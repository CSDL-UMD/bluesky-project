import argparse
import json
import math
import os
import re
import sys
import types
import gzip
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    sys.exit()

try:
    from sklearn.cluster import DPMeans
except ImportError:
    sys.exit()

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    _HAS_SUMY = True
except ImportError:
    _HAS_SUMY = False

def load_embedding_model(embed_dir: str, weights_path: str, device: str):
    embed_dir = os.path.abspath(embed_dir)
    if embed_dir not in sys.path:
        sys.path.insert(0, embed_dir)

    src_path = os.path.join(embed_dir, "embeddings.py")
    if not os.path.exists(src_path):
        sys.exit()

    with open(src_path, encoding="utf-8") as f:
        source = f.read()

    source = re.sub(
        r"torch\.load\(['\"].*?['\"]\)",
        f"torch.load('{weights_path}', map_location=device, weights_only=False)",
        source,
    )

    mod = types.ModuleType("embeddings")
    mod.__file__ = src_path
    sys.modules["embeddings"] = mod
    exec(compile(source, src_path, "exec"), mod.__dict__)

    return mod.get_sentence_embeddings, mod.model

def _normalise_domain(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("http"):
        raw = urlparse(raw).netloc
    return re.sub(r"^www\.", "", raw).rstrip("/")

def load_shortener_domains(csv_path: str) -> set:
    shorteners = set()
    if not os.path.exists(csv_path):
        return shorteners

    try:
        df = pd.read_csv(csv_path, header=None)
        for val in df[0].dropna():
            shorteners.add(_normalise_domain(str(val)))
    except Exception:
        pass
    
    return shorteners

def load_newsguard(newsguard_dir: str) -> dict:
    meta_path = os.path.join(newsguard_dir, "metadata.csv")
    sources_path = os.path.join(newsguard_dir, "all-sources-metadata.csv")

    if not os.path.exists(meta_path):
        raise FileNotFoundError()

    mapping: dict = {}

    df = pd.read_csv(meta_path, low_memory=False, usecols=["Domain", "Score"])
    for _, row in df.dropna(subset=["Score"]).iterrows():
        k = _normalise_domain(str(row["Domain"]))
        if k:
            try:
                mapping[k] = float(row["Score"])
            except ValueError:
                pass

    if os.path.exists(sources_path):
        df2 = pd.read_csv(
            sources_path, low_memory=False,
            usecols=["Source", "Type", "Domain", "Score"],
        ).dropna(subset=["Score"])
        for _, row in df2.iterrows():
            try:
                score = float(row["Score"])
            except ValueError:
                continue
            t = str(row.get("Type", "")).strip().upper()
            parent_key = _normalise_domain(str(row["Domain"]))
            source_key = _normalise_domain(str(row["Source"]))
            if t == "DOMAIN":
                if parent_key and parent_key not in mapping:
                    mapping[parent_key] = score
            else:
                if source_key and source_key not in mapping:
                    mapping[source_key] = score
                if parent_key and parent_key not in mapping:
                    mapping[parent_key] = score

    return mapping

def resolve_short_urls_concurrently(urls: set, timeout: int = 5, workers: int = 20) -> dict:
    resolution_map = {}
    if not urls:
        return resolution_map
        
    def _resolve(url):
        try:
            r = requests.head(url, allow_redirects=True, timeout=(2, timeout), headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 405:
                r = requests.get(url, allow_redirects=True, stream=True, timeout=(2, timeout), headers={"User-Agent": "Mozilla/5.0"})
            return url, r.url
        except requests.RequestException:
            return url, url

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_resolve, u): u for u in urls}
        for fut in tqdm(as_completed(futs), total=len(urls), desc="Unshortening (Parallel)"):
            orig, resolved = fut.result()
            resolution_map[orig] = resolved
            
    return resolution_map

def stream_newsguard_urls(processed_path: str, ng_map: dict, shorteners: set) -> list:
    short_urls_to_resolve = set()
    open_fn = gzip.open if processed_path.endswith('.gz') else open
    
    with open_fn(processed_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                urls = json.loads(line).get("urls", [])
                for url in urls:
                    if _normalise_domain(url) in shorteners:
                        short_urls_to_resolve.add(url)
            except json.JSONDecodeError:
                pass

    resolved_map = resolve_short_urls_concurrently(short_urls_to_resolve, workers=20)

    items = []
    seen = set()    
    with open_fn(processed_path, "rt", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open_fn(processed_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing File"):
            line = line.strip()
            if not line:
                continue
            
            try:
                rec = json.loads(line)
                urls = rec.get("urls", [])
                
                for raw_url in urls:
                    if not raw_url:
                        continue
                        
                    final_url = resolved_map.get(raw_url, raw_url)
                    
                    if final_url in seen:
                        continue
                        
                    domain = _normalise_domain(final_url)
                    parent = ".".join(domain.split(".")[-2:]) if domain.count(".") >= 2 else domain
                    score = ng_map.get(domain) or ng_map.get(parent)
                    
                    if score is not None:
                        seen.add(final_url)
                        items.append({
                            "url":             final_url,
                            "domain":          domain,
                            "newsguard_score": score,
                            "originalDid":     rec.get("originalDid", ""),
                            "createdAt":       rec.get("newCreatedAt", ""),
                            "likeCount":       rec.get("likeCount",   0),
                            "repostCount":     rec.get("repostCount", 0),
                        })
                        
            except json.JSONDecodeError:
                continue
                
    return items

try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NarrativePipeline/1.0)"}

def fetch_article_text(url, timeout=10):
    if _HAS_TRAFILATURA:
        dl = trafilatura.fetch_url(url)
        if dl:
            ext = trafilatura.extract(dl, include_comments=False, include_tables=False)
            if ext: return ext
    if _HAS_BS4:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            return "\n".join(p for p in paras if len(p) > 60) or None
        except Exception:
            return None
    return None

def fetch_all_articles(url_items, cache_dir, workers=8, timeout=10):
    import hashlib
    os.makedirs(cache_dir, exist_ok=True)
    to_fetch = []
    results  = []

    for item in url_items:
        key  = hashlib.md5(item["url"].encode()).hexdigest()
        path = os.path.join(cache_dir, key + ".txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                item = dict(item, article_text=f.read() or None)
            results.append(item)
        else:
            to_fetch.append((item, path))

    if to_fetch:
        def _worker(entry):
            item, path = entry
            text = fetch_article_text(item["url"], timeout)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text or "")
            return dict(item, article_text=text)

        failed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_worker, e): e for e in to_fetch}
            
            with tqdm(total=len(to_fetch), desc="Fetching Articles", unit="url") as pbar:
                for fut in as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    if r["article_text"] is None:
                        failed += 1
                        pbar.set_postfix(failed=failed)
                    pbar.update(1)

    return results

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_BOILERPLATE = [
    "please enable javascript", "all rights reserved", "terms of service", 
    "privacy policy", "about press copyright", "contact us creators", 
    "how youtube works", "test new features", "©", "subscribe to our newsletter",
    "disable any ad blockers", "browser extension", "supported browsers"
]

def process_article(article, min_len=60):
    text = article.get("article_text") or ""
    if not text:
        return None

    if _HAS_LANGDETECT:
        try:
            if detect(text[:2000]) != 'en':
                return None
        except Exception:
            return None

    valid_sents = []
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if len(sent) >= min_len:
            if any(bp in sent.lower() for bp in _BOILERPLATE):
                continue
            valid_sents.append(sent)
            
    if not valid_sents:
        return None
        
    article["clean_text"] = " ".join(valid_sents)
    return article

_STOPS = {
    "the","and","for","that","this","with","are","was","were","have","has",
    "had","but","not","you","they","from","its","our","your","their","about",
    "will","can","been","more","also","when","than","then","who","what","how",
    "just","said","some","there","which","into","one","all","out","get","got",
    "did","she","him","her","his","dont","isnt","cant","wont","would","could",
    "should","may","might","let","way","now","here","very","really","like",
    "even","still","after","over","back","only","think","know","make","take",
}

def _tok(text):
    return [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()) if w not in _STOPS]

def pmi_keywords(cluster_texts, all_texts, top_n=10):
    ct = [w for t in cluster_texts for w in _tok(t)]
    bt = [w for t in all_texts      for w in _tok(t)]
    if not ct or not bt: return []
    cf = Counter(ct); bf = Counter(bt)
    scores = {
        w: math.log2((cf[w] / len(ct)) / (bf.get(w, 1) / len(bt)) + 1e-9)
        for w, c in cf.items() if c >= 2
    }
    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])][:top_n]

def build_narratives(articles, labels, embeddings, all_texts, min_cluster=3, num_extract=2):
    clusters = defaultdict(list)
    for idx, (art, lbl) in enumerate(zip(articles, labels)):
        clusters[int(lbl)].append((art, embeddings[idx]))

    if _HAS_SUMY:
        lexrank_summarizer = LexRankSummarizer()
        
    narratives = []
    for cid, members_data in tqdm(sorted(clusters.items()), desc="Generating Summaries", unit="cluster"):
        if len(members_data) < min_cluster:
            continue
            
        members = [m[0] for m in members_data]
        member_embeddings = np.array([m[1] for m in members_data])
        texts = [m["clean_text"] for m in members]
        keywords = pmi_keywords(texts, all_texts)
        
        final_summary = ""
        if _HAS_SUMY:
            try:
                cluster_text = " ".join(texts)
                parser = PlaintextParser.from_string(cluster_text, Tokenizer("english"))
                
                doc_sents = list(parser.document.sentences)
                extract_count = min(num_extract, len(doc_sents))
                
                if extract_count > 0:
                    summary_sentences = lexrank_summarizer(parser.document, extract_count)
                    final_summary = " ".join([str(sentence) for sentence in summary_sentences])
            except Exception:
                pass
                
        if not final_summary.strip():
            centroid = member_embeddings.mean(axis=0)
            distances = np.linalg.norm(member_embeddings - centroid, axis=1)
            closest_idx = np.argmin(distances)
            closest_text = members[closest_idx]["clean_text"]
            final_summary = " ".join(_SENT_SPLIT.split(closest_text)[:num_extract])
        
        ng_scores = [m["newsguard_score"] for m in members if m["newsguard_score"] is not None]
        domains    = sorted({m["domain"] for m in members if m["domain"]})

        narratives.append({
            "narrative_id":        cid,
            "size":                len(members),
            "summary":             final_summary,
            "pmi_keywords":        keywords,
            "avg_newsguard_score": round(float(np.mean(ng_scores)), 2) if ng_scores else None,
            "total_likes":         sum(m.get("likeCount", 0) for m in members),
            "total_reposts":       sum(m.get("repostCount", 0) for m in members),
            "domains_cited":       domains,
            "articles": [
                {
                    "url":             m["url"],
                    "domain":          m["domain"],
                    "createdAt":       m.get("createdAt", ""),
                    "newsguard_score": m.get("newsguard_score", None),
                    "text_preview":    m["clean_text"][:200] + "..." if len(m["clean_text"]) > 200 else m["clean_text"]
                }
                for m in members
            ],
        })

    narratives.sort(key=lambda n: -n["size"])
    return narratives

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed",  required=True)
    p.add_argument("--newsguard",  default="./NewsGuard")
    p.add_argument("--shorteners", default="./NewsGuard/shorturl-services-list.csv")
    p.add_argument("--embed_dir",  default="./embedding_model")
    p.add_argument("--weights",    default=("./embedding_model/specious_model_weights.pt"))
    p.add_argument("--out_dir",    default="./clusters")
    p.add_argument("--cache_dir",  default="./article_cache")
    p.add_argument("--delta",      type=float, default=0.20)
    p.add_argument("--min_cluster", type=int,  default=2)
    p.add_argument("--workers",    type=int,   default=8)
    p.add_argument("--timeout",    type=int,   default=10)
    p.add_argument("--reliable_threshold", type=float, default=60.0)
    p.add_argument("--num_extract", type=int, default=2)
    return p.parse_args()

def run():
    args    = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    date_tag = (os.path.basename(args.processed).replace("processed_", "").replace(".json", "").replace(".gz", ""))

    ng_map = load_newsguard(args.newsguard)
    shortener_domains = load_shortener_domains(args.shorteners)
    
    url_items = stream_newsguard_urls(args.processed, ng_map, shortener_domains)
    if not url_items:
        sys.exit()
        
    articles = fetch_all_articles(url_items, args.cache_dir, args.workers, args.timeout)
    
    valid_articles = []
    for art in articles:
        processed = process_article(art)
        if processed:
            valid_articles.append(processed)
            
    if not valid_articles:
        sys.exit()
    
    device_id = "cpu" 
    try:
        if torch.backends.mps.is_available():
            device_id = "mps"
    except Exception:
        pass
        
    get_sentence_embeddings, embed_model = load_embedding_model(args.embed_dir, args.weights, device_id)
    texts      = [a["clean_text"] for a in valid_articles]
    embeddings = np.array(get_sentence_embeddings(embed_model, texts), dtype=np.float32)
    
    dp = DPMeans(delta=args.delta, n_init=3, max_iter=100, random_state=42)
    dp.fit(embeddings)
    labels = dp.labels_

    all_texts = texts
    rel_idx   = [i for i, a in enumerate(valid_articles)
                 if a["newsguard_score"] is not None and a["newsguard_score"] >= args.reliable_threshold]
    unrel_idx = [i for i, a in enumerate(valid_articles)
                 if a["newsguard_score"] is not None and a["newsguard_score"] <  args.reliable_threshold]

    def subset_narratives(indices):
        sub_articles = [valid_articles[i] for i in indices]
        sub_l = labels[np.array(indices)] if len(indices) else np.array([], dtype=int)
        sub_e = embeddings[np.array(indices)] if len(indices) else np.array([], dtype=np.float32)
        return build_narratives(sub_articles, sub_l, sub_e, all_texts, args.min_cluster, args.num_extract)
        
    rel_narrs   = subset_narratives(rel_idx)
    unrel_narrs = subset_narratives(unrel_idx)

    def save(fname, narratives, label):
        path = os.path.join(args.out_dir, fname)
        payload = {
            "metadata": {
                "generated_at":        datetime.now().isoformat(),
                "source":              args.processed,
                "newsguard_dir":       args.newsguard,
                "reliability_label":   label,
                "newsguard_threshold": args.reliable_threshold,
                "dpmeans_delta":       args.delta,
                "num_extracted":       args.num_extract,
                "total_narratives":    len(narratives),
                "total_articles":      sum(n["size"] for n in narratives),
            },
            "narratives": narratives,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    save(f"reliable_narratives_{date_tag}.json",   rel_narrs,   "reliable")
    save(f"unreliable_narratives_{date_tag}.json", unrel_narrs, "unreliable")

if __name__ == "__main__":
    run()