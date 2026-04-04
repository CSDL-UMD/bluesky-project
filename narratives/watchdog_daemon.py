import os
import json
import time
import gzip
import re
import types
import sys
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np

import database

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

os.environ["TRANSFORMERS_VERBOSITY"]        = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]        = "false"

FETCH_WORKERS    = 8
EMBED_BATCH_SIZE = 32
INSERT_BATCH_SIZE = 64

_BOILERPLATE = [
    "please enable javascript", "all rights reserved", "terms of service",
    "privacy policy", "about press copyright", "contact us creators",
    "how youtube works", "test new features", "©", "subscribe to our newsletter",
    "disable any ad blockers", "browser extension", "supported browsers",
]
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

SIGNAL_FILE      = "./clusters/.analysis_needed"
ANALYSIS_EVERY_N = 24

def clean_article_text(text, min_len=60):
    if not text:
        return None
    if _HAS_LANGDETECT:
        try:
            if detect(text[:2000]) != "en":
                return None
        except Exception:
            return None
    valid_sents = []
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if len(sent) >= min_len and not any(bp in sent.lower() for bp in _BOILERPLATE):
            valid_sents.append(sent)
    return " ".join(valid_sents) if valid_sents else None


try:
    import trafilatura
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"[Watchdog] FATAL: Missing dependency: {e}", flush=True)
    sys.exit(1)

import torch


def load_embedding_model(embed_dir: str, weights_path: str):
    device    = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    embed_dir = os.path.abspath(embed_dir)
    if embed_dir not in sys.path:
        sys.path.insert(0, embed_dir)
    src_path = os.path.join(embed_dir, "embeddings.py")
    with open(src_path, encoding="utf-8") as f:
        source = f.read()
    source = re.sub(
        r"torch\.load\(['\"].*?['\"]\)",
        f"torch.load('{weights_path}', map_location=device, weights_only=False)",
        source,
    )
    source = source.replace(
        "device = torch.device('cpu')",
        "device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')",
    )
    mod          = types.ModuleType("embeddings")
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
    import pandas as pd
    meta_path = os.path.join(newsguard_dir, "metadata.csv")
    mapping: dict = {}
    if not os.path.exists(meta_path):
        return mapping
    df = pd.read_csv(meta_path, low_memory=False, usecols=["Domain", "Score"])
    for _, row in df.dropna(subset=["Score"]).iterrows():
        k = _normalise_domain(str(row["Domain"]))
        if k:
            try:
                mapping[k] = float(row["Score"])
            except ValueError:
                pass
    return mapping


def fetch_article_text(url: str, timeout: int = 10):
    try:
        dl = trafilatura.fetch_url(url)
        if dl:
            ext = trafilatura.extract(dl, include_comments=False, include_tables=False)
            if ext:
                return ext
    except Exception:
        pass
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return "\n".join(p for p in paras if len(p) > 60) or None
    except Exception:
        return None


def resolve_short_urls_concurrently(urls: set, timeout: int = 5, workers: int = 20) -> dict:
    if not urls:
        return {}

    def _resolve(url):
        try:
            r = requests.head(
                url, allow_redirects=True, timeout=(2, timeout),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.status_code == 405:
                r = requests.get(
                    url, allow_redirects=True, stream=True, timeout=(2, timeout),
                    headers={"User-Agent": "Mozilla/5.0"},
                )
            return url, r.url
        except requests.RequestException:
            return url, url

    resolution_map = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_resolve, u): u for u in urls}
        for fut in as_completed(futs):
            orig, resolved = fut.result()
            resolution_map[orig] = resolved
    return resolution_map


def _signal_analysis():
    if os.path.exists("./clusters/.analysis_running"):
        return
    os.makedirs("./clusters", exist_ok=True)
    open(SIGNAL_FILE, "w").close()


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class BlueskyHandler(FileSystemEventHandler):
    def __init__(self, ng_map, shorteners, embed_fn, model):
        self.ng_map     = ng_map
        self.shorteners = shorteners
        self.embed_fn   = embed_fn
        self.model      = model

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".gz"):
            return
        time.sleep(2)
        self.process_file(event.src_path)

    def process_file(self, filepath):
        filename  = os.path.basename(filepath)
        if database.is_file_processed(filename):
            print(f"[Watchdog] {filename} already processed. Skipping.", flush=True)
            return

        print(f"\n[Watchdog] Ingesting: {filename}", flush=True)
        open_func = gzip.open if filepath.endswith(".gz") else open

        short_urls_to_resolve: set[str] = set()
        try:
            with open_func(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        for url in rec.get("urls", []):
                            if _normalise_domain(url) in self.shorteners:
                                short_urls_to_resolve.add(url)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"[Watchdog] Error in pass 1: {e}", flush=True)
            return

        resolved_map = resolve_short_urls_concurrently(short_urls_to_resolve)

        seen:       set[str]  = set()
        candidates: list[dict] = []
        stat_updates: list[tuple] = []
        all_candidate_urls: list[str] = []

        try:
            with open_func(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec        = json.loads(line)
                        likes      = rec.get("likeCount",    0)
                        reposts    = rec.get("repostCount",  0)
                        did        = rec.get("originalDid", "")
                        created_at = rec.get("newCreatedAt", "")

                        for raw_url in rec.get("urls", []):
                            if not raw_url:
                                continue
                            final_url = resolved_map.get(raw_url, raw_url)
                            if final_url in seen:
                                continue

                            domain = _normalise_domain(final_url)
                            parent = (
                                ".".join(domain.split(".")[-2:])
                                if domain.count(".") >= 2 else domain
                            )
                            score = self.ng_map.get(domain) or self.ng_map.get(parent)
                            if score is None or score < 0:
                                continue

                            seen.add(final_url)
                            all_candidate_urls.append(final_url)
                            candidates.append({
                                "url":        final_url,
                                "domain":     domain,
                                "score":      score,
                                "likes":      likes,
                                "reposts":    reposts,
                                "did":        did,
                                "created_at": created_at,
                            })
                    except Exception as parse_err:
                        print(f"[Watchdog] Line error: {parse_err}", flush=True)
        except Exception as e:
            print(f"[Watchdog] Critical error in pass 2: {e}", flush=True)
            return

        print(
            f"[Watchdog] {len(candidates):,} candidate URLs collected. "
            f"Batch-checking existence in ChromaDB...",
            flush=True,
        )

        existing = database.get_existing_urls(all_candidate_urls)

        new_candidates  = [c for c in candidates if c["url"] not in existing]
        stat_updates    = [
            (c["url"], c["likes"], c["reposts"])
            for c in candidates if c["url"] in existing
        ]

        print(
            f"[Watchdog] {len(existing):,} already in DB (stats will be updated), "
            f"{len(new_candidates):,} new articles to fetch.",
            flush=True,
        )

        database.batch_update_stats(stat_updates)

        def _fetch_and_clean(cand: dict):
            raw_text = fetch_article_text(cand["url"])
            if not raw_text:
                return None
            clean_text = clean_article_text(raw_text)
            if not clean_text:
                return None
            cand["text"] = clean_text
            return cand

        fetched: list[dict] = []
        failed  = 0

        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            futs = {pool.submit(_fetch_and_clean, c): c for c in new_candidates}
            done = 0
            for fut in as_completed(futs):
                done += 1
                result = fut.result()
                if result:
                    fetched.append(result)
                else:
                    failed += 1

        embedded: list[dict] = []
        for batch in _chunked(fetched, EMBED_BATCH_SIZE):
            texts    = [a["text"] for a in batch]
            emb_list = self.embed_fn(self.model, texts)
            for article, emb in zip(batch, emb_list):
                article["embedding"] = np.array(emb, dtype=np.float32)
                embedded.append(article)

        total_inserted = 0
        for batch in _chunked(embedded, INSERT_BATCH_SIZE):
            db_batch = [
                {
                    "url":             a["url"],
                    "domain":          a["domain"],
                    "newsguard_score": a["score"],
                    "text":            a["text"],
                    "embedding":       a["embedding"],
                    "did":             a.get("did", ""),
                    "created_at":      a.get("created_at", ""),
                    "likeCount":       a.get("likes",   0),
                    "repostCount":     a.get("reposts", 0),
                }
                for a in batch
            ]
            database.batch_insert_articles(db_batch)
            total_inserted += len(db_batch)

        database.mark_file_processed(filename)
        print(
            f"[Watchdog] {filename} complete — "
            f"{total_inserted} inserted, {len(stat_updates)} stats updated, "
            f"{failed} fetch failures.",
            flush=True,
        )


if __name__ == "__main__":
    FOLDER_TO_WATCH = "../firehose/Payload/processed_reposts_and_likes"
    os.makedirs(FOLDER_TO_WATCH, exist_ok=True)

    print("[Watchdog] Initializing databases...", flush=True)
    database.init_db()
    database.get_chroma_collection()

    print("[Watchdog] Loading NewsGuard, shorteners, and embedding model...", flush=True)
    ng_map     = load_newsguard("../firehose/NewsGuard")
    shorteners = load_shortener_domains("../firehose/NewsGuard/shorturl-services-list.csv")
    embed_fn, model = load_embedding_model(
        "./embedding_model",
        "./embedding_model/specious_model_weights.pt",
    )

    event_handler = BlueskyHandler(ng_map, shorteners, embed_fn, model)

    print(f"[Watchdog] Initial sweep of {FOLDER_TO_WATCH}...", flush=True)
    processed_count = 0
    for filename in sorted(os.listdir(FOLDER_TO_WATCH)):
        if filename.endswith(".gz") and not database.is_file_processed(filename):
            event_handler.process_file(os.path.join(FOLDER_TO_WATCH, filename))
            processed_count += 1
            if processed_count % ANALYSIS_EVERY_N == 0:
                _signal_analysis()
                print(f"[Watchdog] {processed_count} files done. Analysis signalled.", flush=True)

    _signal_analysis()
    print(f"[Watchdog] Sweep complete. {processed_count} files. Final analysis signalled.", flush=True)

    observer = Observer()
    observer.schedule(event_handler, path=FOLDER_TO_WATCH, recursive=False)
    observer.start()
    print("[Watchdog] Monitoring for new files.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Watchdog] Terminating...", flush=True)
        observer.stop()
    observer.join()