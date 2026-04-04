import os
import sys

CPU_COUNT = os.cpu_count() or 4

os.environ["OMP_NUM_THREADS"]        = str(CPU_COUNT)
os.environ["MKL_NUM_THREADS"]        = str(CPU_COUNT)
os.environ["OPENBLAS_NUM_THREADS"]   = str(CPU_COUNT)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_COUNT)
os.environ["NUMEXPR_NUM_THREADS"]    = str(CPU_COUNT)

import json
import time
import re
import math
from collections import Counter, defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from sklearn.cluster import DPMeans
import database

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)
    from sumy.parsers.plaintext    import PlaintextParser
    from sumy.nlp.tokenizers       import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    _HAS_SUMY = True
except ImportError:
    _HAS_SUMY = False

SIGNAL_FILE = "./clusters/.analysis_needed"
LOCK_FILE   = "./clusters/.analysis_running"

_STOPS = frozenset({
    "the","and","for","that","this","with","are","was","were","have","has",
    "had","but","not","you","they","from","its","our","your","their","about",
    "will","can","been","more","also","when","than","then","who","what","how",
    "just","said","some","there","which","into","one","all","out","get","got",
    "did","she","him","her","his","dont","isnt","cant","wont","would","could",
    "should","may","might","let","way","now","here","very","really","like",
    "even","still","after","over","back","only","think","know","make","take",
})

_WORD_RE    = re.compile(r"\b[a-zA-Z]{3,}\b")
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _is_english(text: str) -> bool:
    if not _HAS_LANGDETECT:
        return True
    try:
        return detect(text[:500]) == "en"
    except Exception:
        return False


def _tok(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPS]


def _tok_batch(texts: list[str]) -> list[str]:
    out = []
    for t in texts:
        out.extend(_tok(t))
    return out


def pmi_keywords(
    cluster_tokens: list[str],
    all_tokens: list[str],
    background_counter: Counter,
    top_n: int = 10,
) -> list[str]:
    if not cluster_tokens or not all_tokens:
        return []

    cf      = Counter(cluster_tokens)
    N_c     = len(cluster_tokens)
    N_b     = len(all_tokens)
    inv_N_c = 1.0 / N_c
    inv_N_b = 1.0 / N_b

    scores = {
        w: math.log2((c * inv_N_c) / (background_counter.get(w, 1) * inv_N_b) + 1e-9)
        for w, c in cf.items() if c >= 2
    }
    return sorted(scores, key=scores.__getitem__, reverse=True)[:top_n]


def _closest_to_centroid(member_embeddings: np.ndarray, members: list) -> dict:
    centroid  = member_embeddings.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(member_embeddings - centroid, axis=1)
    return members[int(np.argmin(distances))]


def _process_cluster(args: tuple) -> dict:
    cid, members, member_embeddings_list, cluster_tokens, all_tokens, background_counter, has_sumy = args
    member_embeddings = np.array(member_embeddings_list, dtype=np.float32)

    keywords = pmi_keywords(cluster_tokens, all_tokens, background_counter)

    final_summary = ""
    if has_sumy:
        try:
            lexrank       = LexRankSummarizer()
            cluster_texts = [m["text"] for m in members]
            joined        = " ".join(cluster_texts)[:50_000]
            parser        = PlaintextParser.from_string(joined, Tokenizer("english"))
            doc_sents     = list(parser.document.sentences)
            if doc_sents:
                sents         = lexrank(parser.document, min(2, len(doc_sents)))
                final_summary = " ".join(str(s) for s in sents)
        except Exception:
            pass

    if not final_summary.strip():
        closest       = _closest_to_centroid(member_embeddings, members)
        final_summary = " ".join(_SENT_SPLIT.split(closest["text"])[:2])

    ng_scores = np.array(
        [m["newsguard_score"] for m in members
         if m["newsguard_score"] is not None and m["newsguard_score"] >= 0],
        dtype=np.float32,
    )

    return {
        "cid":          cid,
        "summary":      final_summary,
        "keywords":     keywords,
        "ng_scores":    ng_scores,
        "total_likes":  sum(m["likeCount"]   for m in members),
        "total_reposts":sum(m["repostCount"] for m in members),
        "domains":      sorted({m["domain"]  for m in members if m["domain"]}),
        "members":      members,
    }


def run_analysis_job():
    if os.path.exists(LOCK_FILE):
        print("[Analysis] Already running. Skipping.", flush=True)
        return

    open(LOCK_FILE, "w").close()

    try:
        t0 = time.time()
        print(f"\n[{datetime.now()}] [Analysis] Starting...", flush=True)

        articles = database.get_all_articles()
        if len(articles) < 2:
            print("[Analysis] Insufficient data. Aborting.", flush=True)
            return

        articles = [a for a in articles if _is_english(a["text"])]
        print(f"[Analysis] {len(articles):,} English articles after language filter.", flush=True)

        if len(articles) < 2:
            print("[Analysis] Insufficient English articles. Aborting.", flush=True)
            return

        texts      = [a["text"] for a in articles]
        chunk_size = max(1, len(texts) // CPU_COUNT)
        chunks     = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

        with ThreadPoolExecutor(max_workers=CPU_COUNT) as pool:
            token_chunks = list(pool.map(_tok_batch, chunks))

        all_tokens         = [tok for chunk in token_chunks for tok in chunk]
        background_counter = Counter(all_tokens)

        embeddings = np.array([a["embedding"] for a in articles], dtype=np.float32)

        print(f"[Analysis] Running DPMeans on {len(embeddings):,} embeddings...", flush=True)
        dp = DPMeans(delta=0.20, n_init=3, max_iter=100, random_state=42)
        dp.fit(embeddings)
        labels = dp.labels_

        n_clusters = len(set(labels))
        print(f"[Analysis] {n_clusters} clusters found.", flush=True)

        if n_clusters < 2 and len(articles) >= 2:
            print("[Analysis] DPMeans < 2 clusters — falling back to KMeans(n=2).", flush=True)
            from sklearn.cluster import KMeans
            labels = KMeans(
                n_clusters=2, random_state=42, n_init=10
            ).fit_predict(embeddings)

        clusters:     dict[int, list] = defaultdict(list)
        cluster_embs: dict[int, list] = defaultdict(list)
        cluster_toks: dict[int, list] = defaultdict(list)

        for art, emb, lbl in zip(articles, embeddings, labels):
            lbl = int(lbl)
            clusters[lbl].append(art)
            cluster_embs[lbl].append(emb)

        for lbl, members in clusters.items():
            cluster_toks[lbl] = _tok_batch([m["text"] for m in members])

        print(f"[Analysis] Computing PMI + summaries for {len(clusters)} clusters in parallel...", flush=True)

        cluster_args = [
            (
                cid,
                members,
                cluster_embs[cid],
                cluster_toks[cid],
                all_tokens,
                background_counter,
                _HAS_SUMY,
            )
            for cid, members in clusters.items()
            if members
        ]

        results_map: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=min(CPU_COUNT, len(cluster_args))) as pool:
            futs = {pool.submit(_process_cluster, a): a for a in cluster_args}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    results_map[r["cid"]] = r
                except Exception as e:
                    print(f"[Analysis] Cluster processing error: {e}", flush=True)

        reliable_threshold = 60.0
        narratives         = []

        for cid, members in clusters.items():
            if not members or cid not in results_map:
                continue
            r         = results_map[cid]
            ng_scores = r["ng_scores"]
            narratives.append({
                "narrative_id":        cid,
                "size":                len(members),
                "summary":             r["summary"],
                "pmi_keywords":        r["keywords"],
                "avg_newsguard_score": round(float(np.mean(ng_scores)), 2) if len(ng_scores) else None,
                "total_likes":         r["total_likes"],
                "total_reposts":       r["total_reposts"],
                "domains_cited":       r["domains"],
                "articles": [
                    {
                        "url":             m["url"],
                        "domain":          m["domain"],
                        "newsguard_score": m["newsguard_score"],
                        "likes":           m["likeCount"],
                    }
                    for m in members
                ],
            })

        def _sort(lst):
            return sorted(lst, key=lambda x: x["size"], reverse=True)

        rel_narrs     = _sort([n for n in narratives
                               if n["avg_newsguard_score"] is not None
                               and n["avg_newsguard_score"] >= reliable_threshold])
        unrel_narrs   = _sort([n for n in narratives
                               if n["avg_newsguard_score"] is not None
                               and n["avg_newsguard_score"] <  reliable_threshold])
        unrated_narrs = _sort([n for n in narratives
                               if n["avg_newsguard_score"] is None])

        os.makedirs("./clusters", exist_ok=True)

        def _save(path, label, data):
            print(f"[Analysis] Saving {len(data)} {label} narratives → {path}", flush=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "generated_at":     datetime.now().isoformat(),
                            "total_narratives": len(data),
                        },
                        "narratives": data,
                    },
                    f, indent=2, ensure_ascii=False,
                )

        _save("./clusters/reliable_narratives_latest.json",   "reliable",   rel_narrs)
        _save("./clusters/unreliable_narratives_latest.json", "unreliable", unrel_narrs)
        _save("./clusters/unrated_narratives_latest.json",    "unrated",    unrated_narrs)

        print(f"[Analysis] Pass complete in {time.time() - t0:.1f}s.", flush=True)

    finally:
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass


if __name__ == "__main__":
    print("[Analysis Worker] Waiting for signals...", flush=True)
    while True:
        if os.path.exists(SIGNAL_FILE):
            try:
                os.remove(SIGNAL_FILE)
            except OSError:
                pass
            run_analysis_job()
        else:
            time.sleep(5)