import os
import gc

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
from multiprocessing import Pool
import numpy as np
import faiss

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

SIGNAL_FILE             = "./clusters/.analysis_needed"
LOCK_FILE               = "./clusters/.analysis_running"
RELIABLE_THRESHOLD      = 60.
RELIABLE_DELTA          = 0.30
MIN_MEMBERS_FOR_LEXRANK = 5

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


def _compute_delta(n_articles, base_delta=0.25):
    if n_articles < 500:
        return 0.45
    if n_articles < 5_000:
        return 0.38
    if n_articles < 50_000:
        return 0.30
    return base_delta


def _is_english(text):
    if not _HAS_LANGDETECT:
        return True
    try:
        return detect(text[:500]) == "en"
    except Exception:
        return False


def _tok(text):
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPS]


def _tok_batch(texts):
    out = []
    for t in texts:
        out.extend(_tok(t))
    return out


def pmi_keywords(cluster_tokens, n_all_tokens, background_counter, top_n=10):
    if not cluster_tokens or not n_all_tokens:
        return []
    cf      = Counter(cluster_tokens)
    N_c     = len(cluster_tokens)
    N_b     = n_all_tokens
    inv_N_c = 1.0 / N_c
    inv_N_b = 1.0 / N_b
    scores  = {
        w: math.log2((c * inv_N_c) / (background_counter.get(w, 1) * inv_N_b) + 1e-9)
        for w, c in cf.items() if c >= 2
    }
    return sorted(scores, key=scores.__getitem__, reverse=True)[:top_n]


def _closest_to_centroid(member_embeddings, members):
    centroid  = member_embeddings.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(member_embeddings - centroid, axis=1)
    return members[int(np.argmin(distances))]


def _process_cluster(args):
    cid, members, member_embeddings_list, cluster_tokens, n_all_tokens, background_counter, has_sumy = args
    member_embeddings = np.array(member_embeddings_list, dtype=np.float32)

    keywords = pmi_keywords(cluster_tokens, n_all_tokens, background_counter)

    final_summary = ""
    if has_sumy and len(members) >= MIN_MEMBERS_FOR_LEXRANK:
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
        "cid":           cid,
        "summary":       final_summary,
        "keywords":      keywords,
        "ng_scores":     ng_scores,
        "total_likes":   sum(m["likeCount"]   for m in members),
        "total_reposts": sum(m["repostCount"] for m in members),
        "domains":       sorted({m["domain"]  for m in members if m["domain"]}),
        "members":       members,
    }


def _cluster_and_process(articles, label, background_counter, n_all_tokens, delta):
    if len(articles) < 2:
        print(f"[Analysis] Not enough {label} articles ({len(articles)}), skipping.", flush=True)
        return []

    max_clusters = max(50, min(2000, len(articles) // 20))

    embeddings = np.array([a["embedding"] for a in articles], dtype=np.float32)
    faiss.normalize_L2(embeddings)

    print(f"[Analysis] [{label}] FAISS Dynamic Clustering on {len(embeddings):,} embeddings "
          f"(delta={delta}, max_clusters={max_clusters})...", flush=True)

    d        = embeddings.shape[1]
    delta_sq = delta ** 2

    centroids = [embeddings[0]]
    index     = faiss.IndexFlatL2(d)
    index.add(np.expand_dims(embeddings[0], axis=0))

    batch_size = 10000

    for start in range(0, len(embeddings), batch_size):
        chunk    = embeddings[start:start + batch_size]
        D, _     = index.search(chunk, 1)
        outliers = np.where(D[:, 0] > delta_sq)[0]
        for idx in outliers:
            vec     = np.expand_dims(chunk[idx], axis=0)
            dist, _ = index.search(vec, 1)
            if dist[0][0] > delta_sq:
                index.add(vec)
                centroids.append(chunk[idx])
                if len(centroids) >= max_clusters:
                    break
        if len(centroids) >= max_clusters:
            print(f"[Analysis] [{label}] Reached cap of {max_clusters} clusters. Stopping generation.", flush=True)
            break

    centroids  = np.array(centroids, dtype=np.float32)
    n_clusters = len(centroids)
    print(f"[Analysis] [{label}] {n_clusters} clusters found.", flush=True)

    if n_clusters < 2:
        print(f"[Analysis] [{label}] Falling back to KMeans(n=5).", flush=True)
        n_clusters    = 5
        kmeans        = faiss.Kmeans(d, n_clusters, niter=20, verbose=False, seed=42)
        kmeans.train(embeddings)
        centroids     = kmeans.centroids
        _, labels_arr = kmeans.index.search(embeddings, 1)
        labels        = labels_arr[:, 0]
    else:
        for _ in range(15):
            index = faiss.IndexFlatL2(d)
            index.add(centroids)
            _, labels_arr = index.search(embeddings, 1)
            labels        = labels_arr[:, 0]

            new_centroids = np.zeros_like(centroids)
            counts        = np.zeros(n_clusters, dtype=np.int32)

            np.add.at(new_centroids, labels, embeddings)
            np.add.at(counts,        labels, 1)

            valid = counts > 0
            new_centroids[valid] /= counts[valid][:, None]
            faiss.normalize_L2(new_centroids)

            if np.allclose(centroids, new_centroids, atol=1e-4, rtol=1e-4):
                break
            centroids = new_centroids

        index = faiss.IndexFlatL2(d)
        index.add(centroids)
        _, labels_arr = index.search(embeddings, 1)
        labels        = labels_arr[:, 0]

    clusters     = defaultdict(list)
    cluster_embs = defaultdict(list)
    cluster_toks = defaultdict(list)

    for art, emb, lbl in zip(articles, embeddings, labels):
        lbl = int(lbl)
        clusters[lbl].append(art)
        cluster_embs[lbl].append(emb)

    del embeddings
    gc.collect()

    for lbl, members in clusters.items():
        cluster_toks[lbl] = [tok for m in members for tok in m["_tokens"]]

    print(f"[Analysis] [{label}] Computing PMI + summaries for {n_clusters} clusters...", flush=True)

    cluster_args = [
        (cid, members, cluster_embs[cid], cluster_toks[cid],
         n_all_tokens, background_counter, _HAS_SUMY)
        for cid, members in clusters.items() if members
    ]

    results_map = {}
    with ThreadPoolExecutor(max_workers=min(CPU_COUNT, len(cluster_args))) as pool:
        futs = {pool.submit(_process_cluster, a): a for a in cluster_args}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results_map[r["cid"]] = r
            except Exception as e:
                print(f"[Analysis] [{label}] Cluster error: {e}", flush=True)

    narratives = []
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
                {"url": m["url"], "domain": m["domain"],
                 "newsguard_score": m["newsguard_score"], "likes": m["likeCount"]}
                for m in members
            ],
        })

    return sorted(narratives, key=lambda x: x["size"], reverse=True)


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
        print(f"[Analysis] {len(articles):,} English articles.", flush=True)

        if len(articles) < 2:
            print("[Analysis] Insufficient English articles. Aborting.", flush=True)
            return

        texts      = [a["text"] for a in articles]
        CHUNK_SIZE = 2000
        chunks     = [texts[i:i + CHUNK_SIZE] for i in range(0, len(texts), CHUNK_SIZE)]
        del texts

        background_counter = Counter()
        n_all_tokens       = 0

        with Pool(processes=CPU_COUNT) as pool:
            for result in pool.imap_unordered(_tok_batch, chunks):
                background_counter.update(result)
                n_all_tokens += len(result)
                del result
        del chunks
        gc.collect()

        for a in articles:
            a["_tokens"] = _tok(a["text"])

        reliable_articles   = [a for a in articles
                                if a["newsguard_score"] is not None
                                and a["newsguard_score"] >= RELIABLE_THRESHOLD]
        unreliable_articles = [a for a in articles
                                if a["newsguard_score"] is not None
                                and a["newsguard_score"] < RELIABLE_THRESHOLD]

        print(f"[Analysis] {len(reliable_articles):,} reliable, "
              f"{len(unreliable_articles):,} unreliable.", flush=True)

        os.makedirs("./clusters", exist_ok=True)

        rel_delta   = _compute_delta(len(reliable_articles))
        unrel_delta = _compute_delta(len(unreliable_articles))

        rel_narratives   = _cluster_and_process(
            reliable_articles,   "reliable",   background_counter, n_all_tokens, rel_delta)
        unrel_narratives = _cluster_and_process(
            unreliable_articles, "unreliable", background_counter, n_all_tokens, unrel_delta)

        def _save(path, label, data):
            print(f"[Analysis] Saving {len(data)} {label} narratives -> {path}", flush=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"metadata": {"generated_at":     datetime.now().isoformat(),
                                        "total_narratives": len(data)},
                           "narratives": data}, f, indent=2, ensure_ascii=False)

        _save("./clusters/reliable_narratives_latest.json",   "reliable",   rel_narratives)
        _save("./clusters/unreliable_narratives_latest.json", "unreliable", unrel_narratives)

        print(f"[Analysis] Complete in {time.time() - t0:.1f}s.", flush=True)

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