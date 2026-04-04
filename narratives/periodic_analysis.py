import os
import json
import time
import re
import math
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

from sklearn.cluster import DPMeans
import database

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

_STOPS = {
    "the","and","for","that","this","with","are","was","were","have","has",
    "had","but","not","you","they","from","its","our","your","their","about",
    "will","can","been","more","also","when","than","then","who","what","how",
    "just","said","some","there","which","into","one","all","out","get","got",
    "did","she","him","her","his","dont","isnt","cant","wont","would","could",
    "should","may","might","let","way","now","here","very","really","like",
    "even","still","after","over","back","only","think","know","make","take",
}

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _tok(text):
    return [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()) if w not in _STOPS]


def pmi_keywords(cluster_texts, all_text_tokens, top_n=10):
    ct = [w for t in cluster_texts for w in _tok(t)]
    if not ct or not all_text_tokens:
        return []

    cf = Counter(ct)
    bf = Counter(all_text_tokens)
    N_c = len(ct)
    N_b = len(all_text_tokens)

    scores = {
        w: math.log2((cf[w] / N_c) / (bf.get(w, 1) / N_b) + 1e-9)
        for w, c in cf.items() if c >= 2
    }
    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])][:top_n]


def _closest_to_centroid(member_embeddings, members):
    centroid  = member_embeddings.mean(axis=0)
    distances = np.linalg.norm(member_embeddings - centroid, axis=1)
    return members[int(np.argmin(distances))]


def run_analysis_job():
    if os.path.exists(LOCK_FILE):
        print("[Analysis] Already running. Skipping.", flush=True)
        return

    open(LOCK_FILE, "w").close()

    try:
        print(f"\n[{datetime.now()}] [Analysis] Starting narrative generation pass...", flush=True)

        articles = database.get_all_articles()
        if len(articles) < 1:
            print("[Analysis] Insufficient data for clustering. Aborting.", flush=True)
            return

        texts      = [a["text"] for a in articles]
        all_tokens = [w for t in texts for w in _tok(t)]
        embeddings = np.array([a["embedding"] for a in articles], dtype=np.float32)

        print("[Analysis] Running DPMeans...", flush=True)
        dp = DPMeans(delta=0.20, n_init=3, max_iter=100, random_state=42)
        dp.fit(embeddings)
        labels = dp.labels_

        if len(set(labels)) < 2 and len(articles) >= 2:
            print("[Analysis] DPMeans < 2 clusters — falling back to KMeans(n=2).", flush=True)
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(embeddings)

        clusters: dict[int, list] = defaultdict(list)
        cluster_embs: dict[int, list] = defaultdict(list)
        for art, emb, lbl in zip(articles, embeddings, labels):
            clusters[int(lbl)].append(art)
            cluster_embs[int(lbl)].append(emb)

        print(f"[Analysis] {len(clusters)} clusters found.", flush=True)

        if _HAS_SUMY:
            lexrank = LexRankSummarizer()

        reliable_threshold = 60.0
        narratives         = []

        for cid, members in clusters.items():
            if not members:
                continue

            member_embeddings = np.array(cluster_embs[cid], dtype=np.float32)
            cluster_texts     = [m["text"] for m in members]
            keywords          = pmi_keywords(cluster_texts, all_tokens)

            final_summary = ""
            if _HAS_SUMY:
                try:
                    parser    = PlaintextParser.from_string(
                        " ".join(cluster_texts), Tokenizer("english")
                    )
                    doc_sents = list(parser.document.sentences)
                    if doc_sents:
                        sents = lexrank(parser.document, min(2, len(doc_sents)))
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
            total_likes   = sum(m["likeCount"]   for m in members)
            total_reposts = sum(m["repostCount"] for m in members)
            domains       = sorted({m["domain"]  for m in members if m["domain"]})

            narratives.append({
                "narrative_id":        int(cid),
                "size":                len(members),
                "summary":             final_summary,
                "pmi_keywords":        keywords,
                "avg_newsguard_score": round(float(np.mean(ng_scores)), 2) if len(ng_scores) else None,
                "total_likes":         total_likes,
                "total_reposts":       total_reposts,
                "domains_cited":       domains,
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
                            "generated_at":    datetime.now().isoformat(),
                            "total_narratives": len(data),
                        },
                        "narratives": data,
                    },
                    f, indent=2, ensure_ascii=False,
                )

        _save("./clusters/reliable_narratives_latest.json",   "reliable",   rel_narrs)
        _save("./clusters/unreliable_narratives_latest.json", "unreliable", unrel_narrs)
        _save("./clusters/unrated_narratives_latest.json",    "unrated",    unrated_narrs)

        print("[Analysis] Pass complete.", flush=True)

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