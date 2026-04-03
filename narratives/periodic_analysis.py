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
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers   import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    _HAS_SUMY = True
except ImportError:
    _HAS_SUMY = False

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

SIGNAL_FILE    = "./clusters/.analysis_needed"
FIXED_INTERVAL = 30 * 60


def _tok(text):
    return [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()) if w not in _STOPS]


def pmi_keywords(cluster_texts, all_texts, top_n=10):
    ct = [w for t in cluster_texts for w in _tok(t)]
    bt = [w for t in all_texts      for w in _tok(t)]
    if not ct or not bt:
        return []
    cf = Counter(ct)
    bf = Counter(bt)
    scores = {
        w: math.log2((cf[w] / len(ct)) / (bf.get(w, 1) / len(bt)) + 1e-9)
        for w, c in cf.items() if c >= 2
    }
    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])][:top_n]


def run_analysis_job():
    print(f"\n[{datetime.now()}] [Analysis] Starting narrative generation pass...", flush=True)
    articles = database.get_all_articles()

    if len(articles) < 1:
        print("[Analysis] Insufficient data for clustering. Aborting pass.", flush=True)
        return

    texts      = [a["text"]      for a in articles]
    embeddings = np.array([a["embedding"] for a in articles], dtype=np.float32)

    print("[Analysis] Executing DPMeans clustering...", flush=True)
    delta = 0.20
    dp    = DPMeans(delta=delta, n_init=3, max_iter=100, random_state=42)
    dp.fit(embeddings)
    labels = dp.labels_

    if len(set(labels)) < 2 and len(articles) >= 2:
        print("[Analysis] DPMeans found < 2 clusters. Falling back to KMeans(n=2).", flush=True)
        from sklearn.cluster import KMeans
        km     = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

    clusters: dict = defaultdict(list)
    for art, lbl in zip(articles, labels):
        clusters[int(lbl)].append(art)

    print(f"[Analysis] Grouped articles into {len(clusters)} unique narratives.", flush=True)

    if _HAS_SUMY:
        lexrank_summarizer = LexRankSummarizer()

    narratives         = []
    reliable_threshold = 60.0

    for cid, members in clusters.items():
        if len(members) < 1:
            continue

        member_embeddings = np.array([m["embedding"] for m in members])
        cluster_texts     = [m["text"] for m in members]
        keywords          = pmi_keywords(cluster_texts, texts)

        final_summary = ""
        if _HAS_SUMY:
            try:
                cluster_text  = " ".join(cluster_texts)
                parser        = PlaintextParser.from_string(cluster_text, Tokenizer("english"))
                doc_sents     = list(parser.document.sentences)
                extract_count = min(2, len(doc_sents))
                if extract_count > 0:
                    summary_sentences = lexrank_summarizer(parser.document, extract_count)
                    final_summary     = " ".join(str(s) for s in summary_sentences)
            except Exception:
                pass

        if not final_summary.strip():
            centroid     = member_embeddings.mean(axis=0)
            distances    = np.linalg.norm(member_embeddings - centroid, axis=1)
            closest_idx  = int(np.argmin(distances))
            closest_text = members[closest_idx]["text"]
            final_summary = " ".join(_SENT_SPLIT.split(closest_text)[:2])

        ng_scores = [
            m["newsguard_score"]
            for m in members
            if m["newsguard_score"] is not None and m["newsguard_score"] >= 0
        ]
        domains = sorted({m["domain"] for m in members if m["domain"]})

        narratives.append({
            "narrative_id":        int(cid),
            "size":                len(members),
            "summary":             final_summary,
            "pmi_keywords":        keywords,
            "avg_newsguard_score": round(float(np.mean(ng_scores)), 2) if ng_scores else None,
            "total_likes":         sum(m["likeCount"]   for m in members),
            "total_reposts":       sum(m["repostCount"]  for m in members),
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

    rel_narrs     = [n for n in narratives if n["avg_newsguard_score"] is not None and n["avg_newsguard_score"] >= reliable_threshold]
    unrel_narrs   = [n for n in narratives if n["avg_newsguard_score"] is not None and n["avg_newsguard_score"] <  reliable_threshold]
    unrated_narrs = [n for n in narratives if n["avg_newsguard_score"] is None]

    rel_narrs.sort(    key=lambda x: x["size"], reverse=True)
    unrel_narrs.sort(  key=lambda x: x["size"], reverse=True)
    unrated_narrs.sort(key=lambda x: x["size"], reverse=True)

    os.makedirs("./clusters", exist_ok=True)

    def _save(path, label, data):
        print(f"[Analysis] Saving {len(data)} {label} narratives to disk...", flush=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata":   {"generated_at": datetime.now().isoformat(), "total_narratives": len(data)},
                    "narratives": data,
                },
                f, indent=2, ensure_ascii=False,
            )

    _save("./clusters/reliable_narratives_latest.json",   "reliable",   rel_narrs)
    _save("./clusters/unreliable_narratives_latest.json", "unreliable", unrel_narrs)
    _save("./clusters/unrated_narratives_latest.json",    "unrated",    unrated_narrs)

    print("[Analysis] Pass complete. Waiting for next cycle.", flush=True)


if __name__ == "__main__":
    print("[Analysis Scheduler] Activated.", flush=True)
    last_run = 0.0
    while True:
        signal_present = os.path.exists(SIGNAL_FILE)
        time_due       = (time.time() - last_run) >= FIXED_INTERVAL

        if signal_present or time_due:
            if signal_present:
                try:
                    os.remove(SIGNAL_FILE)
                except OSError:
                    pass
            run_analysis_job()
            last_run = time.time()
        else:
            time.sleep(60)