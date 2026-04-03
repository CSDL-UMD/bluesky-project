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

os.environ["TRANSFORMERS_VERBOSITY"]       = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]       = "false"

_BOILERPLATE = [
    "please enable javascript", "all rights reserved", "terms of service",
    "privacy policy", "about press copyright", "contact us creators",
    "how youtube works", "test new features", "©", "subscribe to our newsletter",
    "disable any ad blockers", "browser extension", "supported browsers",
]
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

SIGNAL_FILE      = "./clusters/.analysis_needed"
ANALYSIS_EVERY_N = 10

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
        if len(sent) >= min_len:
            if any(bp in sent.lower() for bp in _BOILERPLATE):
                continue
            valid_sents.append(sent)
    if not valid_sents:
        return None
    return " ".join(valid_sents)


try:
    import trafilatura
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"[Watchdog] FATAL: Missing dependency: {e}", flush=True)
    sys.exit(1)

import torch


def load_embedding_model(embed_dir: str, weights_path: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    import pandas as pd
    shorteners = {"bsky.app", "t.co", "bit.ly", "tinyurl.com", "ow.ly", "buff.ly", "lnkd.in"}
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


def fetch_article_text(url, timeout=10):
    dl = trafilatura.fetch_url(url)
    if dl:
        ext = trafilatura.extract(dl, include_comments=False, include_tables=False)
        if ext:
            return ext
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
    resolution_map = {}
    if not urls:
        return resolution_map

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

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_resolve, u): u for u in urls}
        for fut in as_completed(futs):
            orig, resolved = fut.result()
            resolution_map[orig] = resolved

    return resolution_map


def _signal_analysis():
    os.makedirs("./clusters", exist_ok=True)
    open(SIGNAL_FILE, "w").close()


class BlueskyHandler(FileSystemEventHandler):
    def __init__(self, ng_map, shorteners, embed_fn, model):
        self.ng_map     = ng_map
        self.shorteners = shorteners
        self.embed_fn   = embed_fn
        self.model      = model

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith((".gz")):
            return
        time.sleep(2)
        self.process_file(event.src_path)
        _signal_analysis()

    def process_file(self, filepath):
        filename = os.path.basename(filepath)
        if database.is_file_processed(filename):
            print(f"[Watchdog] File {filename} already processed. Skipping.", flush=True)
            return

        print(f"\n[Watchdog] Ingesting new payload: {filename}...", flush=True)
        open_func = gzip.open if filepath.endswith(".gz") else open

        short_urls_to_resolve = set()
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
            print(f"[Watchdog] Error reading file during pass 1: {e}", flush=True)
            return

        resolved_map = resolve_short_urls_concurrently(short_urls_to_resolve)
        seen = set()

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
                                if domain.count(".") >= 2
                                else domain
                            )

                            score = self.ng_map.get(domain)
                            if score is None:
                                score = self.ng_map.get(parent)

                            if score is None or score < 0:
                                continue

                            seen.add(final_url)

                            existing_meta = database.get_article_metadata(final_url)
                            if existing_meta:
                                database.update_article_stats(final_url, likes, reposts)
                            else:
                                raw_text = fetch_article_text(final_url)
                                if not raw_text:
                                    continue

                                clean_text = clean_article_text(raw_text)
                                if not clean_text:
                                    continue

                                emb_list = self.embed_fn(self.model, [clean_text])
                                emb      = np.array(emb_list[0], dtype=np.float32)
                                database.insert_article(
                                    final_url, domain, score, clean_text, emb,
                                    did, created_at, likes, reposts,
                                )

                    except Exception as parse_err:
                        print(f"[Watchdog] Error processing line: {parse_err}", flush=True)
                        continue

            database.mark_file_processed(filename)
            print(
                f"[Watchdog] Payload {filename} completely ingested. "
                f"Processed {len(seen)} unique URLs.",
                flush=True,
            )

        except Exception as e:
            print(f"[Watchdog] Critical error processing {filename}: {e}", flush=True)


if __name__ == "__main__":
    FOLDER_TO_WATCH = "../firehose/Payload/processed_reposts_and_likes"
    os.makedirs(FOLDER_TO_WATCH, exist_ok=True)

    print("[Watchdog] Initializing databases...", flush=True)
    database.init_db()
    database.get_chroma_collection()

    print("[Watchdog] Loading NewsGuard, Shorteners, and Embedding Models into RAM...", flush=True)
    ng_map     = load_newsguard("../firehose/NewsGuard")
    shorteners = load_shortener_domains("../firehose/NewsGuard/shorturl-services-list.csv")
    embed_fn, model = load_embedding_model(
        "./embedding_model",
        "./embedding_model/specious_model_weights.pt",
    )

    event_handler = BlueskyHandler(ng_map, shorteners, embed_fn, model)

    print(f"[Watchdog] Performing initial sweep of {FOLDER_TO_WATCH}...", flush=True)
    processed_count = 0
    for filename in sorted(os.listdir(FOLDER_TO_WATCH)):
        if filename.endswith((".gz")) and not database.is_file_processed(filename):
            event_handler.process_file(os.path.join(FOLDER_TO_WATCH, filename))
            processed_count += 1
            if processed_count % ANALYSIS_EVERY_N == 0:
                _signal_analysis()
                print(f"[Watchdog] {processed_count} files processed. Analysis signalled.", flush=True)

    _signal_analysis()
    print(f"[Watchdog] Sweep complete. {processed_count} total files. Final analysis signalled.", flush=True)

    observer = Observer()
    observer.schedule(event_handler, path=FOLDER_TO_WATCH, recursive=False)
    observer.start()

    print("[Watchdog] Now monitoring directory for new files.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Watchdog] Terminating observer...", flush=True)
        observer.stop()
    observer.join()