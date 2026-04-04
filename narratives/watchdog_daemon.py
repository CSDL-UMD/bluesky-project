import os
import sys

CPU_COUNT = os.cpu_count() or 4

os.environ["TOKENIZERS_PARALLELISM"]        = "false"
os.environ["TRANSFORMERS_VERBOSITY"]        = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["OMP_NUM_THREADS"]               = str(CPU_COUNT)
os.environ["MKL_NUM_THREADS"]               = str(CPU_COUNT)
os.environ["OPENBLAS_NUM_THREADS"]          = str(CPU_COUNT)
os.environ["VECLIB_MAXIMUM_THREADS"]        = str(CPU_COUNT)
os.environ["NUMEXPR_NUM_THREADS"]           = str(CPU_COUNT)

import json
import time
import gzip
import re
import types
import threading
import asyncio
import aiohttp
import requests
from queue import Queue, Empty
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

try:
    import lxml.html as _lxml_html
    _HAS_LXML = True
except Exception:
    _HAS_LXML = False

FETCH_WORKERS    = 256
EMBED_BATCH_SIZE = 16
FETCH_QUEUE_MAX  = 0
EMBED_QUEUE_MAX  = 128
MAX_TEXT_CHARS   = 3000
FETCH_TIMEOUT    = 4
RESOLVE_WORKERS  = 128

_BOILERPLATE_SET = frozenset([
    "please enable javascript", "all rights reserved", "terms of service",
    "privacy policy", "about press copyright", "contact us creators",
    "how youtube works", "test new features", "©", "subscribe to our newsletter",
    "disable any ad blockers", "browser extension", "supported browsers",
])
_SENT_SPLIT  = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_MULTI_SPACE = re.compile(r'\s+')

SIGNAL_FILE      = "./clusters/.analysis_needed"
ANALYSIS_EVERY_N = 24

_embedding_module = None
_http_session: requests.Session = None


def _get_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=FETCH_WORKERS,
            pool_maxsize=FETCH_WORKERS * 2,
            max_retries=0,
        )
        s = requests.Session()
        s.mount("http://",  adapter)
        s.mount("https://", adapter)
        s.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        _http_session = s
    return _http_session


def clean_article_text(text: str, min_len: int = 60) -> str | None:
    if not text:
        return None
    text = text[:MAX_TEXT_CHARS * 3]
    if _HAS_LANGDETECT:
        try:
            if detect(text[:500]) != "en":
                return None
        except Exception:
            return None
    valid_sents = []
    total_chars = 0
    for sent in _SENT_SPLIT.split(text):
        sent = _MULTI_SPACE.sub(" ", sent).strip()
        if len(sent) < min_len:
            continue
        if any(bp in sent.lower() for bp in _BOILERPLATE_SET):
            continue
        valid_sents.append(sent)
        total_chars += len(sent)
        if total_chars >= MAX_TEXT_CHARS:
            break
    return " ".join(valid_sents) if valid_sents else None


try:
    import trafilatura
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"[Watchdog] FATAL: Missing dependency: {e}", flush=True)
    sys.exit(1)

import torch
torch.set_num_threads(CPU_COUNT)
torch.set_num_interop_threads(max(1, CPU_COUNT // 2))


def load_embedding_model(embed_dir: str, weights_path: str):
    global _embedding_module
    device    = torch.device("cpu")
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
    mod          = types.ModuleType("embeddings")
    mod.__file__ = src_path
    sys.modules["embeddings"] = mod
    exec(compile(source, src_path, "exec"), mod.__dict__)

    _embedding_module = mod
    embed_fn = mod.get_sentence_embeddings
    model    = mod.model

    if hasattr(model, "eval"):
        model.eval()

    try:
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")
        print("[Watchdog] torch.compile succeeded (inductor).", flush=True)
    except Exception:
        try:
            model = torch.compile(model)
            print("[Watchdog] torch.compile succeeded (default).", flush=True)
        except Exception:
            print("[Watchdog] torch.compile unavailable, running eager.", flush=True)

    return embed_fn, model


def _normalise_domain(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("http"):
        raw = urlparse(raw).netloc
    return re.sub(r"^www\.", "", raw).rstrip("/")


def load_shortener_domains(csv_path: str) -> set:
    import pandas as pd
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


def _extract_text_lxml(html: str) -> str | None:
    try:
        doc   = _lxml_html.fromstring(html)
        paras = doc.xpath("//p")
        texts = [p.text_content().strip() for p in paras]
        out   = "\n".join(t for t in texts if len(t) > 60)
        return out or None
    except Exception:
        return None


def _extract_text_bs4(html: str) -> str | None:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return "\n".join(p for p in paras if len(p) > 60) or None
    except Exception:
        return None


def fetch_article_text(url: str, timeout: int = FETCH_TIMEOUT) -> str | None:
    try:
        session = _get_session()
        r = session.get(url, timeout=timeout, stream=False)
        r.raise_for_status()
        html = r.text
        result = (_extract_text_lxml(html) if _HAS_LXML else None) or _extract_text_bs4(html)
        if result:
            return result
    except Exception:
        pass

    try:
        dl = trafilatura.fetch_url(url, no_ssl=True)
        if dl:
            ext = trafilatura.extract(
                dl,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_precision=True,
            )
            if ext:
                return ext
    except Exception:
        pass

    return None


async def _async_resolve(
    session: aiohttp.ClientSession, url: str, timeout: int
) -> tuple[str, str]:
    try:
        t = aiohttp.ClientTimeout(total=timeout)
        async with session.head(url, allow_redirects=True, timeout=t) as r:
            if r.status == 405:
                async with session.get(url, allow_redirects=True, timeout=t) as r2:
                    return url, str(r2.url)
            return url, str(r.url)
    except Exception:
        return url, url


async def _resolve_all_async(urls: list[str], timeout: int = 4) -> dict:
    connector = aiohttp.TCPConnector(limit=RESOLVE_WORKERS, ttl_dns_cache=300)
    headers   = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        results = await asyncio.gather(
            *[_async_resolve(session, u, timeout) for u in urls]
        )
    return dict(results)


def resolve_short_urls_concurrently(urls: set, timeout: int = 4) -> dict:
    if not urls:
        return {}
    try:
        loop   = asyncio.new_event_loop()
        result = loop.run_until_complete(_resolve_all_async(list(urls), timeout))
        loop.close()
        return result
    except Exception:
        session = _get_session()
        def _resolve(url):
            try:
                r = session.head(url, allow_redirects=True, timeout=(2, timeout))
                return url, r.url
            except Exception:
                return url, url
        out = {}
        with ThreadPoolExecutor(max_workers=RESOLVE_WORKERS) as pool:
            for orig, resolved in pool.map(_resolve, list(urls)):
                out[orig] = resolved
        return out


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
        filename = os.path.basename(filepath)
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

        seen:               set[str]  = set()
        candidates:       list[dict]  = []
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

        print(f"[Watchdog] {len(candidates):,} candidates. Checking ChromaDB...", flush=True)

        existing       = database.get_existing_urls(all_candidate_urls)
        new_candidates = [c for c in candidates if c["url"] not in existing]
        stat_updates   = [
            (c["url"], c["likes"], c["reposts"])
            for c in candidates if c["url"] in existing
        ]

        print(f"[Watchdog] {len(existing):,} in DB, {len(new_candidates):,} to fetch.", flush=True)

        database.batch_update_stats(stat_updates)

        if not new_candidates:
            database.mark_file_processed(filename)
            return

        fetch_queue   = Queue(maxsize=FETCH_QUEUE_MAX)
        embed_queue   = Queue(maxsize=EMBED_QUEUE_MAX)
        fetcher_done  = threading.Event()
        failed_fetch  = [0]
        failed_embed  = [0]
        fetched_count = [0]
        embed_done    = [0]
        total         = len(new_candidates)
        t_start       = time.time()
        milestones    = {total // 3, (2 * total) // 3, total}
        logged        = set()

        def _fetch_and_clean(cand: dict):
            try:
                raw = fetch_article_text(cand["url"])
            except Exception:
                return None
            if not raw:
                return None
            clean = clean_article_text(raw)
            if not clean:
                return None
            cand["text"] = clean
            return cand

        def _fetcher_worker():
            with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
                futs = {pool.submit(_fetch_and_clean, c): c for c in new_candidates}
                for fut in as_completed(futs):
                    try:
                        result = fut.result(timeout=FETCH_TIMEOUT + 3)
                    except Exception:
                        result = None
                    if result:
                        fetched_count[0] += 1
                        fetch_queue.put(result)
                    else:
                        failed_fetch[0] += 1

                    done = fetched_count[0] + failed_fetch[0]
                    milestone = min(milestones, key=lambda m: abs(m - done))
                    if milestone not in logged and abs(milestone - done) <= max(1, total // 100):
                        logged.add(milestone)
                        elapsed = time.time() - t_start
                        rate    = done / elapsed if elapsed > 0 else 0
                        print(
                            f"[Watchdog] {done}/{total} fetched "
                            f"({fetched_count[0]} ok, {failed_fetch[0]} fail) "
                            f"| {rate:.0f}/s | {embed_done[0]} embedded",
                            flush=True,
                        )
            fetcher_done.set()
            fetch_queue.put(None)

        def _batcher_worker():
            buffer = []
            while True:
                try:
                    item = fetch_queue.get(timeout=5)
                except Empty:
                    if fetcher_done.is_set() and fetch_queue.empty():
                        if buffer:
                            embed_queue.put(buffer)
                        embed_queue.put(None)
                        return
                    continue
                if item is None:
                    if buffer:
                        embed_queue.put(buffer)
                    embed_queue.put(None)
                    return
                buffer.append(item)
                if len(buffer) >= EMBED_BATCH_SIZE:
                    embed_queue.put(buffer)
                    buffer = []

        def _embed_worker():
            with torch.inference_mode():
                while True:
                    try:
                        batch = embed_queue.get(timeout=60)
                    except Empty:
                        break
                    if batch is None:
                        break
                    try:
                        texts   = [a["text"] for a in batch]
                        emb_arr = np.asarray(
                            self.embed_fn(self.model, texts), dtype=np.float32
                        )
                        for j, article in enumerate(batch):
                            article["embedding"] = emb_arr[j]
                        embed_done[0] += len(batch)
                        database.batch_insert_articles([
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
                        ])
                    except Exception as e:
                        print(f"[Watchdog] Embed/insert batch failed: {e}", flush=True)
                        failed_embed[0] += len(batch)

        print(
            f"[Watchdog] fetch({FETCH_WORKERS} threads, {FETCH_TIMEOUT}s timeout) → "
            f"batch({EMBED_BATCH_SIZE}) → embed+insert | {total} articles",
            flush=True,
        )

        t_fetch = threading.Thread(target=_fetcher_worker, daemon=True)
        t_batch = threading.Thread(target=_batcher_worker, daemon=True)
        t_embed = threading.Thread(target=_embed_worker,   daemon=True)

        t_fetch.start()
        t_batch.start()
        t_embed.start()

        t_fetch.join()
        t_batch.join()
        t_embed.join()

        elapsed = time.time() - t_start
        print(
            f"[Watchdog] {filename} done in {elapsed:.0f}s — "
            f"inserted={embed_done[0]}, stats_updated={len(stat_updates)}, "
            f"fetch_fail={failed_fetch[0]}, embed_fail={failed_embed[0]}",
            flush=True,
        )
        database.mark_file_processed(filename)


if __name__ == "__main__":
    FOLDER_TO_WATCH = "../firehose/Payload/processed_reposts_and_likes"
    os.makedirs(FOLDER_TO_WATCH, exist_ok=True)

    print("[Watchdog] Initializing databases...", flush=True)
    database.init_db()
    database.get_chroma_collection()

    _get_session()

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