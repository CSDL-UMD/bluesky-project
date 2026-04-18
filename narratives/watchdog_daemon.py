import os
import sys
import json
import time
import gzip
import re
import types
import threading
import asyncio
import aiohttp
import requests
import random
import gc
import signal
from collections import defaultdict
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from contextlib import nullcontext
from functools import lru_cache
from datetime import datetime
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import numpy as np
import database

CPU_COUNT = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
_langdetect_lock = threading.Lock()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["OMP_NUM_THREADS"] = str(CPU_COUNT)
os.environ["MKL_NUM_THREADS"] = str(CPU_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_COUNT)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_COUNT)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_COUNT)

import torch
import logging

logging.getLogger("readability.readability").setLevel(logging.CRITICAL)
_RE_INVALID_XML = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]')
HAS_GPU = torch.cuda.is_available()

if HAS_GPU:
    FETCH_WORKERS = 150
    EMBED_BATCH_SIZE = 512
    FETCH_TIMEOUT = 15
    RESOLVE_WORKERS = 150
    PIPELINE_CHUNK = 512
    CONNECT_TIMEOUT = 5
    MAX_RESPONSE_BYTES = 5_000_000
    RESOLVE_CONCURRENCY = 150
    DOMAIN_RATE_LIMIT = 0.5
    QUEUE_MAXSIZE = 8
    RESOLVE_POOL_WORKERS = min(CPU_COUNT, 4)
    MGZIP_THREADS = min(CPU_COUNT, 8)
    RESOLVE_BATCH_SIZE = 1000
    EXTRACT_WORKERS = min(CPU_COUNT, 4)
else:
    FETCH_WORKERS = 50
    EMBED_BATCH_SIZE = 32
    FETCH_TIMEOUT = 12
    RESOLVE_WORKERS = 50
    PIPELINE_CHUNK = 128
    CONNECT_TIMEOUT = 4
    MAX_RESPONSE_BYTES = 2_500_000
    RESOLVE_CONCURRENCY = 50
    DOMAIN_RATE_LIMIT = 0.2
    QUEUE_MAXSIZE = 16
    RESOLVE_POOL_WORKERS = 2
    MGZIP_THREADS = 1
    RESOLVE_BATCH_SIZE = 500
    EXTRACT_WORKERS = 2

MAX_TEXT_CHARS = 3000
PROGRESS_EVERY = 5
DOMAIN_FAIL_THRESH = 10
DNS_CACHE_TTL = 600
SIGNAL_FILE = "./clusters/.analysis_needed"
PASSAGE_WORD_LIMIT = 100
ANALYSIS_SIGNAL_INTERVAL = 72

_BSKY_LIKE_TYPE = "app.bsky.feed.like"
_BSKY_REPOST_TYPE = "app.bsky.feed.repost"

_TRACKING_PARAMS = frozenset({
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'utm_id', 'utm_source_platform', 'utm_creative_format',
    'fbclid', 'gclid', 'gclsrc', 'dclid', 'gbraid', 'wbraid',
    'msclkid', 'twclid', 'igshid', 'mc_cid', 'mc_eid',
    'ref', 'ref_src', 'ref_url', 'source', 'src',
    'ftag', 'linkId', 'gift', 'share', 'smid', 'smtyp',
    '_ga', '_gl', 'ncid', 'ocid', 'sr_share',
})

try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 65536)
    if soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f"[Watchdog] Increased file descriptor limit to {target}", flush=True)
        except (ValueError, OSError):
            pass
except ImportError:
    pass

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

try:
    import orjson as _json_lib
    def _json_loads(b): return _json_lib.loads(b)
except ImportError:
    try:
        import ujson as _json_lib
        _json_loads = _json_lib.loads
    except ImportError:
        import json as _json_lib
        _json_loads = _json_lib.loads

try:
    import mgzip as _mgzip
    _HAS_MGZIP = True
except ImportError:
    _HAS_MGZIP = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

try:
    from readability import Document
    from bs4 import BeautifulSoup
    _HAS_READABILITY = True
except ImportError:
    _HAS_READABILITY = False

try:
    import trafilatura as _trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

if not _HAS_READABILITY and not _HAS_TRAFILATURA:
    print("[Watchdog] FATAL: neither readability-lxml nor trafilatura available.", flush=True)
    sys.exit(1)

_embedding_module = None
_http_session = None
_http_session_lock = threading.Lock()
_DEVICE = None
_extract_pool = None
_extract_pool_lock = threading.Lock()

_domain_fail_counts: dict[str, int] = defaultdict(int)
_domain_fail_lock = threading.Lock()
_domain_blacklist: set[str] = set()
_domain_locks: dict[str, asyncio.Lock] = {}
_domain_last_hit: dict[str, float] = {}

_files_processed_since_analysis = 0
_files_processed_lock = threading.Lock()

_WHALES = frozenset(["reuters.com", "bloomberg.com", "nytimes.com", "wsj.com", "ft.com"])

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}


@lru_cache(maxsize=1 << 18)
def normalise_url(url: str) -> str:
    if not url:
        return url
    try:
        parsed = urlparse(url)
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=False)
            filtered = {
                k: v for k, v in params.items()
                if k.lower() not in _TRACKING_PARAMS
            }
            new_query = urlencode(filtered, doseq=True) if filtered else ''
        else:
            new_query = ''
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path.rstrip('/') if parsed.path != '/' else '/',
            parsed.params,
            new_query,
            ''
        ))
        return normalized
    except Exception:
        return url


@lru_cache(maxsize=1 << 17)
def _normalise_domain(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("http"):
        try:
            raw = urlparse(raw).netloc
        except Exception:
            return ""
    return re.sub(r"^www\.", "", raw).rstrip("/")


@lru_cache(maxsize=1 << 17)
def _parent_domain(domain: str) -> str:
    return ".".join(domain.split(".")[-2:]) if domain.count(".") >= 2 else domain


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "done"
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{seconds:.0f}s"


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_MULTI_SPACE = re.compile(r'\s+')
_URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)


class ProgressTracker:
    STATUS_FILE = "./logs/progress_status.json"

    def __init__(self, filename: str, total: int):
        self.filename = filename
        self.total = total
        self.t_start = time.time()
        self._lock = threading.Lock()
        self._counters = dict(
            lines_read=0, candidates=0, existing=0, new=0,
            fetched=0, fetch_fail=0, embedded=0, embed_fail=0,
            inserted=0, db_fail=0, phase="init", skipped_whales=0,
            quality_rejected=0,
            pending_resolve=0, resolved=0, resolve_batches_sent=0,
            interactions_recorded=0,
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reporter, daemon=True)
        self._thread.start()

    def set_phase(self, phase: str):
        with self._lock:
            self._counters["phase"] = phase
        print(f"[Watchdog] ── phase: {phase} ──", flush=True)

    def inc(self, key: str, n: int = 1):
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + n

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._counters)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self._write_status()

    def _reporter(self):
        while not self._stop.wait(timeout=PROGRESS_EVERY):
            self._write_status()

    def _write_status(self):
        snap = self.snapshot()
        elapsed = time.time() - self.t_start
        phase = snap["phase"]

        if phase == "scan+resolve":
            pending = snap.get("pending_resolve", 0)
            resolved = snap.get("resolved", 0)
            batches = snap.get("resolve_batches_sent", 0)
            resolve_rate = resolved / elapsed if elapsed > 0 else 0

            print(
                f"[Watchdog] [{phase:14s}] "
                f"read={snap['lines_read']:,} cands={snap['candidates']:,} | "
                f"resolve: {resolved:,}/{pending:,} pending ({resolve_rate:.0f}/s) batches={batches}",
                flush=True,
            )
        else:
            new = snap["new"]
            fetched = snap["fetched"]
            fail = snap["fetch_fail"]
            quality_rej = snap.get("quality_rejected", 0)
            done = fetched + fail + quality_rej
            rate = done / elapsed if elapsed > 0 else 0
            eta = _fmt_eta((new - done) / rate if rate > 0 and new > done else 0)

            print(
                f"[Watchdog] [{phase:14s}] "
                f"read={snap['lines_read']:,} cands={snap['candidates']:,} "
                f"new={new:,} | fetch {fetched}/{new} "
                f"({fail} fail, {quality_rej} quality_rej, {snap['skipped_whales']} skipped) "
                f"{rate:.0f}/s ETA {eta} | emb={snap['embedded']:,} ins={snap['inserted']:,}",
                flush=True,
            )


def _get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda") if HAS_GPU else torch.device("cpu")
    return _DEVICE


def _make_session() -> requests.Session:
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=FETCH_WORKERS,
        pool_maxsize=FETCH_WORKERS,
        max_retries=0,
    )
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(_HEADERS)
    return s


def _get_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        with _http_session_lock:
            if _http_session is None:
                _http_session = _make_session()
    return _http_session


def segment_into_passages(text: str, word_limit: int = PASSAGE_WORD_LIMIT) -> list:
    if not text:
        return []

    paragraphs = re.split(r'\n\t+', text)
    passages = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        words = para.split()
        if len(words) <= word_limit:
            if para:
                passages.append(para)
        else:
            for i in range(0, len(words), word_limit):
                chunk_words = words[i:i + word_limit]
                if chunk_words:
                    passages.append(" ".join(chunk_words))

    return passages if passages else [text[:500]]


def clean_article_text(text: str, min_len: int = 150) -> str | None:
    if not text:
        return None

    text = text[:MAX_TEXT_CHARS * 3]
    text = _URL_RE.sub(" ", text)
    text = re.sub(r'<[^>]+>', ' ', text)

    if _HAS_LANGDETECT:
        try:
            with _langdetect_lock:
                if detect(text[:500]) != "en":
                    return None
        except Exception:
            return None

    text = _MULTI_SPACE.sub(" ", text).strip()

    if len(text) < min_len:
        return None

    return text


def _extract_with_readability(html: str) -> str | None:
    if not _HAS_READABILITY:
        return None
    try:
        scrubbed_html = _RE_INVALID_XML.sub("", html)
        doc = Document(scrubbed_html)
        soup = BeautifulSoup(doc.summary(), "lxml")
        text = soup.get_text(separator=" ", strip=True)
        return text if len(text) >= 150 else None
    except Exception:
        return None


def _extract_with_trafilatura(html: str) -> str | None:
    if not _HAS_TRAFILATURA:
        return None
    try:
        result = _trafilatura.extract(
            html, include_comments=False, include_tables=False,
            no_fallback=False, favor_precision=True,
        )
        return result if result and len(result) >= 150 else None
    except Exception:
        return None


def _extract_and_clean(html: str) -> str | None:
    text = _extract_with_trafilatura(html)

    if not text:
        text = _extract_with_readability(html)

    if not text:
        return None

    return clean_article_text(text)


def _is_domain_blacklisted(domain: str) -> bool:
    with _domain_fail_lock:
        return domain in _domain_blacklist


def _record_domain_fail(domain: str):
    with _domain_fail_lock:
        _domain_fail_counts[domain] += 1
        if _domain_fail_counts[domain] >= DOMAIN_FAIL_THRESH:
            _domain_blacklist.add(domain)


def _reset_domain_state():
    global _domain_locks, _domain_last_hit
    with _domain_fail_lock:
        _domain_fail_counts.clear()
        _domain_blacklist.clear()
    _domain_locks = {}
    _domain_last_hit = {}


async def _check_connectivity() -> bool:
    test_urls = ["https://www.reuters.com", "https://www.bbc.com"]
    to = aiohttp.ClientTimeout(total=10, connect=CONNECT_TIMEOUT)
    try:
        async with aiohttp.ClientSession(headers=_HEADERS) as session:
            for url in test_urls:
                try:
                    async with session.head(url, timeout=to, ssl=False) as r:
                        if r.status < 500:
                            return True
                except Exception:
                    continue
    except Exception:
        pass
    return False


async def _polite_delay(domain: str):
    if domain not in _domain_locks:
        _domain_locks[domain] = asyncio.Lock()
    async with _domain_locks[domain]:
        now = asyncio.get_event_loop().time()
        last_hit = _domain_last_hit.get(domain, 0)
        gap = now - last_hit
        if gap < DOMAIN_RATE_LIMIT:
            await asyncio.sleep(DOMAIN_RATE_LIMIT - gap)
        _domain_last_hit[domain] = asyncio.get_event_loop().time()


_extraction_lock = threading.Lock()


async def _async_fetch_one(
    session: aiohttp.ClientSession,
    cand: dict,
    timeout: aiohttp.ClientTimeout,
) -> dict | str | None:
    url = cand["url"]
    domain = cand["domain"]

    if any(whale in domain for whale in _WHALES):
        return "WHALE_SKIP"

    if _is_domain_blacklisted(domain):
        return None

    await _polite_delay(domain)

    try:
        async with session.get(url, timeout=timeout, allow_redirects=True, ssl=False) as r:
            if r.status >= 400:
                _record_domain_fail(domain)
                return None
            cl = r.headers.get("Content-Length")
            if cl and int(cl) > MAX_RESPONSE_BYTES:
                return None
            content = await r.content.read(MAX_RESPONSE_BYTES)
    except asyncio.TimeoutError:
        _record_domain_fail(domain)
        return None
    except Exception:
        _record_domain_fail(domain)
        return None

    html = content.decode("utf-8", errors="replace")
    del content

    loop = asyncio.get_event_loop()

    def _safe_extract():
        with _extraction_lock:
            return _extract_and_clean(html)

    try:
        clean = await loop.run_in_executor(None, _safe_extract)
    except Exception:
        return "QUALITY_REJECT"

    if not clean:
        return "QUALITY_REJECT"

    cand = dict(cand)
    cand["text"] = clean
    return cand


def _embed_batch(embed_fn, model, texts: list) -> np.ndarray:
    results = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        sub = texts[i:i + EMBED_BATCH_SIZE]
        results.append(np.asarray(embed_fn(model, sub), dtype=np.float32))
    return np.concatenate(results, axis=0) if results else np.empty((0, 768), dtype=np.float32)


async def _async_fetch_all_pipelined(
    candidates: list,
    prog: ProgressTracker,
    embed_fn,
    model,
    timeout: int = FETCH_TIMEOUT,
):
    to = aiohttp.ClientTimeout(total=timeout, connect=CONNECT_TIMEOUT)
    conn = aiohttp.TCPConnector(
        limit=FETCH_WORKERS,
        ttl_dns_cache=DNS_CACHE_TTL,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=False,
    )

    import queue as _queue
    chunk_ready: _queue.Queue = _queue.Queue(maxsize=QUEUE_MAXSIZE)
    SENTINEL = object()
    embed_thread_done = threading.Event()

    def embed_db_worker():
        if HAS_GPU:
            stream = torch.cuda.Stream()
        else:
            stream = None

        while True:
            chunk = chunk_ready.get()
            if chunk is SENTINEL:
                break
            if not chunk:
                continue

            prog.set_phase("embed+db")
            texts = [a["text"] for a in chunk]

            try:
                if HAS_GPU and stream is not None:
                    ctx = torch.cuda.stream(stream)
                else:
                    ctx = nullcontext()

                with torch.inference_mode(), ctx:
                    emb_arr = _embed_batch(embed_fn, model, texts)

                if HAS_GPU and stream is not None:
                    stream.synchronize()

                for j, article in enumerate(chunk):
                    article["embedding"] = emb_arr[j]
                prog.inc("embedded", len(chunk))

            except Exception as e:
                print(f"[Watchdog] Embed batch failed: {e}", flush=True)
                prog.inc("embed_fail", len(chunk))
                continue

            try:
                database.batch_insert_articles([
                    {
                        "url": a["url"],
                        "domain": a["domain"],
                        "newsguard_score": a["score"],
                        "text": a["text"],
                        "embedding": a["embedding"],
                        "original_did": a.get("original_did", ""),
                        "new_did": a.get("new_did", ""),
                        "created_at": a.get("created_at", ""),
                        "like_count": a.get("likes", 0),
                        "repost_count": a.get("reposts", 0),
                    }
                    for a in chunk if "embedding" in a
                ])
                prog.inc("inserted", len([a for a in chunk if "embedding" in a]))
            except Exception as e:
                print(f"[Watchdog] DB insert failed: {e}", flush=True)
                prog.inc("db_fail", len(chunk))

        embed_thread_done.set()

    embed_thread = threading.Thread(target=embed_db_worker, daemon=True)
    embed_thread.start()

    try:
        async with aiohttp.ClientSession(connector=conn, headers=_HEADERS) as session:
            sem = asyncio.Semaphore(FETCH_WORKERS)
            buffer = []
            buf_lock = asyncio.Lock()

            async def _guarded(cand):
                async with sem:
                    result = await _async_fetch_one(session, cand, to)
                if result == "WHALE_SKIP":
                    prog.inc("skipped_whales")
                    return None
                elif result == "QUALITY_REJECT":
                    prog.inc("quality_rejected")
                    return None
                elif result:
                    prog.inc("fetched")
                else:
                    prog.inc("fetch_fail")
                return result

            async def _flush_if_ready():
                nonlocal buffer
                async with buf_lock:
                    if len(buffer) >= PIPELINE_CHUNK:
                        chunk = buffer[:PIPELINE_CHUNK]
                        buffer = buffer[PIPELINE_CHUNK:]
                        try:
                            chunk_ready.put_nowait(chunk)
                        except _queue.Full:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, chunk_ready.put, chunk)

            tasks = [asyncio.create_task(_guarded(c)) for c in candidates]
            for coro in asyncio.as_completed(tasks):
                try:
                    r = await coro
                    if r and isinstance(r, dict):
                        async with buf_lock:
                            buffer.append(r)
                        await _flush_if_ready()
                except Exception:
                    prog.inc("fetch_fail")

            async with buf_lock:
                if buffer:
                    try:
                        chunk_ready.put_nowait(buffer)
                    except _queue.Full:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, chunk_ready.put, buffer)
    finally:
        chunk_ready.put(SENTINEL)
        embed_thread_done.wait(timeout=60)


async def _async_resolve_one(
    session: aiohttp.ClientSession,
    url: str,
    timeout: aiohttp.ClientTimeout,
    sem: asyncio.Semaphore,
) -> tuple[str, str]:
    async with sem:
        try:
            async with session.head(url, allow_redirects=True, timeout=timeout, ssl=False) as r:
                if r.status == 405:
                    async with session.get(url, allow_redirects=True, timeout=timeout, ssl=False) as r2:
                        return url, str(r2.url)
                return url, str(r.url)
        except Exception:
            return url, url


async def _resolve_batch_async(urls: list[str], timeout: int = 10) -> dict[str, str]:
    if not urls:
        return {}
    to = aiohttp.ClientTimeout(total=timeout, connect=CONNECT_TIMEOUT)
    conn = aiohttp.TCPConnector(limit=RESOLVE_CONCURRENCY, ttl_dns_cache=DNS_CACHE_TTL, ssl=False)
    sem = asyncio.Semaphore(RESOLVE_CONCURRENCY)
    try:
        async with aiohttp.ClientSession(connector=conn, headers=_HEADERS) as session:
            results = await asyncio.gather(
                *[_async_resolve_one(session, u, to, sem) for u in urls],
                return_exceptions=True,
            )
        return {r[0]: r[1] for r in results if isinstance(r, tuple)}
    finally:
        await conn.close()


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
        return result
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()


def _normalise_domain_plain(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("http"):
        try:
            raw = urlparse(raw).netloc
        except Exception:
            return ""
    return re.sub(r"^www\.", "", raw).rstrip("/")


def load_shortener_domains(csv_path: str) -> set:
    import pandas as pd
    shorteners: set[str] = set()
    if not os.path.exists(csv_path):
        return shorteners
    try:
        df = pd.read_csv(csv_path, header=None)
        for val in df[0].dropna():
            shorteners.add(_normalise_domain_plain(str(val)))
    except Exception:
        pass
    return shorteners


def load_newsguard(newsguard_dir: str) -> dict:
    import pandas as pd
    meta_path = os.path.join(newsguard_dir, "metadata.csv")
    mapping: dict[str, float] = {}
    if not os.path.exists(meta_path):
        return mapping
    df = pd.read_csv(meta_path, low_memory=False, usecols=["Domain", "Score"])
    for _, row in df.dropna(subset=["Score"]).iterrows():
        k = _normalise_domain_plain(str(row["Domain"]))
        if k:
            try:
                mapping[k] = float(row["Score"])
            except ValueError:
                pass
    return mapping


def _signal_analysis_if_needed():
    global _files_processed_since_analysis
    with _files_processed_lock:
        _files_processed_since_analysis += 1
        if _files_processed_since_analysis >= ANALYSIS_SIGNAL_INTERVAL:
            _files_processed_since_analysis = 0
            _do_signal_analysis()
        else:
            print(
                f"[Watchdog] Files until next analysis: "
                f"{ANALYSIS_SIGNAL_INTERVAL - _files_processed_since_analysis}",
                flush=True
            )


def _do_signal_analysis():
    running_file = "./clusters/.analysis_running"
    if os.path.exists(running_file):
        print("[Watchdog] Analysis already running, skipping signal", flush=True)
        return

    os.makedirs("./clusters", exist_ok=True)
    open(SIGNAL_FILE, "w").close()
    print("[Watchdog] Signaled analysis needed", flush=True)


def load_embedding_model(embed_dir: str, weights_path: str):
    global _embedding_module
    device = _get_device()
    embed_dir = os.path.abspath(embed_dir)
    if embed_dir not in sys.path:
        sys.path.insert(0, embed_dir)
    src_path = os.path.join(embed_dir, "embeddings.py")
    with open(src_path, encoding="utf-8") as f:
        source = f.read()
    source = re.sub(
        r"torch\.load\([^)]+\)",
        f"torch.load('{weights_path}', map_location=device, weights_only=False)",
        source,
    )
    mod = types.ModuleType("embeddings")
    mod.__file__ = src_path
    mod.__dict__["device"] = device
    sys.modules["embeddings"] = mod
    exec(compile(source, src_path, "exec"), mod.__dict__)
    _embedding_module = mod
    embed_fn = mod.get_sentence_embeddings
    model = mod.model
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return embed_fn, model


def _open_file(filepath: str):
    if not filepath.endswith(".gz"):
        return open(filepath, "rt", encoding="utf-8", errors="replace")
    if _HAS_MGZIP:
        return _mgzip.open(filepath, "rt", encoding="utf-8", errors="replace",
                          thread=MGZIP_THREADS)
    return gzip.open(filepath, "rt", encoding="utf-8", errors="replace")


def _engagement_from_record(rec: dict) -> tuple:
    new_type = rec.get("newType", "")
    explicit_likes = int(rec.get("likeCount", 0) or 0)
    explicit_reposts = int(rec.get("repostCount", 0) or 0)
    event_likes = 1 if new_type == _BSKY_LIKE_TYPE else 0
    event_reposts = 1 if new_type == _BSKY_REPOST_TYPE else 0

    original_did = rec.get("originalDid", "")
    new_did = rec.get("newDid", "")
    created_at = rec.get("newCreatedAt", "")

    return (
        explicit_likes + event_likes,
        explicit_reposts + event_reposts,
        original_did,
        new_did,
        new_type,
        created_at,
    )


def _ingest_single_pass(
    filepath: str,
    filename: str,
    ng_map: dict,
    shorteners: set,
    embed_fn,
    model,
    prog: ProgressTracker,
):
    url_to_candidate: dict[str, dict] = {}
    interactions_buffer: list[dict] = []
    pending_short: list[str] = []
    resolved_map: dict[str, str] = {}
    resolve_futures: list = []

    from concurrent.futures import ThreadPoolExecutor
    _resolve_pool = ThreadPoolExecutor(max_workers=RESOLVE_POOL_WORKERS, thread_name_prefix="resolver")

    def _flush_resolve_batch(urls: list[str]):
        prog.inc("resolve_batches_sent")
        return _resolve_pool.submit(_run_async, _resolve_batch_async(list(urls)))

    def _collect_resolved():
        still_running = []
        for fut, orig_urls in resolve_futures:
            if fut.done():
                try:
                    result = fut.result()
                    resolved_map.update(result)
                    prog.inc("resolved", len(result))
                except Exception:
                    pass
            else:
                still_running.append((fut, orig_urls))
        resolve_futures.clear()
        resolve_futures.extend(still_running)

    def _wait_all_resolved(timeout=60):
        print(f"[Watchdog] Waiting for {len(resolve_futures)} resolve futures...", flush=True)
        for i, (fut, _) in enumerate(resolve_futures):
            try:
                result = fut.result(timeout=timeout)
                resolved_map.update(result)
                prog.inc("resolved", len(result))
            except Exception as e:
                print(f"[Watchdog] Resolve future {i} failed: {e}", flush=True)
        resolve_futures.clear()
        print(f"[Watchdog] All resolve futures complete", flush=True)

    def _add_or_accumulate(final_url, final_domain, score, likes, reposts, original_did, new_did, interaction_type, created_at):
        normalized_url = normalise_url(final_url)

        if normalized_url in url_to_candidate:
            url_to_candidate[normalized_url]["likes"] += likes
            url_to_candidate[normalized_url]["reposts"] += reposts
            if created_at and (not url_to_candidate[normalized_url]["created_at"] or created_at < url_to_candidate[normalized_url]["created_at"]):
                url_to_candidate[normalized_url]["created_at"] = created_at
        else:
            url_to_candidate[normalized_url] = {
                "url": normalized_url,
                "domain": final_domain,
                "score": score,
                "likes": likes,
                "reposts": reposts,
                "original_did": original_did,
                "new_did": new_did,
                "created_at": created_at,
            }
            prog.inc("candidates")

        if new_did and interaction_type:
            interactions_buffer.append({
                "url": normalized_url,
                "actor_did": new_did,
                "interaction_type": interaction_type,
                "created_at": created_at,
            })

    try:
        with _open_file(filepath) as f:
            for raw_line in f:
                if not raw_line or raw_line == "\n":
                    continue
                prog.inc("lines_read")
                n = prog.snapshot()["lines_read"]
                if n % 500_000 == 0:
                    _collect_resolved()
                    if n % 2_000_000 == 0:
                        gc.collect()

                try:
                    rec = _json_loads(raw_line)
                    likes, reposts, original_did, new_did, interaction_type, created_at = _engagement_from_record(rec)
                    urls = rec.get("urls")
                    if not urls:
                        continue
                except Exception:
                    continue

                for raw_url in urls:
                    if not raw_url:
                        continue

                    domain = _normalise_domain(raw_url)

                    if domain in shorteners:
                        if raw_url not in resolved_map and raw_url not in pending_short:
                            pending_short.append(raw_url)
                            prog.inc("pending_resolve")
                            if len(pending_short) >= RESOLVE_BATCH_SIZE:
                                batch = pending_short[:]
                                pending_short.clear()
                                resolve_futures.append(
                                    (_flush_resolve_batch(batch), batch)
                                )
                        final_url = resolved_map.get(raw_url)
                        if final_url is None:
                            continue
                    else:
                        final_url = raw_url

                    final_domain = _normalise_domain(final_url)
                    parent = _parent_domain(final_domain)
                    score = ng_map.get(final_domain) or ng_map.get(parent)
                    if score is None or score < 0:
                        continue

                    _add_or_accumulate(final_url, final_domain, score, likes, reposts, original_did, new_did, interaction_type, created_at)

    except Exception as e:
        print(f"[Watchdog] Error reading file: {e}", flush=True)
        _resolve_pool.shutdown(wait=False)
        return None, None

    if pending_short:
        resolve_futures.append((_flush_resolve_batch(pending_short), pending_short))

    _wait_all_resolved()
    _resolve_pool.shutdown(wait=True)

    if resolved_map:
        try:
            with _open_file(filepath) as f:
                for raw_line in f:
                    if not raw_line or raw_line == "\n":
                        continue
                    try:
                        rec = _json_loads(raw_line)
                        urls = rec.get("urls")
                        if not urls:
                            continue
                        likes, reposts, original_did, new_did, interaction_type, created_at = _engagement_from_record(rec)
                    except Exception:
                        continue

                    for raw_url in urls:
                        if not raw_url:
                            continue
                        if _normalise_domain(raw_url) not in shorteners:
                            continue
                        final_url = resolved_map.get(raw_url, raw_url)
                        final_domain = _normalise_domain(final_url)
                        parent = _parent_domain(final_domain)
                        score = ng_map.get(final_domain) or ng_map.get(parent)
                        if score is None or score < 0:
                            continue
                        _add_or_accumulate(final_url, final_domain, score, likes, reposts, original_did, new_did, interaction_type, created_at)
        except Exception as e:
            print(f"[Watchdog] Error in second pass: {e}", flush=True)

    gc.collect()

    candidates = list(url_to_candidate.values())
    print(
        f"[Watchdog] Scan complete: {len(candidates):,} unique URLs, "
        f"total likes={sum(c['likes'] for c in candidates):,} "
        f"reposts={sum(c['reposts'] for c in candidates):,} "
        f"interactions={len(interactions_buffer):,}",
        flush=True,
    )
    return candidates, interactions_buffer


class BlueskyHandler(FileSystemEventHandler):
    def __init__(self, ng_map, shorteners, embed_fn, model):
        self.ng_map = ng_map
        self.shorteners = shorteners
        self.embed_fn = embed_fn
        self.model = model
        self._processing_lock = threading.Lock()

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".gz"):
            return
        time.sleep(2)
        self.process_file(event.src_path)

    def process_file(self, filepath):
        if not self._processing_lock.acquire(blocking=False):
            print(f"[Watchdog] Already processing a file, skipping {filepath}", flush=True)
            return

        try:
            filename = os.path.basename(filepath)
            if database.is_file_processed(filename):
                print(f"[Watchdog] Skipping already processed: {filename}", flush=True)
                return
            try:
                self._ingest(filepath, filename)
            except Exception as exc:
                print(f"[Watchdog] ERROR processing {filename}: {exc}", flush=True)
                import traceback
                traceback.print_exc()
        finally:
            self._processing_lock.release()
            gc.collect()
            if HAS_GPU:
                torch.cuda.empty_cache()

    def _ingest(self, filepath, filename):
        _reset_domain_state()

        prog = ProgressTracker(filename, total=0)
        prog.set_phase("scan+resolve")

        result = _ingest_single_pass(
            filepath, filename,
            self.ng_map, self.shorteners,
            self.embed_fn, self.model,
            prog,
        )

        if result is None or result[0] is None:
            prog.stop()
            print(f"[Watchdog] Failed to process {filename}", flush=True)
            return

        candidates, interactions_buffer = result

        all_urls = [c["url"] for c in candidates]

        prog.set_phase("db_dedup")
        existing = database.get_existing_urls(all_urls)
        new_candidates = [c for c in candidates if c["url"] not in existing]
        stat_updates = [
            (c["url"], c["likes"], c["reposts"])
            for c in candidates if c["url"] in existing
        ]
        prog.inc("existing", len(existing))
        prog.inc("new", len(new_candidates))

        database.batch_update_stats(stat_updates)

        if interactions_buffer:
            prog.set_phase("interactions")
            inserted_interactions = database.batch_insert_interactions(interactions_buffer)
            prog.inc("interactions_recorded", inserted_interactions)
            print(f"[Watchdog] Recorded {inserted_interactions:,} interactions", flush=True)

        del candidates
        del all_urls
        del interactions_buffer
        gc.collect()

        if not new_candidates:
            prog.stop()
            database.mark_file_processed(filename)
            print(f"[Watchdog] Finished {filename} (no new candidates)", flush=True)
            _signal_analysis_if_needed()
            return

        prog.set_phase("connectivity_check")
        if not _run_async(_check_connectivity()):
            prog.stop()
            print(f"[Watchdog] Connectivity check failed for {filename}", flush=True)
            return

        random.seed(42)
        random.shuffle(new_candidates)
        prog.set_phase("fetching")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _async_fetch_all_pipelined(
                    new_candidates, prog, self.embed_fn, self.model,
                    timeout=FETCH_TIMEOUT,
                )
            )
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()

        snap = prog.snapshot()
        inserted = snap["inserted"]
        fetched = snap["fetched"]
        fetch_fail = snap["fetch_fail"]
        quality_rej = snap["quality_rejected"]
        skipped = snap["skipped_whales"]
        interactions_rec = snap["interactions_recorded"]

        prog.stop()

        if inserted > 0 or len(new_candidates) == 0:
            database.mark_file_processed(filename)

        print(
            f"[Watchdog] Finished {filename}: "
            f"{inserted} inserted, {fetched} fetched, "
            f"{fetch_fail} failed, {quality_rej} quality_rejected, {skipped} skipped, "
            f"{interactions_rec} interactions",
            flush=True
        )

        _signal_analysis_if_needed()


if __name__ == "__main__":
    safe_threads = max(2, CPU_COUNT // 3)
    torch.set_num_threads(safe_threads)
    torch.set_num_interop_threads(max(1, safe_threads // 2))

    FOLDER_TO_WATCH = "./data"
    os.makedirs(FOLDER_TO_WATCH, exist_ok=True)

    database.init_db()
    _get_session()

    ng_map = load_newsguard("./NewsGuard")
    shorteners = load_shortener_domains("./NewsGuard/shorturl-services-list.csv")
    embed_fn, model = load_embedding_model(
        "./embedding_model",
        "./embedding_model/specious_model_weights.pt",
    )

    event_handler = BlueskyHandler(ng_map, shorteners, embed_fn, model)

    files_to_process = [
        f for f in sorted(os.listdir(FOLDER_TO_WATCH))
        if f.endswith(".gz") and not database.is_file_processed(f)
    ]

    print(f"[Watchdog] Found {len(files_to_process)} files to process", flush=True)

    for filename in files_to_process:
        event_handler.process_file(os.path.join(FOLDER_TO_WATCH, filename))

    print("[Watchdog] Initial processing complete. Exiting.", flush=True)
    sys.exit(0)