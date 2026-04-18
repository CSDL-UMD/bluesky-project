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
import ssl
from collections import defaultdict
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from contextlib import nullcontext
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import numpy as np
import database

_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE

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
    FETCH_WORKERS        = 80
    LIMIT_PER_HOST       = 10
    FETCH_TIMEOUT        = 20
    CONNECT_TIMEOUT      = 8
    SOCK_READ_TIMEOUT    = 15 
    DOMAIN_FAIL_THRESH   = 200
    DOMAIN_RATE_LIMIT    = 0.1
    RESOLVE_CONCURRENCY  = 150
    RESOLVE_WORKERS      = 100
    MAX_RESPONSE_BYTES   = 3_000_000
    EMBED_BATCH_SIZE     = 512
    PIPELINE_CHUNK       = 64
    QUEUE_MAXSIZE        = 16
    RESOLVE_POOL_WORKERS = min(CPU_COUNT, 8)
    MGZIP_THREADS        = min(CPU_COUNT, 8)
    RESOLVE_BATCH_SIZE   = 1000
    EXTRACT_WORKERS      = 8
    READ_CHUNK_SIZE      = 1024 * 1024
else:
    FETCH_WORKERS        = 50
    LIMIT_PER_HOST       = 8
    EMBED_BATCH_SIZE     = 32
    FETCH_TIMEOUT        = 12
    RESOLVE_WORKERS      = 100
    PIPELINE_CHUNK       = 32
    CONNECT_TIMEOUT      = 5
    SOCK_READ_TIMEOUT    = 10
    MAX_RESPONSE_BYTES   = 2_000_000
    RESOLVE_CONCURRENCY  = 100
    DOMAIN_RATE_LIMIT    = 0.05
    QUEUE_MAXSIZE        = 8
    DOMAIN_FAIL_THRESH   = 100
    RESOLVE_POOL_WORKERS = 4
    MGZIP_THREADS        = 2
    RESOLVE_BATCH_SIZE   = 1000
    EXTRACT_WORKERS      = 4
    READ_CHUNK_SIZE      = 512 * 1024

MAX_TEXT_CHARS = 3000
PROGRESS_EVERY = 10
DNS_CACHE_TTL = 1800
SIGNAL_FILE = "./clusters/.analysis_needed"
PASSAGE_WORD_LIMIT = 100
ANALYSIS_SIGNAL_INTERVAL = 72

_BSKY_LIKE_TYPE   = "app.bsky.feed.like"
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
    target = min(hard, 131072)
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
    print("[Watchdog] Using uvloop for faster async", flush=True)
except ImportError:
    pass

try:
    import orjson as _json_lib
    def _json_loads(b): return _json_lib.loads(b)
    print("[Watchdog] Using orjson for faster JSON parsing", flush=True)
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
    from bs4 import BeautifulSoup, SoupStrainer
    _HAS_READABILITY = True
    _BODY_STRAINER = SoupStrainer(['p', 'article', 'div', 'section', 'main'])
except ImportError:
    _HAS_READABILITY = False
    _BODY_STRAINER = None

try:
    import trafilatura as _trafilatura
    from trafilatura.settings import use_config
    _TRAF_CONFIG = use_config()
    _TRAF_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "5")
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False
    _TRAF_CONFIG = None

if not _HAS_READABILITY and not _HAS_TRAFILATURA:
    print("[Watchdog] FATAL: neither readability-lxml nor trafilatura available.", flush=True)
    sys.exit(1)

try:
    import aiodns
    _HAS_AIODNS = True
except ImportError:
    _HAS_AIODNS = False

_embedding_module = None
_http_session = None
_http_session_lock = threading.Lock()
_DEVICE = None

_domain_fail_counts: dict[str, int] = defaultdict(int)
_domain_fail_lock = threading.Lock()
_domain_blacklist: set[str] = set()

_files_processed_since_analysis = 0
_files_processed_lock = threading.Lock()

_EXTRACT_POOL: ThreadPoolExecutor | None = None
_EXTRACT_POOL_LOCK = threading.Lock()

_WHALES = frozenset(["reuters.com", "bloomberg.com", "nytimes.com", "wsj.com", "ft.com"])

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

_dns_cache: dict[str, tuple[str, float]] = {}
_dns_cache_lock = threading.Lock()

_MULTI_SPACE    = re.compile(r'\s+')
_URL_RE         = re.compile(r'https?://\S+', re.IGNORECASE)
_WWW_RE         = re.compile(r"^www\.")
_HTML_TAG_RE    = re.compile(r'<[^>]+>')

_error_counts: dict[str, int] = defaultdict(int)
_error_lock = threading.Lock()


def _log_error_type(error_type: str):
    with _error_lock:
        _error_counts[error_type] += 1


def _print_error_summary():
    with _error_lock:
        if _error_counts:
            print(f"[Watchdog] Error summary: {dict(_error_counts)}", flush=True)
            _error_counts.clear()


def _get_extract_pool() -> ThreadPoolExecutor:
    global _EXTRACT_POOL
    if _EXTRACT_POOL is None:
        with _EXTRACT_POOL_LOCK:
            if _EXTRACT_POOL is None:
                _EXTRACT_POOL = ThreadPoolExecutor(
                    max_workers=EXTRACT_WORKERS,
                    thread_name_prefix="extractor",
                )
    return _EXTRACT_POOL


@lru_cache(maxsize=1 << 20)
def normalise_url(url: str) -> str:
    if not url:
        return url
    try:
        parsed = urlparse(url)
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=False)
            filtered = {k: v for k, v in params.items() if k.lower() not in _TRACKING_PARAMS}
            new_query = urlencode(filtered, doseq=True) if filtered else ''
        else:
            new_query = ''
        return urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path.rstrip('/') if parsed.path != '/' else '/',
            parsed.params,
            new_query,
            '',
        ))
    except Exception:
        return url


@lru_cache(maxsize=1 << 18)
def _normalise_domain(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("http"):
        try:
            raw = urlparse(raw).netloc
        except Exception:
            return ""
    return _WWW_RE.sub("", raw).rstrip("/")


@lru_cache(maxsize=1 << 18)
def _parent_domain(domain: str) -> str:
    parts = domain.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else domain


@lru_cache(maxsize=1 << 16)
def _extract_domain_from_url(url: str) -> str:
    try:
        if url.startswith("http://"):
            start = 7
        elif url.startswith("https://"):
            start = 8
        else:
            return ""
        end = url.find("/", start)
        if end == -1:
            end = len(url)
        domain = url[start:end].lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "done"
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{seconds:.0f}s"


class ProgressTracker:
    __slots__ = ('filename', 'total', 't_start', '_lock', '_counters', '_stop', '_thread')

    def __init__(self, filename: str, total: int):
        self.filename  = filename
        self.total     = total
        self.t_start   = time.time()
        self._lock     = threading.Lock()
        self._counters = dict(
            lines_read=0, candidates=0, existing=0, new=0,
            fetched=0, fetch_fail=0, embedded=0, embed_fail=0,
            inserted=0, db_fail=0, phase="init", skipped_whales=0,
            quality_rejected=0,
            pending_resolve=0, resolved=0, resolve_batches_sent=0,
            interactions_recorded=0,
        )
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._reporter, daemon=True)
        self._thread.start()

    def set_phase(self, phase: str):
        with self._lock:
            self._counters["phase"] = phase
        print(f"[Watchdog] ── phase: {phase} ──", flush=True)

    def inc(self, key: str, n: int = 1):
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + n

    def inc_multi(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                self._counters[k] = self._counters.get(k, 0) + v

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
        snap    = self.snapshot()
        elapsed = time.time() - self.t_start
        phase   = snap["phase"]

        if phase == "scan+resolve":
            pending  = snap.get("pending_resolve", 0)
            resolved = snap.get("resolved", 0)
            batches  = snap.get("resolve_batches_sent", 0)
            resolve_rate = resolved / elapsed if elapsed > 0 else 0
            print(
                f"[Watchdog] [{phase:14s}] "
                f"read={snap['lines_read']:,} cands={snap['candidates']:,} | "
                f"resolve: {resolved:,}/{pending:,} pending ({resolve_rate:.0f}/s) batches={batches}",
                flush=True,
            )
        else:
            new         = snap["new"]
            fetched     = snap["fetched"]
            fail        = snap["fetch_fail"]
            quality_rej = snap.get("quality_rejected", 0)
            done        = fetched + fail + quality_rej
            rate        = done / elapsed if elapsed > 0 else 0
            eta         = _fmt_eta((new - done) / rate if rate > 0 and new > done else 0)
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
        pool_maxsize=FETCH_WORKERS * 2,
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


def clean_article_text(text: str, min_len: int = 150) -> str | None:
    if not text:
        return None
    text = text[:MAX_TEXT_CHARS * 3]
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(' ', text)

    if _HAS_LANGDETECT:
        try:
            with _langdetect_lock:
                if detect(text[:300]) != "en":
                    return None
        except Exception:
            return None

    text = _MULTI_SPACE.sub(" ", text).strip()
    return text if len(text) >= min_len else None


def _extract_with_readability(html: str) -> str | None:
    if not _HAS_READABILITY:
        return None
    try:
        scrubbed_html = _RE_INVALID_XML.sub("", html)
        doc  = Document(scrubbed_html)
        soup = BeautifulSoup(doc.summary(), "lxml", parse_only=_BODY_STRAINER)
        text = soup.get_text(separator=" ", strip=True)
        return text if len(text) >= 150 else None
    except Exception:
        return None


def _extract_with_trafilatura(html: str) -> str | None:
    if not _HAS_TRAFILATURA:
        return None
    try:
        result = _trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=True,
            favor_precision=False,
            config=_TRAF_CONFIG,
        )
        return result if result and len(result) >= 150 else None
    except Exception:
        return None


def _extract_and_clean(html: str) -> str | None:
    try:
        text = _extract_with_trafilatura(html)
        if not text:
            text = _extract_with_readability(html)
        if not text:
            return None
        return clean_article_text(text)
    except Exception:
        return None


def _is_domain_blacklisted(domain: str) -> bool:
    with _domain_fail_lock:
        return domain in _domain_blacklist


def _record_domain_fail(domain: str):
    with _domain_fail_lock:
        _domain_fail_counts[domain] += 1
        if _domain_fail_counts[domain] >= DOMAIN_FAIL_THRESH:
            _domain_blacklist.add(domain)


def _reset_domain_state():
    with _domain_fail_lock:
        _domain_fail_counts.clear()
        _domain_blacklist.clear()


async def _check_connectivity() -> bool:
    test_urls = ["https://httpbin.org/get", "https://example.com", "https://www.google.com"]
    to = aiohttp.ClientTimeout(total=10, connect=5)
    try:
        conn = aiohttp.TCPConnector(ssl=_SSL_CONTEXT)
        async with aiohttp.ClientSession(headers=_HEADERS, connector=conn) as session:
            for url in test_urls:
                try:
                    async with session.get(url, timeout=to) as r:
                        if r.status == 200:
                            print(f"[Watchdog] Connectivity confirmed via {url}", flush=True)
                            return True
                except Exception as e:
                    print(f"[Watchdog] Connectivity test failed for {url}: {type(e).__name__}", flush=True)
                    continue
    except Exception as e:
        print(f"[Watchdog] Connectivity check error: {e}", flush=True)
    return False


async def _async_fetch_one(
    session: aiohttp.ClientSession,
    cand: dict,
    timeout: aiohttp.ClientTimeout,
    extract_pool: ThreadPoolExecutor,
    sem: asyncio.Semaphore,
) -> dict | str | None:
    url    = cand["url"]
    domain = cand["domain"]

    if any(whale in domain for whale in _WHALES):
        return "WHALE_SKIP"

    if _is_domain_blacklisted(domain):
        return "BLACKLISTED"

    async with sem:
        try:
            async with session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                max_redirects=5,      
            ) as r:
                if r.status >= 400:
                    _log_error_type(f"http_{r.status}")
                    if r.status in (403, 404, 410, 451):
                        return None
                    _record_domain_fail(domain)
                    return None

                ct = r.headers.get("Content-Type", "")
                if ct and not any(t in ct for t in ("text/html", "text/plain", "application/xhtml")):
                    _log_error_type("bad_content_type")
                    return None

                cl = r.headers.get("Content-Length")
                if cl and int(cl) > MAX_RESPONSE_BYTES:
                    _log_error_type("too_large")
                    return None

                content = await r.content.read(MAX_RESPONSE_BYTES)

        except asyncio.TimeoutError:
            _log_error_type("timeout")
            _record_domain_fail(domain)
            return None
        except aiohttp.TooManyRedirects:
            _log_error_type("too_many_redirects")
            return None
        except aiohttp.ClientSSLError:
            _log_error_type("ssl_error")
            return None
        except aiohttp.ClientConnectorError:
            _log_error_type("connector_error")
            _record_domain_fail(domain)
            return None
        except aiohttp.ServerDisconnectedError:
            _log_error_type("server_disconnected")
            return None
        except Exception as e:
            _log_error_type(f"other_{type(e).__name__}")
            _record_domain_fail(domain)
            return None

    if not content:
        _log_error_type("empty_response")
        return None

    html = content.decode("utf-8", errors="ignore")
    del content

    if len(html) < 500:
        _log_error_type("html_too_short")
        return "QUALITY_REJECT"

    loop = asyncio.get_event_loop()
    try:
        clean = await loop.run_in_executor(extract_pool, _extract_and_clean, html)
    except Exception:
        _log_error_type("extract_error")
        return "QUALITY_REJECT"

    if not clean:
        return "QUALITY_REJECT"

    cand        = dict(cand)
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
    extract_pool = _get_extract_pool()

    to = aiohttp.ClientTimeout(
        total=timeout,
        connect=CONNECT_TIMEOUT,
        sock_read=SOCK_READ_TIMEOUT,
    )

    resolver = aiohttp.AsyncResolver() if _HAS_AIODNS else aiohttp.DefaultResolver()
    conn = aiohttp.TCPConnector(
        limit=FETCH_WORKERS * 2,
        limit_per_host=LIMIT_PER_HOST,
        ttl_dns_cache=DNS_CACHE_TTL,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=_SSL_CONTEXT,
        resolver=resolver,
    )

    import queue as _queue
    chunk_ready: _queue.Queue = _queue.Queue(maxsize=QUEUE_MAXSIZE)
    SENTINEL = object()
    embed_thread_done = threading.Event()

    def embed_db_worker():
        stream = torch.cuda.Stream() if HAS_GPU else None

        while True:
            chunk = chunk_ready.get()
            if chunk is SENTINEL:
                break
            if not chunk:
                continue

            prog.set_phase("embed+db")
            texts = [a["text"] for a in chunk]

            try:
                ctx = torch.cuda.stream(stream) if (HAS_GPU and stream) else nullcontext()
                with torch.inference_mode(), ctx:
                    emb_arr = _embed_batch(embed_fn, model, texts)
                if HAS_GPU and stream:
                    stream.synchronize()
                for j, article in enumerate(chunk):
                    article["embedding"] = emb_arr[j]
                prog.inc("embedded", len(chunk))
            except Exception as e:
                print(f"[Watchdog] Embed batch failed: {e}", flush=True)
                prog.inc("embed_fail", len(chunk))
                continue

            to_insert = [a for a in chunk if "embedding" in a]
            try:
                database.batch_insert_articles([
                    {
                        "url":              a["url"],
                        "domain":           a["domain"],
                        "newsguard_score":  a["score"],
                        "text":             a["text"],
                        "embedding":        a["embedding"],
                        "original_did":     a.get("original_did", ""),
                        "new_did":          a.get("new_did", ""),
                        "created_at":       a.get("created_at", ""),
                        "like_count":       a.get("likes", 0),
                        "repost_count":     a.get("reposts", 0),
                    }
                    for a in to_insert
                ])
                prog.inc("inserted", len(to_insert))
            except Exception as e:
                print(f"[Watchdog] DB insert failed: {e}", flush=True)
                prog.inc("db_fail", len(to_insert))

        embed_thread_done.set()

    embed_thread = threading.Thread(target=embed_db_worker, daemon=True)
    embed_thread.start()

    try:
        async with aiohttp.ClientSession(connector=conn, headers=_HEADERS) as session:
            sem    = asyncio.Semaphore(FETCH_WORKERS)
            buffer = []

            def _flush_buffer():
                nonlocal buffer
                while len(buffer) >= PIPELINE_CHUNK:
                    chunk  = buffer[:PIPELINE_CHUNK]
                    buffer = buffer[PIPELINE_CHUNK:]
                    try:
                        chunk_ready.put_nowait(chunk)
                    except _queue.Full:
                        chunk_ready.put(chunk)

            async def _guarded(cand):
                result = await _async_fetch_one(session, cand, to, extract_pool, sem)
                if result == "WHALE_SKIP":
                    prog.inc("skipped_whales")
                    return None
                elif result == "BLACKLISTED":
                    prog.inc("fetch_fail")
                    return None
                elif result == "QUALITY_REJECT":
                    prog.inc("quality_rejected")
                    return None
                elif result:
                    prog.inc("fetched")
                    return result
                else:
                    prog.inc("fetch_fail")
                    return None
            tasks = [asyncio.create_task(_guarded(c)) for c in candidates]

            for fut in asyncio.as_completed(tasks):
                try:
                    r = await fut
                    if r is not None:
                        buffer.append(r)
                        _flush_buffer()
                except Exception:
                    prog.inc("fetch_fail")

            if buffer:
                try:
                    chunk_ready.put_nowait(buffer)
                except _queue.Full:
                    chunk_ready.put(buffer)

    finally:
        chunk_ready.put(SENTINEL)
        embed_thread_done.wait(timeout=120)
        _print_error_summary()


async def _async_resolve_batch(
    session: aiohttp.ClientSession,
    urls: list[str],
    timeout: aiohttp.ClientTimeout,
    sem: asyncio.Semaphore,
) -> dict[str, str]:
    results = {}

    async def resolve_one(url: str) -> tuple[str, str]:
        async with sem:
            try:
                async with session.head(url, allow_redirects=True, timeout=timeout) as r:
                    return url, str(r.url)
            except Exception:
                try:
                    async with session.get(url, allow_redirects=True, timeout=timeout) as r:
                        return url, str(r.url)
                except Exception:
                    return url, url

    tasks = [resolve_one(u) for u in urls]
    for coro in asyncio.as_completed(tasks):
        try:
            orig, resolved = await coro
            results[orig] = resolved
        except Exception:
            pass

    return results


async def _resolve_all_urls(urls: list[str], prog: ProgressTracker) -> dict[str, str]:
    if not urls:
        return {}

    to       = aiohttp.ClientTimeout(total=10, connect=5)
    resolver = aiohttp.AsyncResolver() if _HAS_AIODNS else aiohttp.DefaultResolver()
    conn     = aiohttp.TCPConnector(
        limit=RESOLVE_CONCURRENCY,
        ttl_dns_cache=DNS_CACHE_TTL,
        ssl=_SSL_CONTEXT,
        resolver=resolver,
    )

    results = {}
    sem     = asyncio.Semaphore(RESOLVE_CONCURRENCY)

    async with aiohttp.ClientSession(connector=conn, headers=_HEADERS) as session:
        for i in range(0, len(urls), RESOLVE_BATCH_SIZE):
            batch        = urls[i:i + RESOLVE_BATCH_SIZE]
            prog.inc("resolve_batches_sent")
            batch_results = await _async_resolve_batch(session, batch, to, sem)
            results.update(batch_results)
            prog.inc("resolved", len(batch_results))

    return results


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
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


def load_shortener_domains(csv_path: str) -> frozenset:
    import pandas as pd
    shorteners: set[str] = set()
    if not os.path.exists(csv_path):
        return frozenset(shorteners)
    try:
        df = pd.read_csv(csv_path, header=None)
        for val in df[0].dropna():
            shorteners.add(_normalise_domain_plain(str(val)))
    except Exception:
        pass
    return frozenset(shorteners)


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
                flush=True,
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
    device    = _get_device()
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
    mod          = types.ModuleType("embeddings")
    mod.__file__ = src_path
    mod.__dict__["device"] = device
    sys.modules["embeddings"] = mod
    exec(compile(source, src_path, "exec"), mod.__dict__)
    _embedding_module = mod
    embed_fn = mod.get_sentence_embeddings
    model    = mod.model
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return embed_fn, model


def _open_file(filepath: str):
    if not filepath.endswith(".gz"):
        return open(filepath, "rt", encoding="utf-8", errors="replace", buffering=READ_CHUNK_SIZE)
    if _HAS_MGZIP:
        return _mgzip.open(filepath, "rt", encoding="utf-8", errors="replace", thread=MGZIP_THREADS)
    return gzip.open(filepath, "rt", encoding="utf-8", errors="replace")


def _engagement_from_record(rec: dict) -> tuple:
    new_type         = rec.get("newType", "")
    explicit_likes   = int(rec.get("likeCount", 0) or 0)
    explicit_reposts = int(rec.get("repostCount", 0) or 0)
    event_likes      = 1 if new_type == _BSKY_LIKE_TYPE else 0
    event_reposts    = 1 if new_type == _BSKY_REPOST_TYPE else 0
    return (
        explicit_likes   + event_likes,
        explicit_reposts + event_reposts,
        rec.get("originalDid", ""),
        rec.get("newDid", ""),
        new_type,
        rec.get("newCreatedAt", ""),
    )


def _ingest_single_pass(
    filepath: str,
    filename: str,
    ng_map: dict,
    shorteners: frozenset,
    embed_fn,
    model,
    prog: ProgressTracker,
):
    url_to_candidate: dict[str, dict] = {}
    interactions_buffer: list[dict]   = []
    short_urls_to_resolve: set[str]   = set()
    lines_since_flush = 0

    prog.set_phase("scan")

    try:
        with _open_file(filepath) as f:
            for raw_line in f:
                if not raw_line or raw_line == "\n":
                    continue
                lines_since_flush += 1

                if lines_since_flush >= 100_000:
                    prog.inc("lines_read", lines_since_flush)
                    lines_since_flush = 0

                try:
                    rec  = _json_loads(raw_line)
                    urls = rec.get("urls")
                    if not urls:
                        continue
                except Exception:
                    continue

                for raw_url in urls:
                    if not raw_url:
                        continue
                    domain = _extract_domain_from_url(raw_url)
                    if domain in shorteners:
                        short_urls_to_resolve.add(raw_url)

    except Exception as e:
        print(f"[Watchdog] Error in scan pass: {e}", flush=True)
        return None, None
    finally:
        if lines_since_flush:
            prog.inc("lines_read", lines_since_flush)

    prog.set_phase("resolve")
    prog.inc("pending_resolve", len(short_urls_to_resolve))

    if short_urls_to_resolve:
        print(f"[Watchdog] Resolving {len(short_urls_to_resolve):,} shortened URLs...", flush=True)
        resolved_map = _run_async(_resolve_all_urls(list(short_urls_to_resolve), prog))
    else:
        resolved_map = {}

    del short_urls_to_resolve
    gc.collect()

    prog.set_phase("process")
    lines_since_flush = 0

    def _add_or_accumulate(final_url, final_domain, score, likes, reposts, original_did, new_did, interaction_type, created_at):
        normalized_url = normalise_url(final_url)
        if normalized_url in url_to_candidate:
            c = url_to_candidate[normalized_url]
            c["likes"]   += likes
            c["reposts"] += reposts
            if created_at and (not c["created_at"] or created_at < c["created_at"]):
                c["created_at"] = created_at
        else:
            url_to_candidate[normalized_url] = {
                "url":          normalized_url,
                "domain":       final_domain,
                "score":        score,
                "likes":        likes,
                "reposts":      reposts,
                "original_did": original_did,
                "new_did":      new_did,
                "created_at":   created_at,
            }
            prog.inc("candidates")

        if new_did and interaction_type:
            interactions_buffer.append({
                "url":              normalized_url,
                "actor_did":        new_did,
                "interaction_type": interaction_type,
                "created_at":       created_at,
            })

    try:
        with _open_file(filepath) as f:
            for raw_line in f:
                if not raw_line or raw_line == "\n":
                    continue
                lines_since_flush += 1

                if lines_since_flush >= 100_000:
                    prog.inc("lines_read", lines_since_flush)
                    lines_since_flush = 0

                try:
                    rec  = _json_loads(raw_line)
                    likes, reposts, original_did, new_did, interaction_type, created_at = _engagement_from_record(rec)
                    urls = rec.get("urls")
                    if not urls:
                        continue
                except Exception:
                    continue

                for raw_url in urls:
                    if not raw_url:
                        continue

                    domain = _extract_domain_from_url(raw_url)

                    if domain in shorteners:
                        final_url    = resolved_map.get(raw_url, raw_url)
                        final_domain = _extract_domain_from_url(final_url)
                    else:
                        final_url    = raw_url
                        final_domain = domain

                    parent = _parent_domain(final_domain)
                    score  = ng_map.get(final_domain) or ng_map.get(parent)
                    if score is None or score < 0:
                        continue

                    _add_or_accumulate(final_url, final_domain, score, likes, reposts, original_did, new_did, interaction_type, created_at)

    except Exception as e:
        print(f"[Watchdog] Error in process pass: {e}", flush=True)
        return None, None
    finally:
        if lines_since_flush:
            prog.inc("lines_read", lines_since_flush)

    del resolved_map
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
        self.ng_map      = ng_map
        self.shorteners  = shorteners
        self.embed_fn    = embed_fn
        self.model       = model
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

        prog   = ProgressTracker(filename, total=0)
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
        existing       = database.get_existing_urls(all_urls)
        new_candidates = [c for c in candidates if c["url"] not in existing]
        stat_updates   = [(c["url"], c["likes"], c["reposts"]) for c in candidates if c["url"] in existing]
        prog.inc_multi(existing=len(existing), new=len(new_candidates))
        database.batch_update_stats(stat_updates)

        if interactions_buffer:
            prog.set_phase("interactions")
            inserted_interactions = database.batch_insert_interactions(interactions_buffer)
            prog.inc("interactions_recorded", inserted_interactions)
            print(f"[Watchdog] Recorded {inserted_interactions:,} interactions", flush=True)

        del candidates, all_urls, interactions_buffer
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
        prog.stop()

        if snap["inserted"] > 0 or len(new_candidates) == 0:
            database.mark_file_processed(filename)

        print(
            f"[Watchdog] Finished {filename}: "
            f"{snap['inserted']} inserted, {snap['fetched']} fetched, "
            f"{snap['fetch_fail']} failed, {snap['quality_rejected']} quality_rejected, "
            f"{snap['skipped_whales']} skipped, {snap['interactions_recorded']} interactions",
            flush=True,
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

    ng_map     = load_newsguard("./NewsGuard")
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