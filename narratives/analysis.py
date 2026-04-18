import os
import gc
import warnings
import random
import threading
import pickle

CPU_COUNT = os.cpu_count() or 4

os.environ["OMP_NUM_THREADS"]          = str(CPU_COUNT)
os.environ["MKL_NUM_THREADS"]          = str(CPU_COUNT)
os.environ["OPENBLAS_NUM_THREADS"]     = str(CPU_COUNT)
os.environ["VECLIB_MAXIMUM_THREADS"]   = str(CPU_COUNT)
os.environ["NUMEXPR_NUM_THREADS"]      = str(CPU_COUNT)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"]           = "0"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"]  = ":4096:8"

RANDOM_SEED = 42

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*min_new_tokens.*")
warnings.filterwarnings("ignore", message=".*forced_bos_token_id.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*token.*")

import json
import time
import re
import math
import logging
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
import torch
import database

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

from sklearn.cluster import MiniBatchDPMeans

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = RANDOM_SEED
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

HAS_GPU = torch.cuda.is_available()

_BART_PIPELINE = None

CLUSTER_CENTERS_DIR = "./clusters/centers"

def _get_bart():
    global _BART_PIPELINE
    if _BART_PIPELINE is None:
        print("[Analysis] Loading BART model...", flush=True)
        from transformers import BartForConditionalGeneration, BartTokenizer
        
        device = "cuda:0" if HAS_GPU else "cpu"
        print(f"[Analysis] Using device: {device}", flush=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenizer = BartTokenizer.from_pretrained(
                "facebook/bart-large-cnn",
                clean_up_tokenization_spaces=True,
            )
            
            if HAS_GPU:
                model = BartForConditionalGeneration.from_pretrained(
                    "facebook/bart-large-cnn",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).to(device)
            else:
                model = BartForConditionalGeneration.from_pretrained(
                    "facebook/bart-large-cnn",
                    low_cpu_mem_usage=True,
                ).to(device)
        
        print("[Analysis] BART model loaded successfully", flush=True)
        _BART_PIPELINE = (tokenizer, model, device)
    return _BART_PIPELINE


def _unload_bart():
    global _BART_PIPELINE
    if _BART_PIPELINE is not None:
        print("[Analysis] Unloading BART model to free memory...", flush=True)
        tokenizer, model, device = _BART_PIPELINE
        del model
        del tokenizer
        _BART_PIPELINE = None
        gc.collect()
        if HAS_GPU:
            torch.cuda.empty_cache()
        print("[Analysis] BART model unloaded", flush=True)


_BART_MAX_INPUT_CHARS = 5_000
_BART_MAX_NEW_TOKENS  = 250
_BART_MIN_NEW_TOKENS  = 50
SIGNAL_FILE = "./clusters/.analysis_needed"
LOCK_FILE   = "./clusters/.analysis_running"

RELIABLE_THRESHOLD = 60.0

DPMEANS_DELTA            = 0.40
DPMEANS_BATCH_SIZE       = 1024
DPMEANS_MAX_ITER         = 100
DPMEANS_N_INIT           = 1
DPMEANS_N_INTRODUCE      = 1
MIN_ARTICLES_PER_CLUSTER = 25

TOP_K_FOR_SUMMARY = 5
PMI_ALPHA         = 1

PMI_MIN_CLUSTER_COUNT  = 3
PMI_MIN_CLUSTER_DOCS   = 2
PMI_MAX_WORD_LEN       = 20
PMI_MIN_WORD_LEN       = 4

MIN_ARTICLE_SENTENCES = 5
MIN_ARTICLE_CHARS     = 200

_STOPS = frozenset({
    "the","and","for","that","this","with","are","was","were","have","has",
    "had","but","not","you","they","from","its","our","your","their","about",
    "will","can","been","more","also","when","than","then","who","what","how",
    "just","said","some","there","which","into","one","all","out","get","got",
    "did","she","him","her","his","dont","isnt","cant","wont","would","could",
    "should","may","might","let","way","now","here","very","really","like",
    "even","still","after","over","back","only","think","know","make","take",
    "trump", "donald", "president", "said", "says", "told", "according",
    "report", "reports", "reported", "new", "year", "years", "day", "days",
    "time", "times", "week", "weeks", "month", "months", "people", "man",
    "woman", "two", "three", "four", "five", "six", "first", "last",
    "former", "former", "senior", "top", "big", "major", "latest", "recent",
    "good", "bad", "old", "long", "high", "low", "large", "small", "right",
    "left", "part", "place", "fact", "case", "point", "end", "show",
    "tell", "put", "look", "come", "see", "use", "find", "give", "keep",
    "call", "try", "ask", "need", "seem", "feel", "become", "include",
    "continue", "move", "turn", "start", "play", "run", "hold",
    "unhinged", "bizarre", "frantic", "deranged", "panicking", "crazed",
    "manic", "furious", "blasts", "slams", "rips", "attacks", "hits",
    "doubles", "doubles", "pushes", "calls", "warns", "urges", "defends",
    "denies", "admits", "reveals", "claims", "says", "suggests",
})

_BOILERPLATE_FRAGMENTS = frozenset([
    "please enable javascript", "all rights reserved", "terms of service",
    "privacy policy", "about press copyright", "contact us",
    "how youtube works", "test new features", "subscribe to our newsletter",
    "disable any ad blockers", "browser extension", "supported browsers",
    "offline for up to 24 hours", "donate by check", "more ways to give",
    "personally identifying information", "cookie policy", "opt out",
    "rights reserved", "log in to continue", "forgot password",
    "create an account", "log in to save", "favorite stories",
    "click here to", "read more at", "sign up for", "free trial",
    "subscribe to", "advertisement", "sponsored content",
    "you may also like", "recommended for you", "related articles",
    "leave a comment", "post a comment", "show comments",
    "follow us on", "share this article", "print this page",
    "most popular", "trending now", "newsletter signup",
    "cookies on this", "we use cookies", "cookie settings",
    "accept all cookies", "our privacy policy",
    "moving the entire site", "new platform", "this community has built",
    "every story, every comment", "moving to a new",
    "we're moving", "we are moving", "site maintenance",
    "under construction", "coming soon", "stay tuned",
    "pardon our dust", "scheduled maintenance",
    "skip to main content", "skip to navigation", "back to top",
    "menu home about", "breadcrumb", "navigation menu",
    "already a subscriber", "not a subscriber", "subscription required",
    "unlimited access", "full access", "exclusive content",
    "thank you for reading", "thank you for visiting",
    "page not found", "article not found", "content unavailable",
    "this content is not available", "region restricted",
    "share on facebook", "share on twitter", "tweet this",
    "pin it", "email this", "print this",
])

_SITE_CHROME_PATTERNS = [
    re.compile(r"moving .* to a new platform", re.IGNORECASE),
    re.compile(r"this community has built", re.IGNORECASE),
    re.compile(r"every story.* every comment", re.IGNORECASE),
    re.compile(r"we're (moving|migrating|transitioning)", re.IGNORECASE),
    re.compile(r"site (maintenance|update|upgrade)", re.IGNORECASE),
    re.compile(r"pardon our (dust|appearance)", re.IGNORECASE),
    re.compile(r"experience.* better.* (app|browser)", re.IGNORECASE),
    re.compile(r"for the best experience", re.IGNORECASE),
    re.compile(r"browser.* (not supported|outdated)", re.IGNORECASE),
    re.compile(r"javascript (is )?(required|disabled)", re.IGNORECASE),
    re.compile(r"ad.?block.* detected", re.IGNORECASE),
    re.compile(r"please (disable|turn off|whitelist)", re.IGNORECASE),
    re.compile(r"support.* by (disabling|turning off)", re.IGNORECASE),
]

_RE_URL      = re.compile(r'https?://\S+|ftp://\S+', re.IGNORECASE)
_RE_DOMAIN   = re.compile(
    r'\b(?:[a-zA-Z0-9\-]+\.)+(?:com|org|net|edu|gov|io|co|uk|de|fr|au|ca|ru|cn)\b',
    re.IGNORECASE,
)
_RE_HEX      = re.compile(r'\b[0-9a-fA-F]{5,}\b')
_RE_NUMERIC  = re.compile(r'\b\d+\b')
_RE_SLUG_SEP = re.compile(r'(?<=[a-zA-Z])-(?=[a-zA-Z])')

def _scrub_noise(text: str) -> str:
    text = _RE_URL.sub(" ", text)
    text = _RE_DOMAIN.sub(" ", text)
    text = _RE_HEX.sub(" ", text)
    text = _RE_NUMERIC.sub(" ", text)
    text = _RE_SLUG_SEP.sub(" ", text)
    return text

_WORD_RE    = re.compile(r"\b[a-zA-Z]{3,}\b")
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

_langdetect_lock = threading.Lock()

def _is_english(text: str) -> bool:
    if not _HAS_LANGDETECT:
        return True
    try:
        with _langdetect_lock:
            return detect(text[:500]) == "en"
    except Exception:
        return True

def _tok(text: str) -> list:
    return [
        w for w in _WORD_RE.findall(_scrub_noise(text).lower())
        if w not in _STOPS
        and PMI_MIN_WORD_LEN <= len(w) <= PMI_MAX_WORD_LEN
    ]

def _tok_batch(texts: list) -> list:
    out = []
    for t in texts:
        out.extend(_tok(t))
    return out

def pmi_keywords(top_k_members, n_all_tokens, background_counter, top_n=15):
    if not top_k_members or not n_all_tokens:
        return []

    cluster_tokens = []
    for m in top_k_members:
        cluster_tokens.extend(_tok(m.get("text", "")))
    
    if not cluster_tokens:
        return []

    doc_word_sets = [set(_tok(m.get("text", ""))) for m in top_k_members]

    cf      = Counter(cluster_tokens)
    N_c     = len(cluster_tokens)
    N_b     = n_all_tokens
    inv_N_c = 1.0 / N_c
    inv_N_b = 1.0 / N_b
    scores  = {}

    for w in sorted(cf.keys()):
        c = cf[w]
        if c < PMI_MIN_CLUSTER_COUNT:
            continue
        doc_freq = sum(1 for doc_ws in doc_word_sets if w in doc_ws)
        if doc_freq < PMI_MIN_CLUSTER_DOCS:
            continue

        c_smoothed = c + PMI_ALPHA
        bg         = background_counter.get(w, 0) + PMI_ALPHA
        scores[w]  = math.log2(
            (c_smoothed * inv_N_c) / (bg * inv_N_b) + 1e-9
        )

    sorted_words = sorted(scores.keys(), key=lambda w: (-scores[w], w))
    return sorted_words[:top_n]

def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)

def _top_k_by_cosine_to_centroid(member_embeddings, members, k=TOP_K_FOR_SUMMARY):
    centroid      = member_embeddings.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims          = member_embeddings @ centroid_norm
    top_idx       = np.argsort(-sims)[:k]
    top_idx       = np.sort(top_idx)
    return [members[i] for i in top_idx]

def _is_boilerplate_sentence(sent: str) -> bool:
    lower = sent.lower()
    
    if any(frag in lower for frag in _BOILERPLATE_FRAGMENTS):
        return True
    
    for pattern in _SITE_CHROME_PATTERNS:
        if pattern.search(sent):
            return True
    
    word_count = len(sent.split())
    if word_count < 5:
        return True
    
    alpha_chars = sum(1 for c in sent if c.isalpha())
    if len(sent) > 0 and alpha_chars / len(sent) < 0.6:
        return True
    
    return False

def _is_quality_article(text: str) -> bool:
    if not text or len(text) < MIN_ARTICLE_CHARS:
        return False
    
    sentences = _SENT_SPLIT.split(text)
    valid_sentences = [
        s for s in sentences 
        if len(s.strip()) > 30 and not _is_boilerplate_sentence(s.strip())
    ]
    
    if len(valid_sentences) < MIN_ARTICLE_SENTENCES:
        return False
    
    total_sents = len([s for s in sentences if len(s.strip()) > 20])
    if total_sents > 0:
        boilerplate_ratio = 1 - (len(valid_sentences) / total_sents)
        if boilerplate_ratio > 0.5:
            return False
    
    return True

def _extract_core_sentences(texts: list, keywords: list, max_chars: int) -> str:
    all_sents = []
    for t in texts:
        all_sents.extend(_SENT_SPLIT.split(t))

    valid_sents = [
        s.strip() for s in all_sents
        if len(s.strip()) > 40 and not _is_boilerplate_sentence(s.strip())
    ]
    
    if not valid_sents:
        valid_sents = [
            s.strip() for s in all_sents
            if len(s.strip()) > 25 and not _is_boilerplate_sentence(s.strip())
        ]

    if not valid_sents:
        valid_sents = [s.strip() for s in all_sents if len(s.strip()) > 20]
        valid_sents = [s for s in valid_sents if not _is_boilerplate_sentence(s)]

    if not valid_sents:
        return ""

    kw_set = set(keywords)
    scored = []
    for idx, s in enumerate(valid_sents):
        words = set(_WORD_RE.findall(s.lower()))
        score = len(words.intersection(kw_set))
        length_bonus = min(len(s) / 200, 1.0)
        scored.append((-(score + length_bonus), idx, s))

    scored.sort()

    best_sents  = []
    current_len = 0
    seen_content = set()
    
    for neg_score, idx, s in scored:
        sig = s[:50].lower()
        if sig in seen_content:
            continue
        seen_content.add(sig)
        
        if current_len + len(s) > max_chars:
            break
        best_sents.append((idx, s))
        current_len += len(s)

    best_sents.sort(key=lambda x: x[0])
    return " ".join(s for idx, s in best_sents)

def _clean_summary_text(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
    text = text.strip('"\'')
    
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    return text.strip()

def _ensure_complete_sentences(text: str) -> str:
    if not text:
        return ""
    
    text = text.strip()
    
    if text and text[-1] in '.!?':
        return text
    
    last_period = text.rfind('.')
    last_exclaim = text.rfind('!')
    last_question = text.rfind('?')
    
    last_end = max(last_period, last_exclaim, last_question)
    
    if last_end > 0:
        if last_end >= len(text) * 0.4:
            return text[:last_end + 1].strip()
        else:
            for punct in [';', ',', ' -', ' –']:
                last_break = text.rfind(punct)
                if last_break > len(text) * 0.6:
                    return text[:last_break].strip() + "..."
            return text.strip() + "..."
    
    return text.strip() + "..."

def _bart_summarise(text: str, cluster_id: int = -1) -> str:
    if not text or len(text.strip()) < 50:
        print(f"[Analysis] Cluster {cluster_id}: Text too short for summarization ({len(text)} chars)", flush=True)
        return ""
    
    tokenizer, model, device = _get_bart()
    
    try:
        with torch.inference_mode():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            ).to(device)
            
            input_len = inputs["input_ids"].shape[1]
            if input_len < 20:
                print(f"[Analysis] Cluster {cluster_id}: Input tokens too few ({input_len})", flush=True)
                return ""
            
            print(f"[Analysis] Cluster {cluster_id}: Generating summary from {input_len} tokens...", flush=True)
            
            num_beams = 4 if HAS_GPU else 2
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summary_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=_BART_MAX_NEW_TOKENS,
                    min_new_tokens=_BART_MIN_NEW_TOKENS,
                    num_beams=num_beams,
                    early_stopping=False,
                    no_repeat_ngram_size=3,
                    forced_bos_token_id=0,
                    length_penalty=1.2,
                    repetition_penalty=1.1,
                    do_sample=False,
                )
            
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            
            del inputs
            del summary_ids
            
            summary = _clean_summary_text(summary)
            summary = _ensure_complete_sentences(summary)
            
            if not summary or len(summary) < 20 or _is_boilerplate_sentence(summary):
                print(f"[Analysis] Cluster {cluster_id}: Summary was boilerplate or too short, discarding", flush=True)
                return ""
            
            print(f"[Analysis] Cluster {cluster_id}: Summary generated ({len(summary)} chars)", flush=True)
            return summary
        
    except Exception as e:
        print(f"[Analysis] Cluster {cluster_id}: BART failed - {e}", flush=True)
        return ""

def _generate_fallback_summary(members: list, keywords: list) -> str:
    for member in members[:3]:
        text = member.get("text", "")
        sentences = _SENT_SPLIT.split(text)
        valid = [s.strip() for s in sentences 
                 if len(s.strip()) > 40 and not _is_boilerplate_sentence(s.strip())]
        if valid:
            fallback = " ".join(valid[:2])
            fallback = _clean_summary_text(fallback)
            fallback = _ensure_complete_sentences(fallback)
            if fallback:
                return fallback
    
    if keywords:
        return f"News cluster covering: {', '.join(keywords[:5])}."
    
    return "News cluster (summary unavailable)."

def _reliability_profile(members: list) -> dict:
    scores   = [m["newsguard_score"] for m in members]
    total    = len(scores)
    scored   = [s for s in scores if s is not None and s >= 0]
    unscored = total - len(scored)

    if not scored:
        return {
            "avg_newsguard_score":   None,
            "pct_reliable":          None,
            "pct_unreliable":        None,
            "pct_unscored":          1.0,
            "is_unreliable_cluster": False,
            "reliability_label":     "unscored",
        }

    n_reliable     = sum(1 for s in scored if s >= RELIABLE_THRESHOLD)
    n_unreliable   = len(scored) - n_reliable
    avg_score      = float(np.mean(scored))
    pct_reliable   = n_reliable   / len(scored)
    pct_unreliable = n_unreliable / len(scored)
    pct_unscored   = unscored     / total

    is_unreliable = avg_score < RELIABLE_THRESHOLD
    label = "unreliable" if is_unreliable else "reliable"

    return {
        "avg_newsguard_score":   round(avg_score, 2),
        "pct_reliable":          round(pct_reliable,   3),
        "pct_unreliable":        round(pct_unreliable, 3),
        "pct_unscored":          round(pct_unscored,   3),
        "is_unreliable_cluster": is_unreliable,
        "reliability_label":     label,
    }

def _get_cluster_time_range(members: list) -> dict:
    timestamps = []
    for m in members:
        created = m.get("created_at") or m.get("createdAt")
        if created:
            timestamps.append(created)
    
    if not timestamps:
        return {"earliest": None, "latest": None}
    
    timestamps.sort()
    return {"earliest": timestamps[0], "latest": timestamps[-1]}

def _collect_unique_dids(members: list) -> list:
    dids = set()
    for m in members:
        did = m.get("original_did", "")
        if did:
            dids.add(did)
    return sorted(dids)

def _collect_unique_new_dids(members: list) -> list:
    dids = set()
    for m in members:
        did = m.get("new_did", "")
        if did:
            dids.add(did)
    return sorted(dids)

def _process_cluster(args):
    (cid, members, member_embeddings_list, n_all_tokens, background_counter) = args

    members = sorted(members, key=lambda m: m["url"])

    print(f"[Analysis] Processing cluster {cid} with {len(members)} members...", flush=True)

    member_embeddings = np.array(member_embeddings_list, dtype=np.float32)

    top_k = _top_k_by_cosine_to_centroid(member_embeddings, members, TOP_K_FOR_SUMMARY)

    keywords = pmi_keywords(top_k, n_all_tokens, background_counter)
    
    print(f"[Analysis] Cluster {cid} keywords: {keywords[:5]}", flush=True)

    dense_text = _extract_core_sentences(
        [m["text"] for m in top_k], 
        keywords, 
        _BART_MAX_INPUT_CHARS
    )
    
    print(f"[Analysis] Cluster {cid}: Extracted {len(dense_text)} chars for summarization", flush=True)
    
    final_summary = ""
    if dense_text and len(dense_text) >= 100:
        final_summary = _bart_summarise(dense_text, cid)
    else:
        print(f"[Analysis] Cluster {cid}: Dense text too short ({len(dense_text)} chars), using fallback", flush=True)
    
    if not final_summary or len(final_summary) < 20:
        print(f"[Analysis] Cluster {cid}: Using fallback summary", flush=True)
        final_summary = _generate_fallback_summary(top_k, keywords)

    total_likes   = sum(m.get("like_count", 0) for m in members)
    total_reposts = sum(m.get("repost_count", 0) for m in members)
    
    time_range = _get_cluster_time_range(members)
    unique_dids = _collect_unique_dids(members)
    unique_new_dids = _collect_unique_new_dids(members)

    return {
        "cid":           cid,
        "summary":       final_summary,
        "keywords":      keywords[:10],
        "reliability":   _reliability_profile(members),
        "total_likes":   total_likes,
        "total_reposts": total_reposts,
        "total_engagement": total_likes + total_reposts,
        "domains":       sorted({m["domain"] for m in members if m.get("domain")}),
        "members":       members,
        "time_range":    time_range,
        "unique_dids":   unique_dids,
        "unique_new_dids": unique_new_dids,
    }

def _passes_quality_filter(members: list) -> bool:
    if len(members) < MIN_ARTICLES_PER_CLUSTER:
        return False
    
    quality_count = sum(1 for m in members if _is_quality_article(m.get("text", "")))
    return quality_count >= MIN_ARTICLES_PER_CLUSTER

def _build_background_counter_threaded(texts: list) -> tuple[Counter, int]:
    CHUNK = 2_000
    chunks = [texts[i:i + CHUNK] for i in range(0, len(texts), CHUNK)]
    counter = Counter()
    n_tokens = 0
    
    for chunk in chunks:
        result = _tok_batch(chunk)
        counter.update(result)
        n_tokens += len(result)
    
    return counter, n_tokens


def _load_existing_centers(domain: str) -> tuple:
    os.makedirs(CLUSTER_CENTERS_DIR, exist_ok=True)
    centers_path = os.path.join(CLUSTER_CENTERS_DIR, f'{domain}_centers.obj')
    counts_path = os.path.join(CLUSTER_CENTERS_DIR, f'{domain}_center_counts.obj')
    
    centers = None
    counts = None
    
    if os.path.exists(centers_path) and os.path.exists(counts_path):
        try:
            with open(centers_path, 'rb') as f:
                centers = pickle.load(f)
            with open(counts_path, 'rb') as f:
                counts = pickle.load(f)
            print(f"[Analysis] Loaded existing centers: {centers.shape[0]} clusters", flush=True)
        except Exception as e:
            print(f"[Analysis] Failed to load existing centers: {e}", flush=True)
            centers = None
            counts = None
    
    return centers, counts


def _save_centers(domain: str, centers: np.ndarray, counts: np.ndarray):
    os.makedirs(CLUSTER_CENTERS_DIR, exist_ok=True)
    centers_path = os.path.join(CLUSTER_CENTERS_DIR, f'{domain}_centers.obj')
    counts_path = os.path.join(CLUSTER_CENTERS_DIR, f'{domain}_center_counts.obj')
    
    try:
        with open(centers_path, 'wb') as f:
            pickle.dump(centers, f)
        with open(counts_path, 'wb') as f:
            pickle.dump(counts, f)
        print(f"[Analysis] Saved centers: {centers.shape[0]} clusters", flush=True)
    except Exception as e:
        print(f"[Analysis] Failed to save centers: {e}", flush=True)


def _assign_labels_to_centers(embeddings: np.ndarray, centers: np.ndarray) -> np.ndarray:
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normalized = embeddings / np.where(emb_norms == 0, 1.0, emb_norms)
    
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers_normalized = centers / np.where(center_norms == 0, 1.0, center_norms)
    
    similarities = emb_normalized @ centers_normalized.T
    
    labels = np.argmax(similarities, axis=1)
    
    return labels


def _cluster_all(articles, background_counter, n_all_tokens):
    if len(articles) < 2:
        return []

    articles = sorted(articles, key=lambda a: a["url"])

    print(f"[Analysis] Starting quality filter on {len(articles)} articles...", flush=True)
    quality_articles = [a for a in articles if _is_quality_article(a.get("text", ""))]
    
    if len(quality_articles) < len(articles):
        rejected = len(articles) - len(quality_articles)
        print(f"[Analysis] Filtered {rejected:,} low-quality articles before clustering", flush=True)
        articles = quality_articles
    
    if len(articles) < 2:
        print("[Analysis] Not enough quality articles for clustering", flush=True)
        return []

    print(f"[Analysis] Building embeddings matrix for {len(articles)} articles...", flush=True)
    embeddings = np.array([a["embedding"] for a in articles], dtype=np.float32)
    embeddings = _l2_normalise(embeddings)
    n = len(embeddings)

    print(f"[Analysis] Running MiniBatchDPMeans clustering on {n} articles...", flush=True)
    
    day_indexes = [n]
    
    domain = "narrative_analysis"
    
    existing_centers, existing_counts = _load_existing_centers(domain)
    
    try:
        clf = MiniBatchDPMeans(
            n_clusters=1 if existing_centers is None else existing_centers.shape[0],
            init=existing_centers if existing_centers is not None else "k-means++",
            delta=DPMEANS_DELTA,
            batch_size=DPMEANS_BATCH_SIZE,
            max_iter=DPMEANS_MAX_ITER,
            n_init=DPMEANS_N_INIT if existing_centers is None else 1,
            random_state=RANDOM_SEED,
            verbose=1,
            day_indexes=day_indexes,
            n_introduce=DPMEANS_N_INTRODUCE,
            domain=domain,
        )
        
        if existing_centers is not None and existing_counts is not None:
            clf._counts = existing_counts.astype(embeddings.dtype)
        
        clf.fit(embeddings)
        labels = clf.labels_
        n_clusters = clf.n_clusters
        
        _save_centers(domain, clf.cluster_centers_, clf._counts)
        
        print(f"[Analysis] MiniBatchDPMeans found {n_clusters} raw clusters", flush=True)
        
    except Exception as exc:
        print(f"[Analysis] MiniBatchDPMeans clustering failed: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        
        if existing_centers is not None:
            print("[Analysis] Attempting fallback with existing centers...", flush=True)
            try:
                labels = _assign_labels_to_centers(embeddings, existing_centers)
                n_clusters = existing_centers.shape[0]
                print(f"[Analysis] Fallback assignment to {n_clusters} existing clusters", flush=True)
            except Exception as fallback_exc:
                print(f"[Analysis] Fallback also failed: {fallback_exc}", flush=True)
                return []
        else:
            return []

    del embeddings
    gc.collect()

    norm_embs = _l2_normalise(
        np.array([a["embedding"] for a in articles], dtype=np.float32)
    )

    clusters = {}
    cluster_embs = {}

    for art, emb, lbl in zip(articles, norm_embs, labels):
        lbl = int(lbl)
        if lbl not in clusters:
            clusters[lbl] = []
            cluster_embs[lbl] = []
        clusters[lbl].append(art)
        cluster_embs[lbl].append(emb)

    del norm_embs
    gc.collect()

    for lbl in clusters:
        clusters[lbl] = sorted(clusters[lbl], key=lambda m: m["url"])

    sorted_cids = sorted(clusters.keys())

    cluster_sizes = [(cid, len(clusters[cid])) for cid in sorted_cids]
    cluster_sizes.sort(key=lambda x: (-x[1], x[0]))
    print(f"[Analysis] Top 10 cluster sizes: {cluster_sizes[:10]}", flush=True)

    valid_cids = [cid for cid in sorted_cids if _passes_quality_filter(clusters[cid])]
    
    print(f"[Analysis] {len(valid_cids)} clusters pass quality filter", flush=True)
    
    cluster_args = [
        (cid, clusters[cid], [cluster_embs[cid][i] for i, m in enumerate(clusters[cid])],
         n_all_tokens, background_counter)
        for cid in valid_cids
    ]

    results_map = {}
    total_clusters = len(cluster_args)
    processed = 0
    
    print(f"[Analysis] Starting summarization of {total_clusters} clusters...", flush=True)
    
    for args in cluster_args:
        try:
            r = _process_cluster(args)
            results_map[r["cid"]] = r
            processed += 1
            if processed % 5 == 0:
                print(f"[Analysis] Progress: {processed}/{total_clusters} clusters processed", flush=True)
                gc.collect()
                if HAS_GPU:
                    torch.cuda.empty_cache()
        except Exception as exc:
            cid = args[0]
            print(f"[Analysis] Cluster {cid} failed: {exc}", flush=True)

    print(f"[Analysis] Summarization complete: {processed}/{total_clusters} successful", flush=True)
    
    _unload_bart()

    narratives = []
    for cid in valid_cids:
        members = clusters[cid]
        if not members or cid not in results_map:
            continue
        r   = results_map[cid]
        rel = r["reliability"]
        time_range = r.get("time_range", {"earliest": None, "latest": None})
        unique_dids = r.get("unique_dids", [])
        unique_new_dids = r.get("unique_new_dids", [])
        narratives.append({
            "narrative_id":          cid,
            "size":                  len(members),
            "summary":               r["summary"],
            "pmi_keywords":          r["keywords"],
            "avg_newsguard_score":   rel["avg_newsguard_score"],
            "pct_reliable":          rel["pct_reliable"],
            "pct_unreliable":        rel["pct_unreliable"],
            "pct_unscored":          rel["pct_unscored"],
            "is_unreliable_cluster": rel["is_unreliable_cluster"],
            "reliability_label":     rel["reliability_label"],
            "total_likes":           r["total_likes"],
            "total_reposts":         r["total_reposts"],
            "total_engagement":      r["total_engagement"],
            "domains_cited":         r["domains"],
            "earliest_article":      time_range["earliest"],
            "latest_article":        time_range["latest"],
            "unique_posters":        len(unique_dids),
            "unique_new_posters":    len(unique_new_dids),
            "articles": [
                {
                    "url":             m["url"],
                    "domain":          m.get("domain", ""),
                    "newsguard_score": m.get("newsguard_score"),
                    "is_reliable":     (
                        m.get("newsguard_score") is not None
                        and m["newsguard_score"] >= RELIABLE_THRESHOLD
                    ),
                    "likes":           m.get("like_count", 0),
                    "reposts":         m.get("repost_count", 0),
                    "original_did":    m.get("original_did", ""),
                    "new_did":         m.get("new_did", ""),
                    "created_at":      m.get("created_at") or m.get("createdAt", ""),
                }
                for m in sorted(members, key=lambda x: x["url"])
            ],
        })

    narratives.sort(key=lambda x: (-x["size"], x["narrative_id"]))
    return narratives

def run_analysis_job():
    if os.path.exists(LOCK_FILE):
        print("[Analysis] Another analysis job is running, skipping", flush=True)
        return

    open(LOCK_FILE, "w").close()

    try:
        t0 = time.time()
        print("[Analysis] Starting analysis job...", flush=True)
        
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
        
        articles = database.get_all_articles()
        
        if len(articles) < 2:
            print("[Analysis] Not enough articles in database", flush=True)
            return

        articles = sorted(articles, key=lambda a: a["url"])

        print(f"[Analysis] Filtering non-English articles from {len(articles)}...", flush=True)
        articles = [a for a in articles if _is_english(a.get("text", ""))]
        if len(articles) < 2:
            print("[Analysis] Not enough English articles", flush=True)
            return

        n_reliable   = sum(1 for a in articles
                           if a.get("newsguard_score") is not None
                           and a["newsguard_score"] >= RELIABLE_THRESHOLD)
        n_unreliable = sum(1 for a in articles
                           if a.get("newsguard_score") is not None
                           and a["newsguard_score"] < RELIABLE_THRESHOLD)
        n_unscored   = len(articles) - n_reliable - n_unreliable
        print(
            f"[Analysis] {len(articles):,} articles — "
            f"{n_reliable:,} reliable / {n_unreliable:,} unreliable / {n_unscored:,} unscored",
            flush=True,
        )

        print("[Analysis] Building background token counter...", flush=True)
        texts = [a.get("text", "") for a in articles]
        background_counter, n_all_tokens = _build_background_counter_threaded(texts)
        print(f"[Analysis] Background counter built: {n_all_tokens:,} tokens", flush=True)
        del texts
        gc.collect()

        print("[Analysis] Tokenizing articles...", flush=True)
        for a in articles:
            a["_tokens"] = _tok(a.get("text", ""))

        os.makedirs("./clusters", exist_ok=True)
        narratives = _cluster_all(articles, background_counter, n_all_tokens)

        n_unreliable_clusters = sum(1 for n in narratives if n["is_unreliable_cluster"])
        n_reliable_clusters   = sum(1 for n in narratives if n["reliability_label"] == "reliable")

        print(f"[Analysis] Final results: {len(narratives)} narratives "
              f"({n_reliable_clusters} reliable, {n_unreliable_clusters} unreliable)",
              flush=True)

        out_path = "./clusters/narratives_latest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "generated_at":            datetime.now().isoformat(),
                        "total_narratives":        len(narratives),
                        "reliable_clusters":       n_reliable_clusters,
                        "unreliable_clusters":     n_unreliable_clusters,
                        "reliable_threshold":      RELIABLE_THRESHOLD,
                        "dpmeans_delta":           DPMEANS_DELTA,
                        "device":                  "GPU" if HAS_GPU else "CPU",
                        "random_seed":             RANDOM_SEED,
                        "elapsed_s":               round(time.time() - t0, 1),
                    },
                    "narratives": narratives,
                },
                f, indent=2, ensure_ascii=False,
            )
        print(
            f"[Analysis] Wrote {len(narratives)} narratives to {out_path} "
            f"in {time.time()-t0:.0f}s",
            flush=True,
        )

    except Exception as e:
        print(f"[Analysis] Job failed with error: {e}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        _unload_bart()
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    database.init_db()

    if args.once:
        run_analysis_job()
    else:
        print("[Analysis] Waiting for analysis signals...", flush=True)
        while True:
            if os.path.exists(SIGNAL_FILE):
                try:
                    os.remove(SIGNAL_FILE)
                except OSError:
                    pass
                run_analysis_job()
            else:
                time.sleep(5)