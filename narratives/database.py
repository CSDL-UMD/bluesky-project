import sqlite3
import os
import threading
import chromadb

DB_DIR     = "./db"
DB_PATH    = os.path.join(DB_DIR, "bluesky_file_ledger.db")
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001

_chroma_client     = None
_chroma_collection = None
_chroma_lock       = threading.Lock()
_CHROMA_MAX_BATCH  = None


def _get_max_batch():
    global _CHROMA_MAX_BATCH
    if _CHROMA_MAX_BATCH is None:
        try:
            _CHROMA_MAX_BATCH = get_chroma_collection()._client.get_max_batch_size()
        except Exception:
            _CHROMA_MAX_BATCH = 5000
    return _CHROMA_MAX_BATCH


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def init_db():
    print("[Database] Initializing SQLite ledger...", flush=True)
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            filename     TEXT PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("[Database] SQLite ledger ready.", flush=True)


def get_chroma_collection():
    global _chroma_client, _chroma_collection
    with _chroma_lock:
        if _chroma_collection is None:
            _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _chroma_collection = _chroma_client.get_or_create_collection(
                name="bluesky_articles",
                metadata={"hnsw:space": "cosine"},
            )
    return _chroma_collection


def is_file_processed(filename):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM processed_files WHERE filename = ?", (filename,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def mark_file_processed(filename):
    print(f"[Database] Marking file as processed: {filename}", flush=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO processed_files (filename) VALUES (?)", (filename,))
    conn.commit()
    conn.close()


def get_article_metadata(url):
    res = get_chroma_collection().get(ids=[url], include=["metadatas"])
    if res and res.get("ids"):
        return res["metadatas"][0]
    return None


def get_existing_urls(urls):
    if not urls:
        return {}
    chunk_size = _get_max_batch()
    found = {}
    col = get_chroma_collection()
    for chunk in _chunked(urls, chunk_size):
        res = col.get(ids=chunk, include=["metadatas"])
        if res and res.get("ids"):
            for uid, meta in zip(res["ids"], res["metadatas"]):
                found[uid] = meta or {}
    return found


def batch_update_stats(updates):
    if not updates:
        return
    chunk_size = _get_max_batch()
    col = get_chroma_collection()
    for chunk in _chunked(updates, chunk_size):
        urls = [u for u, _, _ in chunk]
        res = col.get(ids=urls, include=["metadatas"])
        if not res or not res.get("ids"):
            continue
        delta = {u: (l, r) for u, l, r in chunk}
        new_metas = []
        for uid, meta in zip(res["ids"], res["metadatas"]):
            meta = meta or {}
            dl, dr = delta.get(uid, (0, 0))
            meta["likeCount"]   = meta.get("likeCount",   0) + dl
            meta["repostCount"] = meta.get("repostCount", 0) + dr
            new_metas.append(meta)
        col.update(ids=res["ids"], metadatas=new_metas)


def batch_insert_articles(articles):
    if not articles:
        return
    chunk_size = _get_max_batch()
    col = get_chroma_collection()
    for chunk in _chunked(articles, chunk_size):
        ids        = [a["url"]               for a in chunk]
        embeddings = [a["embedding"].tolist() for a in chunk]
        documents  = [a["text"]              for a in chunk]
        metadatas  = [
            {
                "domain":          str(a["domain"]),
                "newsguard_score": float(a["newsguard_score"]),
                "originalDid":     str(a.get("did", ""))        or "",
                "createdAt":       str(a.get("created_at", "")) or "",
                "likeCount":       int(a.get("likeCount",   0)),
                "repostCount":     int(a.get("repostCount", 0)),
            }
            for a in chunk
        ]
        col.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = documents,
            metadatas  = metadatas,
        )


def get_all_articles():
    print("[Database] Fetching all embedded articles from ChromaDB...", flush=True)
    PAGE_SIZE = 1000
    articles  = []
    offset    = 0
    col = get_chroma_collection()
    while True:
        res = col.get(
            include=["embeddings", "metadatas", "documents"],
            limit=PAGE_SIZE,
            offset=offset,
        )
        if not res or not res.get("ids"):
            break
        for i in range(len(res["ids"])):
            meta = res["metadatas"][i] or {}
            articles.append({
                "url":             res["ids"][i],
                "text":            res["documents"][i],
                "embedding":       res["embeddings"][i],
                "domain":          meta.get("domain", ""),
                "newsguard_score": meta.get("newsguard_score"),
                "originalDid":     meta.get("originalDid", ""),
                "createdAt":       meta.get("createdAt", ""),
                "likeCount":       meta.get("likeCount",  0),
                "repostCount":     meta.get("repostCount", 0),
            })
        fetched = len(res["ids"])
        offset += fetched
        if fetched < PAGE_SIZE:
            break
    if not articles:
        print("[Database] ChromaDB is currently empty.", flush=True)
    else:
        print(f"[Database] Successfully retrieved {len(articles)} articles.", flush=True)
    return articles