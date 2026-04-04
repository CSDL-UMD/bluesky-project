import sqlite3
import os
import threading
import chromadb
from filelock import FileLock

DB_DIR           = "./db"
DB_PATH          = os.path.join(DB_DIR, "bluesky_file_ledger.db")
CHROMA_PATH      = os.path.join(DB_DIR, "chroma_vector_db")
CHROMA_LOCK_PATH = os.path.join(DB_DIR, "chroma.lock")

_chroma_client     = None
_chroma_collection = None
_chroma_lock       = threading.Lock()
_file_lock         = FileLock(CHROMA_LOCK_PATH, timeout=60)

CHROMA_HOST = "localhost"
CHROMA_PORT = 8001


def init_db():
    print("[Database] Initializing SQLite ledger...", flush=True)
    os.makedirs(DB_DIR, exist_ok=True)
    conn   = sqlite3.connect(DB_PATH)
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
            _chroma_client     = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _chroma_collection = _chroma_client.get_or_create_collection(
                name="bluesky_articles",
                metadata={"hnsw:space": "cosine"},
            )
    return _chroma_collection


def is_file_processed(filename):
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM processed_files WHERE filename = ?", (filename,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def mark_file_processed(filename):
    print(f"[Database] Marking file as processed: {filename}", flush=True)
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO processed_files (filename) VALUES (?)", (filename,)
    )
    conn.commit()
    conn.close()


def get_article_metadata(url):
    with _file_lock:
        res = get_chroma_collection().get(ids=[url], include=["metadatas"])
        if res and res.get("ids"):
            return res["metadatas"][0]
        return None


def update_article_stats(url, new_likes, new_reposts):
    with _file_lock:
        col = get_chroma_collection()
        res = col.get(ids=[url], include=["metadatas"])
        if res and res.get("ids"):
            meta                = res["metadatas"][0]
            meta["likeCount"]   = meta.get("likeCount",   0) + new_likes
            meta["repostCount"] = meta.get("repostCount", 0) + new_reposts
            col.update(ids=[url], metadatas=[meta])


def insert_article(url, domain, ng_score, text, embedding, did, created_at, likes, reposts):
    with _file_lock:
        get_chroma_collection().upsert(
            ids        = [url],
            embeddings = [embedding.tolist()],
            documents  = [text],
            metadatas  = [{
                "domain":          str(domain),
                "newsguard_score": float(ng_score),
                "originalDid":     str(did)        if did        else "",
                "createdAt":       str(created_at) if created_at else "",
                "likeCount":       int(likes),
                "repostCount":     int(reposts),
            }],
        )


def get_existing_urls(urls: list[str]) -> dict[str, dict]:
    if not urls:
        return {}
    with _file_lock:
        res = get_chroma_collection().get(ids=urls, include=["metadatas"])
    found = {}
    if res and res.get("ids"):
        for uid, meta in zip(res["ids"], res["metadatas"]):
            found[uid] = meta or {}
    return found


def batch_update_stats(updates: list[tuple[str, int, int]]):
    if not updates:
        return
    urls = [u for u, _, _ in updates]
    with _file_lock:
        col = get_chroma_collection()
        res = col.get(ids=urls, include=["metadatas"])
        if not res or not res.get("ids"):
            return
        new_metas = []
        for uid, meta in zip(res["ids"], res["metadatas"]):
            meta  = meta or {}
            delta = {u: (l, r) for u, l, r in updates}
            dl, dr = delta.get(uid, (0, 0))
            meta["likeCount"]   = meta.get("likeCount",   0) + dl
            meta["repostCount"] = meta.get("repostCount", 0) + dr
            new_metas.append(meta)
        col.update(ids=res["ids"], metadatas=new_metas)


def batch_insert_articles(articles: list[dict]):
    if not articles:
        return
    ids        = [a["url"]   for a in articles]
    embeddings = [a["embedding"].tolist() for a in articles]
    documents  = [a["text"]  for a in articles]
    metadatas  = [
        {
            "domain":          str(a["domain"]),
            "newsguard_score": float(a["newsguard_score"]),
            "originalDid":     str(a.get("did", ""))        or "",
            "createdAt":       str(a.get("created_at", "")) or "",
            "likeCount":       int(a.get("likeCount",   0)),
            "repostCount":     int(a.get("repostCount", 0)),
        }
        for a in articles
    ]
    with _file_lock:
        get_chroma_collection().upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = documents,
            metadatas  = metadatas,
        )


def get_all_articles():
    print("[Database] Fetching all embedded articles from ChromaDB...", flush=True)
    with _file_lock:
        res = get_chroma_collection().get(
            include=["embeddings", "metadatas", "documents"]
        )
    articles = []
    if not res or not res.get("ids"):
        print("[Database] ChromaDB is currently empty.", flush=True)
        return articles
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
    print(f"[Database] Successfully retrieved {len(articles)} articles.", flush=True)
    return articles