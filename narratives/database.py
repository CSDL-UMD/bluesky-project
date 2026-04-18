import os
import gc
import sqlite3
import threading
import numpy as np
import h5py

DB_DIR = "./db"
SQLITE_PATH = os.path.join(DB_DIR, "ledger.db")
HDF5_PATH = os.path.join(DB_DIR, "embeddings.h5")
EMBED_DIM = 768

_lock = threading.Lock()
_local = threading.local()


def _conn():
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(DB_DIR, exist_ok=True)
        con = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA cache_size=-65536")
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("PRAGMA mmap_size=536870912")
        _local.conn = con
    return _local.conn


def _hdf5(mode="r"):
    os.makedirs(DB_DIR, exist_ok=True)
    return h5py.File(HDF5_PATH, mode)


def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = _conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            domain TEXT NOT NULL,
            newsguard_score REAL,
            text TEXT,
            original_did TEXT DEFAULT '',
            new_did TEXT DEFAULT '',
            created_at TEXT DEFAULT '',
            like_count INTEGER DEFAULT 0,
            repost_count INTEGER DEFAULT 0,
            cluster_id INTEGER DEFAULT -1
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            actor_did TEXT NOT NULL,
            interaction_type TEXT NOT NULL,
            created_at TEXT DEFAULT '',
            UNIQUE(url, actor_did, interaction_type, created_at)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_cluster ON articles(cluster_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_created ON articles(created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_url ON interactions(url)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_actor ON interactions(actor_did)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at)")

    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(articles)").fetchall()}
    if "new_did" not in existing_cols:
        cur.execute("ALTER TABLE articles ADD COLUMN new_did TEXT DEFAULT ''")

    conn.commit()

    if not os.path.exists(HDF5_PATH):
        with h5py.File(HDF5_PATH, "w") as f:
            f.create_dataset("urls", shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
            f.create_dataset("embeddings", shape=(0, EMBED_DIM), maxshape=(None, EMBED_DIM), dtype=np.float32)

    print(f"[Database] Initialized at {SQLITE_PATH}", flush=True)


def is_file_processed(filename):
    conn = _conn()
    row = conn.execute(
        "SELECT 1 FROM processed_files WHERE filename=?", (filename,)
    ).fetchone()
    return row is not None


def mark_file_processed(filename):
    conn = _conn()
    conn.execute(
        "INSERT OR IGNORE INTO processed_files (filename) VALUES (?)", (filename,)
    )
    conn.commit()


def url_exists(url):
    conn = _conn()
    row = conn.execute("SELECT 1 FROM articles WHERE url=?", (url,)).fetchone()
    return row is not None


def get_existing_urls(urls):
    if not urls:
        return set()
    conn = _conn()
    placeholders = ",".join("?" * len(urls))
    rows = conn.execute(
        f"SELECT url FROM articles WHERE url IN ({placeholders})", urls
    ).fetchall()
    return {r[0] for r in rows}


def batch_check_urls(urls):
    return get_existing_urls(urls)


def batch_update_stats(updates):
    if not updates:
        return
    with _lock:
        conn = _conn()
        cur = conn.cursor()
        for url, likes, reposts in updates:
            cur.execute(
                """UPDATE articles 
                   SET like_count = like_count + ?, repost_count = repost_count + ?
                   WHERE url = ?""",
                (likes, reposts, url)
            )
        conn.commit()


def batch_insert_interactions(interactions):
    if not interactions:
        return 0
    with _lock:
        conn = _conn()
        inserted = 0
        for inter in interactions:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO interactions 
                       (url, actor_did, interaction_type, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (inter["url"], inter["actor_did"], inter["interaction_type"], inter.get("created_at", "")),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                continue
        conn.commit()
        return inserted


def insert_article(url, domain, newsguard_score, text, embedding, original_did="", new_did="", created_at="", like_count=0, repost_count=0):
    with _lock:
        conn = _conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO articles 
                   (url, domain, newsguard_score, text, original_did, new_did, created_at, like_count, repost_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (url, domain, newsguard_score, text, original_did, new_did, created_at, like_count, repost_count),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return False

        with _hdf5("a") as f:
            urls_ds = f["urls"]
            emb_ds = f["embeddings"]
            n = urls_ds.shape[0]
            urls_ds.resize((n + 1,))
            emb_ds.resize((n + 1, EMBED_DIM))
            urls_ds[n] = url
            emb_ds[n] = embedding.astype(np.float32)

        return True


def batch_insert_articles(articles_data):
    if not articles_data:
        return 0

    articles_data = sorted(articles_data, key=lambda x: x["url"])

    with _lock:
        conn = _conn()
        inserted = 0
        new_urls = []
        new_embeddings = []

        for a in articles_data:
            url = a["url"]
            domain = a["domain"]
            newsguard_score = a["newsguard_score"]
            text = a["text"]
            embedding = a["embedding"]
            original_did = a.get("original_did", "")
            new_did = a.get("new_did", "")
            created_at = a.get("created_at", "")
            like_count = a.get("like_count", 0)
            repost_count = a.get("repost_count", 0)

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO articles 
                       (url, domain, newsguard_score, text, original_did, new_did, created_at, like_count, repost_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (url, domain, newsguard_score, text, original_did, new_did, created_at, like_count, repost_count),
                )
                if conn.total_changes > inserted:
                    new_urls.append(url)
                    new_embeddings.append(embedding.astype(np.float32))
                    inserted += 1
            except sqlite3.IntegrityError:
                continue

        conn.commit()

        if new_urls:
            sorted_pairs = sorted(zip(new_urls, new_embeddings), key=lambda x: x[0])
            new_urls = [p[0] for p in sorted_pairs]
            new_embeddings = [p[1] for p in sorted_pairs]

            with _hdf5("a") as f:
                urls_ds = f["urls"]
                emb_ds = f["embeddings"]
                n = urls_ds.shape[0]
                urls_ds.resize((n + len(new_urls),))
                emb_ds.resize((n + len(new_urls), EMBED_DIM))
                for i, (u, e) in enumerate(zip(new_urls, new_embeddings)):
                    urls_ds[n + i] = u
                    emb_ds[n + i] = e

        return inserted


def get_all_articles():
    conn = _conn()
    rows = conn.execute(
        """SELECT url, domain, newsguard_score, text, original_did, new_did, created_at, like_count, repost_count 
           FROM articles"""
    ).fetchall()

    url_to_row = {r[0]: r for r in rows}

    with _hdf5("r") as f:
        urls = f["urls"][:]
        embeddings = f["embeddings"][:]

    articles = []
    for i, url in enumerate(urls):
        if isinstance(url, bytes):
            url = url.decode("utf-8")
        if url in url_to_row:
            r = url_to_row[url]
            articles.append({
                "url": url,
                "domain": r[1],
                "newsguard_score": r[2],
                "text": r[3],
                "original_did": r[4],
                "new_did": r[5],
                "created_at": r[6],
                "like_count": r[7],
                "repost_count": r[8],
                "embedding": embeddings[i],
            })

    articles.sort(key=lambda x: x["url"])
    return articles


def get_articles_by_date_range(start_date, end_date):
    conn = _conn()
    rows = conn.execute(
        """SELECT url, domain, newsguard_score, text, original_did, new_did, created_at, like_count, repost_count 
           FROM articles
           WHERE created_at >= ? AND created_at <= ?
           ORDER BY created_at""",
        (start_date, end_date)
    ).fetchall()

    url_to_row = {r[0]: r for r in rows}

    if not url_to_row:
        return []

    with _hdf5("r") as f:
        urls = f["urls"][:]
        embeddings = f["embeddings"][:]

    articles = []
    for i, url in enumerate(urls):
        if isinstance(url, bytes):
            url = url.decode("utf-8")
        if url in url_to_row:
            r = url_to_row[url]
            articles.append({
                "url": url,
                "domain": r[1],
                "newsguard_score": r[2],
                "text": r[3],
                "original_did": r[4],
                "new_did": r[5],
                "created_at": r[6],
                "like_count": r[7],
                "repost_count": r[8],
                "embedding": embeddings[i],
            })

    articles.sort(key=lambda x: x["created_at"])
    return articles


def get_article_count():
    conn = _conn()
    row = conn.execute("SELECT COUNT(*) FROM articles").fetchone()
    return row[0] if row else 0


def get_domain_stats():
    conn = _conn()
    rows = conn.execute(
        """SELECT domain, COUNT(*) as cnt, AVG(newsguard_score) as avg_score
           FROM articles GROUP BY domain ORDER BY cnt DESC"""
    ).fetchall()
    return [{"domain": r[0], "count": r[1], "avg_score": r[2]} for r in rows]


def get_temporal_stats():
    conn = _conn()
    rows = conn.execute(
        """SELECT DATE(created_at) as date, COUNT(*) as cnt
           FROM articles 
           WHERE created_at != ''
           GROUP BY DATE(created_at) 
           ORDER BY date"""
    ).fetchall()
    return [{"date": r[0], "count": r[1]} for r in rows]


def get_interactions_for_url(url):
    conn = _conn()
    rows = conn.execute(
        """SELECT actor_did, interaction_type, created_at FROM interactions WHERE url=?""",
        (url,)
    ).fetchall()
    return [{"actor_did": r[0], "interaction_type": r[1], "created_at": r[2]} for r in rows]


def clear_all():
    with _lock:
        conn = _conn()
        conn.execute("DELETE FROM articles")
        conn.execute("DELETE FROM interactions")
        conn.execute("DELETE FROM processed_files")
        conn.commit()

        if os.path.exists(HDF5_PATH):
            os.remove(HDF5_PATH)
            with h5py.File(HDF5_PATH, "w") as f:
                f.create_dataset("urls", shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
                f.create_dataset("embeddings", shape=(0, EMBED_DIM), maxshape=(None, EMBED_DIM), dtype=np.float32)

    print("[Database] Cleared all data", flush=True)