import pandas as pd
from typing import Set, Dict, List, Tuple, Any, Optional
from itertools import islice
from functools import lru_cache
from datetime import datetime, timedelta
import re, time, asyncio, aiohttp, psycopg2, subprocess, gzip, os, json
from urllib.parse import quote, urlsplit
from atproto import CAR, models, AsyncFirehoseSubscribeReposClient, parse_subscribe_repos_message
from collections import Counter
import uvloop, aiofiles
from aiologger import Logger
from async_lru import alru_cache
from dotenv import load_dotenv

MAX_CONCURRENT_REQUESTS = 200
UPDATE_INTERVAL = 3600
BATCH_SIZE = 1000
CACHE_SIZE = 50_000
MAX_QUEUE_SIZE = 5_000
PROCESS_BATCH_SIZE = 20
REPOST_BATCH_SIZE = 5000
URL_PATTERN = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

load_dotenv()
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
    'port': int(os.getenv('DB_PORT', 5432)),
}

try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
except Exception as e:
    print(f"Database connection error: {e}")
    exit(1)
    
class URLShortener:
    def __init__(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
        self.session = session
        self.semaphore = semaphore
        self.shorturl_services = self._load_shorturl_services()
        self._resolve_cache = {}

    def _load_shorturl_services(self) -> Set[str]:
        try:
            return set(pd.read_csv('./Gephi_Like/shorturl-services-list.csv').iloc[:, 0])
        except Exception as e:
            print(f"Error loading shorturl services: {e}")
            return set()

    @alru_cache(maxsize=CACHE_SIZE)
    async def fetch(self, url: str) -> Optional[str]:
        try:
            async with self.semaphore:
                async with self.session.head(url, allow_redirects=False) as response:
                    return str(response.real_url)
        except Exception:
            return None

    async def resolve_url(self, short_url: str) -> Optional[str]:
        if not short_url:
            return None
        if short_url in self._resolve_cache:
            return self._resolve_cache[short_url]
        if urlsplit(short_url).netloc in self.shorturl_services:
            resolved_url = await self.fetch(short_url)
            if resolved_url:
                self._resolve_cache[short_url] = resolved_url
                return resolved_url
        return short_url

class FileManager:
    def __init__(self):
        self.current_date = datetime.now().date()
        self.last_compression_date = datetime.now().date()
        self.post_file = None
        self.repost_file = None
        self.log_dir = "./Messages/bsky_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    async def initialize(self):
        await self._open_new_files()

    async def _open_new_files(self):
        date_str = self.current_date.strftime("%Y-%m-%d")
        post_filename = f"{self.log_dir}/posts_{date_str}.txt"
        repost_filename = f"{self.log_dir}/reposts_{date_str}.txt"
        self.post_file = await aiofiles.open(post_filename, "a", encoding='utf-8')
        self.repost_file = await aiofiles.open(repost_filename, "a", encoding='utf-8')

    async def _close_files(self):
        if self.post_file:
            await self.post_file.close()
        if self.repost_file:
            await self.repost_file.close()

    async def compress_files(self):
        now = datetime.now()
        yesterday_date_str = (now - timedelta(days=1)).date().strftime("%Y-%m-%d")
        yesterday_dir = os.path.join(self.log_dir, yesterday_date_str)
        os.makedirs(yesterday_dir, exist_ok=True)

        await self._close_files()
        for file_name in os.listdir(self.log_dir):
            if file_name.endswith('.txt'):
                file_date_str = file_name.split('_')[-1].split('.')[0]
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()

                if file_date < self.current_date:
                    input_file = os.path.join(self.log_dir, file_name)
                    output_file = os.path.join(yesterday_dir, file_name.replace('.txt', '') + '.gz')

                    async with aiofiles.open(input_file, 'rb') as f_in:
                        async with aiofiles.open(output_file, 'wb') as f_out:
                            while chunk := await f_in.read(1024):
                                await f_out.write(chunk)
                    os.remove(input_file)

    async def write_payload(self, payload: Any, is_repost: bool):
        try:
            if self.post_file is None or self.repost_file is None:
                raise RuntimeError("Files are not properly initialized. Please call 'initialize()' before writing payload.")
            current_date = datetime.now().date()
            if current_date != self.last_compression_date:
                await self.compress_files()
                self.last_compression_date = current_date
                await self._close_files()
                await self._open_new_files()
            if current_date != self.current_date:
                await self._close_files()
                self.current_date = current_date
                await self._open_new_files()
            file_to_write = self.repost_file if is_repost else self.post_file
            payload_str = str(payload).strip()
            if payload_str:
                await file_to_write.write(payload_str + '\n')
                await file_to_write.flush()
        except Exception as e:
            pass

class BatchProcessor:
    def __init__(self, stats_manager: 'StatsManager', file_manager: FileManager):
        self.stats = stats_manager
        self.file_manager = file_manager
        self.session: Optional[aiohttp.ClientSession] = None
        self.url_shortener: Optional[URLShortener] = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.logger = Logger.with_default_handlers(name='batch_processor')
        self.processing_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.url_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.feed_cache = {}
        self.feed_semaphore = asyncio.Semaphore(50)
        self.repost_queue = asyncio.Queue()
        self.processed_urls = set()

    async def initialize(self):
        conn = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(connector=conn)
        self.url_shortener = URLShortener(self.session, self.semaphore)        
        asyncio.create_task(self._process_repost_batches())
        for _ in range(MAX_CONCURRENT_REQUESTS // 2):
            asyncio.create_task(self._process_queue_worker())

    async def _process_repost_batches(self):
        try:
            async for repost_batch in self._generate_repost_batches():
                await asyncio.gather(*[self._fetch_repost_data(*data) for data in repost_batch])
        except Exception as e:
            await self.logger.error(f"Error processing repost batches: {e}")

    async def _generate_repost_batches(self):
        while True:
            batch = []
            while len(batch) < REPOST_BATCH_SIZE and self.repost_queue.qsize() > 0:
                batch.append(await self.repost_queue.get())
            if batch:
                yield batch
            else:
                await asyncio.sleep(0.1)

    async def _fetch_repost_data(self, uri: str, cid: str, new_did: str, created_at: str, py_type: str):
        try:
            did = uri.split('/')[2]
            cache_key = f"{did}:{cid}"
            if cache_key in self.feed_cache:
                await self._process_url_in_record(self.feed_cache[cache_key], is_repost=True)
                return

            async with self.feed_semaphore:
                await self._fetch_and_cache_feed(did, cid, new_did, created_at, py_type, cache_key)
        except Exception as e:
            pass

    async def _fetch_and_cache_feed(self, did, cid, new_did, created_at, py_type, cache_key):
        url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor={did}"
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    matching_item = next((item for item in data.get('feed', []) if item['post']['cid'] == cid), None)
                    if matching_item:
                        post_data = matching_item['post']
                        post_data.update({'originalDid': did, 'newDid': new_did, 'newcreatedAt': created_at, 'newType': py_type})
                        self.feed_cache[cache_key] = matching_item
                        await self._process_url_in_record(matching_item, is_repost=True)
                    else:
                        return
        except asyncio.TimeoutError:
            await self.logger.error(f"Timeout while fetching feed for did: {did}")

    async def _process_url_in_record(self, record, is_repost):
        if is_repost:
            await self.file_manager.write_payload(record, is_repost=is_repost)
            url = record['post'].get('embed', {}).get('external', {}).get('uri')
            if url:
                record['URL'] = url
                record['newsGuardScore'] = -1
        else:
            await self.file_manager.write_payload(record, is_repost=is_repost)
            embed = getattr(record, 'embed', None)
            if embed and hasattr(embed, 'external'):
                url = embed.external.uri
                setattr(record, 'URL', url)
                setattr(record, 'newsGuardScore', -1)
        if url:
            long_url, score = await self._process_url(url)
            record['URL'], record['newsGuardScore'] = long_url, score
            await self.url_queue.put(long_url)

    async def _process_queue_worker(self):
        while True:
            item = await self.processing_queue.get()
            await self._process_single_message(item)
            self.processing_queue.task_done()

    async def _process_single_message(self, commit: models.ComAtprotoSyncSubscribeRepos.Commit):
        try:
            car = CAR.from_bytes(commit.blocks)
            for op in commit.ops:
                if op.action == 'create' and op.cid:
                    raw = car.blocks.get(op.cid)
                    if raw:
                        lexRecord = models.get_or_create(raw, strict=False)
                        if lexRecord:
                            self.stats.increment('total_messages')
                            await self._process_based_on_record_type(lexRecord, commit)
        except Exception as e:
            pass

    async def _process_based_on_record_type(self, lexRecord, commit):
        if models.is_record_type(lexRecord, models.AppBskyFeedPost):
            await self._process_url_in_record(lexRecord, is_repost=False)
        elif lexRecord and (models.is_record_type(lexRecord, (models.AppBskyFeedLike)) or models.is_record_type(lexRecord, (models.AppBskyFeedRepost))):
            await self.repost_queue.put((lexRecord.subject.uri, lexRecord.subject.cid, commit.repo, lexRecord.created_at, lexRecord.py_type))

    async def _process_url(self, url: str) -> Tuple[str, int]:
        try:
            self.stats.increment('total_links')
            long_url = await self.url_shortener.resolve_url(url)
            if long_url:
                domain = extract_domain(long_url)
                if domain:
                    score = self.stats.get_domain_score(domain, long_url) if domain else -1
                    if score >= 60:
                        self.stats.increment('above_60')
                    elif score >= 0:
                        self.stats.increment('below_60')
                    return long_url, score
            return url, -1
        except Exception as e:
            await self.logger.error(f"Error processing URL: {e}")
            return url, -1

    async def process_record(self, commit: models.ComAtprotoSyncSubscribeRepos.Commit):
        await self.processing_queue.put(commit)

class StatsManager:
    def __init__(self):
        self.stats = Counter()
        self.domain_scores = {}
        self.last_update = time.time()

    def increment(self, key: str):
        self.stats[key] += 1

    def get_domain_score(self, domain: str, url: str) -> int:
        return self.domain_scores.get(domain, self.domain_scores.get(url, -1))

    async def load_domain_scores(self):
        try:
            metadata = pd.read_csv('./NewsGuard/metadata.csv', on_bad_lines='skip')
            all_sources = pd.read_csv('./NewsGuard/all-sources-metadata.csv', on_bad_lines='skip')
            
            self.domain_scores.update({
                row['Domain']: row['Score'] 
                for df in [all_sources, metadata] 
                for row in df.to_dict(orient='records')
            })
        except Exception as e:
            print(f"Error loading domain scores: {e}")

    def should_update(self) -> bool:
        return time.time() - self.last_update >= UPDATE_INTERVAL

    def log_stats(self):
        if self.should_update():
            if conn is None or conn.closed:
                print("Database connection is closed or invalid.")
            current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sql = """
            INSERT INTO bsky_news (day, totalmessages, totallinks, newsgreaterthan60, newslessthan60)
            VALUES (%s, %s, %s, %s, %s)
            """
            values = (current_datetime, self.stats.get('total_messages'), self.stats.get('total_links'), self.stats.get('above_60'), self.stats.get('below_60'))
            cursor.execute(sql, values)
            conn.commit()
            if self.stats.get('total_links') == 0:
                try:
                    subprocess.run(["pm2", "restart", "async_bluesky"], check=True, timeout=5)
                except subprocess.TimeoutExpired:
                    print("PM2 restart command timed out.")
                except subprocess.CalledProcessError as e:
                    print(f"PM2 restart failed with error: {e}")
            self.stats.clear()
            self.last_update = time.time()
@lru_cache(maxsize=CACHE_SIZE)
def extract_domain(url: str) -> Optional[str]:
    if not url:
        return None
    match = re.match(r"(?:https?://)?(?:www\d?\.)?([^/]+)", url)
    return match.group(1).lower() if match else None

async def main():
    uvloop.install()    
    loop = asyncio.get_event_loop()
    loop.set_debug(False)
    
    stats_manager = StatsManager()
    file_manager = FileManager()
    await stats_manager.load_domain_scores()
    await file_manager.initialize()
    processor = BatchProcessor(stats_manager, file_manager)
    await processor.initialize()
    
    firehose_client = AsyncFirehoseSubscribeReposClient()    
    firehose_workers = []
    for _ in range(5):
        worker_task = asyncio.create_task(run_firehose_worker(firehose_client, processor))
        firehose_workers.append(worker_task)
    
    try:
        await asyncio.gather(*firehose_workers)
    finally:
        await processor.session.close()
        if processor.batch_processing_task:
            processor.batch_processing_task.cancel()
        for worker in processor.processing_workers:
            worker.cancel()
        await processor.logger.shutdown()

async def run_firehose_worker(firehose_client, processor):
    async def worker(message):
        try:
            commit = parse_subscribe_repos_message(message)
            if isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
                if commit.seq % 100 == 0:
                    firehose_client.update_params(models.ComAtprotoSyncSubscribeRepos.Params(cursor=commit.seq))
                if commit.blocks:
                    await processor.process_record(commit)
                    processor.stats.log_stats()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(e)
    try:
        await firehose_client.start(worker)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())
