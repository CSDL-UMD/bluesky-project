import pandas as pd
from typing import Set, Tuple, Optional
from aiologger import Logger
from datetime import datetime
import re, uvloop, traceback, collections, time, asyncio, aiohttp, psycopg2, subprocess, errno, os
from urllib.parse import urlsplit
from atproto import CAR, models, AsyncFirehoseSubscribeReposClient, parse_subscribe_repos_message
from socket import error as SocketError
from dotenv import load_dotenv
load_dotenv()

MAX_CONCURRENT_REQUESTS = 200
FEED_CONCURRENCY = 50
UPDATE_INTERVAL = 3600
MAX_QUEUE_SIZE = 1000000
MIN_BATCH_SIZE = 100
MAX_BATCH_SIZE = 10_000
REPOST_BATCH_SIZE = 5000
URL_PATTERN = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

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

    def _load_shorturl_services(self) -> Set[str]:
        try:
            services = set(pd.read_csv('./Gephi_Like/shorturl-services-list.csv').iloc[:, 0])
            return services
        except Exception as e:
            return set()

    async def fetch(self, url: str) -> Optional[str]:
        try:
            async with self.semaphore:
                async with self.session.get(url, allow_redirects=False) as response:
                    if 300 <= response.status < 400 and "Location" in response.headers:
                        resolved_url = response.headers["Location"]
                        return resolved_url
                    result = str(response.real_url)
                    return result
        except Exception as e:
            return None

    async def resolve_url(self, short_url: str) -> Optional[str]:
        if not short_url:
            return None
        if urlsplit(short_url).netloc in self.shorturl_services:
            resolved_url = await self.fetch(short_url)
            if resolved_url:
                return resolved_url
        return short_url
            
class BatchProcessor:
    def __init__(self):
        self.session = None
        self.url_shortener = None
        self.feed_semaphore = asyncio.Semaphore(FEED_CONCURRENCY)
        self.rep_semaphore = asyncio.Semaphore(FEED_CONCURRENCY)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.rate_limiter = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS * 2)
        self.processing_queue = asyncio.Queue()
        self.active_reposts = asyncio.Queue()
        self.repost_buffer = []
        self.repost_lock = asyncio.Lock()
        
        self.processing_stats = collections.deque(maxlen=1000)
        self.domain_scores = {}
        self.stats = collections.Counter()
        self.last_overflow_warning = 0
        
        self.processing_window = 30
        self.batch_size = MIN_BATCH_SIZE
        self.batch_lock = asyncio.Lock()
        self.batch_queue = asyncio.Queue(maxsize=500)

        self.logger = Logger.with_default_handlers(name='batch_processor')
        self.last_queue_adjustment = time.time()
        self.adjustment_cooldown = 60
        
    async def initialize(self):
        try:
            retries = 0
            while retries < 3:
                try:
                    conn = aiohttp.TCPConnector(
                        limit=MAX_CONCURRENT_REQUESTS,
                        ttl_dns_cache=300,
                        force_close=False,
                        enable_cleanup_closed=True
                    )
                    
                    self.session = aiohttp.ClientSession(
                        connector=conn,
                        timeout=aiohttp.ClientTimeout(total=30),
                        raise_for_status=False 
                    )
                    
                    self.url_shortener = URLShortener(self.session, self.semaphore)            
                    await self._load_domain_scores()
                    await self._start_workers()
                    await self._initialize_monitoring()
                    break
                    
                except Exception as e:
                    retries += 1
                    await self.logger.error(f"Initialization attempt {retries} failed: {e}\n{traceback.format_exc()}")
                    if retries >= 3:
                        raise
                    await asyncio.sleep(5)
            
        except Exception as e:
            await self.logger.error(f"Initialization failed: {e}\n{traceback.format_exc()}")
            if self.session:
                await self.session.close()
            raise
    
    async def _start_workers(self):
        worker_count = min(os.cpu_count() * 2, MAX_CONCURRENT_REQUESTS)
        self.workers = []        
        for _ in range(worker_count):
            worker = asyncio.create_task(self._process_queue_worker())
            self.workers.append(worker)        
        repost_workers = max(4, FEED_CONCURRENCY // 2)
        for _ in range(repost_workers):
            worker = asyncio.create_task(self._repost_batch_worker())
            self.workers.append(worker)

    async def _initialize_monitoring(self):
        asyncio.create_task(self._monitor_queues())
        asyncio.create_task(self._log_stats_periodically())

    async def _monitor_queues(self):
        while True:
            try:
                await asyncio.sleep(60)                
                proc_size = self.processing_queue.qsize()
                repost_size = self.active_reposts.qsize()                
                if time.time() - self.last_queue_adjustment > 10:
                    self.logger.info(
                        f"Queue status - Processing: {proc_size}/{MAX_QUEUE_SIZE}, "
                        f"Reposts: {repost_size}/{MAX_QUEUE_SIZE}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Queue monitoring error: {e}")
                await asyncio.sleep(5)

    async def _repost_batch_worker(self):
        while True:
            try:
                batch_items = []
                tasks = []
                
                proc_size = self.processing_queue.qsize()
                repost_size = self.active_reposts.qsize()
                batch_size = self.batch_queue.qsize()
                try:
                    while len(batch_items) < REPOST_BATCH_SIZE:
                        try:
                            repost_data = await asyncio.wait_for(
                                self.active_reposts.get(), 
                                timeout=1.0
                            )
                            if not repost_data or not isinstance(repost_data, tuple) or len(repost_data) != 5:
                                continue
                                
                            batch_items.append(repost_data)
                            self.active_reposts.task_done()
                            
                        except asyncio.TimeoutError:
                            if batch_items:
                                break
                            await asyncio.sleep(0.1)
                            continue
                        except Exception as e:
                            await asyncio.sleep(0.1)
                            continue
                            
                except Exception as e:
                    continue

                if not batch_items:
                    await asyncio.sleep(0.1)
                    continue

                try:                    
                    for repost in batch_items:
                        try:
                            task = asyncio.create_task(self._fetch_repost_data(*repost))
                            tasks.append(task)
                        except Exception as e:
                            continue

                    if tasks:
                        done, pending = await asyncio.wait(
                            tasks,
                            timeout=30,
                            return_when=asyncio.ALL_COMPLETED
                        )
                        
                        for task in pending:
                            task.cancel()                            
                        for task in done:
                            try:
                                await task
                            except Exception as e:
                                await self.logger.error(f"Task error: {e}\n{traceback.format_exc()}")
                                
                except Exception as e:
                    await self.logger.error(f"Batch processing error: {e}\n{traceback.format_exc()}")
                    
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    
                    await asyncio.sleep(0.1)
            except Exception as e:
                await self.logger.error(f"Worker loop error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(1)

    async def _process_queue_worker(self):
        while True:
            batch = []
            try:                
                async with self.batch_lock:
                    while len(batch) < self.batch_size:
                        try:
                            item = await asyncio.wait_for(
                                self.processing_queue.get(),
                                timeout=0.05
                            )
                            batch.append(item)
                        except asyncio.TimeoutError:
                            break
                
                if batch:
                    chunk_size = min(10, len(batch))
                    for i in range(0, len(batch), chunk_size):
                        chunk = batch[i:i + chunk_size]
                        await asyncio.gather(*[self._process_single_message(item) for item in chunk])
                        
                    for _ in batch:
                        self.processing_queue.task_done()
                else:
                    await asyncio.sleep(0.05)
                    
            except Exception as e:
                await self.logger.error(f"Error in queue worker: {e}")
                for _ in batch:
                    self.processing_queue.task_done()

# process write to file (separate script)
# another process opens file and processes each line
# optimize batch processing
    async def _process_single_message(self, commit):
        try:
            car = CAR.from_bytes(commit.blocks)
            
            for op in commit.ops:
                if op.action == 'create' and op.cid:
                    raw = car.blocks.get(op.cid)
                    if raw:
                        record = models.get_or_create(raw, strict=False)
                        if record:
                            self.stats['total_messages'] += 1
                            await self._process_based_on_record_type(record, commit)
                            
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")

    async def process_record(self, commit):
        try:
            await self.processing_queue.put(commit)
        except asyncio.QueueFull:
            pass           
        except Exception as e:
            self.logger.error(f"Error queuing record: {e}")

    async def _load_domain_scores(self):
        try:
            metadata = pd.read_csv('./NewsGuard/metadata.csv', on_bad_lines='skip')
            all_sources = pd.read_csv('./NewsGuard/all-sources-metadata.csv', on_bad_lines='skip')
            
            self.domain_scores.update({
                row['Domain']: row['Score'] 
                for df in [all_sources, metadata] 
                for row in df.to_dict(orient='records')
            })
        except Exception as e:
            self.logger.error(f"Error loading domain scores: {e}")

    async def _log_stats_periodically(self):
        while True:
            try:
                await asyncio.sleep(UPDATE_INTERVAL)
                await self._log_stats()
            except Exception as e:
                self.logger.error(f"Stats logging error: {e}")

    async def _log_stats(self):
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
                subprocess.run(["pm2", "restart", "bluesky"], check=True)
            except subprocess.TimeoutExpired:
                print("PM2 restart command timed out.")
            except subprocess.CalledProcessError as e:
                print(f"PM2 restart failed with error: {e}")
        self.stats.clear()
        self.last_update = time.time()

    async def _fetch_repost_data(self, uri: str, cid: str, new_did: str, created_at: str, py_type: str):
        try:
            if not all([uri, cid, new_did, created_at, py_type]):
                await self.logger.warning(f"Invalid repost parameters: {uri}, {cid}, {new_did}, {created_at}, {py_type}")
                return

            did = uri.split('/')[2] if len(uri.split('/')) > 2 else None
            if not did:
                await self.logger.warning(f"Invalid URI format: {uri}")
                return

            url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor={did}"
            
            async with self.feed_semaphore:
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if not data or 'feed' not in data:
                                await self.logger.warning(f"No feed data for {did}")
                                return
                            matching_item = next(
                                (item for item in data.get('feed', [])
                                if item.get('post', {}).get('cid') == cid),
                                None
                            )
                            
                            if matching_item:
                                post_data = matching_item.get('post', {})
                                post_data.update({
                                    'originalDid': did,
                                    'newDid': new_did,
                                    'newcreatedAt': created_at,
                                    'newType': py_type
                                })
                                await self._process_url_in_record(matching_item, is_repost=True)
                        else:
                            pass
                            
                except aiohttp.ClientError as e:
                    if e.errno != errno.ECONNRESET:
                        raise
                    pass
                except SocketError as e:
                    if e.errno != errno.ECONNRESET:
                        raise
                    pass
                except asyncio.TimeoutError:
                    await self.logger.error(f"Timeout fetching repost for {url}")
                except Exception as e:
                    await self.logger.error(f"Error fetching repost data: {e}\n{traceback.format_exc()}")
                    
        except Exception as e:
            await self.logger.error(f"Error in _fetch_repost_data: {e}\n{traceback.format_exc()}")

    async def _process_url(self, url: str) -> Tuple[str, int]:
        try:
            self.stats['total_links'] += 1
            long_url = await self.url_shortener.resolve_url(url)
            if long_url:
                domain = extract_domain(long_url)
                if domain:
                    score = self.domain_scores.get(domain, self.domain_scores.get(long_url, -1)) if domain else -1
                    if score >= 60:
                        self.stats['above_60'] += 1
                    elif score >= 0:
                        self.stats['below_60'] += 1
                    return long_url, score
            return url, -1
        except Exception as e:
            self.logger.error(f"Process Url: {e}")
            return url, -1

    async def _process_single_message(self, commit):
        try:
            car = CAR.from_bytes(commit.blocks)
            
            for op in commit.ops:
                if op.action == 'create' and op.cid:
                    raw = car.blocks.get(op.cid)
                    if raw:
                        record = models.get_or_create(raw, strict=False)
                        if record:
                            self.stats['total_messages'] += 1
                            await self._process_based_on_record_type(record, commit)
                            
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")

    async def _process_based_on_record_type(self, lexRecord, commit):
        try:
            if models.is_record_type(lexRecord, models.AppBskyFeedPost):
                await self._process_url_in_record(lexRecord, is_repost=False)
            elif lexRecord and (models.is_record_type(lexRecord, (models.AppBskyFeedLike)) or
                                models.is_record_type(lexRecord, (models.AppBskyFeedRepost))):
                repost_data = (
                    lexRecord.subject.uri,
                    lexRecord.subject.cid,
                    commit.repo,
                    lexRecord.created_at,
                    lexRecord.py_type
                )
                try:
                    if self.active_reposts.qsize() < MAX_QUEUE_SIZE:
                        await self.active_reposts.put(repost_data)
                    else:
                        await asyncio.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Based on record type error: {e}")
                
        except Exception as e:
            self.logger.error(f"Record processing error: {e}")

    async def _process_url_in_record(self, record, is_repost):
        try:
            url = None
            if is_repost:
                url = record['post'].get('embed', {}).get('external', {}).get('uri', None)
                if url:
                    record['URL'] = url
            else:
                embed = getattr(record, 'embed', None)
                if embed and hasattr(embed, 'external'):
                    url = getattr(embed.external, 'uri', None)
                    if url:
                        setattr(record, 'URL', url)
                        setattr(record, 'newsGuardScore', -1)

            if is_repost and url:
                long_url, score = await self._process_url(url)
                record['URL'] = long_url
                record['newsGuardScore'] = score
            elif url:
                long_url, score = await self._process_url(url)
                setattr(record, 'URL', long_url)
                setattr(record, 'newsGuardScore', score)
        except Exception as e:
            self.logger.error(f"Url in record processing error: {e}")
            pass

def extract_domain(url: str) -> Optional[str]:
    if not url:
        return None
    match = re.match(r"(?:https?://)?(?:www\d?\.)?([^/]+)", url)
    return match.group(1).lower() if match else None

async def run_firehose_worker(firehose_client, processor):
    async def worker(message):
        try:
            commit = parse_subscribe_repos_message(message)
            if isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
                if commit.seq % 100 == 0:
                    firehose_client.update_params(models.ComAtprotoSyncSubscribeRepos.Params(cursor=commit.seq))
                if commit.blocks:
                    await processor.process_record(commit)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            pass 
    try:
        await firehose_client.start(worker)
    except asyncio.CancelledError:
        pass

async def main():    
    try:
        uvloop.install()        
        loop = asyncio.get_event_loop()
        loop.set_debug(False)   
        processor = BatchProcessor()        
        await processor.initialize()        
        firehose_client = AsyncFirehoseSubscribeReposClient()        
        try:
            await run_firehose_worker(firehose_client, processor)
        except Exception as e:
            print(e)
        finally:
            await processor.session.close()
            await processor.logger.shutdown()
            
    except Exception as e:
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        exit(1)
