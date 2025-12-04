import pandas as pd
from typing import Set, Dict, List, Tuple, Any, Optional
from itertools import islice
from functools import lru_cache
from datetime import datetime, timedelta
import re, traceback, collections, time, asyncio, aiohttp, psycopg2, subprocess, gzip, os, gc, json
from urllib.parse import quote, urlsplit
from atproto import CAR, models, AsyncFirehoseSubscribeReposClient, parse_subscribe_repos_message
from collections import defaultdict
import uvloop, aiofiles, statistics
from aiologger import Logger
from socket import error as SocketError
import errno

MAX_CONCURRENT_REQUESTS = 200
FEED_CONCURRENCY = 50
UPDATE_INTERVAL = 86400
MAX_QUEUE_SIZE = 100_000
MIN_BATCH_SIZE = 100
MAX_BATCH_SIZE = 10_000
REPOST_BATCH_SIZE = 500
URL_PATTERN = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

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

class FileManager:
    def __init__(self):
        self.current_date = datetime.now().date()
        self.last_compression_date = None
        self.post_file = None
        self.repost_file = None
        self.log_dir = "./Messages/bsky_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.compressing = asyncio.Lock()
        self.logger = Logger.with_default_handlers(name='file_manager')

    async def initialize(self):
        try:
            await self._open_new_files()
            # Remove immediate compression check
            asyncio.create_task(self._daily_compression_check())
        except Exception as e:
            await self.logger.error(f"Initialization error: {e}")
            raise

    async def _daily_compression_check(self):
        while True:
            # Wait for a full day before first compression
            await asyncio.sleep(24 * 60 * 60)  # 24 hours in seconds
            
            try:
                await self.check_and_compress()
            except Exception as e:
                await self.logger.error(f"Daily compression check failed: {e}")

    async def _open_new_files(self):
        try:
            date_str = self.current_date.strftime("%Y-%m-%d")
            post_filename = os.path.join(self.log_dir, f"posts_{date_str}.txt")
            repost_filename = os.path.join(self.log_dir, f"reposts_{date_str}.txt")

            await self._close_files()

            self.post_file = await aiofiles.open(post_filename, "a", encoding='utf-8')
            self.repost_file = await aiofiles.open(repost_filename, "a", encoding='utf-8')
        except Exception as e:
            await self.logger.error(f"Error opening new files: {e}")
            raise

    async def _close_files(self):
        try:
            if self.post_file:
                await self.post_file.close()
                self.post_file = None
            if self.repost_file:
                await self.repost_file.close()
                self.repost_file = None
        except Exception as e:
            await self.logger.error(f"Error closing files: {e}")
            raise

    async def check_and_compress(self):
        current_date = datetime.now().date()
        if self.last_compression_date is None or current_date > self.last_compression_date:
            await self.compress_files()
            self.last_compression_date = current_date
            return True
        return False

    async def compress_files(self):
        try:
            async with self.compressing:
                compression_task = self._do_compression()
                await asyncio.wait_for(compression_task, timeout=300)

        except asyncio.TimeoutError:
            await self.logger.error("Compression timeout - will retry later")
        except Exception as e:
            await self.logger.error(f"Compression error: {e}")

    async def _do_compression(self):
        now = datetime.now()
        yesterday = (now - timedelta(days=1)).date()
        yesterday_date_str = yesterday.strftime("%Y-%m-%d")
        yesterday_dir = os.path.join(self.log_dir, yesterday_date_str)
        os.makedirs(yesterday_dir, exist_ok=True)

        await self._close_files()

        compressed_count = 0
        for file_name in os.listdir(self.log_dir):
            if not file_name.endswith('.txt'):
                continue

            try:
                file_date_str = file_name.split('_')[-1].split('.')[0]
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            if file_date < yesterday:
                input_file = os.path.join(self.log_dir, file_name)
                
                # Only compress if file is not empty
                if os.path.getsize(input_file) > 0:
                    moved_file = os.path.join(yesterday_dir, file_name)
                    output_file = os.path.join(yesterday_dir, file_name.replace('.txt', '.gz'))

                    try:
                        # Use synchronous compression to ensure file is actually compressed
                        self._compress_single_file(input_file, moved_file, output_file)
                        compressed_count += 1
                    except Exception as e:
                        await self.logger.error(f"Error compressing file {file_name}: {e}")
                else:
                    # Remove empty files
                    os.remove(input_file)

        await self._open_new_files()
        return compressed_count

    def _compress_single_file(self, input_file: str, moved_file: str, output_file: str):
        # Ensure input file is closed before moving
        with open(input_file, 'rb') as f_in:
            # Read full content before moving/compressing
            file_content = f_in.read()

        # Move file first
        os.rename(input_file, moved_file)

        # Compress if content exists
        if file_content:
            with gzip.open(output_file, 'wb') as f_out:
                f_out.write(file_content)
            
            # Remove original file after successful compression
            os.remove(moved_file)

    async def write_payload(self, payload: Any, is_repost: bool):
        try:
            if self.post_file is None or self.repost_file is None:
                await asyncio.wait_for(self.initialize(), timeout=5.0)

            file_to_write = self.repost_file if is_repost else self.post_file
            payload_str = str(payload).strip()

            if payload_str:
                await asyncio.wait_for(
                    self._write_and_flush(file_to_write, payload_str),
                    timeout=2.0
                )

        except asyncio.TimeoutError:
            await self.logger.error("Timeout while writing payload")
            await self._reopen_files()
        except Exception as e:
            await self.logger.error(f"Error writing payload: {e}")

    async def _write_and_flush(self, file, payload_str):
        await file.write(payload_str + '\n')
        await file.flush()

    async def _reopen_files(self):
        try:
            await self._close_files()
            await self._open_new_files()
        except Exception as e:
            await self.logger.error(f"Error reopening files: {e}") 

class BatchProcessor:
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
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
                await asyncio.sleep(3600)                
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
        try:
            print("\n=== Stats Update ===")
            print(f"Total Messages: {self.stats['total_messages']}")
            print(f"Total Links: {self.stats['total_links']}")
            print(f"Scores >= 60: {self.stats['above_60']}")
            print(f"0 <= Scores < 60: {self.stats['below_60']}")
            print(f"Link Processing Rate: {self.stats['total_links']/UPDATE_INTERVAL:.2f} links/second")
            print("===================\n")
            self.stats.clear()
            self.last_update = time.time()
        except Exception as e:
            print(e)

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
                await self.file_manager.write_payload(record, is_repost=True)
                url = record['post'].get('embed', {}).get('external', {}).get('uri', None)
                if url:
                    record['URL'] = url
            else:
                await self.file_manager.write_payload(record, is_repost=False)
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
        file_manager = FileManager()     
        processor = BatchProcessor(file_manager)        
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
