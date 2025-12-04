import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from urllib.parse import urlsplit
from typing import Set, List, Optional, Dict, Union, Tuple
import csv
import gzip
import os
import json
import re
import asyncio
import aiohttp
import aiofiles
import ujson
import gc

CACHE_SIZE = 10000
BATCH_SIZE = 1000
DOMAIN_PATTERN = re.compile(r'(?P<domain>[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(/[^\s]*)?')

class URLProcessor:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(500)
        self._resolve_cache: Dict[str, str] = {}
        self.shorturl_services: Set[str] = set()
        self.orientation_scores: Dict[str, int] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.url_pattern: Optional[re.Pattern] = None

    async def load_shorturl_services(self):
        try:
            with open('./Gephi_Like/shorturl-services-list.csv', mode='r') as f:
                content = f.read()
                self.shorturl_services = set(content.splitlines())
                escaped_services = [re.escape(service) for service in self.shorturl_services]
                self.url_pattern = re.compile(rf'https?://(?:www\.)?(?:{"|".join(escaped_services)})/[^\s]*', re.IGNORECASE)
        except Exception as e:
            print(f"Error loading shorturl services: {e}")

    async def fetch(self, url: str) -> Optional[str]:
        if not url.startswith(('http://', 'https://')):
            return None
        try:
            async with self.semaphore:
                async with self.session.head(url, allow_redirects=True, timeout=5) as response:
                    return str(response.url)
        except Exception:
            return None

    async def unshorten_url(self, short_url: str) -> str:
        if not short_url or short_url in self._resolve_cache:
            return self._resolve_cache.get(short_url, short_url)

        if urlsplit(short_url).netloc in self.shorturl_services:
            resolved_url = await self.fetch(short_url)
            if resolved_url:
                self._resolve_cache[short_url] = resolved_url
                return resolved_url
        return short_url

    def load_orientation_scores(self):
        try:
            metadata = pd.read_csv('./NewsGuard/metadata.csv', usecols=['Domain', 'Orientation'])
            all_sources = pd.read_csv('./NewsGuard/all-sources-metadata.csv', usecols=['Domain', 'Orientation'], on_bad_lines='skip')
            self.orientation_scores = {
                **{row['Domain']: row['Orientation'] for _, row in all_sources.iterrows()},
                **{row['Domain']: row['Orientation'] for _, row in metadata.iterrows()}
            }
            del metadata, all_sources
            gc.collect()
        except Exception as e:
            print(f"Error loading domain scores: {e}")

    def get_orientation_score(self, domain: str) -> int:
        return self.orientation_scores.get(domain, -1)

    @lru_cache(maxsize=CACHE_SIZE)
    def extract_domain(self, url: str) -> Optional[str]:
        try:
            return urlsplit(url).netloc.lower()
        except Exception:
            return None

def extract_url_and_score(text: str):
    try:
        url_pattern = r"URL='([^']*)'"
        score_pattern = r"newsGuardScore=([^\s]+)"

        url_match = re.search(url_pattern, text)
        score_match = re.search(score_pattern, text)

        url = url_match.group(1) if url_match else None
        score = score_match.group(1) if score_match else None

        return url, score
    except Exception as e:
        print(f"Error extracting URL and score: {e}")
        return None, None
    
async def process_batch(batch: List[Dict[str, Union[str, int]]], url_processor: URLProcessor) -> Tuple[List[Dict[str, Union[str, int]]], List[Dict[str, Union[str, int]]]]:
    results = []
    reliable_results = []
    for entry in batch:
        url, news_guard_score = extract_url_and_score(entry)
        if url is not None and news_guard_score is not None:
            if 0 < float(news_guard_score) < 60:
                parsed_url = urlsplit(url)
                domain = parsed_url.netloc
                if domain.startswith("www."):
                    domain = domain[4:]
                orientation = url_processor.get_orientation_score(domain)
                results.append({'url': url, 'newsguard_score': news_guard_score, 'orientation': orientation})
            elif float(news_guard_score) >= 60:
                parsed_url = urlsplit(url)
                domain = parsed_url.netloc
                if domain.startswith("www."):
                    domain = domain[4:]
                orientation = url_processor.get_orientation_score(domain)
                reliable_results.append({'url': url, 'newsguard_score': news_guard_score, 'orientation': orientation})
    return results, reliable_results


async def write_results_to_csv(results: List[Dict[str, Union[str, int]]], filename: str):
    fieldnames = ['url', 'newsguard_score']
    async with aiofiles.open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        await csvfile.seek(0, 2)
        if (await csvfile.tell()) == 0:
            await csvfile.write(','.join(fieldnames) + '\n')
        for row in results:
            await csvfile.write(f"{row['url']},{row['newsguard_score']}, {row['orientation']}\n")

async def process_file(filename: str, url_processor: URLProcessor) -> None:
    try:
        pbar = tqdm(desc=f"Processing {os.path.basename(filename)}", unit="line")

        def read_lines():
            if filename.endswith('.gz'):
                with gzip.open(filename, 'rt') as f:
                    for line in f:
                        yield line
            else:
                with open(filename, 'r') as f:
                    for line in f:
                        yield line

        batch = []
        for line in read_lines():
            try:
                batch.append(line)
                if len(batch) >= BATCH_SIZE:
                    results, reliable_results = await process_batch(batch, url_processor)
                    await write_results_to_csv(results, './Orientation/unreliable.csv')
                    await write_results_to_csv(reliable_results, './Orientation/reliable.csv')
                    pbar.update(len(batch))
                    batch.clear()
                    gc.collect()
            except (json.JSONDecodeError, ujson.JSONDecodeError):
                continue

        if batch:
            results, reliable_results = await process_batch(batch, url_processor)
            await write_results_to_csv(results, './Orientation/unreliable.csv')
            await write_results_to_csv(reliable_results, './Orientation/reliable.csv')
            pbar.update(len(batch))
            batch.clear()
            gc.collect()

        pbar.close()

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

async def process_messages_folder(folder_path: str):
    url_processor = URLProcessor()
    url_processor.load_orientation_scores()

    async with aiohttp.ClientSession() as session:
        url_processor.session = session
        await url_processor.load_shorturl_services()
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(('.gz', '.txt')):
                    files.append(os.path.join(root, filename))
        for file in files:
            await process_file(file, url_processor)
            gc.collect()

if __name__ == "__main__":
    folder_path = './Messages/bsky_logs'
    asyncio.run(process_messages_folder(folder_path))
