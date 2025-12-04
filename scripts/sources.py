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
        self.domain_scores: Dict[str, int] = {}
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

    def load_domain_scores(self):
        try:
            metadata = pd.read_csv('./NewsGuard/metadata.csv', usecols=['Domain', 'Score'])
            all_sources = pd.read_csv('./NewsGuard/all-sources-metadata.csv', usecols=['Domain', 'Score'], on_bad_lines='skip')
            
            self.domain_scores = {
                **{row['Domain']: row['Score'] for _, row in all_sources.iterrows()},
                **{row['Domain']: row['Score'] for _, row in metadata.iterrows()}
            }
            del metadata, all_sources
            gc.collect()
        except Exception as e:
            print(f"Error loading domain scores: {e}")

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

    def get_domain_score(self, domain: str) -> int:
        return self.domain_scores.get(domain, -1)
    
    def get_orientation_score(self, domain: str) -> int:
        return self.orientation_scores.get(domain, -1)

    @lru_cache(maxsize=CACHE_SIZE)
    def extract_domain(self, url: str) -> Optional[str]:
        try:
            return urlsplit(url).netloc.lower()
        except Exception:
            return None

async def process_batch(url_processor: URLProcessor, batch: List[str]) -> Tuple[List[Dict[str, Union[str, int]]], List[Dict[str, Union[str, int]]]]:
    results = []
    reliable_results = []
    for text in batch:
        domains_found = DOMAIN_PATTERN.findall(text)
        for domain, path in domains_found:
            full_url = f"{domain}{path}"
            score = url_processor.get_domain_score(domain)
            orientation = url_processor.get_orientation_score(domain)
            if 0 < score < 60:
                results.append({'url': full_url, 'newsguard_score': score, 'orientation': orientation})
            elif score >= 60:
                reliable_results.append({'url': full_url, 'newsguard_score': score, 'orientation': orientation})

        if not results and url_processor.url_pattern:
            shortened_urls = url_processor.url_pattern.findall(text)
            for url in shortened_urls:
                unshortened_url = await url_processor.unshorten_url(url)
                domain = url_processor.extract_domain(unshortened_url)
                score = url_processor.get_domain_score(domain)
                orientation = url_processor.get_orientation_score(domain)
                if 0 < score < 60:
                    results.append({'url': unshortened_url, 'newsguard_score': score, 'orientation': orientation})
                elif score >= 60:
                    reliable_results.append({'url': unshortened_url, 'newsguard_score': score, 'orientation': orientation})

    return results, reliable_results

async def write_results_to_csv(results: List[Dict[str, Union[str, int]]], filename: str):
    fieldnames = ['url', 'newsguard_score']
    async with aiofiles.open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        await csvfile.seek(0, 2)
        if (await csvfile.tell()) == 0:
            await csvfile.write(','.join(fieldnames) + '\n')
        for row in results:
            await csvfile.write(f"{row['url']},{row['newsguard_score']},{row['orientation']}\n")

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
                data = ujson.loads(line)
                batch.append(data.get('text', ''))
                if len(batch) >= BATCH_SIZE:
                    results, reliable_results = await process_batch(url_processor, batch)
                    await write_results_to_csv(results, './Orientation/unreliable_old.csv')
                    await write_results_to_csv(reliable_results, './Orientation/reliable_old.csv')
                    pbar.update(len(batch))
                    batch.clear()
                    gc.collect()
            except (json.JSONDecodeError, ujson.JSONDecodeError):
                continue

        if batch:
            results, reliable_results = await process_batch(url_processor, batch)
            await write_results_to_csv(results, './Orientation/unreliable_old.csv')
            await write_results_to_csv(reliable_results, './Orientation/reliable_old.csv')
            pbar.update(len(batch))
            batch.clear()
            gc.collect()

        pbar.close()

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

async def process_messages_folder(folder_path: str):
    url_processor = URLProcessor()
    url_processor.load_domain_scores()
    url_processor.load_orientation_scores()

    async with aiohttp.ClientSession() as session:
        url_processor.session = session
        await url_processor.load_shorturl_services()
        files = [
            '2024-06-18.txt', '2024-06-20.txt', '2024-06-21.txt', '2024-06-23.txt',
            '2024-06-24.txt', '2024-06-25.txt', '2024-06-26.txt', '2024-06-27.txt',
            '2024-06-28.txt', '2024-06-29.txt', '2024-07-01.txt', '2024-07-30.txt',
            '2024-08-14.txt', '2024-10-09.txt', '2024-10-15.txt'
        ]
        
        for file in files:
            await process_file(os.path.join(folder_path, file), url_processor)
            gc.collect()

if __name__ == "__main__":
    folder_path = './Messages'
    asyncio.run(process_messages_folder(folder_path))
