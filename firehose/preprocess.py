import os
import sys
import json
import asyncio
import gzip
import aiohttp
import orjson
from datetime import datetime, timezone
from itertools import islice
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="There is no current event loop")

BASE_FOLDER = "./Payload"
MAX_CONCURRENT_REQUESTS = 2000
BATCH_SIZE = 25

def extract_uris(data):
    """
    Extracts unique URIs from the given data.
    Returns:
        list: A list of unique URIs extracted from the data.
    """
    uri_list = []
    if isinstance(data, dict):
        url = data.get('embed', {}).get('external', {}).get('uri')
        if url:
            uri_list.append(url)
    elif isinstance(data, list):
        for item in data:
            uri_list.extend(extract_uris(item))
    return list(set(uri_list))

def chunk_iterator(iterable, chunk_size):
    """
    Splits an iterable into chunks of a specified size.
    Returns:
        iterator: An iterator that yields chunks of the iterable.
    """
    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, chunk_size)), [])

async def fetch_batch_uris(session: aiohttp.ClientSession, uris_batch: List[str], processed_data: Dict[str, Any], error_log: List[str]) -> bool:
    """
    Fetches a batch of URIs and processes the response.
    Returns:
        bool: True if the batch was successfully processed, False otherwise.
    """
    try:
        base_url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts?"
        query_params = "&".join([f"uris[]={uri}" for uri in uris_batch])
        url = f"{base_url}{query_params}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json(loads=orjson.loads)
                for post in data.get('posts', []):
                    processed_data[post['uri']] = post
                return True
            elif response.status == 429:
                error_log.append(f"Rate limited for batch: {uris_batch}")
                return False
            else:
                error_log.append(f"Unexpected status {response.status} for batch: {uris_batch}")
                return False
    except Exception as e:
        error_log.append(f"Error processing batch: {str(e)}")
        return False

async def fetch_all_batches(uris_list: List[str], processed_data: Dict[str, Any], error_log: List[str], batch_size: int):
    """
    Fetches all batches of URIs by splitting them into smaller batches and processing them concurrently.
    Returns:
        bool: True if all batches were processed successfully, False if any batch failed.
    """
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(100)
        async def fetch_with_semaphore(uris_batch: List[str]):
            async with semaphore:
                return await fetch_batch_uris(session, uris_batch, processed_data, error_log)
        tasks = [fetch_with_semaphore(uris_batch) for uris_batch in chunk_iterator(uris_list, batch_size)]        
        results = await asyncio.gather(*tasks)        
        return all(results)
    
def process_files_sync(file_path: str):
    """
    Synchronously processes the given file by invoking an asynchronous file processing function.
    Returns:
        list: A list of processed records.
    """
    loop = asyncio.get_event_loop()    
    result = loop.run_until_complete(process_files(file_path))
    return result

async def process_files(file_path: str):
    """
    Asynchronously processes a file, extracts URIs, fetches corresponding data in batches, 
    and writes processed records to a new file.

    Args:
        file_path (str): The path to the gzipped file to process.

    Returns:
        list: A list of processed records.
    """
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS, 
    )
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        connector=connector
    ) as session:
        try:
            processed_uris = set()
            file_data = []
            post_data = []

            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = orjson.loads(line.strip())
                        commit = data.get("commit", {})
                        collection = commit.get("collection")

                        if collection == "app.bsky.feed.post":
                            urls = extract_uris(commit.get('record', {}))
                            if urls:
                                post_data.extend(urls)

                        if collection in {"app.bsky.feed.like", "app.bsky.feed.repost"}:
                            record = commit.get("record", {})
                            subject_uri = record.get("subject", {}).get("uri")
                            target_cid = record.get("subject", {}).get("cid")
                            
                            if subject_uri and target_cid:
                                if subject_uri not in processed_uris:
                                    processed_uris.add(subject_uri)
                                
                                file_data.append((data, subject_uri, target_cid))
                    except Exception as e:
                        print(f"Error processing line: {e}")
            
            processed_data = {}
            error_log = []            
            uris_list = list(processed_uris)
            await fetch_all_batches(uris_list, processed_data, error_log, BATCH_SIZE)         
            records = []
            for original_payload, uri, target_cid in file_data:
                matching_payload = processed_data.get(uri)
                if matching_payload:
                    updated_payload = {
                        "originalDid": original_payload['commit']['record']['subject']['uri'].split("/")[-3],
                        "newDid": original_payload['did'],
                        "newType": original_payload['commit']['collection'],
                        "text": matching_payload['record']['text'],
                        "urls": extract_uris(matching_payload),
                        "repostCount": matching_payload.get('repostCount', 0),
                        "likeCount": matching_payload.get('likeCount', 0),
                        "quoteCount": matching_payload.get('quoteCount', 0),
                        "newCreatedAt": datetime.fromtimestamp(original_payload['time_us'] / 1_000_000, tz=timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%dT%H:%M:%S.%f')
                    }
                    records.append(updated_payload)
            processed_folder = os.path.join(BASE_FOLDER, "processed_reposts_and_likes")
            os.makedirs(processed_folder, exist_ok=True)
            original_name = os.path.basename(file_path).replace(".gz", "")
            new_file_name = f"processed_{original_name}"
            new_file_path = os.path.join(processed_folder, new_file_name)
            with gzip.open(new_file_path + '.gz', 'at', encoding='utf-8') as f:
                post_json = json.dumps({"urls": post_data}, ensure_ascii=False, separators=(',', ':'), default=str)
                f.write(post_json + '\n')
                for obj in records:
                    obj_json = json.dumps(obj, ensure_ascii=False, separators=(',', ':'), default=str)
                    f.write(obj_json + '\n')
            print(f"Saved {len(records)} records to {new_file_path}")
            if error_log:
                with open('api_errors.log', 'w') as f:
                    f.write('\n'.join(error_log))
            return records
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
def main():
    """
    The main entry point for the script. It processes the file passed as an argument to the script.
    """
    process_files_sync(sys.argv[1])

if __name__ == "__main__":
    main()

