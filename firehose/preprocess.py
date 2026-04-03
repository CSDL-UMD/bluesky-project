import os
import sys
import json
import asyncio
import gzip
import aiohttp
import orjson
from datetime import datetime, timezone
from typing import List, Dict, Any

BASE_FOLDER = "./Payload"
PROCESSED_FOLDER = os.path.join(BASE_FOLDER, "processed_reposts_and_likes")
MAX_CONCURRENT_REQUESTS = 50
BATCH_SIZE = 25

def extract_uris(data):
    uri_list = []
    if not isinstance(data, dict):
        return []
    
    embed = data.get('embed', {})
    if isinstance(embed, dict):
        external = embed.get('external', {})
        if isinstance(external, dict):
            url = external.get('uri')
            if url: uri_list.append(url)
            
    subject = data.get('subject', {})
    if isinstance(subject, dict):
        subject_uri = subject.get('uri')
        if subject_uri: uri_list.append(subject_uri)
        
    return list(set(uri_list))

async def fetch_batch_uris(session, uris_batch, processed_data):
    if not uris_batch: return True
    base_url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts?"
    query_params = "&".join([f"uris[]={uri}" for uri in uris_batch])
    url = f"{base_url}{query_params}"
    
    for attempt in range(3):
        try:
            async with session.get(url, timeout=15) as response:
                if response.status == 200:
                    data = await response.json(loads=orjson.loads)
                    for post in data.get('posts', []):
                        processed_data[post['uri']] = post
                    return True
                elif response.status in {429, 500, 502, 503}:
                    await asyncio.sleep(2 ** attempt)
        except Exception:
            continue
    return False

async def process_file(file_path):
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    processed_data = {}
    file_queue = []
    post_urls_only = []
    uris_to_fetch = set()

    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = orjson.loads(line.strip())
                    commit = data.get("commit", {})
                    
                    if commit.get("operation") != "create":
                        continue
                        
                    collection = commit.get("collection")
                    record = commit.get("record")
                    if not record or not isinstance(record, dict):
                        continue

                    if collection == "app.bsky.feed.post":
                        urls = extract_uris(record)
                        if urls: post_urls_only.extend(urls)

                    elif collection in {"app.bsky.feed.like", "app.bsky.feed.repost"}:
                        subject_uri = record.get("subject", {}).get("uri")
                        if subject_uri:
                            uris_to_fetch.add(subject_uri)
                            file_queue.append((data, subject_uri))
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    async with aiohttp.ClientSession() as session:
        uri_list = list(uris_to_fetch)
        tasks = [
            fetch_batch_uris(session, uri_list[i:i+BATCH_SIZE], processed_data)
            for i in range(0, len(uri_list), BATCH_SIZE)
        ]
        await asyncio.gather(*tasks)

    records = []
    for original_payload, uri in file_queue:
        matching = processed_data.get(uri)
        if matching:
            try:
                time_us = int(original_payload.get('time_us', 0))
            except (TypeError, ValueError):
                time_us = 0
            matching_record = matching.get('record', {})
            
            records.append({
                "originalDid": uri.split("/")[-3] if len(uri.split("/")) >= 3 else "unknown",
                "newDid": original_payload.get('did'),
                "newType": original_payload.get('commit', {}).get('collection'),
                "text": matching_record.get('text', ''),
                "urls": extract_uris(matching_record),
                "newCreatedAt": datetime.fromtimestamp(time_us / 1_000_000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') if time_us else "unknown"
            })

    output_path = os.path.join(PROCESSED_FOLDER, "processed_" + os.path.basename(file_path))
    try:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            if post_urls_only:
                f.write(json.dumps({"urls": post_urls_only}) + '\n')
            for rec in records:
                f.write(json.dumps(rec) + '\n')
        print(f"[*] Done: {len(records)} enriched records saved to {os.path.basename(output_path)}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(process_file(sys.argv[1]))