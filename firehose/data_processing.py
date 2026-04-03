import os
import sys
import json
import asyncio
import aiohttp
import psycopg2
import re
import logging
import gzip
import pandas as pd
from datetime import datetime
from collections import Counter
from urllib.parse import urlsplit
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

domain_scores = {}
shorturl_services = set()

db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
    'port': int(os.getenv('DB_PORT', 5432)),
}
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

def load_data():
    global shorturl_services
    try:
        metadata = pd.read_csv('./NewsGuard/metadata.csv', on_bad_lines='skip', low_memory=False)
        all_sources = pd.read_csv('./NewsGuard/all-sources-metadata.csv', on_bad_lines='skip', low_memory=False)
        domain_scores.update({row['Domain']: row['Score'] for df in [all_sources, metadata] for row in df.to_dict(orient='records')})
        shorturl_services = set(pd.read_csv("shorturl-services-list.csv").iloc[:, 0])
        logging.info("Metadata loaded.")
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")

def extract_domain(url: str):
    if not url: return None
    match = re.match(r"(?:https?://)?(?:www\d?\.)?([^/]+)", url)
    return match.group(1).lower() if match else None

async def unshorten_url(session, url):
    try:
        async with session.get(url, allow_redirects=False, timeout=10) as response:
            if 300 <= response.status < 400 and "Location" in response.headers:
                return response.headers["Location"]
            return str(response.real_url)
    except:
        return url

async def process_db_inserts(urls_to_process, stats):
    current_datetime = datetime.now()
    
    sql_stats = """
    INSERT INTO bsky_news (day, totalmessages, totallinks, newsgreaterthan60, newslessthan60)
    VALUES (%s, %s, %s, %s, %s)
    """
    cursor.execute(sql_stats, (current_datetime, stats['total_messages'], stats['total_links'], stats['above_60'], stats['below_60']))
    
    url_counts = Counter(urls_to_process)
    for url, count in url_counts.items():
        domain = extract_domain(url)
        if domain:
            domain_score = domain_scores.get(domain, -1)
            if domain_score >= 0:
                sql_urls = """
                    INSERT INTO newsguard_counts (url, domain, score, timestamp, count)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (url, timestamp) DO UPDATE
                    SET count = newsguard_counts.count + EXCLUDED.count
                """
                try:
                    cursor.execute(sql_urls, (url, domain, domain_score, current_datetime, count))
                except Exception as e:
                    logging.error(f"DB Insert Error: {e}")
                    conn.rollback()
                    continue
    conn.commit()

async def process_file_data(raw_file_path):
    processed_folder = "./Payload/processed_reposts_and_likes"
    processed_file_path = os.path.join(processed_folder, "processed_" + os.path.basename(raw_file_path))

    if not os.path.exists(processed_file_path):
        logging.error(f"Processed file not found: {processed_file_path}")
        return

    total_urls = []
    stats = {'total_messages': 0, 'total_links': 0, 'above_60': 0, 'below_60': 0}

    try:
        with gzip.open(processed_file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                total_urls.extend(record.get("urls", []))
                stats['total_messages'] += 1
    except Exception as e:
        logging.error(f"Error reading processed file {processed_file_path}: {e}")
        return

    if not total_urls:
        logging.info(f"No URLs found in {os.path.basename(processed_file_path)}. Skipping DB insert.")
        return

    async with aiohttp.ClientSession() as session:
        unshortened_urls = []
        tasks = [unshorten_url(session, url) if extract_domain(url) in shorturl_services else asyncio.sleep(0, result=url) for url in total_urls]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            if result:
                unshortened_urls.append(result)
                domain = extract_domain(result)
                if domain:
                    score = domain_scores.get(domain, -1)
                    if score >= 60: stats['above_60'] += 1
                    elif score >= 0: stats['below_60'] += 1
        
        stats['total_links'] = len(unshortened_urls)
        await process_db_inserts(unshortened_urls, stats)
        logging.info(f"Successfully processed and inserted data for {os.path.basename(raw_file_path)}")

async def main():
    if len(sys.argv) < 2:
        logging.error("No file path provided. Usage: python data_processing.py <path_to_raw_file>")
        return
        
    load_data()
    await process_file_data(sys.argv[1])

if __name__ == "__main__":
    asyncio.run(main())