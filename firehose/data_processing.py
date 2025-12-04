import pandas as pd
import os, json, gzip, aiohttp, asyncio, warnings
from datetime import datetime
import re, os, sys, time, psycopg2
from typing import Set, Optional
from urllib.parse import urlsplit
from aiofiles import open as aio_open
from collections import Counter
from math import isnan
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

load_dotenv()
shorturl_services: Set[str] = set()
domain_scores = {}
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

async def load_domain_scores():
    """
    Asynchronously loads domain trust scores from metadata CSV files (NewsGuard).
    This function reads two CSV files, 'metadata.csv' and 'all-sources-metadata.csv', 
    and updates the global `domain_scores` dictionary with the domain name as the key 
    and its corresponding trust score as the value.

    """
    try:
        metadata = pd.read_csv('./NewsGuard/metadata.csv', on_bad_lines='skip')
        all_sources = pd.read_csv('./NewsGuard/all-sources-metadata.csv', on_bad_lines='skip')
        
        domain_scores.update({
            row['Domain']: row['Score'] 
            for df in [all_sources, metadata] 
            for row in df.to_dict(orient='records')
        })
    except Exception as e:
        print(f"Error loading domains: {e}")

def load_short_url_domains(filepath: str) -> Set[str]:
    """
    Loads and returns a set of short URL domains from a CSV file.
    This function reads a CSV file containing a list of short URL service domains, 
    and returns them as a set of strings.
    Returns:
        Set[str]: A set of short URL domains.
    """
    try:
        services = set(pd.read_csv(filepath).iloc[:, 0])
        return services
    except Exception as e:
        print(f"Error loading short URL domains: {e}")
        return set()

def is_valid_url(url: str) -> bool:
    return len(url) <= 8191

def is_short_url(url: str, short_domains: Set[str]) -> bool:
    """
    Checks if a given URL belongs to a short URL service.
    This function uses the `urlsplit` function to extract the domain from the provided 
    URL and checks if it exists in the provided set of short URL domains.
    Returns:
        bool: True if the URL is a short URL, False otherwise.
    """
    try:
        return urlsplit(url).netloc in short_domains
    except Exception as e:
        return False

def extract_domain(url: str) -> Optional[str]:
    """
    Extracts the domain from a given URL.
    This function uses a regular expression to extract the domain from a URL. 
    It returns the domain in lowercase or `None` if the domain cannot be extracted.
    Returns:
        Optional[str]: The domain extracted from the URL, or None if extraction fails.
    """
    if not url:
        return None
    match = re.match(r"(?:https?://)?(?:www\d?\.)?([^/]+)", url)
    return match.group(1).lower() if match else None

async def unshorten_url(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """
    Resolves a short URL to its final destination using HTTP redirects.
    This function sends an HTTP GET request to the provided URL and checks for a redirect. 
    If a redirect occurs (status code 3xx), the function returns the URL to which it redirects. 
    If no redirect is found, the final resolved URL is returned.
    Returns:
        Optional[str]: The resolved URL or None if an error occurs.
    """
    try:
        async with session.get(url, allow_redirects=False) as response:
            if 300 <= response.status < 400 and "Location" in response.headers:
                resolved_url = response.headers["Location"]
                return resolved_url
            return str(response.real_url)
    except Exception as e:
        return None

def wait_for_file(filepath: str, max_wait_time: int = 60):
    """
    Waits for a file to be created or updated within a given time frame.
    This function checks if the specified file exists and is accessible. If the file 
    is a gzip file, it attempts to read it. If the file is not accessible, the function 
    retries until the maximum wait time is reached.
    """
    start_time = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if not os.path.exists(filepath):
                print(f"[{current_time}] File {filepath} does not exist. Retrying...")
                time.sleep(0.5)
                continue
            if filepath.endswith(".gz"):
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    json.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json.load(f)
            break

        except (OSError, json.JSONDecodeError) as e:
            elapsed_time = time.time() - start_time
            print(f"[{datetime.now()}] Error while reading file {filepath}: {e}")
            if elapsed_time >= max_wait_time:
                raise TimeoutError(f"File {filepath} is not ready after {max_wait_time} seconds.")
            time.sleep(0.5)

def get_urls_from_file(filepath: str) -> tuple:
    """
    Extracts and returns URLs from a gzipped JSON file.
    This function reads a gzipped JSON file line by line, attempts to parse the content 
    as JSON, and extracts all URLs found within the "urls" field of the JSON data.
    Returns:
        tuple: A tuple containing:
            - List of URLs extracted from the file.
            - The total number of messages processed.
    """
    urls = []
    total_messages = 0
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                total_messages += 1
                try:
                    data = json.loads(line)
                    urls.extend(data["urls"])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return urls, total_messages

async def process_urls(short_domains: Set[str], total_urls: list) -> tuple:
    """
    Processes and resolves URLs, unshortening short URLs and keeping long URLs.
    This function iterates over a list of URLs, checks if they are short URLs, 
    and resolves them using the `unshorten_url` function. Long URLs are kept unchanged.
    Returns:
        tuple: A list of resolved long URLs and short URLs that were unshortened.
    """
    unshortened_urls = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in total_urls:
            if is_short_url(url, short_domains):
                tasks.append(unshorten_url(session, url))
            else:
                unshortened_urls.append(url)

        results = await asyncio.gather(*tasks)
        for result in results:
            if result:
                unshortened_urls.append(result)

    return unshortened_urls

async def process_file_pair(processed_file: str, short_domains: Set[str]):
    """
    Processes a file containing URLs, unshortens them, classifies them by domain trustworthiness, 
    and generates a report. This function reads URLs from a processed file, unshortens short URLs, 
    classifies the URLs by their domain trust score (using the `domain_scores` dictionary), 
    and writes the results to a new text file.
    """
    try:
        total_urls, total_messages = get_urls_from_file(processed_file)
        unshortened_urls = await process_urls(short_domains, total_urls)
        stats = {
            'total_messages': total_messages,
            'total_links': len(unshortened_urls),
            'above_60': 0,
            'below_60': 0
        }

        for long_url in unshortened_urls:
            domain = extract_domain(long_url)
            if domain:
                score = domain_scores.get(domain, -1)
                if score >= 60:
                    stats['above_60'] += 1
                elif score >= 0:
                    stats['below_60'] += 1
        
        if conn is None or conn.closed:
            print("Database connection is closed or invalid.")
        current_datetime = datetime.strptime(re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', processed_file).group(1) + " " + re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', processed_file).group(2).replace('-', ':'), "%Y-%m-%d %H:%M:%S")
        sql = """
        INSERT INTO bsky_news (day, totalmessages, totallinks, newsgreaterthan60, newslessthan60)
        VALUES (%s, %s, %s, %s, %s)
        """
        values = (current_datetime, stats['total_messages'], stats['total_links'], stats['above_60'], stats['below_60'])
        cursor.execute(sql, values)
        conn.commit()
        
        url_counts = Counter(unshortened_urls)
        for url, count in url_counts.items():
            if not is_valid_url(url):
                continue
            domain = extract_domain(url)
            if domain is None:
                continue  

            domain_score = domain_scores.get(domain, -1)
            if isnan(domain_score):
                domain_score = -1
            sql = """
                INSERT INTO newsguard_counts (url, domain, score, timestamp, count)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (url, timestamp) DO UPDATE
                SET count = newsguard_counts.count + EXCLUDED.count
            """
            if domain_score >= 0:
                values = (url, domain, domain_score, current_datetime, count)
                try:
                    cursor.execute(sql, values)
                except Exception as e:
                    continue

        conn.commit()
    except Exception as e:
        print(f"Error processing file pair {processed_file}: {e}")

async def main():
    """
    Main function to process data files asynchronously.
    """
    await load_domain_scores()
    shorturl_services = load_short_url_domains("shorturl-services-list.csv")
    destination_path = re.sub(r'(?<=/)\d{4}-\d{2}-\d{2}(?=/)', 'processed_reposts_and_likes', sys.argv[1])
    destination_path = re.sub(r'(?<=/)([^/]+\.json\.gz)$', r'processed_\1', destination_path)
    await process_file_pair(destination_path, shorturl_services)
if __name__ == "__main__":
    asyncio.run(main())
