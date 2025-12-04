import os
import re
from typing import Optional
from collections import Counter
from datetime import datetime
import psycopg2
import pandas as pd
from math import isnan
from dotenv import load_dotenv
load_dotenv()
MAX_URL_LENGTH = 8191

domain_scores = {}

def load_domain_scores():
    """
    Loads domain trust scores from metadata CSV files (NewsGuard).
    This function reads two CSV files, 'metadata.csv' and 'all-sources-metadata.csv', 
    and updates the global `domain_scores` dictionary with the domain name as the key 
    and its corresponding trust score as the value.
    """
    try:
        metadata = pd.read_csv('./NewsGuard/metadata.csv', on_bad_lines='skip')
        all_sources = pd.read_csv('./NewsGuard/all-sources-metadata.csv', on_bad_lines='skip')

        for df in [all_sources, metadata]:
            for row in df.to_dict(orient='records'):
                domain_scores[row['Domain'].lower()] = row['Score']

    except Exception as e:
        print(f"Error loading domains: {e}")


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

directory = './Payload/Urls'

def is_valid_url(url: str) -> bool:
    return len(url) <= MAX_URL_LENGTH

load_domain_scores()
if not domain_scores:
    print("Domain scores are empty. Please check the CSV files.")
    exit(1)

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        match = re.match(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.txt', filename)
        if match:
            file_timestamp = match.group(1) + " " + match.group(2).replace('-', ':')
            current_datetime = datetime.strptime(file_timestamp, "%Y-%m-%d %H:%M:%S")

            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                urls = file.readlines()

            urls = [url.strip() for url in urls]
            url_counts = Counter(urls)
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
                        print(f"Error inserting URL {url}: {e}")
                        continue

            conn.commit()

cursor.close()
conn.close()

print("Link counts have been successfully updated.")

