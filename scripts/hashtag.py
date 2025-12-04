import pandas as pd
import numpy as np
import aiohttp
import asyncio
import requests
from urllib.parse import urlsplit
from urlextract import URLExtract
import csv
import json
import os
import re
import warnings
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
like_edge_weights = {}
repost_edge_weights = {}
warnings.simplefilter(action='ignore', category=FutureWarning)

def initialize_data():
    try:
        metadata_data = pd.read_csv('NewsGuard/metadata.csv', on_bad_lines="skip")
        all_sources_data = pd.read_csv('NewsGuard/all-sources-metadata.csv', on_bad_lines="skip")

        desired_keys = ['Domain', 'Score']
        desired_keys2 = ['Source', 'Domain', 'Score']

        parsed_metadata_filtered = [{key: entry[key] for key in desired_keys if key in entry} for entry in metadata_data.to_dict(orient='records')]
        parsed_all_sources_metadata_filtered = [{key: entry[key] for key in desired_keys2 if key in entry} for entry in all_sources_data.to_dict(orient='records')]

        if not parsed_metadata_filtered or not parsed_all_sources_metadata_filtered:
            raise ValueError('CSV Parsing resulted in undefined data')

        source_score_map = {}
        for row in parsed_all_sources_metadata_filtered:
            if 'Source' in row and 'Score' in row:
                source_score_map[row['Source']] = row['Score']
            if 'Domain' in row and 'Score' in row:
                source_score_map[row['Domain']] = row['Score']

        for row in parsed_metadata_filtered:
            if 'Domain' in row and 'Score' in row and row['Domain'] not in source_score_map:
                source_score_map[row['Domain']] = row['Score']

        return source_score_map

    except Exception as err:
        logging.error(f'Error in initialize_data: {err}')
        return None

def extract_domain(url):
    try:
        pattern = r"(https?://)?(www\d?\.)?(?P<domain>[\w\.-]+\.\w+)(/\S*)?"
        match = re.match(pattern, url)
        if match:
            return match.group("domain")
        else:
            return None
    except Exception as e:
        logging.error(f"Error in extract_domain: {e}")
        return None

async def fetch(url, session):
    try:
        async with session.head(url, allow_redirects=True, timeout=10) as response:
            return str(response.url)
    except asyncio.TimeoutError:
        logging.warning(f"Timeout fetching URL {url}")
        return None
    except aiohttp.ClientError as e:
        logging.error(f"Aiohttp ClientError fetching URL {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unhandled exception fetching URL {url}: {e}")
        return None

async def unshorten_urls(urls, url_record_map, session):
    try:
        shorturl_services_df = pd.read_csv('./Hashtag_Like/shorturl-services-list.csv')
        elements = set(shorturl_services_df.iloc[:, 0])
        semaphore = asyncio.Semaphore(50)  # Limit concurrent requests
        
        async def process_url(url):
            async with semaphore:
                if urlsplit(url).netloc in elements:
                    result = await fetch(url, session)
                    return url, result if result else url
                return url, url

        tasks = [process_url(url) for url in urls]
        results = await asyncio.gather(*tasks)

        for url, result in results:
            url_record_map[url] = {"newUrl": result, **url_record_map.get(url, {})}

        return pd.DataFrame(url_record_map.items(), columns=['URL', 'Record'])
    except Exception as e:
        return pd.DataFrame(columns=['URL', 'Record'])

def modify_data(source, df):
    try:
        filtered_df = df[df['Label'] == source]
        
        if not filtered_df.empty:
            index = filtered_df.index[0]            
            current_weight = df.at[index, 'Weight']
            df.at[index, 'Weight'] = current_weight + 1
        else:
            logging.warning(f"Source {source} not found in DataFrame.")
    except Exception as e:
        logging.error(f"Error modifying data for {source}: {e}")

async def get_profile(did, session):
    url = f"https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile?actor={did}"
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            data = await response.json()
            handle = data.get('handle')
            return handle
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.warning(f"Error fetching profile for {did}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unhandled exception fetching profile for {did}: {e}")
        return None

def check_source_in_csv(df, source):
    return source in df['Label'].values

def get_node(df, node_label):
    try:
        row = df[df['Label'] == node_label]
        if not row.empty:
            return row.iloc[0]['Id']
        else:
            return None
    except Exception as e:
        logging.error(f"Error getting node for {node_label}: {e}")
        return None

def unshorten_url(short_url):
    try:
        with requests.head(short_url, allow_redirects=True, timeout=5) as response:
            return response.url
    except:
        return short_url

async def process_batch(overall_hashtag, like_edge_csv, repost_edge_csv, like_node_csv, repost_node_csv, source_score_map, session):
    like_df = pd.read_csv(like_node_csv)
    repost_df = pd.read_csv(repost_node_csv)
    
    like_edge_weights = {}
    repost_edge_weights = {}
    
    def load_edge_weights(file_path, weight_dict):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    source, target, weight = int(row[0]), int(row[1]), float(row[2])
                    weight_dict[frozenset([source, target])] = weight

    load_edge_weights(like_edge_csv, like_edge_weights)
    load_edge_weights(repost_edge_csv, repost_edge_weights)

    for hashtag_df in overall_hashtag:
        for source, targets in hashtag_df.items():
            if isinstance(targets[0][1], (int, float)) and targets[0][1] >= 60:
                target_df, target_edge_weights = like_df, like_edge_weights
            elif isinstance(targets[0][1], (int, float)) and targets[0][1] >= 0:
                target_df, target_edge_weights = repost_df, repost_edge_weights
            else:
                continue
            if not check_source_in_csv(target_df, source):
                new_row = pd.DataFrame({'Id': [len(target_df)], 'Label': [source]})
                target_df = pd.concat([target_df, new_row], ignore_index=True).drop_duplicates(subset='Label')
            
            processed_edges = set()
            for target in targets:
                if not check_source_in_csv(target_df, target[0]):
                    new_row = pd.DataFrame({'Id': [len(target_df)], 'Label': [target[0]]})
                    target_df = pd.concat([target_df, new_row], ignore_index=True).drop_duplicates(subset='Label')
                
                corresponding_source = get_node(target_df, source)
                corresponding_target = get_node(target_df, target[0])
                
                if corresponding_source is not None and corresponding_target is not None:
                    edge = frozenset([corresponding_source, corresponding_target])                    
                    if edge not in processed_edges:
                        target_edge_weights[edge] = target_edge_weights.get(edge, 0) + 0.5                        
                        processed_edges.add(edge)

            if targets[0][1] >= 60:
                like_df = target_df
            else:
                repost_df = target_df

    def write_edge_weights(file_path, weight_dict):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Source', 'Target', 'Weight', 'Original Timestamp', 'New Timestamp'])
            for edge, weight in weight_dict.items():
                source, target = tuple(edge)
                writer.writerow([source, target, weight])

    write_edge_weights(like_edge_csv, like_edge_weights)
    write_edge_weights(repost_edge_csv, repost_edge_weights)
    
    like_df.to_csv(like_node_csv, index=False)
    repost_df.to_csv(repost_node_csv, index=False)

async def main():
    like_node_csv = "./Hashtag_Like/trustworthy_node.csv"
    repost_node_csv = "./Hashtag_Repost/untrustworthy_node.csv"
    like_edge_csv = "./Hashtag_Like/trustworthy_edge.csv"
    repost_edge_csv = "./Hashtag_Repost/untrustworthy_edge.csv"

    source_score_map = initialize_data()
    if source_score_map is None:
        logging.error("Failed to initialize source score map.")
        return

    input_files = ['./Network/processed_2024-06-07.txt', './Network/2024-06-08.txt','./Network/2024-06-09.txt', './Network/2024-06-10.txt', './Network/2024-06-11.txt', './Network/2024-06-12.txt', './Network/2024-06-13.txt', './Network/2024-06-14.txt', './Network/2024-06-18.txt', './Network/2024-06-20.txt', './Network/2024-06-21.txt', './Network/2024-06-23.txt', './Network/2024-06-24.txt', './Network/2024-06-26.txt', './Network/2024-06-27.txt', './Messages/2024-06-28.txt', './Messages/2024-06-29.txt', './Messages/2024-07-01.txt', './Messages/2024-07-30.txt']
    total_lines = sum(1 for input_file_path in input_files for line in open(input_file_path, 'r'))
    logging.info(f"Processing {total_lines} records...")
    extractor = URLExtract()

    BATCH_SIZE = 100
    url_buffer = set()
    url_record_map = {}
    overall_hash = []
    result_arr = []
    
    async with aiohttp.ClientSession() as session:
        for input_file_path in input_files:
            if not os.path.exists(input_file_path):
                logging.warning(f"File not found: {input_file_path}")
                continue

            with open(input_file_path, 'r') as infile:
                for line in tqdm(infile, total=total_lines, desc=f"Processing {os.path.basename(input_file_path)}"):
                    try:
                        await asyncio.wait_for(process_line(line, extractor, source_score_map, session, url_buffer, url_record_map, overall_hash, result_arr), timeout=30)
                    except asyncio.TimeoutError:
                        logging.warning(f"Timeout processing line: {line[:50]}...")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error in line: {e}")
                    except Exception as e:
                        logging.error(f"Unexpected error processing line: {e}")
                    
                    if len(url_buffer) >= BATCH_SIZE:
                        try:
                            for item in overall_hash:
                                if item not in result_arr:
                                    result_arr.append(item)
                            await process_batch(result_arr, like_edge_csv, repost_edge_csv, like_node_csv, repost_node_csv, source_score_map, session)
                        except Exception as e:
                            continue
                        finally:
                            url_record_map.clear()
                            url_buffer.clear()
                            overall_hash.clear()
                            result_arr.clear()

    logging.info("Processing completed successfully.")

async def process_line(line, extractor, source_score_map, session, url_buffer, url_record_map, overall_hash, result_arr):
    try:
        record = json.loads(line.strip())
        if 'newRecord' not in record:
            return
        
        urls = extractor.find_urls(record.get('text', ''))
        if urls:
            url_record_map.update({url: record for url in urls})
            url_buffer.update(urls)
            for url in urls:
                long_url = unshorten_url(url)
                domain = extract_domain(long_url) or url
                score = source_score_map.get(domain, source_score_map.get(url, -1))
                hashtag_df = {}
                if score >= 0:
                    hashtags = re.findall(r'#\w+', record.get('text', ''))
                    for i, hashtag in enumerate(hashtags):
                        for j, otherhash in enumerate(hashtags):
                            if i != j:
                                if hashtag not in hashtag_df:
                                    hashtag_df[hashtag] = []
                                hashtag_df[hashtag].append([otherhash, score, record.get("newcreatetime")])
                if hashtag_df:
                    overall_hash.append(hashtag_df)
    except ValueError as e:
        logging.error(f"ValueError in process_line: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in process_line: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")
