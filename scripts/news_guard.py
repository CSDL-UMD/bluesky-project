import pandas as pd
import numpy as np
import aiohttp
import asyncio
from urllib.parse import urlsplit
from urlextract import URLExtract
import csv
import json
import os
import re
import warnings
from tqdm import tqdm

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
        print(f'Error in initialize_data: {err}')
        return None

def extract_domain(url):
    try:
        pattern = r"(https?://)?(www\d?\.)?(?P<domain>[\w\.-]+\.\w+)(/\S*)?"
        match = re.match(pattern, url)
        if match:
            return match.group("domain")
        else:
            return None
    except Exception:
        return None

async def get_profile(did, session):
    url = f"https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile?actor={did}"
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
            handle = data.get('handle')
            return handle
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return None
    except Exception as e:
        print(f"Unhandled exception fetching profile for {did}: {e}")
        return None

def modify_data(source, score, df):
    try:
        filtered_df = df[df['Label'] == source]
        if not filtered_df.empty:
            source_id = filtered_df['Id'].iloc[0]
            if pd.notna(source_id):
                index = df.index[df['Id'] == source_id].tolist()

                if index:
                    index = index[0]
                    num_trustworthy = df.at[index, 'Number of Trustworthy Posts']
                    num_untrustworthy = df.at[index, 'Number of Untrustworthy Posts']

                    if score >= 60:
                        num_trustworthy += 1
                    else:
                        num_untrustworthy += 1

                    total_posts = num_trustworthy + num_untrustworthy
                    trustworthiness_score = num_trustworthy / total_posts if total_posts != 0 else 0
                    df.at[index, 'Number of Trustworthy Posts'] = num_trustworthy
                    df.at[index, 'Number of Untrustworthy Posts'] = num_untrustworthy
                    df.at[index, 'Trustworthiness Score'] = trustworthiness_score
    except Exception as e:
        print(f"Error modifying data for {source}: {e}")

def check_source_in_csv(df, source):
    try:
        return source in df['Label'].values
    except Exception as e:
        print(f"Error checking source in CSV: {e}")
        return False

def get_node(df, node_label):
    try:
        row = df[df['Label'] == node_label]
        if not row.empty:
            return row.iloc[0]['Id']
        else:
            return None
    except Exception as e:
        print(f"Error getting node for {node_label}: {e}")
        return None

async def fetch(url, session):
    try:
        async with session.head(url, allow_redirects=False) as response:
            if response.status in (301, 302):
                if 'Location' in response.headers:
                    next_url = response.headers['Location']
                    async with session.head(next_url, allow_redirects=False) as next_response:
                        if next_response.status in (301, 302):
                            if 'Location' in next_response.headers:
                                return next_response.headers['Location']
                    return next_url
            return str(response.url)
    except aiohttp.ClientError as e:
        print(f"Aiohttp ClientError fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Unhandled exception fetching URL {url}: {e}")
        return None

async def unshorten_urls_from_buffer(url_record_map, url_buffer, session):
    try:
        tasks = []
        shorturl_services_df = pd.read_csv('./Gephi_Like/shorturl-services-list.csv')
        elements = set(shorturl_services_df.iloc[:, 0])
        semaphore = asyncio.Semaphore(200)
        async with semaphore:
            for url in url_buffer:
                if urlsplit(url).netloc in elements:
                    task = fetch(url, session)
                    tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for url, result in zip(url_buffer, results):
                if isinstance(result, str):
                    url_record_map[url] = {"newUrl": result, **url_record_map.get(url, {})}
                else:
                    url_record_map[url] = {"newUrl": url, **url_record_map.get(url, {})}

            url_record_df = pd.DataFrame(url_record_map.items(), columns=['URL', 'Record'])
            return url_record_df
    except Exception as e:
        return pd.DataFrame(columns=['URL', 'Record'])
    
async def process_batch(url_record_df, like_edge_csv, repost_edge_csv, like_node_csv, repost_node_csv, source_score_map, session):
    like_edge_csv_data = []
    repost_edge_csv_data = []
    like_df = pd.read_csv(like_node_csv)
    repost_df = pd.read_csv(repost_node_csv)
    for index, row in url_record_df.iterrows():
        try:
            if 'newUrl' in row['Record']:
                url = row['Record']['newUrl']
            else:
                url = row["URL"]
                
            record_dict = row['Record']
            new_record_value = record_dict.get("newRecord")
            domain = extract_domain(url)
            score = source_score_map.get(domain, source_score_map.get(url, -1))

            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    score = -1

            if isinstance(score, (int, float)) and score >= 0:
                originalTimestamp = record_dict.get("originalcreatedAt")
                newTimestamp = record_dict.get("newcreatetime")

                if new_record_value == "app.bsky.feed.like":
                    source = record_dict.get("did")
                    target = record_dict.get("originaldid")
                    original_count = record_dict.get("originallikeCount") or 0
                    weight = original_count + 1
                    source_check = check_source_in_csv(like_df, source)
                    target_check = check_source_in_csv(like_df, target)
                    if not source_check:
                        new_row = pd.DataFrame({'Id': len(like_df), 'Label': source, 'Number of Trustworthy Posts': 0, 'Number of Untrustworthy Posts': 0, 'Trustworthiness Score': np.nan}, index=[len(like_df)])
                        like_df = pd.concat([like_df, new_row], ignore_index=True)
                    if not target_check:
                        new_row = pd.DataFrame({'Id': len(like_df), 'Label': target, 'Number of Trustworthy Posts': 0, 'Number of Untrustworthy Posts': 0, 'Trustworthiness Score': np.nan}, index=[len(like_df)])
                        like_df = pd.concat([like_df, new_row], ignore_index=True)
                    modify_data(source, score, like_df)
                    corresponding_source = get_node(like_df, source)
                    corresponding_target = get_node(like_df, target)
                    like_edge_csv_data.append([corresponding_source, corresponding_target, weight, originalTimestamp, newTimestamp])

                elif new_record_value == "app.bsky.feed.repost":
                    source = record_dict.get("originaldid")
                    target = record_dict.get("did")
                    original_count = record_dict.get("originalrepostCount") or 0
                    weight = original_count + 1
                    source_check = check_source_in_csv(repost_df, source)
                    target_check = check_source_in_csv(repost_df, target)
                    if not source_check:
                        new_row = pd.DataFrame({'Id': len(repost_df), 'Label': source, 'Number of Trustworthy Posts': 0, 'Number of Untrustworthy Posts': 0, 'Trustworthiness Score': np.nan}, index=[len(repost_df)])
                        repost_df = pd.concat([repost_df, new_row], ignore_index=True)
                    if not target_check:
                        new_row = pd.DataFrame({'Id': len(repost_df), 'Label': target, 'Number of Trustworthy Posts': 0, 'Number of Untrustworthy Posts': 0, 'Trustworthiness Score': np.nan}, index=[len(repost_df)])
                        repost_df = pd.concat([repost_df, new_row], ignore_index=True)
                    modify_data(source, score, repost_df)
                    corresponding_source = get_node(repost_df, source)
                    corresponding_target = get_node(repost_df, target)
                    repost_edge_csv_data.append([corresponding_source, corresponding_target, weight, originalTimestamp, newTimestamp])

        except Exception as e:
            print(f"Error processing record: {e}")

    like_df.to_csv(like_node_csv, index=False)
    repost_df.to_csv(repost_node_csv, index=False)

    with open(like_edge_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(like_edge_csv_data)

    with open(repost_edge_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(repost_edge_csv_data)

async def main():
    like_node_csv = "./Gephi_Like/like_node.csv"
    repost_node_csv = "./Gephi_Repost/repost_node.csv"
    like_edge_csv = "./Gephi_Like/like_edge.csv"
    repost_edge_csv = "./Gephi_Repost/repost_edge.csv"

    BATCH_SIZE = 1000
    extractor = URLExtract()
    source_score_map = initialize_data()
    input_files = ['./Network/processed_2024-06-07.txt', './Network/2024-06-08.txt','./Network/2024-06-09.txt', './Network/2024-06-10.txt', './Network/2024-06-11.txt', './Network/2024-06-12.txt', './Network/2024-06-13.txt', './Network/2024-06-14.txt', './Network/2024-06-18.txt', './Network/2024-06-20.txt', './Network/2024-06-21.txt', './Network/2024-06-23.txt', './Network/2024-06-24.txt', './Network/2024-06-26.txt', './Network/2024-06-27.txt', './Messages/2024-06-28.txt', './Messages/2024-06-29.txt', './Messages/2024-07-01.txt', './Messages/2024-07-30.txt']
    total_lines = sum(1 for input_file_path in input_files for line in open(input_file_path, 'r'))
    print(f"Processing {total_lines} records...")

    url_buffer = set()
    url_record_map = {}

    async with aiohttp.ClientSession() as session:
        for input_file_path in input_files:
            if not os.path.exists(input_file_path):
                continue

            with open(input_file_path, 'r') as infile:
                for line in tqdm(infile, total=total_lines, desc=f"Processing {os.path.basename(input_file_path)}"):
                    record = json.loads(line.strip())
                    if 'newRecord' not in record:
                        continue
                    urls = extractor.find_urls(record.get('text'))
                    if urls:
                        url_record_map.update({url: record for url in urls})
                        url_buffer.update(urls)

                    if len(url_buffer) >= BATCH_SIZE:
                        url_record_df = await unshorten_urls_from_buffer(url_record_map, url_buffer, session)
                        await process_batch(url_record_df, like_edge_csv, repost_edge_csv, like_node_csv, repost_node_csv, source_score_map, session)
                        url_record_df.drop(url_record_df.index, inplace=True)
                        url_record_map.clear()
                        url_buffer.clear()

                if url_buffer:
                    url_record_df = await unshorten_urls_from_buffer(url_record_map, url_buffer, session)
                    await process_batch(url_record_df, like_edge_csv, repost_edge_csv, like_node_csv, repost_node_csv, source_score_map, session)
                    url_record_df.drop(url_record_df.index,inplace=True)
                    url_record_map.clear()
                    url_buffer.clear()

    print("Processing completed successfully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Process interrupted by user.")
