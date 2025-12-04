import psycopg2
import pandas as pd
from typing import Literal
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi import APIRouter
import json, subprocess, gzip, io
from fastapi.responses import JSONResponse, StreamingResponse
import signal, sys, re, os
from typing import List, Dict, Any, AsyncGenerator, Tuple
import paramiko, traceback
import tempfile
import shutil, cchardet
import mmap, signal, socket
import orjson, subprocess
from typing import List, Dict, Any, Generator, Optional, Union
import ujson as json 
from io import BytesIO
import gzip
from functools import partial
import aiomultiprocess
from itertools import islice
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

MAX_WORKERS = 48
CHUNK_SIZE = 8388608  # 8MB
BATCH_SIZE = 1000
MAX_RETRIES = 3
RETRY_DELAY = 1 

SSH_POOL_SIZE = 5
ssh_semaphore = asyncio.Semaphore(SSH_POOL_SIZE)
_ssh_pool = []

stop_streaming = False
def handle_sigint(signum, frame):
    global stop_streaming
    stop_streaming = True
    print("SIGINT received. Stopping streaming...")
signal.signal(signal.SIGINT, handle_sigint)

db_config = {
    'host': 'colon.umd.edu',
    'user': 'bluesky_manager',
    'password': 'bluesky',
    'database': 'bluesky_db',
    'port': 5432
}
"""
% FastAPI Endpoint - Fetch Top 10 NewsGuard links for past 30 days / 7 days starting from specified date (Stats) 
All of Time-Series Data - 7 days, 30 days, All Data, Custom, boolean for relative or absolute (Time-Series) 
Post, Repost, Like Paylaods for specified periods of time
"""

router = APIRouter()
ALLOWED_COLUMNS = ["url", "domain"]
def fetch_data_from_database():
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query = "SELECT Day, TotalMessages, TotalLinks, NewsGreaterThan60, NewsLessThan60 FROM bsky_news"
        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()
        return results

    except Exception as e:
        print(f"Error fetching data from the database: {e}")
        return {'error': str(e)}

@router.get("/time_series")
async def time_series(request: Request):
    try:
        period = request.query_params.get('period')
        if not period:
            raise HTTPException(
                status_code=400,
                detail="Period parameter is required. Valid values: 'all', 'seven', 'thirty', 'custom'"
            )
        if period not in ['all', 'seven', 'thirty', 'custom']:
            raise HTTPException(
                status_code=400,
                detail="Invalid period. Valid values: 'all', 'seven', 'thirty', 'custom'"
            )

        relative = request.query_params.get('relative', 'false').lower() == 'true'
        granularity = request.query_params.get('granularity', 'hour')
        if granularity not in ['hour', 'day']:
            raise HTTPException(
                status_code=400,
                detail="Invalid granularity. Valid values: 'hour', 'day'"
            )

        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        
        data = fetch_data_from_database()
        data.sort(key=lambda x: pd.to_datetime(x[0]))
        
        df = pd.DataFrame(data, columns=['date', 'total_messages', 'total_links', 'news_greater_than_60', 'news_less_than_60'])
        df['date'] = pd.to_datetime(df['date'])
        
        current_time = datetime.now()
        
        if period == 'custom':
            if not start_date or not end_date:
                raise HTTPException(
                    status_code=400,
                    detail="Both start_date and end_date are required for custom period"
                )

            try:
                start_datetime = pd.to_datetime(start_date).replace(hour=0, minute=0, second=0)
                end_datetime = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)

                if start_datetime > end_datetime:
                    raise HTTPException(
                        status_code=400,
                        detail="start_date must be before or equal to end_date"
                    )

                df = df[(df['date'] >= start_datetime) & (df['date'] <= end_datetime)]

                if len(df) == 0:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No data found between {start_date} and {end_date}. Make sure that the start_date is after 2023-12-08."
                    )

            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Use YYYY-MM-DD"
                )

        elif period == 'seven':
            cutoff_date = current_time - timedelta(days=7)
            df = df[df['date'] >= cutoff_date]

        elif period == 'thirty':
            cutoff_date = current_time - timedelta(days=30)
            df = df[df['date'] >= cutoff_date]

        if granularity == 'day':
            if relative:
                df['relative_news'] = df['news_less_than_60'] / (df['news_less_than_60'] + df['news_greater_than_60'])
                df['relative_news'] = df['relative_news'].fillna(0)

                daily_stats = df.groupby(df['date'].dt.floor('D')).agg({
                    'relative_news': ['count', 'sum']
                }).reset_index()

                daily_stats.columns = ['date', 'count', 'sum']
                daily_stats['relative_news'] = daily_stats['sum'] / daily_stats['count']

                response_data = [
                    {
                        'date': date.strftime('%Y-%m-%d'),
                        'relative_news': float(rel_news)
                    }
                    for date, rel_news in zip(daily_stats['date'], daily_stats['relative_news'])
                ]
            else:
                daily_stats = df.groupby(df['date'].dt.floor('D')).agg({
                    'total_messages': 'sum',
                    'total_links': 'sum',
                    'news_greater_than_60': 'sum',
                    'news_less_than_60': 'sum'
                }).reset_index()
                response_data = [
                    {
                        'date': date.strftime('%Y-%m-%d'),
                        'total_messages': int(msg),
                        'total_links': int(links),
                        'news_greater_than_60': int(greater),
                        'news_less_than_60': int(less)
                    }
                    for date, msg, links, greater, less in zip(
                        daily_stats['date'],
                        daily_stats['total_messages'],
                        daily_stats['total_links'],
                        daily_stats['news_greater_than_60'],
                        daily_stats['news_less_than_60']
                    )
                ]
        else:
            df['date'] = df['date'].dt.floor('H')
            if relative:
                df['relative_news'] = df['news_less_than_60'] / (df['news_less_than_60'] + df['news_greater_than_60'])
                df['relative_news'] = df['relative_news'].fillna(0)

                response_data = [
                    {
                        'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                        'relative_news': float(rel_news)
                    }
                    for date, rel_news in zip(df['date'], df['relative_news'])
                ]
            else:
                response_data = [
                    {
                        'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                        'total_messages': int(msg),
                        'total_links': int(links),
                        'news_greater_than_60': int(greater),
                        'news_less_than_60': int(less)
                    }
                    for date, msg, links, greater, less in zip(
                        df['date'],
                        df['total_messages'],
                        df['total_links'],
                        df['news_greater_than_60'],
                        df['news_less_than_60']
                    )
                ]
 
        return JSONResponse(content=response_data)
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_urls_from_database(
    type: str,
    end_time: datetime,
    range: str,
    min_score: int,
    max_score: int
):
    try:
        if type not in ALLOWED_COLUMNS:
            raise ValueError(f"Invalid type: {type}. Allowed values: {ALLOWED_COLUMNS}")

        today = datetime.now().date()
        if end_time.date() == today:
            end_time = datetime.now()
        else:
            end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)

        if range == "week":
            start_time = end_time
            end_time = start_time + timedelta(weeks=1)  
        elif range == "month":
            start_time = end_time
            end_time = start_time + relativedelta(months=1)
        else:
            raise ValueError(f"Invalid range. Must be 'week' or 'month'.")

        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query = f"""
            SELECT {type}, SUM(count) AS total_count
            FROM newsguard_counts
            WHERE timestamp >= %s AND timestamp < %s
            AND score >= %s AND score <= %s
            GROUP BY {type}
            ORDER BY total_count DESC
            LIMIT 10;
        """

        cursor.execute(query, (start_time, end_time, min_score, max_score))
        url_results = cursor.fetchall()
        cursor.close()
        conn.close()

        if not url_results:
            raise ValueError(f"No data found between {start_time.date()} and {end_time.date()}. Ensure that the starting date is after 2024-11-28 and before {datetime.now().date()}.")

        return url_results

    except Exception as e:
        print(f"Error fetching data from the database: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching data from the database: {str(e)}")


@router.get("/stats")
async def fetch_urls(
    type: str = Query(..., description="Column to group by (e.g., url or domain)"),
    start_date: str = Query(..., description="Starting date and time in the format YYYY-MM-DD"),
    range: Literal["week", "month"] = Query(..., description="Range: 'week' or 'month'"),
    min_score: int = Query(0, ge=0, description="Minimum score threshold (>= 0)."),
    max_score: int = Query(100, le=100, description="Maximum score threshold (<= 100).")
):
    try:
        try:
            end_time_parsed = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        results = await fetch_urls_from_database(
            type=type,
            end_time=end_time_parsed,
            range=range,
            min_score=min_score,
            max_score=max_score
        )
        return {"results": results}
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from the database: {str(e)}")

class SSHStreamingError(Exception):
    pass

@asynccontextmanager
async def get_ssh_connection():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    async def cleanup():
        try:
            if ssh:
                ssh.close()
        except Exception as e:
            print(f"Error during SSH cleanup: {e}")

    try:
        ssh.connect(
            "carrot.umd.edu",
            username="vikas",
            password="bluesky",
            timeout=30,
            compress=True
        )
        yield ssh
    except Exception as e:
        await cleanup()
        raise SSHStreamingError(f"SSH connection error: {e}")
    finally:
        await cleanup()

async def list_remote_files(
    ssh: paramiko.SSHClient,
    type_filter: str,
    start_datetime: datetime,
    end_datetime: datetime
) -> List[str]:
    files_in_range = []
    try:
        if type_filter in ["like", "repost"]:
            command = "lxc exec bluesky -- ls /root/firehose/Payload/processed_reposts_and_likes/"
            stdin, stdout, stderr = ssh.exec_command(command)
            if stderr.read().decode("utf-8"):
                raise SSHStreamingError("Error listing files")

            files = stdout.read().decode("utf-8").splitlines()
            for filename in files:
                try:
                    date_parts = filename.split('_')[1:3]
                    file_hour = int(date_parts[1].split('-')[0])
                    file_datetime = datetime.strptime(f"{date_parts[0]} {file_hour:02d}", "%Y-%m-%d %H")
                    if start_datetime <= file_datetime < end_datetime:
                        files_in_range.append(filename)
                except (ValueError, IndexError):
                    continue
        else:  # type == "post"
            current_date = start_datetime.date()
            while current_date <= end_datetime.date():
                date_str = current_date.strftime("%Y-%m-%d")
                command = f"lxc exec bluesky -- ls /root/firehose/Payload/{date_str}"
                stdin, stdout, stderr = ssh.exec_command(command)
                if not stderr.read().decode("utf-8"):
                    files = stdout.read().decode("utf-8").splitlines()
                    for file in files:
                        try:
                            time_part = file.split('_')[1].split('.')[0]
                            file_hour, file_minute, file_second = map(int, time_part.split('-'))
                            file_datetime = datetime.strptime(
                                f"{date_str} {file_hour:02d}:{file_minute:02d}:{file_second:02d}",
                                "%Y-%m-%d %H:%M:%S"
                            )
                            if start_datetime <= file_datetime < end_datetime:
                                files_in_range.append(f"{date_str}/{file}")
                        except (ValueError, IndexError):
                            continue
                current_date += timedelta(days=1)

        return sorted(files_in_range)
    except Exception as e:
        raise SSHStreamingError(f"Error listing remote files: {e}")

async def process_files(start_datetime: datetime, end_datetime: datetime, type_filter: str) -> AsyncGenerator[List[Dict], None]:
    async with get_ssh_connection() as ssh:
        files = await list_remote_files(ssh, type_filter, start_datetime, end_datetime)
        if not files:
            raise HTTPException(
                status_code=404,
                detail=f"No files found between {start_datetime} and {end_datetime}."
            )

        for file in files:
            try:
                async for batch in process_file_streaming(ssh, start_datetime, end_datetime, file, type_filter):
                    yield batch
            except asyncio.CancelledError:
                print(f"Processing cancelled for file: {file}")
                raise
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

async def stream_json_response(records_generator: AsyncGenerator[List[Dict], None]) -> AsyncGenerator[str, None]:
    first = True
    yield '['

    try:
        async for batch in records_generator:
            if not first:
                yield ','
            else:
                first = False

            for i, record in enumerate(batch):
                if i > 0:
                    yield ','
                try:
                    yield json.dumps(record, ensure_ascii=False)
                except Exception as e:
                    print(f"Error serializing record: {e}")
                    continue
                await asyncio.sleep(0)

        yield ']'
    except asyncio.CancelledError:
        print("Streaming cancelled by client")
        raise
    except Exception as e:
        print(f"Error during streaming: {e}")
        raise

async def process_file_streaming(
    ssh: paramiko.SSHClient,
    start_datetime: datetime,
    end_datetime: datetime,
    filename: str,
    type_filter: str,
) -> AsyncGenerator[List[Dict[str, Any]], None]:
    buffer = []
    stdin = stdout = stderr = None

    try:
        base_path = "/root/firehose/Payload"
        path = f"{base_path}/processed_reposts_and_likes/{filename}" if type_filter in ["repost", "like"] else f"{base_path}/{filename}"
        tail_cmd = "| tail -n +2" if type_filter in ["repost", "like"] else ""
        command = f'lxc exec bluesky -- /bin/bash -c "dd if={path} bs={CHUNK_SIZE} 2>/dev/null | zcat {tail_cmd}"'

        stdin, stdout, stderr = ssh.exec_command(command)
        stdout._set_mode('rb')

        while True:
            try:
                line = stdout.readline()
                if not line:
                    break

                await asyncio.sleep(0)

                decoded_line = line.decode('utf-8', errors='replace').strip()
                if not decoded_line:
                    continue

                record = json.loads(decoded_line)
                if await process_record(record, type_filter, start_datetime, end_datetime):
                    buffer.append(record)
                    if len(buffer) >= BATCH_SIZE:
                        yield buffer
                        buffer = []

            except (paramiko.SSHException, socket.error) as e:
                print(f"SSH/Socket error during streaming: {e}")
                raise SSHStreamingError(f"Connection error: {e}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing line: {e}")
                continue

        if buffer:
            yield buffer

    except asyncio.CancelledError:
        print("Processing cancelled by client")
        raise
    except Exception as e:
        print(f"Error during file processing: {e}")
        raise
    finally:
        # Clean up resources
        for resource in (stdout, stdin, stderr):
            if resource:
                try:
                    resource.close()
                except Exception as e:
                    print(f"Error closing SSH stream: {e}")

async def process_record(record: Dict, type_filter: str, start_datetime: datetime, end_datetime: datetime) -> bool:
    """Process a single record and return True if it should be included in the output."""
    try:
        if type_filter in ["repost", "like"]:
            record_datetime_str = record.get("newCreatedAt")
            record_type = record.get("newType")
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
        else:
            commit_record = record.get('commit', {}).get('record', {})
            record_datetime_str = commit_record.get('createdAt')
            record_type = commit_record.get('$type')
            fmt = "%Y-%m-%dT%H:%M:%S.%fZ"

        if not record_datetime_str:
            return False

        record_datetime = datetime.strptime(record_datetime_str, fmt)
        return (start_datetime <= record_datetime <= end_datetime and
                record_type == f"app.bsky.feed.{type_filter}")
    except ValueError:
        return False

@router.get("/payload")
async def get_file(
    start_date: str = Query(..., description="Starting date in the format YYYY-MM-DD"),
    start_hour: str = Query(..., description="Hour in 24-hour clock format"),
    end_date: Optional[str] = Query(None, description="End date in the format YYYY-MM-DD (optional)"),
    end_hour: Optional[str] = Query(None, description="End hour in 24-hour clock format (optional)"),
    type_filter: str = Query(..., description="Payload type (post, repost, or like)", alias="type")
):
    try:
        if not (start_hour.isdigit() and 0 <= int(start_hour) <= 23):
            raise ValueError("Invalid start hour format")

        start_datetime = datetime.strptime(f"{start_date} {start_hour}", "%Y-%m-%d %H")
        end_datetime = (datetime.strptime(f"{end_date} {end_hour}", "%Y-%m-%d %H") + timedelta(hours=1)
                       if end_date and end_hour else start_datetime + timedelta(hours=2))

        if start_datetime >= end_datetime:
            raise HTTPException(status_code=400, detail="Start datetime must be earlier than end datetime.")

        return StreamingResponse(
            stream_json_response(process_files(start_datetime, end_datetime, type_filter)),
            media_type="application/json",
            headers={"X-Accel-Buffering": "no"} 
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except SSHStreamingError as se:
        raise HTTPException(status_code=503, detail=str(se))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


