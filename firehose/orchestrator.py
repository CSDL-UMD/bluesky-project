import os
import time
import subprocess
import threading
import asyncio
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor

WATCH_DIR = './Payload'
PROCESSED_DIR = os.path.join(WATCH_DIR, 'processed_reposts_and_likes')
MAX_WORKERS = 4
PYTHON_EXE = sys.executable 

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, queue, loop):
        self.queue = queue
        self.loop = loop

    def on_created(self, event):
        if event.is_directory or not (event.src_path.endswith('.gz') or event.src_path.endswith('.json')):
            return
        
        if os.path.abspath(PROCESSED_DIR) in os.path.abspath(event.src_path):
            return

        print(f"[*] New file detected: {os.path.basename(event.src_path)}")
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event.src_path)

def wait_for_file_ready(file_path, timeout=600):
    last_size = -1
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            current_size = os.path.getsize(file_path)
            if current_size == last_size and current_size > 0:
                return True
            last_size = current_size
        except FileNotFoundError:
            pass
        time.sleep(5) 
    return False

def run_pipeline(file_path):
    try:
        if not wait_for_file_ready(file_path):
            print(f"[!] Timeout waiting for {file_path}")
            return

        print(f"[>] Preprocessing: {os.path.basename(file_path)}")
        subprocess.run([PYTHON_EXE, 'preprocess.py', file_path], check=True)

        print(f"[>] Scoring & DB Insert: {os.path.basename(file_path)}")
        subprocess.run([PYTHON_EXE, 'data_processing.py', file_path], check=True)
        
        print(f"[+] Successfully finished: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"[!!] Error on {file_path}: {e}")

async def monitor_websocket():
    while True:
        print("[*] Launching websocket.py...")
        process = await asyncio.create_subprocess_exec(PYTHON_EXE, 'websocket.py')
        await process.wait()
        print("[!] websocket.py exited. Restarting in 10s...")
        await asyncio.sleep(10)

async def monitor_archiver():
    while True:
        print("[*] Launching rabbit_archiver.py...")
        process = await asyncio.create_subprocess_exec(PYTHON_EXE, 'rabbit_archiver.py')
        await process.wait()
        print("[!] rabbit_archiver.py exited. Restarting in 10s...")
        await asyncio.sleep(10)

def get_initial_files(directory, start_from=None):
    all_files = []
    for root, _, files in os.walk(directory):
        if os.path.abspath(PROCESSED_DIR) in os.path.abspath(root):
            continue
        for f in files:
            if f.endswith('.gz') or f.endswith('.json'):
                all_files.append(os.path.join(root, f))
    
    all_files.sort()
    if start_from:
        try:
            idx = next(i for i, f in enumerate(all_files) if start_from in f)
            return all_files[idx:]
        except StopIteration:
            print(f"[!] Starting file containing '{start_from}' not found. Processing nothing.")
            return []
    return all_files

async def main():
    file_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    event_handler = FileEventHandler(file_queue, loop)
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=True)
    observer.start()

    asyncio.create_task(monitor_websocket())
    asyncio.create_task(monitor_archiver())

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        initial_files = get_initial_files(WATCH_DIR, start_from="2026-03-05_11-07-53")
        print(f"[*] Found {len(initial_files)} files to process.")
        
        for f in initial_files:
            file_queue.put_nowait(f)

        print("[*] Orchestrator is active. Waiting for new data...")
        try:
            while True:
                file_to_process = await file_queue.get()
                executor.submit(run_pipeline, file_to_process)
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()
            observer.join()

if __name__ == "__main__":
    asyncio.run(main())