import os
import time
import subprocess
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor
import asyncio

class FileEventHandler(FileSystemEventHandler):
    """
    Handles file system events such as file creation in a watched directory.
    """
    def __init__(self, queue, event_loop, starting_file=None):
        """
        Initializes the FileEventHandler with necessary parameters.
        """
        self.queue = queue
        self.event_loop = event_loop
        self.preprocess_lock = threading.Lock()
        self.starting_file = starting_file

    def on_created(self, event):
        """
        Handles file creation events. If a new .gz file is created and meets the criteria, it is added to the processing queue.
        """
        if event.is_directory:
            return
        if event.src_path.endswith('.gz') and not os.path.abspath(event.src_path).startswith(os.path.abspath('./Payload/processed_reposts_and_likes')):
            if self.wait_for_file(event.src_path):
                asyncio.run_coroutine_threadsafe(self.queue.put(event.src_path), self.event_loop)

    def wait_for_file(self, file_path, max_wait=600, check_interval=300):
        """
        Waits for the file to finish writing by checking its size over a period of time.
        """
        last_size = -1
        elapsed_time = 0
        while elapsed_time < max_wait:
            time.sleep(check_interval)
            elapsed_time += check_interval
            try:
                current_size = os.path.getsize(file_path)
                if current_size == last_size and current_size > 0:
                    return True
                last_size = current_size
            except FileNotFoundError:
                pass
        return False

def process_file(file_path):
    """
    Processes the given file by running an external script on it.
    """
    try:
        print(f"Running Data Processing: {file_path}")
        subprocess.run(['python3', 'data_processing.py', file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error with file {file_path}: {e}")

def run_preprocess(file_path, preprocess_lock):
    """
    Runs the preprocess script for the given file while ensuring thread safety using a lock.
    """
    with preprocess_lock:
        print(f"Running preprocess for: {file_path}")
        subprocess.run(['python3', 'preprocess.py', file_path], check=True)

def start_websocket():
    """
    Starts the websocket subprocess to connect and receive data.
    """
    print("Starting websocket.py subprocess")
    return subprocess.Popen(['python3', 'websocket.py'])

def get_initial_file_list(directory, starting_file=None):
    """
    Retrieves the list of .gz files from the specified directory that are not already processed.
    """
    files = [
        os.path.join(root, file)
        for root, dirs, files_in_dir in os.walk(directory)
        for file in files_in_dir
        if file.endswith('.gz') and not os.path.commonpath([os.path.abspath(os.path.join(root, file)), os.path.abspath('./Payload/processed_reposts_and_likes')]) == os.path.abspath('./Payload/processed_reposts_and_likes')
    ]
    files.sort()
    if starting_file and starting_file in files:
        return files[files.index(starting_file):]
    return files if starting_file else []

async def orchestrate(starting_file=None, watch_directory='./Payload'):
    """
    Orchestrates the file processing by watching a directory for new files and processing them
    using threads and subprocesses. Also manages WebSocket communication for live data collection.
    """
    file_queue = asyncio.Queue()
    event_loop = asyncio.get_event_loop()
    event_handler = FileEventHandler(file_queue, event_loop, starting_file=starting_file)
    observer = Observer()
    observer.schedule(event_handler, path=watch_directory, recursive=True)
    observer.start()

    websocket_process = start_websocket()

    with ThreadPoolExecutor(max_workers=4) as executor:
        try:
            initial_files = get_initial_file_list(watch_directory, starting_file)
            for file_path in initial_files:
                run_preprocess(file_path, event_handler.preprocess_lock)
                executor.submit(process_file, file_path)

            while True:
                file_path = await file_queue.get()
                if file_path:
                    run_preprocess(file_path, event_handler.preprocess_lock)
                    executor.submit(process_file, file_path)
        except KeyboardInterrupt:
            print("Orchestrator interrupted by user")
        finally:
            observer.stop()
            observer.join()

            if websocket_process.poll() is None:
                print("Stopping websocket.py subprocess")
                websocket_process.terminate()
                websocket_process.wait()

async def main():
    """
    Main entry point to start the orchestrating process with an optional starting file.
    """
    await orchestrate(starting_file="./Payload/2025-12-04/2025-12-04_02-22-16.json.gz")

if __name__ == "__main__":
    asyncio.run(main())
