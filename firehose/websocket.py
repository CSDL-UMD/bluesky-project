import json, asyncio, websockets, os, logging, gzip, time
from datetime import datetime
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
current_day_folder = None

def ensure_daily_folder():
    """
    Ensures that a folder for the current day exists under the './Payload' directory.
    If the folder for today doesn't exist, it creates one.

    Returns:
        str: The path of the current day's folder.
    """
    global current_day_folder
    today = datetime.now().strftime('%Y-%m-%d')
    new_folder_path = f"./Payload/{today}"
    if current_day_folder != new_folder_path:
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        current_day_folder = new_folder_path
    return current_day_folder

async def connect_and_collect(queue, url="wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post&wantedCollections=app.bsky.feed.like&wantedCollections=app.bsky.feed.repost"):
    """
    Connects to a WebSocket server, subscribes to post, repost, and like collections, and collects messages.
    Puts the collected messages into the provided queue.

    Parameters:
        queue (asyncio.Queue): The queue to store the collected messages.
        url (str, optional): The WebSocket URL to connect to. Defaults to the provided default URL.
    """
    while True:
        try:
            async with websockets.connect(url) as websocket:
                while True:
                    try:
                        message = await websocket.recv()
                        try:
                            message_json = json.loads(message)
                            await queue.put(message_json)
                        except json.JSONDecodeError:
                            logging.warning("Failed to decode message JSON")
                    except websockets.ConnectionClosed:
                        logging.warning("WebSocket connection closed")
                        break
        except Exception as e:
            logging.error(f"WebSocket error: {e}")

async def compress_periodically(queue, interval=3600):
    """
    Compresses the messages collected in the queue and writes them to a `.json.gz` file every hour.

    Parameters:
        queue (asyncio.Queue): The queue to collect messages from.
        interval (int, optional): The interval in seconds to wait before compressing. Defaults to 1 hour.
    """
    while True:      
        buffer = deque()
        start_time = time.time()

        while (time.time() - start_time) < interval:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=interval - (time.time() - start_time))
                buffer.append(message)
            except asyncio.TimeoutError:
                continue

        if buffer:
            folder_path = ensure_daily_folder()
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{timestamp}.json.gz"
            full_path = os.path.join(folder_path, filename)  
            try:
               with gzip.open(full_path, 'wt', encoding='utf-8') as f:
                for obj in buffer:
                    obj_json = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
                    f.write(obj_json + '\n')
            except Exception as e:
                logging.error(f"Compression error: {e}")
        while not queue.empty():
            queue.get_nowait()

async def main():
    """
    Initializes an asyncio queue and runs the connection and compression tasks.
    """
    queue = asyncio.Queue()
    await asyncio.gather(connect_and_collect(queue), compress_periodically(queue))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script terminated by user")

