import asyncio
import aio_pika
import os
import gzip
import logging
import shutil
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://localhost/")
PAYLOAD_FOLDER = "./Payload"
TEMP_FOLDER = "./TempPayload"
BATCH_SIZE = 5000

async def consume_and_archive():
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(PAYLOAD_FOLDER, exist_ok=True)
    
    temp_file_path = os.path.join(TEMP_FOLDER, "building_hour.json.gz")    
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=BATCH_SIZE)
        queue = await channel.declare_queue("raw_events", durable=True)

        batch = []
        current_hour = datetime.now().hour
        
        logging.info("Listening to RabbitMQ.")
        
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                batch.append(message)
                now = datetime.now()
                
                if len(batch) >= BATCH_SIZE or now.hour != current_hour:
                    with gzip.open(temp_file_path, 'at', encoding='utf-8') as f:
                        for msg in batch:
                            f.write(msg.body.decode('utf-8') + '\n')
                    
                    for msg in batch:
                        await msg.ack()
                    
                    batch.clear()
                    
                    if now.hour != current_hour:
                        date_str = now.strftime("%Y-%m-%d")
                        final_folder = os.path.join(PAYLOAD_FOLDER, date_str)
                        os.makedirs(final_folder, exist_ok=True)
                        
                        time_str = now.strftime("%Y-%m-%d_%H-00-00")
                        final_file_path = os.path.join(final_folder, f"{time_str}.json.gz")
                        
                        if os.path.exists(temp_file_path):
                            shutil.move(temp_file_path, final_file_path)
                            logging.info(f"[*] Hour complete! Dropped {final_file_path} for Orchestrator.")
                        
                        current_hour = now.hour

if __name__ == "__main__":
    asyncio.run(consume_and_archive())