import asyncio
import websockets
import os
import logging
import aio_pika

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://localhost/")
BSKY_URL = "wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post&wantedCollections=app.bsky.feed.like&wantedCollections=app.bsky.feed.repost"

async def connect_and_collect():
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue("raw_events", durable=True)
        exchange = channel.default_exchange

        while True:
            try:
                async with websockets.connect(BSKY_URL) as websocket:
                    logging.info("Connected to Bluesky Jetstream")
                    while True:
                        try:
                            message = await websocket.recv()
                            await exchange.publish(
                                aio_pika.Message(
                                    body=message.encode(),
                                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                                ),
                                routing_key="raw_events"
                            )
                        except websockets.ConnectionClosed:
                            logging.warning("WebSocket connection closed, reconnecting...")
                            break
            except Exception as e:
                logging.error(f"Connection error: {e}. Retrying in 5s...")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(connect_and_collect())