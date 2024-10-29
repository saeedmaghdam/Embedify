from os import environ
import time
import pika
import json
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

RABBITMQ_HOST = environ.get("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = environ.get("RABBITMQ_PORT", 5672)
RABBITMQ_USER = environ.get("RABBITMQ_USER", "guest")
RABBITMQ_PASS = environ.get("RABBITMQ_PASS", "guest")
RABBITMQ_VHOST = environ.get("RABBITMQ_VHOST", "/")

RABBITMQ_REQUESTS_EXCHANGE = "embeddings_requests"
RABBITMQ_REQUESTS_QUEUE = "embeddings_requests"

RABBITMQ_RESPONSES_EXCHANGE = "embeddings_responses"
RABBITMQ_CODE_RESPONSES_QUEUE = "code_embedding_responses"
RABBITMQ_QUERY_RESPONSES_QUEUE = "query_embedding_responses"

def connect_to_rabbitmq():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_VHOST, credentials))
    channel = connection.channel()

    channel.exchange_declare(exchange=RABBITMQ_REQUESTS_EXCHANGE, exchange_type="direct")
    channel.queue_declare(queue=RABBITMQ_REQUESTS_QUEUE)
    channel.queue_bind(exchange=RABBITMQ_REQUESTS_EXCHANGE, queue=RABBITMQ_REQUESTS_QUEUE, routing_key="")

    channel.exchange_declare(exchange=RABBITMQ_RESPONSES_EXCHANGE, exchange_type="direct")
    channel.queue_declare(queue=RABBITMQ_CODE_RESPONSES_QUEUE)
    channel.queue_declare(queue=RABBITMQ_QUERY_RESPONSES_QUEUE)
    channel.queue_bind(exchange=RABBITMQ_RESPONSES_EXCHANGE, queue=RABBITMQ_CODE_RESPONSES_QUEUE, routing_key="code")
    channel.queue_bind(exchange=RABBITMQ_RESPONSES_EXCHANGE, queue=RABBITMQ_QUERY_RESPONSES_QUEUE, routing_key="query")

    return channel

def generate_embedding(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

def on_message(channel, method, properties, body):
    message = json.loads(body)
    print(f"Received a message with requestId: {message.get('requestId')} and type: {message.get('type')} and length: {len(message.get('content'))}")

    content = message.get("content")
    requestType = message.get("type")
    if requestType != "code" and requestType != "query":
        print("Invalid request type")
        return

    startTime = time.time()
    embedding = generate_embedding(content)
    durationInMs = round((time.time() - startTime) * 1000, 2)
    result = {
        "requestId": message.get("requestId"),
        "embedding": embedding
    }

    channel.basic_publish(
        exchange=RABBITMQ_RESPONSES_EXCHANGE,
        routing_key=requestType,
        body=json.dumps(result)
    )

    print(f"Input embedified in {durationInMs} ms, published the embedding to the {RABBITMQ_RESPONSES_EXCHANGE} exchange with the routing key: {requestType}")

    channel.basic_ack(delivery_tag=method.delivery_tag)

def main():
    channel = connect_to_rabbitmq()
    print("Embedify started, waiting for messages...")
    channel.basic_consume(queue=RABBITMQ_REQUESTS_QUEUE, on_message_callback=on_message)
    channel.start_consuming()

if __name__ == "__main__":
    main()
