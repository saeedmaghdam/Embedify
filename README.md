# Embedify

Here’s a `README.md` file for the Python microservice, explaining how to use it, what inputs it expects, and what outputs it provides.

# Embedify - Python Microservice for Code Embedding Generation

`Embedify` is a Python microservice that listens to a RabbitMQ queue, processes code or query requests, generates embeddings using CodeBERT, and publishes the resulting embeddings back to RabbitMQ. It is designed to run as a long-lived worker service, processing incoming requests continuously.

## Requirements

- Python 3.8+
- RabbitMQ server
- Installed Python packages:
  - `pika` for RabbitMQ messaging
  - `torch` and `transformers` for tokenization and embedding generation

Install the required packages with:

```bash
pip install pika torch transformers
```

## Configuration

The microservice connects to RabbitMQ using environment variables:

- `RABBITMQ_HOST`: RabbitMQ server host (default: `localhost`)
- `RABBITMQ_PORT`: RabbitMQ server port (default: `5672`)
- `RABBITMQ_USER`: RabbitMQ username (default: `guest`)
- `RABBITMQ_PASS`: RabbitMQ password (default: `guest`)
- `RABBITMQ_VHOST`: RabbitMQ virtual host (default: `/`)

Set these environment variables as needed in your system or Docker environment.

## RabbitMQ Setup

This service uses two main exchanges and queues in RabbitMQ:

1. **Request Queue**: Accepts incoming code or query embedding requests.
   - **Exchange**: `embeddings_requests`
   - **Queue**: `embeddings_requests`

2. **Response Queues**: Publishes generated embeddings.
   - **Exchange**: `embeddings_responses`
   - **Code Embeddings Queue**: `code_embedding_responses`
   - **Query Embeddings Queue**: `query_embedding_responses`

## Message Structure

### Request Message Format

Requests to `Embedify` must follow this JSON structure:

```json
{
  "requestId": "unique-request-id",
  "type": "code",         // "code" for code snippets, "query" for user queries
  "content": "public void InitializeDatabase() { ... }"
}
```

- **requestId**: Unique identifier for each request, used to match responses.
- **type**: Specifies if the content is a code snippet (`code`) or a natural language query (`query`).
- **content**: The actual code or query to process.

### Response Message Format

The embedding results published back to RabbitMQ follow this format:

```json
{
  "requestId": "unique-request-id",
  "embedding": [0.12, -0.34, ..., 0.56]  // Embedding vector
}
```

- **requestId**: Matches the request ID to help consumers identify the response.
- **embedding**: The embedding vector generated for the content.

## How It Works

1. **Connect to RabbitMQ**: The microservice establishes a connection to RabbitMQ and subscribes to the `embeddings_requests` queue.

2. **Receive and Process Requests**:
   - For each incoming message, `Embedify` verifies the request type (either `code` or `query`).
   - It then tokenizes and generates an embedding for the provided content using CodeBERT.

3. **Publish Results**:
   - The generated embedding, along with the `requestId`, is published to `embeddings_responses` with a routing key matching the request type (`code` or `query`).
   - The result will be routed to either the `code_embedding_responses` or `query_embedding_responses` queue based on the request type.

## Running the Service

You can run `Embedify` directly:

```bash
python main.py
```

Or set it up as a background process or Docker container to keep it running indefinitely.

### Example Usage

1. **Send a Request**:
   - Send a JSON message to the `embeddings_requests` queue with a code snippet or query.

2. **Receive a Response**:
   - Listen on `code_embedding_responses` or `query_embedding_responses` to receive the embedding result.

## Notes

- Ensure that RabbitMQ is running and configured correctly before starting `Embedify`.
- For larger codebases or high-frequency usage, consider scaling with multiple instances.

## License

This project is licensed under the MIT License.