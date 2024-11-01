# run the web api using `uvicorn sbert-milvus-webapi-example:app --reload`
# docs page: http://127.0.0.1:8000/docs or http://127.0.0.1:8000/redoc

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import uuid

app = FastAPI()

# Load the Sentence Transformer model
model = SentenceTransformer("F:\\SBERT-all-MiniLM-L6-v2")

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the collection name
collection_name = "sentence_embeddings_webapi"

# Define the default sentences
default_sentences = [
    "Implemented the export dialog to enhance user data portability.",
    "Refactored the authentication module to improve security protocols.",
    "Updated the user interface to align with the new branding guidelines.",
    "Optimized the database queries to reduce load times.",
    "Conducted a code review to ensure adherence to coding standards.",
    "Integrated third-party API for real-time data synchronization.",
    "Fixed the bug causing application crashes on startup.",
    "Deployed the latest build to the staging environment for testing.",
    "Documented the new features for the upcoming release notes.",
    "Attended the team meeting to discuss project milestones and deadlines.",
    "Implemented input validation to prevent user errors and enhance data integrity.",
    "Updated the API documentation to reflect recent changes and assist developers.",
    "Optimized image loading times to improve overall application performance.",
    "Conducted usability testing to identify and address user experience issues.",
    "Migrated the database to a new server to increase reliability and scalability.",
    "Developed unit tests for the payment processing module to ensure accuracy.",
    "Resolved synchronization issues between the mobile app and the web platform.",
    "Enhanced logging mechanisms to facilitate easier debugging and monitoring.",
    "Collaborated with the design team to refine the dashboard layout for better user engagement.",
    "Reviewed and merged pull requests to incorporate new features into the main codebase."
]

# Define the collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True, auto_id=False),
    FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]
schema = CollectionSchema(fields)
collection = Collection(name=collection_name, schema=schema)

# Insert default sentences into Milvus
def insert_default_sentences():
    if collection.num_entities == 0:
        ids = [str(uuid.uuid4()) for _ in default_sentences]
        embeddings = model.encode(default_sentences).tolist()
        entities = [
            ids,
            default_sentences,
            embeddings
        ]
        collection.insert(entities)
        collection.flush()
        # Create an index on the embedding field
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 100}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

insert_default_sentences()

# Pydantic models for request bodies
class Sentence(BaseModel):
    sentence: str

class UpdateSentence(BaseModel):
    id: str
    sentence: str

# CRUD Endpoints
@app.post("/add_sentence/")
async def add_sentence(sentence: Sentence):
    id = str(uuid.uuid4())
    embedding = model.encode([sentence.sentence]).tolist()
    entities = [
        [id],
        [sentence.sentence],
        embedding
    ]
    collection.insert(entities)
    collection.flush()
    return {"id": id, "sentence": sentence.sentence}

@app.delete("/delete_sentence/{sentence_id}")
async def delete_sentence(sentence_id: str):
    expr = f'id == "{sentence_id}"'
    collection.delete(expr)
    collection.flush()
    return {"message": f"Sentence with id {sentence_id} deleted."}

@app.put("/update_sentence/")
async def update_sentence(update: UpdateSentence):
    expr = f'id == "{update.id}"'
    collection.delete(expr)
    embedding = model.encode([update.sentence]).tolist()
    entities = [
        [update.id],
        [update.sentence],
        embedding
    ]
    collection.insert(entities)
    collection.flush()
    return {"id": update.id, "sentence": update.sentence}

@app.get("/get_sentence/{sentence_id}")
async def get_sentence(sentence_id: str):
    expr = f'id == "{sentence_id}"'
    results = collection.query(expr, output_fields=["sentence"])
    if results:
        return {"id": sentence_id, "sentence": results[0]["sentence"]}
    else:
        raise HTTPException(status_code=404, detail="Sentence not found.")

# Similarity Search Endpoint
@app.post("/similarity_search/")
async def similarity_search(query: Sentence):
    query_embedding = model.encode([query.sentence])
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=1,
        output_fields=["sentence"]
    )
    if results:
        most_similar_sentence = results[0][0].entity.get("sentence")
        return {"query": query.sentence, "most_similar_sentence": most_similar_sentence}
    else:
        return {"query": query.sentence, "most_similar_sentence": None}
