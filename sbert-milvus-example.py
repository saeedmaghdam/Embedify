from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("F:\\SBERT-all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
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

# 2. Calculate embeddings
embeddings = model.encode(sentences)

# 3. Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# 4. Create a collection in Milvus
collection_name = "sentence_embeddings"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=500, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1])
]
schema = CollectionSchema(fields)
collection = Collection(name=collection_name, schema=schema)

# 5. Insert data into Milvus
entities = [
    sentences,
    embeddings.tolist()
]
collection.insert(entities)
collection.flush()

# 6. Create an index on the embedding field
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 100}
}
collection.create_index(field_name="embedding", index_params=index_params)

# 7. Load the collection into memory
collection.load()

# 8. Encode the query
# query = "How do I optimize image loading times?" # ✅
query = "How I done any database migrations?"

query_embedding = model.encode([query])

# 9. Perform similarity search in Milvus
search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param=search_params,
    limit=1,
    output_fields=["sentence"]
)

# 10. Retrieve and print the most similar sentence
if results:
    most_similar_sentence = results[0][0].entity.get("sentence")
    print(f"Most similar sentence: {most_similar_sentence}")
else:
    print("No similar sentences found.")
