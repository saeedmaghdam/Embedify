from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")
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

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])

# query = "why do I migrate the database to a new server?" # ✅
# query = "have I done any database migration?" # ✅
query = "how do I optimize image loading times?" # ✅


# 4. Encode the query
query_embedding = model.encode(query)

# 5. Calculate the similarity between the query and the sentences
query_similarities = model.similarity(query_embedding, embeddings)
print(query_similarities)

# 6. Find the most similar sentence
most_similar_idx = query_similarities.argmax().item()
most_similar_sentence = sentences[most_similar_idx]
print(most_similar_sentence)