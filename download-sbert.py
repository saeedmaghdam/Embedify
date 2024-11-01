from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the path to save the model
save_path = '.'

# Save the model
model.save(save_path)
