import numpy as np
import ollama

model = 'nomic-embed-text'

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# compare several words
automobile_embedding = ollama.embeddings(model=model, prompt='automobile')['embedding']
vehicle_embedding = ollama.embeddings(model=model, prompt='vehicle')['embedding']
dinosaur_embedding = ollama.embeddings(model=model, prompt='dinosaur')['embedding']
stick_embedding = ollama.embeddings(model=model, prompt='stick')['embedding']

# comparing cosine similarity, automobiles vs automobiles should be 1.0, i.e exactly the same, while automobiles vs dinosaurs should be between 0 and 1, i.e. not the same
print(cosine_similarity(automobile_embedding, automobile_embedding))
print(cosine_similarity(automobile_embedding, vehicle_embedding))
print(cosine_similarity(automobile_embedding, dinosaur_embedding))
print(cosine_similarity(automobile_embedding, stick_embedding))