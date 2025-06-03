# Test the Sentence Transformer model
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example sentences
sentences = [
    "This is an example sentence",
    "Each sentence is converted to a vector",
    "This sentence is similar to the first one",
    "This is completely different"
]

# Encode sentences to get embeddings
embeddings = model.encode(sentences)

# Print the shape of the embeddings
print(f"Shape of embeddings: {embeddings.shape}")

# Calculate cosine similarity between all pairs
cosine_scores = util.cos_sim(embeddings, embeddings)

# Print the similarity scores
print("\nSimilarity scores:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        print(f"Sentence {i+1} vs Sentence {j+1}: {cosine_scores[i][j]:.4f}")