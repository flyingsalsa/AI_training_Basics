
import requests
import json
import time

# Base URL of the API
BASE_URL = "http://localhost:8000"

def test_root():
    response = requests.get(f"{BASE_URL}/")
    print(f"Root endpoint: {response.status_code}")
    print(response.json())
    print()

def test_embedding():
    data = {"text": "This is a test sentence for embedding generation."}
    response = requests.post(f"{BASE_URL}/embedding", json=data)
    print(f"Embedding endpoint: {response.status_code}")
    result = response.json()
    print(f"Embedding dimension: {len(result['embedding'])}")
    print(f"Execution time: {result['execution_time']:.4f} seconds")
    print()

def test_batch_embedding():
    data = {"texts": [
        "This is the first test sentence.",
        "Here is another sentence for embedding.",
        "Let's test how well batch processing works."
    ]}
    response = requests.post(f"{BASE_URL}/batch_embedding", json=data)
    print(f"Batch embedding endpoint: {response.status_code}")
    result = response.json()
    print(f"Number of embeddings: {len(result['embeddings'])}")
    print(f"Embedding dimension: {len(result['embeddings'][0])}")
    print(f"Execution time: {result['execution_time']:.4f} seconds")
    print()

def test_similarity():
    pairs = [
        {"text1": "I love machine learning and artificial intelligence.", 
         "text2": "Deep learning and neural networks are fascinating."},
        {"text1": "The weather is nice today.", 
         "text2": "I need to buy groceries."},
        {"text1": "The quick brown fox jumps over the lazy dog.", 
         "text2": "The fast brown fox leaps over the sleepy dog."}
    ]
    
    print("Similarity tests:")
    for pair in pairs:
        response = requests.post(f"{BASE_URL}/similarity", json=pair)
        result = response.json()
        print(f"Text 1: {pair['text1']}")
        print(f"Text 2: {pair['text2']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Execution time: {result['execution_time']:.4f} seconds")
        print()

if __name__ == "__main__":
    print("Testing Semantic Similarity API\n")
    test_root()
    test_embedding()
    test_batch_embedding()
    test_similarity()