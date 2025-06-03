
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import time
import argparse

def create_sample_corpus():
    """Create a sample corpus of text documents"""
    return [
        "Machine learning is a subset of artificial intelligence that involves building systems that can learn from data.",
        "Deep learning is a type of machine learning that uses neural networks with many layers.",
        "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
        "Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos.",
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment.",
        "Transfer learning is a machine learning technique where a model trained on one task is re-purposed on a second related task.",
        "Generative AI refers to artificial intelligence systems that can generate new content, such as text, images, or music.",
        "Python is a popular programming language for data science and machine learning applications.",
        "TensorFlow and PyTorch are popular deep learning frameworks used by researchers and practitioners.",
        "Data preprocessing is a crucial step in the machine learning pipeline that involves transforming raw data into a suitable format.",
        "Overfitting occurs when a model learns the training data too well, including its noise and outliers.",
        "Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.",
        "Feature engineering is the process of using domain knowledge to extract features from raw data.",
        "Hyperparameter tuning is the process of finding the optimal hyperparameters for a machine learning algorithm.",
        "Ensemble methods combine multiple machine learning models to improve performance.",
        "Clustering is an unsupervised learning technique that groups similar data points together.",
        "Classification is a supervised learning task where the model learns to assign labels to examples.",
        "Regression is a supervised learning task where the model predicts a continuous value.",
        "Recommender systems suggest items or content to users based on their preferences or behavior.",
        "A neural network is a computational model inspired by the structure and function of biological neural networks."
    ]

def semantic_search(model, query, corpus, top_k=5):
    """Search for semantically similar documents in the corpus"""
    start_time = time.time()
    
    # Encode query and corpus
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    # Compute cosine similarities
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # Get top_k results
    top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))
    
    end_time = time.time()
    
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append({
            "score": score.item(),
            "text": corpus[idx]
        })
    
    return results, end_time - start_time

def main():
    parser = argparse.ArgumentParser(description="Semantic Search Demo")
    parser.add_argument("--query", type=str, default="How does AI learn from data?", 
                        help="Query text for semantic search")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                        help="Sentence Transformer model name")
    parser.add_argument("--top_k", type=int, default=5, 
                        help="Number of top results to return")
    args = parser.parse_args()
    
    print(f"Loading model {args.model}...")
    model = SentenceTransformer(args.model)
    print("Model loaded successfully!\n")
    
    corpus = create_sample_corpus()
    print(f"Created a corpus with {len(corpus)} documents.\n")
    
    print(f"Query: {args.query}\n")
    results, execution_time = semantic_search(model, args.query, corpus, top_k=args.top_k)
    
    print(f"Results (execution time: {execution_time:.4f} seconds):")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.4f}")
        print(f"   Text: {result['text']}")
    
    # Create an interactive demo that allows multiple queries
    while True:
        print("\n" + "-"*80)
        query = input("\nEnter a new query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        results, execution_time = semantic_search(model, query, corpus, top_k=args.top_k)
        
        print(f"\nResults (execution time: {execution_time:.4f} seconds):")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Score: {result['score']:.4f}")
            print(f"   Text: {result['text']}")

if __name__ == "__main__":
    main()