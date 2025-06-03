
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import os
import time

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Similarity API",
    description="API for generating text embeddings and calculating semantic similarity",
    version="1.0.0"
)

# Create model directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Initialize the model
MODEL_NAME = "all-MiniLM-L6-v2"
model = None

# Pydantic models for request and response
class TextItem(BaseModel):
    text: str

class TextsItem(BaseModel):
    texts: List[str]

class SimilarityRequest(BaseModel):
    text1: str
    text2: str

class SimilarityResponse(BaseModel):
    similarity: float
    execution_time: float

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    execution_time: float

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    execution_time: float

@app.on_event("startup")
async def startup_event():
    global model
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully!")

@app.get("/")
def read_root():
    return {"message": "Semantic Similarity API is running"}

@app.post("/embedding", response_model=EmbeddingResponse)
def get_embedding(item: TextItem):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    start_time = time.time()
    embedding = model.encode(item.text)
    end_time = time.time()
    
    return {
        "embedding": embedding.tolist(),
        "execution_time": end_time - start_time
    }

@app.post("/batch_embedding", response_model=BatchEmbeddingResponse)
def get_batch_embedding(item: TextsItem):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    start_time = time.time()
    embeddings = model.encode(item.texts)
    end_time = time.time()
    
    return {
        "embeddings": embeddings.tolist(),
        "execution_time": end_time - start_time
    }

@app.post("/similarity", response_model=SimilarityResponse)
def calculate_similarity(request: SimilarityRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    start_time = time.time()
    # Encode both texts
    embedding1 = model.encode(request.text1)
    embedding2 = model.encode(request.text2)
    
    # Calculate cosine similarity
    similarity = util.cos_sim(embedding1, embedding2).item()
    end_time = time.time()
    
    return {
        "similarity": similarity,
        "execution_time": end_time - start_time
    }

# Run this if script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("embedding_service:app", host="0.0.0.0", port=8000, reload=True)