import os
from typing import List, Union
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# Configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/llama-embed-nemotron-8b")
HF_TOKEN = os.getenv("HF_TOKEN")
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
PORT = int(os.getenv("PORT", "80"))

# Create FastAPI app
app = FastAPI(title="RunPod Embedding Worker")

# Global state for model loading
model = None
model_loading = True


# Request/Response models for OpenAI compatibility
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL_NAME


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "runpod"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


@app.on_event("startup")
async def load_model():
    """Load the model during application startup"""
    global model, model_loading
    print(f"Loading model: {MODEL_NAME}")
    try:
        model = SentenceTransformer(
            MODEL_NAME,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        model.max_seq_length = getattr(model, "max_seq_length", 8192)
        print(f"Model loaded successfully. Max sequence length: {model.max_seq_length}")
        model_loading = False
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/ping")
async def health_check(response: Response):
    """
    Required health check endpoint for RunPod load balancing.
    Returns 204 while initializing, 200 when ready, 503 on error.
    """
    if model_loading:
        response.status_code = 204  # Initializing
        return {"status": "initializing"}
    elif model is None:
        response.status_code = 503  # Service unavailable
        return {"status": "error", "message": "Model failed to load"}
    else:
        response.status_code = 200  # Healthy
        return {"status": "healthy", "model": MODEL_NAME}


@app.get("/")
async def root():
    return {"status": "healthy" if not model_loading else "initializing", "model": MODEL_NAME}


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint for RAGFlow validation"""
    return ModelListResponse(
        object="list",
        data=[
            ModelInfo(
                id=MODEL_NAME,
                object="model",
                created=0,
                owned_by="runpod"
            )
        ]
    )


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Handle both string and list inputs
        texts = [request.input] if isinstance(request.input, str) else request.input
        
        if not texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )
        
        # Format response
        data = [
            EmbeddingData(
                object="embedding",
                embedding=emb.tolist(),
                index=idx
            )
            for idx, emb in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            object="list",
            data=data,
            model=request.model
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Use PORT environment variable (RunPod load balancing standard)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
