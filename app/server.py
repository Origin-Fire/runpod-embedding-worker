import os
from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# Configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/llama-embed-nemotron-8b")
HF_TOKEN = os.getenv("HF_TOKEN")
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# Load the model at startup
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    token=HF_TOKEN,
)
model.max_seq_length = getattr(model, "max_seq_length", 8192)
print(f"Model loaded successfully. Max sequence length: {model.max_seq_length}")

# Create FastAPI app
app = FastAPI(title="RunPod Embedding Worker")


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


@app.get("/")
async def root():
    return {"status": "healthy", "model": MODEL_NAME}


@app.get("/health")
async def health():
    return {"status": "healthy"}


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
    # RunPod load balancer expects the server to listen on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
