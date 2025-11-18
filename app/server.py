import os
from typing import List, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/llama-embed-nemotron-8b")
HF_TOKEN = os.getenv("HF_TOKEN")
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

model_kwargs = {}
if HF_TOKEN:
    model_kwargs["token"] = HF_TOKEN

model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN,
)
model.max_seq_length = getattr(model, "max_seq_length", 8192)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str


@app.get("/openai/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/openai/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    texts = req.input if isinstance(req.input, list) else [req.input]
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )
    data = [EmbeddingData(embedding=emb.tolist(), index=i) for i, emb in enumerate(embeddings)]
    return EmbeddingResponse(data=data, model=MODEL_NAME)


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}
