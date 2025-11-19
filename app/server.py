import os
from typing import Any, Dict, List

import runpod.serverless
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/llama-embed-nemotron-8b")
HF_TOKEN = os.getenv("HF_TOKEN")
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN,
)
model.max_seq_length = getattr(model, "max_seq_length", 8192)


def handler(event: Dict[str, Any]):
    body = event.get("input", {})
    req_model = body.get("model", MODEL_NAME)
    raw_input = body.get("input", "")
    texts: List[str] = raw_input if isinstance(raw_input, list) else [raw_input]

    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )

    data = [
        {"object": "embedding", "embedding": emb.tolist(), "index": idx}
        for idx, emb in enumerate(embeddings)
    ]

    return {
        "object": "list",
        "data": data,
        "model": req_model,
    }


runpod.serverless.start({"handler": handler})
