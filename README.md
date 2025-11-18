# RunPod OpenAI-Compatible Embedding Worker

This worker exposes a minimal OpenAI-compatible API (`/openai/v1/models` and `/openai/v1/embeddings`) around a Hugging Face sentence-transformers model. Deploy it to a RunPod Serverless endpoint and point RAGFlow (or any OpenAI client) at the endpoint.

## Features

- Loads any `sentence-transformers` checkpoint via `MODEL_NAME`
- Optional `HF_TOKEN` for private/gated repos
- FastAPI + Uvicorn server listening on port 80
- `/openai/v1/models` returns the configured model ID so RAGFlow's validation passes
- `/openai/v1/embeddings` accepts both string and array inputs just like OpenAI

## Configuration

Environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `MODEL_NAME` | Hugging Face repo or local path for the embedding model | `nvidia/llama-embed-nemotron-8b` |
| `HF_TOKEN` | (Optional) Hugging Face access token for private models | empty |
| `NORMALIZE_EMBEDDINGS` | Set to `false` to disable L2 normalization | `true` |
| `EMBED_BATCH_SIZE` | Batch size for encoding | `16` |

## Building locally

```bash
cd runpod-worker
docker build -t <your-registry>/runpod-embedding-worker:latest .
```

## Deploying to RunPod

1. Push the image to a registry accessible by RunPod (RunPod Registry or Docker Hub).
2. Create a Serverless endpoint and use this image.
3. Set environment variables (MODEL_NAME, HF_TOKEN, etc.).
4. Increase the container disk size to fit the model (e.g., 40GB for 12B checkpoints).
5. Once the endpoint is `Ready`, configure RAGFlow with:
   - Base URL: `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/`
   - Model: same as `MODEL_NAME`
   - API key: your RunPod API key.
