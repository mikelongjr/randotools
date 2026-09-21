"""Minimal OpenAI-compatible embeddings server for Open Notebook."""
from __future__ import annotations

import os
import time
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
SERVED_NAME = os.getenv("SERVED_MODEL_NAME", "qwen3-embedding:0.6b-4k")

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(
    f"Loading embedding model {MODEL_ID} on {DEVICE} "
    f"(hip={getattr(torch.version, 'hip', None)}, "
    f"cuda_avail={torch.cuda.is_available()}, "
    f"count={torch.cuda.device_count()}) ...",
    flush=True,
)
if DEVICE == "cuda":
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"GPU name lookup failed: {exc}", flush=True)

model = SentenceTransformer(MODEL_ID, device=DEVICE)
print("Model ready.", flush=True)

app = FastAPI(title="local-embeddings")


class EmbedRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": DEVICE}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest) -> dict[str, Any]:
    texts = req.input if isinstance(req.input, list) else [req.input]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "model": SERVED_NAME,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
