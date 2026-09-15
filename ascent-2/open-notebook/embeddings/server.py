"""Minimal OpenAI-compatible embeddings server for Open Notebook."""
from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_ID = os.getenv("MODEL_ID", "BAAI/bge-small-en-v1.5")
SERVED_NAME = os.getenv("SERVED_MODEL_NAME", "bge-small-en-v1.5")

print(f"Loading embedding model {MODEL_ID} ...", flush=True)
model = SentenceTransformer(MODEL_ID)
print("Model ready.", flush=True)

app = FastAPI(title="local-embeddings")


class EmbedRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
