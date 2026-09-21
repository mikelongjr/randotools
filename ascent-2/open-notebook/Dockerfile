# Open Notebook with Docling→Scout vision + ARM-safe RapidOCR (onnxruntime).
FROM lfnovo/open_notebook:v1-latest

# Docling defaults RapidOCR to onnxruntime; without this package it falls back to
# CUDA-built torch on CPU (aarch64), which segfaults the surreal worker.
RUN /app/.venv/bin/pip install --no-cache-dir "onnxruntime>=1.20,<2"

COPY patches/content_core_docling.py \
  /app/.venv/lib/python3.12/site-packages/content_core/processors/document/docling.py
