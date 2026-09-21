#!/bin/bash
set -euo pipefail
# Llama 4 Scout for Cursor on GB10 (vLLM).
# - llama4_json tool parser (native JSON tool format)
# - patched tokenizer includes <|python_tag|> required by vLLM parser
# - Meta sampling defaults
# - Multimodal (vision tower in W4A16 checkpoint): up to 2 images/request
# - 64k context / 2 seqs — KV pool supports 64k; concurrency/images cut for GB10 headroom

export HF_TOKEN="${HF_TOKEN:-$(awk -F'= ' '/hf_token/{print $2}' /home/mplong/.cache/huggingface/stored_tokens)}"
MODEL="RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"
TOKENIZER="/home/mplong/Nemotron/scout_tokenizer_cursor"
NAME="llama4-scout"
CACHE="/home/mplong/nim_cache"

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p 8000:8000 \
  -v "$CACHE:/root/.cache/huggingface" \
  -v "$TOKENIZER:/tokenizer:ro" \
  -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  --model "$MODEL" \
  --tokenizer /tokenizer \
  --served-model-name llama4-scout \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":2}' \
  --gpu-memory-utilization 0.78 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-auto-tool-choice \
  --tool-call-parser llama4_json \
  --override-generation-config '{"temperature":0.6,"top_p":0.9,"min_p":0.01}' \
  --trust-remote-code

echo "Started $NAME (:8000) 64k, max_num_seqs=2, mm image=2, util=0.78, tool-parser=llama4_json"
docker ps --filter "name=$NAME"
