#!/usr/bin/env bash
# Wire Open Notebook credentials/models via API.
set -euo pipefail

BASE="${OPEN_NOTEBOOK_API:-http://$(tailscale ip -4):5055}"
PW="${OPEN_NOTEBOOK_PASSWORD:?set OPEN_NOTEBOOK_PASSWORD}"
AUTH=( -H "Authorization: Bearer ${PW}" -H "Content-Type: application/json" )

api() {
  local method=$1 path=$2
  shift 2
  curl -sS -X "$method" "${AUTH[@]}" "$BASE$path" "$@"
}

echo "Using API $BASE"

CREDS="$(api GET /api/credentials)"

ensure_cred() {
  local name=$1 mods=$2 url=$3
  local id
  id=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(next((c['id'] for c in d if c['name']==sys.argv[2]), ''))" "$CREDS" "$name")
  if [[ -n "$id" ]]; then
    echo "OK credential: $name ($id)"
    echo "$id"
    return
  fi
  local body
  body=$(python3 -c "import json,sys; print(json.dumps({'name':sys.argv[1],'provider':'openai_compatible','modalities':json.loads(sys.argv[2]),'api_key':'sk-local','base_url':sys.argv[3]}))" "$name" "$mods" "$url")
  id=$(api POST /api/credentials -d "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  CREDS="$(api GET /api/credentials)"
  echo "CREATED credential: $name ($id)"
  echo "$id"
}

SCOUT_ID=$(ensure_cred "Scout LLM" '["language"]' "http://host.docker.internal:8000/v1" | tail -1)
EMB_ID=$(ensure_cred "Local Embeddings" '["embedding"]' "http://local-embeddings:8080/v1" | tail -1)
SPE_ID=$(ensure_cred "Local Speaches" '["text_to_speech","speech_to_text"]' "http://speaches:8000/v1" | tail -1)

MODELS="$(api GET /api/models)"

ensure_model() {
  local name=$1 typ=$2 cid=$3
  local id
  id=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(next((m['id'] for m in d if m['name']==sys.argv[2] and m['type']==sys.argv[3]), ''))" "$MODELS" "$name" "$typ")
  if [[ -n "$id" ]]; then
    echo "OK model: $name ($typ) $id"
    echo "$id"
    return
  fi
  local body
  body=$(python3 -c "import json,sys; print(json.dumps({'name':sys.argv[1],'provider':'openai_compatible','type':sys.argv[2],'credential':sys.argv[3]}))" "$name" "$typ" "$cid")
  id=$(api POST /api/models -d "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  MODELS="$(api GET /api/models)"
  echo "CREATED model: $name ($typ) $id"
  echo "$id"
}

CHAT_ID=$(ensure_model "llama4-scout" language "$SCOUT_ID" | tail -1)
EMBED_MID=$(ensure_model "bge-small-en-v1.5" embedding "$EMB_ID" | tail -1)
TTS_ID=$(ensure_model "speaches-ai/Kokoro-82M-v1.0-ONNX" text_to_speech "$SPE_ID" | tail -1)
STT_ID=$(ensure_model "Systran/faster-whisper-small" speech_to_text "$SPE_ID" | tail -1)

DEFAULTS=$(python3 -c "import json,sys; print(json.dumps({
  'default_chat_model': sys.argv[1],
  'default_transformation_model': sys.argv[1],
  'large_context_model': sys.argv[1],
  'default_tools_model': sys.argv[1],
  'default_embedding_model': sys.argv[2],
  'default_text_to_speech_model': sys.argv[3],
  'default_speech_to_text_model': sys.argv[4],
}))" "$CHAT_ID" "$EMBED_MID" "$TTS_ID" "$STT_ID")

api PUT /api/models/defaults -d "$DEFAULTS" | python3 -m json.tool
echo "Open Notebook models configured."
