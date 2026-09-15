#!/usr/bin/env bash
# Patch RedHat Llama 4 Scout tokenizer: add <|python_tag|> for vLLM llama4_json.
set -euo pipefail

CACHE="${NIM_CACHE:-/home/mplong/nim_cache}"
MODEL_DIR="$CACHE/hub/models--RedHatAI--Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"
OUT="${TOKENIZER_OUT:-/home/mplong/Nemotron/scout_tokenizer_cursor}"
SNAP="$(ls -d "$MODEL_DIR"/snapshots/* | head -1)"

if [[ ! -d "$SNAP" ]]; then
  echo "Model snapshot not found under $MODEL_DIR — download the model first." >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"
for name in tokenizer.json tokenizer_config.json tokenizer.model \
            special_tokens_map.json chat_template.json \
            processor_config.json preprocessor_config.json; do
  if [[ -e "$SNAP/$name" ]]; then
    cp -L "$SNAP/$name" "$OUT/$name"
    echo "copied $name"
  fi
done

python3 - << PY
import json
from pathlib import Path

out = Path("${OUT}")
special = "<|python_tag|>"

tj = json.loads((out / "tokenizer.json").read_text())
added = tj.setdefault("added_tokens", [])
names = {t.get("content") for t in added if isinstance(t, dict)}
if special not in names:
    next_id = max((t.get("id", 0) for t in added if isinstance(t, dict)), default=200000) + 1
    # Prefer id adjacent to other python_* tokens if present
    for t in added:
        if isinstance(t, dict) and t.get("content") == "<|python_end|>":
            next_id = t["id"] + 1
            break
    added.append({
        "id": next_id,
        "content": special,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    })
    # Keep tokenizer.json vocab in sync when present (dict form)
    vocab = (tj.get("model") or {}).get("vocab")
    if isinstance(vocab, dict) and special not in vocab:
        vocab[special] = next_id
    (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
    print(f"added {special} id={next_id}")
else:
    print(f"{special} already present")

tc_path = out / "tokenizer_config.json"
tc = json.loads(tc_path.read_text())
decoder = tc.setdefault("added_tokens_decoder", {})
# find id from tokenizer.json
aid = next(t["id"] for t in added if t.get("content") == special)
decoder[str(aid)] = {
    "content": special,
    "lstrip": False,
    "normalized": False,
    "rstrip": False,
    "single_word": False,
    "special": True,
}
extra = tc.setdefault("additional_special_tokens", [])
if special not in extra:
    extra.append(special)
tc_path.write_text(json.dumps(tc, indent=2, ensure_ascii=False) + "\n")
print("tokenizer_config.json updated")
PY

chown -R mplong:mplong "$(dirname "$OUT")" 2>/dev/null || true
echo "Patched tokenizer at $OUT"
