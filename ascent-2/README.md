# ascent-2 rebuild recipe

Rebuild the **ascent-2** AI stack from a clean OS install to match the production
capabilities as of 2026-09-15.

**Host:** ASUS GX10 / NVIDIA DGX Spark (GB10), **aarch64**, ~121 GiB unified memory  
**OS baseline:** Ubuntu 24.04 (NVIDIA DGX / Spark image), kernel `6.17.0-*-nvidia`

---

## Capabilities restored by this recipe

| Capability | How |
|---|---|
| **Llama 4 Scout** OpenAI-compatible chat (Cursor / tools) | vLLM container `llama4-scout` on `:8000` |
| **Open Notebook** research UI | Compose stack on Tailscale `:8502` / API `:5055` |
| **Embeddings** | Local CPU service `bge-small-en-v1.5` |
| **TTS / STT** | Speaches (Kokoro + faster-whisper-small), CPU |
| **Hardening** | Tailscale-only UI/API bind, CORS locked, no host publish for embed/STT |
| **Backups** | Nightly local tarball on ascent-2 (`03:00`) |

**Not restored (intentionally disabled):** Laguna NVFP4, Nemotron Lightning NVFP4  
(they fought Scout for unified memory).

---

## Architecture

```
Tailscale peers
    │
    ├─ Cursor / clients ──► :8000  llama4-scout (vLLM, GPU)
    │
    └─ Browser ──► :8502/:5055  open_notebook
                        │
                        ├─ host.docker.internal:8000  (Scout chat)
                        ├─ local-embeddings:8080      (bge-small)
                        ├─ speaches:8000              (TTS/STT)
                        └─ surrealdb:8000             (DB, internal only)
```

---

## Prerequisites (clean OS)

1. **NVIDIA driver + GB10** visible: `nvidia-smi -L` shows `NVIDIA GB10`
2. **Docker Engine** + **NVIDIA Container Toolkit** (`nvidia-ctk`), docker enabled at boot
3. **Tailscale** joined; MagicDNS name `ascent-2`
4. User **`mplong`** with home dirs used below (or adjust paths)
5. **Hugging Face token** with access to:
   - `RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16`
   - (tokenizer files come with that repo)

```bash
# Typical Ubuntu packages (adjust for the Spark image you flash)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 curl jq python3
# Install NVIDIA Container Toolkit per NVIDIA docs for arm64, then:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker docker.socket
sudo usermod -aG docker mplong   # if running Scout as mplong
```

Store HF token where the launch script expects it:

```bash
# /home/mplong/.cache/huggingface/stored_tokens  (or export HF_TOKEN=...)
# format used by run script:  hf_token = hf_xxx...
mkdir -p /home/mplong/.cache/huggingface
printf 'hf_token = %s\n' "$HF_TOKEN" > /home/mplong/.cache/huggingface/stored_tokens
chown -R mplong:mplong /home/mplong/.cache/huggingface
chmod 600 /home/mplong/.cache/huggingface/stored_tokens
```

Copy this recipe tree onto the box (git clone of `randotools`, or rsync):

```bash
# example
sudo mkdir -p /opt/randotools && sudo chown "$USER:$USER" /opt/randotools
# clone/copy repo so /opt/randotools/ascent-2 exists
```

---

## 1. Llama 4 Scout (vLLM)

### 1.1 Download model (~61 GB weights; cache ends up ~170 GB with hub metadata)

```bash
sudo -u mplong -H bash -lc '
  export HF_TOKEN=$(awk -F"= " "/hf_token/{print \$2}" ~/.cache/huggingface/stored_tokens)
  export HF_HOME=/home/mplong/nim_cache
  mkdir -p "$HF_HOME"
  # huggingface-cli or:
  docker run --rm -e HF_TOKEN -v /home/mplong/nim_cache:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    huggingface-cli download RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16
'
```

### 1.2 Patch tokenizer (`<|python_tag|>` required for `llama4_json`)

```bash
sudo bash /opt/randotools/ascent-2/scout/patch_tokenizer.sh
# writes /home/mplong/Nemotron/scout_tokenizer_cursor
```

### 1.3 Install launch script and start

```bash
sudo install -d -o mplong -g mplong /home/mplong/Nemotron
sudo install -m 0755 -o mplong -g mplong \
  /opt/randotools/ascent-2/scout/run_llama4_scout.sh \
  /home/mplong/Nemotron/run_llama4_scout.sh
sudo -u mplong -H /home/mplong/Nemotron/run_llama4_scout.sh
```

### 1.4 Verify

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama4-scout","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
```

**Served name:** `llama4-scout`  
**Endpoint:** `http://ascent-2:8000/v1` (Tailscale) — Cursor API key can be any dummy (`sk-local`)  
**Sampling defaults in vLLM:** temperature `0.6`, top_p `0.9`, min_p `0.01`  
**Context:** `65536`, max concurrent seqs `8`, KV cache `fp8_e4m3`, GPU util `0.80`

> Soft `reboot` on this GX10 can leave the machine powered off — power button may be required.

---

## 2. Open Notebook (+ embeddings + Speaches)

```bash
sudo bash /opt/randotools/ascent-2/scripts/install-open-notebook.sh
# reads password from /opt/open-notebook/.env after install:
set -a; source /opt/open-notebook/.env; set +a
sudo -E bash /opt/randotools/ascent-2/scripts/configure-open-notebook-models.sh
```

What that does:

- Builds `local-embeddings` (`BAAI/bge-small-en-v1.5`)
- Starts Speaches CPU; downloads Kokoro TTS + Whisper-small STT
- Starts SurrealDB + Open Notebook
- Binds UI/API to **Tailscale IP only**
- Sets `CORS_ORIGINS` to `http://ascent-2:8502` and the Tailscale URL
- Enables `open-notebook-backup-local.timer` (03:00)
- Registers three OpenAI-compatible credentials + default models via API

**Important:** Esperanto ignores per-modality `endpoint_*` when `base_url` is set — use **separate credentials** for Scout / embeddings / Speaches (the configure script does this).

### Verify

```bash
TS=$(tailscale ip -4)
curl -sS http://$TS:5055/health
curl -sS -o /dev/null -w "%{http_code}\n" http://$TS:8502/
# embeddings (from compose network)
docker exec open-notebook-open_notebook-1 \
  python -c "import urllib.request; print(urllib.request.urlopen('http://local-embeddings:8080/health').read())"
```

UI: `http://ascent-2:8502` — password in `/opt/open-notebook/.env`.

---

## 3. Optional: omenboy pull backups

Recipe lives under `/opt/randotools/backups/open-notebook/` on omenboy.
On omenboy (host shell with systemd):

```bash
sudo /opt/randotools/backups/open-notebook/bin/install-timer.sh
```

Pulls ascent-2 `latest.tar.gz` nightly at 03:15. See that README for restore notes.

---

## 4. Cursor client

| Setting | Value |
|---|---|
| Base URL | `http://ascent-2:8000/v1` (or Tailscale IP) |
| API key | `sk-local` (ignored by vLLM) |
| Model | `llama4-scout` |
| Temperature / top_p | `0.6` / `0.9` if overridden in Cursor |

---

## 5. LangGraph (optional, on omenboy)

Already in this repo: `langgraph_scout/` — points at Scout over Tailscale. Not required on ascent-2 itself.

---

## File map

```
ascent-2/
├── README.md                          ← this recipe
├── scout/
│   ├── run_llama4_scout.sh            ← vLLM docker run
│   └── patch_tokenizer.sh             ← add <|python_tag|>
├── open-notebook/
│   ├── docker-compose.yml
│   ├── .env.example
│   ├── embeddings/{Dockerfile,server.py}
│   └── backup/backup-open-notebook-local.sh
└── scripts/
    ├── install-open-notebook.sh
    └── configure-open-notebook-models.sh
```

---

## Post-install checklist

- [ ] `docker ps` shows `llama4-scout`, `open_notebook`, `surrealdb`, `local-embeddings`, `speaches`
- [ ] Scout `/v1/models` lists `llama4-scout`
- [ ] Open Notebook Settings shows Chat / Embedding / TTS / STT defaults filled
- [ ] UI not reachable on LAN IP / localhost — only Tailscale
- [ ] `systemctl list-timers open-notebook-backup-local.timer` shows next run
- [ ] `/opt/open-notebook/.env` backed up offline (encryption key!)

---

## Known sharp edges

1. **Memory:** Scout alone uses most of unified memory (~95–105 GiB). Keep Speaches/embeddings on CPU; don’t co-locate another big GPU model.
2. **HF access:** NVIDIA NVFP4 Scout weights were gated; RedHat W4A16 is what we use.
3. **Tokenizer:** Without the patch, tool calling with `llama4_json` fails.
4. **SurrealDB vs Scout ports:** Surreal must **not** publish host `:8000`.
5. **Tailscale IP change:** Re-run `install-open-notebook.sh` (or update `.env` `TAILSCALE_IP` + `docker compose up -d`) if the Tailscale IPv4 changes.
6. **Image tags:** `vllm/vllm-openai:latest` and `lfnovo/open_notebook:v1-latest` float — pin digests for stricter reproducibility once validated.
