# Qwen3.8-Flash-Next NVFP4 on ascent-2 (test lane)

Follows [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark) single-Spark TP1 defaults.

| | |
|---|---|
| Host | ascent-2 (Asus GX10 / GB10) |
| Endpoint | `http://100.70.176.65:8000/v1` |
| Model id | `qwen3.8-flash-next` |
| Checkpoint | `/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia` (`nvidia/Qwen3.8-Flash-Next-NVFP4`) |
| Image | `pilcothink/vllm_spark_qwen38:0.28` (SSD PLE; works on GX10) |
| PLE | `VLLM_PLE_SSD=1` + `--engram-config.cpu_offload false` |
| DSH | omenboy → `gx10-flash` / `qwen3.8-flash-next` |

**Conflicts with Scout** — stop Scout before launching (same `:8000`).

tonyd2wild staged/mmap patches target an older `vllm-openai:nightly` digest (removed from Hub) and produce garbage on current nightly; do not use them with floating `nightly`.

## Tuning applied from the tonyd2wild recipe

| Knob | Value | Why |
|---|---|---|
| `--max-num-seqs` | 6 | recipe single-Spark default (was 2) |
| `cudagraph_capture_sizes` | `[4,8,12,16,20,24]` | `(1+MTP)*seqs`; precondition for seqs > 4 |
| `--max-num-batched-tokens` | 4096 | recipe's biggest single lever |
| compile off + `FULL_DECODE_ONLY` | | must pair with the 4096 chunk |
| `VLLM_PLE_SSD_WORKERS` | 32 | more PLE read threads (image default 8) |
| `VLLM_USE_DEEP_GEMM` | 0 | DeepGEMM faults on sm_121 (vLLM #54125) |

Measured on ascent-2, 200-token replies: prose 19.7 → 21.2 tok/s, code 34.2 → 36.0 tok/s, KV pool 610,724 → 648,983 tokens. MemAvailable ~14 GiB (community danger line is ~10 GiB).

**Not available here:**
- `--kv-cache-dtype fp8_e4m3` — rejected: `Qwen4Exp QSA requires a BF16 main KV cache`. The guard is two lines in `models/qwen4_exp/nvidia/qsa.py`, but the image's QSA path has no fp8 handling at all, so removing it would only corrupt attention. Needs vLLM PR #54846 (RFC #54426). The tonyd2wild overlay carries that PR but against a different vLLM vintage — its `qsa.py`/`ops_qsa.py` differ from this image's by 373 and 515 lines, so it cannot be dropped in. **Decision 2026-09-22: wait for #54846 upstream** rather than hand-rebasing. Worth roughly +59% KV pool (capacity, not decode speed).
- Reduced-vocab MTP draft (`DRAFT_VOCAB=65536`) — recipe's own `mtp.py` overlay, ~15% on prose. Not portable to this image.
- Staged gather — already present as the image's `staged-v3-preadv` SSD PLE reader.

**Watch item:** prefix caching is ON here (agent multi-turn); the recipe keeps it off pending the GDN prefix-cache crash, vLLM #54173.

```bash
# status
docker ps --filter name=vllm_qwen38fn
curl -s http://127.0.0.1:8000/v1/models | jq .
```
