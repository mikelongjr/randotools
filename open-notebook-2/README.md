# Open Notebook #2 (omenboy) — second isolated client dataset

| Instance | UI | API | Data |
|---|---|---|---|
| **ON #1** (existing) | `:8502` | `:5055` | `/opt/open-notebook` + `/space/open-notebook` |
| **ON #2** (this) | `:8503` | `:5056` | `/opt/open-notebook-2` + `/space/open-notebook-2` |

Shared (not client data): `local-embeddings` (Qwen3), `speaches`.  
Isolated per client: SurrealDB, notebook files, password, encryption key.

## Deploy

```bash
sudo mkdir -p /opt/open-notebook-2 /space/open-notebook-2/{surreal_data,notebook_data}
sudo rsync -a --exclude '.env' /opt/randotools/open-notebook-2/ /opt/open-notebook-2/
# .env is created at install with fresh secrets
cd /opt/open-notebook-2
sudo podman-compose --env-file .env up -d
sudo -E bash configure-models.sh   # needs OPEN_NOTEBOOK_PASSWORD from .env
sudo systemctl enable --now open-notebook-2.service
```

UI: `http://<tailscale-ip>:8503` — password in `/opt/open-notebook-2/CREDENTIALS.txt`.

Docling defaults: OCR on, vision/formulas off (bulk-ingest friendly).
