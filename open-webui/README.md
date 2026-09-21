# Open WebUI on omenboy — Scout chat + local Qwen3 RAG + SearXNG web search.
# Open Notebook is unchanged and keeps its own stack.

## Role

| Piece | Endpoint |
|---|---|
| Chat / tools LLM | Scout on ascent-2 `http://100.70.176.65:8000/v1` (`llama4-scout`) |
| RAG embeddings | omenboy `local-embeddings:8080` (`qwen3-embedding:0.6b-4k`) |
| Web search | local SearXNG in this compose |
| UI | Tailscale `:3000` |

Open Notebook remains on `:5055` / `:8502`.

## Deploy (live path `/opt/open-webui`)

```bash
sudo mkdir -p /opt/open-webui
sudo rsync -a /opt/randotools/open-webui/ /opt/open-webui/
# create /opt/open-webui/.env (see .env.example) then:
cd /opt/open-webui
sudo podman-compose --env-file .env up -d
```

First visit: create the admin account (signup enabled). Toggle **Web Search** on a chat to use SearXNG.

Scout defaults to **legacy** function calling so it does not agentically loop on empty `query_knowledge_bases` tools. Re-enable **Native** per-model in Workspace → Models only when you want agentic RAG and have knowledge attached.

## Systemd

```bash
sudo cp /opt/randotools/open-webui/open-webui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now open-webui.service
```

## Notes

- Joins external network `open-notebook_default` so RAG can reach `local-embeddings` without publishing its port.
- `ENABLE_PERSISTENT_CONFIG=false` keeps Scout / embed / SearXNG settings driven by compose.
- Do not point embeddings at Scout; Scout is chat-only here.
