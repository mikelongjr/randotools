# DeepSeek Harness on omenboy → GX-10 Scout

Agent / coding harness UI on omenboy, inference on ascent-2 (GX-10 Spark) Scout.

| Piece | Endpoint |
|---|---|
| LLM | Scout on ascent-2 `http://100.70.176.65:8000/v1` (`llama4-scout`, 64k) |
| Install | `/space/deepseek-harness` (`@deepseek-ai/dsh`) |
| Home / settings | `$DSH_HOME` = `/home/mike/.dsh` |
| Workspace | `/space/dsh-workspace` |
| UI | Loopback `http://127.0.0.1:3080` |

## Deploy

```bash
# one-time package install (already on /space if present)
mkdir -p /space/deepseek-harness /space/dsh-workspace
cd /space/deepseek-harness
npm install @deepseek-ai/dsh@latest
npm rebuild koffi node-pty

# settings + placeholder key (vLLM is keyless but pi-ai still wants a bearer value)
mkdir -p /home/mike/.dsh
cp /opt/randotools/deepseek-harness/settings.yaml /home/mike/.dsh/settings.yaml
cp /opt/randotools/deepseek-harness/.env.example /home/mike/.dsh/.env
chmod 600 /home/mike/.dsh/settings.yaml /home/mike/.dsh/.env
chown -R mike:mike /home/mike/.dsh /space/deepseek-harness /space/dsh-workspace

sudo cp /opt/randotools/deepseek-harness/deepseek-harness.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now deepseek-harness.service
```

Access from another machine: `ssh -L 3080:127.0.0.1:3080 omenboy`, then open the URL printed by `journalctl -u deepseek-harness -n 20` (includes a one-time `?token=`).

## Exa web search

Plugin is installed in the web profile; `cordis.patch.yml` pins `searchProvider: exa` and disables DeepSeek search.

```bash
# put your key in the env file the unit already loads
$EDITOR /home/mike/.dsh/.env   # set EXA_API_KEY=…
sudo systemctl restart deepseek-harness
```

Copy updated patch after recipe edits: `cp /opt/randotools/deepseek-harness/cordis.patch.yml /home/mike/.dsh/profiles/web/`

## Notes

- Provider route id is `gx10-scout` (openai-completions). Compat flags keep Scout happy with `system` + `max_tokens`.
- Scout is multimodal; `input: [text, image]` is declared on the model.
- Web host schema allows only `127.0.0.1` or `0.0.0.0`. Default unit binds loopback. For Tailscale browser access, switch ExecStart to `--host 0.0.0.0 --trusted-host 100.92.114.10 --trusted-host 100.92.114.10:3080`.
- Unit runs `/usr/bin/node …/bin.js` (not the `dsh` shim) because `/space` is SELinux `unlabeled_t` and systemd cannot `Exec` those files directly.
