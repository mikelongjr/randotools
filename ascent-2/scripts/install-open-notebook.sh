#!/usr/bin/env bash
# Install Open Notebook stack from this recipe onto a host (run as root on ascent-2).
set -euo pipefail

RECIPE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${OPEN_NOTEBOOK_ROOT:-/opt/open-notebook}"
TS_IP="$(tailscale ip -4)"

mkdir -p "$DEST" "$DEST/bin" "$DEST/patches"
cp -a "$RECIPE_ROOT/open-notebook/embeddings" "$DEST/"
cp -a "$RECIPE_ROOT/open-notebook/backup" "$DEST/"
cp -a "$RECIPE_ROOT/open-notebook/patches/." "$DEST/patches/"
cp "$RECIPE_ROOT/open-notebook/Dockerfile" "$DEST/Dockerfile"
cp "$RECIPE_ROOT/open-notebook/docker-compose.yml" "$DEST/docker-compose.yml"

if [[ ! -f "$DEST/.env" ]]; then
  cp "$RECIPE_ROOT/open-notebook/.env.example" "$DEST/.env"
  python3 - "$DEST/.env" "$TS_IP" <<'PY'
from pathlib import Path
import secrets, sys
p = Path(sys.argv[1])
ts = sys.argv[2]
text = p.read_text()
text = text.replace("change-me-long-random-string", secrets.token_urlsafe(32))
text = text.replace("change-me-strong-password", secrets.token_urlsafe(18))
text = text.replace("change-me-db-password", secrets.token_urlsafe(18))
text = text.replace("TAILSCALE_IP=\n", f"TAILSCALE_IP={ts}\n")
if "TAILSCALE_IP=" not in text:
    text += f"\nTAILSCALE_IP={ts}\n"
p.write_text(text)
p.chmod(0o600)
print("Wrote", p)
PY
else
  if grep -q '^TAILSCALE_IP=' "$DEST/.env"; then
    sed -i -E "s/^TAILSCALE_IP=.*/TAILSCALE_IP=${TS_IP}/" "$DEST/.env"
  else
    echo "TAILSCALE_IP=${TS_IP}" >> "$DEST/.env"
  fi
fi

# Export for compose variable substitution
set -a
# shellcheck disable=SC1091
source "$DEST/.env"
set +a
export TAILSCALE_IP="${TAILSCALE_IP:-$TS_IP}"

cd "$DEST"
docker compose up -d --build

echo "Waiting for Speaches..."
for i in $(seq 1 30); do
  if docker exec speaches curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then break; fi
  sleep 2
done

docker compose exec -T speaches uv tool run speaches-cli model download speaches-ai/Kokoro-82M-v1.0-ONNX || true
docker compose exec -T speaches uv tool run speaches-cli model download Systran/faster-whisper-small || true

# local backup timer
install -m 0750 "$RECIPE_ROOT/open-notebook/backup/backup-open-notebook-local.sh" "$DEST/bin/backup-open-notebook-local.sh"
install -m 0644 /dev/stdin /etc/systemd/system/open-notebook-backup-local.service << EOF
[Unit]
Description=Local Open Notebook backup on ascent-2
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=${DEST}/bin/backup-open-notebook-local.sh
Nice=10
EOF
install -m 0644 /dev/stdin /etc/systemd/system/open-notebook-backup-local.timer << 'EOF'
[Unit]
Description=Nightly local Open Notebook backup on ascent-2

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true
RandomizedDelaySec=5m

[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now open-notebook-backup-local.timer

cat > "$DEST/CREDENTIALS.txt" << EOF
Open Notebook on $(hostname)
=========================
URL (Tailscale):  http://ascent-2:8502
URL (IP):         http://${TS_IP}:8502
API:              http://ascent-2:5055
Password:         (see .env OPEN_NOTEBOOK_PASSWORD)
Encryption key:   (see .env — do not lose)

Chat:       llama4-scout @ host.docker.internal:8000/v1
Embedding:  bge-small-en-v1.5 @ local-embeddings:8080/v1
TTS:        speaches-ai/Kokoro-82M-v1.0-ONNX @ speaches:8000/v1
STT:        Systran/faster-whisper-small @ speaches:8000/v1
EOF
chmod 600 "$DEST/CREDENTIALS.txt" "$DEST/.env"

echo
echo "Stack up. Next: export OPEN_NOTEBOOK_PASSWORD from .env and run:"
echo "  $RECIPE_ROOT/scripts/configure-open-notebook-models.sh"
