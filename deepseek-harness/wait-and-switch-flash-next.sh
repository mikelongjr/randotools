#!/usr/bin/env bash
set -uo pipefail
API=http://100.70.176.65:8000/v1/models
LOG=/space/dsh-workspace/flash-next-switch.log
mkdir -p /space/dsh-workspace
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiting for Flash-Next on $API"
while true; do
  if curl -fsS --connect-timeout 3 "$API" | grep -q qwen3.8-flash-next; then
    echo "[$(date -Is)] endpoint ready; switching DSH"
    cp /opt/randotools/deepseek-harness/settings.flash-next.yaml /home/mike/.dsh/settings.yaml
    chmod 600 /home/mike/.dsh/settings.yaml
    sudo systemctl restart deepseek-harness.service
    sleep 8
    systemctl is-active deepseek-harness.service
    echo "[$(date -Is)] DSH switched"
    exit 0
  fi
  echo "[$(date -Is)] not ready"
  sleep 60
done
