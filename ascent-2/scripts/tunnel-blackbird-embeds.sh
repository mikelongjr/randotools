#!/usr/bin/env bash
# Tunnel blackbird CUDA Ollama embeds (:11435) onto ascent-2 for Open Notebook.
# Run on omenboy (has SSH to both hosts). Idempotent-ish.
set -euo pipefail

BB_SSH=(ssh -o BatchMode=yes -o IdentitiesOnly=yes -o IdentityAgent=none
  -i "${BLACKBIRD_SSH_KEY:-$HOME/.ssh/id_rsa}" -p "${BLACKBIRD_SSH_PORT:-44422}"
  "${BLACKBIRD_SSH_USER:-mplong}@${BLACKBIRD_SSH_HOST:-eatcenter.tplinkdns.com}")
ASCENT_SSH=(ssh -o BatchMode=yes root@"${ASCENT_HOST:-ascent-2}")

L_SOCK="${CONTROL_DIR:-/tmp}/bb-embed-l.sock"
R_SOCK="${CONTROL_DIR:-/tmp}/ascent2-embed-r.sock"

echo "Checking blackbird embed Ollama..."
"${BB_SSH[@]}" 'curl -sS -m 5 http://127.0.0.1:11435/api/version'

# Local forward: omenboy → blackbird:11435
if ! curl -sS -m 2 http://127.0.0.1:11435/api/version >/dev/null 2>&1; then
  ssh -o BatchMode=yes -o IdentitiesOnly=yes -o IdentityAgent=none \
    -o ExitOnForwardFailure=yes \
    -i "${BLACKBIRD_SSH_KEY:-$HOME/.ssh/id_rsa}" -p "${BLACKBIRD_SSH_PORT:-44422}" \
    -f -N -L 127.0.0.1:11435:127.0.0.1:11435 \
    -o ControlPath="$L_SOCK" -o ControlMaster=yes \
    "${BLACKBIRD_SSH_USER:-mplong}@${BLACKBIRD_SSH_HOST:-eatcenter.tplinkdns.com}"
  echo "Local forward up"
else
  echo "Local forward already up"
fi

# Reverse forward: omenboy → ascent-2:11435
if ! "${ASCENT_SSH[@]}" 'curl -sS -m 2 http://127.0.0.1:11435/api/version' >/dev/null 2>&1; then
  ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -f -N \
    -R 127.0.0.1:11435:127.0.0.1:11435 \
    -o ControlPath="$R_SOCK" -o ControlMaster=yes \
    root@"${ASCENT_HOST:-ascent-2}"
  echo "Reverse forward up"
else
  echo "Reverse forward already up"
fi

# Socat publish for Docker host-gateway (containers cannot hit host 127.0.0.1)
"${ASCENT_SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
command -v socat >/dev/null || apt-get install -y -qq socat
if ! ss -ltn | grep -q ':11436'; then
  nohup socat TCP-LISTEN:11436,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:11435 \
    >/var/log/embed-proxy.log 2>&1 &
  sleep 1
fi
curl -sS -m 5 http://127.0.0.1:11436/api/version
REMOTE

echo
echo "Embed path ready: ascent-2 host.docker.internal:11436 → blackbird :11435"
