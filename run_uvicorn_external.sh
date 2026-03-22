#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
APP="${APP:-server.app:app}"

# Allow explicit override if needed
HOST_IP="${HOST_IP:-}"

if [[ -z "$HOST_IP" ]]; then
  HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi

if [[ -z "$HOST_IP" ]]; then
  HOST_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"
fi

if [[ -z "$HOST_IP" ]]; then
  echo "[ERROR] Impossibile determinare l'IP del server automaticamente."
  echo "Imposta HOST_IP manualmente, ad esempio: HOST_IP=192.168.1.50 ./run_uvicorn_external.sh"
  exit 1
fi

echo "[INFO] Avvio uvicorn su ${HOST_IP}:${PORT}"
echo "[INFO] URL: http://${HOST_IP}:${PORT}"

exec uvicorn "$APP" --host "$HOST_IP" --port "$PORT"
