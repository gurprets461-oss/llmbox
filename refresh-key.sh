#!/usr/bin/env bash
#
# refresh-key.sh -- rotate credentials on a running llmbox without a rebuild.
#
#   ./refresh-key.sh tailscale <NEW_TS_AUTHKEY>   re-auth the node with a new key
#   ./refresh-key.sh api                          mint a new endpoint API key
#
# The Tailscale auth key only matters at join time, but --force-reauth lets you
# rotate the node onto a freshly minted key (e.g. after revoking the old one in
# the admin console). New keys are written ONLY to the RAM secret store.
#
set -euo pipefail
RAM_SECRETS="/run/llmbox"
log() { echo "[refresh] $*" >&2; }
die() { echo "[refresh] FATAL: $*" >&2; exit 1; }

[ -d "$RAM_SECRETS" ] || die "no running llmbox found ($RAM_SECRETS missing)."

case "${1:-}" in
  tailscale)
    NEW="${2:-}"; [ -n "$NEW" ] || die "usage: refresh-key.sh tailscale <NEW_TS_AUTHKEY>"
    printf '%s' "$NEW" > "$RAM_SECRETS/ts_authkey"; chmod 600 "$RAM_SECRETS/ts_authkey"
    tailscale --socket="$RAM_SECRETS/ts.sock" up \
      --auth-key="$NEW" --force-reauth \
      --hostname="${TS_HOSTNAME:-llmbox}" --accept-dns=true
    log "tailscale re-authed onto new key. Revoke the old key in the admin console."
    ;;
  api)
    NEWKEY="$(openssl rand -hex 32)"
    printf '%s' "$NEWKEY" > "$RAM_SECRETS/api_key"; chmod 600 "$RAM_SECRETS/api_key"
    log "new API key minted: $NEWKEY"
    log "restart llama-server to apply (it reads the key at startup)."
    ;;
  *)
    die "usage: refresh-key.sh {tailscale <KEY> | api}"
    ;;
esac
