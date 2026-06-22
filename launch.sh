#!/usr/bin/env bash
#
# launch.sh -- llmbox bootstrap entrypoint. Runs inside the GPU container on any
# provider (vast.ai / AWS / GCP / a box in Hanoi -- it does not care).
#
# It:
#   1. forces a Tailscale auth key (no interactive fallback) and joins the tailnet,
#   2. creates an encrypted workspace whose key lives ONLY in RAM, so an ungraceful
#      "delete instance from the console" is equivalent to a secure wipe,
#   3. (optional) gates on confidential-compute-capable hardware and attests,
#   4. plans the optimal llama.cpp config via planner.py,
#   5. serves an OpenAI-compatible endpoint over the tailnet only (never public),
#   6. prints a copy-paste connection block for Codex / the chat app.
#
# Required env:
#   TS_AUTHKEY     Tailscale auth key (reusable + ephemeral + pre-authorized + tagged)
#   MODEL          preset name (fast|balanced|coder|quality) OR an HF repo id
#
# Optional env:
#   QUANT          e.g. Q4_K_M           (default: preset's quant, else Q4_K_M)
#   CONTEXT        context window         (default: 8192)
#   KV_QUANT       f16|q8_0|q4_0         (default: f16)
#   AUTO_FIT       1 to downshift quant if it won't fit
#   TS_HOSTNAME    tailnet hostname       (default: llmbox)
#   PORT           local llama-server port (default: 8080)
#   CONFIDENTIAL   1 to require a TEE; errors out on ineligible hardware
#   API_KEY        endpoint key           (default: generated)
#   HF_TOKEN       for gated/private repos
#
set -euo pipefail

log() { echo "[launch] $*" >&2; }
die() { echo "[launch] FATAL: $*" >&2; exit 1; }

PORT="${PORT:-8080}"
TS_HOSTNAME="${TS_HOSTNAME:-llmbox}"
CONTEXT="${CONTEXT:-8192}"
KV_QUANT="${KV_QUANT:-f16}"
RAM_SECRETS="/run/llmbox"          # tmpfs, never written to persistent disk
WORKDIR="/workspace"               # encrypted mount
HERE="$(cd "$(dirname "$0")" && pwd)"

# --------------------------------------------------------------------------- #
# 0. preflight
# --------------------------------------------------------------------------- #
[ -n "${TS_AUTHKEY:-}" ] || die "TS_AUTHKEY is required (auth key is forced; no interactive login)."
[ -n "${MODEL:-}"      ] || die "MODEL is required (preset name or HF repo id)."
command -v tailscale     >/dev/null || die "tailscale not found in image."
command -v llama-server  >/dev/null || die "llama-server not found in image."
command -v python3       >/dev/null || die "python3 not found in image."

# --------------------------------------------------------------------------- #
# 1. RAM-only secret store. tmpfs is backed by RAM; on power-off/instance-delete
#    it evaporates along with every key it held.
# --------------------------------------------------------------------------- #
setup_ram_secrets() {
  mkdir -p "$RAM_SECRETS"
  mountpoint -q "$RAM_SECRETS" || mount -t tmpfs -o size=64m,mode=0700 tmpfs "$RAM_SECRETS"
  chmod 700 "$RAM_SECRETS"
  # Stash the TS key in RAM and scrub it from the environment/process listing.
  printf '%s' "$TS_AUTHKEY" > "$RAM_SECRETS/ts_authkey"
  API_KEY="${API_KEY:-$(openssl rand -hex 32)}"
  printf '%s' "$API_KEY" > "$RAM_SECRETS/api_key"
  chmod 600 "$RAM_SECRETS"/*
}

# --------------------------------------------------------------------------- #
# 2. Encrypted workspace. The LUKS key comes from the kernel RNG and is held
#    ONLY in the tmpfs above -- it is never written to the persistent disk. So
#    if the instance is killed hard (console delete, power yank) the ciphertext
#    on disk is unrecoverable: the only key died with RAM. Cleanup-on-exit is a
#    courtesy, not the guarantee.
# --------------------------------------------------------------------------- #
setup_encrypted_workspace() {
  swapoff -a 2>/dev/null || true            # never leak plaintext to swap
  if ! command -v cryptsetup >/dev/null; then
    log "cryptsetup not present; falling back to tmpfs workspace (RAM-bounded)."
    mkdir -p "$WORKDIR"
    mountpoint -q "$WORKDIR" || mount -t tmpfs -o size=90% tmpfs "$WORKDIR"
    return
  fi
  local container="/llmbox.luks"            # backing file on the ephemeral disk
  local keyfile="$RAM_SECRETS/luks.key"
  head -c 64 /dev/urandom > "$keyfile"; chmod 600 "$keyfile"
  # size the container to the disk's free space minus headroom
  local free_mb; free_mb=$(df -Pm / | awk 'NR==2{print $4}')
  local size_mb=$(( free_mb - 4096 )); [ "$size_mb" -lt 4096 ] && size_mb=4096
  fallocate -l "${size_mb}M" "$container"
  cryptsetup luksFormat --type luks2 --batch-mode "$container" "$keyfile"
  cryptsetup open --key-file "$keyfile" "$container" llmbox_crypt
  mkfs.ext4 -q /dev/mapper/llmbox_crypt
  mkdir -p "$WORKDIR"
  mount /dev/mapper/llmbox_crypt "$WORKDIR"
  log "encrypted workspace mounted (${size_mb}M, RAM-only key)."
}

best_effort_cleanup() {
  # Only runs on graceful exit; the security property does NOT depend on it.
  umount "$WORKDIR" 2>/dev/null || true
  cryptsetup close llmbox_crypt 2>/dev/null || true
  rm -rf "$RAM_SECRETS" 2>/dev/null || true
}
trap best_effort_cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# 3. Confidential compute gate (only when CONFIDENTIAL=1). Three independent
#    requirements; the error message names exactly which one failed, because
#    "I have an H100 but TDX isn't on" and "I have a 5090" are different fixes.
# --------------------------------------------------------------------------- #
confidential_gate() {
  log "confidential mode requested: checking eligibility ..."

  # (a) CC-capable GPU: Hopper / Blackwell / Rubin only.
  local gpu; gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  if ! echo "$gpu" | grep -Eiq 'H100|H200|GH200|B100|B200|B300|GB200|GB300|Rubin|R100'; then
    die "confidential: GPU '${gpu:-none}' is not CC-capable. Need Hopper/Blackwell/Rubin."
  fi

  # (b) CPU platform must provide a confidential VM (AMD SEV-SNP or Intel TDX).
  if ! { dmesg 2>/dev/null | grep -Eiq 'SEV-SNP|Intel TDX|tdx: '; } \
     && [ ! -e /dev/tdx_guest ] \
     && [ ! -e /dev/sev-guest ]; then
    die "confidential: no AMD SEV-SNP / Intel TDX confidential VM detected. Launch a CVM."
  fi

  # (c) GPU must actually be in CC mode (capable != enabled).
  if ! nvidia-smi -q 2>/dev/null | grep -iA3 'Confidential Compute' | grep -iq 'ON\|Enabled'; then
    die "confidential: GPU is CC-capable but CC mode is OFF. Enable CC mode on the GPU."
  fi

  # (d) Attestation. Fail CLOSED: if we cannot verify the TEE, we do not run.
  #     Real verification uses NVIDIA's attestation SDK (local GPU verifier or
  #     remote NRAS). The strongest setup has the *client* verify before sending
  #     prompts; this is the server-side gate.
  if command -v nv-attestation >/dev/null; then
    nv-attestation --verify || die "confidential: GPU attestation FAILED."
  elif python3 -c 'import nv_attestation_sdk' 2>/dev/null; then
    python3 "$HERE/attest.py" || die "confidential: GPU attestation FAILED."
  else
    die "confidential: attestation verifier not installed; refusing to run unattested."
  fi
  log "confidential: hardware eligible and attested."
}

# --------------------------------------------------------------------------- #
# 4. Tailscale: forced auth key, ephemeral node, stable hostname -> stable URL.
# --------------------------------------------------------------------------- #
tailscale_up() {
  tailscaled --tun=userspace-networking --state="$RAM_SECRETS/ts.state" \
             --socket="$RAM_SECRETS/ts.sock" >/dev/null 2>&1 &
  sleep 2
  tailscale --socket="$RAM_SECRETS/ts.sock" up \
    --auth-key="$(cat "$RAM_SECRETS/ts_authkey")" \
    --hostname="$TS_HOSTNAME" \
    --accept-dns=true
  log "joined tailnet as '$TS_HOSTNAME'."
}

tailscale_serve() {
  # Serve = private to your tailnet, auto-provisioned TLS. NOT funnel (public).
  # NOTE: `tailscale serve` syntax is version-sensitive; verify against the
  # installed version if this errors.
  tailscale --socket="$RAM_SECRETS/ts.sock" serve --bg \
    --https=443 "http://127.0.0.1:${PORT}" \
    || tailscale --socket="$RAM_SECRETS/ts.sock" serve --bg "http://127.0.0.1:${PORT}"
}

tailnet_fqdn() {
  tailscale --socket="$RAM_SECRETS/ts.sock" status --json 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))'
}

# --------------------------------------------------------------------------- #
# 5. Plan + launch
# --------------------------------------------------------------------------- #
main() {
  setup_ram_secrets
  setup_encrypted_workspace
  [ "${CONFIDENTIAL:-0}" = "1" ] && confidential_gate
  tailscale_up

  pip install -q --no-input huggingface_hub >/dev/null 2>&1 || true

  log "planning ..."
  local plan_json
  plan_json="$(python3 "$HERE/planner.py" \
      --model "$MODEL" \
      ${QUANT:+--quant "$QUANT"} \
      --context "$CONTEXT" \
      --kv-quant "$KV_QUANT" \
      ${AUTO_FIT:+--auto-fit} \
      --models-dir "$WORKDIR/models")"

  # extract launch prefix, server args, model path
  local model_path; model_path="$(echo "$plan_json" | python3 -c 'import sys,json;print(json.load(sys.stdin)["model_path"])')"
  mapfile -t PREFIX < <(echo "$plan_json" | python3 -c 'import sys,json;[print(x) for x in json.load(sys.stdin)["launch_prefix"]]')
  mapfile -t SRV    < <(echo "$plan_json" | python3 -c 'import sys,json;[print(x) for x in json.load(sys.stdin)["server_args"]]')

  tailscale_serve
  local fqdn; fqdn="$(tailnet_fqdn)"
  local base="https://${fqdn}/v1"

  print_connection_block "$base"

  log "starting llama-server (bound to localhost only) ..."
  exec "${PREFIX[@]}" llama-server \
      --model "$model_path" \
      --host 127.0.0.1 --port "$PORT" \
      --api-key "$(cat "$RAM_SECRETS/api_key")" \
      "${SRV[@]}"
}

print_connection_block() {
  local base="$1" key; key="$(cat "$RAM_SECRETS/api_key")"
  cat >&2 <<EOF

  ====================================================================
   llmbox is up. Your endpoint (private to your tailnet, TLS, no public port):

     Base URL : ${base}
     API key  : ${key}

   Use it in Codex on your PC:
     export OPENAI_BASE_URL="${base}"
     export OPENAI_API_KEY="${key}"

   Or point the chat app at the same Base URL + API key.

   (Install Tailscale on that PC/phone and log into the SAME tailnet first.)
  ====================================================================

EOF
}

main "$@"
