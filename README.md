# llmbox

Spin up any GGUF model on any rented GPU, get a private OpenAI-compatible endpoint
you can point Codex or a chat app at. Provider-agnostic (vast.ai / AWS / GCP / a
box in Hanoi — a GPU is a GPU). Beginner-simple on the surface, hand-tuned underneath.

## Run it

One command inside the GPU container:

```bash
TS_AUTHKEY=tskey-auth-xxxx MODEL=coder ./launch.sh
```

`MODEL` is either a **preset** (`fast`, `balanced`, `coder`, `quality`) or a raw HF
repo id for the advanced path:

```bash
TS_AUTHKEY=tskey-auth-xxxx \
MODEL=yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF \
QUANT=Q4_K_M ./launch.sh
```

When it's up it prints a connection block:

```
Base URL : https://llmbox.<your-tailnet>.ts.net/v1
API key  : <generated>
```

Install Tailscale on your PC/phone, log into the **same tailnet**, and point Codex
(`OPENAI_BASE_URL` + `OPENAI_API_KEY`) or the chat app at that Base URL. Same fixed
hostname every spin-up, so the URL is stable even though the VM is disposable.

## Env

| var | default | meaning |
|-----|---------|---------|
| `TS_AUTHKEY` | — (required) | Tailscale auth key — **reusable + ephemeral + pre-authorized + tagged** |
| `MODEL` | — (required) | preset name or HF repo id |
| `QUANT` | preset's / `Q4_K_M` | quant token, e.g. `Q4_K_M`, `IQ4_XS` |
| `AUTO_FIT` | off | downshift quant if the chosen one won't fit |
| `CONTEXT` | `8192` | context window (capped to the model's trained max) |
| `KV_QUANT` | `f16` | `f16` / `q8_0` / `q4_0` to stretch context |
| `TS_HOSTNAME` | `llmbox` | tailnet hostname → stable URL |
| `CONFIDENTIAL` | off | require a TEE; errors out on ineligible hardware |
| `HF_TOKEN` | — | for gated/private repos |

Rotate creds on a live box: `./refresh-key.sh tailscale <NEW_KEY>` or `./refresh-key.sh api`.

## How models are placed (the part that matters)

**Dense** models offload fully to GPU (`-ngl 99 -fa`). Across two consumer cards
(no NVLink) it uses layer split to avoid the PCIe bottleneck; with NVLink it
switches to row split (tensor parallel).

**MoE** models are not run with the naive "dump all experts to CPU" config that
leaves ~2x on the table. The planner reads the GGUF header for exact per-tensor
sizes and:

- keeps attention, KV cache, and **as many whole expert layers as fit** on the GPU,
  spilling only the overflow via an explicit `-ot` regex (deterministic, not the
  version-dependent `--n-cpu-moe` semantics);
- sets threads to **physical** core count (`-t` / `--threads-batch`);
- enables `--numa distribute` + `numactl --interleave=all` on multi-socket boxes,
  where cross-socket bandwidth is otherwise the killer;
- loads weights fully resident (`--no-mmap --mlock`) so cold experts don't fault
  mid-generation;
- keeps flash attention on and KV on GPU.

That's how a 100B-A10B MoE runs usefully on a single 32 GB card plus system RAM:
only the ~10B active experts compute per token, against RAM-resident weights, so
there's no per-token weight streaming over PCIe — CPU memory bandwidth is the limit.

## Privacy model (be honest about the wall)

The host owns the box, so transport security ≠ host security. Two tiers:

**Baseline (always on).** Endpoint bound to localhost, reached only over your
WireGuard tailnet (no public port), TLS, API key. The workspace is LUKS-encrypted
with a key generated from the kernel RNG and held **only in RAM** — never on the
persistent disk. Swap is off. So deleting the instance ungracefully from the
console is equivalent to a secure wipe: the only key dies with RAM, and whatever
the next renter reads off the disk is noise. Cleanup-on-exit exists but nothing
depends on it. This defeats the network and leaves nothing recoverable at rest. It
does **not** hide live inference from a host dumping RAM/VRAM mid-request — if
someone's doing that, you have bigger problems than your endpoint.

**Confidential (`CONFIDENTIAL=1`).** The only config that closes the live-memory
gap. Gated on three independent requirements, with the failing one named:
a CC-capable GPU (Hopper/Blackwell/Rubin), a confidential VM (AMD SEV-SNP or Intel
TDX), and CC mode actually enabled — then attestation, **failing closed** if the
TEE can't be verified.

## Files

- `planner.py` — hardware detection, GGUF header parser, quant/shard selection, dense + MoE placement math. Pure logic, emits JSON.
- `launch.sh` — bootstrap: RAM-keyed encrypted storage, Tailscale up + serve, confidential gate, llama-server launch, connection block.
- `refresh-key.sh` — rotate the Tailscale auth key or the API key on a live VM.
- `test_planner.py` — offline self-tests for the parser and placement math.

## Notes / caveats

- The container is expected to ship `llama-server`, `python3`, `cryptsetup`, and
  `tailscale`. If `cryptsetup` is missing, the workspace falls back to a RAM-backed
  tmpfs (still no plaintext at rest, but bounded by RAM).
- `tailscale serve` syntax is version-sensitive; the script tries the current form
  then falls back.
- At the 8×H100/B200 tier, GGUF on llama.cpp is not throughput-optimal vs vLLM /
  SGLang / TensorRT-LLM. That tier is intentionally a stub here.
- Confidential attestation requires NVIDIA's verifier (`nv-attestation` or the
  `nv_attestation_sdk`) present in the image; without it, confidential mode refuses
  to run rather than running unattested.
