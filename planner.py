#!/usr/bin/env python3
"""
planner.py  --  the brain of llmbox.

Given a model spec (a friendly preset name OR a raw HF repo) and the machine it is
running on, this:

  1. detects the hardware (GPUs, VRAM, CPU cores, sockets/NUMA, system RAM),
  2. resolves the model + quant (presets -> repo+quant, or raw repo+quant),
  3. picks the right GGUF file(s) from the repo (handles sharded GGUFs),
  4. reads the GGUF header to get the *exact* per-tensor sizes,
  5. computes an optimised llama-server launch:
        - dense models: full GPU offload, or layer-split across GPUs,
        - MoE models: keep attention + KV + as many expert layers as fit on the
          GPU, spill ONLY the overflow experts to system RAM, and set threads /
          NUMA / batch / mlock so CPU-side expert compute is not left crippled.

It emits a single JSON object on stdout that launch.sh consumes. All human-readable
logging goes to stderr so stdout stays machine-parseable.

Nothing here is provider-specific. A GPU is a GPU whether it is on vast.ai, AWS or
a kid's box in Hanoi.
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import urllib.request
import urllib.error

# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #

def log(*a):
    print("[planner]", *a, file=sys.stderr, flush=True)


def die(msg, code=2):
    log("FATAL:", msg)
    sys.exit(code)


# --------------------------------------------------------------------------- #
# presets  --  friendly name -> (repo, quant). The advanced path bypasses these
# entirely by passing a raw repo + --quant.
# --------------------------------------------------------------------------- #

PRESETS = {
    # name            repo                                              quant
    "fast":      ("unsloth/Qwen3-4B-Instruct-2507-GGUF",               "Q4_K_M"),
    "balanced":  ("unsloth/Qwen3-14B-GGUF",                            "Q4_K_M"),
    "coder":     ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",        "Q4_K_M"),
    "quality":   ("unsloth/gpt-oss-120b-GGUF",                        "Q4_K_M"),
}

DEFAULT_QUANT = "Q4_K_M"   # the 4-bit default; auto-fit may downshift.

# --------------------------------------------------------------------------- #
# GGUF parsing  --  we only ever read the header (metadata + tensor info table),
# never the tensor payload, so this is cheap and can run over an HTTP range read
# before committing to a multi-GB download.
# --------------------------------------------------------------------------- #

# ggml type -> (block_size_in_elements, bytes_per_block)
GGML_TYPE_SIZE = {
    0:  (1, 4),    # F32
    1:  (1, 2),    # F16
    2:  (32, 18),  # Q4_0
    3:  (32, 20),  # Q4_1
    6:  (32, 22),  # Q5_0
    7:  (32, 24),  # Q5_1
    8:  (32, 34),  # Q8_0
    9:  (32, 40),  # Q8_1
    10: (256, 84),    # Q2_K
    11: (256, 110),   # Q3_K
    12: (256, 144),   # Q4_K
    13: (256, 176),   # Q5_K
    14: (256, 210),   # Q6_K
    15: (256, 292),   # Q8_K
    16: (256, 66),    # IQ2_XXS
    17: (256, 74),    # IQ2_XS
    18: (256, 98),    # IQ3_XXS
    19: (256, 50),    # IQ1_S
    20: (32, 18),     # IQ4_NL
    21: (256, 110),   # IQ3_S
    22: (256, 82),    # IQ2_S
    23: (256, 136),   # IQ4_XS
    24: (1, 1),    # I8
    25: (1, 2),    # I16
    26: (1, 4),    # I32
    27: (1, 8),    # I64
    28: (1, 8),    # F64
    29: (256, 56),    # IQ1_M
    30: (1, 2),    # BF16
}

# GGUF metadata value-type ids
(GT_U8, GT_I8, GT_U16, GT_I16, GT_U32, GT_I32, GT_F32, GT_BOOL,
 GT_STR, GT_ARR, GT_U64, GT_I64, GT_F64) = range(13)


class _Source:
    """Random-access byte source over either a local file or a remote URL.

    For URLs we lazily pull data with HTTP Range requests and cache it, growing
    the buffer as the parser seeks forward. The GGUF header for even very large
    models is at most a few MB, so this stays small.
    """

    CHUNK = 4 * 1024 * 1024  # 4 MiB per remote fetch

    def __init__(self, src):
        self.url = None
        self.fh = None
        self.buf = bytearray()
        self.pos = 0
        if src.startswith("http://") or src.startswith("https://"):
            self.url = src
            self._fetch(0, self.CHUNK)
        else:
            self.fh = open(src, "rb")

    def _fetch(self, start, length):
        end = start + length - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if start == len(self.buf):
            self.buf.extend(data)
        elif start < len(self.buf):
            # overlap; extend tail only
            self.buf.extend(data[len(self.buf) - start:])
        else:
            raise IOError("non-contiguous range fetch")

    def _ensure(self, upto):
        if self.fh is not None:
            return
        while len(self.buf) < upto:
            self._fetch(len(self.buf), max(self.CHUNK, upto - len(self.buf)))

    def read(self, n):
        if self.fh is not None:
            return self.fh.read(n)
        self._ensure(self.pos + n)
        out = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return out

    def close(self):
        if self.fh:
            self.fh.close()


class GGUFHeader:
    def __init__(self, source_path):
        self.meta = {}
        self.tensors = []   # list of dict: name, dims, type, nbytes
        self._s = _Source(source_path)
        try:
            self._parse()
        finally:
            self._s.close()

    # -- primitive readers --------------------------------------------------- #
    def _u32(self): return struct.unpack("<I", self._s.read(4))[0]
    def _u64(self): return struct.unpack("<Q", self._s.read(8))[0]

    def _str(self):
        n = self._u64()
        return self._s.read(n).decode("utf-8", "replace")

    def _read_value(self, vtype):
        if vtype == GT_U8:  return struct.unpack("<B", self._s.read(1))[0]
        if vtype == GT_I8:  return struct.unpack("<b", self._s.read(1))[0]
        if vtype == GT_U16: return struct.unpack("<H", self._s.read(2))[0]
        if vtype == GT_I16: return struct.unpack("<h", self._s.read(2))[0]
        if vtype == GT_U32: return struct.unpack("<I", self._s.read(4))[0]
        if vtype == GT_I32: return struct.unpack("<i", self._s.read(4))[0]
        if vtype == GT_F32: return struct.unpack("<f", self._s.read(4))[0]
        if vtype == GT_BOOL: return struct.unpack("<?", self._s.read(1))[0]
        if vtype == GT_U64: return self._u64()
        if vtype == GT_I64: return struct.unpack("<q", self._s.read(8))[0]
        if vtype == GT_F64: return struct.unpack("<d", self._s.read(8))[0]
        if vtype == GT_STR: return self._str()
        if vtype == GT_ARR:
            sub = self._u32()
            count = self._u64()
            return [self._read_value(sub) for _ in range(count)]
        raise ValueError(f"unknown gguf value type {vtype}")

    def _parse(self):
        magic = self._s.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file (bad magic)")
        version = self._u32()
        if version not in (2, 3):
            log(f"warning: GGUF version {version} (only 2/3 tested), parsing anyway")
        n_tensors = self._u64()
        n_kv = self._u64()

        for _ in range(n_kv):
            key = self._str()
            vtype = self._u32()
            self.meta[key] = self._read_value(vtype)

        for _ in range(n_tensors):
            name = self._str()
            ndim = self._u32()
            dims = [self._u64() for _ in range(ndim)]
            ttype = self._u32()
            _offset = self._u64()
            nbytes = self._tensor_bytes(dims, ttype)
            self.tensors.append({"name": name, "dims": dims, "type": ttype, "nbytes": nbytes})

    @staticmethod
    def _tensor_bytes(dims, ttype):
        n = 1
        for d in dims:
            n *= d
        blck, tsize = GGML_TYPE_SIZE.get(ttype, (1, 2))  # default ~F16 if unknown
        if blck <= 0:
            return n * tsize
        return (n // blck) * tsize

    # -- convenience accessors ---------------------------------------------- #
    def arch(self):
        return self.meta.get("general.architecture", "llama")

    def _ak(self, suffix, default=None):
        return self.meta.get(f"{self.arch()}.{suffix}", default)

    def n_layers(self):  return int(self._ak("block_count", 0) or 0)
    def n_expert(self):  return int(self._ak("expert_count", 0) or 0)
    def n_embd(self):    return int(self._ak("embedding_length", 0) or 0)
    def n_head(self):    return int(self._ak("attention.head_count", 0) or 0)
    def ctx_train(self): return int(self._ak("context_length", 0) or 0)

    def n_head_kv(self):
        v = self._ak("attention.head_count_kv", None)
        if v is None:
            return self.n_head()
        return int(v)

    def head_dim(self):
        kld = self._ak("attention.key_length", None)
        if kld:
            return int(kld)
        h = self.n_head()
        return (self.n_embd() // h) if h else 128

    def is_moe(self):
        return self.n_expert() > 0

    def expert_bytes_by_layer(self):
        """Map layer index -> total bytes of that layer's expert FFN tensors."""
        by_layer = {}
        pat = re.compile(r"blk\.(\d+)\.ffn_(?:gate|up|down)_exps")
        for t in self.tensors:
            m = pat.match(t["name"])
            if m:
                li = int(m.group(1))
                by_layer[li] = by_layer.get(li, 0) + t["nbytes"]
        return by_layer

    def total_bytes(self):
        return sum(t["nbytes"] for t in self.tensors)


# --------------------------------------------------------------------------- #
# hardware detection
# --------------------------------------------------------------------------- #

def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return ""


def detect_gpus():
    """Return list of dicts: {vendor, name, vram_total_mb, vram_free_mb}."""
    gpus = []
    out = _run(["nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits"])
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"vendor": "nvidia", "name": parts[0],
                         "vram_total_mb": int(float(parts[1])),
                         "vram_free_mb": int(float(parts[2]))})
    if gpus:
        return gpus

    # AMD best-effort
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    for line in out.splitlines():
        if "card" in line.lower() and "," in line:
            cols = line.split(",")
            try:
                total = int(cols[1]) // (1024 * 1024)
                gpus.append({"vendor": "amd", "name": "AMD GPU",
                             "vram_total_mb": total, "vram_free_mb": total})
            except Exception:
                pass
    return gpus


def detect_nvlink(n_gpus):
    if n_gpus < 2:
        return False
    out = _run(["nvidia-smi", "nvlink", "-s"])
    return "Link" in out and "inactive" not in out.lower()


def detect_cpu():
    info = {"physical_cores": os.cpu_count() or 4, "sockets": 1, "numa_nodes": 1}
    out = _run(["lscpu"])
    cps = sock = None
    for line in out.splitlines():
        if line.startswith("Core(s) per socket:"):
            cps = int(line.split(":")[1].strip())
        elif line.startswith("Socket(s):"):
            sock = int(line.split(":")[1].strip())
        elif line.startswith("NUMA node(s):"):
            info["numa_nodes"] = int(line.split(":")[1].strip())
    if cps and sock:
        info["physical_cores"] = cps * sock
        info["sockets"] = sock
    return info


def detect_system_ram_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 16000


def detect_hardware():
    gpus = detect_gpus()
    cpu = detect_cpu()
    hw = {
        "gpus": gpus,
        "n_gpus": len(gpus),
        "vram_total_mb": sum(g["vram_total_mb"] for g in gpus),
        "nvlink": detect_nvlink(len(gpus)),
        "cpu": cpu,
        "ram_mb": detect_system_ram_mb(),
    }
    return hw


# --------------------------------------------------------------------------- #
# HF repo file resolution
# --------------------------------------------------------------------------- #

def hf_list_gguf(repo):
    """Return [(filename, size_bytes), ...] for .gguf files in the repo."""
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    token = os.environ.get("HF_TOKEN")
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        die(f"could not query HF repo '{repo}': HTTP {e.code}")
    except Exception as e:
        die(f"could not query HF repo '{repo}': {e}")
    files = []
    for s in data.get("siblings", []):
        fn = s.get("rfilename", "")
        if fn.lower().endswith(".gguf"):
            files.append((fn, s.get("size") or 0))
    if not files:
        die(f"no .gguf files found in repo '{repo}'")
    return files


SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def quant_of(filename):
    """Pull the quant token out of a gguf filename, e.g. Q4_K_M, IQ4_XS, BF16."""
    base = SHARD_RE.sub(".gguf", filename)
    m = re.search(r"(IQ\d+\w*|Q\d+\w*|BF16|F16|F32)", base, re.IGNORECASE)
    return m.group(1).upper() if m else "?"


def select_files(files, requested_quant):
    """Return (list_of_filenames_to_download, first_shard_name) for the quant."""
    wanted = requested_quant.upper()
    matches = [(fn, sz) for fn, sz in files if quant_of(fn) == wanted]
    if not matches:
        avail = sorted({quant_of(fn) for fn, _ in files})
        return None, avail
    # group shards
    shard_groups = {}
    singles = []
    for fn, sz in matches:
        m = SHARD_RE.search(fn)
        if m:
            key = SHARD_RE.sub("", fn)
            shard_groups.setdefault(key, []).append((fn, sz))
        else:
            singles.append((fn, sz))
    if shard_groups:
        # take the group with the most shards (there should be one)
        key = max(shard_groups, key=lambda k: len(shard_groups[k]))
        group = sorted(shard_groups[key], key=lambda x: x[0])
        names = [fn for fn, _ in group]
        return names, names[0]
    # single-file quant
    names = [singles[0][0]]
    return names, names[0]


# --------------------------------------------------------------------------- #
# fit math
# --------------------------------------------------------------------------- #

KV_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
COMPUTE_OVERHEAD_MB = 1024          # CUDA context + compute buffers, per process
VRAM_SAFETY_FRAC = 0.93             # never plan to use 100% of VRAM


def kv_cache_mb(hdr, ctx, kv_quant):
    per = KV_BYTES.get(kv_quant, 2.0)
    # K and V, both quantised the same here for simplicity
    bytes_ = 2 * hdr.n_layers() * hdr.n_head_kv() * hdr.head_dim() * ctx * per
    return int(bytes_ / (1024 * 1024))


def usable_vram_mb(hw):
    return int(hw["vram_total_mb"] * VRAM_SAFETY_FRAC)


# --------------------------------------------------------------------------- #
# planners
# --------------------------------------------------------------------------- #

def plan_dense(hdr, hw, ctx, kv_quant, weights_mb):
    args = ["-ngl", "99", "-fa", "-c", str(ctx), "--mlock"]
    kv_mb = kv_cache_mb(hdr, ctx, kv_quant)
    need = weights_mb + kv_mb + COMPUTE_OVERHEAD_MB
    vram = usable_vram_mb(hw)
    notes = []

    if hw["n_gpus"] >= 2:
        args += ["--split-mode", "layer"]      # consumer multi-GPU = PCIe, layer split
        if hw["nvlink"]:
            args = [a for a in args if a not in ("layer",)]
            args += ["--split-mode", "row"]    # fast interconnect -> tensor parallel
            notes.append("NVLink detected: using row split (tensor parallel).")
        else:
            notes.append("No NVLink: using layer split to avoid PCIe bottleneck.")
        # weight tensor-split by per-GPU VRAM
        split = ",".join(str(g["vram_total_mb"]) for g in hw["gpus"])
        args += ["--tensor-split", split]

    if need > vram and hw["n_gpus"] <= 1:
        notes.append(f"WARNING: needs ~{need} MB, GPU has ~{vram} MB usable. "
                     f"Dense model will partially offload to CPU and be slow.")
    if kv_quant != "f16":
        args += ["--cache-type-k", kv_quant, "--cache-type-v", kv_quant]
    return args, {"kind": "dense", "kv_mb": kv_mb, "need_mb": need,
                  "vram_mb": vram, "notes": notes}


def plan_moe(hdr, hw, ctx, kv_quant, weights_mb):
    """Keep attention+KV+as many expert layers as fit on GPU; spill the rest to RAM."""
    notes = []
    by_layer = hdr.expert_bytes_by_layer()
    if not by_layer:
        notes.append("MoE flagged but no expert tensors located; treating as dense.")
        return plan_dense(hdr, hw, ctx, kv_quant, weights_mb)

    expert_total = sum(by_layer.values())
    expert_total_mb = expert_total / (1024 * 1024)
    dense_part_mb = weights_mb - expert_total_mb           # attention/embd/output/shared
    kv_mb = kv_cache_mb(hdr, ctx, kv_quant)
    vram = usable_vram_mb(hw)

    # VRAM available for experts, after the always-on-GPU parts
    expert_budget_mb = vram - dense_part_mb - kv_mb - COMPUTE_OVERHEAD_MB
    if expert_budget_mb < 0:
        notes.append("WARNING: attention+KV alone exceed VRAM; reduce context.")
        expert_budget_mb = 0

    # Greedily keep whole layers' experts on GPU (sorted by index), spill the rest.
    layers = sorted(by_layer.items())            # [(layer_idx, bytes), ...]
    kept, spilled, used = [], [], 0.0
    for li, b in layers:
        b_mb = b / (1024 * 1024)
        if used + b_mb <= expert_budget_mb:
            kept.append(li)
            used += b_mb
        else:
            spilled.append(li)

    spilled_mb = sum(by_layer[li] for li in spilled) / (1024 * 1024)

    args = ["-ngl", "99", "-fa", "-c", str(ctx),
            "--no-mmap", "--mlock",                       # fully resident, no fault stalls
            "-t", str(hw["cpu"]["physical_cores"]),
            "--threads-batch", str(hw["cpu"]["physical_cores"]),
            "-ub", "512", "-b", "2048"]

    if kv_quant != "f16":
        args += ["--cache-type-k", kv_quant, "--cache-type-v", kv_quant]

    # Build an explicit -ot regex spilling exactly the chosen layers to CPU.
    # Explicit indices avoid version-dependent --n-cpu-moe semantics.
    if spilled:
        if len(kept) == 0:
            regex = r"ffn_(?:gate|up|down)_exps\.weight=CPU"
            notes.append("All experts on CPU/RAM (none fit in VRAM).")
        else:
            idx = "|".join(str(i) for i in spilled)
            regex = rf"blk\.(?:{idx})\.ffn_(?:gate|up|down)_exps\.weight=CPU"
            notes.append(f"{len(kept)} expert layers on GPU, "
                         f"{len(spilled)} spilled to RAM (~{int(spilled_mb)} MB).")
        args += ["-ot", regex]
    else:
        notes.append("All experts fit on GPU.")

    # NUMA handling for multi-socket boxes (big win on dual-socket servers).
    launch_prefix = []
    if hw["cpu"]["numa_nodes"] > 1:
        args += ["--numa", "distribute"]
        launch_prefix = ["numactl", "--interleave=all"]
        notes.append(f"{hw['cpu']['numa_nodes']} NUMA nodes: interleaving memory.")

    # system RAM check
    ram_need_mb = spilled_mb + 2048
    if ram_need_mb > hw["ram_mb"]:
        notes.append(f"WARNING: need ~{int(ram_need_mb)} MB system RAM for spilled "
                     f"experts, box has ~{hw['ram_mb']} MB. Will swap/fail.")

    meta = {"kind": "moe", "n_expert": hdr.n_expert(),
            "expert_layers_on_gpu": len(kept), "expert_layers_on_cpu": len(spilled),
            "spilled_mb": int(spilled_mb), "kv_mb": kv_mb,
            "ram_need_mb": int(ram_need_mb), "notes": notes,
            "launch_prefix": launch_prefix}
    return args, meta


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #

def download(repo, filenames, dest):
    os.makedirs(dest, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        die("huggingface_hub not installed (pip install huggingface_hub)")
    local_first = None
    for fn in filenames:
        log(f"downloading {fn} ...")
        p = hf_hub_download(repo_id=repo, filename=fn, local_dir=dest,
                            local_dir_use_symlinks=False)
        if local_first is None:
            local_first = p
    return local_first


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def resolve_spec(model, quant):
    if model in PRESETS:
        repo, pq = PRESETS[model]
        return repo, (quant or pq)
    # raw repo path
    return model, (quant or DEFAULT_QUANT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="preset name or HF repo id")
    ap.add_argument("--quant", default=None, help="quant token e.g. Q4_K_M")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--kv-quant", default="f16", choices=list(KV_BYTES.keys()))
    ap.add_argument("--auto-fit", action="store_true",
                    help="if requested quant won't fit, drop to the largest that does")
    ap.add_argument("--models-dir", default="/workspace/models")
    ap.add_argument("--no-download", action="store_true",
                    help="plan only using a remote header read; do not download")
    args = ap.parse_args()

    repo, quant = resolve_spec(args.model, args.quant)
    log(f"model repo = {repo}  quant = {quant}")

    hw = detect_hardware()
    log(f"hardware: {hw['n_gpus']} GPU(s), {hw['vram_total_mb']} MB VRAM, "
        f"{hw['cpu']['physical_cores']} cores / {hw['cpu']['sockets']} socket(s), "
        f"{hw['ram_mb']} MB RAM, nvlink={hw['nvlink']}")
    if hw["n_gpus"] == 0:
        log("WARNING: no GPU detected; this will run on CPU only.")

    files = hf_list_gguf(repo)

    # quant selection (+ optional auto-fit downshift)
    names, first = select_files(files, quant)
    if names is None:
        avail = first
        if args.auto_fit and avail:
            # crude: pick the largest-named quant that fits VRAM by file size
            def fsize(q):
                return max((sz for fn, sz in files if quant_of(fn) == q), default=0)
            ordered = sorted(avail, key=fsize, reverse=True)
            vram_b = usable_vram_mb(hw) * 1024 * 1024
            pick = next((q for q in ordered if fsize(q) < vram_b), ordered[-1])
            log(f"requested quant unavailable; auto-fit chose {pick}")
            names, first = select_files(files, pick)
            quant = pick
        else:
            die(f"quant '{quant}' not in repo. Available: {', '.join(avail)}")

    weights_bytes = sum(sz for fn, sz in files if fn in names)
    weights_mb = weights_bytes / (1024 * 1024)

    # read GGUF header: remote (range) for plan-only, else after download
    if args.no_download:
        url = f"https://huggingface.co/{repo}/resolve/main/{first}"
        log("reading GGUF header via remote range request ...")
        hdr = GGUFHeader(url)
        model_path = None
    else:
        model_path = download(repo, names, args.models_dir)
        hdr = GGUFHeader(model_path)

    # cap context to model's trained context
    ctx = args.context
    if hdr.ctx_train() and ctx > hdr.ctx_train():
        log(f"capping context {ctx} -> model max {hdr.ctx_train()}")
        ctx = hdr.ctx_train()

    if hdr.is_moe():
        log(f"MoE detected: {hdr.n_expert()} experts, {hdr.n_layers()} layers")
        srv_args, meta = plan_moe(hdr, hw, ctx, args.kv_quant, weights_mb)
    else:
        log(f"dense model: {hdr.n_layers()} layers")
        srv_args, meta = plan_dense(hdr, hw, ctx, args.kv_quant, weights_mb)

    for n in meta.get("notes", []):
        log("  -", n)

    result = {
        "repo": repo,
        "quant": quant,
        "model_path": model_path,
        "first_shard": first,
        "context": ctx,
        "server_args": srv_args,
        "launch_prefix": meta.get("launch_prefix", []),
        "plan": meta,
        "hardware": hw,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
