"""Produce the transient model directories that arms C2 and D are evaluated on.

Both arms answer the same question from opposite directions — does the
fine-tuning survive being quantized? — so both end in a single self-contained
quantized artifact rather than a base+adapter pair:

  C2: quantized base + adapter -> fuse (requantizes in place, since the base
      is already quantized) -> one quantized model carrying the fine-tuning.
  D:  bf16 base + bf16 adapter -> fuse -> a bf16 fine-tuned model, which is
      then quantized by each scheme's own method, exactly as models/download.py
      quantizes the un-tuned base.

These builds are large (a fused bf16 model is ~6 GB, each quantized variant
1.8-3.2 GB) and there are dozens of them across arms/schemes/seeds, so they
are written under MODELS_DIR/builds and deleted by the orchestrator once the
eval that needs them has finished. Nothing downstream reads them again: the
per-item JSONL is the artifact that persists.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from config import BUILDS_DIR, MODELS_DIR, SCHEMES, SEED


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _scheme(name: str):
    return next(s for s in SCHEMES if s.name == name)


def build_c2(task: str, scheme_name: str, seed: int, adapter_path: Path) -> Path:
    """Fuse a QLoRA adapter back into its quantized base.

    mlx_lm.fuse without --dequantize re-quantizes the merged weights into the
    base's existing format, which is precisely the effect under test: the
    fine-tuning gets rounded onto the same grid the base already lives on.
    """
    out = BUILDS_DIR / "C2" / task / f"seed{seed}" / scheme_name
    if (out / "config.json").exists():
        print(f"[build C2 {task}/{scheme_name}/seed{seed}] exists, reusing")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", str(MODELS_DIR / scheme_name),
        "--adapter-path", str(adapter_path),
        "--save-path", str(out),
    ])
    return out


def build_d_fused_bf16(task: str, seed: int, adapter_path: Path) -> Path:
    """Merge the bf16 adapter into the bf16 base — the input every D variant
    is quantized from.

    Built once per (task, seed) and reused across all four quantized schemes,
    so that the only thing differing between D's variants is the quantization
    method, matching how models/download.py derives every scheme from one base.
    """
    out = BUILDS_DIR / "D" / task / f"seed{seed}" / "fused_bf16"
    if (out / "config.json").exists():
        print(f"[build D {task}/seed{seed}] fused bf16 exists, reusing")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", str(MODELS_DIR / "bf16"),
        "--adapter-path", str(adapter_path),
        "--save-path", str(out),
    ])
    return out


def build_d_quantized(task: str, scheme_name: str, seed: int, fused_bf16: Path) -> Path:
    """Quantize the fine-tuned bf16 model with one scheme's own method.

    Mirrors models/download.py's dispatch (convert / awq / gptq) at the same
    bits and group size, with BASE_MODEL swapped for the fused model — so a
    D-arm int4 differs from the plain int4 in exactly one way: the weights it
    quantized were fine-tuned first.
    """
    scheme = _scheme(scheme_name)
    out = BUILDS_DIR / "D" / task / f"seed{seed}" / scheme_name
    if (out / "config.json").exists():
        print(f"[build D {task}/{scheme_name}/seed{seed}] exists, reusing")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    if scheme.method == "convert":
        _run([
            sys.executable, "-m", "mlx_lm", "convert",
            "--hf-path", str(fused_bf16), "--mlx-path", str(out), "-q",
            "--q-bits", str(scheme.bits), "--q-group-size", str(scheme.group_size),
        ])
    elif scheme.method in ("awq", "gptq"):
        _run([
            sys.executable, "-m", "mlx_lm", scheme.method,
            "--model", str(fused_bf16), "--mlx-path", str(out),
            "--bits", str(scheme.bits), "--group-size", str(scheme.group_size),
            "--seed", str(SEED),
        ])
    else:
        raise ValueError(f"cannot build arm D for method {scheme.method!r}")
    return out


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def discard(path: Path) -> None:
    """Delete a transient build once its eval has been recorded.

    Guarded to BUILDS_DIR so a bad path can never reach the real weights in
    MODELS_DIR (or anything else on disk).
    """
    path = path.resolve()
    builds_root = BUILDS_DIR.resolve()
    if not path.is_relative_to(builds_root):
        raise ValueError(f"refusing to delete {path}: outside {builds_root}")
    if path.exists():
        freed = dir_bytes(path)
        shutil.rmtree(path)
        print(f"[cleanup] removed {path.relative_to(builds_root)} ({freed/1e9:.2f} GB)")
