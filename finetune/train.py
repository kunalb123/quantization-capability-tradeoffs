"""Train one LoRA adapter with mlx_lm, under a budget fixed by config.py.

One training run is identified by (task, scheme, seed). The adapter it
produces is shared by two arms: C1 evaluates it unfused on the quantized
base, C2 fuses it back in and requantizes. Arm D reuses the *bf16* adapter
for every scheme, so it needs no training of its own — which is why the
tune-then-quantize arm is nearly free once C1 has run.

Every hyperparameter comes from config.py and none is per-scheme. That's
deliberate: a per-scheme LR (say) would make "int4 recovers less than bf16"
unfalsifiable, since it could always be blamed on tuning. The known cost of
that choice is that a single fixed LR may not be optimal for every scheme;
the LR sensitivity check in the run plan is what bounds that risk.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from config import (
    ADAPTERS_DIR, LORA_BATCH_SIZE, LORA_DROPOUT, LORA_FINE_TUNE_TYPE, LORA_ITERS,
    LORA_LR, LORA_MASK_PROMPT, LORA_MAX_SEQ_LENGTH, LORA_NUM_LAYERS, LORA_RANK,
    LORA_SCALE, MODELS_DIR,
)


def adapter_dir(task: str, scheme_name: str, seed: int) -> Path:
    return ADAPTERS_DIR / task / f"seed{seed}" / scheme_name


def _write_config(path: Path, seed: int) -> Path:
    """Emit the YAML mlx_lm.lora reads for the settings it has no CLI flag for.

    rank/scale/dropout are only reachable through the config file (they live
    in CONFIG_DEFAULTS["lora_parameters"], not argparse). Everything else is
    written here too so the file is a complete, archivable record of what
    produced this adapter — it's saved next to the weights.
    """
    cfg = {
        "fine_tune_type": LORA_FINE_TUNE_TYPE,
        "num_layers": LORA_NUM_LAYERS,
        "batch_size": LORA_BATCH_SIZE,
        "iters": LORA_ITERS,
        "learning_rate": LORA_LR,
        "max_seq_length": LORA_MAX_SEQ_LENGTH,
        "seed": seed,
        "lora_parameters": {"rank": LORA_RANK, "dropout": LORA_DROPOUT, "scale": LORA_SCALE},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return path


def train(task: str, scheme_name: str, seed: int, data_dir: Path) -> Path:
    """Train (or reuse) the adapter for one (task, scheme, seed).

    Resumable in the only way that matters for an unattended run: a completed
    adapter is detected and skipped, so a restart doesn't redo finished work.
    A run interrupted mid-training is *not* resumed from its checkpoint — the
    partial adapter is discarded and training restarts, because resuming from
    an arbitrary interrupted step would break the "identical budget across
    schemes" guarantee that makes the comparison valid at all.
    """
    out_dir = adapter_dir(task, scheme_name, seed)
    final = out_dir / "adapters.safetensors"
    meta_path = out_dir / "train_meta.json"

    if final.exists() and meta_path.exists():
        print(f"[train {task}/{scheme_name}/seed{seed}] adapter exists, skipping")
        return out_dir

    if out_dir.exists():
        print(f"[train {task}/{scheme_name}/seed{seed}] discarding partial adapter, restarting")
        for f in out_dir.iterdir():
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = _write_config(out_dir / "lora_config.yaml", seed)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", str(MODELS_DIR / scheme_name),
        "--data", str(data_dir),
        "--adapter-path", str(out_dir),
        "--config", str(cfg_path),
        "--train",
    ]
    if LORA_MASK_PROMPT:
        cmd.append("--mask-prompt")

    print(f"$ {' '.join(cmd)}")
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - start

    assert final.exists(), f"training finished but {final} is missing"
    # Only the final adapter ships. mlx_lm also leaves a checkpoint every
    # save_every iters in the same directory, so summing the directory would
    # report ~7x the real deployment cost — and adapter size is exactly what
    # the size/efficiency comparison against the quantized baselines turns on.
    adapter_bytes = final.stat().st_size
    checkpoint_bytes = sum(
        f.stat().st_size for f in out_dir.glob("*_adapters.safetensors")
    )
    meta_path.write_text(json.dumps({
        "task": task, "scheme": scheme_name, "seed": seed,
        "train_seconds": round(elapsed, 1),
        "adapter_bytes": adapter_bytes,
        "checkpoint_bytes": checkpoint_bytes,
        "iters": LORA_ITERS, "batch_size": LORA_BATCH_SIZE,
        "learning_rate": LORA_LR, "rank": LORA_RANK, "scale": LORA_SCALE,
        "num_layers": LORA_NUM_LAYERS, "mask_prompt": LORA_MASK_PROMPT,
    }, indent=2))
    print(f"[train {task}/{scheme_name}/seed{seed}] done in {elapsed/60:.1f} min, "
          f"adapter {adapter_bytes/1e6:.1f} MB "
          f"(+{checkpoint_bytes/1e6:.0f} MB of checkpoints, not deployed)")
    return out_dir
