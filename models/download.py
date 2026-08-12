"""Build the five quantized model variants defined in config.SCHEMES.

Every variant (bf16, int8, awq4, gptq4, int4) is derived from the same
BASE_MODEL, so the only thing that differs between them is the quantization
method — that's what makes the capability comparisons in this project
meaningful. Each variant is smoke-tested right after it's built, and once
all five are done, an environment.json snapshot is written recording
exactly what library versions produced them (see environment.py).

Run directly: `python models/download.py`
"""

import json
import subprocess
import sys
from pathlib import Path

# download.py lives in models/, but it needs to import config.py and
# environment.py from the repo root — this puts the root on sys.path so
# those imports resolve regardless of which directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BASE_MODEL, MODELS_DIR, SCHEMES, SEED
from environment import capture_environment


def _ensure_base_model_cached() -> None:
    """Fully populate the standard Hugging Face cache for BASE_MODEL.

    mlx_lm's convert/awq/gptq commands resolve --model/--hf-path through
    this shared cache, and their save step re-checks the full file list
    against it with network access disabled — if the cache only has a
    partial file set (e.g. because something earlier fetched a restricted
    subset), that check fails instead of fetching what's missing. Doing one
    full, unrestricted snapshot_download up front avoids that.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=BASE_MODEL)


def _run(cmd: list[str]) -> None:
    """Run a subprocess command, echoing it first.

    AWQ/GPTQ calibration can take many minutes with no output otherwise, so
    printing the command up front makes it clear the script is still working
    and shows exactly which mlx-lm CLI invocation is running.
    """
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def build_scheme(scheme) -> None:
    """Produce the model files for one QuantScheme at scheme.path.

    Dispatches on scheme.method:
      - "none"    -> plain download, no quantization (the bf16 baseline)
      - "convert" -> mlx_lm.convert -q (naive affine quantization: int8, int4)
      - "awq"     -> mlx_lm.awq (activation-aware quantization)
      - "gptq"    -> mlx_lm.gptq

    All three quantization methods read BASE_MODEL, scheme.bits, and
    scheme.group_size from config.py, so every 4-bit scheme is quantized
    from the same source weights at the same group size — the only variable
    that changes between them is the quantization algorithm itself.

    If scheme.path already exists, this is a no-op. That makes the script
    resumable: if it crashes or is interrupted partway through the loop over
    all five schemes in main(), rerunning it skips everything already built
    instead of starting over.
    """
    if scheme.path.exists():
        print(f"[{scheme.name}] already exists, skipping build")
        return

    scheme.path.parent.mkdir(parents=True, exist_ok=True)

    if scheme.method == "none":
        # BASE_MODEL is already MLX-formatted bf16 — every quantized variant
        # is derived from this same download, so a plain snapshot is enough.
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=BASE_MODEL, local_dir=str(scheme.path))

    elif scheme.method == "convert":
        _run([
            sys.executable, "-m", "mlx_lm", "convert",
            "--hf-path", BASE_MODEL,
            "--mlx-path", str(scheme.path),
            "-q",
            "--q-bits", str(scheme.bits),
            "--q-group-size", str(scheme.group_size),
        ])

    elif scheme.method == "awq":
        _run([
            sys.executable, "-m", "mlx_lm", "awq",
            "--model", BASE_MODEL,
            "--mlx-path", str(scheme.path),
            "--bits", str(scheme.bits),
            "--group-size", str(scheme.group_size),
            "--seed", str(SEED),
        ])

    elif scheme.method == "gptq":
        _run([
            sys.executable, "-m", "mlx_lm", "gptq",
            "--model", BASE_MODEL,
            "--mlx-path", str(scheme.path),
            "--bits", str(scheme.bits),
            "--group-size", str(scheme.group_size),
            "--seed", str(SEED),
        ])

    else:
        raise ValueError(f"unknown method: {scheme.method!r} for scheme {scheme.name!r}")


def smoke_test(scheme) -> None:
    """Load the just-built model and run one short generation.

    This is a cheap correctness check, not an eval — it only confirms the
    scheme produced a loadable model directory that generates *something*
    non-empty, catching a silently corrupt or truncated build before it
    wastes time downstream (e.g. during the actual capability evals). Also
    reports on-disk size, which is a useful sanity check on its own: bf16
    should be roughly 2x int8 and 4x the int4/awq4/gptq4 variants.
    """
    from mlx_lm import generate, load

    model, tokenizer = load(str(scheme.path))
    messages = [{"role": "user", "content": "What is 12 + 30? Answer with just the number."}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)

    assert text.strip(), f"[{scheme.name}] smoke test produced empty output"

    size_gb = sum(f.stat().st_size for f in scheme.path.rglob("*") if f.is_file()) / 1e9
    print(f"[{scheme.name}] OK — {size_gb:.2f} GB on disk — output: {text.strip()[:80]!r}")


def main():
    """Build + smoke-test every scheme in config.SCHEMES, in order, then
    write one environment.json snapshot.

    The environment snapshot is written once at the end (not per-scheme)
    because it's meant to capture what ran this whole script — mlx-lm
    version, chip, etc — not something that's expected to differ scheme to
    scheme within a single run.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_base_model_cached()

    for scheme in SCHEMES:
        print(f"\n=== {scheme.name} ({scheme.method}, {scheme.bits}-bit) ===")
        build_scheme(scheme)
        smoke_test(scheme)

    env_path = MODELS_DIR / "environment.json"
    env_path.write_text(json.dumps(capture_environment(), indent=2))
    print(f"\nEnvironment snapshot written to {env_path}")


if __name__ == "__main__":
    main()
