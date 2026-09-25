"""Run a set of eval items through one quantization scheme and record results.

This is shared infrastructure used by every eval task (GSM8K, TriviaQA, IFEval,
JSON) — it knows how to load a model, generate greedy completions, and persist
per-item results, but nothing about how to score them. Task-specific scripts
build items (each with a stable "id" and a chat "messages" list), call
run_items() once per scheme, and do their own scoring on the resulting JSONL.

Per-item persistence (rather than only aggregate scores) is what makes the
paired analysis in analyze.py possible later: every scheme's output is keyed
by the same item ids, so schemes can be compared on identical items with
McNemar's test and bootstrap CIs instead of independent accuracy numbers.
"""

import json
import time
from pathlib import Path
from typing import Iterable, TypedDict

from config import MAX_TOKENS, RESULTS_DIR, TEMPERATURE, QuantScheme


class Item(TypedDict):
    id: str
    messages: list[dict[str, str]]


def _output_path(scheme: QuantScheme, task: str, variant: str | None) -> Path:
    """Where one (scheme, task, variant) run's results live.

    variant=None keeps the original layout (results/<task>/<scheme>.jsonl) so
    the pre-existing quantization-only results stay exactly where analyze.py
    expects them. Fine-tuning arms pass a variant like "C1/seed0", giving
    results/<task>/C1/seed0/<scheme>.jsonl — a separate file per arm, keyed by
    the same item ids, which is what keeps every arm paired for McNemar.
    """
    if variant is None:
        return RESULTS_DIR / task / f"{scheme.name}.jsonl"
    return RESULTS_DIR / task / variant / f"{scheme.name}.jsonl"


def _load_done_ids(path: Path) -> set[str]:
    """Ids already present in an existing results file.

    Results are appended one line at a time as they're generated, so a run
    that crashes or is interrupted partway through leaves a valid partial
    JSONL. Reading it back lets run_items() skip completed items on rerun
    instead of regenerating (and re-paying model load + inference cost for)
    everything from scratch.
    """
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["id"])
    return done


def run_items(
    scheme: QuantScheme,
    task: str,
    items: Iterable[Item],
    *,
    adapter_path: Path | str | None = None,
    model_path: Path | str | None = None,
    variant: str | None = None,
) -> Path:
    """Generate a completion for every item under `scheme` and append results
    to results/<task>/<scheme.name>.jsonl, skipping ids already recorded.

    adapter_path attaches a trained LoRA adapter at load time without fusing
    it (arm C1: quantized base + high-precision adapter). model_path overrides
    which weights are loaded, for arms whose model is a transient build rather
    than scheme.path (C2's fused+requantized model, D's tune-then-quantize
    model) — scheme is still passed so the result file is named for the
    quantization scheme being measured. variant selects the output subdirectory.

    Decoding is greedy (temperature 0, per config.TEMPERATURE) so the only
    thing that varies across schemes for a given item is the model weights.
    Each line written is:
      {"id", "input", "output", "prompt_tokens", "generation_tokens",
       "generation_tps", "load_time_s", "peak_memory_gb"}
    "input" is only the final message's content (the real target question) —
    any few-shot examples earlier in item["messages"] are still sent to the
    model as part of the prompt, just not duplicated into every result line,
    since they're fixed per task and available from the eval module itself.
    load_time_s and peak_memory_gb are the same on every line within a run
    (captured once, at model load) — duplicated per line so each result is
    self-contained and analyzable without cross-referencing a separate log.
    """
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    out_path = _output_path(scheme, task, variant)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_ids(out_path)

    items = [item for item in items if item["id"] not in done_ids]
    label = f"{scheme.name}/{task}" + (f"/{variant}" if variant else "")
    if not items:
        print(f"[{label}] all items already done, skipping")
        return out_path

    mx.reset_peak_memory()
    load_start = time.perf_counter()
    load_kwargs = {}
    if adapter_path is not None:
        load_kwargs["adapter_path"] = str(adapter_path)
    model, tokenizer = load(str(model_path or scheme.path), **load_kwargs)
    load_time_s = time.perf_counter() - load_start

    sampler = make_sampler(temp=TEMPERATURE)
    max_tokens = MAX_TOKENS[task]

    with out_path.open("a") as f:
        for i, item in enumerate(items, 1):
            prompt = tokenizer.apply_chat_template(
                item["messages"], add_generation_prompt=True
            )

            text = ""
            response = None
            for response in stream_generate(
                model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler
            ):
                text += response.text

            result = {
                "id": item["id"],
                "input": item["messages"][-1]["content"],
                "output": text,
                "prompt_tokens": response.prompt_tokens,
                "generation_tokens": response.generation_tokens,
                "generation_tps": response.generation_tps,
                "load_time_s": load_time_s,
                "peak_memory_gb": response.peak_memory,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()

            if i % 10 == 0 or i == len(items):
                print(f"[{label}] {i}/{len(items)}")

    return out_path
