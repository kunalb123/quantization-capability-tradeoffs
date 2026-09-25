"""Evaluate one (task, scheme, arm, seed) combination. One process, one eval.

The orchestrator shells out to this rather than calling run_items() inline so
that every model load lives in a process that exits afterwards. Across ~50
evals in an unattended overnight run that matters twice over: MLX's buffer
cache doesn't accumulate across loads, and a crash or OOM on one arm kills a
subprocess instead of the whole run.

Scoring is deliberately not done here — this writes the same per-item JSONL
that the quantization-only runs wrote, and analysis happens later over all
arms at once. The accuracy printed at the end is a progress signal for the
log, not a result.
"""

import argparse
import json

from config import FULL_N, FT_FEW_SHOT, SCHEMES
from evals import factual, reasoning
from harness.runner import run_items

TASK_MODULES = {"gsm8k": reasoning, "triviaqa": factual}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASK_MODULES))
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--variant", required=True, help='e.g. "B" or "C1/seed0"')
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    task_module = TASK_MODULES[args.task]
    n = args.n or FULL_N[args.task]

    # Every fine-tuning arm — tuned or untuned control — is prompted the way
    # the tuned models were trained (0-shot). Item ids are unchanged, so these
    # results stay paired with the existing few-shot runs for McNemar.
    items = task_module.load_items(n, few_shot=FT_FEW_SHOT)
    gold_by_id = {item["id"]: item["gold"] for item in items}

    scheme = next(s for s in SCHEMES if s.name == args.scheme)
    path = run_items(
        scheme, args.task, items,
        adapter_path=args.adapter_path,
        model_path=args.model_path,
        variant=args.variant,
    )

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    correct = sum(1 for r in rows if task_module.score(r["output"], gold_by_id[r["id"]]))
    print(f"RESULT {args.task}/{args.variant}/{args.scheme}: "
          f"{correct}/{len(rows)} = {correct / len(rows):.1%}")


if __name__ == "__main__":
    main()
