"""Main runner — evaluates quantization schemes across eval tasks.

Each task module in evals/ exposes load_items(n) -> list of items (each with
an "id", chat "messages", and a "gold" answer) and score(output, gold) -> bool
to grade one completion. This script wires those into harness.runner.run_items:
for every (task, scheme) pair it generates completions (skipping items
already recorded, per run_items' resumability) and reports accuracy.

All four eval tasks are wired up: GSM8K (evals/reasoning.py), TriviaQA
(evals/factual.py), IFEval (evals/instruction.py), and JSON generation
(evals/structured.py).

Usage:
    python benchmark.py --task gsm8k --schemes bf16 int4
    python benchmark.py --task gsm8k --pilot          # config.PILOT_N items
    python benchmark.py                                # every task, every scheme, full n
"""

import argparse
import json

from config import FULL_N, PILOT_N, SCHEMES
from evals import factual, instruction, reasoning, structured
from harness.runner import run_items

TASK_MODULES = {
    "gsm8k": reasoning,
    "triviaqa": factual,
    "ifeval": instruction,
    "json": structured,
}


def run_task(task: str, scheme_names: list[str], n: int) -> None:
    task_module = TASK_MODULES[task]
    items = task_module.load_items(n)
    gold_by_id = {item["id"]: item["gold"] for item in items}

    scheme_by_name = {s.name: s for s in SCHEMES}
    for name in scheme_names:
        scheme = scheme_by_name[name]
        path = run_items(scheme, task, items)
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        correct = sum(
            1 for line in lines
            if task_module.score(line["output"], gold_by_id[line["id"]])
        )
        print(f"{task}/{name}: {correct}/{len(lines)} = {correct / len(lines):.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=sorted(TASK_MODULES), help="defaults to every registered task"
    )
    parser.add_argument(
        "--schemes", nargs="+", default=[s.name for s in SCHEMES],
        choices=[s.name for s in SCHEMES],
    )
    parser.add_argument(
        "--n", type=int, help="items to evaluate; defaults to --pilot/full size for the task"
    )
    parser.add_argument(
        "--pilot", action="store_true", help="use config.PILOT_N instead of config.FULL_N"
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(TASK_MODULES)
    size_table = PILOT_N if args.pilot else FULL_N

    for task in tasks:
        n = args.n or size_table[task]
        run_task(task, args.schemes, n)


if __name__ == "__main__":
    main()
