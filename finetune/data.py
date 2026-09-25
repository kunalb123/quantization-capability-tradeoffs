"""Build LoRA training data for the fine-tuned tasks.

mlx_lm.lora wants a directory holding train.jsonl / valid.jsonl, one JSON
object per line. We emit the chat format ({"messages": [user, assistant]}),
which matches exactly how the tuned models are prompted at eval time
(0-shot, per config.FT_FEW_SHOT) — training on one prompt shape and
evaluating on another would confound the fine-tuning effect with a format
shift.

Leakage control, which is the whole reason this file exists rather than
being three lines inline:

  - Training rows come from each task's *train* split; eval items come from
    test (GSM8K) / validation (TriviaQA). Disjoint by construction.
  - The first FEW_SHOT_K rows of the train split are already used as the
    few-shot prompt prefix in evals/, so they're skipped here. Otherwise the
    model would be tuned on the very examples the untuned arms get shown,
    which would flatter the tuned arms for the wrong reason.
  - build() asserts both of the above against the actual eval items rather
    than trusting the split boundary, because a dataset revision that
    reshuffles splits would silently break the guarantee.
"""

import json
from pathlib import Path

from datasets import load_dataset

from config import FEW_SHOT_K, FT_TRAIN_N, FT_VALID_N, FT_DATA_DIR, SEED
from evals import factual, reasoning


def _gsm8k_rows(n_needed: int) -> list[dict]:
    """(question, full chain-of-thought answer) pairs from the train split.

    The completion keeps GSM8K's native "#### <answer>" ending, which is what
    evals/reasoning.py's scorer looks for first — so tuning teaches the answer
    format alongside the reasoning, and the tuned model's output is scored by
    the identical function used for every other arm.
    """
    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=SEED)
    rows = train.select(range(FEW_SHOT_K, FEW_SHOT_K + n_needed))
    return [
        {"messages": [
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["answer"]},
        ]}
        for r in rows
    ]


def _triviaqa_rows(n_needed: int) -> list[dict]:
    """(question, short answer) pairs from the train split.

    The completion is the bare answer string, matching the few-shot examples
    the untuned arms see. TriviaQA's scorer is a normalized substring match
    over answer aliases, so a bare answer scores the same as a sentence — but
    training on bare answers is what the format is supposed to teach.
    """
    train = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train").shuffle(seed=SEED)
    rows = train.select(range(FEW_SHOT_K, FEW_SHOT_K + n_needed))
    return [
        {"messages": [
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["answer"]["value"]},
        ]}
        for r in rows
    ]


BUILDERS = {"gsm8k": _gsm8k_rows, "triviaqa": _triviaqa_rows}
EVAL_MODULES = {"gsm8k": reasoning, "triviaqa": factual}


def _assert_no_leakage(task: str, rows: list[dict], eval_n: int) -> None:
    """Fail loudly if any training question also appears as an eval question.

    Compares the literal question text (not ids) against the eval items,
    because that's the thing that would actually contaminate the result. Also
    re-checks the few-shot prefix, which lives in the same train split and is
    the likelier accident of the two.
    """
    train_questions = {r["messages"][0]["content"].strip() for r in rows}

    eval_items = EVAL_MODULES[task].load_items(eval_n, few_shot=True)
    eval_questions = {i["messages"][-1]["content"].strip() for i in eval_items}
    overlap = train_questions & eval_questions
    assert not overlap, (
        f"[{task}] {len(overlap)} training questions also appear in the eval "
        f"set, e.g. {sorted(overlap)[0][:100]!r}"
    )

    few_shot_questions = {
        m["content"].strip()
        for m in eval_items[0]["messages"][:-1]
        if m["role"] == "user"
    }
    fs_overlap = train_questions & few_shot_questions
    assert not fs_overlap, (
        f"[{task}] {len(fs_overlap)} training questions are also few-shot "
        f"prompt examples, e.g. {sorted(fs_overlap)[0][:100]!r}"
    )


def build(task: str, eval_n: int) -> Path:
    """Write train.jsonl / valid.jsonl for `task`, returning the directory.

    Idempotent: if both files already exist they're left alone, so a resumed
    overnight run doesn't rebuild (and re-download) datasets it already has.
    """
    out_dir = FT_DATA_DIR / task
    train_path, valid_path = out_dir / "train.jsonl", out_dir / "valid.jsonl"
    if train_path.exists() and valid_path.exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = BUILDERS[task](FT_TRAIN_N + FT_VALID_N)
    _assert_no_leakage(task, rows, eval_n)

    for path, chunk in ((train_path, rows[:FT_TRAIN_N]), (valid_path, rows[FT_TRAIN_N:])):
        with path.open("w") as f:
            for row in chunk:
                f.write(json.dumps(row) + "\n")
    return out_dir


if __name__ == "__main__":
    from config import FULL_N, FT_TASKS

    for task in FT_TASKS:
        path = build(task, FULL_N[task])
        n_train = sum(1 for _ in (path / "train.jsonl").open())
        n_valid = sum(1 for _ in (path / "valid.jsonl").open())
        print(f"[{task}] {n_train} train / {n_valid} valid -> {path}")
