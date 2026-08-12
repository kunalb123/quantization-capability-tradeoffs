"""GSM8K: grade-school math word problems, scored on final-answer exact match.

Items are built as few-shot chain-of-thought prompts (config.FEW_SHOT_K worked
examples, each ending in "#### <answer>", followed by the target question) so
the model has a concrete pattern to imitate for both the reasoning style and
the answer format. Scoring extracts the number after the last "####" in the
model's output and compares it to the number after "####" in the gold answer.
"""

import re

from datasets import load_dataset

from config import FEW_SHOT_K, SEED

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_LAST_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    """Pull the final numeric answer out of a GSM8K-style completion.

    Gold answers always contain "#### <number>", so that's tried first. Model
    completions are prompted (via the few-shot examples) to follow the same
    convention, but a model — especially a heavily quantized one — may drift
    and never emit "####". As a fallback, the last standalone number in the
    text is used, on the theory that a CoT response tends to state its final
    answer last. Returns None (never matches, never accidentally scores as
    correct) if no number is found at all.
    """
    match = None
    for match in _ANSWER_RE.finditer(text):
        pass
    if match:
        return match.group(1).replace(",", "")

    for match in _LAST_NUMBER_RE.finditer(text):
        pass
    if match:
        return match.group(0).replace(",", "")

    return None


def score(output: str, gold: str) -> bool:
    return extract_answer(output) == gold


def _format_example(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def load_items(n: int) -> list[dict]:
    """Build n GSM8K eval items as chat-format few-shot CoT prompts.

    Few-shot examples come from the train split (fixed by SEED, disjoint from
    the eval items), and the n eval questions come from the test split
    (shuffled by SEED, so pilot and full runs share a prefix — a pilot n=50
    run is the first 50 items of a full n=500 run, not a different sample).
    """
    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=SEED)
    test = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=SEED)

    few_shot = train.select(range(FEW_SHOT_K))
    eval_rows = test.select(range(n))

    messages_prefix = []
    for row in few_shot:
        messages_prefix += _format_example(row["question"], row["answer"])

    items = []
    for i, row in enumerate(eval_rows):
        items.append({
            "id": f"gsm8k-{i}",
            "messages": messages_prefix + [{"role": "user", "content": row["question"]}],
            "gold": extract_answer(row["answer"]),
        })
    return items
