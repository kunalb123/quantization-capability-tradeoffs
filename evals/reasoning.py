"""GSM8K: grade-school math word problems, scored on final-answer exact match.

Items are built as few-shot chain-of-thought prompts (config.FEW_SHOT_K worked
examples, each ending in "#### <answer>", followed by the target question) so
the model has a concrete pattern to imitate for both the reasoning style and
the answer format. Scoring extracts the model's final numeric answer and
compares it to the number after "####" in the gold answer.

Two extraction rules live here. extract_answer() is primary;
extract_answer_legacy() is the original rule, kept so results can be reported
under both — see extract_answer's docstring for why that is worth doing.
"""

import re

from datasets import load_dataset

from config import FEW_SHOT_K, SEED

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_LAST_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    """Pull the final numeric answer out of a GSM8K-style completion.

    Order of attempts: the number following "####" if the model emitted the
    prompted marker, else the number following the last "=", else the legacy
    rule's last-number-in-text fallback.

    The "=" step exists because that fallback systematically misreads verbose
    completions. A model that shows its working tends to close with a
    restatement repeating an *input* quantity after the answer —

        Distance = 3 feet/second x 40 seconds
        Distance = 120 feet
        So, the turtle can crawl 120 feet in 40 seconds.

    — where the last number in the text is 40, not the answer 120. Reading
    after the last "=" recovers the computed value instead.

    This is a correctness fix, but not a neutral one, because it only bites on
    arms that write prose: on this project's GSM8K runs it lifts the untuned
    0-shot arms by ~5-6pp and the fine-tuned arms by 0.0pp (those emit terse
    answers with no trailing restatement). It therefore changes the *size* of
    any tuned-vs-untuned gap — not its sign or its significance, but enough
    that both rules should be reported side by side rather than one being
    silently chosen. Hence extract_answer_legacy() stays.
    """
    if "####" in text:
        match = _LAST_NUMBER_RE.search(text[text.index("####"):])
        if match:
            return match.group(0).replace(",", "")

    last_eq = text.rfind("=")
    if last_eq >= 0:
        match = _LAST_NUMBER_RE.search(text[last_eq:])
        if match:
            return match.group(0).replace(",", "")

    return extract_answer_legacy(text)


def extract_answer_legacy(text: str) -> str | None:
    """The original rule: number after the last "####", else last number in text.

    Superseded by extract_answer() (which still falls back to it), and kept so
    every result can be reported under both rules. Returns None — never
    matching, never accidentally scoring correct — if no number is found.
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


def score_legacy(output: str, gold: str) -> bool:
    """score() under the pre-fix extraction rule, for side-by-side reporting."""
    return extract_answer_legacy(output) == gold


def _format_example(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def load_items(n: int, few_shot: bool = True) -> list[dict]:
    """Build n GSM8K eval items as chat-format few-shot CoT prompts.

    Few-shot examples come from the train split (fixed by SEED, disjoint from
    the eval items), and the n eval questions come from the test split
    (shuffled by SEED, so pilot and full runs share a prefix — a pilot n=50
    run is the first 50 items of a full n=500 run, not a different sample).

    few_shot=False drops the worked examples and sends the bare question.
    Fine-tuned models are trained on — and so evaluated with — that 0-shot
    format, so a 0-shot untuned arm is what makes the tuned/untuned
    comparison fair: scoring a 0-shot tuned model against the existing
    8-shot untuned numbers would confound fine-tuning with the prompt
    format change. Item ids are identical either way, so every arm stays
    paired on the same questions for McNemar.
    """
    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=SEED)
    test = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=SEED)

    few_shot_rows = train.select(range(FEW_SHOT_K))
    eval_rows = test.select(range(n))

    messages_prefix = []
    if few_shot:
        for row in few_shot_rows:
            messages_prefix += _format_example(row["question"], row["answer"])

    items = []
    for i, row in enumerate(eval_rows):
        items.append({
            "id": f"gsm8k-{i}",
            "messages": messages_prefix + [{"role": "user", "content": row["question"]}],
            "gold": extract_answer(row["answer"]),
        })
    return items
