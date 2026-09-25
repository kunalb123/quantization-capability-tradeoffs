"""TriviaQA: closed-book open-domain trivia questions, scored by normalized
answer-alias match.

Uses the "rc.nocontext" config — no supporting passage is given, so the model
must answer from its own parametric knowledge. That's the point: this task
probes whether quantization erodes factual recall specifically, as opposed to
reasoning (GSM8K) or instruction compliance (IFEval). Each question has
multiple acceptable answer strings ("aliases", e.g. "JFK" / "John F. Kennedy")
— scoring counts a completion correct if any alias appears in it after both
are normalized the standard SQuAD/TriviaQA way (lowercased, articles and
punctuation stripped).
"""

import re
import string

from datasets import load_dataset

from config import FEW_SHOT_K, SEED

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def score(output: str, gold_aliases: list[str]) -> bool:
    """True if any gold alias appears (as normalized text) in the output.

    Substring match rather than exact match because models — especially
    prompted for a short answer but not perfectly compliant — may wrap the
    answer in a sentence ("The answer is David Seville.") rather than emit it
    bare; the few-shot examples push toward bare answers, but this scorer
    doesn't rely on that being perfect.
    """
    normalized_output = normalize(output)
    return any(normalize(alias) in normalized_output for alias in gold_aliases)


def _format_example(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def load_items(n: int, few_shot: bool = True) -> list[dict]:
    """Build n TriviaQA eval items as chat-format few-shot prompts.

    Few-shot examples come from the train split (fixed by SEED), eval
    questions from the validation split (shuffled by SEED, so a pilot n=50
    run is a prefix of a full n=500 run). TriviaQA has no held-out test-set
    answers, so validation is the eval split here.

    few_shot=False drops the worked examples (see evals/reasoning.py for
    why the 0-shot arm exists). Item ids are unchanged, so arms stay
    paired item-for-item.
    """
    train = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train").shuffle(seed=SEED)
    val = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=SEED)

    few_shot_rows = train.select(range(FEW_SHOT_K))
    eval_rows = val.select(range(n))

    messages_prefix = []
    if few_shot:
        for row in few_shot_rows:
            messages_prefix += _format_example(row["question"], row["answer"]["value"])

    items = []
    for i, row in enumerate(eval_rows):
        items.append({
            "id": f"triviaqa-{i}",
            "messages": messages_prefix + [{"role": "user", "content": row["question"]}],
            "gold": row["answer"]["normalized_aliases"],
        })
    return items


