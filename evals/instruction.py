"""IFEval: verifiable instruction-following, scored by strict prompt-level
compliance — every instruction attached to a prompt must be satisfied for the
item to count correct.

Unlike GSM8K/TriviaQA, correctness here isn't "does the output match a gold
string" — it's "does the output obey a set of programmatically checkable
constraints" (e.g. "use no commas", "write at least 300 words", "wrap your
title in << >>"). Each google/IFEval item carries an instruction_id_list and
matching kwargs; this module implements one checker function per instruction
id and requires ALL of an item's checkers to pass (IFEval's standard
"strict accuracy" metric — no credit for partial compliance).

This is a from-scratch reimplementation of the checking logic, not a call
into Google's original instruction_following_eval package (no such package
is pip-installable). One instruction type, language:response_language, is
NOT implemented — it needs a language-detection dependency this project
doesn't otherwise use — and items requiring it are filtered out at load time
rather than silently scored wrong.
"""

import json
import re
import string

from datasets import load_dataset

from config import SEED

# --- individual instruction checkers -----------------------------------
# Each takes the raw model output plus whichever kwargs fields IFEval
# populated for that instruction id (unused fields are None and ignored).


def _no_comma(r, **kw):
    return "," not in r


def _number_words(r, num_words=None, relation=None, **kw):
    n = len(r.split())
    if relation == "at least":
        return n >= num_words
    if relation == "less than":
        return n < num_words
    return abs(n - num_words) <= max(1, round(num_words * 0.1))  # "around"


_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")


def _number_sentences(r, num_sentences=None, relation=None, **kw):
    n = len([s for s in _SENTENCE_SPLIT_RE.split(r) if s.strip()])
    if relation == "at least":
        return n >= num_sentences
    if relation == "less than":
        return n < num_sentences
    return n == num_sentences


def _forbidden_words(r, forbidden_words=None, **kw):
    lower = r.lower()
    return not any(re.search(rf"\b{re.escape(w.lower())}\b", lower) for w in (forbidden_words or []))


def _number_highlighted_sections(r, num_highlights=None, **kw):
    return len(re.findall(r"\*[^*]+\*", r)) >= num_highlights


def _keyword_frequency(r, keyword=None, frequency=None, relation=None, **kw):
    n = len(re.findall(rf"\b{re.escape(keyword.lower())}\b", r.lower()))
    if relation == "at least":
        return n >= frequency
    return n < frequency


def _repeat_prompt(r, prompt_to_repeat=None, **kw):
    return r.strip().startswith(prompt_to_repeat.strip())


def _quotation(r, **kw):
    t = r.strip()
    return len(t) >= 2 and t.startswith('"') and t.endswith('"')


def _english_lowercase(r, **kw):
    return r == r.lower()


def _english_capital(r, **kw):
    return r == r.upper()


def _keyword_existence(r, keywords=None, **kw):
    lower = r.lower()
    return all(re.search(rf"\b{re.escape(k.lower())}\b", lower) for k in (keywords or []))


def _title(r, **kw):
    return bool(re.search(r"<<[^\n]+>>", r))


def _letter_frequency(r, letter=None, let_frequency=None, let_relation=None, **kw):
    n = r.lower().count(letter.lower())
    if let_relation == "at least":
        return n >= let_frequency
    return n < let_frequency


def _number_bullet_lists(r, num_bullets=None, **kw):
    n = len(re.findall(r"^\s*[*-]\s", r, flags=re.MULTILINE))
    return n == num_bullets


def _number_placeholders(r, num_placeholders=None, **kw):
    return len(re.findall(r"\[[^\]]+\]", r)) >= num_placeholders


def _number_paragraphs(r, num_paragraphs=None, **kw):
    paras = [p for p in re.split(r"\n\s*\n", r.strip()) if p.strip()]
    return len(paras) == num_paragraphs


def _end_checker(r, end_phrase=None, **kw):
    return r.strip().endswith(end_phrase.strip())


def _postscript(r, postscript_marker=None, **kw):
    return (postscript_marker or "P.S.").lower() in r.lower()


def _capital_word_frequency(r, capital_frequency=None, capital_relation=None, **kw):
    n = len(re.findall(r"\b[A-Z]{2,}\b", r))
    if capital_relation == "at least":
        return n >= capital_frequency
    return n < capital_frequency


def _two_responses(r, **kw):
    parts = [p for p in r.split("******") if p.strip()]
    return len(parts) >= 2


def _json_format(r, **kw):
    text = re.sub(r"^```(?:json)?|```$", "", r.strip(), flags=re.MULTILINE).strip()
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _multiple_sections(r, num_sections=None, section_spliter=None, **kw):
    marker = re.escape(section_spliter or "Section")
    return len(re.findall(rf"{marker}\s*\d+", r)) >= num_sections


def _nth_paragraph_first_word(r, nth_paragraph=None, first_word=None, **kw):
    paras = [p for p in re.split(r"\n\s*\n", r.strip()) if p.strip()]
    if nth_paragraph is None or nth_paragraph > len(paras):
        return False
    words = paras[nth_paragraph - 1].strip().split()
    return bool(words) and words[0].strip(string.punctuation).lower() == first_word.lower()


_CONSTRAINED_RESPONSES = ("my answer is yes", "my answer is no", "my answer is maybe")


def _constrained_response(r, **kw):
    lower = r.lower()
    return any(phrase in lower for phrase in _CONSTRAINED_RESPONSES)


CHECKERS = {
    "punctuation:no_comma": _no_comma,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_sentences": _number_sentences,
    "keywords:forbidden_words": _forbidden_words,
    "detectable_format:number_highlighted_sections": _number_highlighted_sections,
    "keywords:frequency": _keyword_frequency,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:quotation": _quotation,
    "change_case:english_lowercase": _english_lowercase,
    "keywords:existence": _keyword_existence,
    "detectable_format:title": _title,
    "keywords:letter_frequency": _letter_frequency,
    "detectable_format:number_bullet_lists": _number_bullet_lists,
    "detectable_content:number_placeholders": _number_placeholders,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "startend:end_checker": _end_checker,
    "detectable_content:postscript": _postscript,
    "change_case:english_capital": _english_capital,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "combination:two_responses": _two_responses,
    "detectable_format:json_format": _json_format,
    "detectable_format:multiple_sections": _multiple_sections,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_format:constrained_response": _constrained_response,
    # "language:response_language" intentionally unimplemented — see module docstring.
}


def score(output: str, gold: list[tuple[str, dict]]) -> bool:
    """gold is [(instruction_id, kwargs), ...]; every checker must pass."""
    return all(CHECKERS[instruction_id](output, **kwargs) for instruction_id, kwargs in gold)


def load_items(n: int) -> list[dict]:
    """Build n IFEval items, skipping any whose instructions aren't fully
    covered by CHECKERS (currently just items using language:response_language).

    Zero-shot by design — IFEval prompts are self-contained instructions, and
    few-shot examples would risk teaching the model a response *style* that
    biases compliance with the very constraints being measured.
    """
    ds = load_dataset("google/IFEval", split="train").shuffle(seed=SEED)

    items = []
    for row in ds:
        if len(items) >= n:
            break
        if not all(iid in CHECKERS for iid in row["instruction_id_list"]):
            continue

        gold = [
            (iid, {k: v for k, v in kwargs.items() if v is not None})
            for iid, kwargs in zip(row["instruction_id_list"], row["kwargs"])
        ]
        items.append({
            "id": f"ifeval-{row['key']}",
            "messages": [{"role": "user", "content": row["prompt"]}],
            "gold": gold,
        })

    if len(items) < n:
        raise ValueError(
            f"only found {len(items)} IFEval items with fully-checkable instructions, need {n}"
        )
    return items
