"""Is C2's TriviaQA advantage real recall, or the substring scorer rewarding length?

C2 at 4 bits beats both the untuned (B) and the tuned (C1) arm on TriviaQA by
9-13pp, while the weight-level check says its adapter is as eroded as GSM8K's
(~12% retained, at the estimator's floor). The standing hypothesis is that C2's
intermediate output length — 17-20 tokens against 27 untuned and 4 tuned — is
what the scorer rewards: factual.score() counts a completion correct if any
gold alias appears anywhere in it, so more words is more chances to hit.

Exact match kills that channel: the whole normalized completion must equal an
alias. If C2's edge is a length artifact it should collapse under exact match.
If it survives, the adapter is doing real work and the length story is wrong.
"""

import json

import numpy as np

from analyze_finetune import arm_variant, PRIMARY_SEED
from config import FULL_N, FT_FEW_SHOT, RESULTS_DIR
from evals import factual

ARMS = ["B", "C1", "C2", "D"]
SCHEMES = ["int8", "awq4", "gptq4", "int4"]


def exact(output, aliases):
    return factual.normalize(output) in {factual.normalize(a) for a in aliases}


def rows(arm, scheme):
    path = RESULTS_DIR / "triviaqa" / arm_variant(arm, PRIMARY_SEED) / f"{scheme}.jsonl"
    if not path.exists():
        return None
    out = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return out if len(out) >= FULL_N["triviaqa"] else None


def main():
    items = factual.load_items(FULL_N["triviaqa"], few_shot=FT_FEW_SHOT)
    gold = {i["id"]: i["gold"] for i in items}

    print("TriviaQA: substring scorer (as run) vs exact match, n=500\n")
    print(f"{'scheme':8s} {'arm':4s} {'substring':>10s} {'exact':>8s} {'gap':>7s} {'tokens':>7s}")
    table = {}
    for scheme in SCHEMES:
        for arm in ARMS:
            rs = rows(arm, scheme)
            if not rs:
                continue
            sub = np.mean([factual.score(r["output"], gold[r["id"]]) for r in rs])
            ex = np.mean([exact(r["output"], gold[r["id"]]) for r in rs])
            tok = np.mean([r["generation_tokens"] for r in rs])
            table[(scheme, arm)] = (sub, ex)
            print(f"{scheme:8s} {arm:4s} {sub:10.1%} {ex:8.1%} {sub-ex:+7.1%} {tok:7.0f}")
        print()

    print("C2's edge over the other arms, under each scorer:")
    print(f"{'scheme':8s} {'vs':6s} {'substring':>10s} {'exact':>8s}")
    for scheme in SCHEMES:
        if (scheme, "C2") not in table:
            continue
        c2_sub, c2_ex = table[(scheme, "C2")]
        for other in ("B", "C1"):
            if (scheme, other) not in table:
                continue
            o_sub, o_ex = table[(scheme, other)]
            print(f"{scheme:8s} {'vs '+other:6s} {c2_sub-o_sub:+10.1%} {c2_ex-o_ex:+8.1%}")


if __name__ == "__main__":
    main()
