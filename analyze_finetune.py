"""Paired analysis of the fine-tuning arms.

Extends analyze.py's approach rather than replacing it: every arm is scored on
the identical item ids, so all comparisons stay paired and use McNemar's exact
test plus a percentile bootstrap, never independent-sample accuracy math.

Three families of question, and they need different tests:

  1. Did fine-tuning help, within one scheme?
     C1 vs B on the same items -> McNemar exact. This is analyze.py's existing
     machinery pointed at two arms instead of two schemes.

  2. Did fine-tuning help the quantized schemes *more* than it helped bf16 —
     i.e. does tuning close the quantization gap?
     This is an interaction, [acc(C1)-acc(B)]_scheme - [acc(C1)-acc(B)]_bf16,
     and McNemar cannot test it. bootstrap_interaction_ci() resamples item ids
     once per draw and recomputes all four accuracies from that one resample,
     which preserves the pairing across both arms and both schemes. Power for a
     difference-of-differences is roughly half that of a simple delta, so at
     n=500 expect to resolve ~7-8pp interactions, not 4pp — a CI containing 0
     is reported as inconclusive, not as evidence of no interaction.

  3. Does the fine-tuning survive being quantized?
     C1 (unfused, bf16 adapter) vs C2 (fused, requantized) vs D (tuned in bf16,
     then quantized) — all paired, all McNemar.

Multiple comparisons: the primary family is declared in PRIMARY_KEYS and
Holm-corrected. Everything else is printed as exploratory and labelled so, so
a reader isn't invited to treat 40 uncorrected p-values as 40 findings.

Usage:
    python analyze_finetune.py
    python analyze_finetune.py --task gsm8k
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze import bootstrap_delta_ci, mcnemar_exact
from config import (
    ADAPTERS_DIR, ALPHA, BASELINE_SCHEME, FULL_N, FT_FEW_SHOT, FT_TASKS,
    N_BOOTSTRAP, QUANTIZED_SCHEMES, RESULTS_DIR, SCHEMES, SEED,
)
from evals import factual, reasoning

TASK_MODULES = {"gsm8k": reasoning, "triviaqa": factual}
ALL_SCHEMES = [s.name for s in SCHEMES]
PRIMARY_SEED = 0


def arm_variant(arm: str, seed: int = PRIMARY_SEED) -> str:
    """Results subdirectory for an arm (arm B has no seed — it isn't trained)."""
    return "B" if arm == "B" else f"{arm}/seed{seed}"


def load_correctness(task: str, arm: str, scheme: str, seed: int = PRIMARY_SEED,
                     legacy: bool = False) -> dict[str, bool] | None:
    """id -> correct for one arm, or None if that run isn't finished.

    legacy=True scores with the task module's pre-fix rule where it has one
    (GSM8K's extract_answer_legacy; see evals/reasoning.py). The primary
    extractor changes the size of the tuned-vs-untuned gap, so section [1]
    reports both rather than letting a scoring choice pass unremarked.

    "Not finished" deliberately includes *partially* finished. run_items
    appends per item, so an arm being evaluated right now has a valid JSONL
    holding a prefix of the items — and comparing a complete arm against that
    prefix silently produces a paired test over whatever handful of items
    happen to exist, which reads like a result but is noise. An arm is only
    returned once it has all FULL_N items.

    Returning None rather than raising lets the report render mid-run, showing
    what has genuinely landed instead of failing on the first missing arm.
    """
    path = RESULTS_DIR / task / arm_variant(arm, seed) / f"{scheme}.jsonl"
    if not path.exists():
        return None
    items = TASK_MODULES[task].load_items(FULL_N[task], few_shot=FT_FEW_SHOT)
    gold_by_id = {i["id"]: i["gold"] for i in items}
    module = TASK_MODULES[task]
    score = getattr(module, "score_legacy", module.score) if legacy else module.score
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in gold_by_id:
            out[row["id"]] = score(row["output"], gold_by_id[row["id"]])
    if len(out) < FULL_N[task]:
        return None  # still being written — see docstring
    return out


def load_rows(task: str, arm: str, scheme: str, seed: int = PRIMARY_SEED) -> list[dict] | None:
    """Raw result rows for one arm, or None if that run isn't finished.

    Same completeness guard as load_correctness(), for the same reason: a
    mean over however many items happen to be on disk mid-run reads like a
    measurement but is an artifact of when the report was rendered.
    """
    path = RESULTS_DIR / task / arm_variant(arm, seed) / f"{scheme}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows if len(rows) >= FULL_N[task] else None


BARE_MAX_WORDS = 8


def is_bare(output: str) -> bool:
    """True if the completion gives an answer with no working shown.

    Defined structurally — one line, at most BARE_MAX_WORDS words — rather
    than per task, so the same rule describes both tasks and the contrast
    between them is a property of the outputs, not of two different rules.
    """
    stripped = output.strip()
    return "\n" not in stripped and len(stripped.split()) <= BARE_MAX_WORDS


def format_table(task: str) -> None:
    """Mean generation length and bare-answer rate for every arm.

    Fine-tuning on terse gold answers collapses how much the model writes,
    and that collapse is the mechanism behind much of the accuracy movement
    in [1]-[3] — so it belongs beside those tests rather than being inferred
    from them. Which of the two columns carries the signal is task-dependent:
    on TriviaQA the bare rate moves over most of its range, while on GSM8K it
    sits near zero in every arm (even a tuned model shows some working) and
    the token count is what registers the same effect. Both are printed for
    both tasks so that difference stays visible instead of being hidden by
    picking one statistic per task.
    """
    arms = ["B", "C1", "C2", "D"]
    print("\n[0] Output format (mean generation tokens / bare-answer rate)")
    print(f"  {'scheme':10s}" + "".join(f"{a:>16s}" for a in arms))
    for scheme in ALL_SCHEMES:
        cells = []
        for arm in arms:
            rows = load_rows(task, arm, scheme)
            if rows is None:
                cells.append(f"{'(pending)':>16s}")
                continue
            tokens = np.mean([r["generation_tokens"] for r in rows])
            bare = np.mean([is_bare(r["output"]) for r in rows])
            cells.append(f"{tokens:9.0f} /{bare:5.0%}")
        print(f"  {scheme:10s}" + "".join(cells))


def _aligned(*correctness: dict[str, bool]) -> tuple[list[str], list[list[bool]]]:
    """Restrict every arm to the ids all of them have, preserving order.

    A partially-finished arm would otherwise silently compare different item
    subsets across arms, which is exactly the unpaired comparison this whole
    analysis exists to avoid.
    """
    shared = sorted(set.intersection(*(set(c) for c in correctness)),
                    key=lambda i: int(i.rsplit("-", 1)[1]))
    return shared, [[c[i] for i in shared] for c in correctness]


def bootstrap_interaction_ci(
    base_b: list[bool], base_c: list[bool],
    scheme_b: list[bool], scheme_c: list[bool],
    n_bootstrap: int = N_BOOTSTRAP, alpha: float = ALPHA,
) -> tuple[float, float, float]:
    """CI for (scheme_c - scheme_b) - (base_c - base_b), the tuning interaction.

    One set of resampled indices is applied to all four arms per draw, because
    the arms are paired item-for-item; resampling them independently would
    destroy the pairing and inflate the interval.
    """
    rng = np.random.default_rng(SEED)
    arrs = [np.asarray(a, dtype=float) for a in (base_b, base_c, scheme_b, scheme_c)]
    n = len(arrs[0])
    point = (arrs[3].mean() - arrs[2].mean()) - (arrs[1].mean() - arrs[0].mean())

    draws = np.empty(n_bootstrap)
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        bb, bc, sb, sc = (a[idx].mean() for a in arrs)
        draws[k] = (sc - sb) - (bc - bb)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm-Bonferroni step-down: key -> survives correction at `alpha`."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, still_rejecting = {}, True
    for rank, (key, p) in enumerate(ordered):
        threshold = alpha / (m - rank)
        if still_rejecting and p <= threshold:
            out[key] = True
        else:
            still_rejecting = False
            out[key] = False
    return out


def _fmt_pair(label: str, a: list[bool], b: list[bool]) -> tuple[str, float]:
    """One paired comparison row: accuracies, delta with CI, McNemar p."""
    acc_a, acc_b = float(np.mean(a)), float(np.mean(b))
    delta, lo, hi = bootstrap_delta_ci(a, b, N_BOOTSTRAP, ALPHA)
    n_a_only, n_b_only, p = mcnemar_exact(a, b)
    return (
        f"  {label:34s} {acc_a:6.1%} -> {acc_b:6.1%}  "
        f"delta {delta:+6.1%} [{lo:+.1%}, {hi:+.1%}]  "
        f"disc {n_a_only}/{n_b_only}  p={p:.2e}",
        p,
    )


def report_task(task: str) -> dict[str, float]:
    print(f"\n{'=' * 78}\n{task.upper()} — fine-tuning arms (n={FULL_N[task]}, 0-shot)\n{'=' * 78}")
    pvals: dict[str, float] = {}

    format_table(task)

    print("\n[1] Did fine-tuning help? (C1 tuned vs B untuned, same scheme, same items)")
    for scheme in ALL_SCHEMES:
        b, c1 = load_correctness(task, "B", scheme), load_correctness(task, "C1", scheme)
        if not b or not c1:
            print(f"  {scheme:34s} (pending)")
            continue
        _, (ba, ca) = _aligned(b, c1)
        line, p = _fmt_pair(f"{scheme}: B -> C1", ba, ca)
        print(line)
        pvals[f"{task}|recovery|{scheme}"] = p

        # Same comparison under the superseded extraction rule. Reported, not
        # tested: only the primary rule's p-value enters the Holm family, or
        # the correction would be counting each scheme twice.
        b_old = load_correctness(task, "B", scheme, legacy=True)
        c1_old = load_correctness(task, "C1", scheme, legacy=True)
        if b_old and c1_old:
            _, (bo, co) = _aligned(b_old, c1_old)
            acc_b, acc_c = float(np.mean(bo)), float(np.mean(co))
            if (acc_b, acc_c) != (float(np.mean(ba)), float(np.mean(ca))):
                print(f"  {'(legacy extractor)':34s} {acc_b:6.1%} -> {acc_c:6.1%}  "
                      f"delta {acc_c - acc_b:+6.1%}")

    print("\n[2] Does tuning close the quantization gap? (interaction vs bf16)")
    base_b, base_c1 = load_correctness(task, "B", BASELINE_SCHEME), load_correctness(task, "C1", BASELINE_SCHEME)
    for scheme in QUANTIZED_SCHEMES:
        sb, sc = load_correctness(task, "B", scheme), load_correctness(task, "C1", scheme)
        if not all((base_b, base_c1, sb, sc)):
            print(f"  {scheme:34s} (pending)")
            continue
        _, (bb, bc, xb, xc) = _aligned(base_b, base_c1, sb, sc)
        point, lo, hi = bootstrap_interaction_ci(bb, bc, xb, xc)
        verdict = "excludes 0" if (lo > 0 or hi < 0) else "inconclusive"
        print(f"  {scheme:34s} interaction {point:+6.1%} [{lo:+.1%}, {hi:+.1%}]  {verdict}")

    print("\n[3] Does the fine-tuning survive quantization? (arm comparisons)")
    for scheme in QUANTIZED_SCHEMES:
        c1, c2, d = (load_correctness(task, a, scheme) for a in ("C1", "C2", "D"))
        if c1 and c2:
            _, (x, y) = _aligned(c1, c2)
            line, p = _fmt_pair(f"{scheme}: C1 -> C2 (fuse+requant)", x, y)
            print(line)
            pvals[f"{task}|fuse|{scheme}"] = p
        if c2 and d:
            _, (x, y) = _aligned(c2, d)
            line, p = _fmt_pair(f"{scheme}: C2 -> D (order of ops)", x, y)
            print(line)
            pvals[f"{task}|order|{scheme}"] = p
        if not (c1 and c2 and d):
            print(f"  {scheme:34s} (partially pending)")
    return pvals


def adapter_cost_table() -> None:
    """Adapter size on disk — the part of the cost the IpD metric would miss."""
    print(f"\n{'=' * 78}\nADAPTER COST\n{'=' * 78}")
    metas = sorted(ADAPTERS_DIR.rglob("train_meta.json"))
    if not metas:
        print("  (no adapters trained yet)")
        return
    print(f"  {'task':10s} {'scheme':8s} {'seed':5s} {'adapter MB':>11s} {'train min':>10s}")
    for m in metas:
        d = json.loads(m.read_text())
        # Measured from the shipped file, not from meta: adapters trained
        # before the checkpoint fix recorded the whole directory.
        final = m.parent / "adapters.safetensors"
        shipped = final.stat().st_size if final.exists() else d["adapter_bytes"]
        print(f"  {d['task']:10s} {d['scheme']:8s} {d['seed']:<5d} "
              f"{shipped/1e6:11.1f} {d['train_seconds']/60:10.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=sorted(FT_TASKS))
    args = ap.parse_args()

    tasks = [args.task] if args.task else list(FT_TASKS)
    all_p: dict[str, float] = {}
    for task in tasks:
        all_p.update(report_task(task))

    # Primary family: "did fine-tuning recover capability", one test per
    # scheme per task. The arm comparisons in [3] are exploratory.
    primary = {k: v for k, v in all_p.items() if "|recovery|" in k}
    if primary:
        survives = holm(primary, ALPHA)
        print(f"\n{'=' * 78}\nHOLM-CORRECTED PRIMARY FAMILY (recovery tests, alpha={ALPHA})\n{'=' * 78}")
        for key, p in sorted(primary.items(), key=lambda kv: kv[1]):
            print(f"  {key:44s} p={p:.2e}  {'SIGNIFICANT' if survives[key] else 'not significant'}")

    adapter_cost_table()


if __name__ == "__main__":
    main()
