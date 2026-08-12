"""Paired statistical analysis of the benchmark results in results/.

Raw per-scheme accuracy (as printed by benchmark.py) doesn't answer the real
question — whether a scheme's drop from bf16 is a real, item-level capability
loss or just noise. Because every scheme was run on the identical set of
items (same "id"s per task), each scheme can be compared against
config.BASELINE_SCHEME on exactly the same items:

  - McNemar's exact test on the discordant pairs (bf16-right/scheme-wrong vs
    bf16-wrong/scheme-right) — the right test for paired binary outcomes,
    unlike a two-sample proportion test which would ignore the pairing.
  - A bootstrap CI (resampling item ids, config.N_BOOTSTRAP times) on the
    accuracy delta (scheme - baseline), for an effect-size estimate with
    uncertainty rather than just a significant/not-significant verdict.

After the per-task paired tests, an efficiency/IpD summary is printed:
capability retention (mean accuracy across all four tasks, as a fraction of
bf16's) combined with tokens/sec, peak memory, and load time — all already
recorded per-item in the results JSONL — into the README's "intelligence per
dollar" proxy: capability_retention / (memory_gb * seconds_per_token).

Usage:
    python analyze.py                # every task, every non-baseline scheme
    python analyze.py --task gsm8k
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless — this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from config import ALPHA, BASELINE_SCHEME, FULL_N, N_BOOTSTRAP, RESULTS_DIR, SCHEMES, SEED, TASKS
from evals import factual, instruction, reasoning, structured

PLOTS_DIR = RESULTS_DIR / "plots"

TASK_MODULES = {
    "gsm8k": reasoning,
    "triviaqa": factual,
    "ifeval": instruction,
    "json": structured,
}


def _load_correctness(task: str, scheme_name: str, gold_by_id: dict) -> dict[str, bool]:
    """id -> correct, for one task/scheme results file."""
    path = RESULTS_DIR / task / f"{scheme_name}.jsonl"
    task_module = TASK_MODULES[task]
    correctness = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        correctness[row["id"]] = task_module.score(row["output"], gold_by_id[row["id"]])
    return correctness


def mcnemar_exact(baseline: list[bool], scheme: list[bool]) -> tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes.

    Returns (n_baseline_only, n_scheme_only, p_value). Only discordant pairs
    (exactly one of the two got it right) carry information; under the null
    each discordant pair is equally likely to favor either side, so the
    discordant counts follow Binomial(n_discordant, 0.5).
    """
    baseline_only = sum(b and not s for b, s in zip(baseline, scheme))
    scheme_only = sum(s and not b for b, s in zip(baseline, scheme))
    n_discordant = baseline_only + scheme_only
    if n_discordant == 0:
        return baseline_only, scheme_only, 1.0
    p_value = binomtest(
        min(baseline_only, scheme_only), n_discordant, 0.5, alternative="two-sided"
    ).pvalue
    return baseline_only, scheme_only, p_value


def bootstrap_delta_ci(
    baseline: list[bool], scheme: list[bool], n_bootstrap: int, alpha: float
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for accuracy(scheme) - accuracy(baseline).

    Resamples item indices (with replacement) so each bootstrap draw keeps
    the baseline/scheme pairing on the same items intact.
    """
    rng = np.random.default_rng(SEED)
    baseline_arr = np.asarray(baseline, dtype=float)
    scheme_arr = np.asarray(scheme, dtype=float)
    n = len(baseline_arr)

    point_delta = scheme_arr.mean() - baseline_arr.mean()

    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        deltas[i] = scheme_arr[idx].mean() - baseline_arr[idx].mean()

    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point_delta, lo, hi


def analyze_task(task: str) -> dict[str, float]:
    """Print the paired comparison table for one task; return {scheme: accuracy}."""
    task_module = TASK_MODULES[task]

    scheme_names = [s.name for s in SCHEMES]
    for name in scheme_names:
        path = RESULTS_DIR / task / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing results file: {path}")

    # gold answers come from the eval module's own item generation, which is
    # deterministic (same SEED), so re-loading here reproduces exactly the
    # items the benchmark run used.
    items = task_module.load_items(FULL_N[task])
    gold_by_id = {item["id"]: item["gold"] for item in items}

    correctness_by_scheme = {}

    for name in scheme_names:
        correctness_by_scheme[name] = _load_correctness(task, name, gold_by_id)

    ids = sorted(correctness_by_scheme[BASELINE_SCHEME])
    baseline = [correctness_by_scheme[BASELINE_SCHEME][i] for i in ids]
    baseline_acc = sum(baseline) / len(baseline)
    accuracy_by_scheme = {BASELINE_SCHEME: baseline_acc}

    print(f"\n=== {task} (n={len(ids)}, baseline={BASELINE_SCHEME}: {baseline_acc:.1%}) ===")
    print(f"{'scheme':<8} {'acc':>7} {'delta':>8} {'95% CI':>18} {'discordant (b/s)':>18} {'p':>10}")

    for name in scheme_names:
        if name == BASELINE_SCHEME:
            continue
        scheme = [correctness_by_scheme[name][i] for i in ids]
        scheme_acc = sum(scheme) / len(scheme)
        accuracy_by_scheme[name] = scheme_acc

        baseline_only, scheme_only, p_value = mcnemar_exact(baseline, scheme)
        point_delta, lo, hi = bootstrap_delta_ci(baseline, scheme, N_BOOTSTRAP, ALPHA)

        sig = "*" if p_value < ALPHA else " "
        print(
            f"{name:<8} {scheme_acc:>6.1%} {point_delta:>+7.1%} "
            f"[{lo:>+6.1%}, {hi:>+6.1%}] {baseline_only:>8}/{scheme_only:<8} "
            f"{p_value:>9.4f}{sig}"
        )

    return accuracy_by_scheme


def _load_efficiency(task: str, scheme_name: str) -> tuple[float, float, float]:
    """Mean (generation_tps, peak_memory_gb, load_time_s) for one task/scheme
    results file. load_time_s and peak_memory_gb are constant across a run's
    lines (captured once, at model load) so the mean just recovers that
    single value; generation_tps varies per item so it's a genuine average.
    """
    path = RESULTS_DIR / task / f"{scheme_name}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    tps = np.mean([r["generation_tps"] for r in rows])
    memory_gb = np.mean([r["peak_memory_gb"] for r in rows])
    load_time_s = np.mean([r["load_time_s"] for r in rows])
    return tps, memory_gb, load_time_s


def efficiency_and_ipd_summary(
    tasks: list[str], accuracy_by_task: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Print tokens/s, peak memory, and the README's IpD proxy per scheme.

    capability_retention is mean accuracy across `tasks`, as a fraction of
    bf16's mean accuracy over the same tasks — so it only reflects tasks
    actually analyzed this run (e.g. under --task, retention is single-task).
    IpD = capability_retention / (memory_gb * seconds_per_token).

    Returns {scheme_name: {retention, tps, memory_gb, load_time_s, ipd}} so
    generate_plots() doesn't have to recompute or re-read the JSONL files.
    """
    baseline_mean_acc = np.mean([accuracy_by_task[t][BASELINE_SCHEME] for t in tasks])

    print(f"\n=== Efficiency & Intelligence-per-Dollar (baseline={BASELINE_SCHEME}) ===")
    print(f"{'scheme':<8} {'retention':>10} {'tok/s':>8} {'mem (GB)':>9} {'load (s)':>9} {'IpD':>8}")

    metrics_by_scheme = {}
    for scheme in SCHEMES:
        name = scheme.name
        mean_acc = np.mean([accuracy_by_task[t][name] for t in tasks])
        retention = mean_acc / baseline_mean_acc

        tps_by_task, mem_by_task, load_by_task = zip(
            *(_load_efficiency(t, name) for t in tasks)
        )
        tps = float(np.mean(tps_by_task))
        memory_gb = float(np.mean(mem_by_task))
        load_time_s = float(np.mean(load_by_task))

        seconds_per_token = 1.0 / tps
        ipd = retention / (memory_gb * seconds_per_token)
        metrics_by_scheme[name] = {
            "retention": retention, "tps": tps, "memory_gb": memory_gb,
            "load_time_s": load_time_s, "ipd": ipd,
        }

        print(
            f"{name:<8} {retention:>9.1%} {tps:>8.1f} {memory_gb:>9.2f} "
            f"{load_time_s:>9.1f} {ipd:>8.1f}"
        )

    return metrics_by_scheme


def generate_plots(
    tasks: list[str],
    accuracy_by_task: dict[str, dict[str, float]],
    metrics_by_scheme: dict[str, dict[str, float]],
) -> None:
    """Save three PNGs to results/plots/: per-task accuracy, capability
    retention, and IpD score, all grouped by scheme in config.SCHEMES order
    (bf16 -> int8 -> awq4 -> gptq4 -> int4) so bar order is consistent
    across figures and matches the README table's row order.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    scheme_names = [s.name for s in SCHEMES]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(scheme_names)))

    # accuracy per task, grouped by scheme
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(tasks))
    width = 0.8 / len(scheme_names)
    for i, name in enumerate(scheme_names):
        accs = [accuracy_by_task[t][name] * 100 for t in tasks]
        ax.bar(x + i * width, accs, width, label=name, color=colors[i])
    ax.set_xticks(x + width * (len(scheme_names) - 1) / 2)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by task and quantization scheme")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_by_task.png", dpi=150)
    plt.close(fig)

    # capability retention vs IpD score
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    retentions = [metrics_by_scheme[n]["retention"] * 100 for n in scheme_names]
    ax1.bar(scheme_names, retentions, color=colors)
    ax1.set_ylabel("Capability retention (% of bf16)")
    ax1.set_title("Capability retention")
    ax1.set_ylim(0, 105)

    ipds = [metrics_by_scheme[n]["ipd"] for n in scheme_names]
    ax2.bar(scheme_names, ipds, color=colors)
    ax2.set_ylabel("IpD score")
    ax2.set_title("Intelligence per dollar")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "retention_and_ipd.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved plots to {PLOTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), help="defaults to every task")
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(TASKS)
    accuracy_by_task = {task: analyze_task(task) for task in tasks}
    metrics_by_scheme = efficiency_and_ipd_summary(tasks, accuracy_by_task)
    generate_plots(tasks, accuracy_by_task, metrics_by_scheme)


if __name__ == "__main__":
    main()
