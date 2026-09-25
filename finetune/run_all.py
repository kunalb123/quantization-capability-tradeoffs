"""Unattended orchestrator for the fine-tuning experiment.

Runs the whole plan — data prep, training, fusing, requantizing, evaluating —
as one detached process that needs nothing from the caller once started.

Design constraints this exists to satisfy:

  Resumable. Every unit of work has a stable key and is skipped if its output
  already exists (adapters by file, evals by run_items' per-item resume). A
  kill -9 at hour six costs at most the job in flight.

  Fail-soft. A job that raises is recorded and the run continues to the next
  one, because a single broken arm shouldn't cost the remaining twelve hours.
  Failures are summarised at the end and kept in the state file.

  Ordered by value. Jobs run cheapest-and-most-decisive first: the untuned
  0-shot control (no training at all), then TriviaQA end to end (~1 min per
  eval), then GSM8K's arms, then the seed-noise replicates. Stopping the run
  at any point leaves a coherent, analysable subset rather than a scatter of
  half-finished arms.

  Disk-bounded. Arms C2 and D each materialise multi-GB models; each is
  deleted as soon as the eval that needs it has been recorded, so peak usage
  is one or two builds rather than dozens.

Usage:
    python -m finetune.run_all                 # everything, in priority order
    python -m finetune.run_all --dry-run       # print the job list and exit
    python -m finetune.run_all --stages control triviaqa
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config import (
    BUILDS_DIR, FT_SEEDS, FT_TASKS, FULL_N, QUANTIZED_SCHEMES, RESULTS_DIR,
    SCHEMES, SEED_NOISE_SCHEMES,
)
from finetune import build as build_mod
from finetune import data as data_mod
from finetune import train as train_mod

STATE_PATH = RESULTS_DIR / "ft_run_state.json"
ALL_SCHEMES = [s.name for s in SCHEMES]
PRIMARY_SEED = 0


@dataclass
class Job:
    key: str
    stage: str
    kind: str          # control | train | eval | c2 | d
    task: str
    scheme: str
    seed: int = PRIMARY_SEED
    note: str = ""


def plan(extra_seeds: tuple[int, ...] = ()) -> list[Job]:
    """The full job list, in execution order.

    extra_seeds adds further GSM8K seed replicates beyond config.FT_SEEDS.
    They are a follow-up to the main experiment, not part of its design, so
    they are passed in rather than written into FT_SEEDS — that constant
    should keep describing what the headline run actually was.

    Arm B first because it needs no training and is the control every tuned
    comparison is measured against — if it fails, nothing downstream is
    interpretable, so it should fail in minute five rather than hour nine.
    """
    jobs: list[Job] = []

    # TriviaQA first, complete: control then every tuned arm. It finishes in
    # ~2 hrs, so one entire task is banked — and every code path exercised at
    # full n — before GSM8K's much longer evals begin.
    for scheme in ALL_SCHEMES:
        jobs.append(Job("B|triviaqa|" + scheme, "control", "control", "triviaqa", scheme))
    jobs += _task_jobs("triviaqa", "triviaqa", PRIMARY_SEED)

    # GSM8K: the headline task, and the expensive one. Its untuned 0-shot
    # control alone is ~5 hrs, because an untuned model prompted 0-shot writes
    # far longer completions than the few-shot prompting it replaces (measured:
    # ~6x on TriviaQA). Capping max_tokens to speed this up would truncate
    # chain-of-thought and bias the control downward, which would flatter every
    # tuned arm — so the control pays full price.
    for scheme in ALL_SCHEMES:
        jobs.append(Job("B|gsm8k|" + scheme, "control", "control", "gsm8k", scheme))
    jobs += _task_jobs("gsm8k", "gsm8k", PRIMARY_SEED)

    # Stage 6 - seed replicates, to size training noise against scheme gaps.
    # C1 only, and only at the two ends of the quantization range.
    seed_replicates = [s for s in FT_SEEDS["gsm8k"] if s != PRIMARY_SEED]
    seed_replicates += [s for s in extra_seeds if s not in seed_replicates]
    for seed in seed_replicates:
        for scheme in SEED_NOISE_SCHEMES:
            jobs.append(Job(f"train|gsm8k|{scheme}|{seed}", "seeds", "train",
                            "gsm8k", scheme, seed))
            jobs.append(Job(f"C1|gsm8k|{scheme}|{seed}", "seeds", "eval",
                            "gsm8k", scheme, seed))
    return jobs


def _task_jobs(task: str, stage_prefix: str, seed: int) -> list[Job]:
    """Train + C1 + C2 + D for one task at one seed, in that order.

    C1 comes before C2/D because it needs no build step: if fusing or
    requantizing turns out to be broken, the unfused results are already
    banked.
    """
    jobs = []
    for scheme in ALL_SCHEMES:
        jobs.append(Job(f"train|{task}|{scheme}|{seed}", f"{stage_prefix}-train",
                        "train", task, scheme, seed))
    for scheme in ALL_SCHEMES:
        jobs.append(Job(f"C1|{task}|{scheme}|{seed}", f"{stage_prefix}-C1",
                        "eval", task, scheme, seed))
    for scheme in QUANTIZED_SCHEMES:
        jobs.append(Job(f"C2|{task}|{scheme}|{seed}", f"{stage_prefix}-C2",
                        "c2", task, scheme, seed))
    for scheme in QUANTIZED_SCHEMES:
        jobs.append(Job(f"D|{task}|{scheme}|{seed}", f"{stage_prefix}-D",
                        "d", task, scheme, seed))
    return jobs


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def _eval(task: str, scheme: str, variant: str, *, adapter=None, model=None) -> None:
    cmd = [sys.executable, "-m", "finetune.run_one",
           "--task", task, "--scheme", scheme, "--variant", variant]
    if adapter:
        cmd += ["--adapter-path", str(adapter)]
    if model:
        cmd += ["--model-path", str(model)]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def execute(job: Job) -> None:
    data_dir = data_mod.build(job.task, FULL_N[job.task])

    if job.kind == "control":
        _eval(job.task, job.scheme, "B")

    elif job.kind == "train":
        train_mod.train(job.task, job.scheme, job.seed, data_dir)

    elif job.kind == "eval":  # C1: quantized base + unfused adapter
        adapter = train_mod.adapter_dir(job.task, job.scheme, job.seed)
        _eval(job.task, job.scheme, f"C1/seed{job.seed}", adapter=adapter)

    elif job.kind == "c2":  # fuse back into the quantized base, requantizing
        adapter = train_mod.adapter_dir(job.task, job.scheme, job.seed)
        model = build_mod.build_c2(job.task, job.scheme, job.seed, adapter)
        try:
            _eval(job.task, job.scheme, f"C2/seed{job.seed}", model=model)
        finally:
            build_mod.discard(model)

    elif job.kind == "d":  # tune bf16, then quantize the fine-tuned model
        bf16_adapter = train_mod.adapter_dir(job.task, "bf16", job.seed)
        fused = build_mod.build_d_fused_bf16(job.task, job.seed, bf16_adapter)
        model = build_mod.build_d_quantized(job.task, job.scheme, job.seed, fused)
        try:
            _eval(job.task, job.scheme, f"D/seed{job.seed}", model=model)
        finally:
            build_mod.discard(model)
            if job.scheme == QUANTIZED_SCHEMES[-1]:
                # last D variant for this (task, seed) — the 6 GB bf16 fused
                # model has no remaining consumer.
                build_mod.discard(fused)
    else:
        raise ValueError(f"unknown job kind {job.kind!r}")


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"done": {}, "failed": {}, "started": None}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stages", nargs="+", default=None,
                    help="only run jobs whose stage is listed")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt jobs previously recorded as failed")
    ap.add_argument("--extra-seeds", nargs="+", type=int, default=(),
                    help="additional GSM8K seed replicates beyond config.FT_SEEDS")
    ap.add_argument("--schemes", nargs="+", default=None,
                    help="only run jobs for these schemes")
    args = ap.parse_args()

    jobs = plan(tuple(args.extra_seeds))
    if args.stages:
        jobs = [j for j in jobs if j.stage in set(args.stages)]
    if args.schemes:
        jobs = [j for j in jobs if j.scheme in set(args.schemes)]

    if args.dry_run:
        for j in jobs:
            print(f"{j.stage:18s} {j.kind:8s} {j.key}")
        print(f"\n{len(jobs)} jobs")
        return

    state = _load_state()
    state["started"] = state.get("started") or datetime.now(timezone.utc).isoformat()
    if args.retry_failed:
        state["failed"] = {}
    _save_state(state)

    total = len(jobs)
    for i, job in enumerate(jobs, 1):
        if job.key in state["done"]:
            continue
        if job.key in state["failed"] and not args.retry_failed:
            print(f"[{i}/{total}] skipping previously failed {job.key}", flush=True)
            continue

        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n===== [{i}/{total}] {stamp} {job.stage} :: {job.key} =====", flush=True)
        start = time.perf_counter()
        try:
            execute(job)
        except Exception:
            state["failed"][job.key] = {
                "stage": job.stage,
                "when": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc()[-2000:],
            }
            print(f"!!! FAILED {job.key}\n{traceback.format_exc()}", flush=True)
        else:
            state["done"][job.key] = {
                "stage": job.stage,
                "minutes": round((time.perf_counter() - start) / 60, 1),
                "when": datetime.now(timezone.utc).isoformat(),
            }
            print(f"--- done {job.key} in "
                  f"{state['done'][job.key]['minutes']:.1f} min", flush=True)
        _save_state(state)

    done, failed = len(state["done"]), len(state["failed"])
    print(f"\n===== run finished: {done} done, {failed} failed =====", flush=True)
    for key, info in state["failed"].items():
        print(f"  FAILED {key} ({info['stage']})", flush=True)


if __name__ == "__main__":
    main()
