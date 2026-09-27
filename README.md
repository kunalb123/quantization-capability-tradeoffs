# Quantization-Capability Tradeoffs on Apple Silicon

Which quantization scheme maximizes **intelligence per dollar** on constrained hardware?

Most inference benchmarks measure speed. This project measures what actually matters: **how much capability you lose per bit you save** — across quantization schemes, on a real M-series MacBook.

---

## Research Question

Given a fixed memory budget on Apple Silicon, which quantization method preserves the most model capability? Does AWQ outperform GPTQ at INT4? Where does the capability cliff appear? And is the efficiency gain worth the capability cost for real-world tasks?

---

## Experimental Setup

**Model:** Llama 3.2 3B (MLX)

**Quantization schemes:**
| Scheme | Bits | Method |
|--------|------|--------|
| BF16 | 16 | Baseline (no quantization) |
| INT8 | 8 | Post-training quantization |
| GPTQ | 4 | Post-training, weight-only |
| AWQ | 4 | Activation-aware weight quantization |
| INT4 | 4 | Naive symmetric quantization |

**Hardware:** Apple M-series (unified memory, Metal/MPS backend via MLX)

---

## Eval Suite

Four task categories, each probing a different capability dimension:

### 1. Reasoning — GSM8K (subset)
Grade-school math word problems. Tests multi-step reasoning — the capability most sensitive to quantization-induced degradation.

*Scoring note.* The final answer is taken from `####` if the model emits it, else from after the last `=`, else the last number in the text. The middle step matters more than it sounds: a model that shows its working often closes by restating the question ("...so the turtle crawls 120 feet in 40 seconds"), where the last number is an input, not the answer. Reading after the last `=` recovers it. This lifts verbose arms ~5-6pp and terse ones 0.0pp, so it changes the *size* of any verbose-vs-terse gap — every fine-tuning comparison below is therefore reported under both this rule and the original last-number rule (`evals.reasoning.extract_answer_legacy`).

### 2. Factual Recall — TriviaQA (subset)
Open-domain QA. Tests whether compressed models retain factual associations.

### 3. Instruction Following — IFEval (subset)
Structured instruction compliance (format constraints, length constraints). Tests whether models can follow precise instructions after compression.

### 4. Structured Output — JSON generation
Custom prompts requiring valid, schema-conformant JSON. Tests reliability of structured outputs — critical for production automation use cases.

---

## Metrics

**Capability:**
- Accuracy per task category
- Overall capability retention score (% of BF16 baseline)

**Efficiency:**
- Tokens per second
- Peak memory usage (GB)
- Model load time (s)

**Combined — Intelligence per Dollar proxy:**
```
score = capability_retention / (memory_gb * seconds_per_token)
```

---

## Results

n=500 per task per scheme, greedy decoding, on an M3 Pro (18GB unified memory). Accuracy deltas vs BF16 are tested with McNemar's exact test on paired items; significant results (p<0.05) are in the per-task tables below, not this summary table.

| Scheme | GSM8K | TriviaQA | IFEval | JSON | Tokens/s | Memory (GB) | IpD Score |
|--------|-------|----------|--------|------|----------|-------------|-----------|
| BF16 | 79.8% | 57.4% | 63.4% | 100.0% | 22.8 | 6.75 | 3.4 |
| INT8 | 79.6% | 57.6% | 61.2% | 100.0% | 41.6 | 3.93 | 10.5 |
| AWQ-4 | 73.0% | 55.0% | 60.0% | 97.8% | 72.1 | 2.43 | 28.2 |
| GPTQ-4 | 74.2% | 50.4% | 61.8% | 98.8% | 69.4 | 2.51 | 26.3 |
| INT4 | 70.8% | 49.0% | 60.6% | 100.0% | 72.8 | 2.41 | 28.2 |

IpD Score = capability retention (mean accuracy across all four tasks, as a % of BF16's) / (memory_gb × seconds_per_token) — see [Metrics](#metrics). Full paired McNemar/bootstrap comparisons and plots are reproducible via `python analyze.py`.

---

## Key Findings

- **INT8 is effectively free.** Every task is within noise of BF16 (largest McNemar-significant drop: none), while memory drops ~42% and throughput nearly doubles. There's no capability argument for BF16 over INT8 on this hardware.

- **Capability loss is task-dependent, not uniform across "4-bit."** GSM8K (multi-step reasoning) and TriviaQA (factual recall) both show statistically significant drops at every 4-bit scheme (p<0.01 for GSM8K; p<0.001 for TriviaQA except AWQ-4, which is directionally negative but not significant at this n). IFEval (instruction following) shows no significant drop for any scheme at n=500 — compliance with format/length constraints appears to survive 4-bit quantization better than reasoning or recall does. JSON structural validity is nearly saturated everywhere (97.8–100%), with AWQ-4 and GPTQ-4 showing small but statistically significant slips even at that ceiling.

- **INT4 loses the most capability but wins on IpD anyway.** Naive INT4 has the lowest capability retention of any 4-bit scheme (93.2% vs ~95% for AWQ-4/GPTQ-4) but ties AWQ-4 for the best IpD score (28.2), because its throughput/memory profile is marginally better. This is a real limitation of the IpD proxy as specified — it doesn't penalize *which* capabilities are lost, so a scheme that's cheap and generically dumber can score the same as one that's cheap and more selectively degraded. AWQ-4 is the more defensible pick if GSM8K/TriviaQA accuracy specifically matters for a use case, despite the tied score.

- **Quantization also breaks fine-tuning, silently.** LoRA-tuning a 4-bit model and then fusing + requantizing the adapter discards it outright — the resulting model is statistically indistinguishable from the untuned one, because every weight update is smaller than half a quantization step. At 8 bits the same pipeline is lossless. See [Fine-Tuning Under Quantization](#fine-tuning-under-quantization).

- **Fine-tuning a 4-bit model is itself noisy.** Five LoRA seeds at INT4 span 6.0pp of GSM8K accuracy (60.0-66.0%, sd 2.2pp) against 0.8pp across three BF16 seeds. Large effects still read cleanly through that noise, but single-seed gaps between the 4-bit schemes do not — the INT4 seed range covers both AWQ-4 and GPTQ-4.

- **AWQ vs GPTQ has no clear winner at 4-bit.** AWQ-4 does slightly better on GSM8K and JSON; GPTQ-4 does slightly better on TriviaQA and IFEval. Differences between the two are smaller than either's gap from BF16, and neither dominates the other across tasks.

---

## Fine-Tuning Under Quantization

The benchmark above asks what quantization costs a stock model. This second experiment asks what it costs a model you *adapt*: if you LoRA-tune a quantized model, does the fine-tuning survive, and does the order of quantize/tune/fuse matter?

Four arms, all evaluated 0-shot on the identical 500 items so every comparison is paired (McNemar exact + percentile bootstrap, `python analyze_finetune.py`):

| Arm | What it is |
|-----|------------|
| **B** | untuned, 0-shot — the control every tuned arm is measured against |
| **C1** | LoRA-tuned on the quantized base, adapter kept unfused in BF16 |
| **C2** | C1 fused into the quantized base, then requantized — the "ship one file" path |
| **D** | tuned in BF16, fused, then quantized once |

LoRA r=8, scale 20, 600 iters, lr 1e-5, `mask_prompt`. Seeds 0-2 on GSM8K for bf16, extended to seeds 0-4 at int4 once the seed spread turned out to be large (see [the caveat below](#caveat-int4-fine-tuning-is-seed-unstable)).

### Fuse-then-requantize silently destroys a 4-bit adapter

| GSM8K | C1 | C2 | C1→C2 | B→C2 |
|-------|-----|-----|-------|------|
| int8 | 66.6% | 67.0% | +0.4% (p=0.84) | — |
| awq4 | 65.4% | **75.8%** | +10.4% (p=4e-06) | +1.6% (p=0.37) |
| gptq4 | 64.0% | **73.6%** | +9.6% (p=5e-05) | −1.0% (p=0.64) |
| int4 | 62.6% | **72.0%** | +9.4% (p=1e-04) | −2.0% (p=0.31) |

At 4 bits, C2 is statistically indistinguishable from the *untuned* model — the adapter is gone. At 8 bits it survives intact. The cause is grid step size, not the quantizer: AWQ, GPTQ and uncalibrated affine all land on the same number.

This is measurable directly in the weights, without running the model. For each adapted layer, dequantize the base, compute the LoRA delta, and compare it to the quantization step:

| adapter | scheme | bits | median \|delta\|/step | weights below ½ step | delta retained |
|---------|--------|------|------------------|---------------------|----------------|
| gsm8k | int8 | 8 | 0.716 | 36.7% | 99.9% |
| gsm8k | awq4 | 4 | 0.047 | 100.0% | 9.6% |
| gsm8k | int4 | 4 | 0.042 | 100.0% | 12.6% |
| triviaqa | int8 | 8 | 0.647 | 40.3% | 99.8% |
| triviaqa | int4 | 4 | 0.038 | 100.0% | 12.6% |

At 4 bits, *every* weight's update is smaller than half a step, so re-rounding returns it to where it started. int8's step is ~17× finer and the same delta clears it. Retention is measured as ⟨W_after − W_base, delta⟩ / ⟨delta, delta⟩ — a projection, because adding any delta perturbs each group's recomputed scale and nudges every weight by an amount unrelated to the adapter; that noise swamps norm-ratio or unchanged-weight statistics (a *zero* delta already leaves only ~85% of 4-bit weights bit-identical). Against a synthetic calibration curve, the ~10-13% figures sit at the estimator's floor and are consistent with total loss.

### If you must ship one file, tune in BF16 and quantize last

Arm D keeps the fine-tuning at 4 bits, but pays for it:

| GSM8K | C1 | D | C1→D |
|-------|-----|-----|------|
| int8 | 66.6% | 67.4% | +0.8% (p=0.50) |
| awq4 | 65.4% | 62.6% | −2.8% (p=0.16) |
| gptq4 | 64.0% | 59.6% | −4.4% (p=0.03) |
| int4 | 62.6% | 57.6% | −5.0% (p=0.01) |

So the honest ranking for a single-artifact 4-bit deployment is D (keeps the tuning, loses a few points) over C2 (loses the tuning entirely, and *looks* better for it here — see below).

### The scorer decided the sign of the conclusion

On TriviaQA, the same broken C2 pipeline appears to be the *best* arm: 58.6% vs C1's 52.6% at AWQ-4, +9 to +13pp over untuned. That is an artifact. `factual.score()` counts a completion correct if any gold alias appears anywhere in it, and C2's half-reverted format is verbose (17-20 tokens, against 4 tuned and 27 untuned) — more words, more chances to contain the answer. Under exact match the ordering inverts:

| TriviaQA, C2 vs C1 | substring (as run) | exact match |
|---|---|---|
| awq4 | +6.0% | **−33.6%** |
| gptq4 | +9.2% | **−36.8%** |
| int4 | +4.4% | **−32.0%** |

The weight-level check had already said the TriviaQA and GSM8K adapters are eroded identically (0.038 vs 0.042 delta/step). There is no task asymmetry in the mechanism — only in whether the scorer exposed it. A practitioner measuring end-to-end accuracy with a substring scorer would conclude this pipeline was their best configuration.

### Fine-tuning hurt GSM8K at these hyperparameters

Every scheme regressed, Holm-corrected significant across the primary family:

| GSM8K | B | C1 | delta | legacy extractor |
|-------|-----|-----|-------|------------------|
| bf16 | 76.4% | 67.4% | −9.0% (p=7e-05) | −3.8% |
| int8 | 76.4% | 66.6% | −9.8% (p=3e-05) | −4.6% |
| awq4 | 74.2% | 65.4% | −8.8% (p=3e-04) | −2.8% |
| gptq4 | 74.6% | 64.0% | −10.6% (p=1e-05) | −5.0% |
| int4 | 74.0% | 62.6% | −11.4% (p=5e-06) | −6.0% |

Generation collapses from ~165 tokens to ~96: training on GSM8K's terse gold solutions teaches the model to stop showing its working, and multi-step arithmetic needs the working. This is a statement about LoRA at *these* hyperparameters on this data, not about fine-tuning GSM8K in general. TriviaQA, whose gold answers are short by nature, moved the other way (+3 to +6pp, significant only at int4 after correction).

Note how much of the apparent effect size rides on the scoring rule: the same regression is −2.8 to −6.0pp under the original last-number extractor. Sign and significance hold either way; the magnitude does not.

### Caveat: int4 fine-tuning is seed-unstable

C1 accuracy across seeds (GSM8K, n=500 each):

| scheme | s0 | s1 | s2 | s3 | s4 | spread |
|--------|----|----|----|----|----|--------|
| bf16 | 67.4% | 66.8% | 67.6% | — | — | 0.8pp |
| int4 | 62.6% | 63.0% | 66.0% | 60.0% | 64.0% | **6.0pp** (sd 2.2pp) |

Five seeds at int4 span 60.0-66.0%, against 0.8pp of spread across three bf16 seeds. Tuning a 4-bit model is a far noisier process than tuning the same model in BF16, which is worth knowing in its own right: it means a single 4-bit fine-tuning run tells you much less than the same run at BF16.

Two consequences for reading the tables above:

- **The regression survives.** B→C1 at int4 is negative at every seed, from −8.0pp to −14.0pp against an untuned 74.0%. The finding does not depend on seed choice.
- **The 4-bit scheme ordering does not.** int4's seed range covers both awq4 (65.4%) and gptq4 (64.0%), which are single-seed numbers. Any claim that one 4-bit scheme fine-tunes better than another is unsupported at this n — separating them would need several seeds per scheme, not one.


---

## Stack

- [`mlx`](https://github.com/ml-explore/mlx) — Apple Silicon native ML framework
- [`mlx-lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) — LLM loading and inference on MLX
- `datasets` — HuggingFace datasets for GSM8K, TriviaQA, IFEval
- `pandas` / `matplotlib` — results analysis and plotting

---

## Setup

```bash
git clone https://github.com/kunalb123/quantization-capability-tradeoffs
cd quantization-capability-tradeoffs
pip install -r requirements.txt

# Download and quantize models
python models/download.py

# Run full benchmark
python benchmark.py

# Generate analysis and plots
python analyze.py

# Fine-tuning experiment (unattended, resumable; ~14 hrs on an M3 Pro)
python -m finetune.run_all --dry-run   # inspect the job plan first
python -m finetune.run_all

# Paired analysis of the fine-tuning arms
python analyze_finetune.py
python analyze_finetune.py --task gsm8k

# The two mechanism checks behind the fine-tuning findings
python verify_doublequant.py            # LoRA delta vs quantization step
python verify_doublequant.py --calibrate-only   # synthetic reference curve only
python verify_scorer_effect.py          # substring vs exact match on TriviaQA
```

---

## Project Structure

```
quantization-capability-tradeoffs/
├── README.md
├── requirements.txt
├── benchmark.py          # Main runner — evaluates all schemes across all tasks
├── analyze.py            # Generates plots and summary report
├── analyze_finetune.py   # Paired analysis of the fine-tuning arms (B/C1/C2/D)
├── verify_doublequant.py # Weight-level check: LoRA delta vs quantization step
├── verify_scorer_effect.py # Substring vs exact match on the TriviaQA arms
├── models/
│   └── download.py       # Downloads and prepares quantized model variants
├── finetune/
│   ├── run_all.py        # Unattended orchestrator — resumable, fail-soft, disk-bounded
│   ├── run_one.py        # One (task, scheme, arm, seed) eval per process
│   ├── train.py          # LoRA training
│   ├── build.py          # Fuse / requantize the C2 and D artifacts
│   └── data.py           # Builds the 0-shot training sets
├── evals/
│   ├── __init__.py
│   ├── reasoning.py      # GSM8K evaluation
│   ├── factual.py        # TriviaQA evaluation
│   ├── instruction.py    # IFEval evaluation
│   └── structured.py     # JSON generation evaluation
└── results/              # Output CSVs and plots (gitignored except summary)
```

---

## Motivation

Inspired by the "intelligence per dollar" framing: as AI moves to edge devices and constrained hardware, the question isn't just *can we run this model* but *how much of its intelligence survives the journey*. This project is a systematic attempt to answer that with real measurements rather than theoretical bounds.

Prior work in my [cpu-deployable-llms](https://github.com/kunalb123/cpu-deployable-llms) repo benchmarked inference speed and throughput. This project extends that to measure the *capability* dimension — what the model can actually do at each compression level.

---

## References

- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [GPTQ: Accurate Post-training Quantization](https://arxiv.org/abs/2210.17323)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Scaling Laws for Quantization](https://arxiv.org/abs/2411.04330)
