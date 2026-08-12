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
| GPTQ-4 | 74.4% | 50.4% | 61.8% | 98.8% | 69.4 | 2.51 | 26.3 |
| INT4 | 70.6% | 49.0% | 60.6% | 100.0% | 72.8 | 2.41 | 28.2 |

IpD Score = capability retention (mean accuracy across all four tasks, as a % of BF16's) / (memory_gb × seconds_per_token) — see [Metrics](#metrics). Full paired McNemar/bootstrap comparisons and plots are reproducible via `python analyze.py`.

---

## Key Findings

- **INT8 is effectively free.** Every task is within noise of BF16 (largest McNemar-significant drop: none), while memory drops ~42% and throughput nearly doubles. There's no capability argument for BF16 over INT8 on this hardware.

- **Capability loss is task-dependent, not uniform across "4-bit."** GSM8K (multi-step reasoning) and TriviaQA (factual recall) both show statistically significant drops at every 4-bit scheme (p<0.01 for GSM8K; p<0.001 for TriviaQA except AWQ-4, which is directionally negative but not significant at this n). IFEval (instruction following) shows no significant drop for any scheme at n=500 — compliance with format/length constraints appears to survive 4-bit quantization better than reasoning or recall does. JSON structural validity is nearly saturated everywhere (97.8–100%), with AWQ-4 and GPTQ-4 showing small but statistically significant slips even at that ceiling.

- **INT4 loses the most capability but wins on IpD anyway.** Naive INT4 has the lowest capability retention of any 4-bit scheme (93.2% vs ~95% for AWQ-4/GPTQ-4) but ties AWQ-4 for the best IpD score (28.2), because its throughput/memory profile is marginally better. This is a real limitation of the IpD proxy as specified — it doesn't penalize *which* capabilities are lost, so a scheme that's cheap and generically dumber can score the same as one that's cheap and more selectively degraded. AWQ-4 is the more defensible pick if GSM8K/TriviaQA accuracy specifically matters for a use case, despite the tied score.

- **AWQ vs GPTQ has no clear winner at 4-bit.** AWQ-4 does slightly better on GSM8K and JSON; GPTQ-4 does slightly better on TriviaQA and IFEval. Differences between the two are smaller than either's gap from BF16, and neither dominates the other across tasks.

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
```

---

## Project Structure

```
quantization-capability-tradeoffs/
├── README.md
├── requirements.txt
├── benchmark.py          # Main runner — evaluates all schemes across all tasks
├── analyze.py            # Generates plots and summary report
├── models/
│   └── download.py       # Downloads and prepares quantized model variants
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
