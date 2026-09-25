"""Numerical check of the double-quantization mechanism.

The accuracy evidence says arm C2 (fuse a LoRA adapter into a quantized base,
then requantize) loses the fine-tuning at 4 bits and keeps it at 8. The claimed
mechanism is grid step size: mlx_lm's fuse dequantizes the base, adds the LoRA
delta, and re-rounds onto a grid whose spacing is the per-group scale — so a
delta well under a step rounds back to where it started and is discarded rather
than degraded. Bit width is supposed to matter only through that step size.

Measured per scheme:

  delta/step      median |LoRA delta| as a fraction of the quantization step.
                  Reported per task adapter: the GSM8K adapter is erased by
                  4-bit requantization (arm C2 is indistinguishable from
                  untuned) while the TriviaQA one largely survives (C2 beats
                  untuned by 9-13pp), so if step size is really the mechanism
                  these two must land at different points on the curve below.
                  This is the mechanism itself, and needs no requantization.
  below_half      fraction of weights whose delta is under half a step.
  kept            how much of the delta actually survives fuse-and-requantize,
                  as <W_after - W_base, delta> / <delta, delta>.

Why `kept` is a projection and not a norm ratio or an unchanged-weight count:
adding any delta perturbs each group's min/max, hence its recomputed scale,
which nudges every weight in the group by an amount unrelated to the delta.
That noise dominates both of the simpler statistics (a zero delta already
leaves only ~85% of 4-bit weights bit-identical, purely from float
tie-breaking). Projecting onto the delta cancels it, since the perturbation is
uncorrelated with the delta and there are ~10^8 weights to average over.

calibrate() prints the same statistic on synthetic data where delta/step is
known, so the model numbers can be read against a reference curve rather than
in isolation. Note that MLX stores affine `scales` signed — abs() matters, and
omitting it makes the mean of a ratio cancel to nearly zero.
"""

import glob
import json
import sys

import mlx.core as mx

SCHEMES = ["int8", "awq4", "gptq4", "int4"]
TASKS = ["gsm8k", "triviaqa"]


def seed_dir(task):
    return f"finetune/adapters/{task}/seed0"


def _roundtrip(x, group_size, bits, mode):
    packed, scales, biases = mx.quantize(x, group_size=group_size, bits=bits, mode=mode)
    return mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits, mode=mode)


def analyze(scheme, task):
    weights = {}
    for path in sorted(glob.glob(f"models/weights/{scheme}/*.safetensors")):
        weights.update(mx.load(path))
    qcfg = json.load(open(f"models/weights/{scheme}/config.json"))["quantization"]
    bits, group_size, mode = qcfg["bits"], qcfg["group_size"], qcfg.get("mode", "affine")

    adapter = mx.load(f"{seed_dir(task)}/{scheme}/adapters.safetensors")
    lora_scale = json.load(open(f"{seed_dir(task)}/{scheme}/adapter_config.json"))["lora_parameters"]["scale"]

    ratios = []          # |delta|/step samples, subsampled to keep memory flat
    n_below = n_total = 0
    proj_num = proj_den = 0.0

    for key in sorted(adapter):
        if not key.endswith(".lora_a"):
            continue
        prefix = key[: -len(".lora_a")]
        if f"{prefix}.scales" not in weights:
            continue  # layer not quantized

        packed, scales, biases = (weights[f"{prefix}.{p}"] for p in ("weight", "scales", "biases"))
        w = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits, mode=mode)
        delta = (lora_scale * (adapter[key] @ adapter[f"{prefix}.lora_b"]).T).astype(w.dtype)

        # affine scales are one signed value per group of `group_size`
        # consecutive columns; abs() because the sign is a representation
        # detail, not part of the spacing.
        step = mx.abs(mx.repeat(scales.astype(mx.float32), group_size, axis=1))[:, : w.shape[1]]

        abs_delta = mx.abs(delta.astype(mx.float32))
        ratio = abs_delta / step
        n_below += float(mx.sum(ratio < 0.5))
        n_total += w.size
        ratios.append(ratio.reshape(-1)[::997])   # ~0.1% stratified sample

        moved = (_roundtrip(w + delta, group_size, bits, mode) - w).astype(mx.float32)
        d = delta.astype(mx.float32)
        proj_num += float(mx.sum(moved * d))
        proj_den += float(mx.sum(d * d))
        mx.clear_cache()

    sample = mx.concatenate(ratios)
    return {
        "bits": bits,
        "median_ratio": float(mx.median(sample)) if hasattr(mx, "median") else float(mx.mean(sample)),
        "mean_ratio": float(mx.mean(sample)),
        "below_half": n_below / n_total,
        "kept": proj_num / proj_den,
        "n_weights": n_total,
    }


def calibrate():
    """Reference curve: kept vs a known delta/step, on synthetic weights."""
    mx.random.seed(0)
    w = mx.random.normal((256, 3072))
    print("calibration (synthetic, random delta of known size)")
    print(f"  {'delta/step':>11s} {'kept@4bit':>10s} {'kept@8bit':>10s}")
    rows = {}
    for bits in (4, 8):
        packed, scales, biases = mx.quantize(w, group_size=64, bits=bits, mode="affine")
        wq = mx.dequantize(packed, scales, biases, group_size=64, bits=bits, mode="affine")
        step = mx.abs(mx.repeat(scales.astype(mx.float32), 64, axis=1))
        for frac in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
            d = (frac * step * mx.random.normal(w.shape)).astype(wq.dtype)
            moved = (_roundtrip(wq + d, 64, bits, "affine") - wq).astype(mx.float32)
            df = d.astype(mx.float32)
            rows.setdefault(frac, {})[bits] = float(mx.sum(moved * df) / mx.sum(df * df))
    for frac, by_bits in rows.items():
        print(f"  {frac:11.2f} {by_bits[4]:10.1%} {by_bits[8]:10.1%}")


def main():
    args = sys.argv[1:]
    if "--calibrate-only" in args:
        calibrate()
        return
    calibrate()
    # Both task adapters: the accuracy evidence says the GSM8K adapter is
    # erased by 4-bit requantization while the TriviaQA one substantially
    # survives, so their deltas should sit at different points on the
    # calibration curve. Same base weights, same steps — only the adapter
    # differs, which is what makes this a test rather than a description.
    print(f"\n{'task':10s} {'scheme':8s} {'bits':>5s} {'delta/step':>11s} "
          f"{'<half step':>11s} {'kept':>8s}")
    for task in TASKS:
        for scheme in (args or SCHEMES):
            r = analyze(scheme, task)
            print(f"{task:10s} {scheme:8s} {r['bits']:5d} {r['median_ratio']:11.3f} "
                  f"{r['below_half']:10.1%} {r['kept']:7.1%}")
            mx.clear_cache()


if __name__ == "__main__":
    main()
