from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

SEED = 42

# Ungated mirror — swap to "meta-llama/Llama-3.2-3B-Instruct" if you'd rather
# use the official repo (requires accepting the license + an HF token).
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-bf16"


@dataclass(frozen=True)
class QuantScheme:
    name: str
    method: str  # "none" | "convert" | "awq" | "gptq"
    bits: int
    group_size: int | None = None

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.name


FOUR_BIT_GROUP_SIZE = 64

SCHEMES = [
    QuantScheme("bf16", method="none", bits=16),
    QuantScheme("int8", method="convert", bits=8, group_size=64),
    QuantScheme("awq4", method="awq", bits=4, group_size=FOUR_BIT_GROUP_SIZE),
    QuantScheme("gptq4", method="gptq", bits=4, group_size=FOUR_BIT_GROUP_SIZE),
    QuantScheme("int4", method="convert", bits=4, group_size=FOUR_BIT_GROUP_SIZE),
]

BASELINE_SCHEME = "bf16"

TASKS = ("gsm8k", "triviaqa", "ifeval", "json")

PILOT_N = {task: 50 for task in TASKS}
FULL_N = {"gsm8k": 500, "triviaqa": 500, "ifeval": 500, "json": 500}

FEW_SHOT_K = 8  # GSM8K CoT few-shot examples

# Greedy decoding — sampling would add a second source of variance on top
# of quantization effects, which would need its own averaging to separate out.
TEMPERATURE = 0.0

MAX_TOKENS = {"gsm8k": 512, "triviaqa": 64, "ifeval": 512, "json": 256}

N_BOOTSTRAP = 10_000
ALPHA = 0.05


def _validate():
    four_bit_group_sizes = {s.group_size for s in SCHEMES if s.bits == 4}
    assert len(four_bit_group_sizes) == 1, (
        f"4-bit schemes must share group_size to isolate the quant-method "
        f"variable, got {four_bit_group_sizes}"
    )
    assert BASELINE_SCHEME in {s.name for s in SCHEMES}, (
        "BASELINE_SCHEME must be a defined scheme"
    )


_validate()


# ---------------------------------------------------------------------------
# Fine-tuning extension
# ---------------------------------------------------------------------------
# Every scheme is tuned with an identical budget (same rank, scale, steps,
# batch size, LR, data, and data order). That's the whole point: if the LoRA
# config varied by scheme we'd be comparing tuning budgets, not quantization
# schemes. _validate_finetune() below enforces the parts of that which are
# expressible as an assert.

ADAPTERS_DIR = PROJECT_ROOT / "finetune" / "adapters"
FT_DATA_DIR = PROJECT_ROOT / "finetune" / "data"
BUILDS_DIR = MODELS_DIR / "builds"  # transient fused/requantized models

FT_TASKS = ("gsm8k", "triviaqa")

# Arms (see README). B is eval-only; C1/C2 share one training run per
# (task, scheme, seed); D reuses the *bf16* adapter, so it needs no training
# of its own.
#   B  - untuned, 0-shot                      (control for the prompt change)
#   C1 - quantize -> tune, adapter unfused    (QLoRA, deployed as base+adapter)
#   C2 - quantize -> tune, fused + requantized (fine-tuning rounded to the grid)
#   D  - tune bf16 -> fuse -> quantize        (order of operations reversed)
ARMS = ("B", "C1", "C2", "D")

# C2 and D are only meaningful where quantization actually happens: fusing a
# bf16 adapter into a bf16 base is lossless, so C2/D at bf16 would just be C1
# again under another name (and cost a 39-minute GSM8K eval to prove it).
QUANTIZED_SCHEMES = tuple(s.name for s in SCHEMES if s.bits < 16)

FT_SEEDS = {"gsm8k": (0, 1, 2), "triviaqa": (0,)}
# Seeds 1 and 2 exist to size training noise against the between-scheme gaps.
# Running them for all five schemes would cost ~7 hrs of GSM8K eval for a
# variance estimate, so they're restricted to the two extremes of the range.
SEED_NOISE_SCHEMES = ("bf16", "int4")

LORA_FINE_TUNE_TYPE = "lora"
LORA_NUM_LAYERS = 16
LORA_RANK = 8
LORA_SCALE = 20.0   # mlx-lm's own default; note this is not HF's alpha/rank
LORA_DROPOUT = 0.0
LORA_ITERS = 600
LORA_BATCH_SIZE = 4
LORA_LR = 1e-5
LORA_MAX_SEQ_LENGTH = 1024
LORA_MASK_PROMPT = True  # loss on the completion only, not the question

FT_TRAIN_N = 2000
FT_VALID_N = 200

# Tuned models are evaluated 0-shot, matching how they were trained.
FT_FEW_SHOT = False


def _validate_finetune():
    assert set(FT_TASKS) <= set(TASKS), f"FT_TASKS must be a subset of TASKS, got {FT_TASKS}"
    assert set(FT_SEEDS) == set(FT_TASKS), "every FT task needs a seed tuple"
    assert set(SEED_NOISE_SCHEMES) <= {s.name for s in SCHEMES}, (
        "SEED_NOISE_SCHEMES must name defined schemes"
    )
    assert BASELINE_SCHEME not in QUANTIZED_SCHEMES, (
        "the baseline must be the unquantized scheme"
    )
    # Training data is drawn from each task's train split *after* the first
    # FEW_SHOT_K rows, which are already burned as prompt examples. Nothing
    # here can check the split itself, but a nonzero offset is the precondition.
    assert FEW_SHOT_K > 0, "FT data builder skips train[:FEW_SHOT_K] to avoid leakage"


_validate_finetune()
