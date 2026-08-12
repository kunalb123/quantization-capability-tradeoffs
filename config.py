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
