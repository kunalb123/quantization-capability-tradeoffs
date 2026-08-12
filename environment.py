import platform
import subprocess
import sys
from datetime import datetime, timezone


def capture_environment() -> dict:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()

    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "chip": chip,
        # skip editable/local installs ("-e ..." or "@ file://") — no "==" to split on
        "packages": dict(line.split("==", 1) for line in freeze if "==" in line),
    }
