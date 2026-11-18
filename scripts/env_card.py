"""Write ``results/environment.json``: the exact compute stack behind every number.

A reader cannot reproduce an edge-inference measurement without the driver version
and the GPU part number, and a manuscript that names them by hand will eventually
name them wrongly. This reads them from the running interpreter instead.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "environment.json"


def _ver(mod: str, attr: str = "__version__") -> str:
    try:
        return str(getattr(__import__(mod), attr))
    except Exception as exc:  # noqa: BLE001 - an absent optional dep is a fact
        return f"absent ({type(exc).__name__})"


def nvidia_smi(query: str) -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False)
    except OSError:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def main() -> int:
    import torch

    gpus = []
    for name, mem, drv in zip(
            nvidia_smi("name"), nvidia_smi("memory.total"), nvidia_smi("driver_version")):
        gpus.append({"name": name, "memory_total": mem, "driver_version": drv})

    d = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
        "gpu": gpus,
        "gpu_name": gpus[0]["name"] if gpus else None,
        "packages": {
            "transformers": _ver("transformers"),
            "peft": _ver("peft"),
            "ctranslate2": _ver("ctranslate2"),
            "faster_whisper": _ver("faster_whisper"),
            "soundfile": _ver("soundfile"),
            "numpy": _ver("numpy"),
            "psutil": _ver("psutil"),
            "accelerate": _ver("accelerate"),
        },
    }
    try:
        import ctranslate2 as ct2
        d["ct2_cuda_device_count"] = ct2.get_cuda_device_count()
        # Recorded because it decides the quantization axis: a compute type this build
        # cannot execute on the available device has a footprint but no latency, so
        # exporting it would produce a half-measurement.
        d["ct2_supported_compute_types"] = {"cpu": sorted(
            ct2.get_supported_compute_types("cpu"))}
        if d["ct2_cuda_device_count"]:
            d["ct2_supported_compute_types"]["cuda"] = sorted(
                ct2.get_supported_compute_types("cuda"))
    except Exception as exc:  # noqa: BLE001
        d["ct2_cuda_device_count"] = None
        d["ct2_error"] = f"{type(exc).__name__}: {exc}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(d, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
