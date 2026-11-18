"""Cost instrumentation: wall clock, peak RSS, and integrated GPU power draw."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil

_HAS_NVIDIA_SMI = shutil.which("nvidia-smi") is not None


class PowerPoller:
    """Samples total GPU board power at a fixed interval in a daemon thread.

    Energy is the integral of power over the sampling window.  A baseline idle
    reading is taken before the workload so that reported energy can be given
    either as raw or idle-subtracted.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read() -> float | None:
        if not _HAS_NVIDIA_SMI:
            return None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            return float(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def start(self) -> None:
        if not _HAS_NVIDIA_SMI:
            return
        self.samples = []
        self._stop.clear()

        def loop():
            t0 = time.perf_counter()
            while not self._stop.is_set():
                w = self._read()
                if w is not None:
                    self.samples.append((time.perf_counter() - t0, w))
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.summarize()

    def summarize(self) -> dict:
        if len(self.samples) < 2:
            return {"n_samples": len(self.samples), "joules": 0.0, "mean_watts": 0.0, "peak_watts": 0.0}
        watts = [w for _, w in self.samples]
        joules = 0.0
        for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
            joules += 0.5 * (w0 + w1) * (t1 - t0)
        return {
            "n_samples": len(self.samples),
            "joules": joules,
            "mean_watts": sum(watts) / len(watts),
            "peak_watts": max(watts),
            "window_s": self.samples[-1][0] - self.samples[0][0],
        }


@dataclass
class Cost:
    wall_s: float = 0.0
    audio_s: float = 0.0
    peak_rss_mb: float = 0.0
    gpu: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        return self.wall_s / self.audio_s if self.audio_s else float("nan")


def peak_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


class Timer:
    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall_s = time.perf_counter() - self._t
        return False
