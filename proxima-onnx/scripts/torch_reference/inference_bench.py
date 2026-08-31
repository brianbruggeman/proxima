"""Single-image (batch=1) inference latency for mnist.onnx under torch:
50-image warmup, then K measured runs, reporting p50/p95/p99/mean/CoV.
See proxima-tensor/docs/discipline.md row 157 for the recorded reference
numbers this re-proves (measured date/host/loadout noted there; a fresh run
under a loaded host will disagree with those numbers on mean, not on the
qualitative single-thread-vs-eight-thread result).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import dataset_present, load_normalized_images, test_images_path  # noqa: E402
from model import load_model  # noqa: E402

WARMUP_IMAGES = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1, help="torch.set_num_threads value")
    parser.add_argument("--runs", type=int, default=200, help="measured inference count after warmup")
    return parser.parse_args()


def percentile(sorted_samples: list[float], fraction: float) -> float:
    index = min(len(sorted_samples) - 1, int(fraction * len(sorted_samples)))
    return sorted_samples[index]


def load_batches(total: int) -> torch.Tensor:
    if dataset_present():
        images = load_normalized_images(test_images_path(), total)
        if len(images) < total:
            pad = np.zeros((total - len(images), 28, 28), dtype=np.float32)
            images = np.concatenate([images, pad], axis=0)
    else:
        images = np.zeros((total, 28, 28), dtype=np.float32)
    return torch.from_numpy(images).unsqueeze(1)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)

    model = load_model()
    batches = load_batches(WARMUP_IMAGES + args.runs)

    with torch.no_grad():
        for index in range(WARMUP_IMAGES):
            model(batches[index : index + 1])

        samples_ms: list[float] = []
        for index in range(WARMUP_IMAGES, WARMUP_IMAGES + args.runs):
            start = time.perf_counter()
            model(batches[index : index + 1])
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    samples_ms.sort()
    mean = statistics.mean(samples_ms)
    stdev = statistics.pstdev(samples_ms)
    coefficient_of_variation = stdev / mean if mean else 0.0

    print(f"threads={args.threads} runs={args.runs}")
    print(
        f"p50={percentile(samples_ms, 0.50):.4f}ms "
        f"p95={percentile(samples_ms, 0.95):.4f}ms "
        f"p99={percentile(samples_ms, 0.99):.4f}ms "
        f"mean={mean:.4f}ms "
        f"CoV={coefficient_of_variation:.4f}"
    )


if __name__ == "__main__":
    main()
