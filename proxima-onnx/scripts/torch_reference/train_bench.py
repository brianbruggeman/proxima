"""Per-step training latency for a 784-128-10 MLP under torch (Adam,
CrossEntropyLoss, batch 32): 20 warmup steps, then K measured steps,
reporting p50/p95/mean/CoV. Not mnist.onnx -- that checkpoint carries no
training graph -- a standalone MLP sized to the same input/output shape, to
measure torch's own optimizer-step cost. See
proxima-tensor/docs/discipline.md row 157 for the recorded reference number
this re-proves.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1, help="torch.set_num_threads value")
    parser.add_argument("--warmup", type=int, default=20, help="unmeasured steps before timing starts")
    parser.add_argument("--steps", type=int, default=200, help="measured training steps")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def build_model() -> nn.Module:
    return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def percentile(sorted_samples: list[float], fraction: float) -> float:
    index = min(len(sorted_samples) - 1, int(fraction * len(sorted_samples)))
    return sorted_samples[index]


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    total_steps = args.warmup + args.steps
    inputs = torch.randn(total_steps, args.batch_size, 784)
    targets = torch.randint(0, 10, (total_steps, args.batch_size))

    def train_step(index: int) -> None:
        optimizer.zero_grad()
        logits = model(inputs[index])
        loss = loss_fn(logits, targets[index])
        loss.backward()
        optimizer.step()

    for index in range(args.warmup):
        train_step(index)

    samples_ms: list[float] = []
    for index in range(args.warmup, total_steps):
        start = time.perf_counter()
        train_step(index)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    samples_ms.sort()
    mean = statistics.mean(samples_ms)
    stdev = statistics.pstdev(samples_ms)
    coefficient_of_variation = stdev / mean if mean else 0.0

    print(f"threads={args.threads} steps={args.steps} batch_size={args.batch_size}")
    print(
        f"p50={percentile(samples_ms, 0.50):.4f}ms "
        f"p95={percentile(samples_ms, 0.95):.4f}ms "
        f"mean={mean:.4f}ms "
        f"CoV={coefficient_of_variation:.4f}"
    )


if __name__ == "__main__":
    main()
