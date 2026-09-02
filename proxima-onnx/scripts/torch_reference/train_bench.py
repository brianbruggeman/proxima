"""Per-step training latency for a 784-128-10 MLP under torch (Adam,
CrossEntropyLoss, batch 32): 20 warmup steps, then K measured steps,
reporting p50/p95/mean/CoV. Not mnist.onnx -- that checkpoint carries no
training graph -- a standalone MLP sized to the same input/output shape, to
measure torch's own optimizer-step cost. See
proxima-tensor/docs/discipline.md row 157 for the recorded reference number
this re-proves.

Trains on real MNIST train-split batches by default (matching
proxima-autograd/benches/train_step_lane.rs), falling back to synthetic
torch.randn/randint noise only when the fixture under
~/.cache/burn-dataset/mnist/train is absent.
"""

from __future__ import annotations

import argparse
import statistics
import time

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_labels, load_normalized_images, train_dataset_present, train_images_path, train_labels_path  # noqa: E402
from diagnostics import report_and_verify_threads, report_load_average  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1, help="torch.set_num_threads value")
    parser.add_argument("--warmup", type=int, default=20, help="unmeasured steps before timing starts")
    parser.add_argument("--steps", type=int, default=200, help="measured training steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--real-data",
        type=str,
        choices=["auto", "on", "off"],
        default="auto",
        help="use real MNIST train-split batches (auto: on if the fixture is present, else synthetic)",
    )
    return parser.parse_args()


def build_model() -> nn.Module:
    return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def percentile(sorted_samples: list[float], fraction: float) -> float:
    index = min(len(sorted_samples) - 1, int(fraction * len(sorted_samples)))
    return sorted_samples[index]


def load_real_batches(total_steps: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    sample_count = total_steps * batch_size
    images = load_normalized_images(train_images_path(), sample_count)
    labels = load_labels(train_labels_path(), sample_count)
    flat_images = images.reshape(images.shape[0], -1)
    inputs = torch.from_numpy(flat_images).reshape(total_steps, batch_size, 784)
    targets = torch.from_numpy(labels.astype("int64")).reshape(total_steps, batch_size)
    return inputs, targets


def load_synthetic_batches(total_steps: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randn(total_steps, batch_size, 784)
    targets = torch.randint(0, 10, (total_steps, batch_size))
    return inputs, targets


def resolve_data_source(requested: str) -> str:
    if requested == "on":
        if not train_dataset_present():
            raise SystemExit(f"--real-data=on but no MNIST train fixture under {train_images_path().parent}")
        return "real-mnist"
    if requested == "off":
        return "synthetic"
    return "real-mnist" if train_dataset_present() else "synthetic"


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    report_load_average("start")

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    total_steps = args.warmup + args.steps
    data_source = resolve_data_source(args.real_data)
    if data_source == "real-mnist":
        print(f"data source: real-mnist ({train_images_path()})")
        inputs, targets = load_real_batches(total_steps, args.batch_size)
    else:
        reason = "--real-data=off" if args.real_data == "off" else f"no MNIST train fixture under {train_images_path().parent}"
        print(f"data source: synthetic (torch.randn/randint, {reason})")
        inputs, targets = load_synthetic_batches(total_steps, args.batch_size)

    def train_step(index: int) -> None:
        optimizer.zero_grad()
        logits = model(inputs[index])
        loss = loss_fn(logits, targets[index])
        loss.backward()
        optimizer.step()

    for index in range(args.warmup):
        train_step(index)

    report_and_verify_threads(args.threads)

    samples_ms: list[float] = []
    for index in range(args.warmup, total_steps):
        start = time.perf_counter()
        train_step(index)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    report_load_average("end")

    samples_ms.sort()
    mean = statistics.mean(samples_ms)
    stdev = statistics.pstdev(samples_ms)
    coefficient_of_variation = stdev / mean if mean else 0.0

    print(f"threads={args.threads} steps={args.steps} batch_size={args.batch_size} data={data_source}")
    print(
        f"p50={percentile(samples_ms, 0.50):.4f}ms "
        f"p95={percentile(samples_ms, 0.95):.4f}ms "
        f"mean={mean:.4f}ms "
        f"CoV={coefficient_of_variation:.4f}"
    )


if __name__ == "__main__":
    main()
