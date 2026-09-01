"""Per-op attribution of torch's mnist inference latency via torch.profiler.
CPU activities only, with_stack off, single-thread, >=200 measured images.
See proxima-tensor/docs/discipline.md ROW 189's own Phase A table for the
recorded reference numbers this re-proves (measured date/host/loadout noted
there; a fresh run under a different host disagrees on the absolute
per-image number, not on the qualitative conv/fc/other split).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import dataset_present, load_normalized_images, test_images_path  # noqa: E402
from model import load_model  # noqa: E402

WARMUP_IMAGES = 50
MEASURED_IMAGES = 200


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
    torch.set_num_threads(1)
    model = load_model()
    batches = load_batches(WARMUP_IMAGES + MEASURED_IMAGES)

    with torch.no_grad():
        for index in range(WARMUP_IMAGES):
            model(batches[index : index + 1])

        with profile(activities=[ProfilerActivity.CPU], with_stack=False, record_shapes=True) as prof:
            for index in range(WARMUP_IMAGES, WARMUP_IMAGES + MEASURED_IMAGES):
                model(batches[index : index + 1])

    print(f"threads=1 measured_images={MEASURED_IMAGES}")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=40))

    total_self_cpu_us = sum(event.self_cpu_time_total for event in prof.key_averages())
    print(f"\ntotal_self_cpu_us={total_self_cpu_us:.1f} per_image_us={total_self_cpu_us / MEASURED_IMAGES:.4f}")

    conv_us = sum(
        event.self_cpu_time_total
        for event in prof.key_averages()
        if "conv" in event.key.lower() or "convolution" in event.key.lower()
    )
    fc_us = sum(
        event.self_cpu_time_total
        for event in prof.key_averages()
        if "addmm" in event.key.lower() or "linear" in event.key.lower() or event.key.lower() == "aten::mm"
    )
    print(f"conv_self_cpu_us={conv_us:.1f} ({100 * conv_us / total_self_cpu_us:.2f}%)")
    print(f"fc_self_cpu_us={fc_us:.1f} ({100 * fc_us / total_self_cpu_us:.2f}%)")
    print(f"other_self_cpu_us={total_self_cpu_us - conv_us - fc_us:.1f} ({100 * (total_self_cpu_us - conv_us - fc_us) / total_self_cpu_us:.2f}%)")


if __name__ == "__main__":
    main()
