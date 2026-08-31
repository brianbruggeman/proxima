"""Loads the MNIST t10k idx3/idx1 test split from
~/.cache/burn-dataset/mnist, mirroring
proxima-onnx/tests/real_mnist_accuracy.rs's own idx parsing and
normalization ((pixel/255 - 0.1307) / 0.3081 over the raw u8 pixel, never
predivided).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

DATASET_DIR = Path.home() / ".cache" / "burn-dataset" / "mnist"


def test_images_path() -> Path:
    return DATASET_DIR / "test" / "t10k-images-idx3-ubyte"


def test_labels_path() -> Path:
    return DATASET_DIR / "test" / "t10k-labels-idx1-ubyte"


def dataset_present() -> bool:
    return test_images_path().exists() and test_labels_path().exists()


def _idx_header(data: bytes) -> tuple[int, list[int]]:
    """Parses idx3/idx1's big-endian header: a magic number, an item count,
    then dimension_count - 1 per-axis extents -- Yann LeCun's idx format.
    """
    dimension_count = data[3]
    item_count = struct.unpack(">I", data[4:8])[0]
    extents = [struct.unpack(">I", data[4 + axis * 4 : 8 + axis * 4])[0] for axis in range(1, dimension_count)]
    return item_count, extents


def load_normalized_images(path: Path, limit: int) -> np.ndarray:
    """Every test image, normalized exactly as the reference model expects:
    (pixel/255 - 0.1307) / 0.3081, shape (take, rows, cols) float32.
    """
    data = path.read_bytes()
    item_count, extents = _idx_header(data)
    pixel_count = 1
    for extent in extents:
        pixel_count *= extent
    take = min(item_count, limit)
    header_length = 4 + len(extents) * 4 + 4
    raw = np.frombuffer(data, dtype=np.uint8, count=take * pixel_count, offset=header_length)
    raw = raw.reshape(take, *extents).astype(np.float32)
    return (raw / 255.0 - 0.1307) / 0.3081


def load_labels(path: Path, limit: int) -> np.ndarray:
    data = path.read_bytes()
    item_count, _extents = _idx_header(data)
    take = min(item_count, limit)
    return np.frombuffer(data, dtype=np.uint8, count=take, offset=8).copy()
