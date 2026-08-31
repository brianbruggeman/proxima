"""Ports mnist.onnx's node graph (dumped with `onnx.load`, 14 nodes:
3x Conv+Relu, BatchNormalization, Flatten, Gemm+Relu, Gemm,
BatchNormalization, LogSoftmax -- see
proxima-onnx/tests/real_mnist_checkpoint.rs for the same graph parsed and
lowered through proxima-onnx) into an equivalent torch.nn.Module, loading
the model's 18 initializers verbatim as module weights.
"""

from __future__ import annotations

from pathlib import Path

import onnx
import onnx.numpy_helper
import torch
from torch import nn

MODEL_PATH = Path("/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx")

# matches the onnx graph's own BatchNormalization epsilon attribute exactly
BATCH_NORM_EPSILON = 9.999999747378752e-06


class MnistNet(nn.Module):
    """LeNet-style classifier matching mnist.onnx's node graph exactly:
    Conv(1,8,3)+Relu -> Conv(8,16,3)+Relu -> Conv(16,24,3)+Relu ->
    BatchNorm2d(24) -> Flatten -> Linear(11616,32)+Relu -> Linear(32,10) ->
    BatchNorm1d(10) -> LogSoftmax(dim=1).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3)
        self.norm1 = nn.BatchNorm2d(24, eps=BATCH_NORM_EPSILON)
        self.fc1 = nn.Linear(11616, 32)
        self.fc2 = nn.Linear(32, 10)
        self.norm2 = nn.BatchNorm1d(10, eps=BATCH_NORM_EPSILON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.norm1(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm2(x)
        return torch.log_softmax(x, dim=1)


# onnx initializer name -> MnistNet state_dict key (identical here; the
# mapping stays explicit so a future graph change fails loudly, not silently)
INITIALIZER_TO_STATE = {
    "conv1.weight": "conv1.weight",
    "conv1.bias": "conv1.bias",
    "conv2.weight": "conv2.weight",
    "conv2.bias": "conv2.bias",
    "conv3.weight": "conv3.weight",
    "conv3.bias": "conv3.bias",
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "norm1.running_mean": "norm1.running_mean",
    "norm1.running_var": "norm1.running_var",
    "fc1.weight": "fc1.weight",
    "fc1.bias": "fc1.bias",
    "fc2.weight": "fc2.weight",
    "fc2.bias": "fc2.bias",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    "norm2.running_mean": "norm2.running_mean",
    "norm2.running_var": "norm2.running_var",
}


def load_model(model_path: Path = MODEL_PATH) -> MnistNet:
    """Loads mnist.onnx's 18 initializers verbatim into MnistNet, eval mode."""
    onnx_model = onnx.load(str(model_path))
    weights = {init.name: torch.from_numpy(onnx.numpy_helper.to_array(init).copy()) for init in onnx_model.graph.initializer}

    model = MnistNet()
    state = model.state_dict()
    for initializer_name, state_name in INITIALIZER_TO_STATE.items():
        state[state_name].copy_(weights[initializer_name])
    model.load_state_dict(state)
    model.eval()
    return model
