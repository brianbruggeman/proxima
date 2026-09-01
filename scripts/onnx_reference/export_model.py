"""Export BAAI/bge-small-en-v1.5 to ONNX from the cached HF safetensors checkout.

The HF cache snapshot dir (models--BAAI--bge-small-en-v1.5/snapshots/<sha>)
holds only model.safetensors + config/tokenizer files -- no model.onnx is
present there (verified: `find ... -iname '*.onnx'` under that tree is
empty). This script produces one, offline, from the already-cached weights,
so the onnxruntime arm has a model.onnx to point BGE_MODEL_PATH at without
touching any other repo's checkout or the user's HF cache directory itself.

Output goes under this directory's model/ subfolder (gitignored), never
into the HF cache and never into a tracked file.
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModel

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model.onnx")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.eval()

    # same [CLS] sentence A token ids bge_eval.rs::sentences() hardcodes,
    # used only as a tracing input shape for the ONNX export.
    input_ids = torch.tensor([[101, 1996, 4937, 2938, 2006, 1996, 13523, 102]], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)

    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        OUTPUT_PATH,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    size_bytes = os.path.getsize(OUTPUT_PATH)
    print(f"exported model.onnx to {OUTPUT_PATH} ({size_bytes} bytes)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
