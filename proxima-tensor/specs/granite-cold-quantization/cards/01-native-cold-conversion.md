# Card 01 — native Proxima cold conversion

## Objective

Convert the pinned Granite 4 Micro BF16 safetensors shards into Proxima-owned
packed E5M2, E3M2, or E2M2 tensor payloads without Python, full-shard copies,
or source mutation.

## Required evidence

- source repository and revision pinned in the manifest
- SHA256 for config, tokenizer files, index, each source shard, each source
  tensor, and each output tensor
- original tensor name, dtype, and shape for every payload
- counts from bytes consumed and bytes successfully written
- serial conversion command and exact tensor count
- golden codec and malformed-input tests with selected counts

## Current implementation

`proxima-gguf/src/quant/cold_float.rs` provides caller-buffer conversion from
BF16 bytes. `proxima-model-interop/src/cold_artifact.rs` streams a safetensors
shard in tensor-offset order and records the converted payload hashes.
`proxima-model-interop/src/bin/granite-cold-quantize.rs` binds the command to
the Granite 4 Micro repository revision.

## Validation

```sh
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 cargo test -p proxima-gguf --lib quant::fp -- --nocapture
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 cargo test -p proxima-gguf --lib quant::cold_float -- --nocapture
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 cargo check -p proxima-model-interop --features std --bin granite-cold-quantize
```

The full release conversion commands ran serially against the pinned checkpoint:

```sh
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e5m2-release e5m2 32768
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e3m2-release e3m2 32768
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e2m2-release e2m2 32768
bash proxima-tensor/specs/granite-cold-quantization/evidence/verify-converted-output-hashes.sh
```

The three conversion logs record 322 tensors and 6,805,672,960 source payload
bytes for the same pinned revision. They record 3,402,836,480 E5M2 output
bytes, 2,552,127,360 E3M2 output bytes, and 2,126,772,800 E2M2 output bytes.
`converted-artifact-integrity-audit.log` records all 322 output file hashes
and lengths matching each manifest, plus all 10 source artifact hashes
matching the pinned checkpoint files.

This card covers native Proxima conversion and file integrity. The tensor
reconstruction comparisons and separate per-tensor scaling experiment are
recorded in [Card 02](02-tensor-scale-comparison.md). These artifacts compare
tensor values; they do not establish model loading, inference output quality,
or prefill performance.
