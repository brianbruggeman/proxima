# Card 02 — per-tensor scale and reconstruction comparison

## Objective

Compare native Proxima reconstruction of the pinned Granite 4 Micro BF16
checkpoint under two independent encoding conditions: unscaled scalar E5M2,
E3M2, and E2M2; and the same formats preceded by one persisted maximum
magnitude scale per tensor.

The mode is an explicit experiment axis. It does not claim equivalence with
groupwise MLX quantization or establish model output quality.

## Source

- Repository: `ibm-granite/granite-4.0-micro-base`
- Revision: `98e1d641451818f6c8e6b17c1c359b5b4ac495d0`
- Source payloads: 322 BF16 tensors, 3,402,836,480 values.
- Conversion outputs and comparison reports: `evidence/`.

## Commands

Tensor-maximum conversion ran serially with Cargo single-job and niced:

```sh
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e5m2-tensor-scaled e5m2-tensor-scaled 32768
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e3m2-tensor-scaled e3m2-tensor-scaled 32768
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e2m2-tensor-scaled e2m2-tensor-scaled 32768
```

The three logs named `evidence/granite4-*-tensor-scaled-conversion.log`
record 322 tensors each, 6,805,672,960 source bytes scanned and processed,
and output bytes for each selected format.

The separate serial direct-binary timing run is retained in
`evidence/conversion-wall-times-2026-10-10.log`. For this pinned Granite 4
Micro checkpoint on this Mac, unscaled E5M2/E3M2/E2M2 took 57.84/128.21/107.79
seconds; tensor-maximum E5M2/E3M2/E2M2 took 63.44/199.91/202.02 seconds.
These measurements exclude Cargo compilation and include conversion artifact
writes. They do not measure Proxima model loading, inference, or prefill.

The six compare commands then ran serially. Each consumed the source and its
corresponding converted artifact:

```sh
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- compare /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e5m2-release
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- compare /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e3m2-release
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- compare /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e2m2-release
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- compare /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e5m2-tensor-scaled
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- compare /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e3m2-tensor-scaled
RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- compare /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e2m2-tensor-scaled
```

Each run retained a JSON report and stdout log named
`evidence/{e5m2,e3m2,e2m2}-{scalar,tensor-scaled}-comparison.{json,log}`.
The report rows include tensor name, source values, changed count, max error
with source/reconstructed pair, squared error, max magnitudes, zeroed and
saturated counts, and up to eight raw pairs.

## Evidence ledger

All rows below use the same pinned BF16 source revision and all 3,402,836,480
values. These are tensor-reconstruction measurements, not logits or inference
measurements. Exact aggregate records are in each JSON file's root object;
per-tensor records, including the worst absolute-error tensor, follow in that
same file.

| Format and mode | changed values | max absolute error | squared error sum | saturated | nonzero values reconstructed as zero | source record |
|---|---:|---:|---:|---:|---:|---|
| E5M2 scalar | 3,296,186,812 | 4.0 | 14,981.656496071202 | 0 | 1,618,641 | `evidence/e5m2-scalar-comparison.json:6-11` |
| E3M2 scalar | 3,402,209,120 | 32.0 | 587,889.0411454184 | 22 | 3,221,113,464 | `evidence/e3m2-scalar-comparison.json:6-11` |
| E2M2 scalar | 3,402,830,219 | 53.0 | 1,323,703.7614737302 | 5,275 | 3,402,101,121 | `evidence/e2m2-scalar-comparison.json:6-11` |
| E5M2 tensor maximum | 3,357,205,621 | 4.214287 | 10,063.580800308859 | 29 | 6 | `evidence/e5m2-tensor-scaled-comparison.json:6-11` |
| E3M2 tensor maximum | 3,366,248,725 | 4.214287 | 10,122.630547819701 | 29 | 94,267,922 | `evidence/e3m2-tensor-scaled-comparison.json:6-11` |
| E2M2 tensor maximum | 3,398,775,706 | 4.214287 | 75,346.46239732625 | 29 | 1,384,722,601 | `evidence/e2m2-tensor-scaled-comparison.json:6-11` |

## Trace notes

- Scalar E3M2 and E2M2 show many source values reconstructed as zero. Their
  smallest positive magnitudes are 0.0625 and 0.25 from the subnormal decode
  rules ([E3M2 source](../../../../proxima-gguf/src/quant/fp6_e3m2.rs:85),
  [E2M2 source](../../../../proxima-gguf/src/quant/fp5_e2m2.rs:84)); scalar mode
  uses scale 1.0, while scaled conversion measures and applies a tensor scale
  before packing ([converter](../../../../proxima-model-interop/src/cold_artifact.rs:533)).
- Scalar E3M2/E2M2 saturation includes large values in `model.norm.weight`.
  That tensor's recorded worst source value is 60 at index 1028; E3M2
  reconstructs 28 and E2M2 reconstructs 7. The format maxima are below that
  source magnitude.
- Tensor-maximum mode rescales each tensor before encoding and reverses that
  scale on reconstruction; the scale is stored in each tensor manifest row
  ([manifest write](../../../../proxima-model-interop/src/cold_artifact.rs:628)).
  For these reports it reduces the measured worst absolute error for
  these reports, while the scales also produce 29 saturated values in each
  format and do not eliminate zeroing for E3M2/E2M2. The persisted scale is
  determined by each tensor maximum; the record does not identify whether a
  different scaling granularity would improve model behavior.
- The comparison records do not contain per-output logits, token IDs, or
  prefill timings. Tensor error deltas alone do not establish a Granite model
  quality or runtime conclusion.

## Integrity

`evidence/converted-artifact-integrity-audit.log` records all 322 tensor file
hashes and lengths matching manifests for the six format/mode artifacts, and
all 10 pinned source artifact hashes matching the source inventory.
`evidence/cold-artifact-scaled-comparison-tests-v2.log` records 4 passing
focused interop tests, including malformed paths, output-in-source rejection,
and conversion of actual safetensors wire bytes.

## Result status

Measured tensor reconstruction evidence is retained for all six arms. The
mechanism traced here is scalar format range plus per-tensor maximum scaling.
Whether any arm improves model outputs or runtime remains unmeasured by this
card.

