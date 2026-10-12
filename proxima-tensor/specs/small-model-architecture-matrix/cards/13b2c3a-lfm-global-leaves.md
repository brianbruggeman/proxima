# Card 13b2c3a: bind LFM global leaves through the generic profile

## Contract

Bind the scheduled graph's `token_embd.weight`, `output_norm.weight`, and
`output.weight` leaves from a safetensors manifest using the existing
`BindingProfile` rename data and `bind_safetensors_program_leaves`. Keep the
shared embedding/output tensor as one source with two program names. Preserve
the old GGUF `token_embd_norm.weight` spelling as a fallback alias. Do not add
an LFM binder or a tensor operation.

The pinned LFM2.5 header names the two source tensors
`model.embed_tokens.weight` (`[65536,2048]`) and
`model.embedding_norm.weight` (`[2048]`). The small executable fixture uses
the same source names with four embedding rows, allowing exact binding and
logit checks without downloading or claiming execution of the 2.34 GB model.

## Design

Use the existing alias precedence and dense decode path. `output.weight` is a
rank-two matmul input, so the existing role analysis keeps its BF16 source bytes
packed and row-major. `token_embd.weight` and `output_norm.weight` are native
program leaves, so the existing decoder materializes their BF16 values for the
CPU evaluator. The graph is composed from the existing embedding lookup,
RMSNorm, elementwise multiply, and reduction operations.

This extends profile data and the existing binder. It abandons a family branch
inside the binder, a separate embedding-weight container, and a new gather or
projection operation.

## Acceptance

Run the focused test and require exactly one pass and this marker:

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std --lib architecture_matrix_lfm_global_leaves -- --nocapture
```

```text
pinned_sources=2 bound_leaves=3 tied_source_uses=2 evaluated_logits=4 missing_source_rejected=1
```

The test checks pinned source names, BF16 dtypes, and axes; binds a compact
BF16 manifest through the generic profile resolver; proves the embedding and
head reference the same source bytes; checks all four hand-calculated logits;
and rejects a missing norm source with `UnknownTensor`. Then rerun
`architecture_matrix_lfm_layer_bind` and require one pass with its existing
marker, and run `git diff --check` with no diagnostics. Record compiler, host,
full outputs, and exit statuses in
`evidence/card-13b2c3a-lfm-global-leaves.txt`.

The fixture validates profile binding and graph evaluation; it does not claim
an invocation of the pinned full checkpoint.
