# Card 13a: lower one cached LFM short-convolution mixer step

## Contract

The graph helper accepts one LFM decode token, its three short-convolution
projections, convolution weights, output projection, caller-owned
`[channels, l_cache]` history, absolute cache position, and the history roll
permutation. It returns the mixer residual and updated history as separate
graph roots. The transition follows the pinned non-CUDA LFM path: roll, clamp
the position, overwrite that slot, then convolve. The caller uses this path
after prefill initialized history; `l_cache=0` is rejected before graph
mutation.

Call the cached transition only for `cache_position > 0`; the pinned model
selects its full-sequence prefill branch at position zero.

## Pinned transition

The upstream implementation is the retained LFM source at
`fixtures/upstream-source/f399fa2a111dac8c7fc07b2717abb10ee3e82468851321f45553ebb10fbd28b1.py.gz`:

- `:441-469` computes `Bx = B * x`, calls `causal_conv1d_update` for cached
  decode, or initializes the left-padded prefill cache and runs full-sequence
  causal convolution.
- `:476-507` is the non-CUDA slow path. Its cached branch writes at the clamped
  absolute position after rolling. The one-token-prefix overwrite is tested
  in Card 13b1.
- Both branches gate convolution output by `C` and apply `out_proj`.

The Proxima decode graph must preserve that ordering. It must not share an SSM
state contract or substitute attention KV state for convolution history.

## Worked vector

Use one channel, `L_cache=3`, residual input `x=2`, norm weight `1`, epsilon
`1e-6`, projection weights `B=2`, `C=3`, `X=4`, output projection `2`, prior
history `[1,2,3]`, position 3, and roll indices `[1,2,0]`.
With `n = 2 / sqrt(4 + 1e-6)`, the gated input is `8n²`; the shifted and
position-updated history is `[2,3,8n²]`; convolution is `2 + 30 + 800n²`; the
residual output is `2 + 6n(2 + 30 + 800n²)`. The test asserts those three
history values and the output within `1e-3`.

## Acceptance

- Run `taskpolicy -b nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-tensor --lib architecture_matrix_lfm_cached_mixer_step -- --nocapture`.
- The named test prints `decode_tokens=1 reference_pairs=1 state_values=3` and
  asserts the real graph roots against the worked vector.
- The graph uses the existing `Input`, `Elementwise`, `Reduce`, and gather
  operations; no new `Op` variant or mutable hidden state is introduced.
- `git diff --check` reports no whitespace errors.

## Files

- `proxima-tensor/src/spec/short_conv_delta_net.rs`
- `proxima-tensor/src/spec/tests.rs`
- `proxima-tensor/specs/small-model-architecture-matrix/TASKS.md`
- `proxima-tensor/specs/small-model-architecture-matrix/SPEC.md`
