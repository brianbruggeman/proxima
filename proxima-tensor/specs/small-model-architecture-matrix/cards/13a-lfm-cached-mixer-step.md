# Card 13a: lower one cached LFM short-convolution mixer step

## Contract

Add a `proxima-tensor` graph helper for one LFM decode token. It accepts the
token's normalized residual input, the three short-convolution input
projections, convolution weights, output projection, caller-owned `[channels,
l_cache]` history, absolute `cache_position`, and the history roll permutation.
It returns the mixer residual output and the updated history as separate graph
roots. The input sequence has exactly one token; prefill continues to use the
existing full-sequence mixer. Invoke the cached transition only after prefill
has initialized history; the pinned upstream selects full-sequence convolution
when `cache_position` is zero.

## Pinned transition

The upstream implementation is the retained LFM source at
`fixtures/upstream-source/f399fa2a111dac8c7fc07b2717abb10ee3e82468851321f45553ebb10fbd28b1.py.gz`:

- `:441-475` computes `Bx = B * x`, uses the cached update when
  `cache_position[0] > 0`, then gates convolution output by `C` and applies
  `out_proj`.
- `:487-501` rolls history left, clamps absolute `cache_position` to
  `[0, L_cache - 1]`, writes `Bx` at that slot, and computes the weighted tap
  sum.
- `:502-508` retains the existing full-sequence convolution path for prefill.

The Proxima decode graph must preserve that ordering. It must not share an SSM
state contract or substitute attention KV state for convolution history.

## Worked vector

Use one channel, `L_cache=3`, residual input `x=2`, norm weight `1`, epsilon
`1e-6`, projection weights `B=2`, `C=3`, `X=4`, output projection `2`, prior
history `[1,2,3]`, roll indices `[1,2,0]`, and absolute cache position `3`.
With `n = 2 / sqrt(4 + 1e-6)`, the gated input is `8n²`; the shifted and
updated history is `[2,3,8n²]`; convolution is `2 + 30 + 800n²`; the residual
output is `2 + 6n(2 + 30 + 800n²)`. The test asserts those three history
values and the output within `1e-3`.

## Acceptance

- Run `cargo test -p proxima-tensor --lib architecture_matrix_lfm_cached_mixer_step -- --nocapture`.
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
