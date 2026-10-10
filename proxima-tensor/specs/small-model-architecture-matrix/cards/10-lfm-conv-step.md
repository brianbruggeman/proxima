# Card 10: LFM causal short-convolution token step

## Contract

Advance explicit convolution history by one gated `B*x` input. `state_in` and
`state_out` have `[channels, l_cache]` shape; `roll_indices` is the caller's
`(1..l_cache, 0)` permutation. The transition rolls the history left, clamps
the absolute cache position to `[0, l_cache - 1]`, overwrites that slot,
computes the weighted tap sum, and returns both output and state. Cache width
zero is rejected before graph mutation.

The pinned Transformers fixture at
`fixtures/upstream-source/f399fa2a111dac8c7fc07b2717abb10ee3e82468851321f45553ebb10fbd28b1.py.gz`
contains two implementations. `cuda_kernels_forward` calls
`causal_conv1d_update` with its default `cache_seqlens=None` (:455-463), whose
reference implementation appends new samples to the state before computing
the causal window ([causal-conv1d reference at `cd81f041`](https://github.com/Dao-AILab/causal-conv1d/blob/cd81f0413cad2fc1e6f17e785ac39f59aae690cd/causal_conv1d/causal_conv1d_interface.py#L1140-L1175)).
The same Transformers fixture selects `slow_forward` unless running the CUDA
fast path (:522-524); that fallback writes at the clamped absolute cache
position after rolling (:491-499). For a one-token prefill with history
`[0,0,g1]`, the next token at position 1 produces `[0,g2,0]`; this overwrites
the preceding sample. A causal FIFO update would produce `[0,g1,g2]` instead.
The graph test reproduces the pinned non-CUDA `slow_forward` formula; no
Transformers model, runtime, or device execution is included. The full-sequence
causal path remains a distinct reference.

## Worked vector

Use one channel, cache width 3, old history `[1, 2, 3]`, current gated input
`[4]`, weights `[1, 10, 100]`, and absolute cache position 2:

1. Roll: `[1, 2, 3] -> [2, 3, 1]`.
2. Clamp position 2 and overwrite that slot: `state_out = [2, 3, 4]`.
3. Dot the taps: `conv_out = 2*1 + 3*10 + 4*100 = 432`.

The named test checks final-tap and shorter absolute positions against the
pinned overwrite behavior.

## Design pressure

History remains an ordinary caller-owned tensor. The transition uses existing
`Input`, `Elementwise`, `Reduce`, and gather operations; it adds no stateful op
or hidden allocation. The absolute-position overwrite is required to match the
pinned non-CUDA path for short prefills.

## Validation

Run `taskpolicy -b nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-tensor --lib architecture_matrix_lfm_conv_step -- --nocapture`. The expected count is one named test, printing
`steps=2 reference_pairs=2 state_values=6`. Run `git diff --check` as part of
the card evidence.
