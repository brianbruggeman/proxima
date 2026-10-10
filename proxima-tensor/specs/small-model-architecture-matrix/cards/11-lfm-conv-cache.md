# Card 11: carry LFM convolution history from prefill into decode

## Contract

Build the final `l_cache` gated `B*x` samples after prefill, left-padding with
zeros for a short prefix and cropping from the left for a long prefix. Each
decode step consumes the prior history, rolls it left, clamps the absolute
cache position, overwrites that position with the new gated sample, and
returns the next history explicitly. Once the position clamps to the final
tap, this agrees with FIFO append; a shorter prefix can overwrite a previous
sample, as Card 13b1 spells out.

The pinned LFM source stores the prefill `B*x` tail in its cache
(`fixtures/upstream-source/f399fa2a111dac8c7fc07b2717abb10ee3e82468851321f45553ebb10fbd28b1.py.gz:465-469`).
Its CUDA path delegates decode updates to the external `causal_conv1d_update`
(:455-463). The non-CUDA fallback's position-indexed update is the transition
implemented here and exercised by a synthetic operator-graph test on native
arm64 CPU; that test does not invoke the Transformers model runtime. The
transition differs from causal FIFO for a one-token prefix, as Card 13b1
verifies.

## Worked vector

Use one channel, `l_cache=3`, no bias, and taps `[1,10,100]`:

1. Prefill `Bx=[1,2]` produces causal outputs `[100,210]` and history
   `[0,1,2]`.
2. Decode `Bx=3` at absolute position 2: roll to `[1,2,0]`, write slot 2 to get `[1,2,3]`, and compute
   `1+20+300=321`.
3. Decode `Bx=4` at absolute position 3: roll to `[2,3,1]`, write slot 2 to get `[2,3,4]`, and compute
   `2+30+400=432`.

The history builder also checks width-3 `[1,2,3] -> [1,2,3]` and width-4
`[1,2,3,4] -> [2,3,4]`. Decoding once from the width-4 tail with `Bx=5`
returns `[3,4,5]` and `543`. Starting from the stale prefill history instead
of the latest decode history produces a different output.

## Ownership and validation

Each graph returns an ordinary state tensor for the caller to pass into the
next invocation; evaluation does not mutate the input buffer. Run
`taskpolicy -b nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-tensor --lib architecture_matrix_lfm_conv_cache -- --nocapture`; retain the emitted prefill/decode counts and state controls.
