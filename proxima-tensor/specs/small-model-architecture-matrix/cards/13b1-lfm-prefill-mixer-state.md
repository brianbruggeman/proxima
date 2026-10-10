# Card 13b1: return LFM prefill mixer history and use the pinned cache update

## Contract

Return `(residual, history)` from the full-sequence LFM short-convolution
mixer. History comes from the same normalized `B*x` projections as the causal
convolution output, has shape `[channels, l_cache]`, and is left-padded or
cropped by `causal_conv1d_prefill_state`. The cached mixer consumes this
history and applies the pinned non-CUDA update: roll left, clamp the absolute
cache position, overwrite that slot with the new gated input, then convolve
the updated state. Every state remains an explicit graph input/output.

## Pinned source behavior

The retained Transformers fixture
`fixtures/upstream-source/f399fa2a111dac8c7fc07b2717abb10ee3e82468851321f45553ebb10fbd28b1.py.gz`
selects `slow_forward` outside the CUDA fast path (:522-524). Its cached branch
rolls state, clamps `cache_position`, and writes at that absolute slot (:491-499).
The CUDA branch instead calls `causal_conv1d_update(..., None)` (:455-463).
This card models the non-CUDA branch used on the Apple host; it does not assert
that this slot-write behavior is equivalent to the CUDA kernel or full causal
sequence convolution.

## Worked vector

Use one channel, cache width 3, epsilon `1e-6`, norm weight 1, projections
`B=2`, `C=3`, `X=4`, convolution weights `[1,10,100]`, and output projection
2. For input token `i`, let `n_i = i / sqrt(i² + 1e-6)` and `g_i = 8*n_i²`.

Prefill `[1]` returns residual `1 + 6*n_1*100*g_1` and state
`[0,0,g_1]`. Decode `[2]` at absolute cache position 1 rolls that state to
`[0,g_1,0]`, overwrites slot 1, and returns `[0,g_2,0]`. Its residual is
`2 + 6*n_2*10*g_2 = 481.99985` for the chosen float32 operations. A causal FIFO
update instead returns state `[0,g_1,g_2]` and residual `5281.9976`; that
control differs from the pinned non-CUDA path. The separately evaluated
full-sequence causal mixer returns `[4800.993,5281.9976]`.

## Acceptance

Run the named synthetic operator-graph test on native arm64 CPU with one
background-priority Cargo job; this does not invoke the Transformers model
runtime:

```sh
taskpolicy -b nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-tensor --lib architecture_matrix_lfm_prefill_mixer_state -- --nocapture
```

It reports one passing named test and prints the one-token and two-token
prefill/decode vectors, carried state, a zero-history control, and the
full-sequence vector. Its counters include
`prefill_tokens=1 prefill_pairs=1 decode_pairs=1 state_values=3 zero_history_control=1 fifo_divergence=1 invalid_cache_rejected=1 partial_graph_rejected=1`
and
`two_token_prefill=2 carried_pairs=1 state_values=3 zero_history_rejected=1`.
The two-token case must distinguish decode from its zero-history control. The
test asserts the pinned position-indexed output and that zero-width cache
errors occur before graph mutation. Record the full
command, host and compiler versions, output, exit status, and `git diff
--check` in `evidence/card-13b1-lfm-prefill-mixer-state.txt`.
