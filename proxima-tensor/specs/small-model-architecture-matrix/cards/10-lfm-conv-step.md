# Card 10: LFM short-convolution token step

## Worked example (paper, before implementation)

Pinned inputs: `fixtures/semantic-pair-configs/lfm_text.json` declares `conv_L_cache = 3` and `conv_bias = false`. The configuration declares `transformers_version = 4.57.2`. The retained upstream source fixture at `fixtures/upstream-source/f399fa2a111dac8c7fc07b2717abb10ee3e82468851321f45553ebb10fbd28b1.py.gz:491-499` rolls the cached state left, clamps `cache_position` to `[0, L-1]`, writes the current `Bx = B * x` at that slot, and sums the updated taps against `conv.weight[:, 0, :]`. The write is not always at the newest slot.

Concrete one-channel decode case:

- old state, oldest to newest: `[1, 2, 3]`
- current gated input `Bx`: `[4]`
- cache width: `3`
- depthwise weights, oldest to newest: `[1, 10, 100]`
- convolution bias: absent, matching the pinned config
- decode cache position cases: `1` and `3`; clamped write slots are `1` and `2`

At cache position `3`:

1. Roll left: `[1, 2, 3] -> [2, 3, 1]`.
2. Write current `Bx=4` at slot `2`: `state_out = [2, 3, 4]`.
3. Dot updated state with taps: `conv_out = 2*1 + 3*10 + 4*100 = 432`.

At cache position `1`, the same roll produces `[2, 3, 1]`, then the write produces `state_out = [2, 4, 1]` and `conv_out = 2*1 + 4*10 + 1*100 = 142`. This case falsifies an unconditional newest-slot append: that incorrect update would return `[2, 3, 4]` and `432`. The position-3 case also agrees with the final position of the existing batch causal-convolution example (`proxima-tensor/src/spec/tests.rs::causal_conv1d_matches_a_hand_computed_causal_window`).

## Algorithm (pseudocode)

```text
step(state[d, l], gated_input[d], weight[d, l], cache_position):
  write_slot = clamp(cache_position, 0, L - 1)
  rolled[d, l] = state[d, (l + 1) mod L]
  next_state[d, l] = gated_input[d] if l == write_slot else rolled[d, l]
  conv_out[d] = sum_l(next_state[d, l] * weight[d, l])
  return conv_out[d], next_state[d, l]
```

The step precondition is `cache_position > 0`, matching the pinned cached decode branch; initial/prefill cache construction is outside this card.

The state stores prior `B*x` values, not raw hidden states or convolution outputs. The returned `conv_out` is pre-`C` gating and pre-`out_proj`, matching the local batch `causal_conv1d` boundary. The subsequent mixer stages remain the caller's responsibility.

## Design pressure

Use the existing Proxima `Input`, `Elementwise`, `Reduce`, and `Output` graph contract to express caller-owned state-in/state-out. Do not add a stateful `Op` variant: the LFM state is an ordinary fixed-width tensor, and the public `build_forward` graph already represents cache values as input/output leaves. A ring buffer and a persistent hidden allocation are ruled out; this reference step always exposes the exact oldest-to-newest state and writes no state implicitly.

## Implementation mapping

For the pinned `L=3` graph, caller-owned `state_in [D,3]`, `gated_input [D]`, and `weight [D,3]` are inputs. A fixed roll-index leaf `[1,2,0]` feeds the existing computed gather along the tap axis. `Iota` plus equality against the clamped write slot selects current input at one tap and rolled state at the others. Elementwise multiply and tap-axis reduce produce `conv_out`; optional bias is added only when configured. The returned state is an explicit graph root, so the caller owns cache persistence. The named test runs positions 1 and 3, checks both state/output pairs, and asserts the unconditional-append control differs at position 1.
