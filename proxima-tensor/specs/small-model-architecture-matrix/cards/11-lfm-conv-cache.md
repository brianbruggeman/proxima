# Card 11: carry LFM convolution state from prefill into decode

## Worked example (paper, before implementation)

The pinned LFM2 source has two cache paths. For prefill it computes the causal convolution and copies `pad(Bx, (L_cache - Bx.shape[-1], 0))` into the cache (`fixtures/upstream-source/f399fa2a111dac8c7fc07b2717abb10ee3e82468851321f45553ebb10fbd28b1.py.gz:465-469`). For cached decode it rolls that state left, clamps the absolute cache position, writes the new `Bx`, and computes the dot product (`:491-499`). Card 10 implements that decode transition; this card checks the state handoff between the two paths.

Pinned batch-one, one-channel example, `L_cache=3`, no bias, taps `[1,10,100]`. Proxima's local prefill input convention is `[sequence, channel]`; this card fixes batch size at one, matching the source's `[batch, channel, sequence]` with its batch axis removed:

1. Prefill receives `Bx=[1,2]`. Its causal outputs are `[100,210]`; the cache is left-padded to width three: `[0,1,2]`.
2. Decode at absolute position 2 with `Bx=3`: roll `[0,1,2]` to `[1,2,0]`, write slot 2, get state `[1,2,3]` and output `1+20+300=321`.
3. Decode at absolute position 3 with `Bx=4`: roll `[1,2,3]` to `[2,3,1]`, clamp slot to 2, write, get state `[2,3,4]` and output `2+30+400=432`.

The state builder also gets independent tail-window controls: a width-3 prefill `[1,2,3]` yields `[1,2,3]`, and width-4 `[1,2,3,4]` yields `[2,3,4]`. These distinguish exact-width pass-through and left cropping from the width-2 left-padding case.

For the long-prefill control, decoding `Bx=5` at position 4 from `[2,3,4]` yields `[3,4,5]` and `3+40+500=543`.

The incorrect handoff control feeds the prefill state `[0,1,2]` into decode 2 instead of decode 1's returned `[1,2,3]`; after writing 4 it produces `[1,2,4]` and `421`, not `[2,3,4]` and `432`. Each state buffer remains caller-owned and unchanged by evaluation.

## Algorithm

```text
prefill_state(Bx[s,d], prompt_len, L):
  state[d,l] = Bx[prompt_len - L + l,d] when prompt_len - L + l >= 0
               0 otherwise

decode(state[d,l], Bx[d], absolute_position):
  state = step(state, Bx, absolute_position)  # Card 10 transition
  return conv_output, state
```

`prompt_len` is static for this graph and greater than zero; the caller creates a graph for the request's prefill width. `L` is checkpoint-derived. This covers both zero-left-padding when `prompt_len < L` and selecting the trailing window when `prompt_len >= L`.

## Design pressure

The prefill cache is the last `L` values of the gated `B*x` stream, not prefill convolution outputs and not raw hidden states. A one-row fixture alone would not test truncation, so the cache-state builder accepts any static prefill width and uses the final `L` rows. No persistent mutation or cache wrapper is added: each graph returns an ordinary state tensor that the caller passes to the next invocation. A separate stateful Op and a test that reuses the original cache for both decode calls are ruled out because either would hide the ownership transition this card exists to verify.

## Implementation mapping

The prefill builder gathers positions `prompt_len-L+l` along the sequence axis, clamps only the gather index, and selects zero for negative positions. The computed-gather indices must be materialized by `Reduce`, matching the existing convolution builder; a plain `Elementwise` index can remain fusion-held and fail evaluation. The test binds symbol 0 to the static prefill width because the causal-convolution sequence `Iota` uses that symbol. It appends the builder and `causal_conv1d` to one graph, verifies prefill outputs and state, then evaluates Card 10's step graph twice while feeding the exact previous output bytes forward. It also verifies width-3 and width-4 tail-window controls, decodes once from the width-4 state, checks each state snapshot, and rejects the wrong-state control.
