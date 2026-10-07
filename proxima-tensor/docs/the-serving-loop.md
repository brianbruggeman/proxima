# The serving loop

How a loaded model turns a prompt into tokens, and where a caller can watch or stop it. Every
`file:line` was read at commit `365fa7f6` plus this slice. The two examples are runnable:

```
cargo run -p proxima-model-interop --example serving_state_walkthrough --features std
cargo run -p proxima-model-interop --example model_config_load         --features std,metal
```

The first reads no checkpoint. The second is the granite run from
`a-model-is-a-configuration.md`; its `on_token` and statistics output is quoted below.

## One state machine, and what it does not do

`ServingState<Entry, Cache>` is an enum in `proxima-core/src/serving_state.rs:21`, re-exported by
`proxima-model-interop` (`lib.rs`, `pub use proxima_core::ServingState`). It is sans-IO: its
module doc (`serving_state.rs:1`) says every transition performs exactly one program
evaluation, or placement copies only, and the per-layer cache lives inside the variant a
transition returns. The machine evaluates nothing itself. It states which evaluation shape is
legal next, and the caller supplies the result of the evaluation.

Six variants, each carrying only what is legal in that state:

| state | holds | legal next |
|---|---|---|
| `Prefill` | the prompt ids, the cache | `advance_prefill` to `Decode`; `finish` |
| `Decode` | the last accepted token, the cache | `advance_decode` to `Decode`; `enter_verify` to `Verify`; `finish` |
| `Verify` | the draft, a snapshot of the cache, the cache | `accept` or `accept_rows` to `Accept` or `Rollback`; `finish` |
| `Accept` | how many drafts matched, the next token, the cache | `resume` to `Decode` |
| `Rollback` | the snapshot to restore, the token to resume from | `rollback` to `Decode` |
| `Done` | the cache | nothing |

A transition called from the wrong variant returns `ServingFsmError::IllegalTransition` naming
the transition (`serving_state.rs:10-13`) instead of doing anything. In the decode loop `Entry`
is `u32` (a token id) and `Cache` is `usize`, the number of positions the KV cache holds; a
`Verify` snapshot is therefore a `usize` copy, and rolling back is truncating the attention
caches to that cursor.

The walkthrough drives every legal transition with a stand-in target function (the next id is
the current id plus one) and asserts each state:

```rust
    let state: State = ServingState::start(prompt, 0);
    let state = state.advance_prefill(target(12), prompt_rows).expect("prefill settles into decode");
    assert_eq!(state, ServingState::Decode { last: 13, cache: 3 });

    let verifying = state.enter_verify(vec![15, 16]).expect("a decode state may draft");
    let accepted = verifying.accept_rows(2, &[15, 16], vec![5, 6]).expect("every drafted row matched");
    assert_eq!(accepted, ServingState::Accept { n: 2, next: 16, cache: 6 });

    let verifying = state.enter_verify(vec![17, 99]).expect("a second draft");
    let rejected = verifying.accept_rows(1, &[17, target(17)], vec![7, 8]).expect("row 1 differs from the draft");
    assert_eq!(rejected, ServingState::Rollback { snapshot: 8, to: 18 });
```

```
every legal transition walked; after Done: serving fsm: advance_decode is not legal from the current state
```

On acceptance the cache carried forward is the one for the last accepted draft row
(`settle`, `serving_state.rs`: `placement_index`), which is why the loop never replays a verified
prefix.

## How `LoadedModel` drives it

`drive_serving_loop` (`proxima-model-interop/src/generate/decode.rs`, the function after
`run_decode_loop_from_ids`) holds one `Option<ServingState<u32, usize>>` that it takes and puts
back every step, starting at `Prefill` (`decode.rs:3464-3465`). The loop is
`'decode: while step < max_tokens` (`:3674`); inside it one `'evaluate` block (`:3676`) does the
evaluation for the state it took:

1. Read the state. Only `Prefill` and `Decode` may evaluate; anything else is
   `IllegalTransition { attempted: "evaluate_from_settled_state" }` (`:3683-3689`).
2. When the drafter proposes tokens on a `Decode` step, `enter_verify(draft)` (`:3778`). A
   speculative step evaluates the last token plus the drafts in one program run and reads one
   logits row per input row.
3. After the verify evaluation: `accept_rows` (`:5897`), truncate the attention caches to the
   snapshot on `Rollback` (`:5910-5925`), then `resume` and `advance_decode` of the bonus token
   on `Accept` or `rollback` on `Rollback` (`:5928-5931`). The tokens that settled go into
   `settled_tokens`.
4. Otherwise evaluate the single step and settle it: `advance_prefill` from `Prefill`,
   `advance_decode` from `Decode` (`:6356-6359`). A model with no KV layers re-enters `Prefill`
   over the whole sequence each step instead (`:6343-6349`).
5. Leave `'evaluate`; a `for` over `settled_tokens` hands each token to `deliver_token`
   (`:6365-6388`), which is where the stop conditions live. `Break` leaves `'decode`.

The loop ends by turning whatever state is left into `Done` with `finish()` (`:6394`).
`run_decode_loop_from_ids` is a wrapper over it that releases the monolithic expert source on
every exit, the error exits included (`decode.rs`, doc on `run_decode_loop_from_ids`).

There is one loop and one machine. A speculative step is a `Decode` that entered `Verify`; a
plain step is a `Decode` that did not. Whether a model drafts is a `ServingConfig` field
(`speculative`), not a different loop.

## Where hooks sit

A hook is a place a caller sees or changes what the machine does without editing it. In the order
a token meets them:

**Sampling.** After the evaluation, `select_decoded_token` (`decode.rs:1319`) picks the token:
the forced value for that step when a `token_override` is present, otherwise
`proxima_tokenizer::sample_next_token` over the logits row, the recent-token window and the
sampling fields of `ServingConfig` (temperature, top-k, top-p, min-p, the repeat, frequency and
presence penalties, the seed). A speculative batch calls the same function at the same inputs, so
it consumes the random stream in the same order a plain decode would (its doc,
`decode.rs:1309-1322`). `token_override` is a crate-internal parameter of the decode functions;
the public hook for the choice is the `ServingConfig` fields.

**Observation of the machine.** `SpeculativeDecodeStats::record_state`
(`generate/residency_caches.rs:3768-3776`) counts the states the loop passes through: `Prefill`
and `Decode` when about to evaluate, `Accept` and `Rollback` when settled. It is called at
`decode.rs:3781` and `:5903`. A caller passes it to
`generate_from_ids_with_speculative_stats` (or `generate_streaming_with_speculative_stats`).

**Observation of the output, and stopping.** `on_token` is the one callback in every
`generate_*` entry point: `&mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<(), ()>`. `deliver_token`
(`residency_caches.rs:4003`) calls it for each settled token: first, at step 0, with
`Phase::Prefill { prompt_tokens }`, then with `Phase::Token` (`:4037`, `:4054`; `Phase` is
`:3879`). An event carries the token id, its visible text piece, the step, and the cumulative
elapsed milliseconds. `ControlFlow::Break(())` stops generation (the returned flag reports it as
not the model's own stop); the end-of-sequence id stops it too, and is not recorded
(`deliver_token`, `:4048-4050`). Control tokens never stop generation and never print.

**Inside the evaluation.** `LogitsSink::observe` (`decode.rs:6157`) records the logits row at
`last_position` (`LogitsSink::Collect` gathers one `Vec<f32>` per step) and
`NodeValuesSink::observe` (`decode.rs:5331`) records the values of the nodes it was given
(`NodeValuesSink::Collect`, `residency_caches.rs:30-36`). `LogitsSink` is `pub(crate)` (`pregather.rs:2896`); the public route to node values
is `forward_node_values` (`decode.rs:7049`).

The example exercises the output hooks and the machine statistics on the real granite run:

```rust
    let (generated, _text, _stopped) = model
        .generate_from_ids_with_speculative_stats(
            &prompt_ids,
            llama_ids.len(),
            &serving_config,
            &mut |event| {
                phases.push(event.phase);
                ControlFlow::Continue(())
            },
            &mut stats,
            None,
        )
        .expect("decodes");
    assert_eq!(stats.prefill_steps, 1, "one prefill evaluation");
    assert_eq!(stats.decode_steps, 31, "every later token is one single-row decode step");
```

```
on_token: 1 prefill event, 32 token events
serving machine: prefill 1 decode 31 accept 0 rollback 0
on_token Break at the third event: 2 ids kept after 3 events
```

Read the numbers: 32 requested tokens are one `Prefill` evaluation (which also produces token 0)
and 31 `Decode` evaluations; speculation is off, so no `Verify`, `Accept` or `Rollback` occurs.
The step-0 `Prefill` event precedes that step's `Token` event, so 33 events arrive for 32
tokens. A `Break` on the third event stops with two ids kept: the id is appended before its
`Token` event fires (`deliver_token`, `generated_ids.push` before the second `on_token`).

## What a hook cannot be

A hook does not change which state is legal. An `on_token` that wants to inject tokens, or a
sampler that wants another evaluation shape, is a change to the machine (a new variant with its
transitions and its test in `proxima-core`) and then to the loop; that is the same rule as for a
new model primitive in `what-configuration-cannot-do.md`.
