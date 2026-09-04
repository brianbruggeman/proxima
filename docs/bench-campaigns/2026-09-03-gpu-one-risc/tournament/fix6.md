# Round-4 judge residuals (apply after fix5.md, same file, tag `[round-4 judge <id>]`)

## J1 — card 9.1
expect: the REJECT half must DRIVE the autograd side, not only the checker: add a case where a
placed `Reduce` with a `Computed` (data-dependent) out_map is differentiated and the adjoint's
existing guard fires (`proxima-autograd/src/adjoint.rs:806` `is_data_dependent()` →
`ScatterOutputUnsupported`, `proxima-autograd/src/error.rs:73-88`) exactly as it does today — proving
the affine extension changed nothing on the data-dependent path; and a case where two producers
overlap and bind's interval check rejects BEFORE autograd sees the graph.

## J2 — card 6.5
The `UNIFORM_BUFFERS` bound is a leak repair riding inside a performance card, so it gets its own
N, test and rollback line within 6.5: it is commit 1 of 6.5 (the bound + `[spans]
uniform_cache_entries` + a test that inserts `capacity + 1` distinct uniform blobs and asserts
`UNIFORM_CACHE_LEN == capacity` and the evicted entry's buffer is released), commit 2 is the arena
and plan-owned uniforms; rollback reverts commit 2 only — commit 1 is kept regardless (§15). On main
the map has exactly two operations (`get` at `omega/src/metal.rs:2074`, `insert` at `:2092`) and no
recency field, so the bound is a data-structure change on the upload path and N7 asserts
`upload_uniforms` ns/call unchanged within the 0.5 band for `op_setup_ms`.

## J3 — card 11.1
Add a docs gate: `grep -c '[^/]bind\.rs:' docs/bench-campaigns/2026-09-03-gpu-one-risc/plan.md`
must be 0 (G0: every `bind.rs` cite carries its crate path), run in the reprove.

## J4 — every card that runs the MILLI cell (0.2, 1.3, 3.4, 4.2, 6.3, 7.2, 10.1, 10.2)
Label the MILLI rows as **5-token cells**: `profiles_one_real_decode_step_by_per_op_gpu_time`
hardcodes `max_tokens = 5` (`proxima-model-interop/src/bind.rs:3103`) and sets
`PROXIMA_METAL_OP_PROFILE_STEP=3` itself (`:3122`, removed at `:3131`), so `PROXIMA_MAX_TOKENS=8`
and the env profile step are inert on that rung; the row states "milli budget 5, bench budget 8"
so the two rungs are never read as the same cell.

## J5 — card 5.2
opens/commands: the slice the arena hands to `named_blocks` as `QuantizedBlock::Float32` needs an
accessor yielding `&[f32]` from `AlignedBuffer` — the card cites `proxima-tensor/src/align.rs` for
page size only; add `proxima-tensor/src/align.rs (locate: grep -n 'fn as_slice\|fn as_ref\|impl.*Deref'
proxima-tensor/src/align.rs)` to opens and an expect line: "the accessor exists on main or the
card adds it as a `&[f32]` view with a test; its name is recorded on the row" [round-4 judge J5].
