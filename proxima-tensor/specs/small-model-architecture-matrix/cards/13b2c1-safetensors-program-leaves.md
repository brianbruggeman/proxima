# Card 13b2c1: bind safetensors leaves from a lowered program

## Contract

Add one reusable safetensors binder that accepts a parsed `Manifest`, the
whole file byte slice, data-start offset, a lowered `Op` program, the existing
`BindingProfile`, and the runtime-input names for this invocation. It walks
named `Op::Input` leaves, reuses the program consumer-role analysis already
used by `bind_program_leaves`, and resolves direct or `TensorAlias::Rename`
sources from the manifest. It returns existing `BoundWeights` storage via the
current safetensors decode and transpose helpers.

The binder must validate exact axes, dtype byte width, offset range, and file
bounds before binding. For a `Native` role, manifest axes must equal the
program's static axes. For a supported rank-2 `InOut` matmul, manifest axes
`[out, in]` must equal the reverse of the program axes `[in, out]`; equal
element counts with different axes are rejected. Symbolic weight extents are
rejected with a typed error in this slice. An aligned native F32 leaf borrows
its input range; a misaligned F32 leaf is decoded bytewise into owned F32
values. F16/BF16 native leaves use the existing decode path. F16/BF16 matmul
leaves remain packed borrowed blocks; F32 matmul leaves use the existing transpose
helper. `BindingProfile::decode_f32` forces owned F32 decoding for matching
leaves, and `BindingProfile::extra` binds named extra tensors using the same
decode path. An input explicitly listed as runtime state is skipped. An
unresolved required leaf fails with its program name. `Gathered` roles and
requested `Part`/`Join` aliases return typed unsupported errors in this
slice; GGUF gathered/Part/Join behavior stays unchanged.

No tensor `Op`, kernel, model-specific weight type, or family branch is added.

## Worked fixture

Build three two-input/three-output matmul cases from F32, F16, and BF16
safetensors bytes. For each dtype, add a native two-element norm and a renamed
two-element norm alias sharing the same file tensor. Add three named token
inputs to the program's runtime-input set. Evaluate all three matmuls against
the fixture's exact expected vectors. The fixture therefore binds nine weight
leaves, skips three runtime inputs, uses two packed low-precision matmuls and
one transposed F32 matmul, decodes four low-precision native norm/alias leaves,
and resolves three aliases.

## Acceptance

Run:

```sh
env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std --test safetensors_program_leaves architecture_matrix_safetensors_program_leaves -- --nocapture
```

Require one passing named test and the exact marker:

```text
dtypes=3 bound_leaves=9 runtime_inputs_skipped=3 packed_matmuls=2 native_decodes=4 f32_transposes=1 alias_matches=3 evaluated_matmuls=3 decode_rules=1 extra_weights=1 overlapping_ranges_rejected=1 native_f32_borrows=1 misaligned_f32_decodes=1 missing_weights_rejected=1 wrong_axes_rejected=1 dtype_byte_length_rejected=1 short_payload_rejected=1 offset_overflow_rejected=1 part_alias_rejected=1 join_alias_rejected=1 gathered_role_rejected=1 symbolic_weight_rejected=1
```

The test must assert bytes/pointers for packed F16/BF16 matrix blocks, exact
decoded native values, exact F32 transpose values, and all three evaluated
matmul outputs. It must assert aligned F32 borrowing and misaligned F32 owned
decoding with exact values. It must assert typed failures for missing weights,
mismatched axes, dtype byte-width mismatch, truncated payload bytes, overflowing
offsets, Part aliases, Join aliases, gathered roles, and symbolic weight
extents. The forced-decode control must produce an owned F32 block, and the extra tensor must be present
under its configured name. Record the command, host,
compiler, full output, exit status, and `git diff --check` in
`evidence/card-13b2c1-safetensors-program-leaves.txt`.
