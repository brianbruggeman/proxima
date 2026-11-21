# Omega 02: programs as data

**Prerequisites:** [page 00](./00-hello-tensor.md) (a `ProgramSpec` from TOML, run on `proximad` and in Rust) and [page 01](./01-config.md) (the runtime config and `[exec]` as `TensorExecutionConfig`).

**You will:** read a plan's five node kinds and its map grammar, see the difference between a bound (named) plan and a shape-addressed spec, run a bound plan through omega from the shell including its two failure modes, and run one of the eleven real specs in `proxima-tensor/specs/`.

**New concepts (in order):** the node envelope: `id`, `dtype`, `name` · `ExtentSpec`: static and symbolic extents · `MapSpec`: projection and gather · `elementwise` · `reduce` with `init`, `in_map`, `out_map` and `keep` · `constant` and `iota` · `[[bind]]` versus an input node's `name` · `[[output]]` · `plan_parts` and `collect_outputs` · the shape-addressed specs and their tests · what omega does not do yet.

## 1. The node envelope

Every node carries an `id` and a `dtype`, and a node may only reference ids defined above it (`program_spec.rs:26-33`). The `id` is a string the plan author chooses; the lowering assigns each node its position as its `NodeId` (`run.rs:95-100`), so `n0`, `n1`, `n2` are positions in this file order, not some other numbering.

Shapes are written one of two ways (`ExtentSpec`, `program_spec.rs:87`): a bare integer for a static extent, or a string beginning `?` for a symbolic one, where `?0` and `?1` are the zeroth and first symbolic extents (`program_spec.rs:92-96`). `[2, 4]` is a static 2x4; `["?0", 4]` is `seq x 4` and is how a serving plan expresses a sequence length it learns at bind time.

Every node also accepts an optional `name`, which is identity rather than decoration: the backend binds data by name, so a name survives a partition pass that renumbers positions (`proxima-tensor/src/op.rs:195-201`).

## 2. The five node kinds

`NodeSpec` (`program_spec.rs:129`) is told apart by its `op` string.

- **`input`** (`:130`) declares a leaf the caller supplies: `dtype` and `shape`. Its `name` is the key the backend binds it under, and it is the only node kind that may be a bind target (`program_spec.rs:262`).
- **`elementwise`** (`:137`) computes an n-ary scalar map: `body` is a `ScalarOp` such as `add` or `multiply`, `inputs` lists the operands by id, and `maps` gives one addressing rule per operand.
- **`reduce`** (`:146`) contracts or reduces one operand: `body` is the scalar fold (`add`, `maximum`), `init` is the fold's identity (`zero`, `negative_infinity`; `ReduceInit`, `op.rs:141`), `input` is the operand id, `in_map`/`out_map` relate it to the output's axes, and `keep` says whether the reduced axis stays as a size-1 axis (`Keep`, `op.rs:159`). A matmul is this node: a multiply whose result is reduced.
- **`iota`** (`:162`) is a leaf of `0, 1, 2, ...` up to `extent`; `constant` (`:171`) is a leaf of one repeated `value`, with `shape = []` meaning a rank-0 scalar that broadcasts into any consumer. These two are why a plan can run with no data at all, as page 00's did.

## 3. The map grammar

`MapSpec` (`program_spec.rs:111`) has two forms. The common one is a string, `Projection`: an arrow pair where the left side is the operand's axis terms and the right side is the output's axis letters. `si->si` is the identity over a two-axis tensor. `s,2*i->si` reads the operand at `s` along one axis and at twice the iteration index along the other — that is a strided read, and it is how RoPE picks even and odd elements without a gather.

The table form is `Gather { gather, index_map, map, dim }` (`program_spec.rs:115-126`): `gather` names a node supplying indices, `index_map` says how the iteration space addresses it, `map` addresses the operand's non-gathered axes, and `dim` names the operand axis the fetched index selects. An embedding lookup is this form.

## 4. A bound plan, and its two failure modes

A plan that takes data declares one `[[bind]]` per input (`BindSpec`, `program_spec.rs:76`): `name` is the caller's key and `node` is the `input` node's id. The two names are separate on purpose — `run.rs:64-77` finds the caller's tensor by `bind.name` and passes it to the backend under the input node's own `name`. An input node with no `name` cannot be bound at all (`run.rs:87-93`).

The repository's bound fixture is `proxima-tensor/src/spec/fixtures/matmul_silu_softmax.toml`: `y = (x @ w).silu().softmax(-1)` for `x: 2x4` and `w: 4x3`. `omega/examples/run_plan.rs` runs any plan with synthesised inputs, which is how a plan gets exercised without a data file:

```sh
cargo run -p proxima-omega --example run_plan -- \
  proxima-tensor/src/spec/fixtures/matmul_silu_softmax.toml --bind x=2x4 --bind w=4x3
```

```text
backend = cpu
softmax [2, 3] [0.3772, 0.3239, 0.2989, 0.3060, 0.3285, 0.3655]
```

The rows sum to one, which is what the last of its thirteen nodes asserts: node `n13` divides `n11` by the row sum `n12` (`fixtures/matmul_silu_softmax.toml:133-139`).

Forget a `--bind` and the plan says which input it is missing, by the caller's key:

```text
Error: no binding supplied for input `x`
```

Supply `x` only and it names the next one:

```text
Error: no binding supplied for input `w`
```

Both exit `1`. Those messages come from `TensorError::UnboundInputName` through `plan_parts` (`run.rs:34`), which is also where validation runs before any lowering (`run.rs:35-36`).

## 5. Plan to ops to a backend

`plan_parts` (`run.rs:34`) is the one bridge from a spec to any backend, and it returns three things: the `Vec<Op>` the backends already execute, the bound inputs paired with their internal names, and the output `NodeId`s in declaration order. `omega/examples/run_plan.rs` then calls `plan_named_with_config` and `execute_plan_named` exactly as page 00 did, and `collect_outputs` (`run.rs:47`) reads each declared output back by name.

`evaluate_cpu` (`run.rs:22`) is the shortcut when you only want the CPU answer: it calls `plan_parts` and the CPU evaluator, and returns `(name, shape, values)` per output. The CLI's CPU arm is that call (`tools/proxima-cli/src/tensor_run.rs:67-70`), which is why a plan and a `proximad` request produce the same numbers.

## 6. The real specs are shape-addressed, not bound

The eleven files in `proxima-tensor/specs/` are production shapes written as plans: `conv2d`, `rope`, `attention_block`, `causal_attention`, `gqa_attention`, `gqa_attention_group3`, `moe_block`, `moe_topk2_probe`, `mistral_layer`, `mistral_layer_small` and `transformer_block`. None of them carries a `[[bind]]`. They are exercised positionally, the way the CPU evaluator's own tests bind `Op::Input`s in program order, so a test hands data in as an array of blocks and asks for named output nodes by `NodeId`.

That is a real distinction, not a detail: a bound plan is what a caller runs (`proxima tensor run`, a serving request); a shape-addressed spec is what a backend shape or a lowering is proved against. Page 02 uses both, and later pages say which one they mean.

The smallest real spec is `proxima-tensor/specs/rope.toml`: a pairwise rotation of `x: 2x4` by per-position `cos`/`sin` tables of shape `2x2`. Its two interesting lines are the strided reads `s,2*i->si` and `s,2*i+1->si` (`rope.toml:42` and `:50`), which is the map form from section 3 doing real work: no gather node, no intermediate even/odd split.

Its test computes the expected pair norm straight off the raw buffer at indices `2*i` and `2*i+1`, so a map that read `i` instead of `2*i` would fail on a different pair:

```sh
cargo val -p proxima-tensor -E 'test(a_rope_pairwise_rotation_written_as_toml_preserves_pair_norm)'
```

```text
PASS proxima-tensor spec::tests::a_rope_pairwise_rotation_written_as_toml_preserves_pair_norm
SUMMARY  passed=2 failed=0 ignored=0 not_run=0  wall=47.638s discovered=1141 run=2 filtered=1139 binaries=8
```

The second pass in that summary is a doctest, not the rotation; the spec test itself is the one named on the first line.

## 7. What omega does not do yet

- **No plan imports another plan.** A `transformer_block` is one file, not `attention_block` plus `moe_block`; composition of specs is a copy, and nothing in `ProgramSpec` composes sub-plans (`program_spec.rs:26-45` has no include field).
- **No dtype choice per output.** `dtype` is per node and the storage path is float32 today; page 00's gap list has the `numerics` half of that story.
- **No way to run a shape-addressed spec from the CLI.** `proxima tensor run` needs `[[bind]]` entries, because it must name the tensors it sends (`tools/proxima-cli/src/tensor.rs:41-45`); the eleven production specs are reachable from Rust tests, not from a shell command.
- **Symbolic extents are a shape-inference feature, not a data feature.** `?0` lets a plan describe a sequence length, but nothing in the CLI lets a caller declare which length it is sending, so a symbolic plan is a Rust-side plan today.

Next: [03: training](./03-train.md) maps the PyTorch loop onto `proxima-autograd`, loads the optimizer from TOML, and runs four epochs on real MNIST with the loss curve and the test accuracy.
