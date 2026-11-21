# Omega 02: programs as data

**Prerequisites:** [page 00](./00-hello-tensor.md) (a `ProgramSpec` from TOML, run on `proximad` and in Rust) and [page 01](./01-config.md) (the runtime config and `[exec]` as `TensorExecutionConfig`).

**You will:** run a plan that takes caller-supplied data, read the two errors it gives when a tensor is missing, then take the file apart: the five node kinds, the map grammar that says which element feeds which, and how a bind name differs from an input's own name. You finish on the eleven production specs, which are addressed by position instead.

**New concepts (in order):** `[[bind]]` and the two unbound-input errors · `plan_parts` and `collect_outputs` · the node envelope: `id`, `dtype`, `name` · `ExtentSpec`: static and symbolic extents · `MapSpec`: projection and gather · the five node kinds, including `reduce` as a matmul · bind name versus input name · the shape-addressed specs and their tests.

## 1. A plan that takes data

Page 00's program had no inputs: `iota` and `constant` supplied everything. Real plans take tensors, and a plan declares the ones it expects in a `[[bind]]` entry. The repository's fixture is a matmul followed by silu and softmax — `y = (x @ w).silu().softmax(-1)` for `x: 2x4`, `w: 4x3` — and `omega/examples/run_plan.rs` runs any plan with synthesised inputs, so you do not need a data file:

```sh
cargo run -p proxima-omega --example run_plan -- \
  proxima-tensor/src/spec/fixtures/matmul_silu_softmax.toml --bind x=2x4 --bind w=4x3
```

```text
backend = cpu
softmax [2, 3] [0.3772, 0.3239, 0.2989, 0.3060, 0.3285, 0.3655]
```

The rows sum to one, which is what the last of its fourteen nodes computes: node `n13` divides `n11` by the row sum `n12` (`proxima-tensor/src/spec/fixtures/matmul_silu_softmax.toml:133-139`).

Forget a `--bind` and the plan names the input you forgot, by the caller's key:

```text
Error: no binding supplied for input `x`
```

Supply `x` only and it names the next one:

```text
Error: no binding supplied for input `w`
```

Both exit `1`. Those messages come from `TensorError::UnboundInputName` through `plan_parts` (`proxima-tensor/src/spec/run.rs:34`), which validates the plan before lowering anything (`run.rs:35-36`).

`plan_parts` is the one bridge from a spec to any backend, and it returns three things: the `Vec<Op>` the backends execute, the bound inputs paired with their internal names, and the output `NodeId`s in declaration order. `collect_outputs` (`run.rs:47`) reads each declared output back by name. When you only want the CPU answer, `evaluate_cpu` (`run.rs:22`) is both calls in one, and the CLI's CPU arm is that shortcut (`tools/proxima-cli/src/tensor_run.rs:67-70`) — which is why a plan and a `proximad` request produce the same numbers.

## 2. The node envelope

Every node carries an `id` and a `dtype`, and a node may only reference ids defined above it (`program_spec.rs:26-33`). The `id` is a string you choose; the lowering assigns each node its position as its `NodeId` (`run.rs:95-100`), so `n0`, `n1`, `n2` in page 00's file were positions, not a special numbering.

Shapes are written one of two ways (`ExtentSpec`, `program_spec.rs:87`): a bare integer for a static extent, or a string beginning `?` for a symbolic one, where `?0` and `?1` are the zeroth and first symbolic extents (`:92-96`). `[2, 4]` is a static 2x4; `["?0", 4]` is `seq x 4`, and is how a serving plan expresses a sequence length it learns at bind time.

Every node also accepts an optional `name`, which is identity rather than decoration: the backend binds data by name, so a name survives a partition pass that renumbers positions (`proxima-tensor/src/op.rs:195-201`).

## 3. The five node kinds

`NodeSpec` (`program_spec.rs:129`) is told apart by its `op` string.

- **`input`** (`:130`) declares a leaf the caller supplies: `dtype` and `shape`. Its `name` is the key the backend binds it under, and it is the only node kind that may be a bind target (`program_spec.rs:262`).
- **`elementwise`** (`:137`) computes an n-ary scalar map: `body` is a `ScalarOp` such as `add` or `multiply`, `inputs` lists the operands by id, and `maps` gives one addressing rule per operand.
- **`reduce`** (`:146`) is a matmul and every other contraction: multiply, then sum over an axis. `body` is the scalar fold (`add`, `maximum`), `init` is the fold's identity (`ReduceInit`, `op.rs:141`), and `input` is the operand. `in_map` and `out_map` relate the operand's axes to the output's; `keep` says whether the reduced axis survives as a size-1 axis (`Keep`, `op.rs:159`).
- **`iota`** (`:162`) is a leaf of `0, 1, 2, ...` up to `extent`; **`constant`** (`:171`) is a leaf of one repeated `value`, with `shape = []` meaning a rank-0 scalar that broadcasts into any consumer. These two are why a plan can run with no data at all, as page 00's did.

## 4. The map grammar

`MapSpec` (`program_spec.rs:111`) has two forms. The common one is a string, `Projection`: an arrow pair where the left side is the operand's axis terms and the right side is the output's axis letters. `si->si` is the identity over a two-axis tensor. `s,2*i->si` reads the operand at `s` along one axis and at twice the iteration index along the other — that is a strided read, and it is how a rotary position embedding (RoPE) picks even and odd elements without a gather.

The table form is `Gather { gather, index_map, map, dim }` (`program_spec.rs:113-125`): `gather` names a node supplying indices, `index_map` says how the iteration space addresses it, `map` addresses the operand's non-gathered axes, and `dim` names the operand axis the fetched index selects. An embedding lookup is this form.

## 5. Bind name versus input name

The two names are separate on purpose. `run.rs:64-77` finds the caller's tensor by `bind.name` and passes it to the backend under the input node's own `name`. An input node with no `name` cannot be bound at all: the lookup refuses it as `UnnamedInput` (`run.rs:87-93`). That split is what lets a serving caller use a short key like `x` while the graph keeps the name its weights were bound under.

## 6. The real specs are addressed by position

The eleven files in `proxima-tensor/specs/` are production shapes written as plans — a convolution, a rotary embedding, an attention block, a grouped-query block, an MoE block, two Mistral layers, a transformer block, and three probes beside them. None of them carries a `[[bind]]`. They are exercised positionally, the way the CPU evaluator's own tests bind `Op::Input`s in program order, so a test hands data in as an array of blocks and asks for named output nodes by `NodeId`.

The distinction matters in practice: a bound plan is what a caller runs (`proxima tensor run`, a serving request), and a shape-addressed spec is what a shape or a lowering is proved against. Both appear in the pages that follow, and each page says which it means.

The smallest real spec is `proxima-tensor/specs/rope.toml`: a pairwise rotation of `x: 2x4` by per-position `cos`/`sin` tables of shape `2x2`. Its interesting lines are the strided reads `s,2*i->si` and `s,2*i+1->si` (`rope.toml:42` and `:50`), the map form from section 4 doing real work with no gather node and no intermediate even/odd split.

Its test computes the expected pair norm straight off the raw buffer at indices `2*i` and `2*i+1`, so a map that read `i` instead of `2*i` would fail on a different pair:

```sh
cargo val -p proxima-tensor -E 'test(a_rope_pairwise_rotation_written_as_toml_preserves_pair_norm)'
```

```text
PASS proxima-tensor spec::tests::a_rope_pairwise_rotation_written_as_toml_preserves_pair_norm
SUMMARY  passed=2 failed=0 ignored=0 not_run=0  wall=47.638s discovered=1141 run=2 filtered=1139 binaries=8
```

## 7. What omega does not do yet

- **No plan imports another plan.** A `transformer_block` is one file, not `attention_block` plus `moe_block`; nothing in `ProgramSpec` composes sub-plans (`program_spec.rs:26-45` has no include field).
- **No dtype choice per output.** `dtype` is per node and the storage path is float32 today; page 00's gap list has the `numerics` half of that story.
- **No way to run a shape-addressed spec from the CLI.** `proxima tensor run` needs `[[bind]]` entries, because it must name the tensors it sends (`tools/proxima-cli/src/tensor.rs:41-45`); the eleven production specs are reachable from Rust tests, not from a shell command.
- **Symbolic extents stop at the Rust API.** `?0` lets a plan describe a sequence length, but nothing in the CLI lets a caller declare which length it is sending, so a symbolic plan is a Rust-side plan today.

Next: [03: training](./03-train.md) maps the PyTorch loop onto `proxima-autograd`, loads the optimizer from TOML, and runs four epochs on real MNIST with the loss curve and the test accuracy.
