# sketch 3: qwen3.6 MoE with expert offload, as configuration (H4-H6 routed FFN, H8 placement)

status: paper test. Read at main 4b4be6cf (`git show main:<path>`). Nothing was built or run; every
"hand-derived" value below is derived from the cited lines, not executed. Paths relative to the
proxima repo root; `interop` = `proxima-model-interop/src`.

## shape chosen, and the one contested decision

Shape: the routed FFN is descriptor data (H4-H6); everything else the seven `qwen35moe_*` fields do
is ONE config section, `[placement.experts]` (H8), with four parts: an `execution` enum (replaces 4
booleans), a residency rule (3 numbers + 2 budgets), a low-precision tier (a sidecar file), and two
page-retention enums (replace 6 env vars). No family name appears in any key.

Contested decision: should the residency/offload policy become "a pure decision function in core plus
a placement pipe type"? Chosen: the decision half is ALREADY pure and moves to core only by splitting
`residency.rs` along a line that already exists (section 4); the placement half is ALREADY a sink
callback (`interop/residency.rs:384-390`) and does NOT become a new pipe type, because the call site
is the same line before and after (section 4, gate 2). What changes is that the callback's behaviour
is selected by two config enums instead of six env vars.

## 1. ground: what a user writes today

There is no library-level config for these fields. `ServingConfig` derives `Debug, Clone, Copy,
PartialEq` only (`interop/serving.rs:719`); the only conflaguration mirrors are `PromptCacheSettings`
(`interop/prompt_cache_settings.rs:17`) and `SpeculativeSettings` (`interop/speculative_settings.rs:253`,
exported `interop/lib.rs:153-155`). The nearest thing to "the config a user writes" is an
example-local struct, `GenerateConfig` (`examples/gguf_generate.rs:72-162`, prefix `PROXIMA`), whose
env names are `PROXIMA_QWEN35MOE_PRE_GATHER`, `..._PERSISTENT_CUTS`, `..._RESIDENCY_BUDGET_BYTES`,
`..._EXPERT_PREFETCH`, `..._LAYER_WINDOW`, `..._MONOLITHIC_ALL_LOW`, `..._MONOLITHIC_HIGH_MMAP`
(fields at `examples/gguf_generate.rs:77-98`), plus `expert_sidecar` and `expert_sidecar_source`
strings (`:135-138`) that exist ONLY in the example (no field on `ServingConfig`). The sidecar is
attached imperatively after load (`examples/gguf_generate.rs:648-727`) through one of four methods
(`interop/generate/decode.rs:1840-1877`).

The seven fields, their defaults, and every consumer in library source:

| field (`interop/serving.rs`) | default | consumers |
|---|---|---|
| `qwen35moe_pre_gather: bool` :914 | false (:1177) | `decode.rs:3120-3124`, `:4103-4110`, `:4501-4507`, `:6229-6234`; predicate `decode.rs:7255-7257` is `configured && routed_experts` |
| `qwen35moe_persistent_cuts: bool` :917 | false (:1178) | `decode.rs:4589-4596` -> `pregather.rs:196-201,429` (builds caller-owned placed buffers for cut tensors) |
| `qwen35moe_residency_budget_bytes: u64` :920 | 0 (:1180) | `decode.rs:3302-3323` (policy built iff routed && sidecar && budget > 0); ALSO `decode.rs:3424` (read-cache limit) |
| `qwen35moe_expert_prefetch: bool` :942 | false (:1185) | `decode.rs:3407-3408`, `:4644-4708`, `:6267`; every one behind crate feature `qwen35moe-expert-prefetch` (`decode.rs:3403`, `load_model.rs:1069-1072`) |
| `qwen35moe_monolithic_all_low: bool` :944 | false (:1187) | `decode.rs:3132-3137`, `:4110`, `:4509-4514`, `:7264-7271` |
| `qwen35moe_layer_window: usize` :949 | 1 (:1186) | `decode.rs:4833,4958` -> `pregather.rs:615-649`; validated `serving.rs:1395-1410` |
| `qwen35moe_monolithic_high_mmap: bool` :953 | false (:1188) | `decode.rs:3126-3128`, `:4506` |

Adjacent fields that belong to the same decision and live outside the seven:
`expert_residency_schedule.per_layer_budget_bytes` (`serving.rs:663-670`, field :1070),
`expert_weights_budget_bytes` / `dense_weights_budget_bytes` / `activations_budget_bytes` /
`kv_cache_budget_bytes` (`serving.rs:927-940`), `gpu_memory_limit_bytes` (:778), `gpu_memory_fit`
(:774, default true :1148).

## 2. the config a user would write

Real data for the routed shape (`proxima-model-interop/tests/fixtures/llama-parity/qwen35moe/gguf_kv.txt:13,16-19`):
`block_count = 40`, `expert_count = 256`, `expert_used_count = 8`, `expert_feed_forward_length = 512`,
`expert_shared_feed_forward_length = 512`. That file has no `expert_gating_func` key (grep count 0), so
`gating` is a profile value, not GGUF data.

```toml
[model.ffn.routed]
gating = "softmax"
activation = "silu"
projection = "per_route"
expert_bias = false
expert_scale = false

[model.ffn.shared]
gate = "sigmoid_scalar"

[placement.experts]
execution = "pre_gather"
layer_window = 1
persistent_cuts = false

[placement.experts.low]
path = "/models/qwen3.6-35b-a3b.low.pxexsc"
access = "pread"
checkpoint_reads = true
require_byte_identical = true
retain = "route"
retain_checkpoint = "keep"

[placement.experts.high]
budget_bytes = 8589934592
read_cache_bytes = 8589934592

[placement.experts.policy]
kind = "hotness_ema"
ema_rate = 0.1
hysteresis_margin = 0.0
min_dwell_tokens = 0
prefetch = "none"

[placement.budget]
dense_weights = 0
expert_weights = 0
activations = 0
kv_cache = 0
```

Defaults equal today's behaviour: `execution` default is `full_graph`; `ema_rate = 0.1`,
`hysteresis_margin = 0.0`, `min_dwell_tokens = 0` are `ResidencyConfig::default()` (`interop/residency.rs:133-142`);
the file above overrides only `execution`, the sidecar tier and `budget_bytes`.

`execution` is one enum, `full_graph | pre_gather | monolithic_all_low | monolithic_high_mmap`
(section 6, GAP-4). `access = pread|mmap` and `checkpoint_reads` are the 2x2 that the four
`attach_expert_sidecar*` methods encode in their names (`decode.rs:1840-1877`).
`retain` is `keep | route | all_per_layer | all_per_call`; `retain_checkpoint` is
`keep | evicted | per_layer | per_layer_immediate` (GAP-7).

## 3. field map: every key to a consumer, or a gap

"EXISTS" = a field and a consumer exist on main today under another name. "GAP-n" = no consumer;
section 6 gives stage, missing input, smallest generic change.

| key | consumer | status |
|---|---|---|
| `model.ffn.routed.*` (experts, experts_used, expert_ff) | `ModelDescriptor.expert_count/expert_used_count/expert_feed_forward` (`proxima-tensor/src/spec/descriptor.rs:64-71`); `qwen35moe::from_metadata` reads the keys (`interop/qwen35moe/hparams.rs:170-176`) but qwen35moe does NOT lower through the descriptor (`interop/qwen35moe/program.rs:218-222`) | GAP-2, GAP-3 |
| `gating`, `activation`, `projection`, `expert_bias`, `expert_scale` | `MoeFfnSpec` fields (`proxima-tensor/src/spec/mistral_layer_moe.rs:622-652`), set as literals at `interop/qwen35moe/program.rs:122-136`; `MoeFfnSpec` is `Copy` over `NodeId`s with no serde, `LayerFfnConfig` (`attention_forward.rs:304-345`) has no serde | GAP-1 |
| `model.ffn.shared.gate`, shared width | `append_sigmoid_gated_shared_expert` (`interop/qwen35moe/shared_expert.rs:25-34`) called at `program.rs:137-146`; `FfnCombination` has only `Exclusive`, `ParallelDenseMoe` (`attention_forward.rs:283-286`) | GAP-2 |
| `placement.experts.execution` | 3 booleans + `layer_window` (table in section 1) | GAP-4 (enum) |
| `placement.experts.layer_window` | `serving.rs:949`, `pregather.rs:615-649` | EXISTS |
| `placement.experts.persistent_cuts` | `serving.rs:917`, `pregather.rs:429` | EXISTS |
| `low.path`, `low.access`, `low.checkpoint_reads` | example only (`examples/gguf_generate.rs:135-138,648-727`) | GAP-6 |
| `low.require_byte_identical` | hard-coded `source == target` fn pointer (`decode.rs:7259-7261`, passed at `pregather.rs:1499`, type `expert_sidecar.rs:113`) | GAP-5 |
| `low.retain`, `low.retain_checkpoint` | six env vars: `PROXIMA_EXPERT_SIDECAR_KEEP_PAGES` (`pregather.rs:2161`), `PROXIMA_EXPERT_SIDECAR_DISCARD_PER_LAYER` (:2148), `PROXIMA_EXPERT_SIDECAR_DISCARD` (:2310), `PROXIMA_CHECKPOINT_DISCARD_PER_LAYER` (:2176), `..._IMMEDIATE` (:2177), `PROXIMA_EXPERT_CHECKPOINT_DISCARD` (`expert_sidecar.rs:1476`) | GAP-7 |
| `high.budget_bytes` | `serving.rs:920` -> `decode.rs:3302-3323` | EXISTS |
| `high.read_cache_bytes` | `decode.rs:3424` reuses `residency_budget` as the high-read-cache limit (`expert_sidecar.rs:396-401`) | GAP-8 |
| `policy.kind` | single policy, no selector | EXISTS as the only value |
| `policy.ema_rate`, `hysteresis_margin`, `min_dwell_tokens` | `ResidencyConfig` fields (`residency.rs:120-131`) set by `..ResidencyConfig::default()` at `decode.rs:3316` | GAP-5 |
| `policy.prefetch` | bool `serving.rs:942`, crate-feature gated | GAP-5 |
| `placement.budget.*` | `serving.rs:927-940` -> `decode.rs:2575-2583` -> `memory_fit.rs:326-362` | EXISTS, but see GAP-8 |

## 4. can the policy be a pure decision function in core plus a placement pipe?

Decision half: yes, and it is already pure. Evidence from `interop/residency.rs`:
- `observe` (:216-255) reads `self.config`, `self.states`, `self.last_token` and writes `self.states`
  and `self.last_token`. No slab, no IO.
- `reconcile` (:287-347) reads `self.states`, `self.config`, `self.last_token` and writes
  `self.target` and `self.actions`. No slab, no IO.
- `capacity` (:416-424) is `min(budget_bytes / high_bytes_per_expert, layers*experts)`.
- The only IO is `apply_at_boundary` / `apply_actions_at_boundary` (:353-410): they take
  `&mut ExpertSlab` and a callback, and mutate the policy's `resident` bits after the callback.
- The ONLY reasons the file is std-bound: `use crate::bind::Codec; use crate::{ExpertSlab, InteropError}`
  (:8-9), consumed by `ExpertPage` (:88-95) and the two `apply_*` methods; and the module itself is
  `#[cfg(feature = "std")] pub mod residency` (`interop/lib.rs:54-55`, re-exports :138-143). Storage is `Vec` (:186-188), so the
  decision half is tier-1 (alloc), not tier-3.
- Family names in non-test residency.rs: none. The two "qwen35moe" hits are test fn names (:744, :797).

Cut line (no new type): leave in core `ExpertAddress`, `RoutedExpert`, `ServePrecision`, `ServeDecision`,
`ResidencyAction`, `ResidencyConfig`, `ResidencyError`, the private `ExpertState`, and the methods
`observe`, `prefetch_candidates`, `reconcile`, `capacity`, `hotness`, `hottest_not_target`,
`coldest_target` (lines 1-87, 97-347, 412-480 minus `ExpertPage`). Leave in interop `ExpertPage` and
the two `apply_*` methods. The decision half then compiles under `--no-default-features --features alloc`.
To reach tier-3 the three owned `Vec`s (:186-188) become caller slices:

```rust
pub fn observe(heat: &mut ExpertHeat, rule: &HeatRule, token: u64) -> ServePrecision;
pub fn reconcile(
    heats: &[ExpertHeat],
    capacity: usize,
    rule: &HeatRule,
    now: u64,
    target: &mut [bool],
    moves: &mut [ResidencyAction],
) -> Result<usize, MoveOverflow>;
```

`ExpertHeat` is `ExpertState` (:154-161) made public; `HeatRule` is `ResidencyConfig` minus the two
byte fields. Fixed-capacity `moves` forces an overflow policy that `Vec` hid: emit the first N by
priority and leave the rest to the next boundary (safe because `reconcile` recomputes from `heats` each
boundary). `moves.len()` is therefore a real config value, `max_moves_per_boundary`, default
`2 * layers * experts` (today's `Vec::with_capacity(slots*2)`, :203). Not designed here: any change to the
selection algorithm. Read from code, not measured: each `hottest_not_target` / `coldest_target` call is a
linear scan of all `layers*experts` slots (:457-479) and is called once per fill/swap step (:302-325), so
a cold-start reconcile at 40 x 256 = 10240 slots is O(capacity x 10240). Any rewrite must reproduce the
first-max / first-min tie-breaks (:460-465, :473-478) to stay action-identical.

Placement half: it is a sink over `ResidencyAction` and already has a generic, box-free form:
`Apply: FnMut(&mut ExpertSlab, ResidencyAction) -> Result<(), InteropError>` (:384-390). The live call is
`policy.apply_actions_at_boundary(&mut slab, |slab, action| sidecar.apply_action(slab, self.checkpoint_mapping, action))`
(`decode.rs:2001-2003`).

Second gate, call site both ways, for a `PlaceExpert: Pipe<In = ResidencyAction, Out = ResidencyAction>`
type: before, `|slab, action| sidecar.apply_action(slab, checkpoint, action)`; after,
`|slab, action| place.apply(slab, action)`. Identical lines, so the type is a relocation and is not
minted. Two more reasons it fails: `Pipe::call` takes `&self` (`proxima-primitives/src/pipe/primitives.rs:101`)
while the slab mutation needs `&mut`, forcing interior mutability around `ExpertSlab`; and a config-length
stage chain cannot be an `and_then` chain, whose type is fixed at compile time
(`proxima-primitives/src/pipe/ext.rs:48-54`) - the fsm-techniques SPEC hits the same wall and caps cascades
at 4 arms (`fsm-techniques/SPEC.md` reachability row `CascadeTooManyTiers`). The config-selected behaviour
is a closed set of stages, so it is `enum + match` inside the existing callback:

```rust
pub enum RetainLow { Keep, Route, AllPerLayer, AllPerCall }
pub enum RetainCheckpoint { Keep, Evicted, PerLayer, PerLayerImmediate }
```

These two enums replace the six env reads in section 3. They are the only new types in this sketch, and
each is a closed set that today is an `if env::var_os(..)` ladder.

What is NOT config today even though it is a decision: `PROXIMA_DEBUG_*` reads
(`expert_sidecar.rs:1458`, `pregather.rs:622,950,...`) are diagnostics and stay env-gated telemetry;
the repo rule is a structured event plus a file-sink `Exporter`, not a config field.

## 5. worked example (doubles as the test)

Decision half, reusing the repo's own fixture so the number is traceable: `CONFIG` at
`interop/residency.rs:492-498` is `budget_bytes = 144, high_bytes_per_expert = 144, ema_rate = 0.5,
hysteresis_margin = 0.1, min_dwell_tokens = 2`; layers = 1, experts = 4. The ema update is
`ema' = ema * (1 - r)^(elapsed + 1) + r` (:237-241) and hotness is `ema * (1 - r)^(last_token - last_observed)` (:450-455).

Trace A, `stationary_trace_keeps_the_hot_expert_and_serves_high_after_boundary` (:525-558), expert 2 on
tokens 1..=6: token 1 elapsed = 1, ema(2) = 0 * 0.25 + 0.5 = 0.5, served Low (not resident); reconcile:
capacity = min(144 / 144, 4) = 1, fill loop picks expert 2 -> actions `[Page(0,2)]`; tokens 2..=6 serve
High (the test asserts `token > 1 => High`, :545-547) and stage `[]`.

Trace B, `shifting_trace_replaces_a_dwelled_resident_at_the_boundary` (:561-630): expert 0 on tokens
1..=3 gives ema(0) = 0.5, 0.625, 0.65625 and `resident_since = 1`. Token 4 routes expert 3: elapsed = 4,
ema(3) = 0 + 0.5 = 0.5. At `last_token = 4`: hotness(0) = 0.65625 * 0.5^1 = 0.328125, hotness(3) = 0.5.
Swap loop: dwelled = 4 - 1 = 3 >= 2, and 0.5 > 0.328125 + 0.1, so swap. Staged actions (evictions before
pages, :329-343): `[Evict(0,0), Page(0,3)]`. This matches the assertions at :606-629. (Hand-derived from
the cited lines; not executed.)

The sans-IO test for the core split: run Traces A and B through the free functions above over
`heats: [ExpertHeat; 4]`, `target: [bool; 4]`, `moves: [ResidencyAction; 8]` under
`--no-default-features --features alloc`, and assert the same served precisions and the same staged
`moves` as the two interop tests. At Qwen3.6 scale the matrix is `heats: [ExpertHeat; 40 * 256]`
(`block_count`, `expert_count` from the GGUF keys above; today sized at `decode.rs:3318-3319`).

Config parity (P4): the TOML `[placement.experts.high] budget_bytes = 144` +
`[placement.experts.policy] ema_rate = 0.5, hysteresis_margin = 0.1, min_dwell_tokens = 2`, with
`high_bytes_per_expert = 144` taken from the sidecar descriptors (`expert_sidecar.rs:518-529,1517-1519`),
loads to a value equal to `CONFIG` (:492-498), via the same loader/builder/`from_config` triple that
`PromptCacheSettings` ships (`prompt_cache_settings.rs:17-100`).

## 6. HOOK GAPS (stage, missing input, smallest generic change)

GAP-1. H4 describe. Missing input: the routed-FFN knobs are not serializable. `ModelDescriptor`
(`descriptor.rs:53-124`), `LayerFfnConfig` (`attention_forward.rs:304-345`), `FfnCombination` (:283-286),
`ExpertGatingFunc` (`mistral_layer_moe.rs:596-599`) carry no serde/Settings; `MoeFfnSpec` cannot (it holds
`NodeId`s). Smallest change: serde + `Settings` on the first four; `MoeFfnSpec` stays build-time. Already
owned by architecture-as-data slice 10a (fsm-techniques SPEC audit row A11). Counted here because the sketch cannot load without it.

GAP-2. H6 lower. Missing input: a routed layer with a sigmoid-gated shared expert is not a descriptor
shape; `FfnCombination` has `Exclusive` and `ParallelDenseMoe` only, so qwen3.6 keeps its own whole-model
builder (`interop/qwen35moe/program.rs:218-222`). Smallest change: one `FfnCombination` variant carrying
the shared expert's width and gate kind, and move `append_sigmoid_gated_shared_expert` into
`proxima-tensor::spec` (it already imports only `proxima_tensor::spec::{elementwise, reduce, sigmoid, silu}`,
`shared_expert.rs:11`). The GDN mixer half of the same builder is sketch 2's gap, not repeated here.

GAP-3. H6 lower -> H8. Missing input: the pre-gather plan cuts the graph at node ids that only the
bespoke builder returns (`Qwen35MoeLayerDiagnostics.router_logits/block_output/routed_output`,
`program.rs:169-196`; consumed `pregather.rs:196-330,438-440,699-701`), and the executor finds expert
stacks by weight-name substring (`name.contains("_exps.weight")`, `pregather.rs:685`;
`load_model.rs:1189`; `expert_slab.rs:59-63` fixes three projections). The generic builder's `MoeSite`
returns only `layer, selected, weights` (`mistral_layer_moe.rs:663-667`). Smallest change: add
`router_logits: NodeId`, `block_output: NodeId`, `expert_w: [NodeId; 3]` to `MoeSite`. All three are
already in scope inside `append_moe_ffn` (`logits` at :728-747, `expert_w_*` in `MoeFfnSpec` :626-631).
The cut mechanism itself is already generic: `split_layer_program` is documented as the shared cut
semantics for "future routed architectures" over `proxima_tensor::partition` (`execution.rs:179-190`).

GAP-4. H8 place. Missing input: four booleans encode one mode. Illegal combinations are silently inert:
`qwen35moe_monolithic_all_low_enabled(pre_gather = false, ..)` is `false` (`decode.rs:7264-7271`, asserted
`tests_all.rs:146`); `monolithic_high_mmap` also requires `pre_gather` (`decode.rs:3126-3128`) and then
disables it (`:4506`); `layer_window = 2` needs `pre_gather`, `!persistent_cuts`, a byte-preserving sidecar
(`serving.rs:1401-1410`, `pregather.rs:636-649` - the last is a run-time error, not a config error);
`all_low` needs `Serial` dispatch, enforced in the example, not the library
(`examples/gguf_generate.rs:399-407`). Smallest change: the `execution` enum plus `Validate` rows for
window, cuts, sidecar exactness, and dispatch, in the draft's reachability-matrix style
(`fsm-techniques/SPEC.md` "reachability matrix (R3)").

GAP-5. H8 place. Missing input: decision rules that exist but are unreachable or hard-coded:
`ema_rate`, `hysteresis_margin`, `min_dwell_tokens` (`decode.rs:3316` `..default()`); `admit_low_copy =
source == target` (`decode.rs:7259`); the prefetch predictor "previous token's routes at layer + 1" with
threshold `f32::NEG_INFINITY` and capacity 16 (`decode.rs:4660-4698`, `load_model.rs:1060-1063`); the
16-slot staging array (`decode.rs:4625-4633`, should be `expert_used_count` from the descriptor); and
`expert_prefetch` is a bool that is silently inert when the crate feature is off (`decode.rs:3403`; the feature
is off by default, `proxima-model-interop/Cargo.toml:13,17`; no validation in `serving.rs`). Smallest change: build `ResidencyConfig` from settings at `decode.rs:3309-3317`;
make `prefetch` an enum whose `none` is the default and which `Validate` rejects when the crate feature is
absent; size the staging array from `experts_used` (principle 12 `sized` constant for the hard cap).
Cache validity: `CacheKey` already keys on pool size because it "decides which experts run at the low codec"
(`prompt_cache_key.rs:64-66`), and `ServingConfig` is destructured exhaustively there (`:89-180`, so a new
field cannot be added without a key decision). The three rule fields must therefore enter the key, or be
excluded with a stated reason only when `require_byte_identical = true` (then low and high bytes are equal
and no row changes).
Also delete or implement `expert_residency_schedule.per_layer_budget_bytes`: today any non-zero value
returns `UnsupportedServingConfig` at run time (`decode.rs:3287-3296`), a config field whose only consumer
is a refusal.

GAP-6. H8 place. Missing input: nothing in the library reads a sidecar path or access mode. Smallest
change: `[placement.experts.low]` consumed at load by one function that opens the file(s) and calls the
existing constructors (`MappedExpertSidecar::new/from_file` + `with_checkpoint_file`,
`expert_sidecar.rs:486-561,678`), replacing the four `attach_expert_sidecar*` entry points
(`decode.rs:1840-1877`) and the example's string switch. Note `mmap-window` and `pread` call the same
method in the example (`examples/gguf_generate.rs:663-706`): they differ only in the printed label.

GAP-7. H8 place. Missing input: host-page retention is six env vars, and its default is platform-specific
(per-route sidecar discard is on by default only under `cfg(all(feature = "metal", target_os = "macos"))`,
`pregather.rs:2160-2174`). Smallest change: `RetainLow` / `RetainCheckpoint` enums read inside
`ExpertSidecar::apply_action` and the per-layer block; the per-platform default comes from a build.rs
`sized` constant keyed on `CARGO_CFG_TARGET_OS` (principle 8/12, `conflag` skill "per-platform constants").

GAP-8. H8 place. Missing input: budgets are conflated and not placement-aware. (a)
`ExpertSidecarReadScratch::with_high_cache_limit_and_window_capacity(usize::try_from(residency_budget)..)`
reuses the pool budget as the read-cache limit (`decode.rs:3422-3426`), so one number sets two things.
(b) `apply_memory_fit_gate` charges `checkpoint_weight_bytes.expert_bytes` in full
(`decode.rs:2534-2540`; `MemoryBudget::derive` `memory_fit.rs:144-151`; `fit_context_length` `:238`) and
the load-time mapping fit charges the whole file (`pregather.rs:2576-2587`, override only by env
`PROXIMA_MAPPING_FIT_OVERRIDE`), although under `pre_gather` the device-resident expert footprint is the
selected-expert window plus the high pool (`decode.rs:3139-3146` unregisters the whole-checkpoint device buffer
for exactly that reason). I did not measure either gate against a real offload run;
this is read from the arithmetic. Smallest change: `high.read_cache_bytes` as its own field, and
`WeightClassBytes.expert_bytes` becomes an output of the placement decision (window bytes from
`mapped_window_capacity`, `expert_sidecar.rs:596-608`, plus `high.budget_bytes`) instead of the checkpoint
directory sum.

GAP-9. H4/H8. Missing input: offload is reachable only for the family whose `Architecture` trait object
returns `FfnRouting::Routed`; the doc says so (`architecture.rs:247-251`), only `Qwen35MoeArch` overrides it
(`qwen35moe/bind.rs:236-238`), the default is `Dense` (`architecture.rs:469-471`), and the guard is
repeated at `decode.rs:3122,3304,4503,4105,6231` and `pregather.rs:76,99,139,2734` and `decode.rs:1885`. Gemma4's MoE, lfm2moe
and mixtral therefore cannot use the same config even though their lowering reaches `append_moe_ffn` (`mistral_layer_moe.rs:718-721`).
The sidecar also hard-codes the family in error text (`expert_sidecar.rs:1207,1232,1267,1488,1498`).
Smallest change: "routed" is a descriptor fact (any layer with a routed FFN); `execution != full_graph`
is rejected by `Validate` when the descriptor has none; delete `FfnRouting`.

Other findings, not hook gaps:
- Three public entry points have zero callers anywhere in `*.rs` on main (grep over the tree, defs only):
  `execute_qwen35moe_pre_gather` (`decode.rs:2027`), `qwen35moe_layer_boundaries` (`pregather.rs:92`),
  `qwen35moe_layer_segments` (`pregather.rs:126`). They carry the typed `RouterPhase`/`GatherPhase`
  protocol (`qwen35moe/layer_boundary.rs`, `execution.rs`); the live loop uses a different path
  (`qwen35moe_pre_gather_plan` + `evaluate_qwen35moe_pre_gather`, `pregather.rs:196,544`). Two protocols
  for one seam; the unused one is the compensator, per the "follow the compensators inward" check.
- `expert_slab: std::sync::Mutex<ExpertSlab>` (`load_model.rs:1005`) is a bare std mutex; AGENTS.md
  lock discipline forbids it. The slab has one real owner (the decode loop holds the `StepGuard`,
  `decode.rs:4490-4491`).
- Residency is reconciled only when `runtime.uses_gpu()` (`decode.rs:6235`) while `observe` runs on both
  backends (`decode.rs:4709-4724`): on CPU the heat state updates and is never reconciled. Not explained.
- Catalog granularity: H8 is one row ("plan + device -> placed buffers") but this sketch needs three
  decisions with different owners: execution partitioning (graph shape), tier residency (pure policy),
  host-page retention (IO). One sentence; not redesigned.

## 7. what is not a pipe, and why

- `ExpertHeat` / `HeatRule` / `observe` / `reconcile`: a state matrix and pure functions, not an `In -> Out`
  step over a stream. Justified structurally: they run synchronously inside the decode step boundary over
  caller-owned slices (hot path, zero allocation), and R23b of fsm-techniques requires decisions to be
  tier-1 pure functions that interop wraps.
- The placement effect: an existing generic sink callback, kept as is (section 4).
- `RetainLow` / `RetainCheckpoint`: closed enums selected by config; a runtime-length stage chain is not
  expressible as a static `and_then` chain.
- The graph partition (`partition_at`) is a graph transform at lowering time, outside the algebra's runtime
  pipe form.

## 8. designs abandoned

- `PlaceExpert` as a `Pipe` type, composed with `and_then` from a config list: fails the second gate
  (identical call site) and the static-chain limit above.
- `Box<dyn PlacementStage>` list: ruled out by the box-free default; the stage set is closed.
- A `trait ResidencyPolicy` with per-technique impls: there is one rule family and three numbers; a trait
  would be the blanket-impl-under-a-new-name the repo rules forbid.
- Moving `ExpertResidency` wholesale into core: it carries `ExpertPage`/`apply_*` which need `ExpertSlab`
  (memmap2); the split along :88-95 / :349-410 is the only cut that removes the std imports.
- Keeping `per_layer_budget_bytes` in the config: its only consumer refuses it.

## verdict

fits after 9 named hook changes: GAP-1 (already owned by architecture-as-data 10a), GAP-2, GAP-3, GAP-4,
GAP-5, GAP-6, GAP-7, GAP-8, GAP-9. The pure-decision question is answered "yes, by splitting
`residency.rs` at :88-95/:349-410; the placement half stays a sink callback". Not claimed: that any of this
preserves token ids or speed; none of it was run.
