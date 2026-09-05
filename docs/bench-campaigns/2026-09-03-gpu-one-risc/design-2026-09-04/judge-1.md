# judge-1 (Plan-1 = design-A, Plan-2 = design-B, Plan-3 = design-AB)

Ranking: **[Plan-3, Plan-2, Plan-1]**

Grounding runs (welded `cd /Users/brianbruggeman/repos/slot-0/proxima`, read-only, no cargo):
`grep -n '^pub fn \(plan\|execute\)' omega/src/metal.rs` = **11** (468, 519, 535, 1049, 1280,
1295, 1313, 1512, 1587, 1612, 1716) — Plan-3's count is right, the brief's "nine" is not;
`omega/src/msl.rs:444` `Q4K_BLOCK_ELEMENTS = 256`; `gather_count` gates at `msl.rs:1047`, `1218`,
`1389`; `proxima-tensor/src/bind.rs:160-162` states `StepArg` is "a plain index into a side
table ... never a `Box<dyn>` recursive tree"; `proxima-tensor/Cargo.toml:37` `config = ["std",
"dep:bon", "dep:conflaguration", ...]` and `lib.rs:103-104` "The `alloc` tier never sees it";
`proxima-tensor/src/map.rs:99-107` "every backend ... carries every buffer as f32 ... Lifting the
ceiling means adding real integer buffers"; `proxima-primitives/src/pipe/primitives.rs:91-101`
`Pipe { type In; type Out; type Err; fn call(&self, input: Self::In) -> impl Future<..> }` — no
lifetime parameters, no GATs, `&self` only; `proxima-model-interop/src/generate.rs:1541`
`impl<'file> Pipe for LoadedModel<'file>`.

Bytes recomputed independently: FFN 3,170.9 + attn 755.0 + output.weight 107.5 + KV 134.2 =
**4,167.6 MB** vs the brief's 4,169 (0.03%). Plan-1's "125 MB unexplained residual" IS the KV
term: it priced KV at 34 positions (8.9 MB) instead of the brief's 0.134 GB, then scaled that
residual by the Q3_K factor in its D.7 table. Plan-2 and Plan-3 both reconcile; only Plan-3
prices the union-density interaction (k=4 union `1-(1-d)^4`, given at the 0.6 kill boundary) and
the down-projection contraction-axis constraint (elision granularity forced to 256 =
`Q4K_BLOCK_ELEMENTS`, because `ffn_down` carries the hidden axis as `reduce_dim`,
`msl.rs:3184-3189`, so an unstructured selection saves down nothing).

## Scores (0-5)

| axis | Plan-1 | Plan-2 | Plan-3 | one-line reason |
|---|---:|---:|---:|---|
| risk surface | 3 | 3 | 5 | P1 deletes the hand attention kernel (`msl.rs:2586`, already rescaled online softmax + threadgroup staging) before landing the replacement schedule; P2 stages it but has no seam; P3 lands R5 + cooperative staging in the same card and keeps `cached-attention-streaming` (`Cargo.toml:33`) selectable |
| ordering | 2 | 3 | 5 | P1 CARD A2's greps require CARD A3's deletions; P2's cards 3/4 need the integer `cached_len` its card 8 delivers; P3 names both defects and emits one ordering |
| rollback | 1 | 2 | 5 | P1 renames the feature (no bisect seam); P2 stages by commit; P3 bisects by feature selection, deleting the old path only after a full measurement round |
| missing steps | 3 | 3 | 4 | P1/P2 assume router weights that do not exist and (P2) omit selector dispatch cost; P3 adds D3-calib (fit 4096x56/layer, 4.1 MB/token, sidecar) and B0 (`CountingAllocator`, which exists only privately in `proxima-telemetry`), but leaves ~12 signature types undefined in an explicit UNFINISHED block |
| hidden coupling | 3 | 2 | 5 | P2 couples §D's KV arithmetic to a band its own R4 calls advisory and to `BandBound::Dynamic`'s integer operand, which `map.rs:99-107` says no backend plumbs; P3 makes the domain binding and the length a symbol |
| observability | 2 | 4 | 5 | P1 leaves `classify_kind` (`metal.rs:1631-1692`) grepping MSL while collapsing every fusion class into one kind; P2/P3 return `KernelFamily`, P3 adds a gate that reproduces the census 225/391 split before fusion lands |
| scope discipline | 3 | 4 | 5 | P1 mints nine plan-time pipes plus `ModelSpec`/`StageSpec`/`lower_into`; P3 deletes eleven such types by writing both call sites |
| pipe questions | 2 | 4 | 5 | P1's `PlanSlots`/`ScheduleBarriers`/`PackUniforms` are the identical line as functions and are async for pure interval assignment; P2 kills a `Fuse` stage on the second question but keeps `Advance`/`BindInputs`/`StopPolicy`; P3 answers both questions in code for all eleven |
| no_std tier | 4 | 2 | 5 | P2 derives `Settings + Deserialize` on `FusionRules`/`FusionCost` on the bind path, which is the `config` feature = std (`Cargo.toml:37`, `lib.rs:103-104`), and needs integer buffers; P1 forces the symbol form; P3 does both and gates the alloc build to name its modules |
| RISC claim | 3 | 5 | 5 | P1's `FusedRegion { ops: Vec<BoundOp> }` makes `BoundOpKind` the recursive tree `bind.rs:160-162` rejects by name, duplicates extents/domain, and has no mechanism for RoPE/GQA (no `IndexPattern::compose`); P2/P3's `Loop { stages }` is 5 kinds to 3, P3 additionally states the cost (a multi-stage traversal interpreter five emitters must implement) |
| bytes arithmetic | 2 | 3 | 5 | P1 mis-sources KV and scales an unexplained residual; P2 reconciles the decomposition but takes 161 GB/s and `dispatch_ns = 10_300` from the ROW 281/282 serialized profile the brief marks superseded, and carries no host term; P3 uses ROW 287 in-buffer (23.24 ms / 4.034 GB = 173.6 GB/s achieved; wall - ALL = 5.96 ms host residual), prices union density and selector cost, and states the negative answer (7.25 ms at the only measured bandwidth) |
| quality gates / kill | 3 | 5 | 5 | P1 gives EM/ppl thresholds without a degenerate control; P2 pre-registers the hypothesis, mechanism metric (mass recall), ns/element gate against the arm that already lost (ROW 180/181, 0.2-0.29 vs 0.057 ns/element) and a random-row control that must fail; P3 adds kills for k' and union density and retracts text-identity on cards that reassociate floating point |
| signatures type-check | 1 | 2 | 5 | `Pipe` has no lifetime params: P1's `Read::Out = LogitsView<'_>` and `In = (Encoded<'static>, ..)` are not expressible; P2's `Advance: Pipe<Out = ReadyStep<'p>>` needs `&'p mut` from `&self` and its `encode<'p>(.., out: &'p mut [Command<'p>])` permits one encode per plan; P3 restricts every pipe boundary to POD and cites `primitives.rs:91-101` for why |
| abandoned designs | 5 | 4 | 5 | all three name kills; P1's `FlashAttention` paragraph-as-finding and the draft-source-is-a-pipe / selector-is-not asymmetry are the sharpest single results, and P3 carries them forward |
| **total /70** | **37** | **46** | **69** | |

## Ranking, deciding axis per slot

1. **Plan-3** — the only candidate whose bytes section is computed from the ROW 287 in-buffer
   decomposition of record and therefore surfaces the 5.96 ms host residual and 173.6 GB/s
   achieved, and the only one that prices the two interactions the task names (union density,
   the `ffn_down` contraction axis at `Q4K_BLOCK_ELEMENTS = 256`).
2. **Plan-2** — the algebra is right (`Loop`, 5 kinds to 3, `IndexPattern::compose` as the
   minimal operation for RoPE/GQA) but it is scored down for a tier violation on the bind path
   and a band derived by peephole, which reproduces audit item 1 one level down.
3. **Plan-1** — the deciding axis is the bytes arithmetic: KV taken at 34 positions produces a
   125 MB "unexplained residual" that is the brief's own KV term, and that residual is then
   scaled by a codec factor; compounded by `FusedRegion` violating the recursion rule stated at
   `bind.rs:160-162` and nine plan-time pipes that fail the second binary question.

No axis was left unscored.
