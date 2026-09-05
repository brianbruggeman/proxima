# design-final — design-AB2 with the round-2 panel's fixes applied in place

<!-- r2-fix: base pin corrected. AB2 wrote "HEAD 35a139f"; `git rev-parse --short HEAD` in this
     session reads **ce05362**, which is what judge-r2-2 and judge-r2-3 also recorded. Every
     file:line below was re-opened at ce05362 this session. -->

Base pinned: `/Users/brianbruggeman/repos/slot-0/proxima`, HEAD **ce05362**. No build, no cargo, no
measurement was run in this session; every number is tagged MEASURED / DERIVED / ASSUMED with its
source. Provenance of each step: `AB §x`, `B2 §y`, `crit #n` (critique-AB finding n), or
`r2-fix` (round-2 panel, judge-r2-1/2/3).

---

## 0. The shape, and the five contested decisions

**Shape.** (i) `BoundOpKind::Loop { stages }` with a **declared** `Domain` of half-spaces on the shipped
`AxisIndex` replaces `Elementwise`/`Reduce`/`CachedAttention` (5 kinds -> 3). (ii) A fusion rule **is a
pipe** `Pipe<In = BoundProgram, Out = BoundProgram>`; a backend's capability is the composition it
builds — there is no capability set type. (iii) `plan()` is pure, alloc-tier, and **takes no codecs**;
codec-dependent kernel selection is `omega::schedule(&Plan, &[Option<QuantizedBlock<'_>>])`, std side.
(iv) The step is one typestate FSM; POD crosses every pipe boundary. (v) **Placement is a per-op field
on the schedule, not a process-wide backend** — a placement rule is a pipe, exactly like a fusion rule
(§B.4). (vi) Bytes: **3.5 ms/token is not reachable**, and at the *measured* device ceiling it is not
reachable *even with the FFN entirely deleted* — proof in §D.4.

**The five decisions the brief names, each decided with one reason.**

| # | conflict | taken | reason |
|---|---|---|---|
| 1 | `Loop { stages }` (AB) vs `ReduceChain { stages, operands, band, epilogue }` (B2) | **`Loop`** | <!-- r2-fix: the panel is right that "a `Band { axis, lower, upper }` cannot state causal and sliding-window jointly" is FALSE — affine `lower`/`upper` on one axis state exactly that pair. The reason is withdrawn and replaced with the axis that actually decides it: declared-vs-derived. --> `ReduceChain` is additive (Elementwise + Reduce + ReduceChain = 3 kinds, one of them new); `Loop` is subtractive (5 -> 3, `Elementwise` = 1 stage `fold: None`, `Reduce` = 1 stage with a `Fold`). The deciding axis is **declared vs derived**: B2's `Band` is *recognised* from an `Iota`/`Greater`/`Select` spelling by a peephole (its own §A.2), so an unrecognised spelling of the same mask silently falls back to reading every key row — the exact silent-fallback failure `lowering-audit` finding 1 records one level up, and KV bytes then depend on whether an analysis fired. AB2's `Domain` is **written by the author of the program and binding on the lowering**: a point outside it is not read, not accumulated, not written, so KV bytes are a property of the program. `Domain: SmallVec<[HalfSpace; 4]>` also generalises past two endpoints on one axis (ALiBi's per-head slope bound, MoE's expert-membership constraint) at the same size. B2's `epilogue: Option<ComposedBody>` is subsumed by a trailing `fold: None` stage — one mechanism, not two. |
| 2 | `FusionRules` set (AB) vs rule-is-a-pipe (B2) | **B2's** | B2 §A.4 wrote the composition (`FusePrologue.and_then(FuseEpilogue).and_then(ShareAxisChain)`) and the call sites are **not** identical: a backend that composes rules can insert a rule the core does not ship; a backend handed a 4-bool set cannot. It passes both binary questions where AB's set passes only the first. It also deletes the type `crit #4` proved uncompilable (`PlanKey: Hash + Ord` over a `FusionRules` that derives neither). |
| 3 | barrier schedule: per-point `resources` (B2) vs `barriers: Vec<BarrierSet>` (AB) | **B2's `BarrierPoint`** | AB's `BarrierSet` names the slots and destroys both *where* and *why*; `BarrierPoint { before, resources, cause }` keeps all three. It also collapses `crit #4`'s three spellings (`BarrierSet` / `BarrierRange` / `MAX_INLINE_BARRIER_SLOTS`) into one name plus one sizing const. `Hazard` extends to cross-engine dependencies in §B.4 without a fourth spelling. |
| 4 | spec loader tier (B2 §C) | **B2's tier split, without the pipe** | Take: std-gated loader producing `Vec<Op>`, `build.rs` baking `&'static [Op]` below std (conflaguration tier-2, the `proxima-telemetry` `sized` pattern). Reject: wrapping it in `LoadSpec: Pipe`. Written both ways, `LoadSpec.call(src).await?` and `Vec::<Op>::try_from(&spec)?` are the same line — a relocation (P1 second question). §C now writes this out against the shipped `#[cfg(feature = "config")] pub mod spec;` (`proxima-tensor/src/lib.rs:219-220`). |
| 5 | what pure `plan()` takes, given `QuantizedBlock` is std-gated (`crit #3`) | **nothing codec-shaped** | `grep -rn BlockCodec` = 0 repo-wide: AB rested a signature on a type that does not exist. The shipped spelling is `QuantizedBlock<'a>` at `proxima-tensor/src/cpu.rs:3091` inside `#[cfg(feature = "std")] pub mod cpu;` (`proxima-tensor/src/lib.rs:193-194`, opened). Minting an alloc-tier twin duplicates a variant list. So `plan()` drops the argument: kernel selection by codec is an **emitter** decision and lives where the emitter lives. |

**The discriminator that decides which plan-time things are pipes** (this is what AB and B2 disagreed
about without naming it): a stage is a pipe when *a caller can supply one the library does not ship*.
Fusion rules pass — a backend composes its own set, including its own rule. <!-- r2-fix: placement
rules pass for the identical reason and are added here. --> **Placement rules pass** — an operator can
supply "layers 28..31 on CPU" without the library shipping that policy. `PlanSlots`,
`ScheduleBarriers`, `PackUniforms`, `Encode`, `Read`, `Sample`, `BindInputs`, `Advance`, `StopPolicy`
fail: the library ships the only implementation of each and the call site is the identical line, so
they stay plain functions (AB's result, kept).

---

# D. Bytes and time — redone from the steady-state decomposition (PRIMARY)

## D.1 The anchor, which now reproduces

`dispatch-census.md:25-30` is the decomposition of record and it supersedes AB §0.3 entirely
(`crit #1`, `crit #2`): **wall 28.82 = gpu_exec 27.52 + host 1.30** on plan-HIT steps 3..7. There is no
5.96 ms host residual — that number was `33.00 − 27.04` over a 7-step mean containing two plan-miss
steps that pay bind+compile. AB's ordering rationale (`design-AB.md:1168-1170`) rested on it and is
**void**; the card order below is rebuilt from "gap 11.4 ms is GPU-side".

| term | ops | ms | provenance |
|---|---:|---:|---|
| MATVEC | 225 | 23.24 (CoV 0.7%) | MEASURED, census:32 |
| NOT-MATVEC aggregate | 391 | 4.05 (CoV 1.5%) | MEASURED, census:33 |
| host | — | 1.30 | MEASURED, census:27 |
| **sum** | 616 | **28.59** | DERIVED; wall 28.82 MEASURED — 0.23 ms (0.8%) unaccounted |

Per-class arms (attn 3.0-3.6 CoV 9.9%, coop 1.3-1.8 CoV 18%, elem 1.76 CoV 1.1%) sum to 6.1-7.2 ms
against a 4.05 ms aggregate, so **2-3 ms of non-matvec work overlaps in the production command buffer**
and a per-class arm may not be subtracted from the aggregate. Every route row below uses the aggregate.

**Bandwidth reconciliation (`crit #15`).** The census's 179 GB/s divides *all* 4.169 GB by the *matvec*
23.24 ms. KV is streamed by the 32 attention ops, not by the 225 matvecs. Weights-only:
4,033.4 MB / 23.24 ms = **173.6 GB/s MEASURED-derived** on the matvec stream, and
134.2 MB / 3.3 ms = **40.7 GB/s** on the attention/KV stream. Both figures are used, never mixed.

## D.2 The byte table, and the ffn_down granularity that forces the group size

Weights (openchat-3.5 Q4_K_S, Q4_K = 144 B / 256 elt, `omega/src/msl.rs:439`, `Q4K_BLOCK_ELEMENTS = 256`
at `msl.rs:444`, opened): gate+up 2,113.9 MB, down 1,056.96, attn 755.0, `output.weight` (Q6_K) 107.5
= 4,033.4 MB. KV f32 at this context 134.2 MB. Total 4,167.6 MB MEASURED-consistent with the census.

`ffn_down` contracts over the 14336 axis (`PackedRowBlock { weight, other, reduce_dim, codec }`,
`msl.rs:3184-3189`), and the Q4_K superblock is 256 elements **along that same axis**. Unstructured
elision at density d leaves `1-(1-d)^256` of down's blocks live: at d = 0.10 that is 1.000 —
**zero bytes saved**. So the selection granularity is a whole superblock: **G = 14336/256 = 56 groups**,
gate/up elide 256 whole rows per group, down drops one whole block per output row. Group density
`g_u >= d_u` always, and both are measured, never assumed.

Union: a k-token pass reads the union of k row sets. Independent bound `d_u = 1-(1-d)^k`; correlation
pulls it down by an unmeasured amount. Both densities are card D3's deliverable.

## D.3 The model

```
ms/token = [ (gate+up)·c_ffn·d_u + down·c_ffn·g_u + attn·c_attn + out·c_out·s + KV·c_kv + router ] / (A · B_w)
         + non_matvec / A + host / A
```
`A` = accepted tokens/pass (needs the s-axis fold first, `msl.rs:3171-3200` — without it weights
re-stream k times and A cancels to 1.0). `router` = 4.1 MB/token (56 groups x 4096 x Q4_K x 32 layers).
`non_matvec` = 4.05 MEASURED today; 1.5 ms DERIVED if the attention arm reaches its own byte floor
(134.2 MB / 173.6 GB/s = 0.77 ms) and the 616 -> 264 dispatch ceiling (census:62) lands.

## D.4 What is reachable, plainly

<!-- r2-fix: D0 is now MEASURED (`proxima-tensor/docs/discipline.md:21198-21258` on branch
     `perf/device-streaming-ceiling`, harness `omega/tests/device_streaming_ceiling.rs`, commit
     a8d9c06): best cell = `nocopy_resident`, wide grid, 4 GB, **run 1 = 264.29 GB/s (CoV 26.84%),
     run 2 = 237.79 GB/s (CoV 29.40%)** on a box carrying 30-47 concurrent builder processes. The
     ASSUMED `@300`/`@400` columns are therefore RETRACTED and replaced with the measured range. The
     empty-dispatch fixed cost (0.48-0.53 ms, CoV 62-65%) carries the same contention signature, so
     the range is reported as a range, not resolved to a point. -->

**At the *measured* device ceiling, 3.5 ms/token is arithmetically impossible — even with the entire
FFN deleted.** Budget at A = 2.18 with the attention arm fixed: non_matvec/A + host/A = 1.5/2.18 +
1.30/2.18 = 1.29 ms, leaving 2.21 ms. At the best ceiling ever observed (264.29 GB/s) that buys
584 MB/token = **1,273 MB/pass**; at the second run's 237.79 GB/s, 525 MB/token = **1,146 MB/pass**.
The terms no elision lever touches — attn 755.0 + KV(f16) 67.1 + out(Q4_K) 73.7 = **895.9 MB/pass** —
leave an FFN budget of 250-377 MB/pass against 932.6 MB/pass in the most favourable elision row below.
FFN would have to shrink a further **2.5-3.7x beyond the favourable elision case**. This is the number
that hurts and it is stated first.

<!-- measured 2026-09-04: k' --> The `A=2.18` column below was ASSUMED (~~struck~~); §D.4a recomputes
every row at the two MEASURED k' values (1.36 at k=4, 1.49 at k=8).

| route | MB/pass | MB/token (~~A=2.18~~ ASSUMED) | @173.6 M (today's matvec) | @237.79 M (ceiling low) | @264.29 M (ceiling high) | vs 3.5 |
|---|---:|---:|---:|---:|---:|---|
| today (A=1) | 4,167.6 | 4,167.6 | 23.24+4.05+1.30 = **28.59** | — | — | — |
| R1 lossless+codecs (Q3_K FFN, out Q4_K, KV f16, no elision) | 3,318.4 | 1,522.2 | 8.77+1.86+0.60 = **11.23** | 6.40+1.86+0.60 = 8.86 | 5.76+1.86+0.60 = 8.22 | misses |
| R2 + elision at the kill boundary (d_u .60 / g_u .70) | 2,434.1 | 1,116.6 | 6.43+1.86+0.60 = **8.89** | 4.70+..= 7.16 | 4.22+..= 6.68 | misses |
| R2 favourable (d_u .35 / g_u .45) | 1,828.5 | 838.8 | 4.83+1.86+0.60 = **7.29** | 3.53+..= 5.99 | 3.17+..= 5.63 | misses |
| R2 favourable + attention arm at its byte floor (non_matvec 1.5) | 1,828.5 | 838.8 | 4.83+0.69+0.60 = **6.12** | 3.53+0.69+0.60 = **4.82** | 3.17+0.69+0.60 = **4.46** | **misses at the measured ceiling** |

<!-- r2-fix: the last row previously read "meets only @400 ASSUMED". With D0 measured, the whole
     favourable stack at the best ceiling ever observed is 4.46 ms — the target misses on every
     column of the table, and no column is assumed any more. -->

## D.4a Recomputed at the measured k' (2026-09-04)

<!-- measured 2026-09-04: k' --> Every cell below is §D.3's own formula, `MB/token(A) / B_w +
non_matvec/A + host/A`, evaluated at MEASURED A in place of the ASSUMED 2.18. No bandwidth number is
invented — `173.6`, `237.79`, `264.29` are the same three measured figures as the table above.

**At A = 1.36 (mean k', k=4, ngram 2-4, `real_run3.log`):**

| route | MB/pass | MB/token (A=1.36) | @173.6 M | @237.79 M | @264.29 M | vs 3.5 |
|---|---:|---:|---:|---:|---:|---|
| R1 lossless+codecs | 3,318.4 | 2,440.0 | 14.06+2.98+0.96 = **18.00** | 10.26+2.98+0.96 = 14.20 | 9.23+2.98+0.96 = 13.17 | misses |
| R2 + elision at kill boundary | 2,434.1 | 1,789.8 | 10.31+2.98+0.96 = **14.25** | 7.53+2.98+0.96 = 11.47 | 6.77+2.98+0.96 = 10.71 | misses |
| R2 favourable | 1,828.5 | 1,344.5 | 7.74+2.98+0.96 = **11.68** | 5.65+2.98+0.96 = 9.59 | 5.09+2.98+0.96 = 9.03 | misses |
| R2 favourable + attention arm at its byte floor (non_matvec 1.5) | 1,828.5 | 1,344.5 | 7.74+1.10+0.96 = **9.80** | 5.65+1.10+0.96 = 7.71 | 5.09+1.10+0.96 = **7.15** | misses |

**At A = 1.49 (mean k', k=8, ngram 2-4, `real_run3.log`):**

| route | MB/pass | MB/token (A=1.49) | @173.6 M | @237.79 M | @264.29 M | vs 3.5 |
|---|---:|---:|---:|---:|---:|---|
| R1 lossless+codecs | 3,318.4 | 2,227.1 | 12.83+2.72+0.87 = **16.42** | 9.37+2.72+0.87 = 12.96 | 8.43+2.72+0.87 = 12.02 | misses |
| R2 + elision at kill boundary | 2,434.1 | 1,633.6 | 9.41+2.72+0.87 = **13.00** | 6.87+2.72+0.87 = 10.46 | 6.18+2.72+0.87 = 9.77 | misses |
| R2 favourable | 1,828.5 | 1,227.2 | 7.07+2.72+0.87 = **10.66** | 5.16+2.72+0.87 = 8.75 | 4.64+2.72+0.87 = 8.23 | misses |
| R2 favourable + attention arm at its byte floor (non_matvec 1.5) | 1,828.5 | 1,227.2 | 7.07+1.01+0.87 = **8.95** | 5.16+1.01+0.87 = 7.04 | 4.64+1.01+0.87 = **6.52** | misses |

Required bandwidth, same method as the ASSUMED-A calculation below (`favourable MB/token / (3.5 -
non_matvec/A - host/A)`): at A=1.36 the budget is 3.5 - 1.10 - 0.96 = 1.44 ms, so B_w >= 1344.5/1.44 =
**932.8 GB/s** — 3.53-3.92x above the measured 237.79-264.29 GB/s ceiling, not the 1.44-1.60x the
ASSUMED A produced. At A=1.49 the budget is 3.5 - 1.01 - 0.87 = 1.62 ms, so B_w >= 1227.2/1.62 =
**757.1 GB/s** — 2.86-3.18x above the ceiling.

**Reachable floor, recomputed:** the design's own "4.46-4.82 ms" line below assumed A=2.18. At the
measured A the same row (R2 favourable + attention arm at its byte floor) reads **7.15-9.80 ms/token
at A=1.36** and **6.52-8.95 ms/token at A=1.49**, against llama.cpp's 17.45: 1.78-2.44x at A=1.36,
1.95-2.68x at A=1.49 — not the 3.6-3.9x the ASSUMED A produced. R1 alone (no elision, no attention-arm
fix) at A=1.36 and today's 173.6 GB/s reads 18.00 ms/token, which is slower than llama.cpp's 17.45; at
A=1.49 it reads 16.42 ms/token, 1.06x llama.cpp, not the 1.55x the ASSUMED A produced.

The design's own kill criterion below, unedited: `A < 1.5` at k=4 kills multi-token. Measured
A(k=4) = 1.36 < 1.5 — the criterion is met on this corpus with n-gram drafting alone. Measured
A(k=8) = 1.49 is also < 1.5.

Solving for each requirement independently, holding the others at their favourable value:
- **bandwidth B_w >= 379.5 GB/s** (838.8 MB/token / 2.21 ms) — **1.44-1.60x above the measured ceiling
  of 237.79-264.29 GB/s**. Not a tuning target; unreachable on this host. Previously stated as "95% of
  the M1 Max spec sheet", which was an ASSUMED spec-sheet number and is withdrawn.
- **A (accepted tokens/pass)**: design required ~~>= 2.18 ASSUMED~~. <!-- measured 2026-09-04: k' -->
  MEASURED on real greedy Metal streams (8 prompts x 64 tokens, n-gram prompt-lookup drafting,
  ngram 2-4, branch `test/draft-acceptance-harness`, `draft_acceptance_aggregate` lines in
  `real_run3.log`): mean k' = 1.2330 (k=2), **1.3609 (k=4)**, 1.4850 (k=8). By category at k=4: code
  1.93, chat 1.35, list 1.15, prose 1.01 — n-gram drafting tracks repetition in the prompt, not a
  general property of the model. The s-axis fold must still land first, or the denominator is 1.0.
- **d_u(k=4) <= 0.35 and g_u(k=4) <= 0.45** at the 256-group granularity.
- **attention arm at its byte floor** (3.0-3.6 -> ~0.77 ms), which is not a byte lever at all.
All four, simultaneously, plus a bandwidth the box does not have. 3.5 ms/token does not survive.

<!-- measured 2026-09-04: k' --> The paragraph below is at the ASSUMED A=2.18; §D.4a recomputes it at
the two MEASURED k' values.

**Reachable floor under measured conditions:** R1 at **11.23 ms/token** — 2.55x today, and 1.55x
*faster* than llama.cpp's 17.45 (today is 1.65x slower). With the elision stack clearing its quality
gate at favourable densities, the attention arm fixed, and the matvec stream reaching D0's own measured
ceiling, **4.46-4.82 ms** = 3.6-3.9x llama. The largest single remaining term is still the matvec
stream's shortfall against its *own device*: 173.6 GB/s is **65.7-73.0% of the measured ceiling**, and
closing it is worth 4.0334 GB x (1/173.6 − 1/237.79 .. 1/264.29) = **6.28-7.98 ms/token at zero quality
risk**, more than every lossy lever combined. That headroom is 1.37-1.52x, not the 2.3x a 400 GB/s
spec sheet implied — the cards below are ordered on the measured figure.

**Kill criteria** (each halts its lever and the sensitivity row says whether the target survives):
`d_u(4) > 0.6` or `g_u(4) > 0.75` kills the elision composition; `A < 1.5` at k=4 kills multi-token
<!-- measured 2026-09-04: k' --> (measured A(k=4) = 1.36 < 1.5: this criterion is met — §D.4a);
gathered packed-row GB/s < 0.85x the ungathered kernel kills the elision *kernel* regardless of bytes
(the landed CPU probe already measured the element-granular form at 0.20-0.29 ns/element vs 0.057 DRAM);
attention `gpu_exec_ms` worse than 3.6 + 9.9% CoV kills the `Loop` attention card; exact-match-at-64
< 0.98 (Q3_K/Q4_K codec cards) / < 0.95 (elision) kills that lever; **D0's own best cell failing to
clear 5% CoV on a quiet box retracts the ceiling range and re-prices every column above** (it has not
cleared it yet: 35 of 36 measured cells exceed the 5% trust line).

## D.4a addendum — repriced at the quiet ceiling (2026-09-05, `discipline.md` ROW 302)

D0's own kill criterion above has now fired the other way: `proxima-tensor/docs/discipline.md` ROW
302 clears 5% CoV on every WIDE-grid cell across two quiet runs (0.44-0.81% CoV), retiring the loud
237.79-264.29 GB/s range in favour of **381.24 GB/s** (worst quiet WIDE cell, `private_blit`, 4 GB)
to **381.88 GB/s** (best, same shape). Every ratio above computed against the loud range is
recomputed here against the quiet one; nothing above is edited in place, this addendum supersedes it.

- **Naive full budget.** 3.5 ms x 381.24 GB/s = **1.334 GB/token** (worst quiet cell) to 3.5 x
  381.88 = 1.336 GB/token (best) — previously priced at 3.5 x 237.79 = 0.832 GB/token to 3.5 x
  264.29 = 0.925 GB/token against the loud range. This is the naive figure (all 3.5 ms to the matvec
  stream, host and non-matvec residuals ignored); the B_w-solved figures below carry the real
  requirement.
- **The "933 GB/s needed" ratio (line 160, A=1.36).** 932.8 GB/s was "3.53-3.92x above the measured
  237.79-264.29 GB/s ceiling"; against the quiet ceiling, 932.8 / 381.24 = **2.446x**, 932.8 /
  381.88 = 2.443x — **2.44-2.45x above the quiet ceiling**, not 3.53-3.92x.
- **The A=1.49 figure (line 162).** 757.1 GB/s was "2.86-3.18x above the measured ceiling"; against
  the quiet ceiling, 757.1 / 381.24 = 1.986x, 757.1 / 381.88 = 1.983x — **1.98-1.99x above the quiet
  ceiling**.
- **The ASSUMED-A=2.18 figure (line 176).** 379.5 GB/s was "1.44-1.60x above the measured ceiling";
  against the quiet ceiling, 379.5 / 381.24 = 0.9954x, 379.5 / 381.88 = 0.9938x — **within 0.5-0.6%
  of the quiet ceiling, not above it**. This bandwidth requirement (the ASSUMED-A branch only, not
  the MEASURED-k' branch above it, which stays far above every ceiling measured) is now inside the
  device's own quiet streaming envelope.

**Reachable floor, re-derived term by term, each term naming the lever that attacks it (no lever
combined, no adjective, every number cites a ROW):**

```
reachable_floor_ms = bytes/ceiling + host_residual + non_matvec_residual
                    = 10.92         + 1.30           + 11.4
                    = 23.62 ms/token
```

- **bytes/ceiling = 10.92 ms** — ROW 302's own floor arithmetic at the best solo GPU cells (381.77
  GB/s run 1, 381.88 GB/s run 2: floor_ms 10.920/10.917). Attacked by closing the matvec kernel to
  the device's own measured streaming ceiling — the D1/D1b/D1c/D1d cards (packed-row addressing,
  s-fold, per-codec pair-dot bodies), already 65.7-73.0% there per the in-buffer figure (`discipline.md`
  ROW 296, 247.26 GB/s of 381.24).
- **host_residual = 1.30 ms** — `dispatch-census.md:27` / ROW 288's own steady-state decomposition
  (`step_wall_ms` 28.82 = `gpu_exec_ms` 27.52 + host 1.30). Attacked by the host-side dispatch/session
  cards (B2c one command-buffer owner, B3 thread-local collapse, H0 the `KernelKey` POD struct).
- **non_matvec_residual = 11.4 ms** — measured today as the gap between the full decode-program wall
  clock and what the matvec stream alone would cost at its own already-achieved in-buffer rate:
  28.3 ms (`discipline.md` ROW 298 A / ROW 300 M-R, 28.25-28.34 mean) minus 4.169 GB / 0.247 GB/ms
  (ROW 296's Relaxed in-buffer peak, 247.26 GB/s) = 28.3 - 16.9 = **11.4 ms**. Attacked by the
  dispatch-fusion cards (A5 epilogue 616->520 MEASURED, A6 prologue 520->264 MEASURED) and the
  attention-arm byte-floor card (A0-attn, ~2.5 ms above its own floor per §D.1).

23.62 ms/token is still **1.35x slower than llama.cpp's 17.53** (`discipline.md` ROW 298 D) — the
11.4 ms non-matvec/host residual, not the matvec stream, is now the larger of the two gaps to close
against llama, and dispatch fusion (already landed 616->520, ROW 294) plus the attention-arm fix are
what the ordering in §E below prices next.

---

# A. Algebra — `Loop`, a declared `Domain`, rules as pipes

```rust
// proxima-tensor/src/map.rs — pure, alloc tier, NO serde on this path.
/// Offset that is known at spec time or bound per call from the same `symbols:
/// &[u64]` slice `shape::infer` already takes. Widens `AxisIndex::offset: i32`;
/// one affine type in the crate, not two (P1 reuse-first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Offset { #[default] Static(i32), Symbol(SymbolId) }

/// `crit #4`: `Zero` is deleted (it was `Static(0)` under a derived `PartialEq`,
/// two spellings of one value, and the A1 migration gate compares `PartialEq`
/// node-for-node). Serde is EXTERNALLY tagged, never `untagged`: under
/// `untagged` both arms are a bare integer and `Symbol` is unreachable from
/// TOML — in a design whose §C thesis is that a TOML author writes the domain.
#[cfg(feature = "config")]
// #[serde(tag = "kind", content = "value")] on the spec-side mirror only.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(pub u16);            // `Extent::Symbolic(u16)` moves to this in the same card

pub type HalfSpace = AxisIndex;          // a half-space IS an affine expr + a sign convention

/// Beyond-rectangular shape of an iteration space. Empty = the full box.
/// BINDING, not advisory: a point failing any constraint is not read, not
/// accumulated, not written. That is what makes KV bytes a property of the
/// program instead of a property of whether an analysis fired.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Domain { pub constraints: SmallVec<[HalfSpace; MAX_INLINE_CONSTRAINTS]> }
```

`MAX_INLINE_CONSTRAINTS` default is **4**, not AB's 2 (`crit #7`: causal + sliding window is exactly 2,
so 2 sits on the spill boundary at the default), and its overflow policy is **bind errors**, matching
`MAX_INLINE_STAGES`'s decline-rather-than-spill — one policy for one hazard on one path.

```rust
// proxima-tensor/src/bind.rs
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    pub reduced_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
    pub body: ComposedBody,
    pub fold: Option<Fold>,              // None = a map over the live axes (this is B2's `epilogue`)
    pub domain: BoundDomain,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct Fold { pub op: ScalarOp, pub init: ReduceInit, pub keep: Keep }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum BoundOffset { Static(i64), Symbol(SymbolId) }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundHalfSpace { pub coefficients: SmallVec<[i64; MAX_INLINE_RANK]>, pub offset: BoundOffset }
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoundDomain { pub constraints: SmallVec<[BoundHalfSpace; MAX_INLINE_CONSTRAINTS]> }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum StepArg { Operand(u16), Step(u16), Stage(u16) }

pub enum BoundOpKind {
    Loop { stages: SmallVec<[Stage; MAX_INLINE_STAGES]>, operands: BoundOperands,
           output_axes: SmallVec<[u16; MAX_INLINE_RANK]>, out_layout: Layout,
           out_scatter: Option<Lookup> },
    Iota,
    Constant { value: f32 },
}
```

Flat side table, backwards-only, exactly as `bind.rs:160-162` requires ("a plain index into a side
table, never a `Box<dyn>` recursive tree"). The live cache length is `BoundOffset::Symbol`, never an
operand: `map.rs:99-107` states no backend plumbs integer buffers, so B2's `BandBound::Runtime { node }`
is rejected. Two map extensions, both B2's and both required: `IndexPattern::compose` (substitution —
nothing weaker pushes a non-identity map through a held node) and strided extent resolution in
`unify_iteration_space` (`shape.rs:224-244`), unit projection always winning so existing programs bind
byte-identically. `is_identity_projection` (`bind.rs:1156-1164`) is the second gate and is relaxed in
the same card (judge-r2-2's grounded check: two distinct gates, not one).

**Rules are pipes; capability is composition (decision 2).**

```rust
pub struct FusePrologue; pub struct FuseEpilogue; pub struct FuseBroadcastEpilogue; pub struct ShareAxis;

impl Pipe for FuseEpilogue {                     // primitives.rs:91-102: no lifetimes, call(&self, ..), !Send root
    type In = BoundProgram; type Out = BoundProgram; type Err = TensorError;
    fn call(&self, program: BoundProgram) -> impl Future<Output = Result<BoundProgram, TensorError>>
    { async move { fuse_epilogue(program) } }
}
// metal:  FusePrologue.and_then(FuseEpilogue).and_then(FuseBroadcastEpilogue).and_then(ShareAxis)
// wgpu:   FusePrologue.and_then(FuseEpilogue)                      // declines what it cannot render
```

No `FusionRules`, no bitfield, no bool. `fuse_cached_attention: bool` (`bind.rs:2635-2640`) is deleted
and **nothing replaces it**. The plan cache instead keys on a `RuleSetId(pub u32)` the driver supplies —
a newtype id (P11's permitted class), because `ShareAxis` reassociates floating-point and two rule
compositions must not share a cache entry. This is what makes `PlanKey` derive `Hash + Ord` legally
(`crit #4`: it could not over `FusionRules`).

**The cross-backend numeric contract (`crit #12`), which AB owed and did not state:** a rule composition
is part of the program's identity. Parity is asserted **within** a composition (fused vs unfused under
the *same* `RuleSetId`, `assert_close 1e-6` on the CPU oracle), never across two backends running two
compositions. The `X` gate compares each backend to the CPU evaluator *running that backend's own
composition*.

---

# B. Orchestration — one FSM, POD pipe boundaries, one device edge per engine

`Pipe` at `proxima-primitives/src/pipe/primitives.rs:91-102` (opened): `type In; type Out; type Err:
Debug + 'static; fn call(&self, input: Self::In) -> impl Future<..>`. No lifetime parameters, no GATs,
no `Send`. Therefore no borrowed view is ever an `In` or an `Out`: `LogitsView` stays inside the
`InFlight -> Sampled` inherent transition, which is free to borrow.

```rust
pub struct ReadyStep<'p>{ plan:&'p Plan, cursor: Cursor }
pub struct EncodedStep<'p>{ plan:&'p Plan, cursor: Cursor, commands: CommandRange }
pub struct InFlightStep<'p>{ plan:&'p Plan, cursor: Cursor, ticket: SubmitTicket }
pub struct SampledStep<'p>{ plan:&'p Plan, run: AcceptedRun, counters: StepCounters }

#[must_use]
pub enum Step<'p> { Ready(ReadyStep<'p>), Encoded(EncodedStep<'p>), InFlight(InFlightStep<'p>), Sampled(SampledStep<'p>) }

/// B2's shape, taken over AB's `Cursor.accepted: u16` + `StepOutcome.accepted:
/// ArrayVec` + `halt: Option<Halt>` (`crit #15`: three encodings of one
/// quantity). The rejected suffix is kept because destroying it is what forces
/// a downstream reconstruction of what the verifier already knew.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedRun {
    pub tokens: ArrayVec<TokenId, MAX_ACCEPTED>,
    pub rejected: ArrayVec<TokenId, MAX_DRAFT>,
    pub cursor: Cursor,
    pub halt: Option<Halt>,
}
```

`AcceptedRun` is `Clone`, **not `Copy`** — `crit #4` is correct that `arrayvec 0.7.6` has a `Drop` impl
and cannot be `Copy`. So the `Session` pipe's boundary contract is restated honestly: **`Out` is
move-only and allocation-free**, not "POD". `StepCounters` is `Copy` and exists unconditionally (its
*emission* is `#[cfg(feature = "instrument")]`, its shape is not — `crit #4`: a public `Out` may not
change shape with a feature flag).

Pipes on the path, and only these: `Draft: Pipe<In = TokenWindow, Out = DraftSpan>` (a caller can swap
n-gram for a draft model), `Submit: Pipe<In = SubmitRequest, Out = SubmitTicket>` (**one device edge per
engine**, §B.4), `Session: Pipe<In = StepInputs, Out = AcceptedRun>` (one call = one pass). `crit #11`
is accepted: `LoadedModel::call` (`proxima-model-interop/src/generate.rs:1542-1556`, `In = (String,
usize)`, `Out = (Vec<u32>, String, bool)`) is a **different granularity and a lossy payload**; AB's
claim that `Session` "is the shape it already has" is withdrawn, and CARD H4 re-faces it onto
`AcceptedRun` so the bare `bool` and the per-call `String`/`Vec` stop existing.

```rust
/// Pure, alloc tier. Takes NO codecs (decision 5): `QuantizedBlock<'a>` is
/// `#[cfg(feature = "std")]` (`cpu.rs:3091`, `lib.rs:193-194`) and borrowed, and
/// `BlockCodec` does not exist (`grep -rn BlockCodec` = 0). Codec-dependent
/// kernel choice belongs to the emitter, so it moves to `omega::schedule`.
#[must_use] pub fn plan(program: &[Op], symbols: &[u64], outputs: &[NodeId], rules: RuleSetId)
    -> Result<Plan, TensorError>;

/// omega, std. Already holds `QuantizedBlock` (`metal.rs:450-456` maps it to
/// `PackedCodec`, `msl.rs:788`), so no crate-graph inversion and no new enum.
#[must_use] pub fn schedule(plan: &Plan, blocks: &[Option<QuantizedBlock<'_>>]) -> Result<Schedule, EmitError>;

#[derive(Debug, Clone, PartialEq)]
pub struct BarrierPoint {                 // decision 3, from B2 §B.3
    pub before: u32,
    pub resources: SmallVec<[SlotId; MAX_BARRIER_SLOTS]>,   // one name; `BarrierSet`/`BarrierRange` deleted
    pub cause: Hazard,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hazard {
    ReadAfterWrite, WriteAfterWrite, WriteAfterRead,
    /// <!-- r2-fix (owner): cross-engine dependency, §B.4. Same three hazard
    /// kinds, but the producer and consumer sit on different engines, so the
    /// join is a command-buffer completion rather than an intra-encoder
    /// barrier. Kept as a variant of the SAME enum: a fourth spelling of
    /// "barrier" is what decision 3 deleted.
    CrossEngine { producer: Engine, consumer: Engine, inner: HazardKind },
}
#[must_use] pub fn schedule_barriers(ops: &[BoundOp], residency: &ResidencyPlan) -> SmallVec<[BarrierPoint; 32]>;
```

**`KernelKey` is a POD struct, not a `String` (`crit #5`, and it is a correctness fix as well as an
allocation fix).** `omega/src/msl.rs:930` `kernel_cache_key(..) -> Result<String, EmitError>` is called
at `omega/src/metal.rs:3875` inside `encode_op` once per op per step — 616 `String`s per token on the
plan-**hit** path (the comment at `metal.rs:3871-3874` says so itself), which is precisely the path the
zero-allocation gate asserts. The same key **omits the reduce extents** while `cooperative_reduce_width`
bakes an extent-derived width into the source, so `pipeline_for` can serve a wrong kernel from cache.
One change closes both: a POD `KernelKey` carrying the extents (fields in §B.2). `KernelFamily` is
returned by the emitter (`Emitted { source, family, key }`) instead of recovered by substring-grepping
MSL (`metal.rs:1734` `classify_kind`).

## B.2 Support types — field lists, or deleted

<!-- r2-fix: the panel's "12 types in signatures with no field lists" (judge-r2-1 missing-steps 2,
     judge-r2-2 missing-steps 2). Each gets the two binary questions. Seven are deleted; six get
     field lists; two AB spellings collapse into one shipped name. -->

**Kept, with fields.** Each answers question 2 with something a caller can do that they could not:

```rust
/// The half-open span of encoded commands one step owns. A caller can ask
/// "which commands are mine" — that is what `BarrierPoint.before` indexes and
/// what `poll_complete` reports progress against. A bare `u32` count cannot
/// answer it once two steps are in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandRange { pub first: u32, pub count: u32 }

/// FNV-1a over an op's RESOLVED reduce extents. Exists because
/// `cooperative_reduce_width` bakes an extent-derived width into the kernel
/// source while `kernel_cache_key` (msl.rs:930-981) omits extents entirely —
/// a caller can now distinguish two ops that differ only in reduce extent,
/// which today silently share a pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct ExtentDigest(pub u64);

/// Four bits per operand slot, up to 8 operands: `Option<PackedCodec>` per
/// operand packed into one `Copy + Hash` word. A caller can key a cache on the
/// codec set without allocating; a `SmallVec<[Option<PackedCodec>; 8]>` is
/// neither `Copy` nor `Hash`-cheap and cannot sit in `KernelKey`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct CodecMask(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelKey { pub family: KernelFamily, pub codecs: CodecMask,
                       pub extents: ExtentDigest, pub engine: Engine, pub rules: RuleSetId }

/// What a drafter proposes AND with what probability. `proposal.len() ==
/// tokens.len()` always — a deterministic n-gram drafter reports 1.0, so there
/// is no `Option` and no empty case. Returning bare tokens destroys q(x), and
/// speculative sampling at temperature > 0 cannot be stated without it; the
/// downstream would reconstruct it or silently degrade to argmax verification.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftSpan { pub tokens: ArrayVec<TokenId, MAX_DRAFT>,
                       pub proposal: ArrayVec<f32, MAX_DRAFT> }

/// Owned and fixed-cap because `Pipe` has NO lifetime parameters
/// (primitives.rs:91-102) — the constraint, not a preference. `cursor` is what
/// anchors the drafted span to an absolute position.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenWindow { pub tokens: ArrayVec<TokenId, MAX_DRAFT_CONTEXT>, pub cursor: Cursor }

/// One name for what AB spelled `ResidentSet` and `schedule_barriers` needs:
/// which node lives in which slot, how big, and whether it survives the step.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidencyPlan { pub slots: SmallVec<[SlotBinding; MAX_PLAN_SLOTS]> }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotBinding { pub slot: SlotId, pub node: NodeId, pub bytes: u32, pub kind: SlotKind }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind { Weight, KeyValue, Activation, Uniform, Output }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct Cursor { pub position: u32, pub capacity: u32 }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum Halt { EndOfText, MaxTokens, StopToken(TokenId) }
#[derive(Debug, Clone, Copy, PartialEq)] pub struct StepCounters { pub dispatches: u32, pub bytes_read: u64,
    pub gpu_exec_ns: u64, pub host_ns: u64, pub cpu_engine_ns: u64, pub plan_hit: bool }
#[derive(Debug, Clone, PartialEq)] pub struct StepInputs { pub window: TokenWindow, pub draft: DraftSpan }
#[derive(Debug, Clone, PartialEq)] pub struct SubmitRequest { pub commands: CommandRange, pub engine: Engine }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct SubmitTicket(pub u64);
```

**Deleted, each on question 2 — the call site is the identical line without it:**

| type | why it dies |
|---|---|
| `Wiring` / `StackWiring` (AB) | two spellings of `&[NodeId]`. No call site changes. |
| `Bindings` (AB) | the shipped bind path already takes named `&[(&str, &[f32])]`; a newtype over it is a relocation. |
| `Grid` (AB) | `omega` already computes dispatch geometry in `msl.rs:996 kernel_dispatch_shape`; a second holder for the same three numbers is the `BarrierSet`/`BarrierRange` defect again. |
| `UniformPatch` (AB) | `metal.rs:2286 pack_uniforms(&BoundOp) -> Vec<u8>` and `metal.rs:3748 PlanUniforms` own this; the patch type moves bytes from one owner to another and back. |
| `Completion` (AB) | replaced by `poll_complete(&self, ticket: SubmitTicket, cx: &mut Context<'_>) -> Poll<Result<StepCounters, EmitError>>` — P20's reactor surface, which is what a cancellable caller actually needs. |
| `ResidentSet` (AB) | collapsed into `ResidencyPlan` above. One concept, one name. |
| `Budget` (AB) | a stop policy. `AcceptedRun.halt` states why a run ended; the iteration bound is the caller's `while`. Nothing a caller could not already write. |

## B.4 Placement: two engines, a driver that is a property of one of them, a rule that is a pipe

**KILLED 2026-09-05 by ROW 302** (`proxima-tensor/docs/discipline.md`): concurrent CPU+GPU streaming
measured 291.85 GB/s, GPU-alone measured 381.88 GB/s — concurrent is 23.6% below solo-GPU, not
above it, and card D0c's own gate ("this card kills or admits §B.4's placement work outright")
answers NO. The mixed-engine placement work below does not land on this host; the section is left
as written because the algebra (`Engine::{Cpu, Gpu}`, `ScheduledOp.engine`, `BandwidthTable`) is
still correct, it simply has no bandwidth-bound placement case left to serve.

<!-- r2-fix (owner, verbatim): "omega should support cpu, gpu and _mixed_ backends" and
     "how is there any more than 2 backends?" — `Backend` collapses to `Engine::{Cpu, Gpu}`. -->

**There are two engines, and `Backend::Mixed` is not a third.** `omega/src/backend.rs:78-91` enumerates
seven variants — `Cpu`, `Metal`, `Wgpu`, `Vulkan`, `Cuda`, `Npu`, `Ane` — and only **three** of them
execute: `backend.rs:265-375` gives `Cpu`, `Metal` and `Wgpu` real arms and returns
`BackendError::NotImplemented` for the other four (`lowering-audit.md:51`, row 37: "four of seven
variants are name reservations with no arm"). Of the three that execute, `Metal` and `Wgpu` are *the
same engine reached through two drivers* — `omega/Cargo.toml`'s own words for `wgpu` are "one
abstraction layer over `Backend::Metal`: same `BoundOp` descriptor, same emit-then-drive split, a WGSL
emitter and a `wgpu::Device` instead of an MSL emitter and an `objc2-metal` device" (`backend.rs:81-86`).
So the enum spends seven variants stating two facts:

```rust
/// omega. Where an op runs. Two variants, because there are two places:
/// a CPU core and a GPU. `Metal`/`Wgpu`/`Vulkan`/`Cuda` are DRIVERS of the
/// Gpu engine, not peers of the Cpu engine, and `Npu`/`Ane` were name
/// reservations with no lowering — deleted, not carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine { Cpu, Gpu }

/// Which driver renders the Gpu engine on this target. Resolved ONCE at
/// plan/schedule time from the compiled features and `target_os`, never
/// carried per op — every Gpu op on a host uses the same driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuDriver { Metal, Wgpu }

impl GpuDriver {
    /// The resolution `backend.rs:279-300`'s `cfg!` cascade already performs,
    /// hoisted to one place instead of once per `match backend` arm.
    #[must_use] pub const fn for_target() -> Option<Self> {
        if cfg!(all(feature = "metal", target_os = "macos")) { Some(Self::Metal) }
        else if cfg!(feature = "wgpu-backend") { Some(Self::Wgpu) }
        else { None }
    }
}
```

"Mixed" is not a third engine either — no emitter emits for it and no device implements it. It is a
**placement over ops**, so it is a field on the scheduled op, not a variant beside the things it mixes.
Adding `Backend::Mixed` would force every `match backend` to answer "which one, for this op?" at a point
where the per-op information (codec, bytes, extents) has already been discarded — the rich-to-poor
boundary this design keeps looking for.

**What a caller can do that they could not — the call site both ways.** Today:

```rust
// before: one engine for the whole program, named as a string-parsed process global
let plan = omega::plan_named(&program, &symbols, &named, &outputs, Backend::Metal)?;
// running two layers on the CPU is not expressible at all: there is no second call to make,
// and `Backend::Cpu` here moves the ENTIRE program, all 616 ops.
```

After:

```rust
// after: one plan, one schedule, placement as data
let plan     = omega::plan(&program, &symbols, &outputs, RuleSetId(0))?;
let schedule = omega::schedule(&plan, &blocks)?;                       // every op on Engine::Gpu
let hybrid   = PlaceLastLayers { layers: 2, engine: Engine::Cpu }
                 .and_then(PlaceByBandwidth { table: &measured })
                 .call(schedule).await?;                                // 38 ops moved, 578 unchanged
```

The two lines are not the same line: the second expresses a program *split across both engines in one
pass*, which the first cannot express at any argument value. That is question 2 answered, and it is why
`ScheduledOp` earns a field rather than `Backend` earning a variant.

**What is actually new: one field and one two-variant enum; five variants are deleted.**

```rust
/// omega, std. One scheduled op = one bound op + where it runs.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledOp { pub op: BoundOp, pub engine: Engine, pub kernel: KernelKey }

#[derive(Debug, Clone, PartialEq)]
pub struct Schedule { pub ops: Vec<ScheduledOp>, pub barriers: SmallVec<[BarrierPoint; 32]>,
                      pub residency: ResidencyPlan }
```

`schedule()` assigns a default engine structurally (the engine whose emitter can render the op at all);
a **placement rule is a pipe**, composed exactly like a fusion rule, and it passes the same
discriminator — an operator can supply "last 4 layers on CPU" without the library shipping that policy:

```rust
pub struct PlaceByCodec;                                  // Q6_K per-element bodies are CPU-cheap, GPU-poor
pub struct PlaceLastLayers { pub layers: u16, pub engine: Engine }
pub struct PlaceByBandwidth<'t> { pub table: &'t BandwidthTable }   // D0's measured per-engine GB/s

impl Pipe for PlaceLastLayers {
    type In = Schedule; type Out = Schedule; type Err = EmitError;
    fn call(&self, schedule: Schedule) -> impl Future<Output = Result<Schedule, EmitError>>
    { async move { place_last_layers(schedule, self.layers, self.engine) } }
}
// hybrid: PlaceByCodec.and_then(PlaceLastLayers { layers: 4, engine: Engine::Cpu })
//                     .and_then(PlaceByBandwidth { table: &measured })
// gpu-only today: the identity composition — no rule, no change, byte-identical plans.

/// Per-engine achieved streaming bandwidth, MEASURED by D0's arms, never
/// assumed. A rule reads it; nothing else does.
#[derive(Debug, Clone, PartialEq)]
pub struct BandwidthTable { pub rows: SmallVec<[BandwidthRow; 8]> }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandwidthRow { pub engine: Engine, pub codec: Option<PackedCodec>,
                          pub gb_per_s: f32, pub cov: f32, pub samples: u16 }
```

`BandwidthTable`/`BandwidthRow` answer question 2: a caller can supply a table measured on *their* host
and get a different placement with no recompile — that is P4's config-as-composition, and it is the
whole point of the hybrid arm. `ScheduledOp` answers it too: a caller can read *which* ops went where,
which a process-wide engine argument destroys.

**Buffers: one mapping, no copies.** Apple silicon is unified memory and `omega/src/metal.rs:53` already
states the driver creates buffers with `MTLResourceOptions::StorageModeShared` **only** — the same
pages the CPU sink reads and writes. So a hybrid step performs **zero copies**; what it needs is
ordering, not transfer:

- **CPU writes, then GPU reads** — the CPU sink's stores must be complete before the command buffer that
  reads them is `commit()`ted. Encoding happens after the join, so the ordering is the join itself; no
  `didModifyRange` is required (that is `StorageModeManaged`, which this driver never uses).
- **GPU writes, then CPU reads** — `commit()` then `waitUntilCompleted()` on that command buffer before
  the CPU sink touches the slot (`metal.rs:680` is the shipped call). This is the only synchronization
  primitive the cross-engine join needs, and it is already in the driver.
- Both directions are recorded as `Hazard::CrossEngine { producer, consumer, inner }` at the
  `BarrierPoint` whose `before` index is the first op on the consuming engine, so the barrier schedule
  is one list across engines rather than one list per engine.

**How the driver runs it.** The CPU sink runs on `proxima_tensor::cpu::matmul_worker_count()` threads
(`cpu.rs:12933-12939`, `PROXIMA_MATMUL_WORKERS` via `OnceLock`) **concurrently with** the Metal sink's
command buffer, and joins only at `CrossEngine` points. `MetalDriver` stays the `!Send` owner of device
handles; the CPU sink owns its own slice of the residency plan. Two sinks, one schedule, one barrier
list — no `Arc<Mutex<Schedule>>` anywhere (P21 rung 1: the split is by *ownership of slots*, and
`ResidencyPlan` is what states that split).

**Not proven, and carded rather than assumed:** whether concurrent CPU+GPU streaming *exceeds* the
GPU-alone ceiling at all. Both engines share one memory controller; D0's measured 237.79-264.29 GB/s is
a whole-chip figure, so the hybrid arm may be a zero-sum split. Card D0c measures it before any
placement rule is written (§E).

---

# C. Generic model programs

<!-- r2-fix: the panel's largest hole (judge-r2-1/-2/-3 all scored missing-steps 2 on this section
     being asserted rather than written). Written here from the shipped code, taking B2 §C's
     completeness and re-deciding two of its calls against AB2's own shape. -->

## C.1 Production decode is std-gated *by the program builders themselves*

`proxima-tensor/src/lib.rs:219-220` is `#[cfg(feature = "config")] pub mod spec;` and
`proxima-tensor/Cargo.toml:37` is `config = ["std", "dep:bon", "dep:conflaguration", "dep:serde",
"smallvec/serde"]`. So **every** model-program builder — all 18 `#[allow(clippy::too_many_arguments)]`
functions in `spec.rs` (`grep -c too_many_arguments proxima-tensor/src/spec.rs` = 18; the list runs
`append_mistral_layer:898`, `append_moe_ffn:1407`, `append_mistral_moe_layer:1705`,
`mistral_forward_program:2067`, `append_mistral_cached_layer:2378`,
`append_qwen35_dense_attention_layer:2957`, `append_mistral_single_range_cached_layer:3517`,
`append_mistral_cached_moe_layer:4167`, `append_lfm2_conv_mixer:4920`,
`append_qwen35_delta_net_step:5059`, `append_qwen35_conv_branch:5490`, `append_qwen35_ssm_mixer:5636`,
`append_attention_mixer:5959`, `lfm2_forward_program_with_experts:6292`,
`mistral_cached_forward_program:6742`, `qwen3_cached_forward_program:6775`,
`mistral_cached_forward_program_with_experts:6811`, `qwen35_forward_program:7184`) — sits behind
`std + serde + bon + conflaguration`. `proxima-model-interop/Cargo.toml:9-18` says so outright:
`std = [.., "proxima-tensor/config"]`, with the comment "`generate.rs` is a LIB module ... and imports
`proxima_tensor::spec`, which is `config`-gated."

That is the finding: **the alloc tier cannot construct a model program at all**, and production drags a
TOML deserializer in to build one it never deserializes (`toml::from_str` appears in `spec.rs` only
under `#[cfg(test)]`). The tier split is therefore not a preference — it is a repair:

```rust
/// std + `config`. Already shipped: `spec.rs:361-364` is
/// `impl TryFrom<&ProgramSpec> for Vec<Op>`, and `ProgramSpec` (spec.rs:69-77)
/// already derives `Builder + Deserialize + Serialize + Settings`. No loader
/// type is added; `LoadSpec: Pipe` was rejected in decision 4 because
/// `LoadSpec.call(src).await?` and `Vec::<Op>::try_from(&spec)?` are the same
/// line.
#[cfg(feature = "config")]
pub fn program_from_toml(text: &str) -> Result<Vec<Op>, TensorError>;

/// alloc tier, no serde, no bon. The program is DATA that arrived from
/// somewhere — a `build.rs` bake, a mapped file, a caller's Vec.
pub fn plan(program: &[Op], symbols: &[u64], outputs: &[NodeId], rules: RuleSetId)
    -> Result<Plan, TensorError>;
```

Below std the same TOML resolves at **build time** into `&'static [Op]` — the conflaguration tier-2
pattern (`build.rs` IS the config surface), the mechanism `proxima-telemetry`'s `sized` module and
`proxima-tensor/build.rs:269-291` already use. One source of truth, two tiers, and the alloc-tier bind
path never sees `serde`. **CARD C1 moves the layer builders out of `spec.rs`'s `config` gate**, because
a program builder is not a config face; only `ProgramSpec` is.

## C.2 The positional types, and the leaf that stops existing

```rust
/// `CachedLayerRoots = (NodeId, NodeId, NodeId)` (spec.rs:2333), consumed at
/// 15+ sites (spec.rs:2404, 3541, 3943, 3990, 4194, 6750, 6783, 6822, 6872):
/// three same-typed positions, so any two can be swapped silently and the
/// program still binds. `Qwen35DenseAttentionRoots` (spec.rs:2344) is the same
/// defect one wider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedLayerRoots { pub hidden: NodeId, pub key: NodeId, pub value: NodeId }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitRotaryRoots { pub first: NodeId, pub second: NodeId, pub pass: NodeId, pub value: NodeId }

/// `append_mistral_single_range_cached_layer` (spec.rs:3516-3539) carries
/// `#[allow(clippy::too_many_arguments)]` over 23 parameters, **21 of them
/// bare `NodeId`** — x, inv_dim, eps, ones, inv_sqrt_head_dim, cos_new,
/// sin_new, group_ones, is_future, attn_norm_weight, ffn_norm_weight, wq, wk,
/// wv, wo, w_gate, w_up, w_down, k_even_cache, k_odd_cache, v_cache. Any two
/// swapped is a silent wrong program. One struct + a `bon` builder (P4, both
/// surfaces first-class) and the `allow` disappears rather than being
/// justified. 18 such functions exist; C2 converts them one per card.
#[derive(Debug, Clone, bon::Builder)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerBindings {
    pub input: NodeId,
    pub attn_norm: NodeId, pub ffn_norm: NodeId,
    pub q: NodeId, pub k: NodeId, pub v: NodeId, pub o: NodeId,
    pub gate: NodeId, pub up: NodeId, pub down: NodeId,
    pub cos: NodeId, pub sin: NodeId,
    pub cache: CachedLayerRoots,
    pub group: u32,
}
```

Note the `cfg_attr`: `LayerBindings` is a *spec-side* type, so the serde derive is legal there and never
reaches the alloc-tier bind path. If this struct were needed at the alloc tier the derive would have to
go — the constraint doing its job.

**`cached_len` stops being a leaf.** <!-- r2-fix: this is where AB2 and B2 genuinely differ, and the
synthesis takes AB2's shape. B2 moves the leaf from `DType::Float32` to `DType::Int32`. AB2 deletes it.
--> Today `spec.rs:3987` builds `input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len")`, a
rank-0 f32 input; `bind.rs:2435-2449` then carries it as `CachedAttention`'s **ninth operand**
(`bind.rs:230` documents it as exactly that) purely so the band can read the true value rather than a
shape difference the KV bucket widens. But `spec.rs:6724` shows the caller **already** passes it:
`symbols = [new_positions, cached_len]`. The value exists twice, and `exact_merged_causal_mask_cached_len`
(`bind.rs:2016-2034`) is machinery that recovers one copy from the graph.

Under §A the domain's offset is `BoundOffset::Symbol(SymbolId)`, bound from the same `symbols: &[u64]`
slice `shape::infer` already takes. So the f32 leaf, the ninth operand, and the recovery peephole all
delete together — three mechanisms replaced by the one the caller was already using. This is also why
B2's `DType::Int32` change is not taken: there is no leaf left to retype.

**KV bucketing is a serving knob, not a build constant.** <!-- r2-fix: `KV_BUCKET_TOKENS` was a build
const in `proxima-tensor/src/sized.rs:326`, and commit 283002e (branch
`refactor/kv-bucket-policy-owner`) relocated it to `proxima-model-interop/src/sized.rs` — correct
owner, still the wrong tier. --> `generate.rs:710-726 kv_extent(merged_len, capacity)` rounds the KV
extent to a bucket so the Metal plan-cache key stays constant across a bucket of decode steps. That is a
**per-serving trade** (padding vs plan-cache misses) whose right value depends on context length and
prompt mix, so it belongs on `ServingConfig` beside `-c`, `-np`, `-ctk` (`serving.rs:69-98`), not baked
at build time:

```rust
pub struct ServingConfig<'model> {
    /* .. shipped fields, serving.rs:69-.. */
    /// Plan-cache key bucket for the placed-KV decode loop: the growing KV
    /// extent rounds up to this many tokens so the key is stable across a
    /// bucket of steps. `1` disables bucketing (bind the true `merged_len`).
    /// Default seeded from `sized::KV_BUCKET_TOKENS` — the conflaguration
    /// bridge: the build const IS the config below std, and seeds the runtime
    /// default at std, never typed twice.
    pub kv_bucket_tokens: u32,
}
fn default_kv_bucket_tokens() -> u32 { crate::sized::KV_BUCKET_TOKENS as u32 }
```

The 0-ULP argument is unchanged and already proven for **any** bucket size — `spec.rs:11529-11548`
`cpu_mask_zero_ulp` sweeps `bucket_tokens` as a *parameter*, which is exactly why a runtime value is
admissible. Two consequences that must land in the same card: the bucket enters `PlanKey` (two servings
with different buckets may not share a plan entry), and `sized::KV_BUCKET_TOKENS` keeps its
`kv-capacity-bucket` gate as the tier-3 floor value only.

## C.3 What a new architecture needs to add

| new architecture needs | today | after |
|---|---|---|
| a new layer shape (different norm, different FFN) | Rust: one of 18 `append_*` fns in `spec.rs`, each with `too_many_arguments` | **data only** — a TOML layer + `LayerBindings` |
| GQA with a different group map | Rust: a ninth stride tuple at `bind.rs:2465-2474` (eight literal tuples today) | **data only** — `Layout` carries strides; the rule never compares them to a literal |
| sliding-window attention | Rust: new fields on `CachedAttention` | **data only** — a second `HalfSpace` in the same `Domain` (`MAX_INLINE_CONSTRAINTS = 4`) |
| causal + window + ALiBi slope bound jointly | not expressible | **data only** — three half-spaces, still inline |
| a per-expert MoE membership constraint | Rust: `append_moe_ffn:1407` | **data only** — a half-space on the expert axis + `out_scatter` |
| an SSM/delta-net mixer (`append_qwen35_ssm_mixer:5636`) | Rust | **Rust** — a sequential scan is `Keep::Scan` over a `Loop`, but the recurrence's operand-to-operand carry is not a half-space; named honestly, not claimed as data |
| a genuinely new scalar primitive | Rust: `ScalarOp` variant | Rust — and `op.rs:52-57` says this set stays closed, so the answer is "desugar it" |
| a new weight codec (Q3_K) | Rust: `PackedCodec` + unpack body | **Rust** — one table row + one MSL body, plus a `CodecLayout` row (§E card D1d) |
| a new backend | Rust: a driver | **Rust** — but it composes the fusion and placement rules it can render (§A, §B.4) instead of passing a bool |
| running some layers on CPU | not expressible (the engine is a per-plan argument) | **data only** — a placement rule + a `BandwidthTable` (§B.4) |

Four Rust-required rows remain and each is honest: a codec is machine-level, a backend is a driver, a
scalar primitive is closed by decree, and a recurrence is not a polyhedron.

---

# E. Cards — 30-minute slices, one gated deliverable each, ordered by measured value

Standing battery: **P** parity <=1e-4 vs `proxima_tensor::cpu::evaluate` on the real program;
**B** 20 runs byte-identical logits; **T** text identical (never on a card that reassociates FP —
there the gate is `assert_close 1e-6` + EM-at-64 >= 0.99); **M** ms/token and `gpu_exec_ms` not worse
beyond CoV (matvec 0.7%, attn **9.9%** — the attention CoV binds any attention card); **A** allocation
counter == 0 with the **step count asserted** (zero work and successful work emit the same signal);
**G** bytes/token from the driver's own counters against §D.3's prediction, >5% disagreement halts;
**X** the measurement is the deliverable, N and CoV reported.

Ordering rationale, stated because AB's is void (`crit #2`): the gap is **11.4 ms GPU-side**, of which
matvec bandwidth is the largest single term (23.24 ms at **65.7-73.0% of the measured ceiling**) and the
attention arm is ~2.5 ms above its own byte floor. Host is 1.30 ms total, so no host card precedes a GPU
card except the one that is also a *correctness* fix (H0, the wrong-kernel-served hole).

<!-- r2-fix: every card below is now a row with ONE gated deliverable and a 30-minute scope; the
     "then, in order:" prose paragraph AB2 ended with is expanded into rows 8-31. -->

| # | card (30 min) | one deliverable | gate |
|---|---|---|---|
| **D0a** | device streaming ceiling, **quiet-box repeat** — branch `perf/device-streaming-ceiling`, harness `omega/tests/device_streaming_ceiling.rs` exists (a8d9c06); run it with `pgrep -c 'cargo|rustc|nextest' == 0` asserted in the log | the best cell (`nocopy_resident`, wide, 4 GB) re-measured under 5% CoV | **X**: GB/s + CoV + N. 35 of 36 landed cells exceed the 5% line (`discipline.md:21208-21252`); if the quiet box still cannot clear it, the **range 237.79-264.29 stands as a range** and §D.4's ceiling columns stay a range, not a point |
| **D0b** | ceiling at the **Q4_K superblock stride** — add one arm reading 144 B / 256 elt strided (`msl.rs:439`, `:444`) beside the shipped `float4`/`uint4` arms | GB/s for the access shape decode actually uses | **X**; if it is materially below the `float4` arm, the matvec shortfall is partly *access shape*, and D1b's arms are re-ordered on it |
| **D0c** | **hybrid ceiling** — branch `test/cpu-gpu-streaming-ceiling` (today at 5d801d7, no hybrid arm yet): add a CPU-only streaming arm (`PROXIMA_MATMUL_WORKERS` threads over the same mapped GGUF) and a **concurrent CPU+GPU** arm | aggregate GB/s of the concurrent arm vs the GPU-alone ceiling | **X**. **This card kills or admits §B.4's placement work outright**: if concurrent < GPU-alone x 1.05, both engines are contending for one memory controller, hybrid placement is zero-sum for bandwidth-bound ops, and the only surviving placement case is compute-bound or codec-mismatched ops. **KILLED 2026-09-05 by ROW 302: 291.85 GB/s concurrent < 381.88 GB/s GPU-alone (23.6% below, not the >=1.05x admission threshold) — hybrid placement does not clear this card** |
| **D1** | matvec roofline ladder — branch `perf/matvec-roofline-ladder` (aa99186 hoists packed-row addressing out of the block loop; 5053e59 loads Q4_K/Q5_K weight words as native `u16`) | achieved GB/s per landed change vs D0's ceiling, at 1/2/4/8 rows per activation | **X**: GB/s per arm + CoV; a mechanism named for any arm below 0.85x the ceiling (a measurement without a mechanism is rung 1, not a result) |
| **D1b** | packed-row addressing arms — branch `perf/packed-row-addressing` (a3eef89 adds a Q6_K pair-dot body on top of D1's two commits) | ns/element and GB/s per addressing arm | **X**: which arm, if any, moves 173.6 toward the ceiling; **M** unchanged on the production program |
| **D1c** | **s-fold reshaped into the ONE packed-row body** — branch `perf/packed-row-multi-activation` today adds `push_packed_row_multi_activation_body` as a SECOND body behind `metal-packed-row-multi-activation` (`omega/Cargo.toml:315-330`). Fold it into `push_packed_row_blocked_body` with `s = 1` as the degenerate case; the feature disappears, `sized::PACKED_ROW_ACTIVATION_GROUP` (`omega/omega-runtime.toml:138-158`, group = 8) stays as the P12 build const | one body, no flag; `grep -c 'metal-packed-row-multi-activation' omega/` == 0 | **B** byte-identical at s = 1 (the degenerate case must reproduce today's kernel exactly), **P** at s = 4, **M**. Rationale is this design's own RISC lint: two bodies for one traversal is the `Elementwise`/`Reduce` split one level down |
| **D1d** | **Q6_K pair-dot becomes a codec-structural choice, not a feature** — `metal-q6k-pair-dot` (a3eef89, `omega/Cargo.toml:269-287`) sits beside `metal-q5k-pair-dot` (`Cargo.toml:268`), both default-off, and `msl.rs:3274-3277` reads `cfg!(feature = ..)` to pick a body. Whether a codec *has* a paired body is a property of its block layout: Q4_K's 144 B block yields a nibble out of an already-loaded word, Q5_K needs a `qh` bit from another byte (`msl.rs:3454-3466`), Q6_K's 210 B block is not a multiple of 4 so its wide loads are `ushort`, not `ulong` (`msl.rs:3490-3499`). Replace all three `cfg!`s with a `const fn CodecLayout::pair_width(codec) -> Option<PairWidth>` the emitter reads | `grep -c 'cfg!(feature = "metal-q.k-pair-dot")' omega/src/msl.rs` == 0; the body is selected from the codec | **B** per codec against the feature-on build; **M**. A default-off flag means the production build never runs the fast body and `KernelKey` cannot state which body it wants — the same class as `classify_kind` grepping MSL |
| **A0-attn** | attention arm measured against its own byte floor (134.2 MB / 3.3 ms = 40.7 GB/s vs matvec 173.6) before any `Loop` work | measured loop trips vs `t`, from the emitted MSL bound at `msl.rs:2586` (`cached_key_rows = new_key_rows = key_shape[0]`, `bind.rs:2504-2506`, means 2t trips for t of work) | **X**: if trips == 2t the fix is the `Domain` lowering and it is priced at ~1.65 ms |
| **H0** | `KernelKey` POD struct replaces `kernel_cache_key -> String` (`msl.rs:930`, sole production caller `metal.rs:3875`), **extents included** (`crit #5`, `crit #6`) | 616 `String`s/token deleted **and** the wrong-kernel-served hole closed | **A** on plan-hit steps with N = 100 asserted; **B**; a regression test that two ops differing only in reduce extent get different keys |
| **T0** | omega tier repair: `--no-default-features` EXIT 101, 11 errors, `msl.rs:2546` `.to_string()` with no alloc import (`tiers-census.md:3-9`) | omega builds at `--no-default-features` | **X**: exit code 0 **and the module list the restricted build compiled** (P3's N==0 clause) |
| **T1** | `proxima-model-interop` gains an `alloc` feature (it has none today) | the crate builds `--no-default-features --features alloc` | **X**: exit 0 + module list. Lands **before** B4 puts the FSM in this crate |
| **A0** | `IndexPattern::compose` + strided extent resolution (`shape.rs:224-244`) + `is_identity_projection` relaxation (`bind.rs:1156-1164`) | both gates relaxed, unit projection still winning | **B**: every existing program binds byte-identically |
| **A1a** | `Offset`/`SymbolId` in `map.rs` only, `Zero` deleted | one affine offset type | **B**; `PartialEq` node-for-node against the pre-card bind |
| **A1b** | `Domain` on `Op`, `#[serde(default)]`, the 32 struct-literal sites updated | `Domain` exists and is empty everywhere | **B** byte-identical (empty domain = full box) |
| **A2-lower** | shared `lower::LoopWalk` in `proxima-tensor` — stage order, live axes, domain-as-bound-vs-guard, cooperative width, computed **once** | one lowering the emitters consume | **X**: `lowering-audit.md`'s 28 duplicated emitter functions and 3 diverged `reduce_is_cooperative` signatures reduced to 1 |
| **A2a-e** | `Loop` behind `feature = "loop-fusion"`, **one emitter per card** (cpu, metal, wgpu, vulkan, cuda) — `crit #9`: AB's single A2 was 255 sites with no rollback granularity below the whole port | that emitter renders `Loop` | **P** + **B** per emitter |
| **B1** | `plan()` pure + `omega::schedule` | codecs out of `plan()` | **B**, **M** |
| **B2c** | one driver: `grep -c 'commandBuffer()' omega/src/metal.rs` from 3 to 1 | one command-buffer owner | **P**, **M** |
| **B3** | six thread-locals become `MetalSession` fields: `grep -c 'thread_local!' omega/src/metal.rs` from 6 to 0 | one owner, no ambient state | **P**, **M**, `Plan` free-standing |
| **B4** | `Step` FSM + `Session` pipe in interop (after T1) | the four typestates | **A** with N = 128 asserted, **T** |
| **B5** | `BarrierPoint` with `resources` + `cause`; `memoryBarrierWithResources` replaces scope-wide barriers | per-point barriers | **P**, **M** (expect a gain; CoV-gated), barrier count logged |
| **B6a** | `Backend`(7) -> `Engine::{Cpu, Gpu}` + `GpuDriver::{Metal, Wgpu}` resolved once by `for_target()`; `Vulkan`/`Cuda`/`Npu`/`Ane` deleted (`backend.rs:265-375` returns `NotImplemented` for all four, `lowering-audit.md:51`) | 7 variants -> 2 + 2 | **B** byte-identical; `grep -c 'BackendError::NotImplemented' omega/src/backend.rs` == 0 |
| **B6b** | placement: `ScheduledOp.engine` field + identity composition | schedules carry an engine; GPU-only path byte-identical | **B**; **only if D0c admitted the work**. **KILLED 2026-09-05 by ROW 302 (D0c did not admit)** |
| **B7** | placement rule pipes + `BandwidthTable`; layer-split experiment (last N layers on `Engine::Cpu`) | ms/token and text for N in {2, 4, 8} | **T** text-identical **and** **X** steady-state ms/token vs GPU-only; a regression halts placement |
| **B8** | cross-engine `Hazard::CrossEngine` + the `waitUntilCompleted` join, CPU sink on `matmul_worker_count()` threads | one barrier list across two engines | **P**, **M**; a control with the join removed **must** produce wrong logits |
| **A4** | `ShareAxis` + cooperative staging **in one card** — `msl.rs:2586` already implements rescaled online softmax with threadgroup staging; landing a generic emitter first regresses the 3.0-3.6 ms arm (P14, the incumbent wins) | fused attention through `Loop` | `assert_close 1e-6` + EM-at-64 >= 0.99, **M** against attn CoV 9.9% |
| **A5** | epilogue + broadcast-epilogue fusion | dispatch count 616 -> 520 MEASURED | **X** measured, never projected; **B** |
| **A6** | prologue fusion; `CachedAttention` and `fuse_cached_attention: bool` deleted | 520 -> 264 MEASURED | **X**; **B** |
| **C1** | layer builders leave `spec.rs`'s `config` gate | a program builds at the alloc tier | **X**: `--no-default-features --features alloc` builds the builder module, module list named |
| **C2** | `LayerBindings` + typed `CachedLayerRoots`/`SplitRotaryRoots`; one `too_many_arguments` allow removed per card | that builder takes one struct | **B**; `grep -c too_many_arguments proxima-tensor/src/spec.rs` decrements |
| **C3** | `cached_len` leaf + ninth operand + `exact_merged_causal_mask_cached_len` deleted; the domain reads `BoundOffset::Symbol` | three mechanisms become one | **B** byte-identical at every `cached_len`, including the bucketed extents `cpu_mask_zero_ulp` sweeps |
| **C4** | `kv_bucket_tokens` moves to `ServingConfig`, default seeded from `sized::KV_BUCKET_TOKENS`; bucket enters `PlanKey` | a runtime knob, one source of truth | **B** at the seeded default; a parity test that two buckets do not share a plan entry; config<->builder round-trip asserted (P4) |
| **D2** | s-axis fold enabled in the production decode path (after D1c) | A > 1 actually reached | **G**: weight bytes/pass **flat** in k. <!-- measured 2026-09-04: k' --> MEASURED (n-gram drafting, `test/draft-acceptance-harness`, `real_run3.log`): mean k' = 1.36 (k=4), 1.49 (k=8), both < 1.5. **A draft MODEL is required for A > 1.5; its bytes must be priced.** Kill: draft-model bytes per accepted token >= the weight bytes it saves |
| **D3** | union density + acceptance harness: `d_u(k)`, `g_u(k)`, `A(k)` for k in {2,4,8} on the 200-prompt set | the three measured curves | **X**; **kills the elision lever before any kernel is written** if `d_u(4) > 0.6` or `g_u(4) > 0.75` |
| **E0** | exact-match harness with its degenerate control (full-model reference logits cached) | the harness | **X**: must read exact-match **1.000** against the full model, or the metric measures something else |
| **Q1-Q5** | the codec and elision cards (Q3_K FFN, out Q4_K, KV f16, group elision kernel, router) — one lever per card, in §D.4's route order | that lever's measured MB/pass and ms/token | **G** within 5% of §D.3's prediction; **Q** EM-at-64 >= 0.98 (codecs) / >= 0.95 (elision), re-measured on the whole lossy **stack**, not the lever alone |
| **H4** | `LoadedModel::call` re-faced onto `AcceptedRun` (`generate.rs:1542-1556`: `In = (String, usize)`, `Out = (Vec<u32>, String, bool)`) | the bare `bool` and per-call `String`/`Vec` stop existing | **T**, **A** with N asserted |

---

# F. What is NOT a pipe, and why each is justified

1. **IR and plan data** — `Op`, `BoundOp`, `Loop`, `Stage`, `Domain`, `HalfSpace`, `Offset`, `Plan`,
   `Schedule`, `ScheduledOp`, `BarrierPoint`, `ResidencyPlan`, `KernelKey`, `BandwidthTable`,
   `AcceptedRun`, `LayerBindings`. Values pipes carry. Each earns its place on the second question:
   `Domain` lets a caller write a banded reduce whose iteration space is triangular; `Loop` lets a
   caller express five folds sharing one traversal; `BarrierPoint` lets a caller see *which* resources
   conflict and *why*, which `MTLBarrierScope::Buffers` destroys; `ScheduledOp` lets a caller read which
   ops ran on which engine, which a per-plan engine argument destroys; `LayerBindings` makes 21 swappable
   `NodeId` positions non-swappable. <!-- r2-fix: `LayerInputs` was a name that appeared nowhere else;
   it is `LayerBindings` (§C.2). The seven deleted support types of §B.2 are gone from this list. -->
2. **`Step` and its four typestate structs** — a sans-IO FSM. Transitions take different argument types
   and are driven an arbitrary number of times by a caller who owns the loop; `Pipe::call` is one
   `In -> Out`. Precedent: `proxima-primitives/src/pipe/sans_io.rs:41-52`.
3. **`MetalDriver`/`MetalSession` and the CPU sink** — resource owners for `!Send` device handles and
   for a worker pool (`RefCell` not a mutex; a thread-local is a missing owner, not a missing lock,
   P21 rung 1). Their *behaviour* (`Submit`) is a pipe; `poll_complete(&self, ticket, cx) -> Poll<..>`
   is the P20 reactor surface and is **not** claimed to be a pipe.
4. **`plan()`, `schedule()`, `schedule_barriers()`, `program_from_toml()`** — plain functions. Judgement
   call, named as one: written both ways the call sites are identical lines, and `call`'s `async` +
   by-value `In` make the pipe form strictly worse for pure plan-time work. The moment a caller supplies
   an alternative implementation they become pipes and nothing else changes — which is exactly the
   discriminator that *admitted* the fusion rules **and the placement rules**.
5. **Newtype ids** — `TokenId`, `SlotId`, `SymbolId`, `SubmitTicket`, `RuleSetId`, `PlanKey`, `StageId`,
   `ExtentDigest`, `CodecMask`. P11's compile-time clause; `NodeId` (`op.rs:26-28`) is the shipped
   precedent.
6. **`Engine` and `GpuDriver`** — data, not pipes, and both are *subtractive*: they replace
   `omega/src/backend.rs:78-91`'s seven variants (three executable, four `NotImplemented`) with two
   engines plus a driver resolved once per target. Deliberately **no** `Mixed` variant: mixing is a
   placement over ops and it lives on `ScheduledOp.engine`. `Npu`/`Ane` are deleted rather than
   carried — a variant with no lowering is a promise the type system makes on behalf of code that
   does not exist.

**Defect declared, not hidden (`crit #14`):** `Loop` is RISC at the `Op` face and a multi-stage
traversal form at the `BoundOp` face that five emitters must interpret, and `lowering-audit.md` already
measures 28 duplicated emitter functions with `reduce_is_cooperative` implemented three times with
diverged signatures. Hence CARD A2-lower above. Without it the port multiplies the duplication the audit
measured, and R5-as-a-per-emitter-theorem is one rule implemented five times.

---

# G. Worked example, which is the test (P17)

```rust
/// Two programs computing the SAME banded softmax-weighted reduction, differing only in
/// memory layout: keys head-major, and keys position-major. Both must bind to ONE `Loop`
/// with five stages and identical results, because the rule reads strides from `Layout`
/// and never compares them to a literal — `bind.rs:2465-2474` compares against eight
/// literal stride tuples today, so this test cannot pass on main.
#[proxima::test]
#[case::head_major(KeyLayout::HeadMajor)]
#[case::position_major(KeyLayout::PositionMajor)]
async fn a_banded_softmax_reduction_fuses_at_any_operand_layout(#[case] layout: KeyLayout) {
    let program = banded_softmax_program(layout);            // real openchat dims: 8 kv heads, group 4, head_dim 128
    let shapes  = shape::infer(&program, &[SEQUENCE, MERGED_LENGTH]).expect("infers");
    let rules   = FusePrologue.and_then(FuseEpilogue).and_then(ShareAxis);

    let bound = rules.call(bind_plain(&program, &shapes, &[OUTPUT]).expect("binds")).await.expect("rules apply");

    assert_eq!(bound.len(), 1, "the whole chain is one Loop at any layout");
    let BoundOpKind::Loop { stages, operands, .. } = &bound[0].kind else { panic!("must be a Loop") };
    assert_eq!(stages.len(), 5, "score, max, exp-sum, value-sum, divide");
    assert_eq!(operands.len(), 3, "Q, K, V — the cache length is a symbol, not a ninth operand");
    assert!(matches!(stages[1].domain.constraints[0].offset, BoundOffset::Symbol(_)),
            "the live length is a uniform read, not a compiled-in sentinel");

    let fused = cpu::evaluate_with(&program, &shapes, &blocks, &rules).await.expect("fused runs");
    let plain = cpu::evaluate_with(&program, &shapes, &blocks, &NoRules).await.expect("unfused runs");
    assert_close(&fused, &plain, 1e-6);        // one rule against its own unfused form, not two implementations
}

/// <!-- r2-fix: the placement counterpart, which is the §B.4 test. --> A placement rule
/// moves the last two layers to the CPU engine and changes NOTHING about the logits:
/// same schedule, same barriers plus two cross-engine joins, same text.
#[proxima::test]
async fn a_placement_rule_moves_layers_without_changing_a_single_logit() {
    let plan     = plan(&program, &symbols, &[OUTPUT], RuleSetId(0)).expect("plans");
    let gpu_only = schedule(&plan, &blocks).expect("schedules");
    let hybrid   = PlaceLastLayers { layers: 2, engine: Engine::Cpu }
        .call(gpu_only.clone()).await.expect("places");

    assert_eq!(hybrid.ops.len(), gpu_only.ops.len(), "placement moves ops, it never adds or drops one");
    assert_eq!(hybrid.ops.iter().filter(|op| op.engine == Engine::Cpu).count(),
               ops_in_last_layers(&plan, 2), "exactly the named layers moved");
    assert!(hybrid.barriers.iter().any(|point| matches!(point.cause, Hazard::CrossEngine { .. })),
            "a cross-engine dependency is a barrier point, not an implicit hope");

    let (gpu_logits, hybrid_logits) = (run(&gpu_only).await, run(&hybrid).await);
    assert_eq!(gpu_logits, hybrid_logits, "byte-identical: the engines share one unified-memory mapping");
}
```

Green here means does-what-it-does, never is-what-it-should-be. The shape argument is §A's minimality
(the `Lookup::extent` candidate was examined and rejected: it bounds an *address*, not an iteration
point, and acts per operand where the band must constrain the key read, the value read and the
accumulation together) and §F's lint — falsifiable by the list in §F, not by this test.

---

<<UNFINISHED: written at the 30-minute mark. Closed since AB2: §C in full (C.1 tier split from the
shipped `config` gate, C.2 `LayerBindings`/typed roots/`cached_len` leaf deletion/`ServingConfig`
bucket, C.3 the new-architecture table); the twelve support types (six defined with fields, seven
deleted, `ResidentSet`->`ResidencyPlan` collapsed); every card as a gated 30-minute row ordered
GPU-gap first with the measured D0 ceiling; decision 1's rationale corrected to declared-vs-derived;
§D.4 re-priced onto the measured 237.79-264.29 GB/s; §B.4 placement, including the `Backend`(7) -> `Engine`(2) + `GpuDriver`(2) collapse and card B6a. Still open: (1)
`ScatterBounds::{Fault, Drop}` and its interaction with the binding `Domain` (a dropped scatter is an
out-of-domain write) — the semantics of the elision lever, unworked; (2) the `sized::` const table with
per-const overflow policy is not re-tabulated, and `crit #6`'s reachability gate
(`PACKED_ROW_BLOCK_SIMDGROUPS`'s only consumer is unreachable — a swept constant that never reached a
dispatch) needs one cell per new const, including the three added here
(`PACKED_ROW_ACTIVATION_GROUP`, `MAX_INLINE_CONSTRAINTS`, `MAX_PLAN_SLOTS`); (3) the CPU sink's own
allocation budget under `matmul_worker_count()` threads is not stated, and B8's **A** gate needs it;
(4) `Q1-Q5` are one row for five levers — each needs its own row with its own MB/pass before D3
reports; (5) <!-- measured 2026-09-04: k' --> the tokenizer raised `Tokenizer(InvalidUtf8)`
(`proxima-model-interop/src/bind.rs:3931`) on a prose-category prompt during the draft-acceptance
harness run — fix branch `fix/tokenizer-decode-split-utf8`, not yet landed>>
