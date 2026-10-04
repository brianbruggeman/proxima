---
status: draft (research input to SPEC.md; not audited)
date: 2026-10-04
base: proxima-windows main 4b4be6cf
---

# research classification of the pipeline-as-data catalog

Supersedes the THIN/ACTIVE split in `frontier.md`. Test applied (owner, 2026-10-04):

- **engineering**: a known method answers the core question and a correctness oracle exists.
- **active research**: no settled method, others are publishing on it in 2025-2026. It gets a hook.
- **open research**: no known method answers the core question in our setting, and no one is visibly
  working on that exact question.

Scope notes from the owner and main thread, applied here:
- **Dropped, not classified:** model merging in any form, including merging inside a quantized
  format (frontier row C23 and the research-track row "merging inside a quantized format" in SPEC.md).
  It is a tuning and training knob with no inference-level benefit.
- Scope question this raises for the designed slices, not decided here: `cartridge_train` (fsm-techniques
  R14) and the fitters (R18) are training-side work of the same kind.

## provenance

- "opened" = I fetched that page this session and read what it returned. The fetch tool returns a
  model-written summary of the page, so "opened" for an arXiv id means the abstract-level summary, not the
  paper body. One paper body was read from its PDF text (2607.17283, lines cited).
- "snippet" = appeared only in a search-result listing.
- Counts of `cached` arithmetic are labelled `arithmetic, not measured`.
- Code anchors are `proxima-windows` main 4b4be6cf, read this session unless marked "as cited by spec".

## section 1: open research (1 item)

Searches that returned nothing for the owner's two named examples are recorded under "reclassified"
below. They did not stay open.

### OPEN-1: what exactness can a technique honestly promise on a shape-dependent quantized kernel set

**Core question, stated precisely.** For omega's Metal kernels, where the kernel chosen for an op depends on
the row count (cached-attention form, tiled GEMM at `TILED_GEMM_MIN_TOKENS`, cooperative reduce at
`COOPERATIVE_REDUCE_MIN_LEN`; `omega/src/msl/emit_and_classify.rs:1942`, `:2261`), can a per-op forward
error bound be derived from each kernel pair's reduction tree such that, at every decode step where the
top-1 minus top-2 logit margin exceeds the bound, the argmax of the multi-row path (verify-N, tree,
lookahead, chunked-prefill continuation, prefix-cache continuation) is certified equal to the argmax of the
decode-1 path, with the remaining steps re-evaluated at the decode-1 shape?

**Why the design assumes an answer.**
- decode-as-data R7: lossless techniques give output byte-identical to non-speculative decode for any
  `SamplingConfig` (`decode-as-data/SPEC.md`, requirement list, R7).
- decode-as-data R13 and AC12: prefill at ubatch sizes {1, 7, whole} gives ids equal to llama-server.
- prefix-cache-reuse problem statement: reuse is "token-identical to a fresh prefill".
- fsm-techniques R6a/R13a: content-keyed sealed blocks restored byte-identically, which treats equal key as
  equal bytes. That holds only if the producing kernel is shape-invariant.
- decode-as-data R18 tests each shape-selected kernel against the CPU reference on both sides of its
  threshold, with a tolerance. It does not test equality across the threshold.
- decode-as-data states GDN chunked vs recurrent is "equal in exact arithmetic, not bitwise" (the
  "two kinds of split" section).

**Why no known method answers it (what I searched, what came closest).**
- Searched: "bitwise batch invariant kernels speculative decoding verification exact greedy equality",
  "certified exact speculative decoding numerical error bound logit margin fallback", "rigorous floating-point
  rounding error bound transformer logits reduction order argmax certification greedy decoding".
- A body-level measurement on this hardware class exists: on a quantized Metal backend, the same context
  evaluated as a batch of five versus one at a time shifts logits by up to about 0.1, enough to flip argmax at
  near-ties, with top-1/top-2 probability gaps observed at 1.8e-3 to 5.8e-3. The authors ran their greedy leg on
  fp32 CPU because of it and wrote that bit-exact greedy equivalence on the quantized backend is unattainable
  for any implementation whose baseline and verifier use different batch shapes. Source: arXiv 2607.17283
  (https://arxiv.org/abs/2607.17283), PDF text lines 472-478 and 843-847, opened. That sentence is the paper's
  own claim; it is contradicted in scope by fixed-config and verify-and-rollback work below.
- Batch-invariant kernels remove the dependence at a kernel-design cost. Vosti (2609.38981,
  https://arxiv.org/abs/2609.38981, opened) certifies bitwise-identical logits across batch and cache
  configurations; its Triton analyzer is for GPU kernels and the fetched summary says it does not address Metal or
  quantized kernels.
- Verify-and-rollback (LLM-42, 2601.17768, https://arxiv.org/abs/2601.17768, opened) keeps a non-deterministic fast path
  and replays candidates under a fixed-shape reduction schedule.
- Margin-triggered verification (MarginGate, 2605.30218, https://arxiv.org/abs/2605.30218, opened): flips are 0.3-1.3% of
  steps across the models it tested; it verifies only low-margin steps, by an empirical threshold.
  Its multicandidate/speculative case is stated as needing other policies (snippet of its limitations section).
- A margin theorem that assumes the perturbation bound exists for a VLM drafter (2609.00355,
  https://arxiv.org/abs/2609.00355). Opened, but the fetched page did not show the theorem, so this is snippet-level.
- The closest rigorous machinery: the classical sequential-accumulation bound and a layerwise deviation
  recursion for Lipschitz certification under floating point (2603.13334, https://arxiv.org/abs/2603.13334, opened; feed-forward
  ReLU networks, abstract does not state reduction-order handling). A rounding-error tool reports that the reduction tree changes the bound by
  about 50x (serial 2.91e-11 vs tree 5.68e-13 over 1,024 points; Satire,
  https://arxiv.org/html/2503.05924, snippet).
- Not found: a bound for a transformer forward on block-quantized (Q4_0/Q4_K) kernels whose reduction trees
  are known, used as a margin certificate with fallback.

**What it would let us do.** Stop asserting byte-identical ids for speculation, tree, lookahead, chunked
prefill and prefix-cache continuation, and instead publish a contract per kernel pair. It makes the
decode-as-data AC4 `lossless_shape_` count (9 tests) and prefix-cache AC "token-identical to fresh
prefill" satisfiable on a stated basis rather than by agreement of near-tied prompts. It also fixes what a
content key means across plan shapes (catalog change CC-2).

**Benefit line.**
- Who: the owner's lossless gates (decode-as-data AC4, AC12, AC17; prefix-cache refutation condition) and every
  later technique that changes evaluation shape.
- Mechanism: re-evaluation only at steps whose margin is below the certified bound.
- Order of magnitude: extra work is at most the fraction f of steps below the bound, times one decode-1 pass.
  For f = 0.1856 (the verified-step fraction MarginGate reports on Llama-3.1-8B, snippet of
  https://arxiv.org/pdf/2605.30218, a different engine and an empirical threshold) the added cost is at most
  18.6% of decode-1 time per token. arithmetic, not measured on omega. Against verify-all (LLM-42's
  baseline), the saving is 1 - f.

**Hypothesis (falsifiable).** For each omega kernel pair, a bound epsilon derived from the reduction tree
(count of partial sums times unit roundoff times the sum of absolute terms, composed layer by layer) holds on
every recorded step, and the fraction of steps with margin below 2 x epsilon is at most 25% on gemma4 E2B, openchat
and qwen3.

**Baseline.** (a) verify-all at decode-1 shape (LLM-42 style); (b) empirical margin threshold (MarginGate style);
(c) no contract (today).

**Measurement.** Instrumentation first, no kernel change: dump final-layer logits for the same 256-token
continuation of the three vendored prompts per checkpoint (`proxima-model-interop/tests/fixtures/llama-parity/`
per architecture-as-data slice 0) evaluated at shapes {1, 5, 9, whole prompt}. Per step record max
absolute logit difference between shapes, the top-1/top-2 margin, and whether the argmax flipped. Control that
should fail: a bound computed with the wrong reduction tree (serial instead of the actual cooperative tree)
must be violated by observed differences or be so loose that f > 25%.

**Kill criterion.** Any recorded step with observed difference above epsilon (the bound is unsound); or f
above 25% (the fallback cost approaches the speculation gain, to be re-derived from the parity spec's R19
verify-width cost curve before the run); or any flip at a step whose margin exceeds 2 x epsilon.

## section 2: active research (33 of the 45 candidates, plus residuals)

Each row: the hook it needs, whether SPEC.md's catalog (H1-H17, N1-N10) already has it, and one opened source.
"Cat." = catalog has the hook. Sources are abstract-level fetches unless stated.

| id | core question | hook | cat. | source (opened) |
|---|---|---|---|---|
| C1 | which steering law applied at which layer to which rows | N2 | yes | Spherical Steering 2602.08169, https://arxiv.org/abs/2602.08169: geodesic rotation with a confidence gate |
| C3 | per-token layer skip or exit without retraining; where the skipped layers' KV comes from | N3 | yes | TIDE 2603.21365, https://arxiv.org/abs/2603.21365: learned routers at checkpoint layers; vSkipper 2609.37062, https://arxiv.org/abs/2609.37062: groups tokens by skip decision inside the serving engine's batching and cache interfaces |
| C4 | batching rows that need different loop counts | N3 plus H10 | partly (H10 has no depth-aware admission) | continuous depth batching 2608.09444, https://arxiv.org/abs/2608.09444 |
| C5 | per-row adapter application and the adapter cache key | N4 | yes; key rule is CC-1 | aLoRA serving 2512.17910, https://arxiv.org/abs/2512.17910: base-aligned block hashing, adapter id enters the hash only for tokens after activation; InfiniLoRA 2604.07173, https://arxiv.org/abs/2604.07173 |
| C6 | KV codec and error at 2-bit and below | N7 | yes | CommVQ 2506.18879, https://arxiv.org/abs/2506.18879 |
| C7 | which rows to keep under a budget | H13 | yes | PruLong 2506.17121, https://arxiv.org/abs/2506.17121 |
| C8 | cross-layer low-rank KV | N7 | yes | xKV 2503.18893, https://arxiv.org/abs/2503.18893 |
| C9 | retrieval over sealed blocks | H12 | yes | FIER 2508.08256, https://arxiv.org/abs/2508.08256: token-level retrieval from 1-bit keys; AB-Sparse 2605.12110, https://arxiv.org/abs/2605.12110: per-head block sizes |
| C13 | keyed logit bias or tournament sampling and its detection | N1 / H14 | yes | SynthID-Text analysis 2603.03410, https://arxiv.org/abs/2603.03410: mean score vulnerable to more tournament layers |
| C15 | dynamic patches replace a fixed vocabulary | H3 plus H4/H6 (patch encoder is model structure) | H3 yes; H6 no | BLT 2412.09871, https://arxiv.org/abs/2412.09871 |
| C16 | prefill/decode split and the transfer ordering | N8, H8, H10 | yes | PPD disaggregation 2603.13358, https://arxiv.org/abs/2603.13358 |
| C17 | preempt at operator boundaries | H10 (needs yield points in H7 plans) | H10 yes; yield points no | FlowPrefill 2602.16603, https://arxiv.org/abs/2602.16603 |
| C18 | a readout that predicts an exit point (coordinator item 2) | N2 feeds N3/H11 propose | yes | Draft & Verify 2309.08168, https://arxiv.org/abs/2309.08168: training-free self-speculation by skipping layers, up to 1.99x; SWIFT 2410.06916, https://arxiv.org/abs/2410.06916: on-the-fly choice of skipped layers, 1.3-1.6x, output distribution preserved; ConfLayers 2604.14612, https://arxiv.org/abs/2604.14612; SimLens 2507.17618, https://arxiv.org/abs/2507.17618 |
| C19 | probes in the decode loop | N2 read by H15/H17 | yes | hidden-state probes 2606.10487, https://arxiv.org/abs/2606.10487 |
| C20 | N completions from one state | H11 fork | no (frontier.md gap list) | prefix-confidence 2507.18122, https://arxiv.org/abs/2507.18122 |
| C21 | multi-token heads | H11 plus readouts | yes (decode-as-data R5) | FastMTP 2509.18362, https://arxiv.org/abs/2509.18362 |
| C22 | block diffusion and its caches | H11 non-autoregressive entry | yes | Fast-dLLM v2 2509.26328, https://arxiv.org/abs/2509.26328 |
| C24 | per-layer precision from sensitivity | H5 | yes | MXSens 2607.17733, https://arxiv.org/abs/2607.17733 |
| C25 | force or forbid end of thinking | H11 / H15 | stop proposal not in H11 | s1 2501.19393, https://arxiv.org/abs/2501.19393 |
| C26 | position-agnostic cache assemble | H9, N9 | yes | LazyAttention 2606.04302, https://arxiv.org/abs/2606.04302 |
| C27 | learned cache artifacts | H9, N8 | yes | Cartridges 2506.06266, https://arxiv.org/abs/2506.06266 |
| C28 | idle-time derived context | H10 | yes | sleep-time compute 2504.13171, https://arxiv.org/abs/2504.13171 |
| C29 | sparse decode with dense rectification | H11, H12 | yes | Rectified Sparse Attention 2506.04108, https://arxiv.org/abs/2506.04108 |
| C31 | speculation over actions | H11 generic entry | yes | Speculative Actions 2510.04371, https://arxiv.org/abs/2510.04371 |
| C32 | prompt compression | N5 | yes | empirical study 2505.00019, https://arxiv.org/abs/2505.00019 |
| C34 | determinism under dynamic batching | N10 | yes, but see CC-2 | LLM-42 2601.17768, https://arxiv.org/abs/2601.17768; MarginGate 2605.30218, https://arxiv.org/abs/2605.30218 |
| C35 | request-owned fast weights | N2 plus H11 commit | yes | LaCT 2505.23884, https://arxiv.org/abs/2505.23884 |
| C36 | expert override and prefetch | N6 | yes | speculating experts 2603.19289, https://arxiv.org/abs/2603.19289 |
| C37 | activation sparsity | N2 / H7 | yes | TEAL 2408.14690, https://arxiv.org/abs/2408.14690 |
| C38 | latent hand-off between models | N8 | yes | survey 2606.05711, https://arxiv.org/abs/2606.05711 |
| C39 | cross-vocabulary drafting | H3 x H11 | yes | OmniDraft 2507.02659, https://arxiv.org/abs/2507.02659 |
| C44 | observability tap | N2 | yes | DMI-Lib 2605.11093, https://arxiv.org/abs/2605.11093 |
| C45 | probe as reasoning stop signal | H15 | yes | LYNX 2512.05325, https://arxiv.org/abs/2512.05325 |

### design-derived active items (from the ablation)

| id | core question | hook needed | cat. | source (opened) |
|---|---|---|---|---|
| A-1 | tree or path verification on recurrent (GDN) layers without per-node state snapshots. SPEC decode-as-data restricts recurrent layers to chain shapes (R3b in its FSM section) | H11 shape/accept/commit gated by a layer capability, not an architecture | partly: decode-as-data has the chain-only rule | TreeWY 2608.20961, https://arxiv.org/abs/2608.20961: tree-structured WY transform, Qwen3.5 35B and 397B; Bole 2608.01651 and SpecLA 2607.16673 (snippet) |
| A-2 | prefix caching for hybrid models where recurrent state forces exact-match hits | H13 grain (entry vs block) | sketch 10 chose entry grain | Marconi 2411.19379, https://arxiv.org/abs/2411.19379 |
| A-3 | selection block size decoupled from page/seal block size. fsm-techniques says "No second chunk size exists" and requires `block_tokens` multiple of 16 (`fsm-techniques/SPEC.md:153`) | H12 summary granularity vs H13 key/tier granularity | no | AB-Sparse 2605.12110, https://arxiv.org/abs/2605.12110: block size too large for sensitive heads misses tokens (snippet of the same paper) |
| A-4 | approximate reuse of rows produced under a different producer (adapter, steering, shifted position) with a bounded error | H9 assemble with recompute selection; key carries an "approximate" provenance | recompute selection is fsm-techniques slice 9 | Shared-prefix reuse across standard LoRA adapters 2609.17109, https://arxiv.org/abs/2609.17109: 16x TTFT at 8K, exact-match change from -4.6 to -0.8 points on GSM8K, "neither quality equivalence nor a general boundary-selection rule is established"; DroidSpeak, https://arxiv.org/pdf/2411.02820 (snippet): about 10% of layers critical, found by offline profiling; AgentKVShift 2607.21604, https://arxiv.org/abs/2607.21604 |
| A-5 | error across a chain of KV codecs (seal, tier demotion, cross-layer merge, rectification) | N7 plus a per-block witness | no | WitCert 2607.28699, https://arxiv.org/abs/2607.28699: per-layer, per-head, per-step upper bound on total variation, "sound for any cache-preserving black-box quantizer"; transform-coding analysis 2608.14191, https://arxiv.org/abs/2608.14191: additive key and value distortion under a white-noise model; QEvict 2608.05326, https://arxiv.org/abs/2608.05326: three tiers, no error bound across migration |
| A-6 | placement under one memory cap across experts, KV and prefix entries | H8/H13 placement as a pure decision | yes | VAMP 2609.13537, https://arxiv.org/abs/2609.13537; placement study 2609.16215, https://arxiv.org/abs/2609.16215: gains come from tier capacity, not placement policy |
| A-7 | quantized KV composed with deviation-based recompute selection (CacheBlend's HKVD ranks loaded vs recomputed rows) | H9 assemble | recompute selection is a designed slice | AgentKVShift 2607.21604, https://arxiv.org/abs/2607.21604: reports compounding when reuse and low-bit KV stack (CacheBlend's own abstract in https://arxiv.org/abs/2405.16444 does not address quantization) |

Residual open sub-questions inside active items (not counted as open; each has people working on it):
- A-4: a certified, not profiled, bound for approximate reuse. 2609.17109 states the rule is not established.
- A-5: whether a runtime witness replaces the designed static composition of codec error. WitCert covers a
  black-box quantizer; a chain was not checked by me.
- C2 approximate case: reuse of unsteered rows above the steering layer with recompute of a chosen subset.

## section 3: engineering (11 of the 45 candidates, plus design-derived)

| id | core question | oracle | source / evidence (opened) |
|---|---|---|---|
| C2 exact | per-row steering vector table, and exact sharing of rows below the steering layer. Rows at layers up to L are a pure function of ids and config restricted to layers up to L, so the key is a per-layer-group hash chain; vLLM keys blocks by `(block_hash, group_id)` and takes the intersection of per-group hits | bitwise equality of shared rows against an unsteered prefill, given determinism (OPEN-1) | EasySteer 2509.25175, https://arxiv.org/abs/2509.25175 (vLLM-integrated, no prefix-cache discussion); vLLM hybrid KV manager, https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ (opened); KV cache steering 2507.08799, https://arxiv.org/abs/2507.08799 |
| C10 | pin sink rows; per-head learned sink logit in softmax | llama.cpp has `ggml_flash_attn_ext_add_sinks` (`/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml.c:5570`, f1ea20621) | 2510.06477, https://arxiv.org/abs/2510.06477 |
| C11 | grammar token mask with state advance on commit | llama.cpp grammar sampler (`include/llama.h:1524`) | XGrammar 2411.15100, https://arxiv.org/abs/2411.15100 |
| C12 | classifier-free guidance combine of two logits rows (the layer-contrastive "when to contrast" part is active: 2505.23657, https://arxiv.org/abs/2505.23657) | offline reference formula; `grep -in cfg include/llama.h` prints nothing at f1ea20621, so llama.cpp is not an oracle here | CFG-for-LMs 2306.17806 not opened |
| C14 | prompt-boundary correction | offline brute-force over tokenizations on short prompts (reference I would write; none exists in the repo) | 2412.03719, https://arxiv.org/abs/2412.03719: exact and approximate algorithms |
| C30 | conformal accept/escalate | empirical coverage on held-out data under the same serving configuration | conformal cascade 2607.25018, https://arxiv.org/abs/2607.25018 |
| C33 | RoPE scaling | `transformers` YaRN reference worked example (long-context R10a-c) | LongRoPE2 2502.20082, https://arxiv.org/abs/2502.20082 |
| C40 | truncation chain and its order | llama.cpp sampler constructors at `include/llama.h:1480, 1492, 1495, 1562` | 2609.15476, https://arxiv.org/abs/2609.15476: order of scaling and truncation matters |
| C41 | template as a pure, sandboxable program | llama.cpp `common/jinja/` renderer output | template backdoors 2602.04653, https://arxiv.org/abs/2602.04653: attack surface report |
| C42 | streaming detokenize with hold-back | streamed bytes equal batch decode; llama-server `content` (decode-as-data AC15) | UTF-8 plumbing 2511.05578, https://arxiv.org/abs/2511.05578 |
| C43 | descriptor to graph at today's speed | op-graph digests 7/7 and token parity 4 checkpoints (SPEC.md:28-30); speed by interleaved arms (architecture-as-data AC12) | AttentionEngine 2502.15349, https://arxiv.org/abs/2502.15349: attention variants as primitive ops fused into kernel templates |

Design-derived engineering (the ablation's "assumed answer" has a known method and an oracle):

| id | design component | what it assumes | method | oracle |
|---|---|---|---|---|
| E-1 | cache key under hooks (SPEC.md cross-cutting rule; coordinator item 3) | a key can be built so a hit is valid. Exact case: dependency-closure memoization. Salsa tracks which inputs each query read and returns the memoized value if none changed (https://salsa-rs.github.io/salsa/overview.html, opened); Bazel keys actions by hashed inputs and states a cache hit requires the same inputs to give the same output (https://bazel.build/basics/hermeticity, opened); Nominal Adapton proves from-scratch consistency (1503.07792, https://arxiv.org/abs/1503.07792). For ops written in the 5-op algebra the closure is static: `Op::Input` leaves are named (`proxima-tensor/src/op.rs:196-199`) | rows equal an uncached run. The premise "same inputs, same output" fails across plan shapes, which is OPEN-1; until then the key must include the plan shape |
| E-2 | footprint of a hook written as a host pipe | the author declares which layers and rows it reads and writes | differential check: perturb inputs outside the declared footprint and assert unchanged outputs | the check itself |
| E-3 | visibility as data (decode-as-data) | a chain-shaped tree gives bit-identical logits to chain verify | masked entries contribute exp(-inf) = 0 | AC2 (`visibility_chain_tree_equal_`), llama seq-id layout AC3/AC5 |
| E-4 | top-fraction selection by rank-count (`fsm-techniques/SPEC.md:105`, A16) | lowered to a selection kernel with the same set | known selection kernels | CPU reference with a stated tie rule (see CC-5) |
| E-5 | sample-and-match under speculation | stochastic output identical to sequential | `rng` consumed once per accepted draft in sequential order, stated at `proxima-model-interop/src/generate/decode.rs:1315-1316` | sequential run, same seed |
| E-6 | FSM generic over entry, sans-IO conformance (fsm-techniques R23) | every transition is testable without IO | scripted backend | R23c properties |
| E-7 | calibration tables for conformal cascade | coverage holds | table keyed by serving-config digest, as in E-1 | empirical coverage per configuration |
| E-8 | descriptor config round-trip (architecture-as-data R9) | GGUF to config to TOML to config lowers to the same digest | serde round trip | AC9 digests |

## ablation of the designed architecture

For each designed component: what question does it assume has an answer, and its class.

| id | component | assumed answer | class |
|---|---|---|---|
| D-1 | lossless claims for spec, tree, lookahead, chunked prefill, prefix continuation | multi-row and single-row paths agree bitwise | **open** (OPEN-1) |
| D-2 | content key of a sealed block (fsm-techniques R5, R13) | same key means same bytes | engineering given D-1 (E-1) |
| D-3 | cross-cutting cache key rule (SPEC.md) | a key exists that stays valid under an intervention | engineering for exact, active for approximate (E-1, A-4) |
| D-4 | seal codec N7 across tiers | error composes across seal, demote, rectify | active (A-5) |
| D-5 | placement policy under a cap (H8, H13) | an optimal policy exists | active (A-6); the one measurement found shows capacity dominates |
| D-6 | one block size for seal, key, summary, tier, page (`fsm-techniques/SPEC.md:153`) | one size serves selection quality and tier I/O | active (A-3) |
| D-7 | recurrent layers admit chain shapes only (decode-as-data) | no tree verify on GDN | active (A-1): published methods exist |
| D-8 | tier unit for hybrid models (sketch 10 chose the entry) | block grain unavailable on recurrent layers | active (A-2) |
| D-9 | per-row depth variation (N3) against plans built once (decode-as-data R9, AC8: 3 plans over 64 steps) | row subsets per layer do not multiply plans | active: vSkipper packs cohorts per routed layer (https://arxiv.org/abs/2609.37062) |
| D-10 | descriptor to op graph at speed | digests plus parity oracle exist | engineering (C43) |
| D-11 | five-op algebra needs no sixth op (fsm-techniques refutation condition) | top-k, gather, scatter expressible | engineering with a stated tie rule (E-4) |
| D-12 | FSM state set suffices | Prefill, Decode, Verify(refine), Accept, Rollback, Done cover all | falsifiable claim in the specs, not a research question; checked by the paper test |
| D-13 | sealed rows immutable (R5b) against ReSA rectification overwriting the last f rows (R10b) | horizon is at least f | defect: no `Validate` row exists for it (`fsm-techniques/SPEC.md:356-372` lists `RectifyNeedsSparseRead`, `RectifyNeedsRewind` only). See CC-4 |
| D-14 | conformal judge guarantee under lossy hooks | calibration transfers | engineering (E-7) |
| D-15 | tier names device/host/disk | the device/host distinction carries cost | not researched; M1 unified memory, measure |

## reclassified from `frontier.md` (the six THIN rows and the two owner examples)

| frontier row | was | now | reason |
|---|---|---|---|
| C2 steering in a batch with the cache key bound to the intervention | THIN, research track | engineering (exact), active (approximate) | Exact reuse is dependency-closure memoization with per-group keys (E-1; salsa and Bazel pages opened; vLLM per-group `(block_hash, group_id)` keys opened). Three searches returned no steering-specific cache paper: "activation steering prefix cache sharing...", "steering vector serving system multi-tenant...", plus the EasySteer fetch. That absence is why it looked open. It is not open once the closure view is applied. |
| C14 token healing | THIN | engineering | 2412.03719 gives exact and approximate algorithms |
| C18 lens inside the serving loop | THIN, research track | active | exit-point prediction by layer skipping is published: Draft & Verify, SWIFT, ConfLayers, SimLens (all opened) |
| C23 merging | THIN | dropped | owner scope |
| C42 streaming detokenize | THIN | engineering | oracle exists; one 2025 paper gives the incremental UTF-8 algorithm |
| C43 one generic engine | THIN | engineering | oracle exists (digests, parity) |

## catalog changes implied

- CC-1. H9/H13 cache key is a hash chain over layer groups, each link binding (token ids, producer
  configuration restricted to layers up to that group), mirroring vLLM's `(block_hash, group_id)`.
  Replaces the single-key "cross-cutting rule" in SPEC.md.
- CC-2. N10 grows a field: the exactness contract of a plan, one of bitwise, margin-certified (bound attached),
  tolerance. The content key carries the plan shape or the contract. Blocks of different contract classes are
  not interchangeable until OPEN-1 resolves.
- CC-3. Separate `attention.read.summary_tokens` from `kv.block_tokens` (A-3). Today fsm-techniques R3 makes
  them one value.
- CC-4. `Validate` rows: `seal.horizon_rows >= decode.rectify.every`, and `seal.horizon_rows >=` the widest
  verify shape. Neither is in the reachability matrix (`fsm-techniques/SPEC.md:356-372`).
- CC-5. State tie semantics for the rank-count selection (equal scores must select the stated count), so R8b's
  data-independent count holds (`fsm-techniques/SPEC.md:244-246`).
- CC-6. H11 gets a fork (C20, N completions from one state) and a stop proposal toward H15 (C25, C45). Both
  were already listed as gaps in `frontier.md`.
- CC-7. H11 shape validity on recurrent layers is a layer capability (tree-structured verify kernel present),
  not "chain only" (A-1).
- CC-8. H13 tier grain: entry for layers that cannot be cut into rows, block for the rest (A-2).
- CC-9. N3 plans are cohorts (row subsets per layer); R9's plan count is stated per cohort shape (D-9).
- CC-10. H17 calibration tables are keyed by serving-configuration digest (E-7).
- CC-11. H10 yield points: operator-boundary preemption (C17) needs H7 plans to expose them.
- CC-12. SPEC.md research-track table: remove "merging inside a quantized format" (dropped); mark the steering
  row as engineering (exact) plus active (approximate); mark the logit-lens row active; add OPEN-1.

## counts

| class | candidates from the 45 | design-derived |
|---|---|---|
| open | 0 | 1 (OPEN-1) |
| active | 33 | 7 (A-1 to A-7) |
| engineering | 11 (C2 counted once) | 8 (E-1 to E-8) |
| dropped | 1 (C23) | 0 |

33 + 11 + 1 = 45.

## not done

- No source was read beyond the abstract-level fetch, except the PDF text of 2607.17283.
- The fetch tool's summary of 2607.17283's abstract page said the paper achieved "exact greedy-sequence
  agreement"; the PDF text says the greedy leg ran on fp32 CPU because quantized Metal logits are not
  batch-invariant. The PDF text is what OPEN-1 cites. The abstract-page summary disagrees with it and I did not reconcile the two beyond reading the PDF.
- Not checked: whether WitCert's bound extends to a chain of codecs (A-5); the DroidSpeak, Bole, SpecLA and
  Satire pages were snippet only.
- OPEN-1's bound derivation was not attempted; no kernel was instrumented. Every number in its benefit line is
  from another engine or is arithmetic.
- The claim that no source covers a certified margin bound on block-quantized kernels is the result of the
  searches listed in OPEN-1, not of an exhaustive survey.
