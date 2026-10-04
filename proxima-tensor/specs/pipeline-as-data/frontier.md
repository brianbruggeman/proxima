---
status: draft (research input to SPEC.md; not audited)
date: 2026-10-04
---

# frontier cross-reference for the pipeline-as-data hook catalog

Companion to `SPEC.md` (H1-H17). Ablation: for each stage and each gap between stages, what
would a swappable hook enable. Cross-reference: is the technique under active research.

## provenance of the sources (read this before trusting a row)

- Every arXiv id below was returned by a WebSearch result listing in this session (2026-10-04).
  The month in a citation is the arXiv id prefix (YYMM), not the submission date, except where
  stated.
- One paper was opened in full (WebFetch): 2605.11093 (submitted 2026-05-11). Every other row
  rests on the search-result snippet of that paper, which is a summary, not the paper.
  Mechanism lines are therefore "as the snippet describes it". Status: plausible, not proven.
- No citation was constructed from memory. Items I recognise but did not see listed (DistServe,
  CacheBlend's own arXiv id, RetrievalAttention's details) are either omitted or marked
  "unverified".
- ACTIVE means: at least one 2025-2026 primary source was listed, plus the incumbent paper
  where relevant. It does not mean the technique works or that the claims in the sources hold.

## new stages the catalog lacks (found by the ablation)

SPEC.md has no row for any of these. Each is either a new hook or a gap between existing hooks.

| id | name | where it sits | what it is |
|---|---|---|---|
| N1 | shape | between H11 step and H14 sample | logits row -> logits row (grammar mask, CFG, watermark bias, repetition penalty, steering readout). H14 today owns "sampler chain" but the multi-pass cases (CFG, contrastive) need a second forward and state, which a sampler chain on one logits row cannot host. |
| N2 | tap and edit | inside H6/H11, between layers | hidden state [rows x d] at layer L -> same shape. Hosts steering, probes, lens readouts, TTT writes, activation sparsity. |
| N3 | depth program | H6 lowering + H11 step | per row, which layers run and how many times (skip, early exit, loop, mixture-of-depths). Lowering emits a static graph today; this makes depth a per-step decision. |
| N4 | adapter bind | between H5 bind and H11 step | per-request low-rank delta selected per row in a batch. |
| N5 | compress | before H3 tokenize / H9 assemble | text or ids -> shorter text or ids (prompt compression). |
| N6 | route | inside H6, MoE layers | router scores -> chosen experts (override, skip, substitute-on-miss, prefetch hint). Distinct from H8 residency. |
| N7 | codec on seal | H13 | rows -> bytes at seal and bytes -> rows at read (quantization, low-rank, cross-layer). H13 names seal/evict/demote but not the codec. |
| N8 | import/export | H9 / H13 boundary | cache blocks or hidden states crossing an instance, model or agent boundary (disaggregation, cartridges, latent hand-off). |
| N9 | position | H6/H12 | position-id and RoPE scaling per row (YaRN, deferred positional encoding for position-independent caching). |
| N10 | determinism mode | H7 | kernel-selection constraint (batch-invariant reductions). Not a technique; a property H7 must be able to demand. |

Gaps between existing stages (not new rows, but missing edges):
- H2 -> H3: the template is an executable program (see C41); the hook output must be treated as
  untrusted data by H3.
- H11 -> H15: reasoning-budget control and hidden-state early stop need H11 to emit a stop
  proposal, not only a token entry.
- H14 -> H17: a verifier needs N samples from one state; H14 returns one entry. Best-of-N and
  self-consistency need H11 to fork.
- H5 -> H6: a merge or per-layer precision choice changes the descriptor the lowerer sees; today
  H5 is name resolution only.

## the table

Class: ACTIVE or THIN. Stage: existing H# or NEW (N# above). "Exposes" is what the hook must take
and return.

| # | candidate hook | stage | what the hook must expose | class | sources (arXiv id, month; mechanism) |
|---|---|---|---|---|---|
| C1 | activation steering / representation engineering | N2 | in: layer id, hidden rows, per-row intervention params; out: edited hidden rows | ACTIVE | 2602.08169 (2026-02) rotate activations along a geodesic with a confidence gate; 2510.04309 (2025-10) per-layer PID-controlled steering vector; 2605.11093 (2026-05, opened) DMI-Lib, asynchronous access to model-internal state at 0.4-6.8% offline overhead |
| C2 | per-request steering inside a continuous batch, cache key includes the intervention | N2 + H9 | in: per-row index into a vector table; prefix-cache key must bind the intervention id | THIN | 2509.25175 (2025-09) EasySteer, vLLM-integrated, 71-84% of baseline throughput with multiple vectors. Only one batched-serving source found. No source found on prefix-cache coherence under steering. |
| C3 | layer skip, early exit, mixture-of-depths | N3 | in: hidden row, layer index; out: continue, skip, or exit with readout | ACTIVE | 2603.21365 (2026-03) TIDE per-token early exit via small routers; 2510.13876 (2025-10) GateSkip residual gates; 2606.06574 (2026-06) predicts a per-input program that skips or repeats layers; 2507.10524 (2025-07) Mixture-of-Recursions |
| C4 | recurrent depth / looped models, depth-adaptive batching | N3 | in: loop counter, halting probability; out: exit or iterate; scheduler must batch rows at different depths | ACTIVE | 2510.25741 (2025-10) Ouro, shared stack up to 4 iterations with learned halting; 2502.05171 (2025-02) Huginn recurrent depth; 2608.09444 (2026-08) continuous depth batching for looped models, 1.5-1.9x offline throughput on Ouro 1.4B and Huginn 3.5B |
| C5 | multi-LoRA adapter composition and hot-swap | N4 | in: per-row adapter id, scale; out: base output + low-rank delta; adapter residency (host/device) | ACTIVE | 2311.03285 S-LoRA and 2310.18547 Punica (incumbents, 2023); 2604.07173 (2026-04) disaggregated multi-LoRA serving; 2608.03579 (2026-08) cuts PCIe swap and VRAM up to 16x; 2511.07129 (2025-11) per-instance dynamic LoRA selection and merging |
| C6 | KV quantization | N7 | in: sealed rows; out: packed bytes + dequant-on-read contract; per-channel vs per-token axis | ACTIVE | 2402.02750 KIVI (2024 incumbent); 2506.18879 (2025-06) CommVQ 87.5% size cut at 2-bit; 2511.18643 (2025-11) Kitty dynamic channel-wise precision boost; 2608.07915 (2026-08) beyond the 2-bit cliff |
| C7 | KV eviction | H13 | in: attention statistics per row, budget per layer/head; out: keep set | ACTIVE | 2506.17121 (2025-06) chunked eviction during prefill, KV-footprint metric; 2504.14051 (2025-04) CAOTE uses value vectors in the score; 2603.10899 (2026-03) LookaheadKV predicts future attention; 2605.08840 (2026-05) layer-wise output reconstruction |
| C8 | KV merging and cross-layer sharing | N7 / H13 | in: rows from adjacent layers or tokens; out: merged rows + reconstruction map | ACTIVE | 2503.18893 (2025-03) xKV cross-layer SVD, 8x at <=3% loss; 2508.16134 (2025-08) CommonKV training-free; 2604.13556 (2026-04) YOCO++ |
| C9 | KV retrieval attention (top-k over full cache) | H12 | in: query rows, summaries per block; out: read set of block ids | ACTIVE | 2508.08256 (2025-08) FIER token-level retrieval from 1-bit keys; 2505.13109 (2025-05) FreeKV; 2603.14224 (2026-03) self-indexing KV cache; 2607.27692 (2026-07) top-k reuse across steps |
| C10 | attention sinks (pin, learned sink logit) | H12 / H13 | in: sink set per layer/head; out: pinned rows; plus a per-head learned sink logit in the softmax | ACTIVE | 2309.17453 StreamingLLM (2023 incumbent); 2510.06477 (2025-10) sinks and compression valleys, describes gpt-oss per-head learnable sink logit; 2604.10098 (2026-04) survey; 2603.11487 (2026-03) sinks necessary in softmax transformers |
| C11 | grammar / structured-output constrained decoding | N1 | in: automaton state per row, vocab; out: token mask; state advance on commit | ACTIVE | 2411.15100 XGrammar (MLSys); 2502.05111 (2025-02) flexible and efficient grammar-constrained decoding; 2608.03065 (2026-08) parser stack classification; 2510.17376 (2025-10) AdapTrack on intent distortion; 2602.00612 (2026-02) for diffusion LLMs |
| C12 | classifier-free guidance and contrastive decoding | N1 (multi-pass) | in: two or more logits rows from forks of one state; out: combined row | ACTIVE | 2306.17806 CFG for LMs (2023 incumbent); 2505.23657 (2025-05, EMNLP 2025) active layer-contrastive decoding picks when to contrast; 2608.08082 (2026-08) CFG in masked diffusion LMs. Autoregressive-LLM CFG sources since 2025: only the first of these, so this is the weakest ACTIVE row. |
| C13 | text watermarking | H14 / N1 | in: key, context window hash, logits row; out: biased or tournament-sampled entry; detector reads ids offline | ACTIVE | 2603.03410 (2026-03) theory of SynthID-Text tournament sampling; 2602.01752 (2026-02) WorldCup multi-bit; 2605.12456 (2026-05) TextSeal; 2607.16010 (2026-07) paraphrase defeats detection for 98.3% of detected texts |
| C14 | token healing / prompt-boundary correction | H3 | in: prompt ids, tokenizer; out: backed-up prefix + constraint on first generated token | THIN | 2412.03719 (2024-12, revised 2025-06) exact and approximate algorithms for the prompt boundary problem; 2403.06988 (2024) guiding LLMs, non-invasive constrained generation. One 2025 revision; no 2026 source. |
| C15 | dynamic and byte-level tokenization | H3 / H16 | in: byte stream; out: variable-size patches or session-local vocabulary entries; embeddings computed at runtime | ACTIVE | 2412.09871 BLT (ACL 2025) entropy-segmented patches; 2605.08044 (2026-05) Fast BLT, block diffusion over bytes; 2506.01084 (2025-06) zip2zip, LZW hypertokens at inference, 20-60% shorter sequences |
| C16 | prefill/decode disaggregation | H8 / H10 / N8 | in: sealed cache blocks + target instance; out: transfer completion; ordering with decode admission | ACTIVE | 2407.00079 Mooncake (2024 incumbent); 2603.13358 (2026-03) PPD disaggregation for multi-turn; 2606.08635 (2026-06) per-token mixed-precision KV transfer; 2508.01989 (2025-08) aggregate vs disaggregate depends on SLO |
| C17 | continuous batching, chunked prefill, preemption | H10 | in: queue, token budget, deadlines; out: next batch + chunk sizes; preempt points at operator boundaries | ACTIVE | 2609.07883 (2026-09) deadline-aware adaptive chunking; 2602.16603 (2026-02) FlowPrefill operator-boundary preemption; 2608.15171 (2026-08) P-PAS adaptive token budget; 2510.14392 (2025-10) FairBatching |
| C18 | logit-lens and tuned-lens readouts at serving time | N2 | in: hidden rows at chosen layers; out: vocab distribution or scalar probe; must not perturb the forward | THIN | 2303.08112 tuned lens (2023, v6 2025-11); 2507.17618 (2025-07) SimLens early exit on single-token decisions; 2602.00462 (2026-02) LatentLens finds tuned lens gives at most +1.8pp. Used as an analysis tool in the sources; no source found that runs a lens inside a serving loop. |
| C19 | guardrail classifiers and probes in settle | H17 / H15 | in: streamed text or hidden-state rows; out: pass, hold, or stop with reason | ACTIVE | 2604.03962 (2026-04) StreamGuard forecasting for streaming moderation; 2606.10487 (2026-06) hidden-state probes inside the decode loop; 2603.02219 (2026-03) NExT-Guard training-free; 2510.14276 Qwen3Guard-Stream (2025-10) |
| C20 | self-consistency and verifier loops | H17 + fork in H11 | in: N completions from one state; out: selected answer; needs cheap fork of cache | ACTIVE | 2502.20379 (2025-02) multi-agent verification; 2507.18122 (2025-07) prefix-confidence without a verifier; 2603.03417 (2026-03) multi-sequence verifiers; 2608.07424 (2026-08) CoBa compute-balanced routing |
| C21 | multi-token prediction heads | H11 | in: last hidden row; out: K draft tokens with tree structure; verify in one pass | ACTIVE | 2401.10774 Medusa and 2503.01840 EAGLE-3 (incumbents); 2509.18362 (2025-09) FastMTP; 2507.11851 (2025-07) LLM knows the future, sampler head; 2603.17942 (2026-03) training-free MTP |
| C22 | diffusion / block-diffusion decoding | H11 (non-autoregressive entry) | in: partially masked block; out: committed positions per iteration; block-level and sub-block caches | ACTIVE | 2509.26328 (2025-09) Fast-dLLM v2, 2.5x; 2505.22618 Fast-dLLM v1 (ICLR 2026); 2604.15750 (2026-04) DepCap adaptive block size; 2607.04206 (2026-07) serving diffusion LLMs on the AR stack |
| C23 | model merging at load | H4 / H5 | in: N checkpoints + merge recipe; out: one bound weight set | THIN | 2603.09938 (2026-03) survey; 2505.10833 (2025-05) MergeBench; 2607.11997 (2026-07) merging empirical study. All offline. The request-level work found merges adapters, not base weights: 2511.07129. No source on merge-at-bind inside an engine. |
| C24 | per-layer mixed precision at bind | H5 | in: per-layer sensitivity, memory budget; out: codec choice per tensor | ACTIVE | 2607.17733 (2026-07) MXSens; 2608.24945 (2026-08) Fisher-based allocation; 2604.13440 (2026-04) forward-only KL sensitivity; 2608.28003 (2026-08) measures quantize/dequantize overhead in TensorRT-LLM |
| C25 | reasoning-budget control (force or forbid end-of-thinking) | H11 / H15 | in: think-token count, delimiter ids; out: suppress, append, or force text into the stream | ACTIVE | 2501.19393 (2025-01) s1 budget forcing; 2604.10739 (2026-04) overthinking under budget; 2510.06557 (2025-10) budget forcing worse than long CoT on R1-distill |
| C26 | position-independent cache assemble | H9 / N9 | in: chunk ids, target positions; out: re-rotated cached rows + recompute set | ACTIVE | 2410.15332 EPIC (2024); 2606.04302 (2026-06) deferred positional encoding; 2607.28069 (2026-07) SemPIC; 2609.10266 (2026-09) KVShareArena across contexts and checkpoints; CacheBlend via 2510.09665 (arXiv id for CacheBlend itself not seen: unverified) |
| C27 | learned cache artifacts (cartridges) | H9 / N8 | in: artifact id; out: preloaded rows; composition of several | ACTIVE | 2506.06266 (2025-06) Cartridges, self-study, composable at inference; 2508.17032 (2025-08) keys as shareable routers; 2606.07878 (2026-06) Still, amortized KV compaction in one forward pass |
| C28 | sleep-time precompute | H10 | in: idle slot, context; out: rewritten context or warmed cache | ACTIVE | 2504.13171 (2025-04) sleep-time compute; 2602.15156 (2026-02) Panini, structured memory |
| C29 | sparse decode with periodic dense rectification | H11 + H12 | in: step counter; out: sparse read set, or dense refresh that rewrites rows | ACTIVE | 2506.04108 (2025-06) Rectified Sparse Attention, up to 2.42x at 256K; 2506.08889 (2025-06) SeerAttention-R; 2601.17702 (2026-01) S3-Attention |
| C30 | conformal cascade | H17 | in: answer + calibrated score; out: accept, or escalate to next tier | ACTIVE | 2607.25018 (2026-07) conformal cascade, set-size deferral; 2604.23577 (2026-04) RouteNLP; 2510.17543 (2025-10) edge-cloud conformal alignment |
| C31 | action speculation over non-token entries | H11 generic entry | in: state; out: speculative action + rollback or commit; privacy contract for side effects | ACTIVE | 2510.04371 (2025-10) Speculative Actions; 2607.03333 (2026-07) SPORK; 2608.00881 (2026-08) AOSpec for stateful tools; 2606.02483 (2026-06) ghost tool calls leak intent |
| C32 | prompt compression | N5 | in: text, budget, query; out: shorter text | ACTIVE | 2310.06839 LongLLMLingua (2023); 2505.00019 (2025-05) empirical study, compression raises hallucination; 2603.19733 (2026-03) PoC; 2606.09659 (2026-06) end-to-end compression at scale |
| C33 | RoPE scaling and context extension | N9 | in: sequence length, per-dimension frequency table; out: position transform | ACTIVE | 2309.00071 YaRN (2023); 2502.20082 (2025-02) LongRoPE2; 2510.00028 (2025-10) RoPE scaling in quantized LLMs |
| C34 | deterministic (batch-invariant) mode | N10 / H7 | in: determinism flag; out: kernel plan with fixed reduction order | ACTIVE | 2506.09501 (2025-06) batch-invariant kernels; 2511.17826 (2025-11) across tensor-parallel sizes; 2601.17768 (2026-01) LLM-42 verify-and-rollback; 2608.14376 (2026-08) CoRun padding |
| C35 | test-time training / fast-weight writes | N2 + H11 commit | in: chunk of hidden rows; out: updated per-request weights (request-owned state) | ACTIVE | 2505.23884 (2025-05) LaCT large-chunk TTT; 2604.06169 (2026-04) In-Place TTT on MLP projection; 2606.21803 (2026-06) closed-form write; 2605.28053 (2026-05) RW-TTT batched serving of request-owned state |
| C36 | MoE routing override, skip, prefetch | N6 | in: router scores, resident set; out: chosen experts; hint to H8 | ACTIVE | 2603.19289 (2026-03) execute prefetched experts instead of missing; 2511.02237 (2025-11) batch-aware expert routing; 2610.01950 (2026-10) coordinated offload and residency; 2607.24787 (2026-07) SpecPrefetch |
| C37 | activation sparsity gate | N2 / H7 | in: hidden row, per-layer threshold; out: sparse row + kernel that skips weight columns | ACTIVE | 2408.14690 TEAL (ICLR 2025), 40-50% sparsity, up to 1.8x decode; 2505.14884 (2025-05) Polar Sparsity for batched serving; 2411.12692 (2024-11) SparseInfer |
| C38 | latent hand-off between models or agents | N8 | in: hidden rows or KV rows from sender; out: receiver cache entries (projected if models differ) | ACTIVE | 2606.05711 (2026-06) survey; 2605.22786 (2026-05) LCGuard; 2606.28958 (2026-06) integrity of relayed KV; 2608.04893 (2026-08) audit of whether relayed caches carry information |
| C39 | cross-tokenizer drafting | H3 x H11 | in: draft ids; out: string-level re-tokenisation into target vocabulary and back | ACTIVE | 2507.02659 (2025-07) OmniDraft cross-vocabulary drafter; 2506.06607 (2025-06) tokenizer transplantation; 2604.16368 (2026-04) string-level exact matching in MLX-LM |
| C40 | sampler truncation chain (min-p, top-n sigma, p-less) | H14 | in: logits row, temperature, ordering of scale vs truncate; out: entry | ACTIVE | 2407.01082 min-p (2024); 2609.15476 (2026-09) temperature fragility, order of scaling and truncation matters; 2606.13982 (2026-06) adaptive nucleus truncation |
| C41 | chat template as an executable program | H2 | in: messages; out: prompt text; must be pure and sandboxable, and its source must be an audited input | ACTIVE | 2602.04653 (2026-02, v4 2026-05) template backdoors, 18 models, 4 engines. The sources are about attack surface, not a technique. |
| C42 | streaming detokenize, UTF-8 hold, stop-string hold-back | H16 / H15 | in: id deltas, stop set; out: text delta with minimal hold-back; text and id deltas must stay consistent on stop | THIN | 2511.05578 (2025-11) byte-level tokenizers can emit ill-formed UTF-8, gives an incremental algorithm. The rest I found is engine issue trackers and blog posts, not primary papers. |
| C43 | model described as data, run by one generic engine | H4 / H6 | in: descriptor; out: lowered graph | THIN | No primary source. Closest listing: 2606.31093 (2026-06, a framework paper whose title the search did not return, so unverified) with a declarative graph DSL; 2608.23841 (2026-08) co-designs engine and architecture, not config-driven. |
| C44 | model-internal observability tap | N2 | in: layer ids, row selectors; out: asynchronous copy to a host ring | ACTIVE | 2605.11093 (2026-05, opened) DMI-Lib, ring buffer GPU to CPU; 2604.28129 (2026-04) residual-stream probing for multi-turn attacks |
| C45 | hidden-state probe as stop signal for reasoning | H15 | in: hidden row at cue tokens; out: stop proposal with calibrated risk | ACTIVE | 2512.05325 (2025-12) LYNX probe + split conformal; 2505.18404 (2025-05) thought calibration; 2604.04930 (2026-04) confidence dynamics, no probe |

## counts

| class | rows | ids |
|---|---|---|
| ACTIVE | 39 | all rows not listed below |
| THIN | 6 | C2, C14, C18, C23, C42, C43 |
| total candidates | 45 | 24 from the seed list, 21 added |

The seed list was 24 items; the 24 seed items are C1, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12,
C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24 and C2 (steering in a batch, split out
of the first seed item). Rows C25-C45 were found by the ablation.

## ACTIVE, must be hooks (owner rule: anything actively researched gets a hook)

Grouped by the hook each one needs. A group with no existing H# needs a new stage from the
section above.

- **N1 shape (logits row -> logits row, possibly multi-pass):** C11 grammar, C12 CFG and
  contrastive, C13 watermark, C40 truncation chain. Constraint the sources impose: C12 needs a
  fork of one state into two forward passes; C11 needs per-row automaton state advanced on
  commit; C13 needs a context-window hash.
- **N2 tap and edit:** C1 steering, C35 fast-weight writes, C37 activation sparsity, C44
  observability tap. C45 and C19 (probe variants) read from this tap.
- **N3 depth program:** C3 skip/exit/MoD, C4 looped depth. C4 additionally requires the H10
  scheduler to batch rows at different depths (2608.09444).
- **N4 adapter bind:** C5. Requires per-row adapter id in the batch and a residency policy.
- **N7 codec on seal:** C6 quantization, C8 merging and cross-layer sharing. C7 eviction and C10
  sink pinning share the H13 seal decision.
- **H12 read:** C9 retrieval attention, C29 rectified sparse (also H11), C10 sinks.
- **H9 assemble and N8 import/export:** C26 position-independent assemble, C27 cartridges, C38
  latent hand-off, C16 disaggregation transfer.
- **H10 schedule:** C17 batching and preemption, C28 sleep-time.
- **H11 step:** C21 MTP, C22 diffusion, C25 reasoning budget, C31 action speculation, C20
  verifier fork.
- **H14 sample:** C13, C40.
- **H15/H17 settle:** C19 guardrails, C20 verifier, C30 conformal cascade, C45 reasoning probe.
- **H3 tokenize:** C15 dynamic and byte-level tokenization, C39 cross-tokenizer drafting.
- **H5 bind:** C24 per-layer precision.
- **N5 compress:** C32. **N6 route:** C36. **N9 position:** C33 (and C26). **N10 determinism:**
  C34. **H2 template:** C41 (hook must be pure; the source is data that can be hostile).

Cross-cutting requirement visible in the sources (stated as plausible, not proven): C2, C26,
C27, C35 and C38 all alter what a cached row means. The H9 assemble key and the H13 seal key
must therefore bind to the producing intervention, adapter, position scheme and codec, not to
token ids alone. 2602.04653 and 2606.28958 are separate reports of what goes wrong when the
producer of state is not bound to the consumer.

## THIN, possible novel ground

For each: the exact searches run (all WebSearch, 2026-10-04) and what they returned. "Not found"
here means these queries returned nothing relevant in the top 10 listings, not that nothing
exists.

- **C2 per-request steering in a continuous batch, prefix-cache key bound to the intervention.**
  - "activation steering representation engineering 2026 arXiv serving inference system"
  - "per-request batched activation steering vectors serving engine vLLM hooks intervention
    continuous batching" (returned only general continuous-batching pages)
  - "steering vectors multi-tenant serving throughput overhead system paper steering LLM
    inference engine" (returned EasySteer 2509.25175 and nothing on per-request tenancy)
  - Found: one vLLM-integrated steering framework. Not found: any source on cache-key coherence
    when two requests share a prefix but differ in steering.
- **C14 token healing.**
  - "token healing prompt boundary tokenization bias constrained generation arXiv"
  - Found: 2412.03719 (revised 2025-06), 2403.06988 (2024), 2402.01035 (2024), a
    transformers issue. Not found: any 2026 source or any treatment inside a batched engine.
- **C18 lens readouts inside a serving loop.**
  - "tuned lens logit lens readout 2025 2026 arXiv latent reasoning interpretability"
  - "tuned lens early exit readout intermediate layer decoding hook inference system speculative
    layer skip draft 2025 2026" (the search tool itself reported it found no paper combining
    tuned-lens readout, hook-based inference systems and speculative layer-skip drafting)
  - Found: analysis uses and early-exit uses (2507.17618, 2602.00462, 2609.09902 listing only).
    Not found: a lens as a first-class serving hook, or a lens used as a draft head.
- **C23 model merging at bind.**
  - "model merging at load time task arithmetic TIES LLM merge 2025 2026 arXiv" (the tool
    reported nothing about load-time merging)
  - "runtime weight merging at inference serving dynamic merge LoRA task vectors on-the-fly
    request-level model merging system" (returned adapter-level merging only, 2511.07129,
    2602.21222)
  - Found: offline merging surveys and benchmarks, adapter-level request merging. Not found:
    base-weight merge performed by the engine at bind, or the cost of re-binding.
- **C42 streaming detokenize and stop-string hold-back.**
  - "incremental streaming detokenization partial UTF-8 codepoint stop string matching streaming
    LLM serving"
  - "byte latent transformer dynamic tokenization patches 2025 arXiv tokenizer-free" (surfaced
    2511.05578)
  - Found: one 2025 paper (2511.05578); remaining hits were a GitHub project (stopedge), vLLM
    issues and blog posts, which I do not count as primary sources. Not found: a paper on
    minimal hold-back for stop sequences.
- **C43 a model as data for one generic engine.**
  - "declarative model architecture description single generic inference engine config-driven
    transformer variants arXiv" (the tool reported no paper matching exactly)
  - "programmable LLM inference engine extensibility hooks DSL custom decoding policy KV policy
    arXiv 2025 2026" (returned Leyline 2606.01065 listing only, microserving 2412.12488 from
    2024, SGLang survey 2505.01658)
  - Found: nothing that makes the whole forward pass data. Leyline (KV directives for agentic
    inference) appeared only as a reference list and was not read; unverified.

Observation, not a finding: five of the six THIN rows sit at the engine's seams (cache key,
tokenizer boundary, stream boundary, bind, descriptor), where the literature is about models
rather than about the serving system around them.

## what was not done

- No source was read beyond the search snippet, except 2605.11093. Mechanism lines are summaries.
- No hook was checked against the code anchors in SPEC.md; the new stages N1-N10 are proposals
  from the ablation, not read from `proxima-tensor`.
- The paper test in SPEC.md has no sketch for N1-N10. Candidate sketches that would stress them:
  steering vector (N2), grammar mask (N1), looped depth (N3), multi-LoRA (N4).
- Counts are of my own classification and depend on the threshold "at least one 2025-2026
  primary source listed". C12 and C41 are the two ACTIVE rows nearest to that threshold.
