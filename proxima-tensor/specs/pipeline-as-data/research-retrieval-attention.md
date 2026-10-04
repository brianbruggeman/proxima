---
status: draft (research input to SPEC.md; not audited)
date: 2026-10-04
base: proxima-windows main 4b4be6cf (design files read from the fsm-techniques checkout, read-only)
---

# research classification: certified retrieval attention over a bounded cache

Format and rules follow `research.md` (OPEN-1 entry). Provenance as in `research.md`: "opened" = fetched
or PDF text read this session; the PDF text lines are cited where I read the body; "snippet" = search-result
listing only. Arithmetic is labelled `arithmetic, not measured`.

## class

**Active research.** The soundness mathematics is engineering (a proof exists, the oracle is full
attention). The open part is empirical and systems-shaped: how often a box-bound certificate fires on real
long-context LLM decoding, with the unread blocks held in an outside tier. Nobody has published that
measurement. Not open research: a deterministic certificate with per-block upper bounds is already proven.

## precise question

For one decode step, one layer, one query head h, with sealed blocks b = 1..M, per-block per-dimension key
bounds (lo_b, hi_b) and a per-block value bound, and a read set R of blocks fetched through the generic read
hook: let U_b(q) = sum_d max(q_d lo_bd, q_d hi_bd) / sqrt(d) + rounding slack. Let m = max over read keys of
the exact logit, S_R = sum over read tokens of exp(s - m), T = sum over unread b of n_b exp(U_b - m).
Is tau = T / (S_R + T) a sound upper bound on the unread softmax mass, and does
||o_full - o_R||_2 <= tau * (value diameter bound) hold, so that when tau * Vdiam <= epsilon the step is
certified and otherwise the step reads every block (fallback)? And at what rate does the certificate fire on
gemma4 E2B, openchat and qwen3 contexts, per layer and head, at a stated read budget?

## sources opened

Soundness side (all opened this session):
- Quest, arXiv 2406.10774 (https://arxiv.org/abs/2406.10774), PDF text lines ~291-301: per-page
  channel-wise min m_i and max M_i of keys; U_i = max(Q_i m_i, Q_i M_i); the sum "is the upper bound of
  attention weights across all Key vectors on this page". Selection is top-K pages by that bound. No
  statement of output error. Abstract: up to 7.03x self-attention speedup, 2.23x end-to-end (line 33-35).
  Shows: the bound is sound on the logit; it is used to select, not to certify.
- Tzachristas et al., "A Mathematical Theory of Top-k Sparse Attention via Total Variation Distance",
  arXiv 2512.07647 (https://arxiv.org/abs/2512.07647), PDF text. Shows: TV(P, P_k) equals the discarded
  tail mass (abstract); head-tail identity ||Attn - Attn_k||_2 = tau ||mu_tail - mu_head||_2 (Theorem 5.2,
  line ~735); output bound <= 2C TV when ||v_j|| <= C (Proposition 5.1, line ~718) and <= tau diam_{H,T}
  (abstract). MC-Search (Algorithm 2, lines 1121-1175) keys partitioned into cells, "U_j(q) an upper bound on
  q.k over that cell", tau_hat_k = (S_tail_known + sum_j e^{U_j}) / (S_head_known + S_tail_known + ...),
  Theorem 7.2: if it terminates the certificate TV <= epsilon is proven. Cells are centre-radius balls
  (U_j = q.c_j + ||q|| r_j, line 1080), not Quest boxes. Blockwise lower/upper bound version with
  deterministic guarantee (eq. 4.4, lines 669-682). Experiments: bert-base-uncased attention maps,
  n = 128/256, and synthetic logits; at epsilon = 0.01, k_mc ~ 58.8 (n=128) and ~107.8 (n=256), speedup
  n/k_mc about 2.2x and 2.4x (lines 1507-1511), counted in scored keys. No fallback tier, no rounding
  analysis (grep for round/quantiz returned no match), no KV-cache system. Closest source.
- Runtime-Certified Bounded-Error Quantized Attention, arXiv 2605.20868
  (https://arxiv.org/abs/2605.20868), PDF text. Shows: a per-head per-step certified mass bound with a
  fallback ladder ending at dense attention (Theorem 4.4, lines 562-650). The bound covers quantization
  error of INT8 scores, not unread blocks: the paper states it addresses quantization, "not token eviction"
  (line 947). Absorbs rounding by substituting e^{3 Delta} for e^{2 Delta} (fetched summary, Theorem 4.4
  note). Output error bound E_key <= 2 V_max e^{2 Delta}(1 - tau_cov)(e^{2 Delta} - 1) (line 640) uses a
  value bound V_max. Closest source for "bound plus fallback plus rounding slack".
- Error Certificates for KV-Cache Eviction via Randomized Design, arXiv 2607.21475
  (https://arxiv.org/abs/2607.21475), PDF text lines 1-150. Shows: Theorem 1 (line 191): for a deterministic
  value-blind eviction nothing computable from the retained set identifies the eviction error, because evicted
  values are unconstrained. Their fix is Poisson sampling plus an empirical-Bernstein certificate: coverage
  96.9-97.7% of realized errors in 12,096 replay cells (line 290), so probabilistic, not a deterministic
  bound. Shows why the value bound is a required extra input.
- Tail-Mass Law for Softmax Attention (https://zenodo.org/records/20770653), opened (fetch summary of the
  record page; not a PDF body): tail mass <= (n-k)/k e^{-beta Delta_k}, output error <= epsilon diam(V),
  "certifiable regime is rare everywhere, peaking near 3.6% and vanishing at strict tolerances", over eight
  model families, 256-16,384 tokens, 1.5e8 attention rows, bound overestimates tail mass by a median 80-200x.
  This is a gap-based top-k bound, not a box bound. It is the strongest counter-evidence on fire rate.
- CertKV, arXiv 2608.21541 (https://arxiv.org/html/2608.21541), opened: exact slot accounting is a
  posteriori, needs the full cache before eviction; no unread-block handling, no fallback. Not a certificate
  for the setting.
- PrHS, arXiv 2602.08329 (https://arxiv.org/abs/2602.08329), PDF text lines 31-39, 191-282: a-priori
  routing mutual-information certificate that is a function of the average dropped mass; a statement about
  an information quantity, "not a direct guarantee of downstream task accuracy" (line ~282).
- BLASST, arXiv 2512.12087 (https://arxiv.org/abs/2512.12087), PDF text lines ~1193: skips blocks when
  local max minus running max is below ln lambda. A threshold criterion with a mass argument, no
  output-error certificate stated; prefill-oriented.

Selection-only sources (the "SELECT" half of the distinction), snippet-level unless noted:
- InfLLM, arXiv 2402.04617 (https://arxiv.org/abs/2402.04617): opened; abstract only, representative
  tokens per block, lookup of "token-relevant units"; no error bound stated at abstract level.
- MagicPIG, arXiv 2410.16179 (https://arxiv.org/abs/2410.16179): opened abstract: "sampling with theoretical
  guarantees" via LSH, estimator not a deterministic certificate; abstract does not state variance bounds.
- RetroInfer, arXiv 2505.02922 (https://arxiv.org/abs/2505.02922): snippet of a search listing and a third
  party review: "accuracy-bound attention estimation" using cluster centroids weighted by cluster size for the
  estimation zone. Not opened beyond that, so whether its "bound" is deterministic is unmeasured by me.
- Twilight, arXiv 2502.02770 (https://arxiv.org/abs/2502.02770): snippet: top-p adaptive budget on top of a
  base selector; 15.4x self-attention, 3.9x end-to-end; a 2026 offloading paper (arXiv 2604.08426, snippet)
  reports little gain from top-p over top-k under its protocol. Target is mass coverage on the approximate
  attention distribution, no sound bound on the pruned part stated in the snippet.
- ShadowKV, arXiv 2410.21465 (https://arxiv.org/abs/2410.21465): snippet: chunk-mean landmarks plus about
  0.3% outlier chunks; a mean is not an upper bound, so no certificate possible from it.
- Landmark attention: searched for it in the three queries, not found in any result, not opened.
- Others in the listings, snippet only: BFLA 2605.12193, FlashPrefill V2 2608.19758 (surrogate term for pruned
  blocks), EntmaxKV 2605.21649 (exact when selection contains the support), WitCert 2607.28699,
  Vertex-Softmax 2605.10974 (interval bounds on softmax for verification).

Searches run: "certified error bound sparse attention KV cache retrieval upper bound of omitted softmax mass
guarantee fallback"; "block-level min max key bound upper bound attention logit omitted blocks softmax tail
mass epsilon exact fallback long-context KV retrieval 2026"; "certified OR provable top-k attention retrieval
guarantee omitted softmax mass bound block bounding box keys sparse decoding".

## does anything certify

- Certifies the output of sparse attention over unread blocks with a sound deterministic bound: arXiv
  2512.07647 (Theorem 7.2 plus Theorem 5.2), on cells with centre-radius bounds, in an offline evaluation.
  Its algorithm iteratively scores keys; it has no memory tier and no dense fallback step.
- Certifies with a fallback to dense: arXiv 2605.20868, for quantized-score error, not for unread blocks.
- Probabilistic certificate: arXiv 2607.21475, 97% coverage, not sound.
- Quest, InfLLM, ShadowKV, MagicPIG (estimator), Twilight use a bound or summary to select or estimate; none
  of their sources I opened states an output-error guarantee with a fallback.
- Not found: a paper that combines Quest-style per-dimension key min/max boxes, a mass certificate, a value
  bound, and a dense fallback on a tiered KV cache, with measured fire rates on a long-context LLM. Search
  returned no such paper (three queries above); an unverified negative beyond that.

## design check: what the fsm-techniques draft and sketch 8 already carry

- `fsm-techniques/SPEC.md:203` `seal.summaries` subset of {content_key, key_minmax}; `SPEC.md:362` reachability
  row `attention.read = block` requires `key_minmax`; `tasks/04-seal-and-reserve.md:159` per-block per-dimension
  min and max of the concatenated even and odd planes (`block_key_min`, `block_key_max`).
- `sketches/08-rectified-sparse-attention.md:90-118`: `fold_key_bounds(rows, width, low, high)` folds min/max
  per dimension; `block_score(query, low, high) = sum_d max(q_d high_d, q_d low_d)`, the Quest bound.
  Worked example scores blocks 0..3 = 2, 1, 7, 3 (lines 130-131).
- `SPEC.md:231-243` already has `sampled{..., epsilon, delta}` and R8c "relative error <= epsilon with
  probability >= 1 - delta" (MagicPIG-shaped, probabilistic). No deterministic read variant exists.

Extra the certificate needs, each one checked against those lines:
1. The exact logit is `block_score / sqrt(d)` plus the model's logit scale and soft-cap if any; the bound
   must be applied after the same transform the attention kernel applies. Not in the sketch.
2. A per-head bound. The draft pools the query over every head and both rotated planes (sketch line 11-12).
   A pooled score is not an upper bound on any single head's logit, so a certificate must use per-head
   `block_score`, or a sound upper bound on the pooled sum, and per-layer tau is a max over heads in a KV group.
   This contradicts the draft's selection design and is a real change, not a rename.
3. A value bound per block (for example max ||v|| or a value min/max box). Without it the output error is
   unbounded (the Theorem 1 argument of 2607.21475 applies to values). Not carried by `seal.summaries`.
4. A rounding slack term added to U_b (below).
5. A fallback action: if tau * Vdiam > epsilon, read every unread block through the hook. The draft's read
   variants are dense | block | sampled; this is a fourth variant or a wrapper over `block`.
6. Summaries must themselves stay resident. Per block, key bounds are 2d floats and a value bound 1 float,
   against 2*b*d floats of KV: (2d+1)/(2*b*d), about 6.3% at b = 16 and 0.39% at b = 256 (arithmetic, not
   measured). "Bounded memory" holds only if summaries are also tiered or boxed hierarchically; a box over
   child boxes is a sound parent bound and gives a tree with O(log M) probing.

## interaction with OPEN-1 (research.md, kernel-shape exactness)

- The certificate bounds the attention-output deviation per layer and head, in real arithmetic. OPEN-1 asks
  for a forward error bound in floating point that composes layer by layer into a logit margin. The two share one
  recursion: this epsilon enters OPEN-1's per-layer perturbation as an additional term, and the final margin
  test uses the sum. A certified-sparse step is therefore only certified at the argmax level if OPEN-1's
  machinery exists. Without it, the claim is limited to "attention output within epsilon in exact arithmetic".
- Rounding enters three places: (a) U_b is a sum of d products computed in f32; a sound bound adds a
  slack of order gamma_d * sum_d |q_d| max(|lo_d|, |hi_d|) to the logit, which multiplies the mass by e^{slack}
  (same device as 2605.20868's e^{3 Delta}); (b) the exact logits of read blocks and their online-softmax
  accumulation depend on the reduction order and block order; (c) the read set changes the row count, so the
  kernel shape changes, which is OPEN-1's exact subject (cached-attention form and cooperative reduce
  thresholds, `omega/src/msl/emit_and_classify.rs:1942`, `:2261`, as cited in research.md).
- OPEN-1 gets its per-kernel bound from a reduction tree; here the same tree bound is needed for the
  bound kernel itself. If OPEN-1 stays open, this item's soundness claim stays at real arithmetic plus an
  explicit slack parameter measured against a CPU f64 reference.

## why active, not open, not engineering

Not open: a proven deterministic mass certificate with unexplored-cell upper bounds exists
(2512.07647 Theorem 7.2), as does the head-tail output identity, and dense-fallback certification machinery
exists (2605.20868). Several groups published certificate work on KV caches in 2026
(2605.20868, 2607.21475, 2608.21541).
Not engineering: the question that decides whether this is useful has no published number. The one
direct evidence on fire rate is adverse (Tail-Mass Law: certifiable regime peaks near 3.6% for the gap-based
bound; bound looseness 80-200x), and Quest's box bound is known to be loose in high dimension (my reading of
the geometry, not measured: a box over 16 keys of dimension 128 bounds each coordinate separately, so the sum
of per-coordinate maxima exceeds any actual key's score). Also no source combines the four required parts
(box bounds, value bound, rounding slack, memory-tier fallback).

## benefit line

- Who: long-context decode where the KV cache exceeds the device tier and blocks arrive through the read
  hook (tier stores, an outside memory system); anyone who today uses Quest-style selection and cannot say
  what it cost on a given step.
- Mechanism: reads only the selected blocks when the certificate holds, so per-step KV traffic drops by the
  unread fraction; fallback bounds the damage to the already-paid dense cost plus the probe.
- Order of magnitude, from cited numbers: Quest reports up to 7.03x self-attention and 2.23x end-to-end for
  selection with no certificate (arXiv 2406.10774 abstract). MC-Search certified at epsilon = 0.01 scores
  2.2x-2.4x fewer keys on bert-base n = 128-256 (arXiv 2512.07647 lines 1507-1511); different model, short
  contexts, scored keys not read bytes. If the certificate fires on a fraction f of layer-head-steps and
  reads a fraction r of blocks when it fires, traffic is f*r + (1 - f) of dense (arithmetic, not measured on
  omega); with the 3.6% peak (adverse source) f*r + (1 - f) is above 0.96 for any r, so the benefit is
  only real if f is far above that.

## hypothesis (falsifiable)

For per-head `block_score` boxes at b = 16, a per-block max-value-norm bound, a read budget of at most
25% of sealed blocks chosen by `block_score`, and epsilon = 0.01 relative to full-attention output norm: (H1)
the certificate is sound, observed ||o_full - o_R|| <= tau * Vdiam + slack on every recorded layer-head-step;
(H2) it fires (tau * Vdiam <= epsilon without fallback) on at least 50% of layer-head-steps at 32k context on
gemma4 E2B, openchat and qwen3 on the vendored prompts.

## baseline

(a) dense attention (oracle, correctness); (b) Quest top-K at the same budget with no certificate (observed
error, no guarantee); (c) the Tail-Mass Law gap bound (arXiv, zenodo 20770653) computed on the same rows;
(d) MC-Search with ball cells (arXiv 2512.07647) on the same keys.

## measurement

Instrumentation first, no kernel change: dump per layer, head and decode step the exact logits for all
cached keys and the per-block `block_score` (f64), the true unread mass, the true output error against dense,
and the value-norm bound, on 256-token continuations of the three vendored prompts per checkpoint
(`proxima-model-interop/tests/fixtures/llama-parity/`) at context 8k, 16k, 32k (extend by repeating a real
document, not synthetic noise). Record per record: tau bound, true tail mass, ratio, certificate fires, observed
error vs bound. Report the full per-head distribution, not a mean. Controls that should fail: (i) replace U_b
by the block mean-key logit (ShadowKV-style landmark, not an upper bound): violations of H1 must appear; (ii)
shuffle per-block bounds across blocks: f must collapse. If (i) shows no violation, the harness is not measuring
the bound. Also run pooled-over-heads `block_score` (the draft's choice) to quantify how often it fails to
upper-bound a single head's logit.

## kill criterion

Any record with true tail mass above the computed bound, or true output error above tau * Vdiam plus rounding
slack (unsound; also kills any claim at argmax level until OPEN-1 closes); or f below 10% at 32k context
at the 25% budget (the certificate is too loose to save reads; Quest-like selection with a post-hoc check
is then the honest option, and the item reclassifies as engineering with a negative result); or summary
residency above 10% of KV bytes at b = 16 without a hierarchical box tree (the bounded-memory premise fails).
