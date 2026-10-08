# tiered MoE: three senses (research notes, 2026-10-08)

All rows from pages fetched this session; "unsourced" = not in what was read. Fetched via WebFetch summaries of arXiv abs/html pages (summaries, not full-text reads, unless noted).

## Table 1. Hierarchical routing, 3+ gating levels

| System | Levels | What each level selects | Size | Reported effect | Source |
|---|---|---|---|---|---|
| Hierarchical Mixtures of Experts (Jordan, Jacobs 1994) | tree-structured, nested gates (depth: unsourced here) | internal nodes are gates, leaves are experts; EM fit | unsourced | robot dynamics simulations (numbers unsourced) | [S1] |
| DeepSeek-V3 | 2 selection stages, not 3: per-token top-8 of 256 routed, constrained to at most M=4 nodes; plus 1 always-on shared expert | node set, then experts within | 671B total / 37B active; 58 MoE layers; expert hidden dim 2048 | training/serving numbers unsourced here; deploy: prefill EP32 (4 nodes), decode EP320 (40 nodes) | [S2] |
| PEER (Mixture of A Million Experts) | 2 sub-key sets, each sqrt(N) keys of dim d/2; full key = concatenation | top-k over each sub-key set, then combine | N=1024^2 (~1M) single-neuron experts, 8 heads, k=16 per head (128 active) | C4 ppl at 6e18 FLOPs: dense 23.84, MoE 21.41, PKM 21.92, PEER 20.63; at 2e19: 18.31/17.12/17.36/16.45. Top-k search O((sqrt(N)+k^2)d) | [S3] |
| Hi-MoE (2605.08292) | 2: inter-group balance, intra-group specialization | group, then expert in group; groups often device-aligned | unsourced | unsourced | [S4] |
| HoME (3D medical segmentation) | 2 MoE levels (+ slot assignment step = 3 steps) | local features, then global refinement | unsourced | vision model, not LLM | [S5] |
| HDMoLE (ASR) | 2: global router (pre-trained accent model) + per-layer local routers | accent/global then LoRA expert | unsourced | ASR, not LLM | [S6] |
| Mixtral | flat top-2 of 8, 1 level (the "two-level" premise is not supported by anything read) | experts | 45B total / 14B active (from HOBBIT table) | - | [S7] |
| 3-level LLM router | none found. A GitHub issue (placement / outer model / inner expert) is a proposal, not a paper | - | - | - | [S8] |

## Table 2. Heterogeneous expert tiers (3+ classes)

| System | Tiers | How routing picks | Numbers | Source |
|---|---|---|---|---|
| MoE++ | 4 classes: FFN, zero (discard), copy (skip layer), constant (replace by learned vector) | token gets variable number of FFN experts; gating residuals carry previous layer path | 1.1-2.1x expert forward throughput vs same-size vanilla MoE; params/benchmarks unsourced | [S9] |
| HMoE (Tencent) | experts of different sizes (count of size classes unsourced) | routing unsourced; loss encourages frequent activation of smaller experts | lower loss with fewer activated params (no figures on page) | [S10] |
| DeepSeek-V3 | 2: 1 shared (always on) + 256 routed (top-8) | shared always; routed by router, <=4 nodes | see Table 1; shared+redundant experts on 64 of 320 decode GPUs | [S2] |
| NPUMoE (Apple M2 Max/Ultra) | 3 static capacity tiers C(1)>C(2)>C(3), tau(e)=max{j | n_e<=C(j)}; example: top quartile gets 4x base capacity (other multipliers unsourced) | offline-calibration popularity (routed-token counts per expert per layer); dynamic routing unchanged | prefill latency 1.32x-5.55x vs CoreML/ANEMLL; energy 1.81x-7.37x | [S11] |
| Mixture of Depths | 2 per layer: processed vs skipped tokens | top-k tokens per layer, k fixed in advance (static graph) | iso-FLOP parity; up to 50% faster post-training sampling | [S12] |
| DraftExpert | shared + top-1 routed + 1 draft expert per layer (drafter footprint) | router top-1 for drafter; target experts prefetched | see Table 3 | [S13] |
| Expert-choice / token-choice hybrids | not researched this session: unsourced | | | |

## Table 3. Serving systems with 3+ placement tiers

| System | Tiers | Promotion / demotion rule | Signal | Reported | Source |
|---|---|---|---|---|---|
| NPUMoE | NPU (hot expert groups, attention, FFN static shapes) / CPU (router, top-k, scatter-gather, cold experts) / GPU idle | residency of hot grouped graphs by offline calibration; graphs >~1.2 GB on M2 Ultra fall to CPU | offline routed-token counts | prefill 1.32-5.55x; end-to-end up to 3.86x vs CoreML naive, 1.26x vs CoreML CPU, 1.19x vs ANEMLL; Qwen3-30B-A3B 0.572 s vs 1.139 s CoreML | [S11] |
| PowerInfer-2 (Snapdragon 8 Gen 3; dense-sparse FFN, incl. TurboSparse-Mixtral 47B) | NPU hot region / CPU cold region / UFS flash (GPU unused) | hot: LRU at cluster granularity; cold: LRU per neuron; evicted weights discarded; hot region grows with batch | neuron activation frequency | TurboSparse-Mixtral 47B 11.68 tok/s (19 GB avail), 9.96 at 50% FFN offload, 2.13 at 7 GB; cache hit 95% (Bamboo 7B), Mixtral miss 3.5% avg, P99 18.9% | [S14] |
| HOBBIT | GPU cache (FP16 + low precision copies) / CPU DRAM / SSD; precisions FP16->Int4 (4090), Int8->Int2 (Jetson) | per-token: rank by gate weight, cumulative score <=0.6 high precision, 0.6-0.9 low precision, >0.9 skip; cache score = LRU+LFU+LHU+FLD weighted | gate weights; next-layer predictor top-1 ~96% (next layer), ~90% (2-3 layers) | up to 9.93x decode (Phi-MoE, Jetson AGX Orin vs MoE-Infinity); 3.21x/3.29x vs MoE-Offloading on 4090 (Mixtral/Phi-MoE); Mixtral ~67% high / 30% low / 3% skipped; miss penalty -4.69..-8.68% vs LRU | [S15] |
| ProMoE | GPU (per-layer LRU cache) / host DRAM (PCIe4 23.9 GB/s measured) | predicted experts queued low priority; misses high priority with preemption | per-layer 2-layer MLP (~2M params) on layer input, run on CPU, ~200 us; accuracy ~84.7% | prefill 2.20x avg (3.21x max), decode 2.07x avg (5.02x max) vs offloading baselines; RTX 4090 | [S16] |
| MoE-Infinity | GPU / host DRAM (SSD not mentioned on page) | eviction: lowest (predicted likelihood x (1-layer/L)); prefetch from EAM cosine-match to history | expert activation matrix traces | DeepSeek-V2-Lite TPOT avg 155 ms vs vLLM 485, DeepSpeed 737, Mixtral-Offloading 1250, Ollama 2590 (A5000); 3.1-16.7x claim; hit rates unsourced | [S17] |
| Fiddler | CPU / GPU | run on GPU if resident; else copy only if est. CPU latency > GPU latency + transfer; else run on CPU | input size, latency model | 1.26x single batch, 1.30x long prefill (abstract), 11.57x beam search; Mixtral-8x7B 16-bit | [S18] |
| ExpertFlow | GPU / CPU | cache engine with routing-path prediction and error correction | T5-encoder predictor (7.21 MB), 73-87% batch accuracy; (device: GPU implied) | hit ratio up to 91.96%; 2-10x speed; GPU mem savings up to 93.72% (A40) | [S19] |
| DraftExpert | (a) CPU->RTX 4090; (b) Flash->Hexagon HTP NPU (Snapdragon 8 Elite) | routed experts offloaded, non-blocking prefetch of predicted verifier experts; misses on demand | draft-model prediction | decode TPS DS/CG 2.19->2.99, DS/MN 10.18->15.47, ML/CG 1.94->2.50, ML/MN 8.50->13.69; avg 1.45x; prefetch hit 86-88% | [S13] |
| Mixtral-offloading (Eliseev, Mazur) | tiers not on abstract page; numbers unsourced | unsourced | unsourced | unsourced | [S20] |

## Direct answer

Router / predictor on a different device than the experts:
- NPUMoE: yes. Router, top-k dispatch, scatter/gather run on CPU; hot expert FFN and attention run on NPU; cold experts on CPU [S11].
- ProMoE: predictor runs on CPU (~200 us) while experts compute on GPU [S16].
- DraftExpert: routers, shared experts, draft experts and attention are accelerator-resident; page does not state where the draft router runs [S13]. Not a split.
- ExpertFlow: predictor placement not stated; future work says it could move to CPU, implying GPU now [S19].
- NPU router with GPU experts: none found. A Hexagon PR notes quantization differences flip near-tied experts, relevant to cross-device routers [S21].
- "3 or 4 tier MoE": Table 1 found no 3-level gated LLM; DeepSeek-V3 is a 2-stage select + shared. Table 2 has NPUMoE (3 capacity tiers) and MoE++ (4 expert classes). Table 3 has 3-tier placements in HOBBIT (GPU/CPU/SSD), PowerInfer-2 (NPU/CPU/flash), NPUMoE (NPU/CPU/GPU-idle).

## Sources
- S1 https://direct.mit.edu/neco/article/6/2/181/5779/Hierarchical-Mixtures-of-Experts-and-the-EM (search result only)
- S2 https://arxiv.org/html/2412.19437v2 (DeepSeek-V3)
- S3 https://arxiv.org/html/2407.04153 (PEER)
- S4 https://arxiv.org/pdf/2605.08292 (search result only)
- S5 https://arxiv.org/pdf/2507.06363 (search result only)
- S6 https://arxiv.org/pdf/2409.19878 (search result only)
- S7 https://arxiv.org/html/2411.01433 (Mixtral figures)
- S8 https://github.com/pH34r-pH/long-haul/issues/48
- S9 https://arxiv.org/abs/2410.07348 (MoE++)
- S10 https://arxiv.org/abs/2408.10681 (HMoE)
- S11 https://arxiv.org/html/2604.18788v1 (NPUMoE; last 3070 chars unread)
- S12 https://arxiv.org/abs/2404.02258
- S13 https://arxiv.org/html/2607.24434 (DraftExpert)
- S14 https://arxiv.org/html/2406.06282 (PowerInfer-2)
- S15 https://arxiv.org/html/2411.01433 (HOBBIT)
- S16 https://arxiv.org/html/2410.22134 (ProMoE)
- S17 https://arxiv.org/html/2401.14361 (MoE-Infinity)
- S18 https://arxiv.org/html/2402.07033 (Fiddler)
- S19 https://arxiv.org/html/2410.17954v1 (ExpertFlow)
- S20 https://arxiv.org/abs/2312.17238
- S21 https://github.com/hyeons-lab/cera/pull/466 (search snippet only)
