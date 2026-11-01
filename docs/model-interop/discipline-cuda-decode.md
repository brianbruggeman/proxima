## C5 — CUDA M=1 decode (token generation) vs llama.cpp f1ea20621 (OPEN)

Objective (owner): proxima tg >= llama.cpp tg on RTX 2070 (sm_75). Branch
`perf/cuda-decode`. Measurement: `decode_compare <gguf> "The capital of France is" 32 cuda`,
mean of the 30 decode steps after the first (step 0 carries the plan switch), 5 runs,
mean +- sd across runs. Gate per change: `decode_compare ... 32 both` matched=32,
`cargo val -p proxima-omega --features cuda-driver --lib -E 'test(cuda)'`, `binary(cuda_parity)`.
Instrumentation: `PROXIMA_CUDA_EVENT_TIMING=1` (cuda events per node, summed per kernel) and
`PROXIMA_CUDA_HOST_TIMING=1`, plus the `cuda plan refit phases` / `cuda execute phases` debug events.
Allocation budget: hot path (per launch) zero new allocations; the patch path allocates one
`Vec<NodeId>` (changed set) per step, setup path unchanged.
Box is shared with two other agents, so every number carries loadavg; GPU was never idle.

| change | evidence (qwen3-0.6b-q8_0 unless noted) | result |
|---|---|---|
| baseline 080bded3 | host timing: bind 9.4-10.6 ms (full re-bind, `bind_refused=matcher fused kind`), upload 1.1, submit 2.1, sync 10.3, readback 0.5; step 26.9-28.3 ms | owner: 20.26 ms |
| refit cached-attention rows in place (`refit_cached_attention_rows`, already used by Metal) | bind 9.4 -> 0.5 ms, step 28.3 -> 17.7 ms | kept |
| skip unchanged uniform uploads | event GPU total 15.4 -> 12.3 ms (455 pageable htod copies per token removed) | kept |
| rank-1 reduction needs no coordinate divmod; unroll 8 | norm kernel 45 -> 24 us | kept |
| coop block width 32 -> 128 | 16.6 vs 16.7 ms, no effect | ROLLBACK |
| lean packed loop + chunk8 for q4k/q1_0/q2_0 | matvec 41 -> 38 us, no effect alone | kept (prerequisite) |
| 4-chunk load batching | no effect | kept |
| float4 activation loads (lane-chunked scalar loads were 8 L1 wavefronts each) | GPU 10.2 -> 7.1 ms; lm_head 1555 -> 676 us; Bonsai Q1_0 matvec 131 -> 39 us | kept |
| broadcast reduce launched one warp per element (4096 redundant warps per norm) -> one per row; single-row norm takes a 256-thread block | norm 24 -> 7.2 us; Bonsai GPU 11.9 -> 7.6 ms | kept |
| 4 independent accumulators per lane | 6.6 vs 6.1 ms (noise), no effect | ROLLBACK |
| patch only touched ops (memoized touched set, in-place kernel/uniform refit) | refit 1.59 -> 0.6 ms (infer 0.44 + patch 0.16), Bonsai step 11.8 -> 10.0 ms | kept |
| cached attention: 4 warps per vector, merged softmax state | 27 -> 10.6 us per layer | kept |
| cache resolved CudaFunction per node | submit 2.1 -> 0.93 ms per token at 455 nodes | kept |

Substrate: unit tests `a_single_row_broadcast_reduce_takes_a_whole_block`,
`a_multi_row_broadcast_reduce_keeps_one_warp_per_row`,
`a_rank_one_packed_reduce_hoists_row_bases_and_decodes_unrolled_chunks`,
`cached_attention_gives_each_vector_a_block_of_warps_that_merge_softmax_state`; parity
`cuda_parity` (GPU); decode_compare matched=32 on all five models is a manual gate, not CI.
Saved criterion-style baselines do not exist for this path; the missing proof substrate is a
CI job that runs `decode_compare ... both` on a small model and a baseline tg file.

