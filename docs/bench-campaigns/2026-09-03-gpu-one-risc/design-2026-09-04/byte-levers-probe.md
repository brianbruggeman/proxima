# What the algebra supports today for the byte levers (read-only probe, main af918bb)

## Multi-token pass — PARTIAL
- Program and plan cache are `new_count`-generic: symbols `[new_count, kv_bound_extent]`
  (generate.rs:2394), cache key `(new_count, bucket)` (generate.rs:1248,1323).
- The loop hardwires steady state to one token: `next_ids = vec![token_id]` (generate.rs:2503,
  2069); `sample_next_token` takes one logits row (proxima-tokenizer/src/sample.rs:277).
- Packed-row kernel batches 4 OUTPUT rows per simdgroup over ONE activation (ggml nr0), no
  s-axis fold (msl.rs:3171-3200) — with s = k it would re-stream weights k times. The weight-once
  multi-token kernel exists only as `metal-tiled-gemm` (simdgroup_matrix, Q4_K only, token axis,
  min tokens gate; omega/Cargo.toml:129, msl.rs:1575-1608, 3961-4030), unreachable from decode.
- No draft/speculative/verify machinery anywhere.

## Dynamic row elision — PARTIAL
- `IndexMap::Computed { indices, index_map, base, gathered_dim }` (proxima-tensor/src/map.rs:134-151)
  is live in production for MoE expert routing on CPU (`run_reduce_quantized` gather arm,
  cpu.rs:7091, sizing 7205-7261) — the same mechanism expresses `y[j] = Σ_k W[idx[j],k]·x[k]`,
  exercised only for whole-expert slabs.
- Metal: `classify_packed_row_block` requires `gather_count == 0` (msl.rs:1039-1052, 1389) → any
  gathered weight falls to the serial one-thread-per-output kernel (`push_gather_fetch`
  msl.rs:2286-2318, element-granular, fault slot per operand metal.rs:3221-3233).
- No in-graph selector: ScalarOp has no top-k/threshold-count/argsort; Keep = Reduce|Scan
  (op.rs:60-78, 142-147). The landed elision probe (proxima-tensor/benches/bench_dynamic_elision.rs,
  ROW 180/181 discipline.md:16430,16471) built the skip set on the HOST, CPU-only, and measured
  dispatch-bound (0.2-0.29 ns/element vs DRAM 0.057).

## Lower-bit codecs — PARTIAL
- CPU `QuantizedBlock` (cpu.rs:3091) and Metal `PackedCodec` (msl.rs:788-805): F32, Q4_K, Q5_K,
  Q6_K, Q8_0 (+F16/BF16, Q4_0 unimplemented downstream). Q2_K/Q3_K/Q4_0/Q5_0 parse
  (proxima-gguf/src/types.rs:109-133, 245-281) and are rejected at bind
  (`UnrepresentableGgmlType`, proxima-model-interop/src/bind.rs:70-73; capability.rs:141-156);
  IQ*/TQ* never attempted. output.weight is Q6_K (107 MB) and binds today.

## Implication for the ≤1.4 GB/token target (4.169 GB today)
- k tokens per weight pass divides weight bytes by k for the accepted tokens: needs (a) the
  packed-row kernel to fold s activations per streamed weight row (nr1), (b) a verify step in
  the decode FSM (accept longest matching prefix; placed KV rollback = do not advance
  cached_len past the accepted count), (c) a draft source as a pipe (n-gram prompt lookup
  needs no second model; a small draft model is the same pipe shape).
- Row elision needs a selector op or a host-side selector pipe + the packed-row kernel accepting
  a row-axis gather; bytes scale with the selected fraction.
- Codecs: Q3_K/Q2_K need encoder/decoder in proxima_gguf::quant + kernel bodies; bytes ×0.75/×0.6.
