//! Resident-corpus GEMV + top-k selection: the search-serving shape.
//!
//! One query at a time against a resident corpus matrix: `Plan` is built
//! ONCE per corpus size, `execute_plan` runs per query against the SAME
//! page-aligned `[NxD]` buffer every call (`proxima_tensor::AlignedBuffer`),
//! so the no-copy path (`omega::metal::upload_block_no_copy`,
//! `NOCOPY_BUFFERS` keyed by `(pointer, byte_length)`) wires the corpus once
//! and every subsequent query reuses that wrapper instead of re-uploading —
//! see `omega/src/metal.rs`'s `Plan` doc and the module's "Host buffer
//! upload" section for the mechanism this measures.
//!
//! Query vectors are NOT resident (3 KiB at d=768, a fresh `Vec<f32>` per
//! query, deliberately never page-aligned) — the small operand always takes
//! the copy path, which is correct and cheap; only the corpus matters here.
//!
//! This probe carries TWO top-k selection arms over the SAME `execute_plan`
//! output per query, so job 2's fix is measured against job 1's landing in
//! one artifact instead of two:
//!   - `full_sort`: `sort_by` the whole distance vector, `truncate(k)` — the
//!     host-side incumbent this probe exists to retire.
//!   - `partial_select`: `select_nth_unstable_by` to partition around the
//!     k-th element (`O(N)` expected), then `sort` only the k-slice for
//!     order. `Vec::select_nth_unstable_by` is `core`/`std`, no new
//!     dependency — see the module doc section "why not on-GPU" for why a
//!     device-side reduction was not reached.
//!
//! # Why not on-GPU (job 2, options 1 and 2 in the brief)
//!
//! `proxima_tensor::ScalarOp` (audited: `Identity, Add, Subtract, Multiply,
//! Divide, Maximum, Minimum, Negate, Reciprocal, Exponential, Logarithm,
//! SquareRoot, Tanh, Erf, Greater, Equal, Select`) has no index-returning
//! primitive, and `Op::Reduce`/`NodeSpec::Reduce` writes exactly one scalar
//! per output-map cell — it can find the maximum VALUE (`ScalarOp::Maximum`)
//! but never which index produced it. Exact top-20 selection needs the
//! index, not just the value, so it is not expressible in this algebra
//! without minting a new IR primitive (an argmax-reduce or an indexed partial
//! sort) — out of scope for a 90-minute budget and a violation of "reuse
//! first" (`/guiding-principles` §1) to mint one under time pressure. A
//! threadgroup-level partial reduction emitting per-group candidates (option
//! 2) has the same index problem at the kernel-argument level and was not
//! reached for the same reason. `partial_select` is therefore the floor this
//! probe measures, not the ceiling — a future row can revisit options 1/2 if
//! an index-carrying reduce primitive lands on its own merits elsewhere.
//!
//! # 4M gate failure — a DIFFERENT failure than previously characterized
//!
//! This probe's own run reproduces a gate failure at 4,000,000 rows, but NOT
//! the one previously reported ("rank-3 tie, max abs diff 2.476e-4"). This
//! run's failure is a rank-0 mismatch, wrong INDEX entirely (oracle
//! idx=2953695 val=207.692368 vs candidate idx=2150258 val=205.864502, abs
//! diff 1.83 -- three orders of magnitude past the 1e-4 tolerance and not a
//! tie at all). Hypothesized mechanism: `rows*DIMENSION` (3,072,000,000)
//! crosses `i32::MAX` (2,147,483,647) while the passing 1M size
//! (768,000,000) does not, suggesting a signed 32-bit index overflow
//! somewhere in the emitted kernel or its dispatch. **This hypothesis was
//! TESTED AND REFUTED**: rows=2,796,192 (`rows*DIMENSION` = 2,147,475,456,
//! just under `i32::MAX`) and rows=2,796,208 (2,147,487,744, just over) BOTH
//! passed the gate cleanly in an isolated run. The real boundary lies
//! somewhere between 2,796,208 and 4,000,000, unidentified within this
//! session's time budget -- reported as a genuine, unresolved finding, not
//! papered over. Do not trust an extrapolation from the 1M row figures to
//! 4M until this is root-caused.
//!
//! # Correctness gate
//!
//! BEFORE any timing, for the first query at each corpus size: computes the
//! csr-db-vector `DotProduct` oracle exactly
//! (`csr/crates/csr-db-vector/src/distance.rs:19-52`,
//! `dot_product_distance = -dot_product_simd(a, b)`, 8-lane manual
//! accumulation) over the WHOLE corpus, ranks ascending distance (= descending
//! dot) with index as the deterministic tie-break, and compares the top-20
//! slots (value within 1e-4 abs, same order) against BOTH `full_sort` and
//! `partial_select` reading the SAME `execute_plan` output. A mismatch
//! prints the disagreement and withholds that size's timing entirely — the
//! row is marked GATE FAILED, not silently skipped.

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    run();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("resident_gemv_topk requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run() {
    use std::time::Instant;

    use proxima_tensor::test_support::Lcg;
    use proxima_tensor::{
        AlignedBuffer, DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce,
        ReduceInit, ScalarOp, append, projection,
    };

    const DIMENSION: usize = 768;
    const TOP_K: usize = 20;
    const TIMED_QUERIES: usize = 50;
    const WARMUP_QUERIES: usize = 5;
    const TOLERANCE: f32 = 1e-4;

    /// The csr-db-vector oracle, LANES=8 manual accumulation, copied verbatim
    /// from `csr/crates/csr-db-vector/src/distance.rs:19-33` so the gate
    /// compares bit-for-bit against the same accumulation order the
    /// incumbent uses, not merely the same mathematical dot product.
    fn oracle_dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
        const LANES: usize = 8;
        let chunks = a.len() / LANES;
        let mut sum = [0.0f32; LANES];
        for i in 0..chunks {
            let offset = i * LANES;
            for j in 0..LANES {
                sum[j] += a[offset + j] * b[offset + j];
            }
        }
        let mut total: f32 = sum.iter().sum();
        for i in (chunks * LANES)..a.len() {
            total += a[i] * b[i];
        }
        total
    }

    fn gemv_program(rows: u32, k: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let corpus = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(rows), Extent::Static(k)],
                name: None,
            },
        );
        let query = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k), Extent::Static(1)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (corpus, IndexMap::Affine(projection(3, &[0, 2]))),
                    (query, IndexMap::Affine(projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    /// Ranked (value, original_index) pairs, descending dot product, index
    /// ascending as the deterministic tie-break. Shared by the oracle and
    /// both device-side selection arms so a disagreement can only be about
    /// WHICH values were selected, never how ties within one method break.
    fn rank_descending(values: &[f32]) -> Vec<(f32, usize)> {
        let mut indexed: Vec<(f32, usize)> =
            values.iter().copied().zip(0..).collect::<Vec<_>>();
        indexed.sort_by(|left, right| {
            right.0.total_cmp(&left.0).then(left.1.cmp(&right.1))
        });
        indexed
    }

    /// The host-side incumbent this probe exists to retire: `O(N log N)`.
    fn full_sort_topk(values: &[f32], k: usize) -> Vec<(f32, usize)> {
        let mut indexed: Vec<(f32, usize)> =
            values.iter().copied().zip(0..).collect::<Vec<_>>();
        indexed.sort_by(|left, right| {
            right.0.total_cmp(&left.0).then(left.1.cmp(&right.1))
        });
        indexed.truncate(k);
        indexed
    }

    /// The fix: partition around the k-th element (`O(N)` expected), sort
    /// only the k-slice for order. Never fully orders the tail.
    fn partial_select_topk(values: &[f32], k: usize) -> Vec<(f32, usize)> {
        let mut indexed: Vec<(f32, usize)> =
            values.iter().copied().zip(0..).collect::<Vec<_>>();
        let bound = k.min(indexed.len());
        if bound > 0 && bound < indexed.len() {
            indexed.select_nth_unstable_by(bound - 1, |left, right| {
                right.0.total_cmp(&left.0).then(left.1.cmp(&right.1))
            });
        }
        indexed.truncate(bound);
        indexed.sort_by(|left, right| {
            right.0.total_cmp(&left.0).then(left.1.cmp(&right.1))
        });
        indexed
    }

    fn slots_agree(oracle: &[(f32, usize)], candidate: &[(f32, usize)]) -> Option<String> {
        if oracle.len() != candidate.len() {
            return Some(format!(
                "slot count mismatch: oracle={} candidate={}",
                oracle.len(),
                candidate.len()
            ));
        }
        for (rank, (oracle_slot, candidate_slot)) in oracle.iter().zip(candidate.iter()).enumerate() {
            let diff = (oracle_slot.0 - candidate_slot.0).abs();
            if diff > TOLERANCE || oracle_slot.1 != candidate_slot.1 {
                return Some(format!(
                    "rank {rank} mismatch: oracle=(idx={}, val={:.6}) candidate=(idx={}, val={:.6}) abs_diff={diff:.6e}",
                    oracle_slot.1, oracle_slot.0, candidate_slot.1, candidate_slot.0
                ));
            }
        }
        None
    }

    fn ru_maxrss_gib() -> f64 {
        let mut usage: libc::rusage = unsafe { core::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
            return 0.0;
        }
        // macOS reports ru_maxrss in BYTES, unlike Linux's KiB.
        usage.ru_maxrss as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    fn percentile(sorted_ms: &[f64], fraction: f64) -> f64 {
        let index = ((sorted_ms.len() - 1) as f64 * fraction).round() as usize;
        sorted_ms[index]
    }

    fn mean_and_cov(samples_ms: &[f64]) -> (f64, f64) {
        let mean = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
        let variance = samples_ms.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
            / samples_ms.len() as f64;
        let stddev = variance.sqrt();
        let cov_pct = if mean > 0.0 { stddev / mean * 100.0 } else { 0.0 };
        (mean, cov_pct)
    }

    struct SizeRow {
        rows: usize,
        gemv_p50_ms: f64,
        gemv_p99_ms: f64,
        gemv_cov_pct: f64,
        full_sort_p50_ms: f64,
        full_sort_cov_pct: f64,
        partial_select_p50_ms: f64,
        partial_select_cov_pct: f64,
        total_full_sort_p50_ms: f64,
        total_partial_select_p50_ms: f64,
        eff_gbs_full_sort: f64,
        eff_gbs_partial_select: f64,
        peak_rss_gib: f64,
        nocopy_attempts: u64,
        nocopy_reuses: u64,
        real_wires: u64,
        query_count: usize,
    }

    fn bench_size(rows: usize) -> Result<SizeRow, String> {
        let page = omega::metal::page_size();
        let mut corpus = AlignedBuffer::new(rows * DIMENSION, page)
            .map_err(|error| format!("aligned corpus allocation failed at rows={rows}: {error}"))?;
        assert_eq!(
            corpus.len(),
            rows * DIMENSION,
            "rows*DIMENSION must already be a page multiple for the no-copy \
             path's (pointer, byte_length) key to hold across calls -- see \
             the module doc"
        );
        let mut filler = Lcg(0x5EED_C0DE ^ rows as u64);
        for value in corpus.iter_mut() {
            *value = filler.next_unit();
        }

        let (program, sum) = gemv_program(rows as u32, DIMENSION as u32);

        let attempts_before = omega::metal::NOCOPY_BUFFER_UPLOADS.get();
        let reuses_before = omega::metal::NOCOPY_BUFFER_REUSES.get();

        let mut gemv_samples_ms = Vec::with_capacity(TIMED_QUERIES);
        let mut full_sort_samples_ms = Vec::with_capacity(TIMED_QUERIES);
        let mut partial_select_samples_ms = Vec::with_capacity(TIMED_QUERIES);
        let mut gate_failure: Option<String> = None;

        let query_at = |index: u64| -> Vec<f32> {
            let mut lcg = Lcg(0x1357_9BDF ^ index);
            (0..DIMENSION).map(|_| lcg.next_unit()).collect()
        };

        let resolved = {
            let warmup_query = query_at(0);
            let blocks = [QuantizedBlock::Float32(&corpus), QuantizedBlock::Float32(&warmup_query)];
            let resolved = omega::plan(&program, &[], &blocks, &[sum])
                .map_err(|error| format!("plan failed at rows={rows}: {error}"))?;
            omega::execute_plan(&resolved, &blocks)
                .map_err(|error| format!("warmup execute failed at rows={rows}: {error}"))?;
            resolved
        };

        for warmup_index in 1..=WARMUP_QUERIES as u64 {
            let query = query_at(warmup_index);
            let blocks = [QuantizedBlock::Float32(&corpus), QuantizedBlock::Float32(&query)];
            omega::execute_plan(&resolved, &blocks)
                .map_err(|error| format!("warmup execute failed at rows={rows}: {error}"))?;
        }

        for query_index in 0..TIMED_QUERIES as u64 {
            let query = query_at(1000 + query_index);
            let blocks = [QuantizedBlock::Float32(&corpus), QuantizedBlock::Float32(&query)];

            let gemv_started = Instant::now();
            let evaluated = omega::execute_plan(&resolved, &blocks)
                .map_err(|error| format!("execute failed at rows={rows}: {error}"))?;
            let gemv_elapsed_ms = gemv_started.elapsed().as_secs_f64() * 1000.0;

            let distances = evaluated.root();
            assert_eq!(distances.len(), rows, "degenerate probe: no output at rows={rows}");

            if query_index == 0 {
                let oracle_dots: Vec<f32> = (0..rows)
                    .map(|row| {
                        let row_start = row * DIMENSION;
                        oracle_dot_product_simd(&corpus[row_start..row_start + DIMENSION], &query)
                    })
                    .collect();
                let oracle_ranked = rank_descending(&oracle_dots);
                let oracle_top = &oracle_ranked[..TOP_K.min(oracle_ranked.len())];

                let full_sort_top = full_sort_topk(distances, TOP_K);
                let partial_select_top = partial_select_topk(distances, TOP_K);

                if let Some(mismatch) = slots_agree(oracle_top, &full_sort_top) {
                    gate_failure =
                        Some(format!("full_sort vs oracle at rows={rows}: {mismatch}"));
                } else if let Some(mismatch) = slots_agree(oracle_top, &partial_select_top) {
                    gate_failure =
                        Some(format!("partial_select vs oracle at rows={rows}: {mismatch}"));
                }
            }

            let full_sort_values = distances.to_vec();
            let full_sort_started = Instant::now();
            let full_sort_result = full_sort_topk(&full_sort_values, TOP_K);
            let full_sort_elapsed_ms = full_sort_started.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(&full_sort_result);

            let partial_select_values = distances.to_vec();
            let partial_select_started = Instant::now();
            let partial_select_result = partial_select_topk(&partial_select_values, TOP_K);
            let partial_select_elapsed_ms = partial_select_started.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(&partial_select_result);

            gemv_samples_ms.push(gemv_elapsed_ms);
            full_sort_samples_ms.push(full_sort_elapsed_ms);
            partial_select_samples_ms.push(partial_select_elapsed_ms);
        }

        if let Some(failure) = gate_failure {
            return Err(failure);
        }

        let attempts_after = omega::metal::NOCOPY_BUFFER_UPLOADS.get();
        let reuses_after = omega::metal::NOCOPY_BUFFER_REUSES.get();
        let nocopy_attempts = attempts_after - attempts_before;
        let nocopy_reuses = reuses_after - reuses_before;

        gemv_samples_ms.sort_by(f64::total_cmp);
        full_sort_samples_ms.sort_by(f64::total_cmp);
        partial_select_samples_ms.sort_by(f64::total_cmp);

        let (_, gemv_cov_pct) = mean_and_cov(&gemv_samples_ms);
        let (_, full_sort_cov_pct) = mean_and_cov(&full_sort_samples_ms);
        let (_, partial_select_cov_pct) = mean_and_cov(&partial_select_samples_ms);

        let gemv_p50_ms = percentile(&gemv_samples_ms, 0.50);
        let gemv_p99_ms = percentile(&gemv_samples_ms, 0.99);
        let full_sort_p50_ms = percentile(&full_sort_samples_ms, 0.50);
        let partial_select_p50_ms = percentile(&partial_select_samples_ms, 0.50);

        let corpus_bytes = (rows * DIMENSION * 4) as f64;
        let total_full_sort_p50_ms = gemv_p50_ms + full_sort_p50_ms;
        let total_partial_select_p50_ms = gemv_p50_ms + partial_select_p50_ms;
        let eff_gbs_full_sort = (corpus_bytes / 1e9) / (total_full_sort_p50_ms / 1000.0);
        let eff_gbs_partial_select = (corpus_bytes / 1e9) / (total_partial_select_p50_ms / 1000.0);

        Ok(SizeRow {
            rows,
            gemv_p50_ms,
            gemv_p99_ms,
            gemv_cov_pct,
            full_sort_p50_ms,
            full_sort_cov_pct,
            partial_select_p50_ms,
            partial_select_cov_pct,
            total_full_sort_p50_ms,
            total_partial_select_p50_ms,
            eff_gbs_full_sort,
            eff_gbs_partial_select,
            peak_rss_gib: ru_maxrss_gib(),
            nocopy_attempts,
            nocopy_reuses,
            real_wires: nocopy_attempts - nocopy_reuses,
            query_count: TIMED_QUERIES,
        })
    }

    let sizes = [10_000usize, 100_000, 1_000_000, 4_000_000];
    let mut rows_out: Vec<SizeRow> = Vec::new();
    let mut any_gate_failed = false;

    println!(
        "resident_gemv_topk d={DIMENSION} k={TOP_K} timed_queries={TIMED_QUERIES} \
         warmup_queries={WARMUP_QUERIES} tolerance={TOLERANCE:.0e}"
    );

    for &rows in &sizes {
        print!("--- rows={rows} gate ... ");
        match bench_size(rows) {
            Ok(row) => {
                println!("PASSED (query 0, top-{TOP_K} vs csr-db-vector DotProduct oracle)");
                rows_out.push(row);
            }
            Err(failure) => {
                println!("FAILED");
                println!("  {failure}");
                println!("  timing WITHHELD for rows={rows} per the correctness-first gate.");
                any_gate_failed = true;
            }
        }
    }

    println!();
    println!(
        "{:>9} | {:>16} | {:>16} | {:>7} | {:>14} | {:>7} | {:>18} | {:>7} | {:>13} | {:>13} | {:>9} | {:>9} | {:>10} | {:>18}",
        "rows", "gemv p50 (ms)", "gemv p99 (ms)", "gemv CoV%", "full_sort p50", "fs CoV%",
        "partial_select p50", "ps CoV%", "total(fs) ms", "total(ps) ms", "GB/s(fs)", "GB/s(ps)",
        "peak RSS GiB", "nocopy att/reuse/wire"
    );
    for row in &rows_out {
        println!(
            "{:>9} | {:>16.3} | {:>16.3} | {:>7.1} | {:>14.3} | {:>7.1} | {:>18.3} | {:>7.1} | {:>13.3} | {:>13.3} | {:>9.1} | {:>9.1} | {:>10.3} | {}/{}/{}",
            row.rows,
            row.gemv_p50_ms,
            row.gemv_p99_ms,
            row.gemv_cov_pct,
            row.full_sort_p50_ms,
            row.full_sort_cov_pct,
            row.partial_select_p50_ms,
            row.partial_select_cov_pct,
            row.total_full_sort_p50_ms,
            row.total_partial_select_p50_ms,
            row.eff_gbs_full_sort,
            row.eff_gbs_partial_select,
            row.peak_rss_gib,
            row.nocopy_attempts,
            row.nocopy_reuses,
            row.real_wires,
        );
    }

    println!();
    for row in &rows_out {
        let topk_speedup = row.full_sort_p50_ms / row.partial_select_p50_ms.max(1e-9);
        let total_speedup = row.total_full_sort_p50_ms / row.total_partial_select_p50_ms.max(1e-9);
        println!(
            "rows={:>9}  topk speedup(full_sort/partial_select)={:.2}x  \
             total/query speedup={:.2}x  (N={} timed queries per arm)",
            row.rows, topk_speedup, total_speedup, row.query_count
        );
    }

    let baseline_path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/resident_gemv_topk.baseline.json");
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"dimension\": {DIMENSION},\n"));
    json.push_str(&format!("  \"top_k\": {TOP_K},\n"));
    json.push_str(&format!("  \"timed_queries\": {TIMED_QUERIES},\n"));
    json.push_str(&format!("  \"tolerance\": {TOLERANCE},\n"));
    json.push_str("  \"sizes\": [\n");
    for (index, row) in rows_out.iter().enumerate() {
        let comma = if index + 1 == rows_out.len() { "" } else { "," };
        json.push_str(&format!(
            "    {{\"rows\": {}, \"gemv_p50_ms\": {:.6}, \"gemv_p99_ms\": {:.6}, \
             \"gemv_cov_pct\": {:.3}, \"full_sort_p50_ms\": {:.6}, \
             \"full_sort_cov_pct\": {:.3}, \"partial_select_p50_ms\": {:.6}, \
             \"partial_select_cov_pct\": {:.3}, \"total_full_sort_p50_ms\": {:.6}, \
             \"total_partial_select_p50_ms\": {:.6}, \"eff_gbs_full_sort\": {:.3}, \
             \"eff_gbs_partial_select\": {:.3}, \"peak_rss_gib\": {:.3}, \
             \"nocopy_attempts\": {}, \"nocopy_reuses\": {}, \"real_wires\": {}}}{comma}\n",
            row.rows,
            row.gemv_p50_ms,
            row.gemv_p99_ms,
            row.gemv_cov_pct,
            row.full_sort_p50_ms,
            row.full_sort_cov_pct,
            row.partial_select_p50_ms,
            row.partial_select_cov_pct,
            row.total_full_sort_p50_ms,
            row.total_partial_select_p50_ms,
            row.eff_gbs_full_sort,
            row.eff_gbs_partial_select,
            row.peak_rss_gib,
            row.nocopy_attempts,
            row.nocopy_reuses,
            row.real_wires,
        ));
    }
    json.push_str("  ],\n");
    json.push_str(&format!("  \"any_gate_failed\": {any_gate_failed}\n"));
    json.push_str("}\n");
    std::fs::write(baseline_path, &json)
        .unwrap_or_else(|error| panic!("failed to write baseline json to {baseline_path}: {error}"));
    println!();
    println!("baseline written: {baseline_path}");

    if any_gate_failed {
        std::process::exit(1);
    }
}
