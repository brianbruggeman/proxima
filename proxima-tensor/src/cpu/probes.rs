use super::*;
#[cfg(target_arch = "aarch64")]
use crate::test_support::Lcg;
use std::time::Instant;

/// Dispatches `rows` through the same `claim_and_run_rows` shared-cursor
/// mechanism [`matmul_rows_threaded`] uses, but with `oversubscribe`
/// passed in instead of hard-coded to [`crate::sized::ROW_OVERSUBSCRIBE`],
/// so [`row_oversubscribe_picks_the_multiplier`] can sweep the multiplier
/// without a rebuild per value.
fn dispatch_rows_with_oversubscribe<Row>(
    rows: usize,
    workers: usize,
    oversubscribe: usize,
    dot_row: Row,
) -> Vec<f32>
where
    Row: Fn(usize) -> Result<f32, TensorError> + Sync,
{
    // adapts the scalar `Row` this test builds (matches every real
    // caller's own dot-product closure shape) into
    // `claim_and_run_rows`'s `width`-slot form with `width == 1`, then
    // hands off to a generic-over-the-adapted-closure-type inner
    // dispatcher so the unsafe pointer cast below can name that type via
    // its own fresh generic parameter (a bare closure has no nameable
    // type to turbofish with).
    dispatch_rows_widened(rows, workers, oversubscribe, 1, move |row, slot| {
        slot[0] = dot_row(row)?;
        Ok(())
    })
}

fn dispatch_rows_widened<Wide>(
    rows: usize,
    workers: usize,
    oversubscribe: usize,
    width: usize,
    dot_row: Wide,
) -> Vec<f32>
where
    Wide: Fn(usize, &mut [f32]) -> Result<(), TensorError> + Sync,
{
    let mut output = vec![0.0f32; rows * width];
    let chunk_count = (workers.saturating_mul(oversubscribe)).clamp(1, rows.max(1));
    let chunk_len = rows.div_ceil(chunk_count);

    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining = output.as_mut_slice();
    let mut row_start = 0usize;
    while !remaining.is_empty() {
        let take_rows = chunk_len.min(remaining.len() / width);
        let (slice, rest) = remaining.split_at_mut(take_rows * width);
        remaining = rest;
        chunk_ranges.push((row_start, slice.as_mut_ptr() as usize, slice.len()));
        row_start += take_rows;
    }
    let chunk_ranges_len = chunk_ranges.len();

    let pool = nest_pool().expect("pool builds under test");
    let dot_row_address = &dot_row as *const Wide as usize;
    let next_index = Arc::new(AtomicUsize::new(0));
    let chunk_ranges: Arc<Vec<(usize, usize, usize)>> = Arc::new(chunk_ranges);
    let spawned_count = workers
        .saturating_sub(1)
        .min(chunk_ranges_len.saturating_sub(1));
    let (result_sender, result_receiver) = sync_channel(chunk_ranges_len);

    for _ in 0..spawned_count {
        let sender = result_sender.clone();
        let next_index = Arc::clone(&next_index);
        let chunk_ranges = Arc::clone(&chunk_ranges);
        drop(pool.spawn(move || {
            claim_and_run_rows::<Wide>(&next_index, dot_row_address, width, &chunk_ranges, &sender);
            Ok::<(), _>(())
        }));
    }
    claim_and_run_rows::<Wide>(
        &next_index,
        dot_row_address,
        width,
        &chunk_ranges,
        &result_sender,
    );
    drop(result_sender);

    for _ in 0..chunk_ranges_len {
        let _ = result_receiver.recv();
    }
    output
}

/// Manual microbench picking [`crate::sized::ROW_OVERSUBSCRIBE`] —
/// principle 18/19: a design constant needs a measurement artifact, not
/// reasoning. Synthetic per-row cost is deliberately imbalanced (the
/// last 1/8 of rows costs ~8x a normal row, echoing the 2.04x
/// equal-row-count spread [`OVERSUBSCRIBE`]'s own doc records for a real
/// GEMM) so a static 1:1 split leaves the calling thread idling in
/// `Receiver::recv` for whichever puller drew the straggler range.
/// Run via the `cpu_probes` example.
pub fn row_oversubscribe_picks_the_multiplier() {
    let workers = thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(1);
    let rows = 4096usize;
    let straggler_start = rows - rows / 8;
    let cost_of = |row: usize| -> u64 { if row >= straggler_start { 4000 } else { 500 } };
    let dot_row = |row: usize| -> Result<f32, TensorError> {
        let mut accumulator = 0.0f32;
        for iteration in 0..cost_of(row) {
            accumulator += (iteration as f32).sin();
        }
        Ok(accumulator)
    };

    let load = std::process::Command::new("uptime")
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime unavailable".to_string());
    eprintln!("workers={workers} rows={rows} ambient_load={load}");

    eprintln!("-- imbalanced (straggler last 1/8 of rows) --");
    for oversubscribe in [1usize, 2, 4, 8, 16, 32] {
        let mut samples_micros = Vec::with_capacity(5);
        for _ in 0..5 {
            let started = Instant::now();
            let output = dispatch_rows_with_oversubscribe(rows, workers, oversubscribe, dot_row);
            let elapsed = started.elapsed().as_micros() as f64;
            assert_eq!(output.len(), rows);
            samples_micros.push(elapsed);
        }
        let mean = samples_micros.iter().sum::<f64>() / samples_micros.len() as f64;
        let variance = samples_micros
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / samples_micros.len() as f64;
        let coefficient_of_variation = variance.sqrt() / mean;
        eprintln!(
            "oversubscribe={oversubscribe} mean_us={mean:.1} cov={coefficient_of_variation:.4} samples={samples_micros:?}"
        );
    }

    // degenerate control (principle 19/V5): a UNIFORM per-row cost has
    // nothing to steal around, so this arm isolates the atomic-cursor
    // and `SyncSender` overhead oversubscription adds without any
    // imbalance to pay for it. If a high multiplier regresses here, that
    // is the ceiling on how far oversubscription can be pushed once real
    // rows are small enough for per-chunk overhead to matter.
    let uniform_dot_row = |row: usize| -> Result<f32, TensorError> {
        let mut accumulator = 0.0f32;
        for iteration in 0..1200u64 {
            accumulator += ((row as f32) + iteration as f32).sin();
        }
        Ok(accumulator)
    };
    eprintln!("-- uniform cost (degenerate control) --");
    for oversubscribe in [1usize, 2, 4, 8, 16, 32] {
        let mut samples_micros = Vec::with_capacity(5);
        for _ in 0..5 {
            let started = Instant::now();
            let output =
                dispatch_rows_with_oversubscribe(rows, workers, oversubscribe, uniform_dot_row);
            let elapsed = started.elapsed().as_micros() as f64;
            assert_eq!(output.len(), rows);
            samples_micros.push(elapsed);
        }
        let mean = samples_micros.iter().sum::<f64>() / samples_micros.len() as f64;
        let variance = samples_micros
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / samples_micros.len() as f64;
        let coefficient_of_variation = variance.sqrt() / mean;
        eprintln!(
            "oversubscribe={oversubscribe} mean_us={mean:.1} cov={coefficient_of_variation:.4} samples={samples_micros:?}"
        );
    }
}

/// KILL-EARLY nano (attention-tile task, 2026-09-01, binding on ROW
/// 198/199): the untiled scalar path vs the NEON width tile widened by
/// this task's `outer_extent` field, vs one `cblas_sgemm` call per head
/// (`try_run_accelerate_sgemm`), at BGE's own real deployed attention
/// shapes -- `Q@K^T` (`K=32`, `N=seq_len`) and `softmax@V`
/// (`K=seq_len`, `N=32`), `seq_len` in `{7, 8, 9}`, `heads=12`. `Q@K^T`'s
/// own `N` is always `< WIDTH_TILE_VECS * 4 == 16` for every `seq_len`
/// in this set, so `width_tile_plan` declines it with `NarrowWidth`
/// regardless of this task's own fix (verified separately by the
/// `bge_route_census` example) -- its NEON number here is `N/A`, never
/// routed. Run via the `cpu_probes` example.
#[cfg(target_arch = "aarch64")]
pub fn attention_tile_shapes_nano() {
    const HEADS: usize = 12;
    const REPEATS: usize = 7;

    fn scalar_reference(
        a: &[f32],
        b_kn: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        for head in 0..heads {
            let a_head = &a[head * m * k..(head + 1) * m * k];
            let b_head = &b_kn[head * k * n..(head + 1) * k * n];
            let out_head = &mut out[head * m * n..(head + 1) * m * n];
            for row in 0..m {
                for col in 0..n {
                    let value = width_tile_scalar_cell(
                        KStridedTile {
                            data: a_head,
                            base: (row * k) as i64,
                            k_stride: 1,
                        },
                        KStridedTile {
                            data: b_head,
                            base: col as i64,
                            k_stride: n as i64,
                        },
                        k,
                        0.0,
                    );
                    out_head[row * n + col] = value;
                }
            }
        }
        out
    }

    fn neon_route<const VECS: usize>(
        a: &[f32],
        b_kn: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        let raw: [&[f32]; 2] = [a, b_kn];
        let plan = WidthTilePlan {
            a_operand: 0,
            b_operand: 1,
            row_stride_a: k as i64,
            base_a: 0,
            k_stride_a: 1,
            base_b: 0,
            k_stride_b: n as i64,
            out_base: 0,
            out_row_stride: n as i64,
            out_col_stride: 1,
            leading_total: m,
            reduction_total: k,
            width: n,
            seed: 0.0,
            outer_extent: heads,
            outer_stride_a: (m * k) as i64,
            outer_stride_b: (k * n) as i64,
            outer_stride_out: (m * n) as i64,
            vecs: VECS,
        };
        run_width_tile_neon::<VECS>(&plan, &raw, None, &mut out);
        out
    }

    #[cfg(target_os = "macos")]
    fn accelerate_route(
        a: &[f32],
        b_nk: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        for head in 0..heads {
            let a_head = &a[head * m * k..(head + 1) * m * k];
            let b_head = &b_nk[head * n * k..(head + 1) * n * k];
            let out_head = &mut out[head * m * n..(head + 1) * m * n];
            // SAFETY: `a_head`/`b_head` are exactly `m*k`/`n*k` elements
            // (`m x k` and `n x k` row-major, this function's own doc),
            // `out_head` exactly `m*n` -- every bound `try_run_accelerate_sgemm`'s
            // own `# Safety` requires.
            let accelerated = unsafe {
                try_run_accelerate_sgemm(
                    a_head, 0, k, b_head, 0, k, out_head, 0, n, m, n, k, 1, 0.0, true,
                )
            };
            assert!(
                accelerated,
                "try_run_accelerate_sgemm declined a shape this nano expects to run: m={m} n={n} k={k}"
            );
        }
        out
    }

    fn mean_cov(samples: &[f64]) -> (f64, f64) {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        (mean, variance.sqrt() / mean)
    }

    let load = std::process::Command::new("uptime")
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime unavailable".to_string());
    eprintln!("attention_tile_shapes_nano: heads={HEADS} repeats={REPEATS} ambient_load={load}");

    for seq_len in [7usize, 8, 9] {
        for (name, k, n) in [("Q@K^T", 32usize, seq_len), ("softmax@V", seq_len, 32usize)] {
            let m = seq_len;
            let mut rng = Lcg(0x9E37_79B9 ^ (m as u64) ^ ((k as u64) << 8) ^ ((n as u64) << 16));
            let a: Vec<f32> = (0..HEADS * m * k).map(|_| rng.next_unit()).collect();
            let b_kn: Vec<f32> = (0..HEADS * k * n).map(|_| rng.next_unit()).collect();
            // `n x k` row-major transpose of `b_kn`, per-head -- the
            // layout `try_run_accelerate_sgemm`'s own doc requires.
            let mut b_nk = vec![0.0f32; HEADS * n * k];
            for head in 0..HEADS {
                for row in 0..k {
                    for col in 0..n {
                        b_nk[head * n * k + col * k + row] = b_kn[head * k * n + row * n + col];
                    }
                }
            }

            let reference = scalar_reference(&a, &b_kn, HEADS, m, k, n);

            let mut scalar_samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let started = Instant::now();
                let out = scalar_reference(&a, &b_kn, HEADS, m, k, n);
                scalar_samples.push(started.elapsed().as_nanos() as f64);
                std::hint::black_box(&out);
            }
            let (scalar_mean, scalar_cov) = mean_cov(&scalar_samples);

            let width_eligible = n >= WIDTH_TILE_VECS * 4;
            let neon_report = if width_eligible {
                let mut neon_out = Vec::new();
                let mut neon_samples = Vec::with_capacity(REPEATS);
                for _ in 0..REPEATS {
                    let started = Instant::now();
                    neon_out = neon_route::<WIDTH_TILE_VECS>(&a, &b_kn, HEADS, m, k, n);
                    neon_samples.push(started.elapsed().as_nanos() as f64);
                }
                for (got, want) in neon_out.iter().zip(&reference) {
                    assert!(
                        (got - want).abs() < 1e-4,
                        "neon width tile diverged from scalar reference: got={got} want={want}"
                    );
                }
                let (neon_mean, neon_cov) = mean_cov(&neon_samples);
                format!(
                    "neon_ns={neon_mean:>9.1} (cov={neon_cov:.3}) neon_speedup={:.3}x",
                    scalar_mean / neon_mean
                )
            } else {
                "neon_ns=N/A (NarrowWidth-declined, N < WIDTH_TILE_VECS*4, never routed)"
                    .to_string()
            };

            #[cfg(target_os = "macos")]
            let accelerate_report = {
                let mut accelerate_out = Vec::new();
                let mut accelerate_samples = Vec::with_capacity(REPEATS);
                for _ in 0..REPEATS {
                    let started = Instant::now();
                    accelerate_out = accelerate_route(&a, &b_nk, HEADS, m, k, n);
                    accelerate_samples.push(started.elapsed().as_nanos() as f64);
                }
                for (got, want) in accelerate_out.iter().zip(&reference) {
                    assert!(
                        (got - want).abs() < 1e-3,
                        "accelerate diverged from scalar reference: got={got} want={want}"
                    );
                }
                let (accelerate_mean, accelerate_cov) = mean_cov(&accelerate_samples);
                format!(
                    "accelerate_ns={accelerate_mean:>9.1} (cov={accelerate_cov:.3}) accelerate_speedup={:.3}x",
                    scalar_mean / accelerate_mean
                )
            };
            #[cfg(not(target_os = "macos"))]
            let accelerate_report = "accelerate_ns=N/A (non-macos)".to_string();

            eprintln!(
                "{name:<10} M={m:<2} K={k:<2} N={n:<2} width_eligible={width_eligible:<5} scalar_ns={scalar_mean:>9.1} (cov={scalar_cov:.3}) {neon_report} {accelerate_report}"
            );
        }
    }
}

/// KILL-EARLY nano (narrow-tile task, 2026-09-01, binding on ROW
/// 198/199): the untiled scalar path vs the narrow-`VECS` NEON width
/// tile at BGE's `Q@K^T` inner shape `(M,32)x(32,M)`, `M` in `{7, 8,
/// 9}`, `heads=12` — the exact class [`attention_tile_shapes_nano`]
/// above reports `neon_ns=N/A` for. Two arms per `M`: `forced_vecs2`
/// runs `gemm_width_tile_neon::<_, 2>` exactly as the binding
/// instruction names it (at `N=7` this walks ZERO full 8-wide tiles —
/// `col_tiles = 7/8 = 0` — so every element falls through the same
/// scalar column-tail loop the untiled path already uses, measuring
/// only the tile/outer-loop overhead with no compensating NEON work);
/// `admitted` runs whichever `VECS` [`width_tile_vecs_for`] actually
/// selects for that `N` (`1` at `N=7`, `2` at `N=8/9`, identical to
/// `forced_vecs2` at `N=8/9`). Run via the `cpu_probes` example.
#[cfg(target_arch = "aarch64")]
pub fn narrow_width_tile_shapes_nano() {
    const HEADS: usize = 12;
    const REPEATS: usize = 7;

    fn scalar_reference(
        a: &[f32],
        b_kn: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        for head in 0..heads {
            let a_head = &a[head * m * k..(head + 1) * m * k];
            let b_head = &b_kn[head * k * n..(head + 1) * k * n];
            let out_head = &mut out[head * m * n..(head + 1) * m * n];
            for row in 0..m {
                for col in 0..n {
                    let value = width_tile_scalar_cell(
                        KStridedTile {
                            data: a_head,
                            base: (row * k) as i64,
                            k_stride: 1,
                        },
                        KStridedTile {
                            data: b_head,
                            base: col as i64,
                            k_stride: n as i64,
                        },
                        k,
                        0.0,
                    );
                    out_head[row * n + col] = value;
                }
            }
        }
        out
    }

    fn neon_route<const VECS: usize>(
        a: &[f32],
        b_kn: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * m * n];
        let raw: [&[f32]; 2] = [a, b_kn];
        let plan = WidthTilePlan {
            a_operand: 0,
            b_operand: 1,
            row_stride_a: k as i64,
            base_a: 0,
            k_stride_a: 1,
            base_b: 0,
            k_stride_b: n as i64,
            out_base: 0,
            out_row_stride: n as i64,
            out_col_stride: 1,
            leading_total: m,
            reduction_total: k,
            width: n,
            seed: 0.0,
            outer_extent: heads,
            outer_stride_a: (m * k) as i64,
            outer_stride_b: (k * n) as i64,
            outer_stride_out: (m * n) as i64,
            vecs: VECS,
        };
        run_width_tile_neon::<VECS>(&plan, &raw, None, &mut out);
        out
    }

    fn mean_cov(samples: &[f64]) -> (f64, f64) {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        (mean, variance.sqrt() / mean)
    }

    let load = std::process::Command::new("uptime")
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime unavailable".to_string());
    eprintln!("narrow_width_tile_shapes_nano: heads={HEADS} repeats={REPEATS} ambient_load={load}");

    for m in [7usize, 8, 9] {
        let (k, n) = (32usize, m);
        let mut rng = Lcg(0x9E37_79B9 ^ (m as u64) ^ ((k as u64) << 8) ^ ((n as u64) << 16));
        let a: Vec<f32> = (0..HEADS * m * k).map(|_| rng.next_unit()).collect();
        let b_kn: Vec<f32> = (0..HEADS * k * n).map(|_| rng.next_unit()).collect();

        let reference = scalar_reference(&a, &b_kn, HEADS, m, k, n);

        let mut scalar_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = Instant::now();
            let out = scalar_reference(&a, &b_kn, HEADS, m, k, n);
            scalar_samples.push(started.elapsed().as_nanos() as f64);
            std::hint::black_box(&out);
        }
        let (scalar_mean, scalar_cov) = mean_cov(&scalar_samples);

        let mut forced2_out = neon_route::<2>(&a, &b_kn, HEADS, m, k, n);
        for (got, want) in forced2_out.iter().zip(&reference) {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "vecs=2 not bit-identical to scalar reference: got={got} want={want}"
            );
        }
        let mut forced2_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = Instant::now();
            forced2_out = neon_route::<2>(&a, &b_kn, HEADS, m, k, n);
            forced2_samples.push(started.elapsed().as_nanos() as f64);
        }
        std::hint::black_box(&forced2_out);
        let (forced2_mean, forced2_cov) = mean_cov(&forced2_samples);

        let admitted_vecs = width_tile_vecs_for(n).expect("N=7/8/9 all admit at least VECS=1");
        let (admitted_mean, admitted_cov) = if admitted_vecs == 2 {
            (forced2_mean, forced2_cov)
        } else {
            let mut admitted_out = neon_route::<1>(&a, &b_kn, HEADS, m, k, n);
            for (got, want) in admitted_out.iter().zip(&reference) {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "vecs=1 not bit-identical to scalar reference: got={got} want={want}"
                );
            }
            let mut admitted_samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let started = Instant::now();
                admitted_out = neon_route::<1>(&a, &b_kn, HEADS, m, k, n);
                admitted_samples.push(started.elapsed().as_nanos() as f64);
            }
            std::hint::black_box(&admitted_out);
            mean_cov(&admitted_samples)
        };
        let admitted_speedup = scalar_mean / admitted_mean;

        eprintln!(
            "Q@K^T M={m:<2} K={k:<2} N={n:<2} scalar_ns={scalar_mean:>9.1} (cov={scalar_cov:.3}) forced_vecs2_ns={forced2_mean:>9.1} (cov={forced2_cov:.3}) forced_vecs2_speedup={:.3}x admitted_vecs={admitted_vecs} admitted_ns={admitted_mean:>9.1} (cov={admitted_cov:.3}) admitted_speedup={admitted_speedup:.3}x",
            scalar_mean / forced2_mean
        );
    }
}
