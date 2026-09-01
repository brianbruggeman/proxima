//! Width-tile accumulator sweep: `gemm_width_tile_neon`
//! (`proxima-tensor/src/cpu.rs:8573`, now `<const ROWS: usize, const VECS:
//! usize>`) holds `ROWS * VECS` `float32x4_t` accumulators in registers.
//! Production ships `WIDTH_TILE_ROWS=4 * WIDTH_TILE_VECS=4` = 16
//! (`proxima-tensor/src/sized.rs:252,258`). ROW 20's own accumulator sweep
//! (`docs/discipline.md:1903-1910`) puts NEON f32 saturation at 24
//! accumulators / 48.5 GMAC/s. At `VECS=6` the kernel's live register count
//! is `ROWS*VECS` accumulators + `VECS` b-vectors + 1 broadcast a-value =
//! `4*6 + 6 + 1 = 31` of aarch64's 32 vector registers -- fits without
//! spilling. Both BGE hidden widths divide evenly by 24 columns (384 % 24
//! == 0, 1536 % 24 == 0), so `VECS=6` introduces no new column tail on the
//! deployed shapes.
//!
//! PRE-REGISTERED PREDICTION (recorded before this file was ever run): at
//! M=8 on the QKVO (384x384) and FFN (384x1536, 1536x384) shapes, VECS=6
//! beats VECS=4 by >=1.15x. VECS=5 lands between VECS=4 and VECS=6. Any
//! config whose live register count exceeds 32 (none in this sweep) would
//! be predicted to regress from spilling. Current measured rate to beat:
//! ~17.3 GMAC/s implied (ROW 206), 36% of the 48.5 GMAC/s NEON ceiling.
//!
//! METHODOLOGY: each cell measures the REAL per-shape dispatch a config
//! would run in production -- `row_tiles = M / ROWS` (floor) row-tiles times
//! `col_tiles = N / (VECS*4)` (floor) column-tiles, each an actual
//! `gemm_width_tile_neon::<ROWS, VECS>` call over the shape's full `K`
//! reduction, against a packed-panel `B` (panel k_stride == tile_cols,
//! matching `PackedWidthPanels`' own layout -- packing is default-on since
//! ROW 207). A `(ROWS, VECS)` config that cannot form even one row-tile at a
//! given `M` (`M < ROWS`) is reported `n/a` rather than fabricated -- this is
//! itself the reason `(1, VECS)` variants exist: mnist's M=1 fc path can
//! only ever reach a `ROWS=1` kernel in production, never `ROWS=4`.
//! Same-buffer reuse across repeats is a warm-cache measurement (register/
//! compute-bound), the regime the accumulator-count hypothesis is about.

#![allow(clippy::unwrap_used, clippy::expect_used)]

// the whole sweep is aarch64-only: `gemm_width_tile_neon`/`KStridedTile`
// only exist under `target_arch = "aarch64"` (`proxima-tensor/src/cpu.rs`).
#[cfg(target_arch = "aarch64")]
mod sweep {
    use proxima_tensor::cpu::{KStridedTile, gemm_width_tile_neon};

    const REPEATS: usize = 5;
    const CALLS_PER_REPEAT: usize = 50;

    fn deterministic_data(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let mixed = (index as u32).wrapping_mul(2654435761).wrapping_add(salt);
                (mixed as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    /// Packs `b` (`k_total x n` row-major) into `PackedWidthPanels`' own
    /// panel-major layout for a given `tile_cols`: panel `p`'s `k_total *
    /// tile_cols` block starts at `p * k_total * tile_cols`, row `k` inside
    /// panel `p` at `k * tile_cols`. Only full column tiles are packed --
    /// exactly what `run_width_tile_neon` walks through the packed path.
    fn pack_panels(b: &[f32], k_total: usize, n: usize, tile_cols: usize) -> (Vec<f32>, usize) {
        let full_col_tiles = n / tile_cols;
        let mut data = vec![0.0f32; full_col_tiles * k_total * tile_cols];
        for tile in 0..full_col_tiles {
            for row in 0..k_total {
                for col in 0..tile_cols {
                    let source = row * n + tile * tile_cols + col;
                    let dest = tile * k_total * tile_cols + row * tile_cols + col;
                    data[dest] = b[source];
                }
            }
        }
        (data, full_col_tiles)
    }

    struct Timed {
        mean_ns: f64,
        cov_pct: f64,
        samples: Vec<f64>,
    }

    /// One full `row_tiles x col_tiles` walk over the packed shape -- the
    /// unit `time_calls` repeats `CALLS_PER_REPEAT` times per repeat.
    fn run_shape_pass<const ROWS: usize, const VECS: usize>(a: &[f32], k_total: usize, row_tiles: usize, packed_b: &[f32], col_tiles: usize) {
        let tile_cols = VECS * 4;
        for row_tile in 0..row_tiles {
            let base_a = (row_tile * ROWS * k_total) as i64;
            for col_tile in 0..col_tiles {
                let base_b = (col_tile * k_total * tile_cols) as i64;
                let mut tile_out = [[[0.0f32; 4]; VECS]; ROWS];
                // caller-checked: `a` holds `row_tiles * ROWS` rows of
                // `k_total` contiguous elements each; `packed_b` holds
                // `col_tiles` panels of `k_total * tile_cols` contiguous
                // elements each -- both sized exactly to match by the
                // callers below.
                unsafe {
                    gemm_width_tile_neon::<ROWS, VECS>(
                        KStridedTile { data: a, base: base_a, k_stride: 1 },
                        k_total as i64,
                        KStridedTile { data: packed_b, base: base_b, k_stride: tile_cols as i64 },
                        k_total,
                        &mut tile_out,
                    );
                }
                std::hint::black_box(&tile_out);
            }
        }
    }

    fn time_calls<F: FnMut()>(mut call: F) -> Timed {
        let mut ns_per_call_per_repeat = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            for _ in 0..CALLS_PER_REPEAT {
                call();
            }
            let elapsed = start.elapsed();
            ns_per_call_per_repeat.push(elapsed.as_nanos() as f64 / CALLS_PER_REPEAT as f64);
        }
        let mean = ns_per_call_per_repeat.iter().sum::<f64>() / ns_per_call_per_repeat.len() as f64;
        let variance = ns_per_call_per_repeat.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / ns_per_call_per_repeat.len() as f64;
        let cov = variance.sqrt() / mean * 100.0;
        Timed { mean_ns: mean, cov_pct: cov, samples: ns_per_call_per_repeat }
    }

    fn report(config: &str, shape: &str, m: usize, macs_per_pass: f64, timed: &Timed, gate_state: &str) {
        let gmac_s = macs_per_pass / (timed.mean_ns / 1e9) / 1e9;
        let cov_flag = if timed.cov_pct > 5.0 { " (CoV>5%, range not point mean)" } else { "" };
        let range = if timed.cov_pct > 5.0 {
            let min = timed.samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = timed.samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            format!(" range=[{min:.1},{max:.1}]ns")
        } else {
            String::new()
        };
        println!(
            "{config:<22} | {shape:<9} | M={m:<2} | ns/call={:>10.1} | GMAC/s={:>8.3} | CoV={:>6.2}%{cov_flag}{range} | gate={gate_state}",
            timed.mean_ns, gmac_s, timed.cov_pct
        );
    }

    fn shape_ms() -> [usize; 4] {
        [1, 7, 8, 9]
    }

    macro_rules! run_config {
        ($rows:literal, $vecs:literal, $label:expr, $gate_state:expr) => {{
            let tile_cols = $vecs * 4;
            for &(shape_name, k_total, n) in &[("QKVO", 384usize, 384usize), ("FFN-up", 384usize, 1536usize), ("FFN-down", 1536usize, 384usize)] {
                let a_full = deterministic_data(9 * k_total, 0x1000_0000u32.wrapping_add(($rows * 31 + $vecs) as u32));
                let b_full = deterministic_data(k_total * n, 0x2000_0000u32.wrapping_add(($rows * 31 + $vecs) as u32));
                let (packed_b, col_tiles) = pack_panels(&b_full, k_total, n, tile_cols);
                let macs_per_invocation = ($rows * $vecs * 4 * k_total) as f64;

                for &m in &shape_ms() {
                    let row_tiles = m / $rows;
                    if row_tiles == 0 {
                        println!("{:<22} | {shape_name:<9} | M={m:<2} | n/a: M < ROWS={}", $label, $rows);
                        continue;
                    }
                    let macs_per_pass = macs_per_invocation * (row_tiles * col_tiles) as f64;
                    let a_needed = &a_full[..row_tiles * $rows * k_total];
                    let timed = time_calls(|| {
                        run_shape_pass::<$rows, $vecs>(a_needed, k_total, row_tiles, &packed_b, col_tiles);
                    });
                    report($label, shape_name, m, macs_per_pass, &timed, $gate_state);
                }
            }
        }};
    }

    pub fn run(gate_state: &str) {
        println!("bge_width_tile_accs: width-tile accumulator sweep -- ROWS*VECS registers, packed weights, M in {{1,7,8,9}}");
        println!(
            "PRE-REGISTRATION (see file doc comment): VECS=6 beats VECS=4 by >=1.15x at M=8 on QKVO/FFN; VECS=5 lands between; any config spilling past 32 vector registers regresses (none in this sweep spill)."
        );
        println!("gate_state={gate_state}");
        println!();

        run_config!(4, 4, "ROWS4_VECS4_incumbent", gate_state);
        run_config!(4, 5, "ROWS4_VECS5", gate_state);
        run_config!(4, 6, "ROWS4_VECS6", gate_state);
        run_config!(2, 6, "ROWS2_VECS6_remainder", gate_state);
        run_config!(1, 6, "ROWS1_VECS6_remainder", gate_state);
    }
}

#[cfg(target_arch = "aarch64")]
fn main() {
    // the launching shell already ran the 60s-apart pgrep quiet-gate before
    // invoking this binary; this string is the label baked into the run.
    let gate_state = std::env::var("ACCS_GATE_STATE").unwrap_or_else(|_| "unlabeled".to_string());
    sweep::run(&gate_state);
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    println!("bge_width_tile_accs: aarch64-only sweep, skipping on this target");
}
