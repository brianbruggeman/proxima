//! kernel-latency task (2026-09-01): mechanism verification, BEFORE any
//! attack on the gap `docs/discipline.md` ROW 213 named -- `gemm_width_tile_neon`
//! (`proxima-tensor/src/cpu.rs:9723`) runs at 11.0-14.5 GMAC/s in-graph vs
//! 48.0-48.8 GMAC/s isolated-warm (ROW 210, `bge_width_tile_accs.rs`). That
//! isolated number reuses the SAME packed-`b` buffer across all 250 timed
//! calls per cell -- cache-hot after the first touch, exactly the ROW 181
//! precedent (`bge_matmul_cache_regime.rs`) warns against as "the regime
//! that produced the misleading 48.5".
//!
//! This probe adds a CACHE-RESIDENT CONTROL: the same kernel, same shapes,
//! same `ROWS=4,VECS=4` production config, but the packed-`b` buffer is
//! rotated across `ROTATION=64` independently-allocated instances (ROW 181's
//! own method) so no weight buffer is touched twice within any window
//! smaller than 64 calls -- `64 * 2.25MiB(FFN) / 0.5625MiB(QKVO)` =
//! 144.0/36.0 MiB rotated, both comfortably exceeding any on-chip cache
//! tier on this machine.
//!
//! PRE-REGISTRATION (recorded before this file was ever run): if the kernel
//! is LATENCY-bound on the cold weight read (too few outstanding loads to
//! cover DRAM latency), `cold` GMAC/s should land near the in-graph figure
//! (11.0-14.5 GMAC/s, ROW 213) and `warm` GMAC/s should reproduce ROW 210's
//! 48.0-48.8 GMAC/s. If instead the kernel is issue-limited or something
//! else dominates, `cold` should land close to `warm` (within ~1.3x) since
//! an issue-bound kernel does not care whether its reads are cache-resident.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(target_arch = "aarch64")]
mod probe {
    use proxima_tensor::cpu::{KStridedTile, gemm_width_tile_neon};

    const ROTATION: usize = 64;
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

    /// same panel-major packing `bge_width_tile_accs.rs`'s own `pack_panels`
    /// and `PackedWidthPanels` use: panel `p`'s block starts at
    /// `p * k_total * tile_cols`, contiguous.
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

    fn run_shape_pass<const ROWS: usize, const VECS: usize>(
        a: &[f32],
        k_total: usize,
        row_tiles: usize,
        packed_b: &[f32],
        col_tiles: usize,
    ) {
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
                        KStridedTile {
                            data: a,
                            base: base_a,
                            k_stride: 1,
                        },
                        k_total as i64,
                        KStridedTile {
                            data: packed_b,
                            base: base_b,
                            k_stride: tile_cols as i64,
                        },
                        k_total,
                        &mut tile_out,
                    );
                }
                std::hint::black_box(&tile_out);
            }
        }
    }

    fn time_calls<F: FnMut(usize)>(mut call: F) -> Timed {
        let mut ns_per_call_per_repeat = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            for index in 0..CALLS_PER_REPEAT {
                call(index);
            }
            let elapsed = start.elapsed();
            ns_per_call_per_repeat.push(elapsed.as_nanos() as f64 / CALLS_PER_REPEAT as f64);
        }
        let mean = ns_per_call_per_repeat.iter().sum::<f64>() / ns_per_call_per_repeat.len() as f64;
        let variance = ns_per_call_per_repeat
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / ns_per_call_per_repeat.len() as f64;
        let cov = variance.sqrt() / mean * 100.0;
        Timed {
            mean_ns: mean,
            cov_pct: cov,
            samples: ns_per_call_per_repeat,
        }
    }

    fn report(label: &str, shape: &str, m: usize, macs_per_pass: f64, triad_bytes: f64, timed: &Timed) {
        let gmac_s = macs_per_pass / (timed.mean_ns / 1e9) / 1e9;
        let gb_s = triad_bytes / (timed.mean_ns / 1e9) / 1e9;
        let range = if timed.cov_pct > 5.0 {
            let min = timed.samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = timed
                .samples
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            format!(" range=[{min:.1},{max:.1}]ns (CoV>5%, range not point mean)")
        } else {
            String::new()
        };
        println!(
            "  {label:<8} | {shape:<9} | M={m:<2} | ns/call={:>10.1} | GMAC/s={:>8.3} | GB/s={:>7.3} | CoV={:>6.2}%{range}",
            timed.mean_ns, gmac_s, gb_s, timed.cov_pct
        );
    }

    /// existing form: ONE packed `b` buffer, reused every timed call --
    /// cache-hot after the first touch (ROW 210's own regime).
    fn warm_arm(m: usize, k_total: usize, n: usize) -> (Timed, f64, f64) {
        const ROWS: usize = 4;
        const VECS: usize = 4;
        let tile_cols = VECS * 4;
        let a_full = deterministic_data(m * k_total, 0x1000_0000);
        let b_full = deterministic_data(k_total * n, 0x2000_0000);
        let (packed_b, col_tiles) = pack_panels(&b_full, k_total, n, tile_cols);
        let row_tiles = m / ROWS;
        let macs_per_pass = (ROWS * VECS * 4 * k_total * row_tiles * col_tiles) as f64;
        let triad_bytes =
            ((k_total * tile_cols * col_tiles) + (m * k_total) + (row_tiles * ROWS * tile_cols * col_tiles)) as f64
                * 4.0;
        let a_needed = &a_full[..row_tiles * ROWS * k_total];
        let timed = time_calls(|_index| {
            run_shape_pass::<ROWS, VECS>(a_needed, k_total, row_tiles, &packed_b, col_tiles);
        });
        (timed, macs_per_pass, triad_bytes)
    }

    /// ROW 181 round-robin: `ROTATION` independently-allocated packed `b`
    /// buffers, round-robined across the timed loop -- no buffer touched
    /// twice within a `ROTATION`-call window.
    fn cold_arm(m: usize, k_total: usize, n: usize) -> (Timed, f64, f64) {
        const ROWS: usize = 4;
        const VECS: usize = 4;
        let tile_cols = VECS * 4;
        let a_full = deterministic_data(m * k_total, 0x1000_0000);
        let row_tiles = m / ROWS;
        let a_needed = a_full[..row_tiles * ROWS * k_total].to_vec();

        let buffers: Vec<(Vec<f32>, usize)> = (0..ROTATION)
            .map(|index| {
                let salt = 0x3000_0000u32.wrapping_add((index as u32).wrapping_mul(0x9e37_79b9));
                let b_full = deterministic_data(k_total * n, salt);
                pack_panels(&b_full, k_total, n, tile_cols)
            })
            .collect();
        let col_tiles = buffers[0].1;
        let macs_per_pass = (ROWS * VECS * 4 * k_total * row_tiles * col_tiles) as f64;
        let triad_bytes =
            ((k_total * tile_cols * col_tiles) + (m * k_total) + (row_tiles * ROWS * tile_cols * col_tiles)) as f64
                * 4.0;

        // untimed warm-up over the WHOLE rotation set -- forces first-touch
        // page faults without leaving any single buffer resident, since
        // ROTATION * shape-bytes already exceeds cache before timing starts.
        for (packed_b, tiles) in &buffers {
            run_shape_pass::<ROWS, VECS>(&a_needed, k_total, row_tiles, packed_b, *tiles);
        }

        let timed = time_calls(|index| {
            let (packed_b, tiles) = &buffers[index % ROTATION];
            run_shape_pass::<ROWS, VECS>(&a_needed, k_total, row_tiles, packed_b, *tiles);
        });
        (timed, macs_per_pass, triad_bytes)
    }

    /// owner-directed correction: single-shape ROTATION=64 rotates only
    /// 36-144 MiB, sized against L2 (12 MiB, `sysctl hw.perflevel0.l2cachesize`
    /// on this box), not against BGE's real 132.85 MB per-sentence working
    /// set. This arm interleaves all 3 real shapes at their real per-sentence
    /// call counts (QKVO x48 node-calls, FFN-up x12, FFN-down x12 -- 72
    /// `gemm_width_tile_neon`-dispatching nodes, matching `bge_route_census`'s
    /// own 96-call total minus the attn_qk/attn_v narrow-tile nodes this
    /// probe does not model) inside ONE timed "sentence pass", rotated across
    /// `SENTENCE_ROTATION=8` independently-allocated sentence buffer sets (8
    /// x ~81 MiB = ~648 MiB, far beyond any cache tier on this machine) so no
    /// node's weight buffer repeats within a 72-call window. No intervening
    /// non-GEMM work (softmax/LayerNorm/GELU) -- isolates "many concurrent
    /// cold streams" from "heterogeneous surrounding traffic" as a variable.
    const SENTENCE_ROTATION: usize = 8;
    const M_FIXED: usize = 8;

    struct SentenceNode {
        k_total: usize,
        packed_b: Vec<f32>,
        col_tiles: usize,
    }

    fn build_sentence(salt_base: u32) -> Vec<SentenceNode> {
        const ROWS: usize = 4;
        const VECS: usize = 4;
        let tile_cols = VECS * 4;
        let mut nodes = Vec::with_capacity(72);
        for index in 0..48 {
            let salt = salt_base.wrapping_add((index as u32).wrapping_mul(0x1234_5678));
            let b_full = deterministic_data(384 * 384, salt);
            let (packed_b, col_tiles) = pack_panels(&b_full, 384, 384, tile_cols);
            nodes.push(SentenceNode {
                k_total: 384,
                packed_b,
                col_tiles,
            });
        }
        for index in 0..12 {
            let salt = salt_base
                .wrapping_add(0x0aaa_0000)
                .wrapping_add((index as u32).wrapping_mul(0x1234_5678));
            let b_full = deterministic_data(384 * 1536, salt);
            let (packed_b, col_tiles) = pack_panels(&b_full, 384, 1536, tile_cols);
            nodes.push(SentenceNode {
                k_total: 384,
                packed_b,
                col_tiles,
            });
        }
        for index in 0..12 {
            let salt = salt_base
                .wrapping_add(0x0bbb_0000)
                .wrapping_add((index as u32).wrapping_mul(0x1234_5678));
            let b_full = deterministic_data(1536 * 384, salt);
            let (packed_b, col_tiles) = pack_panels(&b_full, 1536, 384, tile_cols);
            nodes.push(SentenceNode {
                k_total: 1536,
                packed_b,
                col_tiles,
            });
        }
        let _ = ROWS;
        nodes
    }

    fn interleaved_sentence_arm() -> (Timed, f64, f64) {
        const ROWS: usize = 4;
        const VECS: usize = 4;
        let row_tiles_384 = M_FIXED / ROWS;
        let a_384 = deterministic_data(row_tiles_384 * ROWS * 384, 0x1000_0000);
        let a_1536 = deterministic_data(row_tiles_384 * ROWS * 1536, 0x1000_0001);

        let sentences: Vec<Vec<SentenceNode>> = (0..SENTENCE_ROTATION)
            .map(|sentence_index| {
                build_sentence(0x5000_0000u32.wrapping_add((sentence_index as u32).wrapping_mul(0x9e37_79b9)))
            })
            .collect();

        let mut macs_per_sentence = 0f64;
        let mut bytes_per_sentence = 0f64;
        for node in &sentences[0] {
            let row_tiles = M_FIXED / ROWS;
            let macs = (ROWS * VECS * 4 * node.k_total * row_tiles * node.col_tiles) as f64;
            let bytes = ((node.k_total * VECS * 4 * node.col_tiles) + (M_FIXED * node.k_total)) as f64 * 4.0;
            macs_per_sentence += macs;
            bytes_per_sentence += bytes;
        }

        let run_one_sentence = |nodes: &[SentenceNode]| {
            for node in nodes {
                let a_needed = if node.k_total == 384 { &a_384 } else { &a_1536 };
                run_shape_pass::<ROWS, VECS>(
                    a_needed,
                    node.k_total,
                    row_tiles_384,
                    &node.packed_b,
                    node.col_tiles,
                );
            }
        };

        // untimed warm-up over the whole rotation set.
        for nodes in &sentences {
            run_one_sentence(nodes);
        }

        let timed = time_calls(|index| {
            run_one_sentence(&sentences[index % SENTENCE_ROTATION]);
        });
        (timed, macs_per_sentence, bytes_per_sentence)
    }

    pub fn run(gate_state: &str) {
        println!(
            "bge_width_tile_kernel_latency_probe: mechanism verification -- warm (ROW 210 regime, cache-hot) vs cold (ROW 181 round-robin, {ROTATION} distinct packed-b buffers)"
        );
        println!(
            "PRE-REGISTRATION: latency-bound predicts cold near 11.0-14.5 GMAC/s (ROW 213 in-graph) and warm near 48.0-48.8 GMAC/s (ROW 210); issue-limited predicts cold within ~1.3x of warm."
        );
        println!("gate_state={gate_state}");
        println!(
            "host L2 (perflevel0, sysctl hw.perflevel0.l2cachesize): 12 MiB shared/4 cores. single-shape ROTATION={ROTATION} rotates 36-144 MiB (clears L2, not necessarily SLC)."
        );
        println!();

        for &(shape_name, k_total, n) in &[
            ("QKVO", 384usize, 384usize),
            ("FFN-up", 384usize, 1536usize),
            ("FFN-down", 1536usize, 384usize),
        ] {
            for &m in &[7usize, 8, 9] {
                let (warm, warm_macs, warm_bytes) = warm_arm(m, k_total, n);
                report("warm", shape_name, m, warm_macs, warm_bytes, &warm);
                let (cold, cold_macs, cold_bytes) = cold_arm(m, k_total, n);
                report("cold", shape_name, m, cold_macs, cold_bytes, &cold);
                let slowdown = cold.mean_ns / warm.mean_ns;
                println!("    -> cold/warm slowdown: {slowdown:.3}x\n");
            }
        }

        println!(
            "=== interleaved multi-shape sentence pass: 72 node-calls (48 QKVO + 12 FFN-up + 12 FFN-down) at M={M_FIXED}, rotated across {SENTENCE_ROTATION} independent sentence buffer sets (~648 MiB total, clears any plausible cache tier) ==="
        );
        println!(
            "PRE-REGISTRATION: if 'many concurrent cold streams' (without intervening non-GEMM work) is sufficient to reproduce the in-graph gap, this arm should land near 11.0-15.3 GMAC/s; if it instead lands near the single-shape cold rate (34-48 GMAC/s), the residual gap implicates something NOT modeled here -- intervening non-GEMM traffic, real-arena TLB/allocation pattern, or heterogeneous shape scheduling."
        );
        let (interleaved, interleaved_macs, interleaved_bytes) = interleaved_sentence_arm();
        report(
            "sentence",
            "72-node",
            M_FIXED,
            interleaved_macs,
            interleaved_bytes,
            &interleaved,
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn main() {
    let gate_state =
        std::env::var("LAT_GATE_STATE").unwrap_or_else(|_| "unlabeled".to_string());
    probe::run(&gate_state);
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    println!("bge_width_tile_kernel_latency_probe: aarch64-only probe, skipping on this target");
}
