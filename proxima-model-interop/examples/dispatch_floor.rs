//! What the dispatch stream of one decode step costs, by replaying it under
//! changed conditions, with a per-dispatch table for every run.
//!
//! One decode runs with `PROXIMA_CAPTURE_LIVE` on a chosen step; its dispatches
//! are kept. Then `DF_ROUNDS` rounds, each replaying the whole step as ONE
//! command buffer under every arm, in an order that rotates so box state hits
//! every arm alike. `DF_MODE=floor` (serial capture):
//! - `recorded`: the captured step as it ran, on a serial encoder.
//! - `empty_same_bindings`: every kernel body replaced by an empty kernel that
//!   declares the same buffer bindings and static threadgroup memory, launched
//!   with the same grid, threadgroup shape and dynamic threadgroup length.
//! - `empty_minimal_bindings`: one shared empty kernel, one small buffer, no
//!   threadgroup memory, the same grids and threadgroup shapes.
//! - `empty_one_threadgroup`: `empty_same_bindings` with every grid cut to a
//!   single threadgroup of the same shape.
//! - `recorded_unordered`: the recorded kernels on a concurrent encoder with no
//!   barrier; the output is not valid, it is the overlap ceiling.
//!
//! `DF_MODE=barriers` (capture under concurrent dispatch, each dispatch marked
//! with whether the live hazard tracker fired a barrier ahead of it):
//! - `live_tracker`: a buffer barrier exactly where the live tracker fired one.
//! - `node_keyed`: a barrier only where a dispatch reads a node an earlier
//!   dispatch wrote with no barrier between them, from each record's bindings.
//! - `no_barriers`: concurrent dispatch with none; the output is not valid.
//! - `serial_encoder`: the serial replay the other arms' output is compared with.
//!
//! `DF_MODE=stamped` (serial capture; one structured `dispatch_stamp` event per
//! stamped dispatch to `DF_STAMP_EVENTS_FILE`):
//! - `recorded`: the captured step on one serial encoder (the reference).
//! - `split_unstamped`: one serial encoder per dispatch in one command buffer, nothing sampled.
//! - `split_stamped`: the same split, each encoder sampling start and end GPU timestamps.
//! - `empty_split_stamped`: the split and stamped replay with every kernel empty, the
//!   instrument's reading of a dispatch with no work in it.
//! - `empty_same_bindings`: the one-encoder empty stream of `floor` mode.
//!
//! `dispatch_instream.csv` and `dispatch_instream_groups.csv` carry the per-dispatch and
//! per-kernel medians over the rounds beside the isolated marginal.
//!
//! The per-dispatch table (`DF_OUT_DIR/dispatch_table.csv`) carries, per
//! dispatch, its kernel, grid, threadgroups, bindings, bytes bound, threadgroup
//! memory, pipeline switch, barrier flags, the isolated marginal GPU time of its
//! kernel group (a batch of identical dispatches in one serial encoder) and each
//! arm's whole-step median divided by the dispatch count, a mean and not an
//! attribution. `DF_OUT_DIR/group_rollup.csv` sums the table per kernel.
//! Every timed cell prints wall, process CPU, CPU%, peak RSS, physical
//! footprint, Metal allocated bytes and load before and after
//! (`cell_resources/cell.rs`). Knobs: `DF_MODE`, `DF_STEP` (23), `DF_ROUNDS`
//! (21), `DF_OUT_DIR`, `PROXIMA_DECODE_MODEL_GGUF`, `PROXIMA_PROMPT_FILE`.
#![allow(clippy::expect_used, clippy::unwrap_used)]

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "cell_resources/cell.rs"]
mod cell;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod harness {
    use core::ops::ControlFlow;
    use std::collections::{BTreeMap, BTreeSet};
    use std::fmt::Write as _;
    use std::fs::File;
    use std::sync::Arc;

    use memmap2::{Mmap, MmapOptions};
    use omega::CapturedDispatch;
    use proxima_gguf::parse_complete;
    use proxima_gguf::types::GgmlType;
    use proxima_telemetry::export::Exporter;
    use proxima_telemetry::recorder::Recorder;
    use conflaguration::Settings as _;
    use proxima_model_interop::{
        GPU_LAYERS_ALL, LoadedModel, ServingConfig, SpeculativeConfig, SpeculativeSettings, TokenEvent,
    };

    use super::cell::Cell;

    const ISOLATED_BATCH: usize = 8;
    const ISOLATED_REPEATS: usize = 7;
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
        sorted[sorted.len() / 2]
    }

    fn cov_percent(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
            / (values.len() as f64 - 1.0).max(1.0);
        100.0 * variance.sqrt() / mean
    }

    fn fnv(hash: u64, bytes: &[u8]) -> u64 {
        bytes.iter().fold(hash, |state, byte| {
            (state ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    fn serving_config(dispatch_type: omega::DispatchType, speculative: SpeculativeConfig<'_>) -> ServingConfig<'_> {
        ServingConfig {
            speculative,
            gpu_layers: GPU_LAYERS_ALL,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            reasoning_budget: 0,
            dispatch_type,
            ..ServingConfig::default()
        }
    }

    fn spread(label: &str, values: &[f64]) {
        println!(
            "df {label} median={:.0} min={:.0} max={:.0}",
            median(values),
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
    }

    fn capture(model: &LoadedModel<'_>, prompt: &str, step: usize, concurrent: bool) -> Vec<CapturedDispatch> {
        // SAFETY: called before any thread other than the telemetry-free main one exists.
        unsafe {
            std::env::set_var("PROXIMA_CAPTURE_NODES", "all");
            std::env::set_var("PROXIMA_CAPTURE_STEPS", step.to_string());
            std::env::set_var("PROXIMA_CAPTURE_LIVE", "1");
        }
        let dispatch_type = if concurrent {
            omega::DispatchType::Concurrent
        } else {
            omega::DispatchType::Serial
        };
        let mut on_token = |_event: TokenEvent<'_>| ControlFlow::Continue(());
        let speculative_settings = SpeculativeSettings::from_env().expect("PROXIMA_SPECULATIVE_* env parses");
        println!("df speculative_types={}", speculative_settings.speculative_types);
        model
            .generate_streaming(
                prompt,
                step + 1,
                serving_config(dispatch_type, speculative_settings.as_speculative_config()),
                &mut on_token,
            )
            .expect("greedy decode");
        let dispatches: Vec<CapturedDispatch> = omega::take_captured_dispatches()
            .into_iter()
            .filter(|dispatch| dispatch.grid.threads > 0)
            .collect();
        // SAFETY: single-threaded here; later generations must not capture again.
        unsafe {
            std::env::remove_var("PROXIMA_CAPTURE_NODES");
            std::env::remove_var("PROXIMA_CAPTURE_STEPS");
            std::env::remove_var("PROXIMA_CAPTURE_LIVE");
        }
        assert!(!dispatches.is_empty(), "N==0: nothing captured");
        assert!(
            dispatches.iter().all(|dispatch| dispatch.unreplayable.is_none()),
            "a captured dispatch is not replayable"
        );
        dispatches
    }

    /// `(entry, threads, uniform bytes hash)`: dispatches sharing it do the same work.
    fn group_key(dispatch: &CapturedDispatch) -> (String, u64, u64) {
        (
            dispatch.entry.clone(),
            dispatch.grid.threads,
            fnv(FNV_OFFSET, &dispatch.uniform_bytes),
        )
    }

    fn isolated_marginal_ns(dispatch: &CapturedDispatch) -> f64 {
        let span = |batch: usize| -> f64 {
            let samples: Vec<f64> = (0..ISOLATED_REPEATS)
                .map(|_| dispatch.time_gpu_ns(batch).expect("isolated replay"))
                .collect();
            median(&samples)
        };
        (span(ISOLATED_BATCH) - span(1)) / (ISOLATED_BATCH - 1) as f64
    }

    /// Median over `ISOLATED_REPEATS` of one dispatch alone in a command buffer after a cache flush, less the same
    /// without the flush: the command buffer floor cancels, what remains is the weights coming from DRAM.
    fn cold_penalty_ns(dispatch: &CapturedDispatch) -> f64 {
        let flush_bytes = env_usize("DF_FLUSH_MIB", 256) * 1024 * 1024;
        let sample = |flush: bool| -> f64 {
            let spans: Vec<f64> = (0..ISOLATED_REPEATS)
                .map(|_| {
                    if flush {
                        omega::flush_gpu_caches(flush_bytes).expect("flush gpu caches");
                    }
                    dispatch.time_gpu_ns(1).expect("isolated replay")
                })
                .collect();
            median(&spans)
        };
        sample(true) - sample(false)
    }

    /// The dispatch repeated `ISOLATED_BATCH` times alone, one stamped encoder each: the median duration of
    /// the repeats after the first, median over `ISOLATED_REPEATS` replays. Same instrument as the stream, so
    /// the per-encoder cost cancels against the in-stream stamp. `DF_STAMPED_ISOLATED=0` skips it (zeros).
    fn stamped_isolated_ns(dispatch: &CapturedDispatch) -> f64 {
        if env_usize("DF_STAMPED_ISOLATED", 1) == 0 {
            return 0.0;
        }
        let batch: Vec<&CapturedDispatch> = vec![dispatch; ISOLATED_BATCH];
        let per_replay: Vec<f64> = (0..ISOLATED_REPEATS)
            .map(|_| {
                let (_, stamps) = CapturedDispatch::time_gpu_sequence_split_ns(&batch, true, false)
                    .expect("stamped isolated replay");
                let durations: Vec<f64> = stamps.iter().skip(1).map(|stamp| (stamp[1] - stamp[0]) as f64).collect();
                median(&durations)
            })
            .collect();
        median(&per_replay)
    }

    fn longest_run(flags: &[bool]) -> usize {
        let mut longest = 0;
        let mut run = 0;
        for flag in flags {
            if *flag {
                run = 1;
            } else {
                run += 1;
            }
            longest = longest.max(run);
        }
        longest
    }

    enum Replay {
        Serial,
        Noop { minimal_bindings: bool, one_threadgroup: bool },
        Barriered { flags: Vec<bool>, prepass_barrier: bool },
        Split { sample: bool, empty_kernels: bool },
    }

    impl Replay {
        fn run(&self, dispatches: &[CapturedDispatch]) -> (f64, Vec<[u64; 2]>) {
            let span = match self {
                Self::Serial => CapturedDispatch::time_gpu_sequence_ns(dispatches),
                Self::Noop { minimal_bindings, one_threadgroup } => {
                    CapturedDispatch::time_gpu_noop_sequence_ns(dispatches, *minimal_bindings, *one_threadgroup)
                }
                Self::Barriered { flags, prepass_barrier } => {
                    CapturedDispatch::time_gpu_sequence_barriered_ns(dispatches, flags, *prepass_barrier)
                }
                Self::Split { sample, empty_kernels } => {
                    return CapturedDispatch::time_gpu_sequence_split_ns(dispatches, *sample, *empty_kernels)
                        .expect("split whole-step replay");
                }
            };
            (span.expect("whole-step replay"), Vec::new())
        }
    }

    struct Arm {
        label: &'static str,
        replay: Replay,
    }

    fn floor_arms(records: usize) -> Vec<Arm> {
        vec![
            Arm { label: "recorded", replay: Replay::Serial },
            Arm {
                label: "empty_same_bindings",
                replay: Replay::Noop { minimal_bindings: false, one_threadgroup: false },
            },
            Arm {
                label: "empty_minimal_bindings",
                replay: Replay::Noop { minimal_bindings: true, one_threadgroup: false },
            },
            Arm {
                label: "empty_one_threadgroup",
                replay: Replay::Noop { minimal_bindings: false, one_threadgroup: true },
            },
            Arm {
                label: "recorded_unordered",
                replay: Replay::Barriered { flags: vec![false; records], prepass_barrier: false },
            },
        ]
    }

    fn stamped_arms() -> Vec<Arm> {
        vec![
            Arm { label: "recorded", replay: Replay::Serial },
            Arm {
                label: "split_unstamped",
                replay: Replay::Split { sample: false, empty_kernels: false },
            },
            Arm {
                label: "split_stamped",
                replay: Replay::Split { sample: true, empty_kernels: false },
            },
            Arm {
                label: "empty_split_stamped",
                replay: Replay::Split { sample: true, empty_kernels: true },
            },
            Arm {
                label: "empty_same_bindings",
                replay: Replay::Noop { minimal_bindings: false, one_threadgroup: false },
            },
        ]
    }

    /// One dispatch of the stamped replay, medians over the rounds.
    struct InstreamRow {
        index: usize,
        node: u32,
        kernel: String,
        grid_threads: u64,
        threadgroups: u64,
        bindings: usize,
        bytes_bound_full: u64,
        static_threadgroup_bytes: usize,
        dynamic_threadgroup_bytes: usize,
        pipeline_switch: bool,
        isolated_ns: f64,
        round4_isolated_ns: f64,
        cold_penalty_ns: f64,
        stamped_isolated_ns: f64,
        instream_ns: f64,
        instream_min_ns: f64,
        instream_max_ns: f64,
        instream_cov_pct: f64,
        gap_ns: f64,
        empty_instream_ns: f64,
        empty_gap_ns: f64,
    }

    impl InstreamRow {
        fn excess_ns(&self) -> f64 {
            self.instream_ns - self.isolated_ns
        }

        fn ratio(&self) -> f64 {
            self.instream_ns / self.isolated_ns.max(1.0)
        }

        /// In-stream time less what the same instrument reads for an empty kernel with the same launch.
        fn work_ns(&self) -> f64 {
            self.instream_ns - self.empty_instream_ns
        }

        fn work_excess_ns(&self) -> f64 {
            self.work_ns() - self.isolated_ns
        }

        /// In-stream time less the same dispatch repeated alone under the same instrument.
        fn stamped_excess_ns(&self) -> f64 {
            self.instream_ns - self.stamped_isolated_ns
        }
    }

    /// Per dispatch, over the rounds: `(duration ns, gap ns)`. The gap of dispatch 0 is 0:
    /// nothing precedes it in the stamped span.
    fn duration_and_gap(rounds: &[Vec<[u64; 2]>], count: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let durations = (0..count)
            .map(|index| rounds.iter().map(|round| (round[index][1] - round[index][0]) as f64).collect())
            .collect();
        let gaps = (0..count)
            .map(|index| {
                rounds
                    .iter()
                    .map(|round| match index {
                        0 => 0.0,
                        _ => round[index][0].saturating_sub(round[index - 1][1]) as f64,
                    })
                    .collect()
            })
            .collect();
        (durations, gaps)
    }

    fn instream_rows(
        dispatches: &[CapturedDispatch],
        marginals: &BTreeMap<(String, u64, u64), f64>,
        stamped: &[Vec<[u64; 2]>],
        empty: &[Vec<[u64; 2]>],
        round4: &BTreeMap<usize, (u32, String, f64)>,
        cold_penalties: &[f64],
        stamped_isolated: &[f64],
    ) -> Vec<InstreamRow> {
        let (durations, gaps) = duration_and_gap(stamped, dispatches.len());
        let (empty_durations, empty_gaps) = duration_and_gap(empty, dispatches.len());
        let mut previous_pipeline = usize::MAX;
        dispatches
            .iter()
            .enumerate()
            .map(|(index, dispatch)| {
                let switch = dispatch.pipeline_identity() != previous_pipeline;
                previous_pipeline = dispatch.pipeline_identity();
                InstreamRow {
                    index,
                    node: dispatch.node,
                    kernel: dispatch.entry.clone(),
                    grid_threads: dispatch.grid.threads,
                    threadgroups: dispatch.threadgroup_count(),
                    bindings: dispatch.buffer_binding_count(),
                    bytes_bound_full: dispatch.bound_buffer_bytes_full(),
                    static_threadgroup_bytes: dispatch.pipeline_resources().0,
                    dynamic_threadgroup_bytes: dispatch.dynamic_threadgroup_bytes(),
                    pipeline_switch: switch,
                    isolated_ns: marginals[&group_key(dispatch)],
                    cold_penalty_ns: cold_penalties[index],
                    stamped_isolated_ns: stamped_isolated[index],
                    round4_isolated_ns: round4
                        .get(&index)
                        .filter(|(node, kernel, _)| *node == dispatch.node && *kernel == dispatch.entry)
                        .map_or(f64::NAN, |(_, _, isolated)| *isolated),
                    instream_ns: median(&durations[index]),
                    instream_min_ns: durations[index].iter().copied().fold(f64::INFINITY, f64::min),
                    instream_max_ns: durations[index].iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    instream_cov_pct: cov_percent(&durations[index]),
                    gap_ns: median(&gaps[index]),
                    empty_instream_ns: median(&empty_durations[index]),
                    empty_gap_ns: median(&empty_gaps[index]),
                }
            })
            .collect()
    }

    fn write_instream_csv(directory: &str, rows: &[InstreamRow]) {
        let mut table = String::from(
            "index,node,kernel,grid_threads,threadgroups,buffer_bindings,bytes_bound_full,static_threadgroup_bytes,dynamic_threadgroup_bytes,pipeline_switch,isolated_marginal_ns,round4_isolated_marginal_ns,cold_penalty_ns,stamped_isolated_ns,instream_ns_median,instream_ns_min,instream_ns_max,instream_cov_pct,gap_ns_median,empty_instream_ns_median,empty_gap_ns_median,instream_minus_isolated_ns,instream_over_isolated,work_ns,work_minus_isolated_ns,stamped_excess_ns\n",
        );
        for row in rows {
            writeln!(
                table,
                "{},{},{},{},{},{},{},{},{},{},{:.0},{:.0},{:.0},{:.0},{:.0},{:.0},{:.0},{:.2},{:.0},{:.0},{:.0},{:.0},{:.3},{:.0},{:.0},{:.0}",
                row.index,
                row.node,
                row.kernel,
                row.grid_threads,
                row.threadgroups,
                row.bindings,
                row.bytes_bound_full,
                row.static_threadgroup_bytes,
                row.dynamic_threadgroup_bytes,
                u8::from(row.pipeline_switch),
                row.isolated_ns,
                row.round4_isolated_ns,
                row.cold_penalty_ns,
                row.stamped_isolated_ns,
                row.instream_ns,
                row.instream_min_ns,
                row.instream_max_ns,
                row.instream_cov_pct,
                row.gap_ns,
                row.empty_instream_ns,
                row.empty_gap_ns,
                row.excess_ns(),
                row.ratio(),
                row.work_ns(),
                row.work_excess_ns(),
                row.stamped_excess_ns()
            )
            .unwrap();
        }
        std::fs::write(format!("{directory}/dispatch_instream.csv"), table).expect("write dispatch_instream");
    }

    fn write_instream_groups(directory: &str, rows: &[InstreamRow]) {
        let mut groups: BTreeMap<&str, Vec<&InstreamRow>> = BTreeMap::new();
        for row in rows {
            groups.entry(row.kernel.as_str()).or_default().push(row);
        }
        let mut table = String::from(
            "kernel,dispatches,threadgroups_mean,single_threadgroup_dispatches,isolated_ms,instream_ms,gap_ms,empty_instream_ms,empty_gap_ms,instream_minus_isolated_ms,instream_over_isolated,work_ms,work_minus_isolated_ms,cold_penalty_ms,stamped_isolated_ms,stamped_excess_ms\n",
        );
        for (kernel, members) in &groups {
            let sum = |field: fn(&InstreamRow) -> f64| members.iter().map(|row| field(row)).sum::<f64>() / 1e6;
            let (isolated, instream) = (sum(|row| row.isolated_ns), sum(|row| row.instream_ns));
            writeln!(
                table,
                "{kernel},{},{:.1},{},{isolated:.4},{instream:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.4},{:.4},{:.4},{:.4},{:.4}",
                members.len(),
                members.iter().map(|row| row.threadgroups as f64).sum::<f64>() / members.len() as f64,
                members.iter().filter(|row| row.threadgroups <= 1).count(),
                sum(|row| row.gap_ns),
                sum(|row| row.empty_instream_ns),
                sum(|row| row.empty_gap_ns),
                instream - isolated,
                instream / isolated.max(f64::MIN_POSITIVE),
                sum(InstreamRow::work_ns),
                sum(InstreamRow::work_excess_ns),
                sum(|row| row.cold_penalty_ns),
                sum(|row| row.stamped_isolated_ns),
                sum(InstreamRow::stamped_excess_ns)
            )
            .unwrap();
        }
        std::fs::write(format!("{directory}/dispatch_instream_groups.csv"), table).expect("write instream groups");
    }

    fn print_top(label: &str, rows: &[InstreamRow], min_isolated_ns: f64, key: fn(&InstreamRow) -> f64, count: usize) {
        let mut ranked: Vec<&InstreamRow> = rows.iter().filter(|row| row.isolated_ns >= min_isolated_ns).collect();
        ranked.sort_by(|left, right| key(right).partial_cmp(&key(left)).expect("finite"));
        for row in ranked.into_iter().take(count) {
            println!(
                "df top {label} index={} node={} kernel={} threadgroups={} bindings={} bytes_bound_full={} isolated_ns={:.0} instream_ns={:.0} excess_ns={:.0} ratio={:.2} gap_ns={:.0} empty_instream_ns={:.0} work_ns={:.0} work_excess_ns={:.0} stamped_isolated_ns={:.0} stamped_excess_ns={:.0} cold_penalty_ns={:.0}",
                row.index,
                row.node,
                row.kernel,
                row.threadgroups,
                row.bindings,
                row.bytes_bound_full,
                row.isolated_ns,
                row.instream_ns,
                row.excess_ns(),
                row.ratio(),
                row.gap_ns,
                row.empty_instream_ns,
                row.work_ns(),
                row.work_excess_ns(),
                row.stamped_isolated_ns,
                row.stamped_excess_ns(),
                row.cold_penalty_ns
            );
        }
    }

    fn span_ms(rounds: &[Vec<[u64; 2]>]) -> Vec<f64> {
        rounds
            .iter()
            .map(|round| (round[round.len() - 1][1] - round[0][0]) as f64 / 1e6)
            .collect()
    }

    fn print_group_sums(label: &str, members: &[&InstreamRow]) {
        let sum_ms = |field: fn(&InstreamRow) -> f64| members.iter().map(|row| field(row)).sum::<f64>() / 1e6;
        println!(
            "df instream {label} dispatches={} isolated_ms={:.4} round4_isolated_ms={:.4} instream_ms={:.4} gap_ms={:.4} empty_instream_ms={:.4} empty_gap_ms={:.4} excess_ms={:.4} work_ms={:.4} work_excess_ms={:.4} cold_penalty_ms={:.4} stamped_isolated_ms={:.4} stamped_excess_ms={:.4}",
            members.len(),
            sum_ms(|row| row.isolated_ns),
            sum_ms(|row| row.round4_isolated_ns),
            sum_ms(|row| row.instream_ns),
            sum_ms(|row| row.gap_ns),
            sum_ms(|row| row.empty_instream_ns),
            sum_ms(|row| row.empty_gap_ns),
            sum_ms(InstreamRow::excess_ns),
            sum_ms(InstreamRow::work_ns),
            sum_ms(InstreamRow::work_excess_ns),
            sum_ms(|row| row.cold_penalty_ns),
            sum_ms(|row| row.stamped_isolated_ns),
            sum_ms(InstreamRow::stamped_excess_ns),
        );
    }

    fn print_instream_summary(rows: &[InstreamRow], stamped: &[Vec<[u64; 2]>], empty: &[Vec<[u64; 2]>]) {
        let (span, empty_span) = (span_ms(stamped), span_ms(empty));
        println!(
            "df instream spans stamped_p50_ms={:.4} stamped_cov_pct={:.2} empty_stamped_p50_ms={:.4} empty_stamped_cov_pct={:.2} round4_matched={}",
            median(&span),
            cov_percent(&span),
            median(&empty_span),
            cov_percent(&empty_span),
            rows.iter().filter(|row| !row.round4_isolated_ns.is_nan()).count()
        );
        let all: Vec<&InstreamRow> = rows.iter().collect();
        let singles: Vec<&InstreamRow> = rows.iter().filter(|row| row.threadgroups <= 1).collect();
        let others: Vec<&InstreamRow> = rows.iter().filter(|row| row.threadgroups > 1).collect();
        print_group_sums("all", &all);
        print_group_sums("single_threadgroup", &singles);
        print_group_sums("multi_threadgroup", &others);
        print_top("excess_absolute", rows, 0.0, InstreamRow::excess_ns, 40);
        print_top("excess_ratio_isolated_at_least_2us", rows, 2000.0, InstreamRow::ratio, 40);
        print_top("work_excess_absolute", rows, 0.0, InstreamRow::work_excess_ns, 40);
        print_top("stamped_excess_absolute", rows, 0.0, InstreamRow::stamped_excess_ns, 40);
    }

    fn barrier_arms(live: Vec<bool>, node: Vec<bool>, slot: Vec<bool>) -> Vec<Arm> {
        let records = live.len();
        vec![
            Arm { label: "live_tracker", replay: Replay::Barriered { flags: live, prepass_barrier: true } },
            Arm { label: "node_keyed", replay: Replay::Barriered { flags: node, prepass_barrier: true } },
            Arm { label: "node_and_slot_keyed", replay: Replay::Barriered { flags: slot, prepass_barrier: true } },
            Arm {
                label: "no_barriers",
                replay: Replay::Barriered { flags: vec![false; records], prepass_barrier: false },
            },
            Arm { label: "serial_encoder", replay: Replay::Serial },
        ]
    }

    fn digests(dispatches: &[CapturedDispatch]) -> (u64, u64) {
        let last = dispatches
            .last()
            .and_then(CapturedDispatch::output_bytes)
            .map_or(0, |bytes| fnv(FNV_OFFSET, &bytes));
        let all = dispatches
            .iter()
            .filter_map(CapturedDispatch::output_bytes)
            .fold(FNV_OFFSET, |state, bytes| fnv(state, &bytes));
        (last, all)
    }

    fn write_tables(
        directory: &str,
        dispatches: &[CapturedDispatch],
        flags: [&[bool]; 3],
        arm_labels: &[&str],
        arm_means_us: &[f64],
        marginals: &BTreeMap<(String, u64, u64), f64>,
    ) {
        std::fs::create_dir_all(directory).expect("create output directory");
        let mut table = String::from(
            "index,node,kernel,grid_threads,threadgroups,buffer_bindings,bytes_bound_full,static_threadgroup_bytes,dynamic_threadgroup_bytes,pipeline_switch,live_barrier_before,node_keyed_barrier_before,node_and_slot_keyed_barrier_before,isolated_marginal_ns",
        );
        for label in arm_labels {
            write!(table, ",{label}_mean_us").unwrap();
        }
        table.push('\n');
        let mut rollup: BTreeMap<String, (usize, f64, f64, usize, usize)> = BTreeMap::new();
        let mut previous_pipeline = usize::MAX;
        for (index, dispatch) in dispatches.iter().enumerate() {
            let marginal = marginals[&group_key(dispatch)];
            let switch = dispatch.pipeline_identity() != previous_pipeline;
            previous_pipeline = dispatch.pipeline_identity();
            write!(
                table,
                "{index},{},{},{},{},{},{},{},{},{},{},{},{},{marginal:.0}",
                dispatch.node,
                dispatch.entry,
                dispatch.grid.threads,
                dispatch.threadgroup_count(),
                dispatch.buffer_binding_count(),
                dispatch.bound_buffer_bytes_full(),
                dispatch.pipeline_resources().0,
                dispatch.dynamic_threadgroup_bytes(),
                u8::from(switch),
                u8::from(flags[0][index]),
                u8::from(flags[1][index]),
                u8::from(flags[2][index]),
            )
            .unwrap();
            for mean in arm_means_us {
                write!(table, ",{mean:.3}").unwrap();
            }
            table.push('\n');
            let row = rollup.entry(dispatch.entry.clone()).or_default();
            row.0 += 1;
            row.1 += marginal;
            row.2 += dispatch.threadgroup_count() as f64;
            row.3 += usize::from(dispatch.threadgroup_count() <= 1);
            row.4 += usize::from(switch);
        }
        std::fs::write(format!("{directory}/dispatch_table.csv"), table).expect("write dispatch table");
        let mut groups = String::from(
            "kernel,dispatches,isolated_marginal_ms_sum,threadgroups_mean,single_threadgroup_dispatches,pipeline_switches",
        );
        for label in arm_labels {
            write!(groups, ",{label}_mean_ms_sum").unwrap();
        }
        groups.push('\n');
        for (kernel, (count, marginal_sum, groups_sum, singles, switches)) in &rollup {
            write!(
                groups,
                "{kernel},{count},{:.4},{:.1},{singles},{switches}",
                marginal_sum / 1e6,
                groups_sum / *count as f64
            )
            .unwrap();
            for mean in arm_means_us {
                write!(groups, ",{:.4}", mean * *count as f64 / 1e3).unwrap();
            }
            groups.push('\n');
        }
        std::fs::write(format!("{directory}/group_rollup.csv"), groups).expect("write group rollup");
    }

    type StampRecorder = Arc<Recorder<proxima_telemetry::clock::GlobalClock>>;

    /// `DF_STAMP_EVENTS_FILE=<path>` points a file-sink exporter at the process recorder so the
    /// per-dispatch `dispatch_stamp` events of the stamped arms land in a file. The ring is
    /// drained from this thread between replays, never during one.
    fn install_stamp_recorder() -> Option<StampRecorder> {
        let path = std::env::var("DF_STAMP_EVENTS_FILE").ok()?;
        proxima_telemetry::emit::global::install(proxima_telemetry::emit::EnvFilter::parse(
            "dispatch_floor=info,omega::metal::arena_encode_dispatch_finish=debug",
        ));
        let exporter = Exporter::file(path);
        Some(
            Recorder::builder()
                .ring_capacity(2048)
                .export(exporter)
                .expect("file exporter installs")
                .install()
                .expect("telemetry recorder installs"),
        )
    }

    fn drain(recorder: Option<&StampRecorder>) {
        if let Some(recorder) = recorder {
            let mut drained = 0;
            loop {
                let batch = recorder.drain();
                drained += batch;
                if batch == 0 {
                    break;
                }
            }
            println!("df events_drained={drained}");
        }
    }

    /// `DF_ISOLATED_TABLE=<dispatch_table.csv>` is an earlier run's table: `index -> (node, kernel, isolated_marginal_ns)`.
    fn round4_table() -> BTreeMap<usize, (u32, String, f64)> {
        let Ok(path) = std::env::var("DF_ISOLATED_TABLE") else {
            return BTreeMap::new();
        };
        let text = std::fs::read_to_string(&path).expect("read DF_ISOLATED_TABLE");
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().expect("table header").split(',').collect();
        let column = |name: &str| header.iter().position(|candidate| *candidate == name).expect("column present");
        let (index, node, kernel, isolated) =
            (column("index"), column("node"), column("kernel"), column("isolated_marginal_ns"));
        lines
            .map(|line| {
                let fields: Vec<&str> = line.split(',').collect();
                (
                    fields[index].parse().expect("index"),
                    (fields[node].parse().expect("node"), fields[kernel].to_string(), fields[isolated].parse().expect("isolated")),
                )
            })
            .collect()
    }

    fn report_instream(
        directory: &str,
        dispatches: &[CapturedDispatch],
        marginals: &BTreeMap<(String, u64, u64), f64>,
        labels: &[&str],
        stamps: &[Vec<Vec<[u64; 2]>>],
    ) {
        let rounds_of = |label: &str| {
            let arm = labels.iter().position(|candidate| *candidate == label).expect("arm exists");
            assert!(!stamps[arm].is_empty(), "N==0: arm {label} produced no stamps");
            &stamps[arm]
        };
        let (stamped, empty) = (rounds_of("split_stamped"), rounds_of("empty_split_stamped"));
        let cold_penalties: Vec<f64> = dispatches.iter().map(cold_penalty_ns).collect();
        let stamped_isolated: Vec<f64> = dispatches.iter().map(stamped_isolated_ns).collect();
        let rows = instream_rows(dispatches, marginals, stamped, empty, &round4_table(), &cold_penalties, &stamped_isolated);
        write_instream_csv(directory, &rows);
        write_instream_groups(directory, &rows);
        print_instream_summary(&rows, stamped, empty);
    }

    pub fn run() {
        let step = env_usize("DF_STEP", 23);
        let rounds = env_usize("DF_ROUNDS", 21);
        let mode = std::env::var("DF_MODE").unwrap_or_else(|_| "floor".to_string());
        let barriers_mode = mode == "barriers";
        let stamped_mode = mode == "stamped";
        let directory = std::env::var("DF_OUT_DIR").unwrap_or_else(|_| "df_out".to_string());
        let path = std::env::var("PROXIMA_DECODE_MODEL_GGUF").expect("PROXIMA_DECODE_MODEL_GGUF");
        let file = File::open(path).expect("open checkpoint");
        // SAFETY: read-only mapping of a checkpoint no other process writes.
        let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map checkpoint");
        let parsed = parse_complete(&bytes).expect("parse header");
        let model = LoadedModel::load(&parsed, &bytes).expect("bind model");
        let prompt = std::fs::read_to_string(
            std::env::var("PROXIMA_PROMPT_FILE").expect("PROXIMA_PROMPT_FILE"),
        )
        .expect("read prompt file");
        let dispatches = capture(&model, &prompt, step, barriers_mode);
        let live_flags = CapturedDispatch::recorded_barriers(&dispatches);
        let node_flags = CapturedDispatch::node_keyed_barriers(&dispatches);
        let slot_flags = CapturedDispatch::node_and_slot_keyed_barriers(&dispatches);
        let prepass_records = dispatches
            .iter()
            .filter(|dispatch| dispatch.route_prepass_dispatches > 0)
            .count();
        let physical: usize = dispatches.len()
            + dispatches
                .iter()
                .map(|dispatch| dispatch.route_prepass_dispatches as usize)
                .sum::<usize>();
        let identities: BTreeSet<usize> =
            dispatches.iter().map(CapturedDispatch::pipeline_identity).collect();
        let bindings: Vec<f64> = dispatches
            .iter()
            .map(|dispatch| dispatch.buffer_binding_count() as f64)
            .collect();
        let bound_bytes: Vec<f64> = dispatches
            .iter()
            .map(|dispatch| dispatch.bound_buffer_bytes_full() as f64)
            .collect();
        let groups: Vec<f64> = dispatches
            .iter()
            .map(|dispatch| dispatch.threadgroup_count() as f64)
            .collect();
        let single_threadgroup = groups.iter().filter(|count| **count <= 1.0).count();
        let switches = dispatches
            .windows(2)
            .filter(|pair| pair[0].pipeline_identity() != pair[1].pipeline_identity())
            .count();
        println!(
            "df capture mode={mode} step={step} records={} physical_dispatches={physical} prepass_records={prepass_records} rounds={rounds}",
            dispatches.len()
        );
        println!("df pipeline_states distinct={} pipeline_switches_between_dispatches={switches}", identities.len());
        if !barriers_mode {
            let (_, original_static, empty_static) =
                CapturedDispatch::floor_pipeline_resources(&dispatches).expect("empty kernels compile");
            println!(
                "df static_threadgroup_bytes_over_distinct_pipelines original={original_static} empty={empty_static}"
            );
        }
        spread("buffer_bindings_per_dispatch", &bindings);
        spread("threadgroups_per_dispatch", &groups);
        spread("bound_bytes_per_dispatch_full", &bound_bytes);
        println!("df single_threadgroup_dispatches={single_threadgroup}");
        println!(
            "df barriers live_tracker={} node_keyed={} node_and_slot_keyed={} live_only={} node_only={} longest_barrier_free_run_live={} longest_barrier_free_run_node_keyed={} longest_barrier_free_run_node_and_slot_keyed={}",
            live_flags.iter().filter(|flag| **flag).count(),
            node_flags.iter().filter(|flag| **flag).count(),
            slot_flags.iter().filter(|flag| **flag).count(),
            live_flags.iter().zip(&node_flags).filter(|(live, node)| **live && !**node).count(),
            live_flags.iter().zip(&node_flags).filter(|(live, node)| !**live && **node).count(),
            longest_run(&live_flags),
            longest_run(&node_flags),
            longest_run(&slot_flags)
        );
        let arms = if barriers_mode {
            barrier_arms(live_flags.clone(), node_flags.clone(), slot_flags.clone())
        } else if stamped_mode {
            stamped_arms()
        } else {
            floor_arms(dispatches.len())
        };
        if barriers_mode {
            for arm in [&arms[0], &arms[0], &arms[1], &arms[2], &arms[3], &arms[4]] {
                arm.replay.run(&dispatches);
                let (last, all) = digests(&dispatches);
                println!("df digest arm={} last_output={last:016x} all_outputs={all:016x}", arm.label);
            }
        }
        let mut marginals: BTreeMap<(String, u64, u64), f64> = BTreeMap::new();
        for dispatch in &dispatches {
            marginals
                .entry(group_key(dispatch))
                .or_insert_with(|| isolated_marginal_ns(dispatch));
        }
        println!(
            "df isolated_marginal groups={} sum_over_dispatches_ms={:.4}",
            marginals.len(),
            dispatches.iter().map(|dispatch| marginals[&group_key(dispatch)]).sum::<f64>() / 1e6
        );
        let recorder = stamped_mode.then(install_stamp_recorder).flatten();
        if stamped_mode {
            for arm in &arms {
                arm.replay.run(&dispatches);
            }
            drain(recorder.as_ref());
        }
        let mut spans: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        let mut stamps: Vec<Vec<Vec<[u64; 2]>>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            for offset in 0..arms.len() {
                let arm = (round + offset) % arms.len();
                proxima_telemetry::info!(arm = arms[arm].label, round = round as u64, "dispatch_stamp_replay");
                let cell = Cell::begin();
                let (span, round_stamps) = arms[arm].replay.run(&dispatches);
                let stamped_span_ms = round_stamps.last().zip(round_stamps.first()).map(|(last, first)| (last[1] - first[0]) as f64 / 1e6);
                if let Some(stamped_ms) = stamped_span_ms {
                    assert!(
                        (stamped_ms - span / 1e6).abs() <= 0.02 * span / 1e6,
                        "stamp unit check: first start to last end {stamped_ms} ms against command buffer span {} ms",
                        span / 1e6
                    );
                }
                println!(
                    "df replay arm={} round={round} gpu_span_ms={:.4} stamped_first_start_to_last_end_ms={stamped_span_ms:?}",
                    arms[arm].label,
                    span / 1e6
                );
                println!("{}", cell.end(arms[arm].label));
                drain(recorder.as_ref());
                spans[arm].push(span / 1e6);
                if !round_stamps.is_empty() {
                    stamps[arm].push(round_stamps);
                }
            }
        }
        let mut means_us = Vec::new();
        for (arm, values) in arms.iter().zip(&spans) {
            let p50 = median(values);
            means_us.push(p50 * 1e3 / dispatches.len() as f64);
            println!(
                "df summary arm={} n={} p50_ms={p50:.4} min={:.4} max={:.4} cov_pct={:.2} us_per_record={:.3} us_per_physical_dispatch={:.3}",
                arm.label,
                values.len(),
                values.iter().copied().fold(f64::INFINITY, f64::min),
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                cov_percent(values),
                p50 * 1e3 / dispatches.len() as f64,
                p50 * 1e3 / physical as f64
            );
        }
        let labels: Vec<&str> = arms.iter().map(|arm| arm.label).collect();
        write_tables(&directory, &dispatches, [&live_flags, &node_flags, &slot_flags], &labels, &means_us, &marginals);
        let p50: Vec<f64> = spans.iter().map(|values| median(values)).collect();
        let line: Vec<String> = labels
            .iter()
            .zip(&p50)
            .map(|(label, value)| format!("{label}={value:.4}"))
            .collect();
        println!("df p50_ms {}", line.join(" "));
        if stamped_mode {
            report_instream(&directory, &dispatches, &marginals, &labels, &stamps);
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    harness::run();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("unsupported target for this example");
    std::process::exit(1);
}
