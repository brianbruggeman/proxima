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

    use memmap2::{Mmap, MmapOptions};
    use omega::CapturedDispatch;
    use proxima_gguf::parse_complete;
    use proxima_gguf::types::GgmlType;
    use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig, TokenEvent};

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

    fn serving_config(dispatch_type: omega::DispatchType) -> ServingConfig<'static> {
        ServingConfig {
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
        model
            .generate_streaming(prompt, step + 1, serving_config(dispatch_type), &mut on_token)
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
    }

    impl Replay {
        fn run(&self, dispatches: &[CapturedDispatch]) -> f64 {
            match self {
                Self::Serial => CapturedDispatch::time_gpu_sequence_ns(dispatches),
                Self::Noop { minimal_bindings, one_threadgroup } => {
                    CapturedDispatch::time_gpu_noop_sequence_ns(dispatches, *minimal_bindings, *one_threadgroup)
                }
                Self::Barriered { flags, prepass_barrier } => {
                    CapturedDispatch::time_gpu_sequence_barriered_ns(dispatches, flags, *prepass_barrier)
                }
            }
            .expect("whole-step replay")
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

    pub fn run() {
        let step = env_usize("DF_STEP", 23);
        let rounds = env_usize("DF_ROUNDS", 21);
        let mode = std::env::var("DF_MODE").unwrap_or_else(|_| "floor".to_string());
        let barriers_mode = mode == "barriers";
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
        let mut spans: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            for offset in 0..arms.len() {
                let arm = (round + offset) % arms.len();
                let cell = Cell::begin();
                let span = arms[arm].replay.run(&dispatches);
                println!("df replay arm={} round={round} gpu_span_ms={:.4}", arms[arm].label, span / 1e6);
                println!("{}", cell.end(arms[arm].label));
                spans[arm].push(span / 1e6);
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
