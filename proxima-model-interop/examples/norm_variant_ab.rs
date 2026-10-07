//! Interleaved A/B replay of kernel-body variants on captured gemma4-E2B
//! decode dispatches.
//!
//! The decode runs once; the dispatches of one step are captured live
//! (`omega::take_captured_dispatches`). For every `<sha16>.<tag>.metal` file in
//! `AB_VARIANT_DIR` whose `sha16` prefixes a captured kernel's `msl_sha256`,
//! the variant is compiled against that dispatch's own buffers and uniform
//! bytes (`CapturedDispatch::with_kernel_variant`) and timed against the
//! production kernel in alternating rounds inside this one process, so GPU
//! clock state and background load hit every arm alike. An optional
//! `<sha16>.<tag>.width` file holds the new threadgroup width; the thread
//! count scales with it. An optional `<sha16>.<tag>.scale` file holds `n/d`, a further
//! multiplier on the thread count, for a variant that changes how many rows one
//! simdgroup folds.
//!
//! Knobs: `AB_VARIANT_DIR`, `AB_STEP` (5), `AB_ROUNDS` (60), `AB_BATCH` (16),
//! `PROXIMA_PROMPT`.
//!
//! Packed-weight kernels (a `Q4_0` matvec) are skipped unless `AB_PACKED` is set.
//! With `AB_SEQUENCE` set, an arm is timed as one command buffer running every
//! captured member of the group in program order (each member reads its own weight
//! tensor, as the live step does), after `AB_FLUSH_MIB` (384) of CPU writes evict the
//! system cache; the figure is microseconds per dispatch. `AB_SHA` restricts the run to
//! groups whose kernel sha256 starts with the given prefix. With `AB_STEP_SEQUENCE` set
//! (implies `AB_SEQUENCE`), an arm is the whole captured step replayed in program order with
//! the group's members swapped for the variant, so a variant is judged by the step time it
//! moves with the real kernel interleaving and real weight streaming; set `AB_FLUSH_MIB=0`
//! there, the step streams more bytes than the system cache holds. With `AB_OMIT_ALL` set,
//! every kernel group is timed by omission instead: the step with the group's dispatches
//! removed against the whole step, one `ab omit` line per group (its in-situ cost on a
//! GPU kept busy, unlike an isolated single-dispatch command buffer).
#![allow(clippy::expect_used, clippy::unwrap_used)]

#[cfg(all(feature = "metal", target_os = "macos"))]
mod harness {
    use core::ops::ControlFlow;
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::path::PathBuf;

    use memmap2::{Mmap, MmapOptions};
    use omega::CapturedDispatch;
    use proxima_gguf::parse_complete;
    use proxima_gguf::types::GgmlType;
    use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig, TokenEvent};
    use proxima_tensor::NumericPolicy;

    const MODEL_ENV: &str = "PROXIMA_GEMMA4_E2B_GGUF";
    const SHA_PREFIX_CHARS: usize = 16;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn numeric_policy() -> NumericPolicy {
        match std::env::var("AB_NUMERIC").as_deref() {
            Ok("bit_exact") => NumericPolicy::bit_exact(),
            Ok(other) => panic!("AB_NUMERIC={other}: expected `bit_exact` or unset"),
            Err(_) => ServingConfig::default().numeric_policy,
        }
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
        values[values.len() / 2]
    }

    fn decode(step: usize) {
        let path = std::env::var(MODEL_ENV).expect("model path env");
        let file = File::open(path).expect("open gemma4-E2B blob");
        // SAFETY: read-only mapping of a checkpoint no other process writes.
        let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map blob");
        let parsed = parse_complete(&bytes).expect("parse header");
        let model = LoadedModel::load(&parsed, &bytes).expect("bind gemma4-E2B");
        let serving_config = ServingConfig {
            gpu_layers: GPU_LAYERS_ALL,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            reasoning_budget: 0,
            dispatch_type: omega::DispatchType::Serial,
            ..ServingConfig::default()
        };
        let prompt = std::fs::read_to_string(
            std::env::var("PROXIMA_PROMPT_FILE").expect("PROXIMA_PROMPT_FILE"),
        )
        .expect("read prompt file");
        let mut on_token = |_event: TokenEvent<'_>| ControlFlow::Continue(());
        model
            .generate_streaming(&prompt, step + 1, serving_config, &mut on_token)
            .expect("greedy decode");
    }

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct GroupKey {
        sha: String,
        threads: u64,
        width: Option<u64>,
        extents: Vec<u64>,
    }

    struct Arm {
        label: String,
        dispatches: Vec<CapturedDispatch>,
    }

    fn load_variants(
        dir: &PathBuf,
        members: &[&CapturedDispatch],
        base_width: u64,
    ) -> Vec<Arm> {
        let base = members[0];
        let needle = format!("kernel void {}(", base.entry);
        let mut arms = Vec::new();
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .expect("read variant dir")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.ends_with(".metal"))
            .filter(|name| name.starts_with(&base.msl_sha256[..SHA_PREFIX_CHARS]))
            .collect();
        names.sort();
        for name in names {
            let source = std::fs::read_to_string(dir.join(&name)).expect("read variant");
            if !source.contains(&needle) {
                continue;
            }
            let width_path = dir.join(name.replace(".metal", ".width"));
            let (threads, width) = match std::fs::read_to_string(&width_path) {
                Ok(text) => {
                    let width: u64 = text.trim().parse().expect("width integer");
                    (base.grid.threads / base_width * width, Some(width))
                }
                Err(_) => (base.grid.threads, base.grid.threadgroup_width),
            };
            let scale_path = dir.join(name.replace(".metal", ".scale"));
            let threads = match std::fs::read_to_string(&scale_path) {
                Ok(text) => {
                    let (numerator, denominator) = text.trim().split_once('/').expect("scale n/d");
                    let numerator: u64 = numerator.parse().expect("scale numerator integer");
                    let denominator: u64 = denominator.parse().expect("scale denominator integer");
                    threads * numerator / denominator
                }
                Err(_) => threads,
            };
            let template = base
                .with_kernel_variant(&source, &base.entry, threads, width, numeric_policy())
                .unwrap_or_else(|error| panic!("variant {name} does not compile: {error}"));
            let dispatches = members
                .iter()
                .map(|member| member.with_pipeline_of(&template))
                .collect();
            arms.push(Arm {
                label: name.trim_end_matches(".metal").to_string(),
                dispatches,
            });
        }
        arms
    }

    fn ulp_distance(left: f32, right: f32) -> u32 {
        let ordered = |value: f32| {
            let bits = value.to_bits() as i32;
            if bits < 0 {
                i32::MIN.wrapping_sub(bits)
            } else {
                bits
            }
        };
        ordered(left).abs_diff(ordered(right))
    }

    fn compare_outputs(base: &[u8], other: &[u8]) -> String {
        let to_floats = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()
        };
        let (left, right) = (to_floats(base), to_floats(other));
        let pairs = || left.iter().zip(&right);
        let differing = pairs()
            .filter(|(base_value, other_value)| base_value.to_bits() != other_value.to_bits())
            .count();
        let max_ulp = pairs()
            .map(|(base_value, other_value)| ulp_distance(*base_value, *other_value))
            .max()
            .unwrap_or(0);
        let max_abs = pairs()
            .map(|(base_value, other_value)| (base_value - other_value).abs())
            .fold(0.0_f32, f32::max);
        let largest = left.iter().map(|value| value.abs()).fold(0.0_f32, f32::max);
        format!(
            "elements={} differing={differing} max_ulp={max_ulp} max_abs={max_abs:e} max_abs_over_largest={:e}",
            left.len(),
            max_abs / largest.max(f32::MIN_POSITIVE)
        )
    }

    fn evict_system_cache(scratch: &mut [u8], round: usize) {
        scratch.fill(round as u8);
        std::hint::black_box(&scratch);
    }

    fn measure_sequence(
        arms: &[(String, Vec<&CapturedDispatch>)],
        rounds: usize,
        scratch: &mut [u8],
        per_dispatch: bool,
    ) -> Vec<f64> {
        let mut spans: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            for offset in 0..arms.len() {
                let index = (round + offset) % arms.len();
                evict_system_cache(scratch, round + offset);
                let total = CapturedDispatch::time_gpu_sequence_ns(&arms[index].1)
                    .expect("sequence replay");
                let divisor = if per_dispatch { arms[index].1.len() as f64 } else { 1.0 };
                spans[index].push(total / divisor);
            }
        }
        spans.iter_mut().map(|samples| median(samples)).collect()
    }

    fn step_with_group_replaced<'a>(
        dispatches: &'a [CapturedDispatch],
        members: &[usize],
        replacements: &'a [CapturedDispatch],
    ) -> Vec<&'a CapturedDispatch> {
        let replaced: BTreeMap<usize, &CapturedDispatch> =
            members.iter().copied().zip(replacements.iter()).collect();
        dispatches
            .iter()
            .enumerate()
            .map(|(index, dispatch)| replaced.get(&index).copied().unwrap_or(dispatch))
            .collect()
    }

    fn measure(
        arms: &[(String, Vec<&CapturedDispatch>)],
        rounds: usize,
        batch: usize,
    ) -> Vec<(f64, f64)> {
        let mut single: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        let mut batched: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            for offset in 0..arms.len() {
                let index = (round + offset) % arms.len();
                single[index].push(arms[index].1[0].time_gpu_ns(1).expect("replay"));
                batched[index].push(arms[index].1[0].time_gpu_ns(batch).expect("replay"));
            }
        }
        (0..arms.len())
            .map(|index| {
                let single_ns = median(&mut single[index]);
                let batched_ns = median(&mut batched[index]);
                let marginal = (batched_ns - single_ns) / (batch as f64 - 1.0);
                (marginal, single_ns)
            })
            .collect()
    }

    pub fn run() {
        let step = env_usize("AB_STEP", 5);
        let rounds = env_usize("AB_ROUNDS", 60);
        let batch = env_usize("AB_BATCH", 16);
        let passes = env_usize("AB_PASSES", 1);
        let allow_packed = std::env::var_os("AB_PACKED").is_some();
        let describe = std::env::var_os("AB_DESCRIBE").is_some();
        let omit_all = std::env::var_os("AB_OMIT_ALL").is_some();
        let step_sequence = std::env::var_os("AB_STEP_SEQUENCE").is_some();
        let sequence = step_sequence || std::env::var_os("AB_SEQUENCE").is_some();
        let mut scratch = vec![0u8; env_usize("AB_FLUSH_MIB", 384) << 20];
        let variant_dir = PathBuf::from(std::env::var("AB_VARIANT_DIR").expect("AB_VARIANT_DIR"));
        // SAFETY: called from `main` before any thread is spawned.
        unsafe {
            std::env::set_var("PROXIMA_CAPTURE_NODES", "all");
            std::env::set_var("PROXIMA_CAPTURE_STEPS", step.to_string());
            std::env::set_var("PROXIMA_CAPTURE_LIVE", "1");
        }
        decode(step);
        let dispatches: Vec<CapturedDispatch> = omega::take_captured_dispatches()
            .into_iter()
            .filter(|dispatch| dispatch.grid.threads > 0)
            .collect();
        assert!(!dispatches.is_empty(), "N==0: nothing captured");
        let mut groups: BTreeMap<GroupKey, Vec<usize>> = BTreeMap::new();
        for (index, dispatch) in dispatches.iter().enumerate() {
            let key = GroupKey {
                sha: dispatch.msl_sha256.clone(),
                threads: dispatch.grid.threads,
                width: dispatch.grid.threadgroup_width,
                extents: dispatch.extents.clone(),
            };
            groups.entry(key).or_default().push(index);
        }
        println!(
            "ab capture: step={step} dispatches={} groups={}",
            dispatches.len(),
            groups.len()
        );
        let mut timed_groups = 0usize;
        for (
            GroupKey {
                sha,
                threads,
                width,
                extents,
            },
            members,
        ) in &groups
        {
            let base = &dispatches[members[0]];
            if !allow_packed && base.operands.iter().any(|(_, codec)| codec != "unpacked") {
                continue;
            }
            let sha_filter = std::env::var("AB_SHA").ok();
            if sha_filter.as_ref().is_some_and(|prefix| !sha.starts_with(prefix.as_str())) {
                continue;
            }
            if describe {
                println!(
                    "ab describe sha={} extents={extents:?} count={} entry={}",
                    &sha[..SHA_PREFIX_CHARS],
                    members.len(),
                    base.entry
                );
                for line in base.describe_buffers() {
                    println!("ab describe   {line}");
                }
                timed_groups += 1;
                continue;
            }
            if omit_all {
                let kept: Vec<&CapturedDispatch> = dispatches
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !members.contains(index))
                    .map(|(_, dispatch)| dispatch)
                    .collect();
                let arms = vec![
                    ("base".to_string(), dispatches.iter().collect::<Vec<_>>()),
                    ("omit".to_string(), kept),
                ];
                let totals = measure_sequence(&arms, rounds, &mut scratch, false);
                let cost_ms = (totals[0] - totals[1]) / 1e6;
                println!(
                    "ab omit entry={} sha={} threads={threads} tg={width:?} extents={extents:?} count={} base_ms={:.4} cost_ms={cost_ms:.4} per_dispatch_us={:.2}",
                    base.entry,
                    &sha[..SHA_PREFIX_CHARS],
                    members.len(),
                    totals[0] / 1e6,
                    cost_ms * 1e3 / members.len() as f64
                );
                timed_groups += 1;
                continue;
            }
            let base_width = width.unwrap_or(*threads).max(1);
            let member_refs: Vec<&CapturedDispatch> = if sequence {
                members.iter().map(|index| &dispatches[*index]).collect()
            } else {
                vec![base]
            };
            let variants = load_variants(&variant_dir, &member_refs, base_width);
            if variants.is_empty() {
                continue;
            }
            let mut arms: Vec<(String, Vec<&CapturedDispatch>)> =
                vec![("base".to_string(), member_refs.clone())];
            arms.extend(
                variants
                    .iter()
                    .map(|arm| (arm.label.clone(), arm.dispatches.iter().collect())),
            );
            timed_groups += 1;
            let reference_label =
                std::env::var("AB_REFERENCE").unwrap_or_else(|_| "base".to_string());
            let reference = arms
                .iter()
                .find(|arm| arm.0 == reference_label)
                .map_or(base, |arm| arm.1[0]);
            let base_output = reference.replay_output().expect("reference output");
            for arm in &arms {
                let output = arm.1[0].replay_output().expect("arm output");
                println!(
                    "ab bits sha={} threads={threads} extents={extents:?} arm={} {}",
                    &sha[..SHA_PREFIX_CHARS],
                    arm.0,
                    compare_outputs(&base_output, &output)
                );
            }
            for pass in 0..passes {
                if step_sequence {
                    let omitted: Vec<&CapturedDispatch> = dispatches
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| !members.contains(index))
                        .map(|(_, dispatch)| dispatch)
                        .collect();
                    let step_arms: Vec<(String, Vec<&CapturedDispatch>)> =
                        std::iter::once(("base".to_string(), dispatches.iter().collect()))
                            .chain(std::iter::once(("omit_group".to_string(), omitted)))
                            .chain(variants.iter().map(|arm| {
                                (
                                    arm.label.clone(),
                                    step_with_group_replaced(&dispatches, members, &arm.dispatches),
                                )
                            }))
                            .collect();
                    let results = measure_sequence(&step_arms, rounds, &mut scratch, false);
                    for (arm, total) in step_arms.iter().zip(results) {
                        println!(
                            "ab step sha={} threads={threads} extents={extents:?} count={} pass={pass} arm={} step_ms={:.4}",
                            &sha[..SHA_PREFIX_CHARS],
                            members.len(),
                            arm.0,
                            total / 1e6
                        );
                    }
                    continue;
                }
                if sequence {
                    let results = measure_sequence(&arms, rounds, &mut scratch, true);
                    for (arm, per_dispatch) in arms.iter().zip(results) {
                        println!(
                            "ab sequence sha={} threads={threads} extents={extents:?} tg={width:?} count={} pass={pass} arm={} per_dispatch_us={:.3}",
                            &sha[..SHA_PREFIX_CHARS],
                            members.len(),
                            arm.0,
                            per_dispatch / 1e3
                        );
                    }
                    continue;
                }
                let results = measure(&arms, rounds, batch);
                for (arm, (marginal, single)) in arms.iter().zip(results) {
                    println!(
                        "ab group sha={} threads={threads} extents={extents:?} tg={width:?} count={} pass={pass} arm={} marginal_us={:.3} single_us={:.3}",
                        &sha[..SHA_PREFIX_CHARS],
                        members.len(),
                        arm.0,
                        marginal / 1e3,
                        single / 1e3
                    );
                }
            }
        }
        assert!(
            timed_groups > 0,
            "N==0: no captured kernel matched a variant file"
        );
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
