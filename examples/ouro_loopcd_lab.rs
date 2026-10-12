#[cfg(feature = "proxima-tensor")]
use std::fs::{self, File};
use std::io;

#[cfg(feature = "proxima-tensor")]
#[path = "ouro_loopcd_lab/checkpoint.rs"]
mod checkpoint;
#[cfg(feature = "proxima-tensor")]
#[path = "ouro_loopcd_lab/controls.rs"]
mod controls;
#[cfg(feature = "proxima-tensor")]
#[path = "ouro_loopcd_lab/engine.rs"]
mod engine;
#[cfg(feature = "proxima-tensor")]
#[path = "ouro_loopcd_lab/graph.rs"]
mod graph;
#[cfg(feature = "proxima-tensor")]
#[path = "ouro_loopcd_lab/guidance.rs"]
mod guidance;
#[cfg(feature = "proxima-tensor")]
#[path = "ouro_loopcd_lab/model.rs"]
mod model;
#[cfg(feature = "proxima-tensor")]
#[path = "ouro_loopcd_lab/panel.rs"]
mod panel;
#[cfg(feature = "proxima-tensor")]
#[path = "ouro_loopcd_lab/paper.rs"]
mod paper;
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
#[path = "ouro_loopcd_lab/performance.rs"]
mod performance;
#[cfg(all(feature = "proxima-tensor", not(target_os = "macos")))]
mod performance {
    use std::io;

    use super::runner::Arguments;

    pub(super) fn run(_: &Arguments) -> io::Result<()> {
        Err(io::Error::other(
            "performance mode requires native macOS Metal",
        ))
    }

    pub(super) fn microbench(arguments: &Arguments) -> io::Result<()> {
        run(arguments)
    }

    pub(super) fn cached(arguments: &Arguments) -> io::Result<()> {
        run(arguments)
    }
}

#[cfg(feature = "proxima-tensor")]
mod runner {
    use std::env;
    use std::fs::File;
    use std::io::{self, BufRead, BufReader, BufWriter, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    use omega::config::{RuntimeConfig, SelectionMode};
    use serde_json::{Value, json};

    use super::checkpoint::{Checkpoint, WeightMappingMode};
    use super::controls;
    use super::guidance::Guidance;
    use super::model::{ARMS, Model, pick};

    pub struct Arguments {
        pub output: PathBuf,
        pub(super) directory: PathBuf,
        pub(super) backend: String,
        pub(super) prefix: Vec<u32>,
        pub(super) samples: usize,
        pub(super) metal_copy_weights: bool,
        pub(super) panel: Option<PathBuf>,
        pub(super) panel_cases: Option<Vec<String>>,
        inventory_only: bool,
        controls_only: bool,
        compare_cpu: Option<PathBuf>,
        pub(super) guidance: Option<Guidance>,
        pub(super) depth: usize,
        pub(super) max_new_tokens: usize,
        paper_controls: bool,
        performance: bool,
        cached_performance: bool,
        pub(super) gpu_profile: bool,
        pub(super) omega_config: Option<RuntimeConfig>,
        pub(super) microbench: bool,
    }

    pub fn arguments() -> io::Result<Arguments> {
        let mut output = None;
        let mut directory = PathBuf::from("/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/models/ouro-1.4b");
        let mut backend = String::from("cpu");
        let mut prefix = vec![19556, 905, 30];
        let mut samples = 20;
        let mut metal_copy_weights = false;
        let mut inventory_only = false;
        let mut controls_only = false;
        let mut compare_cpu = None;
        let mut panel = None;
        let mut panel_cases = None;
        let mut explicit_prefix = false;
        let mut method = None;
        let mut depth = 4;
        let mut strength = 0.5f32;
        let mut max_new_tokens = 1;
        let mut paper_controls = false;
        let mut performance = false;
        let mut cached_performance = false;
        let mut gpu_profile = false;
        let mut omega_config_path = None;
        let mut microbench = false;
        let mut explicit_method_options = false;
        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            if argument == "--inventory-only" {
                inventory_only = true;
                continue;
            }
            if argument == "--controls-only" {
                controls_only = true;
                continue;
            }
            if argument == "--metal-copy-weights" {
                metal_copy_weights = true;
                continue;
            }
            if argument == "--paper-controls" {
                paper_controls = true;
                continue;
            }
            if argument == "--performance" {
                performance = true;
                continue;
            }
            if argument == "--cached-performance" {
                cached_performance = true;
                continue;
            }
            if argument == "--microbench" {
                microbench = true;
                continue;
            }
            if argument == "--gpu-profile" {
                gpu_profile = true;
                continue;
            }
            let value = arguments
                .next()
                .ok_or_else(|| io::Error::other(format!("missing value for {argument}")))?;
            match argument.as_str() {
                "--model-dir" => directory = PathBuf::from(value),
                "--backend" => backend = value,
                "--output" => output = Some(PathBuf::from(value)),
                "--omega-config" => omega_config_path = Some(PathBuf::from(value)),
                "--compare-cpu" => compare_cpu = Some(PathBuf::from(value)),
                "--panel" => panel = Some(PathBuf::from(value)),
                "--panel-cases" => panel_cases = Some(value.split(',').map(String::from).collect()),
                "--samples" => samples = value.parse().map_err(io::Error::other)?,
                "--method" => method = Some(value),
                "--depth" => {
                    explicit_method_options = true;
                    depth = value.parse().map_err(io::Error::other)?;
                }
                "--strength" => {
                    explicit_method_options = true;
                    strength = value.parse().map_err(io::Error::other)?;
                }
                "--max-new-tokens" => {
                    explicit_method_options = true;
                    max_new_tokens = value.parse().map_err(io::Error::other)?;
                }
                "--prefix-ids" => {
                    explicit_prefix = true;
                    prefix = value
                        .split(',')
                        .map(|piece| piece.parse().map_err(io::Error::other))
                        .collect::<io::Result<_>>()?
                }
                _ => return Err(io::Error::other(format!("unknown argument {argument}"))),
            }
        }
        let guidance = method
            .as_deref()
            .map(|method| Guidance::parse(method, strength))
            .transpose()?;
        if cached_performance
            && (performance
                || backend != "metal"
                || samples != 5
                || max_new_tokens != 64
                || method.is_some()
                || panel.is_some()
                || explicit_prefix
                || controls_only
                || inventory_only
                || paper_controls
                || compare_cpu.is_some()
                || microbench
                || metal_copy_weights
                || panel_cases.is_some())
        {
            return Err(io::Error::other(
                "cached performance requires Metal, five samples, 64-token budget and exclusive mode",
            ));
        }
        if performance {
            if backend != "metal"
                || samples != 5
                || method.is_some()
                || panel.is_some()
                || explicit_prefix
                || explicit_method_options
                || paper_controls
                || inventory_only
                || controls_only
                || compare_cpu.is_some()
                || metal_copy_weights
                || panel_cases.is_some()
            {
                return Err(io::Error::other(
                    "performance mode requires Metal, exactly five samples, and no other mode or per-run overrides",
                ));
            }
        } else if gpu_profile && !cached_performance {
            return Err(io::Error::other(
                "--gpu-profile requires a performance mode",
            ));
        }
        if depth == 0 || !strength.is_finite() || strength < 0.0 {
            return Err(io::Error::other(
                "require positive depth and finite nonnegative strength",
            ));
        }
        if omega_config_path.is_some() && !performance && !cached_performance {
            return Err(io::Error::other(
                "--omega-config requires --performance in this experiment runner",
            ));
        }
        if microbench && (!performance || omega_config_path.is_none() || gpu_profile) {
            return Err(io::Error::other(
                "--microbench requires --performance and --omega-config without --gpu-profile",
            ));
        }
        let omega_config = omega_config_path
            .as_deref()
            .map(RuntimeConfig::from_file)
            .transpose()
            .map_err(io::Error::other)?;
        if let Some(config) = &omega_config
            && (config.engine != "gpu"
                || config.gpu_driver != "metal"
                || config.selection.mode != SelectionMode::Configured)
        {
            return Err(io::Error::other(
                "this frozen performance cohort requires configured selection with explicit gpu/metal",
            ));
        }
        if guidance.is_some()
            && (panel.is_some()
                || inventory_only
                || controls_only
                || paper_controls
                || compare_cpu.is_some())
            || paper_controls
                && (panel.is_some() || inventory_only || controls_only || compare_cpu.is_some())
            || explicit_method_options && guidance.is_none() && !cached_performance
        {
            return Err(io::Error::other(
                "selected methods and paper controls require an exclusive mode; depth, strength and token budget require --method",
            ));
        }
        if panel.is_some()
            && (inventory_only || controls_only || compare_cpu.is_some() || explicit_prefix)
            || panel.is_none() && panel_cases.is_some()
        {
            return Err(io::Error::other(
                "panel mode cannot combine with original prefix, controls, inventory or backend comparison modes; --panel-cases requires --panel",
            ));
        }
        if panel.is_some() && samples < 2 {
            return Err(io::Error::other("panel mode requires at least two samples"));
        }
        if !matches!(backend.as_str(), "cpu" | "metal")
            || prefix.is_empty()
            || prefix.len() > 65536
            || panel.is_none()
                && guidance.is_none()
                && !paper_controls
                && !performance
                && !cached_performance
                && samples != 0
                && samples < 20
        {
            return Err(io::Error::other(
                "require cpu/metal backend, nonempty bounded prefix and zero or at least twenty samples",
            ));
        }
        Ok(Arguments {
            output: output.ok_or_else(|| io::Error::other("--output required"))?,
            directory,
            backend,
            prefix,
            samples,
            metal_copy_weights,
            inventory_only,
            controls_only,
            compare_cpu,
            panel,
            panel_cases,
            guidance,
            depth,
            max_new_tokens,
            paper_controls,
            performance,
            cached_performance,
            gpu_profile,
            omega_config,
            microbench,
        })
    }

    fn compare_cpu(
        path: &PathBuf,
        metal_records: &[Value],
        writer: &mut impl Write,
    ) -> io::Result<()> {
        let reader = BufReader::new(File::open(path)?);
        let mut cpu_records = Vec::new();
        for (line_index, line) in reader.lines().enumerate() {
            let value: Value = serde_json::from_str(&line?).map_err(io::Error::other)?;
            if value["kind"] == "diagnostic_readouts" {
                cpu_records.push((line_index + 1, value));
            }
        }
        if cpu_records.len() != ARMS.len() || metal_records.len() != ARMS.len() {
            return Err(io::Error::other(
                "diagnostic backend comparison requires all four arms",
            ));
        }
        for metal in metal_records {
            let (line, cpu) = cpu_records
                .iter()
                .find(|(_, cpu)| cpu["context"]["arm"] == metal["context"]["arm"])
                .ok_or_else(|| io::Error::other("CPU comparison arm missing"))?;
            if cpu["context"]["backend"] != "cpu"
                || metal["context"]["backend"] != "metal"
                || cpu["context"]["prefix_ids"] != metal["context"]["prefix_ids"]
                || cpu["context"]["checkpoint_observed_sha256"]
                    != metal["context"]["checkpoint_observed_sha256"]
            {
                return Err(io::Error::other(
                    "backend comparison prefix, checkpoint or backend differs",
                ));
            }
            for field in ["reference_logits", "terminal_logits", "guided_logits"] {
                let cpu_values = cpu[field]
                    .as_array()
                    .ok_or_else(|| io::Error::other("CPU diagnostic vector absent"))?;
                let metal_values = metal[field]
                    .as_array()
                    .ok_or_else(|| io::Error::other("Metal diagnostic vector absent"))?;
                if cpu_values.len() != metal_values.len() {
                    return Err(io::Error::other("backend comparison vector lengths differ"));
                }
                let mut differences = Vec::new();
                for (coordinate, (cpu_value, metal_value)) in
                    cpu_values.iter().zip(metal_values).enumerate()
                {
                    let cpu_number = cpu_value
                        .as_f64()
                        .ok_or_else(|| io::Error::other("CPU logit not numeric"))?;
                    let metal_number = metal_value
                        .as_f64()
                        .ok_or_else(|| io::Error::other("Metal logit not numeric"))?;
                    if cpu_number != metal_number {
                        differences.push(json!({"coordinate":coordinate,"cpu":cpu_number,"metal":metal_number,"difference":metal_number-cpu_number}));
                    }
                }
                record(
                    writer,
                    &json!({"kind":"cpu_metal_diagnostic_comparison","arm":metal["context"]["arm"],"boundary":field,"cpu_source":format!("{}:{line}",path.display()),"cpu_numeric_policy":cpu["context"]["numeric_policy"],"metal_numeric_policy":metal["context"]["numeric_policy"],"prefix_ids":metal["context"]["prefix_ids"],"compared_elements":cpu_values.len(),"exact_value_difference_count":differences.len(),"complete_coordinate_differences":differences,"cpu_selected_id":cpu["selected_id"],"metal_selected_id":metal["selected_id"],"selected_id_changed":cpu["selected_id"] != metal["selected_id"]}),
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn record(writer: &mut impl Write, value: &Value) -> io::Result<()> {
        serde_json::to_writer(&mut *writer, value).map_err(io::Error::other)?;
        writer.write_all(b"\n")
    }

    pub(super) fn stats(values: &[f64], positions: usize) -> Value {
        if values.is_empty() {
            return json!({"sample_count":0,"mean_ms":null,"sample_standard_deviation_ms":null,"cov_percent":null,"p50_ms":null,"p90_ms":null,"p95_ms":null,"p99_ms":null,"p999_ms":null,"input_tokens_per_second":null});
        }
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let deviation = if count >= 2 {
            Some(
                (values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / (count - 1) as f64)
                    .sqrt(),
            )
        } else {
            None
        };
        let mut ordered = values.to_vec();
        ordered.sort_by(f64::total_cmp);
        let percentile = |fraction: f64| {
            ordered[((fraction * count as f64).ceil() as usize)
                .saturating_sub(1)
                .min(count - 1)]
        };
        json!({"sample_count":count,"mean_ms":mean,"sample_standard_deviation_ms":deviation,"cov_percent":deviation.filter(|_| mean > 0.0).map(|deviation| 100.0*deviation/mean),"p50_ms":percentile(0.5),"p90_ms":percentile(0.9),"p95_ms":percentile(0.95),"p99_ms":percentile(0.99),"p999_ms":percentile(0.999),"input_tokens_per_second":(positions > 0 && mean > 0.0).then_some(positions as f64*1000.0/mean),"raw_durations_ms":values})
    }

    pub fn run(arguments: &Arguments) -> io::Result<()> {
        if arguments.cached_performance {
            return super::performance::cached(arguments);
        }
        if arguments.performance {
            if arguments.microbench {
                return super::performance::microbench(arguments);
            }
            return super::performance::run(arguments);
        }
        if let Some(path) = &arguments.panel {
            return super::panel::run(arguments, path);
        }
        let mut writer = BufWriter::new(File::create(arguments.output.join("records.jsonl"))?);
        if arguments.paper_controls {
            record(&mut writer, &controls::paper(&arguments.backend)?)?;
            return writer.flush();
        }
        if arguments.guidance.is_some() {
            return super::paper::run(arguments, &mut writer);
        }
        if arguments.controls_only {
            record(&mut writer, &controls::run(&arguments.backend)?)?;
            return writer.flush();
        }
        let setup_started = Instant::now();
        let mapping_mode =
            WeightMappingMode::for_run(&arguments.backend, arguments.metal_copy_weights);
        let checkpoint = Checkpoint::open(&arguments.directory, mapping_mode)?;
        for value in &checkpoint.inventory {
            record(&mut writer, value)?;
        }
        if arguments.inventory_only {
            return writer.flush();
        }
        record(&mut writer, &controls::run(&arguments.backend)?)?;
        let prefix_tokens: Vec<Value> = arguments.prefix.iter().map(|identifier| json!({"id":identifier,"spelling":checkpoint.vocab.token_str(*identifier),"bytes":checkpoint.vocab.token_bytes(*identifier)})).collect();
        record(
            &mut writer,
            &json!({"kind":"prefix","ids":arguments.prefix,"tokens":prefix_tokens,"no_special_tokens_inserted":true,"text_encoding_executed":false}),
        )?;
        let mut model = Model::prepare(&checkpoint, arguments.prefix.clone(), &arguments.backend)?;
        record(
            &mut writer,
            &json!({"kind":"setup","backend":arguments.backend,"setup_wall_ms":setup_started.elapsed().as_secs_f64()*1000.0,"timing_excludes":"checkpoint parsing, embedding staging, plan preparation, controls, diagnostics and JSON serialization","timing_includes":"full recurrent layers, loop norms, required head readouts, contrast, native submission/readback and output copies"}),
        )?;
        let mut diagnostic_records = Vec::new();
        for arm in ARMS {
            let context = model.context(arm);
            let diagnostic = model.forward(arm, true)?;
            for mut value in diagnostic.trace {
                value["context"] = context.clone();
                record(&mut writer, &value)?;
            }
            let readout_record = json!({"kind":"diagnostic_readouts","context":context,"reference_logits":diagnostic.reference,"terminal_logits":diagnostic.terminal,"guided_logits":diagnostic.logits,"selected_id":diagnostic.selected,"selected_spelling":checkpoint.vocab.token_str(diagnostic.selected),"selected_bytes":checkpoint.vocab.token_bytes(diagnostic.selected),"diagnostic_head_executions":diagnostic.head_executions,"diagnostic_only":true});
            record(&mut writer, &readout_record)?;
            diagnostic_records.push(readout_record);
        }
        if let Some(path) = &arguments.compare_cpu {
            let mut comparison_writer = BufWriter::new(File::create(
                arguments.output.join("cpu-metal-comparison.jsonl"),
            )?);
            compare_cpu(path, &diagnostic_records, &mut comparison_writer)?;
            comparison_writer.flush()?;
        }
        writer.flush()?;
        let mut sample_writer =
            BufWriter::new(File::create(arguments.output.join("samples.jsonl"))?);
        let mut durations: Vec<Vec<f64>> = (0..ARMS.len())
            .map(|_| Vec::with_capacity(arguments.samples))
            .collect();
        for sample in 0..arguments.samples {
            let mut selections = Vec::new();
            for (index, arm) in ARMS.iter().copied().enumerate() {
                let started = Instant::now();
                let observed = model.forward(arm, false)?;
                let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
                durations[index].push(wall_ms);
                let terminal_id = pick(&observed.terminal)?;
                selections.push((arm.name, observed.selected));
                record(
                    &mut sample_writer,
                    &json!({"kind":"sample","context":model.context(arm),"sample":sample,"wall_ms":wall_ms,"selected_id":observed.selected,"selected_spelling":checkpoint.vocab.token_str(observed.selected),"selected_bytes":checkpoint.vocab.token_bytes(observed.selected),"terminal_selected_id":terminal_id,"changed_from_terminal":observed.selected!=terminal_id,"head_executions":observed.head_executions,"layer_executions":observed.layer_executions,"branch_norm_sites":observed.branch_norm_sites,"loop_norm_sites":observed.loop_norm_sites,"terminal_logits":observed.terminal,"reference_logits":observed.reference,"output_logits":observed.logits}),
                )?;
            }
            for (name, identifier) in &selections {
                record(
                    &mut sample_writer,
                    &json!({"kind":"selected_id_comparison","sample":sample,"arm":name,"selected_id":identifier,"baseline4_id":selections[0].1,"baseline2_id":selections[1].1,"changed_from_baseline4":*identifier!=selections[0].1,"changed_from_baseline2":*identifier!=selections[1].1,"backend":arguments.backend,"prefix_ids":arguments.prefix}),
                )?;
            }
            sample_writer.flush()?;
            eprintln!(
                "completed native model sample {} of {} on {}",
                sample + 1,
                arguments.samples,
                arguments.backend
            );
        }
        let mut summaries = Vec::new();
        for (index, arm) in ARMS.iter().copied().enumerate() {
            summaries.push(json!({"kind":"summary","context":model.context(arm),"cost":stats(&durations[index],arguments.prefix.len()),"head_executions_per_sample":if arm.strength().is_some() {2} else {1},"dispatch_count":null,"readback_count":null,"device_allocation_bytes":null,"unmeasured_counter_reason":"this example uses existing plan/execute surface without attributing global concurrent counters","quality_status":"unmeasured"}));
        }
        serde_json::to_writer_pretty(
            File::create(arguments.output.join("summary.json"))?,
            &summaries,
        )
        .map_err(io::Error::other)?;
        writer.flush()?;
        sample_writer.flush()
    }
}

#[cfg(feature = "proxima-tensor")]
fn main() -> io::Result<()> {
    let arguments = runner::arguments()?;
    if let Some(parent) = arguments.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&arguments.output)?;
    let result = runner::run(&arguments);
    if let Err(error) = &result {
        serde_json::to_writer_pretty(File::create(arguments.output.join("failure.json"))?,&serde_json::json!({"kind":"native_ouro_failure","error":error.to_string(),"cpu_fallback_executed":false,"model_completion_not_claimed":true})).map_err(io::Error::other)?;
    }
    result
}

#[cfg(not(feature = "proxima-tensor"))]
fn main() -> io::Result<()> {
    Err(io::Error::other("build with --features proxima-tensor"))
}
