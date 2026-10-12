use std::io;

#[cfg(feature = "proxima-tensor")]
mod lab {
    use std::env;
    use std::fs::{self, File};
    use std::io::{self, BufWriter, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    use omega::backend::{self, Engine, GpuDriver, Plan};
    use proxima_tensor::{
        DType, Extent, IndexMap, NodeId, NumericPolicy, Op, QuantizedBlock, ScalarOp, append,
        projection,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct StatePair {
        origin: String,
        model_id: String,
        checkpoint_id: String,
        representation: String,
        boundary: String,
        reference_prefix_token_ids: Vec<u32>,
        final_prefix_token_ids: Vec<u32>,
        reference_iteration: u32,
        final_iteration: u32,
        shape: Vec<u32>,
        reference: Vec<f32>,
        #[serde(rename = "final", alias = "final_state")]
        final_state: Vec<f32>,
        strength: f32,
    }

    #[derive(Debug)]
    struct Arguments {
        backend: String,
        input: Option<PathBuf>,
        output: PathBuf,
        samples: usize,
    }

    fn arguments() -> io::Result<Arguments> {
        let mut backend = String::from("cpu");
        let mut input = None;
        let mut output = None;
        let mut samples = 30;
        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| io::Error::other(format!("missing value for {argument}")))?;
            match argument.as_str() {
                "--backend" => backend = value,
                "--input" => input = Some(PathBuf::from(value)),
                "--output" => output = Some(PathBuf::from(value)),
                "--samples" => {
                    samples = value.parse().map_err(|error| {
                        io::Error::other(format!("invalid sample count: {error}"))
                    })?;
                }
                _ => return Err(io::Error::other(format!("unknown argument {argument}"))),
            }
        }
        if !matches!(backend.as_str(), "cpu" | "metal") {
            return Err(io::Error::other("--backend must be cpu or metal"));
        }
        if samples < 20 {
            return Err(io::Error::other("--samples must be at least 20"));
        }
        Ok(Arguments {
            backend,
            input,
            output: output.ok_or_else(|| io::Error::other("--output is required"))?,
            samples,
        })
    }

    fn fixture() -> StatePair {
        StatePair {
            origin: String::from("synthetic_paper_algebra_fixture"),
            model_id: String::from("none:literal-fixture"),
            checkpoint_id: String::from("none:literal-fixture"),
            representation: String::from("hidden"),
            boundary: String::from("pre-coda-pre-output-normalization"),
            reference_prefix_token_ids: vec![101, 202],
            final_prefix_token_ids: vec![101, 202],
            reference_iteration: 1,
            final_iteration: 2,
            shape: vec![4],
            reference: vec![1.0, 2.0, -1.0, 4.0],
            final_state: vec![3.0, 1.0, 2.0, 0.0],
            strength: 0.5,
        }
    }

    fn validate(pair: &StatePair) -> io::Result<()> {
        if [&pair.origin, &pair.model_id, &pair.checkpoint_id]
            .iter()
            .any(|identifier| identifier.trim().is_empty())
        {
            return Err(io::Error::other(
                "origin, model and checkpoint identifiers are required",
            ));
        }
        if pair.reference_prefix_token_ids.is_empty()
            || pair.reference_prefix_token_ids != pair.final_prefix_token_ids
        {
            return Err(io::Error::other(
                "states must share exact nonempty prefix token ids",
            ));
        }
        if pair.reference_iteration == 0 || pair.reference_iteration >= pair.final_iteration {
            return Err(io::Error::other(
                "completed reference iteration must precede final iteration",
            ));
        }
        if !matches!(
            (pair.representation.as_str(), pair.boundary.as_str()),
            ("hidden", "pre-coda-pre-output-normalization") | ("logits", "post-readout-logits")
        ) {
            return Err(io::Error::other(
                "representation does not identify the required contrast boundary",
            ));
        }
        if pair.shape.is_empty() || pair.shape.contains(&0) {
            return Err(io::Error::other("state shape must have nonzero extents"));
        }
        let elements = pair.shape.iter().try_fold(1usize, |product, extent| {
            let extent = usize::try_from(*extent).map_err(io::Error::other)?;
            product
                .checked_mul(extent)
                .ok_or_else(|| io::Error::other("shape product overflow"))
        })?;
        if pair.reference.len() != elements || pair.final_state.len() != elements {
            return Err(io::Error::other(
                "both state vectors must match the exact shape",
            ));
        }
        if !pair.strength.is_finite() || pair.strength < 0.0 {
            return Err(io::Error::other("strength must be finite and nonnegative"));
        }
        if pair
            .reference
            .iter()
            .chain(&pair.final_state)
            .any(|value| !value.is_finite())
        {
            return Err(io::Error::other("state elements must be finite"));
        }
        Ok(())
    }

    fn program(pair: &StatePair, reversed: bool) -> io::Result<(Vec<Op>, NodeId)> {
        let rank = u16::try_from(pair.shape.len()).map_err(io::Error::other)?;
        let axes: Vec<u16> = (0..rank).collect();
        let identity = IndexMap::Affine(projection(rank, &axes));
        let mut program = Vec::with_capacity(6);
        let mut inputs = Vec::with_capacity(3);
        for name in ["reference", "final", "strength"] {
            inputs.push(append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: pair
                        .shape
                        .iter()
                        .map(|extent| Extent::Static(*extent))
                        .collect(),
                    name: Some(String::from(name)),
                },
            ));
        }
        let (first, second) = if reversed {
            (inputs[0], inputs[1])
        } else {
            (inputs[1], inputs[0])
        };
        let difference = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Subtract,
                operands: vec![(first, identity.clone()), (second, identity.clone())],
                name: Some(String::from("contrast_difference")),
            },
        );
        let scaled = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (difference, identity.clone()),
                    (inputs[2], identity.clone()),
                ],
                name: Some(String::from("scaled_contrast")),
            },
        );
        let guided = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![(inputs[1], identity.clone()), (scaled, identity)],
                name: Some(String::from("guided_state")),
            },
        );
        Ok((program, guided))
    }

    fn named_blocks<'pair>(
        pair: &'pair StatePair,
        strength: &'pair [f32],
    ) -> [(&'static str, QuantizedBlock<'pair>); 3] {
        [
            ("reference", QuantizedBlock::Float32(&pair.reference)),
            ("final", QuantizedBlock::Float32(&pair.final_state)),
            ("strength", QuantizedBlock::Float32(strength)),
        ]
    }

    fn prepare(
        pair: &StatePair,
        backend_name: &str,
        reversed: bool,
        strength: &[f32],
    ) -> io::Result<(Plan, NodeId, Value)> {
        validate(pair)?;
        let (program, output) = program(pair, reversed)?;
        let graph = json!({
            "inputs":program.iter().filter(|operation| matches!(operation, Op::Input { .. })).count(),
            "elementwise_nodes":program.iter().filter(|operation| matches!(operation, Op::Elementwise { .. })).count(),
            "subtract_nodes":program.iter().filter(|operation| matches!(operation,
                Op::Elementwise { body: ScalarOp::Subtract, .. })).count(),
            "multiply_nodes":program.iter().filter(|operation| matches!(operation,
                Op::Elementwise { body: ScalarOp::Multiply, .. })).count(),
            "add_nodes":program.iter().filter(|operation| matches!(operation,
                Op::Elementwise { body: ScalarOp::Add, .. })).count(),
            "program_nodes":program.len(),
        });
        let engine = if backend_name == "cpu" {
            Engine::Cpu
        } else {
            Engine::Gpu
        };
        let plan = backend::plan_named(
            engine,
            Some(GpuDriver::Metal),
            &program,
            &[],
            &named_blocks(pair, strength),
            &[output],
            NumericPolicy::default(),
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        Ok((plan, output, graph))
    }

    fn execute(pair: &StatePair, backend_name: &str, reversed: bool) -> io::Result<Vec<f32>> {
        let strength = vec![pair.strength; pair.reference.len()];
        let (mut plan, output, _graph) = prepare(pair, backend_name, reversed, &strength)?;
        let evaluated = backend::execute_plan_named(&mut plan, &named_blocks(pair, &strength))
            .map_err(|error| io::Error::other(error.to_string()))?;
        let (values, shape) = evaluated
            .get(output)
            .ok_or_else(|| io::Error::other("backend returned no requested contrast output"))?;
        validate_output(pair, values, shape)?;
        Ok(values.to_vec())
    }

    fn validate_output(pair: &StatePair, values: &[f32], shape: &[u64]) -> io::Result<()> {
        if shape
            .iter()
            .copied()
            .ne(pair.shape.iter().map(|extent| u64::from(*extent)))
            || values.len() != pair.final_state.len()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(io::Error::other(
                "backend output violated exact shape or finite-element contract",
            ));
        }
        Ok(())
    }

    fn mismatches(observed: &[f32], expected: &[f32]) -> usize {
        observed
            .iter()
            .zip(expected)
            .filter(|(observed, expected)| observed != expected)
            .count()
            + observed.len().abs_diff(expected.len())
    }

    fn record(writer: &mut impl Write, value: &Value) -> io::Result<()> {
        serde_json::to_writer(&mut *writer, value).map_err(io::Error::other)?;
        writer.write_all(b"\n")
    }

    fn controls(backend_name: &str, writer: &mut impl Write) -> io::Result<()> {
        let pair = fixture();
        let expected = [4.0, 0.5, 3.5, -2.0];
        let guided = execute(&pair, backend_name, false)?;
        let mut identity_pair = pair.clone();
        identity_pair.strength = 0.0;
        let identity = execute(&identity_pair, backend_name, false)?;
        let reversed = execute(&pair, backend_name, true)?;
        let reversed_expected = [2.0, 1.5, 0.5, 2.0];
        let guided_mismatches = mismatches(&guided, &expected);
        let identity_mismatches = mismatches(&identity, &pair.final_state);
        let reversed_direction_mismatches = mismatches(&reversed, &expected);
        let reversed_literal_mismatches = mismatches(&reversed, &reversed_expected);
        record(
            writer,
            &json!({
                "event":"literal_controls", "backend":backend_name, "pair":pair,
                "expected_guided":expected, "guided_output":guided,
                "identity_output":identity, "reversed_output":reversed,
                "reversed_expected":reversed_expected,
                "guided_elements":guided.len(), "guided_mismatches":guided_mismatches,
                "identity_elements":identity.len(), "identity_mismatches":identity_mismatches,
                "reversed_direction_mismatches":reversed_direction_mismatches,
                "reversed_literal_mismatches":reversed_literal_mismatches,
                "plan_build_count":3, "model_inference_executed":false,
            }),
        )?;
        if guided_mismatches != 0
            || identity_mismatches != 0
            || reversed_direction_mismatches != 4
            || reversed_literal_mismatches != 0
        {
            return Err(io::Error::other(
                "literal payload control disagreed with expected vector",
            ));
        }
        if backend_name == "metal" {
            let cpu = execute(&pair, "cpu", false)?;
            let differences: Vec<Value> = cpu
                .iter()
                .zip(&guided)
                .enumerate()
                .filter(|(_, (cpu, metal))| cpu != metal)
                .map(|(index, (cpu, metal))| json!({"index":index,"cpu":cpu,"metal":metal}))
                .collect();
            record(
                writer,
                &json!({
                    "event":"cpu_metal_comparison", "compared_elements":cpu.len(),
                    "mismatches":mismatches(&cpu, &guided), "cpu_output":cpu,
                    "metal_output":guided, "differing_elements":differences,
                }),
            )?;
            if mismatches(&cpu, &guided) != 0 {
                return Err(io::Error::other("CPU and Metal literal payloads differ"));
            }
        }
        Ok(())
    }

    fn invalid_controls(writer: &mut impl Write) -> io::Result<()> {
        let mut rejected_cases = 0;
        for case in [
            "unequal_prefixes",
            "unequal_vector_lengths",
            "iteration_order",
            "invalid_boundary",
            "invalid_strength",
        ] {
            let mut pair = fixture();
            match case {
                "unequal_prefixes" => pair.final_prefix_token_ids.push(303),
                "unequal_vector_lengths" => {
                    pair.reference.pop();
                }
                "iteration_order" => pair.reference_iteration = pair.final_iteration,
                "invalid_boundary" => pair.boundary = String::from("post-output-normalization"),
                "invalid_strength" => pair.strength = -0.5,
                _ => return Err(io::Error::other("unknown malformed control")),
            }
            let rejection = validate(&pair).err().map(|error| error.to_string());
            record(
                writer,
                &json!({"event":"invalid_input_control", "case":case,
                "pair":pair, "rejection":rejection, "backend_execution_count":0}),
            )?;
            if rejection.is_none() {
                return Err(io::Error::other(format!(
                    "invalid control accepted: {case}"
                )));
            }
            rejected_cases += 1;
        }
        let mut nonfinite_pair = fixture();
        nonfinite_pair.strength = f32::INFINITY;
        let nonfinite_rejection = validate(&nonfinite_pair)
            .err()
            .map(|error| error.to_string());
        record(
            writer,
            &json!({"event":"nonfinite_strength_control",
            "strength_encoding":"positive_infinity_not_representable_in_json",
            "rejection":nonfinite_rejection, "backend_execution_count":0}),
        )?;
        if nonfinite_rejection.is_none() {
            return Err(io::Error::other("nonfinite strength accepted"));
        }
        record(
            writer,
            &json!({"event":"invalid_input_summary", "rejected_cases":rejected_cases,
            "additional_nonfinite_strength_rejections":1}),
        )
    }

    fn statistics(durations: &[f64], elements: usize) -> Value {
        let sample_count = durations.len();
        let mean = durations.iter().sum::<f64>() / sample_count as f64;
        let sample_sd = (sample_count > 1).then(|| {
            (durations
                .iter()
                .map(|duration| (duration - mean).powi(2))
                .sum::<f64>()
                / (sample_count - 1) as f64)
                .sqrt()
        });
        let cov = sample_sd
            .filter(|_| mean > 0.0)
            .map(|deviation| 100.0 * deviation / mean);
        let mut sorted = durations.to_vec();
        sorted.sort_by(f64::total_cmp);
        let percentile = |percent: usize| sorted[(sample_count * percent).div_ceil(100) - 1];
        json!({
            "sample_count":sample_count, "mean_ns":mean, "sample_sd_ns":sample_sd,
            "standard_deviation_ns":sample_sd,
            "cov_percent":cov, "cov_formula":"100 * sample_sd / mean; sample_sd denominator n-1",
            "p50_ns":percentile(50), "p90_ns":percentile(90), "p95_ns":percentile(95),
            "p99_ns":percentile(99), "max_ns":sorted.last(),
            "percentile_method":"nearest_rank", "raw_duration_ns":durations,
            "elements_per_second":(mean > 0.0).then(|| elements as f64 * 1.0e9 / mean),
            "rate_scope":"contrast_output_elements_not_generated_tokens",
        })
    }

    fn measured_arm(
        pair: &StatePair,
        arguments: &Arguments,
        arm: &str,
        writer: &mut impl Write,
    ) -> io::Result<()> {
        let strength = vec![pair.strength; pair.reference.len()];
        let setup_started = Instant::now();
        let (mut plan, output, graph) = prepare(pair, &arguments.backend, false, &strength)?;
        let setup_ns = setup_started.elapsed().as_secs_f64() * 1.0e9;
        record(
            writer,
            &json!({
                "event":"arm_setup", "backend":arguments.backend, "arm":arm, "pair":pair,
                "setup_wall_ns":setup_ns, "graph_census":graph,
                "inputs":graph["inputs"], "input_nodes":graph["inputs"],
                "elementwise_nodes":graph["elementwise_nodes"], "subtract_nodes":graph["subtract_nodes"],
                "multiply_nodes":graph["multiply_nodes"], "add_nodes":graph["add_nodes"],
                "program_nodes":graph["program_nodes"], "plan_build_count":1,
                "equation":"final + strength * (final - reference)",
                "operator_order":["subtract(final,reference)","multiply(difference,strength)","add(final,scaled)"],
                "strength_storage":"setup_allocated_full_shape_vector",
                "allocation_count":null, "allocation_count_status":"unmeasured",
                "measurement_scope":"native_contrast_operator",
                "timing_scope":"execute_plan_named including backend allocation,dispatch,synchronization,host output; excludes JSON formatting",
                "cpu_plan_behavior":"reuses existing scratch; evaluator repeats infer/bind",
                "model_inference_executed":false, "coda_normalization_readout_executed":false,
            }),
        )?;
        let blocks = named_blocks(pair, &strength);
        let mut durations = Vec::with_capacity(arguments.samples);
        let mut previous_output = None;
        for sample in 0..=arguments.samples {
            let started = Instant::now();
            let evaluated = backend::execute_plan_named(&mut plan, &blocks)
                .map_err(|error| io::Error::other(error.to_string()))?;
            let duration = started.elapsed().as_secs_f64() * 1.0e9;
            let (values, shape) = evaluated
                .get(output)
                .ok_or_else(|| io::Error::other("measured execution returned no output"))?;
            record(
                writer,
                &json!({
                    "event":"execution_sample", "backend":arguments.backend, "arm":arm,
                    "sample_index":sample, "warmup":sample == 0, "wall_ns":duration,
                    "output":values, "output_shape":shape, "strength":pair.strength,
                    "output_bits":values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    "peak_live_buffers":evaluated.peak_live_buffers(),
                    "allocation_count":null, "allocation_count_status":"unmeasured",
                }),
            )?;
            validate_output(pair, values, shape)?;
            if arm == "identity" && mismatches(values, &pair.final_state) != 0 {
                return Err(io::Error::other(
                    "zero strength did not retain final state values",
                ));
            }
            if previous_output
                .as_ref()
                .is_some_and(|previous: &Vec<f32>| mismatches(values, previous) != 0)
            {
                return Err(io::Error::other(
                    "repeated plan produced different output elements",
                ));
            }
            if previous_output.is_none() {
                previous_output = Some(values.to_vec());
            }
            if sample > 0 {
                durations.push(duration);
            }
        }
        let mut summary = statistics(&durations, pair.reference.len());
        summary["event"] = json!("arm_summary");
        summary["backend"] = json!(arguments.backend);
        summary["arm"] = json!(arm);
        summary["warmup_count"] = json!(1);
        summary["plan_build_count"] = json!(1);
        summary["plan_execute_count"] = json!(arguments.samples + 1);
        summary["error_count"] = json!(0);
        summary["allocation_count"] = Value::Null;
        summary["allocation_count_status"] = json!("unmeasured");
        #[cfg(target_os = "macos")]
        if let Plan::Metal(metal_plan) = &plan {
            summary["metal_arena_allocated_bytes"] = json!(metal_plan.arena_allocated_bytes());
            summary["metal_arena_peak_bytes"] = json!(metal_plan.arena_peak_bytes());
            summary["metal_arena_slot_count"] = json!(metal_plan.arena_slot_count());
            summary["metal_arena_counter_scope"] =
                json!("optional existing plan arena counters; not total allocation counts");
        }
        record(writer, &summary)
    }

    fn session(arguments: &Arguments, writer: &mut impl Write) -> io::Result<()> {
        let pair = match &arguments.input {
            Some(input) => {
                let raw = fs::read(input)?;
                fs::write(arguments.output.join("input.raw.json"), &raw)?;
                serde_json::from_slice(&raw).map_err(io::Error::other)?
            }
            None => fixture(),
        };
        record(
            writer,
            &json!({"event":"state_pair", "backend":arguments.backend, "pair":pair,
            "input_origin_verification":"metadata supplied by input; lab does not execute checkpoint inference"}),
        )?;
        validate(&pair)?;
        fs::write(
            arguments.output.join("pair.json"),
            serde_json::to_vec_pretty(&pair).map_err(io::Error::other)?,
        )?;
        invalid_controls(writer)?;
        controls(&arguments.backend, writer)?;
        measured_arm(&pair, arguments, "guided", writer)?;
        let mut identity = pair;
        identity.strength = 0.0;
        measured_arm(&identity, arguments, "identity", writer)
    }

    pub fn run() -> io::Result<()> {
        let arguments = arguments()?;
        if let Some(parent) = arguments
            .output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir(&arguments.output)?;
        let mut writer = BufWriter::new(File::create(arguments.output.join("execution.jsonl"))?);
        record(
            &mut writer,
            &json!({
                "event":"run_metadata", "backend_requested":arguments.backend,
                "arguments":env::args().collect::<Vec<_>>(), "architecture":env::consts::ARCH,
                "operating_system":env::consts::OS, "samples_requested":arguments.samples,
                "model_inference_executed":false, "backend_fallback_permitted":false,
                "process_resource_measurement":"external native /usr/bin/time -l",
            }),
        )?;
        let result = session(&arguments, &mut writer);
        if let Err(error) = &result {
            record(
                &mut writer,
                &json!({"event":"run_failure", "backend":arguments.backend,
                "error":error.to_string(), "cpu_fallback_executed":false}),
            )?;
            fs::write(
                arguments.output.join("failure.json"),
                serde_json::to_vec_pretty(&json!({
                    "backend_requested":arguments.backend, "error":error.to_string(),
                    "cpu_fallback_executed":false,
                }))
                .map_err(io::Error::other)?,
            )?;
        }
        writer.flush()?;
        println!(
            "execution artifact: {}",
            arguments.output.join("execution.jsonl").display()
        );
        result
    }
}

#[cfg(feature = "proxima-tensor")]
fn main() -> io::Result<()> {
    lab::run()
}

#[cfg(not(feature = "proxima-tensor"))]
fn main() -> io::Result<()> {
    Err(io::Error::other(
        "loopcd_lab requires --features proxima-tensor",
    ))
}
