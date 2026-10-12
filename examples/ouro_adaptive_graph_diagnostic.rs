use std::process;

#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use std::collections::BTreeMap;
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use std::env;
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use std::fs::{self, File, OpenOptions};
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use std::io::{self, BufRead, BufReader, Write};
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use std::path::PathBuf;

#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use omega::backend::{self, Engine, GpuDriver};
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use proxima_model_interop::{
    HfConfig, RecurrentConfig, ouro_descriptor_from_hf_config, parse_hf_config,
};
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use proxima_tensor::NumericPolicy;
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use proxima_tensor::QuantizedBlock;
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use proxima_tensor::op::ScalarOp;
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use proxima_tensor::partition::partition_between_with_mapping_ancestor_closed;
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use proxima_tensor::shape;
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use proxima_tensor::spec::build_forward_with_readouts;
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use proxima_tensor::{LoopCdMode, NodeId, Op};
#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
use serde_json::{Value, json};

#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
#[path = "ouro_loopcd_lab/guidance.rs"]
mod guidance;

#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
mod diagnostic {
    use super::*;

    const MAX_STRENGTH: f32 = 0.5;

    struct SavedCase {
        label: &'static str,
        source: PathBuf,
        source_line: usize,
        value: Value,
        prefix_ids: Vec<u32>,
        terminal_logits: Vec<f32>,
        captured_strength: f32,
    }

    fn invalid_data(message: &'static str) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, message)
    }

    fn read_case(label: &'static str, path: PathBuf) -> io::Result<SavedCase> {
        let mut found = None;
        for (line_index, line) in BufReader::new(File::open(&path)?).lines().enumerate() {
            let line = line?;
            let value: Value = serde_json::from_str(&line).map_err(io::Error::other)?;
            if value.get("kind").and_then(Value::as_str) != Some("paper_selected_output")
                || value.get("method").and_then(Value::as_str) != Some("adaptive-logits")
            {
                continue;
            }
            if found.is_some() {
                return Err(invalid_data(
                    "C2 source contains multiple adaptive selected outputs",
                ));
            }
            let context = value
                .get("context")
                .ok_or_else(|| invalid_data("C2 selected output has no context"))?;
            if context.get("depth").and_then(Value::as_u64) != Some(4)
                || value.get("strength").and_then(Value::as_f64) != Some(f64::from(MAX_STRENGTH))
                || context.get("checkpoint_revision").and_then(Value::as_str)
                    != Some("574fa66cb8bf5abdc979642d01cf2b79b16bfab1")
                || context
                    .get("checkpoint_expected_sha256")
                    .and_then(Value::as_str)
                    != Some("58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af")
            {
                return Err(invalid_data(
                    "C2 selected output differs from the pinned four-depth half-strength checkpoint",
                ));
            }
            let numeric = context
                .get("backend_numeric_policy")
                .ok_or_else(|| invalid_data("C2 selected output lacks backend numeric policy"))?;
            for permission in [
                "approx_functions",
                "contraction",
                "epilogue_sources",
                "nan_assumptions",
                "reassociation",
                "signed_zero",
            ] {
                if numeric.get(permission).and_then(Value::as_bool) != Some(false) {
                    return Err(invalid_data("C2 source numeric policy is not bit exact"));
                }
            }
            if context.get("omega_config") != Some(&Value::Null)
                || context.get("use_cache").and_then(Value::as_bool) != Some(false)
                || context.get("backend").and_then(Value::as_str) != Some("metal")
                || context
                    .get("digest_verified_during_setup")
                    .and_then(Value::as_bool)
                    != Some(true)
                || context
                    .get("checkpoint_observed_sha256")
                    .and_then(Value::as_str)
                    != context
                        .get("checkpoint_expected_sha256")
                        .and_then(Value::as_str)
            {
                return Err(invalid_data(
                    "C2 source execution configuration differs from the uncached bit-exact capture",
                ));
            }
            let prefix_ids = value
                .get("prefix_ids")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid_data("C2 selected output has no prefix IDs"))?
                .iter()
                .map(|token| {
                    token
                        .as_u64()
                        .and_then(|token| u32::try_from(token).ok())
                        .ok_or_else(|| invalid_data("C2 prefix token is not a u32"))
                })
                .collect::<io::Result<Vec<_>>>()?;
            if context.get("positions").and_then(Value::as_u64) != Some(prefix_ids.len() as u64) {
                return Err(invalid_data(
                    "C2 prefix length differs from its recorded context",
                ));
            }
            let terminal_logits = value
                .get("terminal_logits")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid_data("C2 selected output has no terminal logits"))?
                .iter()
                .map(|logit| {
                    logit
                        .as_f64()
                        .map(|logit| logit as f32)
                        .filter(|logit| logit.is_finite())
                        .ok_or_else(|| invalid_data("C2 terminal logit is not finite"))
                })
                .collect::<io::Result<Vec<_>>>()?;
            let captured_strength = value
                .get("adaptive")
                .and_then(|adaptive| adaptive.get("effective_strength"))
                .and_then(Value::as_f64)
                .map(|strength| strength as f32)
                .filter(|strength| strength.is_finite())
                .ok_or_else(|| {
                    invalid_data("C2 selected output has no finite adaptive strength")
                })?;
            if value.get("effective_strength").and_then(Value::as_f64)
                != Some(f64::from(captured_strength))
            {
                return Err(invalid_data(
                    "C2 top-level strength differs from adaptive record",
                ));
            }
            found = Some(SavedCase {
                label,
                source: path.clone(),
                source_line: line_index + 1,
                value,
                prefix_ids,
                terminal_logits,
                captured_strength,
            });
        }
        found.ok_or_else(|| invalid_data("C2 source has no adaptive selected output"))
    }

    fn evaluate(
        engine: Engine,
        driver: Option<GpuDriver>,
        program: &[Op],
        named: &[(&str, QuantizedBlock<'_>)],
        symbols: &[u64],
        outputs: &[NodeId],
    ) -> io::Result<Vec<Vec<f32>>> {
        let shapes = shape::infer(program, symbols).map_err(io::Error::other)?;
        let mut plan = backend::plan_named_exact(
            engine,
            driver,
            program,
            symbols,
            named,
            outputs,
            NumericPolicy::bit_exact(),
        )
        .map_err(io::Error::other)?;
        let evaluated = backend::execute_plan_named(&mut plan, named).map_err(io::Error::other)?;
        outputs
            .iter()
            .map(|node| {
                let (values, concrete_shape) = evaluated.get(*node).ok_or_else(|| {
                    io::Error::other(format!("diagnostic output {} missing", node.0))
                })?;
                let expected_elements = shapes
                    .of(*node)
                    .iter()
                    .try_fold(1_u64, |product, extent| product.checked_mul(*extent))
                    .ok_or_else(|| {
                        invalid_data("diagnostic output shape element count overflow")
                    })?;
                if concrete_shape != shapes.of(*node)
                    || usize::try_from(expected_elements).ok() != Some(values.len())
                    || values.iter().any(|value| !value.is_finite())
                {
                    return Err(invalid_data(
                        "diagnostic output shape, payload length, or finiteness is invalid",
                    ));
                }
                Ok(values.to_vec())
            })
            .collect()
    }

    fn one_value(values: &[f32]) -> io::Result<f32> {
        match values {
            [value] => Ok(*value),
            _ => Err(invalid_data("adaptive strength root is not scalar")),
        }
    }

    fn denominator_diagnostics(
        segment: &[Op],
        metal_values: &[Vec<f32>],
        cpu_values: &[Vec<f32>],
    ) -> Value {
        let exponential =
            segment
                .iter()
                .enumerate()
                .find_map(|(index, operation)| match operation {
                    Op::Elementwise {
                        body: ScalarOp::Exponential,
                        ..
                    } => Some(NodeId(index as u32)),
                    _ => None,
                });
        let Some(exponential) = exponential else {
            return json!({"status": "no exponential node in extracted segment"});
        };
        let denominator =
            segment
                .iter()
                .enumerate()
                .find_map(|(index, operation)| match operation {
                    Op::Reduce(reduce)
                        if reduce.body == ScalarOp::Add && reduce.operand == exponential =>
                    {
                        Some(NodeId(index as u32))
                    }
                    _ => None,
                });
        let metal_exponentials = &metal_values[exponential.0 as usize];
        let cpu_exponentials = &cpu_values[exponential.0 as usize];
        let sum_in_f64 = |values: &[f32]| values.iter().map(|value| f64::from(*value)).sum::<f64>();
        json!({
            "exponential_node": exponential.0,
            "exponential_payload_len": metal_exponentials.len(),
            "derived_f64_sum_of_captured_metal_exponentials": sum_in_f64(metal_exponentials),
            "derived_f64_sum_of_captured_cpu_exponentials": sum_in_f64(cpu_exponentials),
            "denominator_reduce_node": denominator.map(|node| node.0),
            "metal_denominator": denominator
                .and_then(|node| metal_values.get(node.0 as usize))
                .and_then(|values| values.first())
                .copied(),
            "cpu_denominator": denominator
                .and_then(|node| cpu_values.get(node.0 as usize))
                .and_then(|values| values.first())
                .copied(),
        })
    }

    fn case_record(
        case: &SavedCase,
        segment: &[Op],
        cuts: &[(NodeId, String)],
        mapping: &BTreeMap<NodeId, NodeId>,
        mapped_strength: NodeId,
        mapped_terminal: NodeId,
        symbols: &[u64],
    ) -> io::Result<Value> {
        if cuts.len() != 1 || cuts[0].0 != original_for_segment(mapping, mapped_terminal)? {
            return Err(invalid_data(
                "adaptive partition did not produce its single terminal-logit cut",
            ));
        }
        let (cut_node, cut_name) = &cuts[0];
        let named_values = [(
            cut_name.as_str(),
            QuantizedBlock::Float32(&case.terminal_logits),
        )];
        let all_outputs: Vec<NodeId> = (0..segment.len())
            .map(|node| u32::try_from(node).map(NodeId).map_err(io::Error::other))
            .collect::<io::Result<_>>()?;
        let strength_only = evaluate(
            Engine::Gpu,
            Some(GpuDriver::Metal),
            segment,
            &named_values,
            symbols,
            &[mapped_strength],
        )?;
        let metal_strength_only = one_value(&strength_only[0])?;
        let cpu_strength_only = evaluate(
            Engine::Cpu,
            None,
            segment,
            &named_values,
            symbols,
            &[mapped_strength],
        )?;
        let cpu_strength_only = one_value(&cpu_strength_only[0])?;
        let metal_all = evaluate(
            Engine::Gpu,
            Some(GpuDriver::Metal),
            segment,
            &named_values,
            symbols,
            &all_outputs,
        )?;
        let cpu_all = evaluate(
            Engine::Cpu,
            None,
            segment,
            &named_values,
            symbols,
            &all_outputs,
        )?;
        let host =
            guidance::adaptive(&case.terminal_logits, MAX_STRENGTH).map_err(io::Error::other)?;
        if metal_all[mapped_terminal.0 as usize] != case.terminal_logits
            || cpu_all[mapped_terminal.0 as usize] != case.terminal_logits
        {
            return Err(invalid_data(
                "partition input node did not preserve the saved C2 terminal vector",
            ));
        }

        let maximum_index = case
            .terminal_logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or_else(|| invalid_data("terminal logits are empty"))?;
        let mut perturbed_terminal = case.terminal_logits.clone();
        perturbed_terminal[maximum_index] += 0.25;
        let perturbed_named = [(
            cut_name.as_str(),
            QuantizedBlock::Float32(&perturbed_terminal),
        )];
        let cpu_perturbed = evaluate(
            Engine::Cpu,
            None,
            segment,
            &perturbed_named,
            symbols,
            &[mapped_strength],
        )?;
        let cpu_perturbed_strength = one_value(&cpu_perturbed[0])?;

        let shapes = shape::infer(segment, symbols).map_err(io::Error::other)?;
        if metal_all.len() != cpu_all.len() || metal_all.len() != segment.len() {
            return Err(invalid_data(
                "backend output sets differ from the extracted operations",
            ));
        }
        let operations = segment
            .iter()
            .enumerate()
            .map(|(index, operation)| -> io::Result<Value> {
                let node = NodeId(index as u32);
                let original_node = original_for_segment(mapping, node)?;
                let differences = signed_difference(&metal_all[index], &cpu_all[index])?;
                Ok(json!({
                    "segment_node": node.0,
                    "original_node": original_node.0,
                    "kind": operation.kind(),
                    "name": operation.name(),
                    "operation": format!("{operation:?}"),
                    "shape": shapes.of(node),
                    "metal_values": &metal_all[index],
                    "cpu_values": &cpu_all[index],
                    "metal_minus_cpu": differences,
                }))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let metal_all_strength = one_value(&metal_all[mapped_strength.0 as usize])?;
        let cpu_strength = one_value(&cpu_all[mapped_strength.0 as usize])?;
        let difference_control = cpu_perturbed_strength - cpu_strength;
        if difference_control == 0.0 {
            return Err(invalid_data(
                "mutated-input negative control did not change strength",
            ));
        }
        Ok(json!({
            "case": case.label,
            "source_jsonl": case.source,
            "source_line": case.source_line,
            "source_kind": case.value.get("kind"),
            "source_method": case.value.get("method"),
            "source_context": case.value.get("context"),
            "source_adaptive_statistics": case.value.get("adaptive"),
        "prefix_ids": case.prefix_ids,
        "prefix_length": case.prefix_ids.len(),
        "symbols": symbols,
            "terminal_logits": case.terminal_logits,
            "terminal_root_original_node": cut_node.0,
            "terminal_input_name": cut_name,
            "terminal_cut_shape": shapes.of(mapped_terminal),
            "adaptive_strength_original_node": original_for_segment(mapping, mapped_strength)?.0,
            "adaptive_strength_segment_node": mapped_strength.0,
            "operations": operations,
            "softmax_denominator_diagnostics": denominator_diagnostics(segment, &metal_all, &cpu_all),
            "source_c2_strength": case.captured_strength,
            "metal_strength_only": metal_strength_only,
            "metal_strength_all_nodes_requested": metal_all_strength,
            "cpu_strength_only": cpu_strength_only,
            "cpu_strength": cpu_strength,
            "cpu_strength_only_minus_all_nodes": cpu_strength_only - cpu_strength,
            "metal_all_nodes_minus_cpu_all_nodes_strength": metal_all_strength - cpu_strength,
            "source_c2_minus_metal_strength_only": case.captured_strength - metal_strength_only,
            "source_c2_minus_metal_all_nodes_strength": case.captured_strength - metal_all_strength,
            "host_serial_guidance_statistics": {
                "top_probability": host.top_probability,
                "second_probability": host.second_probability,
                "margin": host.margin,
                "effective_strength": host.effective_strength,
            },
            "negative_control": {
                "mutated_input_index": maximum_index,
                "delta_added_to_maximum_logit": 0.25,
                "cpu_strength_after_mutation": cpu_perturbed_strength,
                "cpu_strength_delta": difference_control,
            },
        }))
    }

    fn signed_difference(left: &[f32], right: &[f32]) -> io::Result<Vec<f32>> {
        if left.len() != right.len() {
            return Err(invalid_data(
                "signed-difference vectors have different lengths",
            ));
        }
        Ok(left
            .iter()
            .zip(right)
            .map(|(left, right)| left - right)
            .collect())
    }

    fn original_for_segment(
        mapping: &BTreeMap<NodeId, NodeId>,
        segment_node: NodeId,
    ) -> io::Result<NodeId> {
        mapping
            .iter()
            .find_map(|(original, mapped)| (*mapped == segment_node).then_some(*original))
            .ok_or_else(|| {
                io::Error::other(format!(
                    "segment node {} lacks original mapping",
                    segment_node.0
                ))
            })
    }

    fn parse_args() -> io::Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
        let values = env::args_os().skip(1).collect::<Vec<_>>();
        if values.len() != 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "usage: ouro_adaptive_graph_diagnostic <model-dir> <france-records.jsonl> <ball-records.jsonl> <fresh-output.json>",
            ));
        }
        let mut values = values.into_iter();
        Ok((
            PathBuf::from(
                values
                    .next()
                    .ok_or_else(|| invalid_data("model-dir missing"))?,
            ),
            PathBuf::from(
                values
                    .next()
                    .ok_or_else(|| invalid_data("france records missing"))?,
            ),
            PathBuf::from(
                values
                    .next()
                    .ok_or_else(|| invalid_data("ball records missing"))?,
            ),
            PathBuf::from(
                values
                    .next()
                    .ok_or_else(|| invalid_data("output path missing"))?,
            ),
        ))
    }

    fn run() -> io::Result<()> {
        let host_guidance = guidance::Guidance::parse("adaptive-logits", MAX_STRENGTH)?;
        if host_guidance.name() != "adaptive-logits"
            || host_guidance.strength() != Some(MAX_STRENGTH)
            || !host_guidance.uses_logit_reference()
        {
            return Err(invalid_data(
                "diagnostic host guidance does not match the C2 adaptive arm",
            ));
        }
        let (model_directory, france_path, ball_path, output_path) = parse_args()?;
        let config_bytes = fs::read(model_directory.join("config.json"))?;
        let hf_config: HfConfig = parse_hf_config(&config_bytes).map_err(io::Error::other)?;
        let recurrent = RecurrentConfig {
            iterations: Some(4),
            mode: LoopCdMode::AdaptiveLogits,
            strength: MAX_STRENGTH,
            reference_iteration: 1,
        };
        let (architecture, mut descriptor) =
            ouro_descriptor_from_hf_config(&hf_config, recurrent).map_err(io::Error::other)?;
        if architecture.vocab != descriptor.vocab {
            return Err(invalid_data(
                "descriptor vocabulary differs from HF architecture",
            ));
        }
        descriptor.loopcd.mode = LoopCdMode::AdaptiveLogits;
        descriptor.loopcd.strength = MAX_STRENGTH;
        descriptor.loopcd.reference_iteration = 1;
        let built = build_forward_with_readouts(&descriptor, true).map_err(io::Error::other)?;
        let mut zero_cap_descriptor = descriptor.clone();
        zero_cap_descriptor.loopcd.strength = 0.0;
        let zero_cap_built =
            build_forward_with_readouts(&zero_cap_descriptor, true).map_err(io::Error::other)?;
        let readouts = built
            .readouts
            .ok_or_else(|| invalid_data("adaptive graph has no readout roots"))?;
        let terminal = readouts
            .terminal_logits
            .ok_or_else(|| invalid_data("adaptive graph has no terminal-logit root"))?;
        let strength = readouts
            .adaptive_strength
            .ok_or_else(|| invalid_data("adaptive graph has no strength root"))?;
        let cases = [
            read_case("france", france_path)?,
            read_case("ball", ball_path)?,
        ];
        let frozen_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("proxima-tensor/specs/omega-programmable/frozen-prefixes.json");
        let frozen: Value =
            serde_json::from_slice(&fs::read(frozen_path)?).map_err(io::Error::other)?;
        if frozen.get("checkpoint_revision").and_then(Value::as_str)
            != Some("574fa66cb8bf5abdc979642d01cf2b79b16bfab1")
            || frozen.get("checkpoint_sha256").and_then(Value::as_str)
                != Some("58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af")
        {
            return Err(invalid_data(
                "frozen prefix register differs from the pinned checkpoint",
            ));
        }
        let mut records = Vec::with_capacity(cases.len());
        let mut graph_description = None;
        for case in &cases {
            if case.terminal_logits.len() != descriptor.vocab as usize
                || case.value.get("vocabulary_length").and_then(Value::as_u64)
                    != Some(u64::from(descriptor.vocab))
            {
                return Err(invalid_data(
                    "saved terminal logits differ from the pinned vocabulary length",
                ));
            }
            let frozen_ids = frozen
                .get("cases")
                .and_then(Value::as_array)
                .and_then(|cases| {
                    cases
                        .iter()
                        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(case.label))
                })
                .and_then(|entry| entry.get("prefix_ids"))
                .and_then(Value::as_array)
                .ok_or_else(|| invalid_data("frozen prefix register lacks a named case"))?
                .iter()
                .map(|token| {
                    token
                        .as_u64()
                        .and_then(|token| u32::try_from(token).ok())
                        .ok_or_else(|| invalid_data("frozen prefix token is not a u32"))
                })
                .collect::<io::Result<Vec<_>>>()?;
            if case.prefix_ids != frozen_ids {
                return Err(invalid_data(
                    "C2 prefix does not match the named frozen-prefix record",
                ));
            }
            let prefix_length = case.prefix_ids.len() as u64;
            let symbols = [prefix_length, prefix_length, 0];
            let (segment, cuts, mapping) = partition_between_with_mapping_ancestor_closed(
                &built.forward.0,
                &symbols,
                Some(terminal),
                strength,
            )
            .map_err(io::Error::other)?;
            let mapped_terminal = mapping
                .get(&terminal)
                .copied()
                .ok_or_else(|| invalid_data("partition omitted the terminal-logit boundary"))?;
            let mapped_strength = mapping
                .get(&strength)
                .copied()
                .ok_or_else(|| invalid_data("partition omitted adaptive strength"))?;
            let zero_cap_readouts = zero_cap_built
                .readouts
                .ok_or_else(|| invalid_data("zero-cap graph has no readout roots"))?;
            let zero_cap_terminal_original = zero_cap_readouts
                .terminal_logits
                .ok_or_else(|| invalid_data("zero-cap graph has no terminal-logit root"))?;
            let zero_cap_strength_original = zero_cap_readouts
                .adaptive_strength
                .ok_or_else(|| invalid_data("zero-cap graph has no adaptive root"))?;
            if zero_cap_terminal_original != terminal {
                return Err(invalid_data(
                    "zero-cap graph changed the terminal-logit root identity",
                ));
            }
            let (zero_cap_segment, zero_cap_cuts, zero_cap_mapping) =
                partition_between_with_mapping_ancestor_closed(
                    &zero_cap_built.forward.0,
                    &symbols,
                    Some(zero_cap_terminal_original),
                    zero_cap_strength_original,
                )
                .map_err(io::Error::other)?;
            let zero_cap_terminal = zero_cap_mapping
                .get(&zero_cap_terminal_original)
                .copied()
                .ok_or_else(|| invalid_data("zero-cap graph omitted terminal-logit boundary"))?;
            let zero_cap_strength = zero_cap_mapping
                .get(&zero_cap_strength_original)
                .copied()
                .ok_or_else(|| invalid_data("zero-cap graph omitted strength output"))?;
            if zero_cap_cuts.len() != 1 || zero_cap_cuts[0].0 != zero_cap_terminal_original {
                return Err(invalid_data(
                    "zero-cap graph did not retain one terminal cut",
                ));
            }
            let zero_cap_named = [(
                zero_cap_cuts[0].1.as_str(),
                QuantizedBlock::Float32(&case.terminal_logits),
            )];
            let zero_cap_outputs = [zero_cap_terminal, zero_cap_strength];
            let zero_cap_values = evaluate(
                Engine::Cpu,
                None,
                &zero_cap_segment,
                &zero_cap_named,
                &symbols,
                &zero_cap_outputs,
            )?;
            if zero_cap_values[0] != case.terminal_logits || one_value(&zero_cap_values[1])? != 0.0
            {
                return Err(invalid_data(
                    "zero-cap graph did not preserve input and produce zero strength",
                ));
            }
            if graph_description.is_none() {
                graph_description = Some(json!({
                "architecture": {
                    "vocab": architecture.vocab,
                    "embedding": architecture.embedding,
                    "block_count": architecture.block_count,
                    "query_heads": architecture.query_heads,
                    "kv_heads": architecture.kv_heads,
                    "head_dim": architecture.head_dim,
                },
                "descriptor_vocab": descriptor.vocab,
                    "terminal_logits_original_node": terminal.0,
                    "adaptive_strength_original_node": strength.0,
                    "symbols": symbols,
                    "partition_node_count": segment.len(),
                    "partition_cut_count": cuts.len(),
                }));
            }
            records.push(case_record(
                case,
                &segment,
                &cuts,
                &mapping,
                mapped_strength,
                mapped_terminal,
                &symbols,
            )?);
            let record = records
                .last_mut()
                .ok_or_else(|| invalid_data("case record was not appended"))?;
            let cap_operation = &zero_cap_built.forward.0[zero_cap_strength_original.0 as usize];
            let cap_dependencies = cap_operation.dependencies();
            let cap_constant = cap_dependencies.iter().find_map(|node| {
                match zero_cap_built.forward.0.get(node.0 as usize) {
                    Some(Op::Constant { value, .. }) => Some((node, value)),
                    _ => None,
                }
            });
            let Some((cap_node, cap_value)) = cap_constant else {
                return Err(invalid_data(
                    "zero-cap strength operation does not expose its cap constant",
                ));
            };
            if *cap_value != 0.0 {
                return Err(invalid_data("zero-cap graph constant is not zero"));
            }
            record["negative_control"]["zero_cap"] = json!({
                "cap": 0.0,
                "strength_node_original": zero_cap_strength_original.0,
                "strength_operation": format!("{cap_operation:?}"),
                "cap_constant_original_node": cap_node.0,
                "cap_constant_operation": format!("{:?}", zero_cap_built.forward.0[cap_node.0 as usize]),
                "cpu_strength": one_value(&zero_cap_values[1])?,
            });
        }
        let document = json!({
            "diagnostic": "ouro_shared_adaptive_strength_graph",
            "contract": "proxima-tensor/specs/ouro-mainline/results/weight-owner/adaptive-strength-diagnostic-contract.md",
        "timed": false,
        "numeric_policy": format!("{:?}", NumericPolicy::bit_exact()),
        "planner_api": "omega::backend::plan_named_exact; Metal exact activation path delegates to plan_named with its existing output-set-dependent planner behavior",
        "graph": graph_description,
            "records": records,
        });
        let bytes = serde_json::to_vec(&document).map_err(io::Error::other)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output_path)?;
        output.write_all(&bytes)?;
        output.write_all(b"\n")?;
        Ok(())
    }

    pub(super) fn run_cli() {
        if let Err(error) = run() {
            eprintln!("ouro adaptive graph diagnostic failed: {error}");
            process::exit(1);
        }
    }
}

#[cfg(all(feature = "proxima-tensor", target_os = "macos"))]
fn main() {
    diagnostic::run_cli();
}

#[cfg(not(all(feature = "proxima-tensor", target_os = "macos")))]
fn main() {
    eprintln!("build with --features proxima-tensor on macOS to run the Metal diagnostic");
    process::exit(1);
}
