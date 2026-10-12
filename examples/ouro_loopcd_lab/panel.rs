use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

use proxima_tensor::QuantizedBlock;
use proxima_tokenizer::bpe::{decode_ids, encode_pretoken};
use serde_json::{Value, json};

use super::checkpoint::{Checkpoint, WeightMappingMode};
use super::controls;
use super::engine::Executor;
use super::graph;
use super::model::{Arm, Model, PANEL_ARMS, Readouts, pick};
use super::runner::{Arguments, record, stats};

const MANIFEST_DIGEST: &str = "887808271b373587c13192b2249bf743b7ffc31e4c804beb843a50d23e4a7830";
const EQUATION_TOLERANCE: f32 = 0.00001;

struct Case {
    identifier: String,
    pieces: Vec<String>,
    prompt: Vec<u8>,
    prefix: Vec<u32>,
    gold: Vec<u8>,
    gold_id: u32,
    timing: bool,
}

struct Score {
    selected: u32,
    gold_id: u32,
    hit: bool,
    rank: usize,
    logit: f32,
    log_probability: f64,
}

impl Score {
    fn value(&self, checkpoint: &Checkpoint) -> Value {
        json!({
            "selected_id": self.selected,
            "selected_bytes": checkpoint.vocab.token_bytes(self.selected),
            "selected_spelling": checkpoint.vocab.token_str(self.selected),
            "gold_id": self.gold_id,
            "gold_bytes": checkpoint.vocab.token_bytes(self.gold_id),
            "exact_hit": self.hit,
            "gold_rank": self.rank,
            "gold_logit": self.logit,
            "gold_log_probability": self.log_probability,
            "score_meaning": "exact next-token byte target; not generated-answer correctness",
            "confidence": null,
            "confidence_status": "unmeasured",
        })
    }
}

fn score(values: &[f32], gold_id: u32) -> io::Result<Score> {
    let selected = pick(values)?;
    let gold = *values
        .get(gold_id as usize)
        .ok_or_else(|| io::Error::other("gold ID outside vocabulary readout"))?;
    let maximum = values[selected as usize] as f64;
    let denominator = values
        .iter()
        .map(|value| ((*value as f64) - maximum).exp())
        .sum::<f64>();
    Ok(Score {
        selected,
        gold_id,
        hit: selected == gold_id,
        rank: 1 + values.iter().filter(|value| **value > gold).count(),
        logit: gold,
        log_probability: (gold as f64 - maximum) - denominator.ln(),
    })
}

fn equation_check(
    observed: &[f32],
    reference: &[f32],
    terminal: &[f32],
    strength: f32,
    tolerance: f32,
) -> io::Result<Value> {
    if observed.len() != reference.len() || observed.len() != terminal.len() {
        return Err(io::Error::other(
            "contrast equation vector dimensions differ",
        ));
    }
    let mut differences = Vec::new();
    for (coordinate, ((actual, early), final_value)) in
        observed.iter().zip(reference).zip(terminal).enumerate()
    {
        let expected = final_value + strength * (final_value - early);
        if !actual.is_finite() || !expected.is_finite() {
            return Err(io::Error::other(
                "contrast equation contains nonfinite logits",
            ));
        }
        if (actual - expected).abs() > tolerance {
            differences.push(json!({"coordinate":coordinate,"observed":actual,"expected":expected,"difference":actual-expected}));
        }
    }
    Ok(
        json!({"compared_elements":observed.len(),"strength":strength,"absolute_tolerance":tolerance,"mismatch_count":differences.len(),"complete_mismatches":differences}),
    )
}

fn deltas(diagnostic: &[f32], sample: &[f32]) -> io::Result<Value> {
    if diagnostic.len() != sample.len() {
        return Err(io::Error::other("sample and diagnostic dimensions differ"));
    }
    let mut differences = Vec::new();
    for (coordinate, (old, new)) in diagnostic.iter().zip(sample).enumerate() {
        if !old.is_finite() || !new.is_finite() {
            return Err(io::Error::other("sample delta contains nonfinite logits"));
        }
        if old != new {
            differences
                .push(json!({"coordinate":coordinate,"old":old,"new":new,"difference":new-old}));
        }
    }
    Ok(
        json!({"compared_elements":sample.len(),"difference_count":differences.len(),"complete_coordinate_differences":differences}),
    )
}

fn scoring_controls(writer: &mut impl Write) -> io::Result<()> {
    let fixtures = [
        (vec![0.0, 2.0, 1.0], 1, 1, true, 1, -0.4076059644443804),
        (vec![0.0, 2.0, 1.0], 0, 1, false, 3, -2.4076059644443806),
        (vec![2.0, 2.0, 1.0], 1, 0, false, 1, -0.8619948040582511),
        (
            vec![100.0, 102.0, 101.0],
            1,
            1,
            true,
            1,
            -0.4076059644443804,
        ),
    ];
    let mut observations = Vec::new();
    let mut mismatches = 0;
    let mut log_probabilities = Vec::new();
    for (logits, gold_id, expected_id, expected_hit, expected_rank, expected_probability) in
        fixtures
    {
        let observed = score(&logits, gold_id)?;
        let mismatch = observed.selected != expected_id
            || observed.hit != expected_hit
            || observed.rank != expected_rank
            || (observed.log_probability - expected_probability).abs() > 0.00001;
        mismatches += usize::from(mismatch);
        log_probabilities.push(observed.log_probability);
        observations.push(json!({"logits":logits,"gold_id":gold_id,"selected_id":observed.selected,"hit":observed.hit,"rank":observed.rank,"gold_log_probability":observed.log_probability,"expected":{"selected_id":expected_id,"hit":expected_hit,"rank":expected_rank,"gold_log_probability":expected_probability},"mismatch":mismatch}));
    }
    let wrong_label = score(&[0.0, 2.0, 1.0], 0)?;
    let shift_difference = (log_probabilities[0] - log_probabilities[3]).abs();
    record(
        writer,
        &json!({"kind":"scoring_oracle","fixture_checks":observations.len(),"fixture_mismatches":mismatches,"fixtures":observations,"shift_log_probability_difference":shift_difference,"absolute_tolerance":0.00001,"wrong_label_hit":wrong_label.hit,"wrong_label_gold_id":0,"wrong_label_expected_hit":false}),
    )?;
    if mismatches != 0 || wrong_label.hit || shift_difference > 0.00001 {
        return Err(io::Error::other(
            "independent scoring oracle differs from literal expectations",
        ));
    }
    Ok(())
}

fn contrast_controls(backend: &str, writer: &mut impl Write) -> io::Result<()> {
    let reference = [1.0, 4.0, 2.0];
    let terminal = [2.0, 1.0, 2.0];
    let expected = [2.5, -0.5, 2.0];
    let wrong_sign = [1.5, 2.5, 2.0];
    let strengths = [0.5; 3];
    let literal = equation_check(&expected, &reference, &terminal, 0.5, EQUATION_TOLERANCE)?;
    let wrong = equation_check(&wrong_sign, &reference, &terminal, 0.5, EQUATION_TOLERANCE)?;
    let bindings = [
        ("reference", QuantizedBlock::Float32(&reference)),
        ("terminal", QuantizedBlock::Float32(&terminal)),
        ("strength", QuantizedBlock::Float32(&strengths)),
    ];
    let mut executor = Executor::prepare(graph::contrast(3)?, backend, 1, &bindings, false)?;
    let actual = executor.run(&bindings, false)?.0;
    let tensor = equation_check(&actual, &reference, &terminal, 0.5, EQUATION_TOLERANCE)?;
    record(
        writer,
        &json!({"kind":"contrast_oracle","backend":backend,"reference":reference,"terminal":terminal,"strength":0.5,"independent_expected":expected,"literal_expected_check":literal,"actual_tensor_output":actual,"actual_tensor_check":tensor,"wrong_sign_literal":wrong_sign,"wrong_sign_check":wrong}),
    )?;
    if literal["mismatch_count"] != 0
        || tensor["mismatch_count"] != 0
        || wrong["mismatch_count"] != 2
    {
        return Err(io::Error::other(
            "contrast oracle or deliberately wrong direction control differs",
        ));
    }
    Ok(())
}

fn manifest(path: &Path) -> io::Result<Value> {
    let bytes = fs::read(path)?;
    let mut process = Command::new("/usr/bin/openssl")
        .args(["dgst", "-sha256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    process
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("manifest digest stdin absent"))?
        .write_all(&bytes)?;
    let output = process.wait_with_output()?;
    let digest = String::from_utf8(output.stdout).map_err(io::Error::other)?;
    if !output.status.success() || digest.split_whitespace().last() != Some(MANIFEST_DIGEST) {
        return Err(io::Error::other(
            "panel manifest bytes differ from frozen manifest SHA256",
        ));
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
    if value["cases"].as_array().map(Vec::len) != Some(8)
        || value["status"] != "frozen-before-inference"
        || value["strength"] != 0.5
    {
        return Err(io::Error::other(
            "panel manifest structure differs from admitted corpus",
        ));
    }
    Ok(value)
}

fn corpus(
    manifest: &Value,
    filter: Option<&[String]>,
    checkpoint: &Checkpoint,
    writer: &mut impl Write,
) -> io::Result<Vec<Case>> {
    let records = manifest["cases"]
        .as_array()
        .ok_or_else(|| io::Error::other("panel cases absent"))?;
    let requested: BTreeSet<&str> = filter.into_iter().flatten().map(String::as_str).collect();
    if let Some(filter) = filter {
        if filter.is_empty() || requested.len() != filter.len() || requested.contains("") {
            return Err(io::Error::other("panel case filter is empty or duplicated"));
        }
        for identifier in &requested {
            if !records
                .iter()
                .any(|value| value["id"].as_str() == Some(*identifier))
            {
                return Err(io::Error::other(format!("unknown panel case {identifier}")));
            }
        }
    }
    let mut cases = Vec::new();
    for value in records {
        let identifier = value["id"]
            .as_str()
            .ok_or_else(|| io::Error::other("case ID absent"))?;
        if filter.is_some() && !requested.contains(identifier) {
            continue;
        }
        let pieces: Vec<String> = value["pieces"]
            .as_array()
            .ok_or_else(|| io::Error::other("case pieces absent"))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(String::from)
                    .ok_or_else(|| io::Error::other("case piece is not text"))
            })
            .collect::<io::Result<_>>()?;
        let prompt = pieces.concat().into_bytes();
        let mut prefix = Vec::new();
        let mut piece_tokens = Vec::new();
        for piece in &pieces {
            let encoded =
                encode_pretoken(piece.as_bytes(), &checkpoint.vocab).map_err(io::Error::other)?;
            if decode_ids(&encoded, &checkpoint.vocab).map_err(io::Error::other)?
                != piece.as_bytes()
            {
                return Err(io::Error::other(format!(
                    "case {identifier} piece byte round trip differs"
                )));
            }
            piece_tokens.push(json!({"piece":piece,"bytes":piece.as_bytes(),"ids":encoded}));
            prefix.extend(encoded);
        }
        if prefix.is_empty()
            || prefix.len() > 65536
            || decode_ids(&prefix, &checkpoint.vocab).map_err(io::Error::other)? != prompt
        {
            return Err(io::Error::other(format!(
                "case {identifier} prompt byte round trip differs or prefix is empty/unbounded"
            )));
        }
        let gold = value["gold"]
            .as_str()
            .ok_or_else(|| io::Error::other("case gold text absent"))?
            .as_bytes()
            .to_vec();
        let gold_tokens = encode_pretoken(&gold, &checkpoint.vocab).map_err(io::Error::other)?;
        if gold_tokens.len() != 1
            || decode_ids(&gold_tokens, &checkpoint.vocab).map_err(io::Error::other)? != gold
        {
            return Err(io::Error::other(format!(
                "case {identifier} gold must round trip as one token"
            )));
        }
        let timing = value["timing"]
            .as_bool()
            .ok_or_else(|| io::Error::other("case timing flag absent"))?;
        record(
            writer,
            &json!({"kind":"panel_corpus","case_id":identifier,"pieces":pieces,"piece_tokens":piece_tokens,"prompt_bytes":prompt,"prompt":String::from_utf8_lossy(&prompt),"prefix_ids":prefix,"prefix_tokens":prefix.iter().map(|identifier| json!({"id":identifier,"bytes":checkpoint.vocab.token_bytes(*identifier),"spelling":checkpoint.vocab.token_str(*identifier)})).collect::<Vec<_>>(),"gold_id":gold_tokens[0],"gold_bytes":gold,"timing":timing,"byte_round_trip_violations":0,"special_tokens_inserted":false,"independent_hf_tokenizer_parity":"unmeasured"}),
        )?;
        cases.push(Case {
            identifier: identifier.to_owned(),
            pieces,
            prompt,
            prefix,
            gold,
            gold_id: gold_tokens[0],
            timing,
        });
    }
    Ok(cases)
}

fn candidate_scores(identifier: u32, output: &Readouts, checkpoint: &Checkpoint) -> Value {
    json!({"id":identifier,"bytes":checkpoint.vocab.token_bytes(identifier),"spelling":checkpoint.vocab.token_str(identifier),"first_logit":output.reference.as_ref().map(|values| values[identifier as usize]),"terminal_logit":output.head_logits[identifier as usize],"guided_logit":output.logits[identifier as usize]})
}

fn compare(
    case: &Case,
    indices: (usize, usize),
    scope: &str,
    outputs: &[Readouts],
    model: &Model<'_>,
    sources: &[String],
    writer: &mut impl Write,
) -> io::Result<()> {
    let (baseline_index, candidate_index) = indices;
    let checkpoint = model.checkpoint;
    let backend = &model.backend;
    let baseline = &outputs[baseline_index];
    let candidate = &outputs[candidate_index];
    let baseline_score = score(&baseline.logits, case.gold_id)?;
    let candidate_score = score(&candidate.logits, case.gold_id)?;
    let direction = match (baseline_score.hit, candidate_score.hit) {
        (false, true) => "gain",
        (true, false) => "loss",
        (true, true) => "unchanged_hit",
        (false, false) => "unchanged_miss",
    };
    let candidates: Vec<Value> = [baseline_score.selected, candidate_score.selected]
        .into_iter().map(|identifier| json!({"id":identifier,"bytes":checkpoint.vocab.token_bytes(identifier),"baseline":candidate_scores(identifier,baseline,checkpoint),"candidate":candidate_scores(identifier,candidate,checkpoint)})).collect();
    record(
        writer,
        &json!({"kind":"panel_comparison","case_id":case.identifier,"backend":backend,"prefix_ids":case.prefix,"gold_id":case.gold_id,"gold_bytes":case.gold,"scope":scope,"baseline_arm":PANEL_ARMS[baseline_index].name,"candidate_arm":PANEL_ARMS[candidate_index].name,"baseline_source":sources[baseline_index],"candidate_source":sources[candidate_index],"baseline_hit":baseline_score.hit,"candidate_hit":candidate_score.hit,"direction":direction,"reverse_direction":match direction {"gain"=>"loss","loss"=>"gain",other=>other},"selected_id_changed":baseline_score.selected!=candidate_score.selected,"baseline_score":baseline_score.value(checkpoint),"candidate_score":candidate_score.value(checkpoint),"gold_logit_difference":candidate_score.logit-baseline_score.logit,"gold_log_probability_difference":candidate_score.log_probability-baseline_score.log_probability,"both_candidates":candidates,"mechanism_boundary":"full readout vectors and candidate scores trace reranking only; pretrained factual representation remains unexplained","generalization_status":"unmeasured"}),
    )
}

fn model_controls(
    case: &Case,
    outputs: &[Readouts],
    sources: &[String],
    writer: &mut impl Write,
) -> io::Result<()> {
    for (guided_index, baseline_index) in [(2, 0), (3, 0), (4, 1), (5, 1)] {
        let arm = PANEL_ARMS[guided_index];
        let output = &outputs[guided_index];
        let reference = output
            .reference
            .as_deref()
            .ok_or_else(|| io::Error::other("diagnostic reference absent"))?;
        let strength = arm
            .strength()
            .ok_or_else(|| io::Error::other("guided strength absent"))?;
        let check = if strength == 0.0 {
            let mut check = deltas(&outputs[baseline_index].logits, &output.logits)?;
            check["mismatch_count"] = check["difference_count"].clone();
            check["absolute_tolerance"] = json!(0.0);
            check
        } else {
            equation_check(
                &output.logits,
                reference,
                &output.head_logits,
                strength,
                EQUATION_TOLERANCE,
            )?
        };
        record(
            writer,
            &json!({"kind":"model_contrast_control","case_id":case.identifier,"arm":arm.name,"depth":arm.depth,"strength":strength,"guided_source":sources[guided_index],"baseline_source":sources[baseline_index],"check":check}),
        )?;
        if check["mismatch_count"] != 0 {
            return Err(io::Error::other(format!(
                "case {} arm {} contrast control differs",
                case.identifier, arm.name
            )));
        }
    }
    Ok(())
}

fn sample_record(
    model: &Model<'_>,
    case: &Case,
    arm: Arm,
    timing: (usize, usize, f64),
    observed: &Readouts,
    diagnostic: &Readouts,
    source: &str,
) -> io::Result<Value> {
    let (sample, order, wall_ms) = timing;
    let expected_heads = if arm.strength().is_some() { 2 } else { 1 };
    if observed.head_executions != expected_heads
        || observed.reference.is_some() != arm.strength().is_some()
    {
        return Err(io::Error::other(
            "timed arm readout head count or reference boundary differs",
        ));
    }
    let reference_deltas = match (&diagnostic.reference, &observed.reference) {
        (Some(previous), Some(current)) => Some(deltas(previous, current)?),
        (Some(_), None) if arm.strength().is_none() => None,
        _ => return Err(io::Error::other("timed guided reference absent")),
    };
    Ok(
        json!({"kind":"panel_sample","case_id":case.identifier,"context":model.context(arm),"sample":sample,"arm_order_position":order,"wall_ms":wall_ms,"score":score(&observed.logits,case.gold_id)?.value(model.checkpoint),"terminal_score":score(&observed.head_logits,case.gold_id)?.value(model.checkpoint),"reference_score":observed.reference.as_ref().map(|values| score(values,case.gold_id).map(|score| score.value(model.checkpoint))).transpose()?,"head_executions":observed.head_executions,"layer_executions":observed.layer_executions,"branch_norm_sites":observed.branch_norm_sites,"loop_norm_sites":observed.loop_norm_sites,"diagnostic_source":source,"reference_logits_deltas":reference_deltas,"reference_absent_reason":if observed.reference.is_none() {Some("timed baseline executes terminal head only")} else {None},"terminal_logits_deltas":deltas(&diagnostic.head_logits,&observed.head_logits)?,"guided_logits_deltas":deltas(&diagnostic.logits,&observed.logits)?,"delta_reconstruction":"apply every listed old/new coordinate to corresponding diagnostic vector; absent changes retain diagnostic values; absent baseline reference is not reconstructed","dispatch_count":null,"readback_count":null,"device_allocation_bytes":null,"cpu_percent":null,"rss_bytes":null,"resource_counter_status":"unmeasured","cpu_fallback_executed":false}),
    )
}

pub fn run(arguments: &Arguments, path: &Path) -> io::Result<()> {
    let manifest = manifest(path)?;
    let mut writer = BufWriter::new(File::create(arguments.output.join("records.jsonl"))?);
    let mut samples = BufWriter::new(File::create(arguments.output.join("samples.jsonl"))?);
    let mut comparisons = BufWriter::new(File::create(arguments.output.join("comparisons.jsonl"))?);
    let setup_started = Instant::now();
    let mapping_mode = WeightMappingMode::for_run(&arguments.backend, arguments.metal_copy_weights);
    let checkpoint = Checkpoint::open(&arguments.directory, mapping_mode)?;
    for value in &checkpoint.inventory {
        record(&mut writer, value)?;
    }
    record(&mut writer, &controls::run(&arguments.backend)?)?;
    scoring_controls(&mut writer)?;
    contrast_controls(&arguments.backend, &mut writer)?;
    let cases = corpus(
        &manifest,
        arguments.panel_cases.as_deref(),
        &checkpoint,
        &mut writer,
    )?;
    record(
        &mut writer,
        &json!({"kind":"panel_setup","backend":arguments.backend,"manifest_path":path,"manifest_sha256":MANIFEST_DIGEST,"manifest_digest_source":"native openssl SHA256 over exact parsed input bytes","manifest":manifest,"selected_case_ids":cases.iter().map(|case| case.identifier.as_str()).collect::<Vec<_>>(),"arms":PANEL_ARMS.iter().map(|arm| json!({"name":arm.name,"depth":arm.depth,"strength":arm.strength()})).collect::<Vec<_>>(),"setup_wall_ms":setup_started.elapsed().as_secs_f64()*1000.0,"timing_excludes":"checkpoint parsing, embedding staging, plan preparation, controls, diagnostics, argmax selection, gold scoring, vector comparisons and JSON serialization","timing_includes":"full recurrent layers, loop norms, required head readouts, contrast, native submission/readback and output copies","original_single_prefix_timing_boundary":"original forward includes argmax; wall times from that campaign are not matched to this panel boundary","numeric_policy":"BF16 weight values, f32 activations; author BF16 activation rounding unmeasured","cpu_fallback_executions":0,"tuning_runs":0,"pretraining_contamination":"unknown","quality_interpretation":"frozen local exact next-token probes; no generalization verdict"}),
    )?;
    let mut summaries = Vec::new();
    let mut diagnostic_count = 0;
    let mut sample_count = 0;
    let mut comparison_count = 0;
    for case in &cases {
        let preparation_started = Instant::now();
        let mut model = Model::prepare(&checkpoint, case.prefix.clone(), &arguments.backend)?;
        record(
            &mut writer,
            &json!({"kind":"panel_case_setup","case_id":case.identifier,"pieces":case.pieces,"prompt_bytes":case.prompt,"prepare_wall_ms":preparation_started.elapsed().as_secs_f64()*1000.0}),
        )?;
        let mut diagnostics = Vec::new();
        let mut sources = Vec::new();
        for arm in PANEL_ARMS {
            let output = model.forward_readouts(arm, true)?;
            let source = format!(
                "{}#panel_diagnostic_readouts/{}/{}",
                arguments.output.join("records.jsonl").display(),
                case.identifier,
                arm.name
            );
            record(
                &mut writer,
                &json!({"kind":"panel_diagnostic_readouts","record_id":source,"case_id":case.identifier,"context":model.context(arm),"reference_logits":output.reference,"terminal_logits":output.head_logits,"guided_logits":output.logits,"score":score(&output.logits,case.gold_id)?.value(&checkpoint),"terminal_score":score(&output.head_logits,case.gold_id)?.value(&checkpoint),"reference_score":output.reference.as_ref().map(|values| score(values,case.gold_id).map(|score| score.value(&checkpoint))).transpose()?,"diagnostic_head_executions":output.head_executions,"vocabulary_length":output.logits.len(),"diagnostic_only":true,"warmup_before_timing":true,"hidden_activations_serialized":false}),
            )?;
            sources.push(source);
            diagnostics.push(output);
            diagnostic_count += 1;
        }
        model_controls(case, &diagnostics, &sources, &mut writer)?;
        for candidate in 1..PANEL_ARMS.len() {
            compare(
                case,
                (0, candidate),
                "versus_baseline4",
                &diagnostics,
                &model,
                &sources,
                &mut comparisons,
            )?;
            comparison_count += 1;
        }
        for (guided, baseline) in [(2, 0), (3, 0), (4, 1), (5, 1)] {
            compare(
                case,
                (baseline, guided),
                "guidance_versus_same_depth",
                &diagnostics,
                &model,
                &sources,
                &mut comparisons,
            )?;
            comparison_count += 1;
        }
        writer.flush()?;
        comparisons.flush()?;
        if !case.timing {
            continue;
        }
        let mut durations: Vec<Vec<f64>> = (0..PANEL_ARMS.len())
            .map(|_| Vec::with_capacity(arguments.samples))
            .collect();
        for sample in 0..arguments.samples {
            for order in 0..PANEL_ARMS.len() {
                let arm_index = (sample + order) % PANEL_ARMS.len();
                let arm = PANEL_ARMS[arm_index];
                let started = Instant::now();
                let observed = model.forward_readouts(arm, false)?;
                let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
                durations[arm_index].push(wall_ms);
                let value = sample_record(
                    &model,
                    case,
                    arm,
                    (sample, order, wall_ms),
                    &observed,
                    &diagnostics[arm_index],
                    &sources[arm_index],
                )?;
                record(&mut samples, &value)?;
                sample_count += 1;
            }
            samples.flush()?;
            eprintln!(
                "completed panel case {} sample {} of {} on {}",
                case.identifier,
                sample + 1,
                arguments.samples,
                arguments.backend
            );
        }
        for (arm_index, arm) in PANEL_ARMS.into_iter().enumerate() {
            summaries.push(json!({"kind":"panel_timing_summary","case_id":case.identifier,"context":model.context(arm),"cost":stats(&durations[arm_index],case.prefix.len()),"head_executions_per_sample":if arm.strength().is_some(){2}else{1},"dispatch_count":null,"readback_count":null,"device_allocation_bytes":null,"cpu_percent":null,"rss_bytes":null,"resource_counter_status":"unmeasured","unmeasured_counter_reason":"existing plan/execute surface does not attribute per-arm resources; native time wrapper records whole-process CPU/RSS separately","observed_error_count":0,"error_rate":0.0,"quality_status":"exact next-token target only; generalization unmeasured"}));
        }
    }
    record(
        &mut writer,
        &json!({"kind":"panel_control","gold_ranking_fixture_checks":4,"fixture_mismatches":0,"wrong_label_hit":0,"wrong_direction_contrast_mismatches":2,"model_zero_strength_comparisons":cases.len()*2,"model_zero_strength_compared_elements_per_vector":checkpoint.shape.vocab,"model_zero_strength_mismatch_count":0,"model_nonzero_equation_comparisons":cases.len()*2,"model_nonzero_compared_elements_per_vector":checkpoint.shape.vocab,"model_nonzero_absolute_tolerance":EQUATION_TOLERANCE,"model_nonzero_mismatch_count":0,"source":"scoring_oracle, contrast_oracle, and every model_contrast_control in records.jsonl; controls abort on mismatches"}),
    )?;
    let summary = json!({"kind":"panel_campaign_summary","backend":arguments.backend,"manifest_sha256":MANIFEST_DIGEST,"corpus_records":cases.len(),"diagnostic_records":diagnostic_count,"sample_records":sample_count,"comparison_records":comparison_count,"timed_case_arm_cells":summaries.len(),"byte_round_trip_violations":0,"observed_error_count":0,"error_rate":0.0,"cpu_fallback_executions":0,"cov_definition":"100 * sample SD / mean; sample SD divisor n-1","percentile_definition":"nearest rank ceil(p*n)","arm_order":"rotation: index=(sample+order_position)%6","timing_summaries":summaries,"resource_counter_status":"per-arm unmeasured; whole-process counters captured by external native time wrapper","author_activation_oracle_status":"unmeasured","generalization_status":"unmeasured"});
    serde_json::to_writer_pretty(
        File::create(arguments.output.join("summary.json"))?,
        &summary,
    )
    .map_err(io::Error::other)?;
    writer.flush()?;
    samples.flush()?;
    comparisons.flush()
}
