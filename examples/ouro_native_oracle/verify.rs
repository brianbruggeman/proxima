use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::compare::{self, ABSOLUTE, Archive, Comparisons, JsonLines, RELATIVE, Vector};

pub fn pick(values: &[f32]) -> io::Result<u32> {
    let mut selected = None;
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(io::Error::other("selection logits nonfinite"));
        }
        if selected.is_none_or(|(_, maximum)| *value > maximum) {
            selected = Some((index, *value));
        }
    }
    u32::try_from(
        selected
            .ok_or_else(|| io::Error::other("selection logits absent"))?
            .0,
    )
    .map_err(io::Error::other)
}

fn report(path: &Path, value: &Value) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value).map_err(io::Error::other)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!(
        "{}",
        serde_json::to_string(value).map_err(io::Error::other)?
    );
    Ok(())
}

fn serialized_float(value: f64) -> io::Result<Value> {
    let text = serde_json::to_string(&value).map_err(io::Error::other)?;
    serde_json::from_str(&text).map_err(io::Error::other)
}

fn vector_records(path: &Path) -> io::Result<BTreeMap<String, Vector>> {
    let mut vectors = BTreeMap::new();
    compare::visit_records(path, |line, record| {
        let key = record["key"]
            .as_str()
            .ok_or_else(|| io::Error::other("verification vector key absent"))?;
        let shape: Vec<usize> =
            serde_json::from_value(record["shape"].clone()).map_err(io::Error::other)?;
        let values = compare::read_values(&record, "values")?;
        if values.len() != shape.iter().product::<usize>() {
            return Err(io::Error::other("verification shape differs"));
        }
        if vectors
            .insert(
                key.to_owned(),
                Vector {
                    values,
                    shape,
                    path: path.to_path_buf(),
                    line,
                    field: "values".to_owned(),
                },
            )
            .is_some()
        {
            return Err(io::Error::other("verification vector duplicate"));
        }
        Ok(())
    })?;
    Ok(vectors)
}

struct AuditVector {
    actual: Vec<f32>,
    reference: Vec<f32>,
    seen: Vec<bool>,
    shape: Vec<usize>,
    non_exact: usize,
}

pub fn artifacts(directory: &Path) -> io::Result<()> {
    let oracle = Archive::oracle(directory)?;
    oracle.require_complete()?;
    let prefix: Vec<u32> =
        serde_json::from_value(oracle.context["prefix_ids"].clone()).map_err(io::Error::other)?;
    let archive_path = PathBuf::from(
        oracle.context["source_archive"]
            .as_str()
            .ok_or_else(|| io::Error::other("source archive path absent"))?,
    );
    let source = Archive::open(&archive_path, &prefix)?;
    let replay_path = directory.join("replay-vectors.jsonl");
    let replay = vector_records(&replay_path)?;
    if replay.len() != 96 {
        return Err(io::Error::other("replay vectors expected=96"));
    }
    let mut audits = Vec::new();
    let mut keys = BTreeSet::new();
    let mut total_coordinates = 0;
    let mut total_non_exact = 0;
    let mut total_exceeding = 0;
    let mut end_to_end = 0;
    let mut replay_count = 0;
    compare::visit_records(&directory.join("comparisons.jsonl"), |line, record| {
        let identifier = record["id"]
            .as_u64()
            .ok_or_else(|| io::Error::other("comparison identifier absent"))?
            as usize;
        if identifier != audits.len() {
            return Err(io::Error::other("comparison IDs duplicate or incomplete"));
        }
        let key = record["key"]
            .as_str()
            .ok_or_else(|| io::Error::other("comparison key absent"))?;
        let condition = record["condition"]
            .as_str()
            .ok_or_else(|| io::Error::other("comparison condition absent"))?;
        if !keys.insert(format!("{condition}/{key}")) {
            return Err(io::Error::other("comparison boundary duplicated"));
        }
        let actual = match condition {
            "end_to_end_f32" => {
                end_to_end += 1;
                oracle.get(key)?
            }
            "archived_input_layer_replay" => {
                replay_count += 1;
                replay
                    .get(key)
                    .ok_or_else(|| io::Error::other("replay comparison key absent"))?
            }
            _ => return Err(io::Error::other("unexpected comparison condition")),
        };
        let reference = source.get(key)?;
        if actual.shape != reference.shape
            || record["shape"] != json!(actual.shape)
            || record["actual"]["path"] != json!(actual.path)
            || record["actual"]["line"] != actual.line
            || record["actual"]["field"] != actual.field
            || record["reference"]["path"] != json!(reference.path)
            || record["reference"]["line"] != reference.line
            || record["reference"]["field"] != reference.field
            || record["absolute_tolerance"] != ABSOLUTE
            || record["relative_tolerance"] != RELATIVE
        {
            return Err(io::Error::other(format!(
                "comparison provenance or threshold differs at line {line}"
            )));
        }
        let mut non_exact = 0;
        let mut exceeding = 0;
        let mut positive = 0;
        let mut negative = 0;
        let mut zero_bits = 0;
        for (actual_value, reference_value) in actual.values.iter().zip(&reference.values) {
            let delta = f64::from(*actual_value) - f64::from(*reference_value);
            let tolerance =
                0.003 + 0.003 * f64::from(actual_value.abs().max(reference_value.abs()));
            if actual_value.to_bits() != reference_value.to_bits() {
                non_exact += 1;
                if delta > 0.0 {
                    positive += 1;
                } else if delta < 0.0 {
                    negative += 1;
                } else {
                    zero_bits += 1;
                }
            }
            if delta.abs() > tolerance {
                exceeding += 1;
            }
        }
        if record["coordinates"] != actual.values.len()
            || record["exact_equal"] != actual.values.len() - non_exact
            || record["exceeding"] != exceeding
            || record["positive"] != positive
            || record["negative"] != negative
            || record["zero_difference_bits"] != zero_bits
            || record["within_tolerance"] != actual.values.len() - exceeding
            || record["nonfinite"] != 0
        {
            return Err(io::Error::other(format!(
                "comparison recount differs at line {line}"
            )));
        }
        total_coordinates += actual.values.len();
        total_non_exact += non_exact;
        total_exceeding += exceeding;
        audits.push(AuditVector {
            actual: actual.values.clone(),
            reference: reference.values.clone(),
            seen: vec![false; actual.values.len()],
            shape: actual.shape.clone(),
            non_exact,
        });
        Ok(())
    })?;
    let mut difference_rows = 0;
    compare::visit_records(&directory.join("differences.jsonl"), |line, record| {
        let identifier = record["comparison"]
            .as_u64()
            .ok_or_else(|| io::Error::other("difference comparison ID absent"))?
            as usize;
        let index = record["index"]
            .as_u64()
            .ok_or_else(|| io::Error::other("difference index absent"))?
            as usize;
        let vector = audits
            .get_mut(identifier)
            .ok_or_else(|| io::Error::other("difference comparison ID outside set"))?;
        let actual = *vector
            .actual
            .get(index)
            .ok_or_else(|| io::Error::other("difference index outside vector"))?;
        let reference = vector.reference[index];
        if vector.seen[index] || actual.to_bits() == reference.to_bits() {
            return Err(io::Error::other("difference row duplicate or exact"));
        }
        vector.seen[index] = true;
        let difference = f64::from(actual) - f64::from(reference);
        let threshold = 0.003 + 0.003 * f64::from(actual.abs().max(reference.abs()));
        let direction = if difference > 0.0 {
            "positive"
        } else if difference < 0.0 {
            "negative"
        } else {
            "zero_difference_bits"
        };
        let coordinates: Vec<usize> = vector
            .shape
            .iter()
            .enumerate()
            .map(|(dimension, extent)| {
                (index / vector.shape[dimension + 1..].iter().product::<usize>()) % extent
            })
            .collect();
        if record["actual_bits"] != actual.to_bits()
            || record["reference_bits"] != reference.to_bits()
            || record["actual"]
                .as_f64()
                .map(|value| (value as f32).to_bits())
                != Some(actual.to_bits())
            || record["reference"]
                .as_f64()
                .map(|value| (value as f32).to_bits())
                != Some(reference.to_bits())
            || record["difference"] != serialized_float(difference)?
            || record["absolute_error"] != serialized_float(difference.abs())?
            || record["threshold"] != serialized_float(threshold)?
            || record["exceeding"] != (difference.abs() > threshold)
            || record["direction"] != direction
            || record["coordinates"] != json!(coordinates)
        {
            return Err(io::Error::other(format!(
                "difference payload differs at line {line}"
            )));
        }
        difference_rows += 1;
        Ok(())
    })?;
    for vector in &audits {
        if vector.seen.iter().filter(|seen| **seen).count() != vector.non_exact {
            return Err(io::Error::other("difference coverage incomplete"));
        }
    }
    if audits.len() != 298
        || end_to_end != 202
        || replay_count != 96
        || total_coordinates != prefix.len() * 2048 * 296 + 49152 * 2
        || difference_rows != total_non_exact
    {
        return Err(io::Error::other("artifact acceptance coverage differs"));
    }
    report(
        &directory.join("verification.json"),
        &json!({"comparison_records":audits.len(),"end_to_end_comparisons":end_to_end,"replay_comparisons":replay_count,"total_compared_coordinates":total_coordinates,"non_exact_coordinates":total_non_exact,"difference_rows":difference_rows,"exceeding_coordinates":total_exceeding,"missing_duplicate_records":0,"coverage_violations":0,"provenance_violations":0,"recount_path":"reread all vector and coordinate payloads; independent coordinate loop and bitset coverage"}),
    )
}

pub fn contrasts(directory: &Path) -> io::Result<()> {
    let archive = Archive::oracle(directory)?;
    let reference = &archive.get("reference_logits")?.values;
    let terminal = &archive.get("terminal_logits")?.values;
    let mut strengths = BTreeSet::new();
    let mut coordinates = 0;
    let mut zero_differences = 0;
    let mut half_differences = 0;
    compare::visit_records(&directory.join("contrasts.jsonl"), |_, record| {
        let strength = record["strength"]
            .as_f64()
            .ok_or_else(|| io::Error::other("contrast strength absent"))?
            as f32;
        if ![0.0, 0.5].contains(&strength) || !strengths.insert(strength.to_bits()) {
            return Err(io::Error::other("contrast strengths differ"));
        }
        let actual = compare::read_values(&record, "values")?;
        if actual.len() != 49152 {
            return Err(io::Error::other("contrast vocabulary shape differs"));
        }
        for (index, value) in actual.iter().enumerate() {
            let difference = terminal[index] - reference[index];
            let scaled = strength * difference;
            let expected = terminal[index] + scaled;
            if strength == 0.0 && value.to_bits() != terminal[index].to_bits() {
                zero_differences += 1;
            }
            if strength == 0.5 && value.to_bits() != expected.to_bits() {
                half_differences += 1;
            }
            coordinates += 1;
        }
        Ok(())
    })?;
    if strengths.len() != 2 || zero_differences != 0 || half_differences != 0 {
        return Err(io::Error::other("contrast verification differs"));
    }
    report(
        &directory.join("contrast-verification.json"),
        &json!({"strengths":2,"vocabulary_per_strength":49152,"contrast_coordinates":coordinates,"zero_strength_bit_differences":zero_differences,"half_strength_equation_mismatches":half_differences,"reference_source":archive.get("reference_logits")?.line,"terminal_source":archive.get("terminal_logits")?.line}),
    )
}

pub fn modes(f32_directory: &Path, bf16_directory: &Path, output: &Path) -> io::Result<Value> {
    let f32_archive = Archive::oracle(f32_directory)?;
    let bf16_archive = Archive::oracle(bf16_directory)?;
    f32_archive.require_complete()?;
    bf16_archive.require_complete()?;
    if f32_archive.context["mode"] != "f32" || bf16_archive.context["mode"] != "bf16" {
        return Err(io::Error::other(
            "mode comparison requires labeled f32 and bf16 references",
        ));
    }
    for field in [
        "prefix_ids",
        "checkpoint_revision",
        "checkpoint_observed_sha256",
        "depth",
        "use_cache",
        "readout",
    ] {
        if f32_archive.context[field] != bf16_archive.context[field] {
            return Err(io::Error::other(format!(
                "dtype comparison condition {field} differs"
            )));
        }
    }
    let mut comparisons = Comparisons::create(output)?;
    for (key, actual) in &bf16_archive.vectors {
        comparisons.compare(
            key,
            actual,
            f32_archive.get(key)?,
            "bf16_source_semantics_vs_f32",
        )?;
    }
    comparisons.finish()
}

pub fn independence(directory: &Path) -> io::Result<Value> {
    let mut writer = JsonLines::create(&directory.join("independence.jsonl"))?;
    let mut raw_accessors = 0;
    let mut violations = 0;
    for (path, source) in [
        (
            "examples/ouro_native_oracle/math.rs",
            include_str!("math.rs"),
        ),
        (
            "examples/ouro_native_oracle/model.rs",
            include_str!("model.rs"),
        ),
        (
            "examples/ouro_native_oracle/compare.rs",
            include_str!("compare.rs"),
        ),
    ] {
        let mut calls = Vec::new();
        for (index, line) in source.lines().enumerate() {
            for prohibited in [
                "proxima_tensor::",
                "omega::",
                "proxima_gguf::",
                "checkpoint.packed(",
                "checkpoint.embedding_rows(",
                "checkpoint.layer_scales",
                "checkpoint.final_scale",
                "Executor::",
                "Model::forward",
            ] {
                if line.contains(prohibited) {
                    violations += 1;
                    calls.push(json!({"line":index+1,"prohibited":prohibited}));
                }
            }
            raw_accessors += line.matches("checkpoint.bytes(").count();
        }
        writer.emit(&json!({"kind":"arithmetic_source_audit","path":path,"read_line_range":[1,source.lines().count()],"prohibited_calls":calls,"ffi_call_lines":source.lines().enumerate().filter(|(_,line)|line.contains("cblas_sgemm(")).map(|(index,_)|index+1).collect::<Vec<_>>()}))?;
    }
    writer.emit(&json!({"kind":"shared_boundaries","modules":["examples/ouro_loopcd_lab/checkpoint.rs","examples/ouro_loopcd_lab/graph.rs::Shape"],"checkpoint_open_side_effect":"cached Proxima scale decode occurs but its fields never enter oracle computation","manual_source_inspection_required":true}))?;
    writer.flush()?;
    if violations != 0 || raw_accessors == 0 {
        return Err(io::Error::other("oracle independence lexical gate differs"));
    }
    Ok(
        json!({"arithmetic_modules_audited":3,"prohibited_arithmetic_calls":violations,"shared_source_modules":2,"raw_byte_accessor_sites":raw_accessors,"dependency_declaration_edits":0,"semantic_independence_verdict":"not inferred from lexical audit"}),
    )
}

pub fn paper(
    candidate_path: &Path,
    oracle_directory: &Path,
    directory: &Path,
) -> io::Result<Value> {
    let oracle_path = oracle_directory.join("paper-output.jsonl");
    let mut oracle_record = None;
    let mut oracle_line = 0;
    compare::visit_records(&oracle_path, |line, record| {
        if record["kind"] == "paper_selected_output" {
            if oracle_record.is_some() {
                return Err(io::Error::other("oracle paper record duplicated"));
            }
            oracle_line = line;
            oracle_record = Some(record);
        }
        Ok(())
    })?;
    let oracle = oracle_record.ok_or_else(|| io::Error::other("oracle paper record absent"))?;
    if oracle["context"]["mode"] != "f32" {
        return Err(io::Error::other(
            "paper numerical comparison requires F32 oracle mode",
        ));
    }
    let method = oracle["method"]
        .as_str()
        .ok_or_else(|| io::Error::other("oracle paper method absent"))?;
    let mut comparisons = Comparisons::create(directory)?;
    let mut selected_records = 0;
    let mut scalar_exceeding = 0usize;
    let mut selection_differences = 0usize;
    let mut scalar_comparisons = JsonLines::create(&directory.join("scalar-comparisons.jsonl"))?;
    let mut selections = JsonLines::create(&directory.join("selection-comparisons.jsonl"))?;
    compare::visit_records(candidate_path, |line, candidate| {
        if candidate["kind"] != "paper_selected_output"
            || candidate["method"] != method
            || candidate["context"]["prefix_ids"] != oracle["context"]["prefix_ids"]
        {
            return Ok(());
        }
        for field in [
            "checkpoint_revision",
            "checkpoint_expected_sha256",
            "checkpoint_observed_sha256",
            "depth",
            "use_cache",
        ] {
            if candidate["context"][field] != oracle["context"][field] {
                return Err(io::Error::other(format!(
                    "paper comparison condition {field} differs"
                )));
            }
        }
        let actual_logits = compare::read_values(&candidate, "guided_logits")?;
        let reference_logits = compare::read_values(&oracle, "guided_logits")?;
        let actual_selected = pick(&actual_logits)?;
        let reference_selected = pick(&reference_logits)?;
        if candidate["selected_id"] != actual_selected
            || oracle["selected_id"] != reference_selected
        {
            return Err(io::Error::other(
                "selected ID differs from full-vector argmax",
            ));
        }
        if candidate["selected_bytes"].is_null() || oracle["selected_bytes"].is_null() {
            return Err(io::Error::other("selected candidate bytes absent"));
        }
        if actual_selected != reference_selected
            || candidate["selected_bytes"] != oracle["selected_bytes"]
        {
            selection_differences += 1;
        }
        selections.emit(&json!({"kind":"selection_comparison","actual_id":actual_selected,"reference_id":reference_selected,"actual_bytes":candidate["selected_bytes"],"reference_bytes":oracle["selected_bytes"],"equal":actual_selected==reference_selected && candidate["selected_bytes"]==oracle["selected_bytes"],"actual_source":{"path":candidate_path,"line":line},"reference_source":{"path":oracle_path,"line":oracle_line}}))?;
        if candidate["strength"] != oracle["strength"]
            || candidate["context"]["strength"] != oracle["context"]["strength"]
        {
            return Err(io::Error::other(
                "paper comparison strength or adaptive cap differs",
            ));
        }
        if method == "baseline"
            && (!candidate["effective_strength"].is_null()
                || !oracle["effective_strength"].is_null())
        {
            return Err(io::Error::other(
                "baseline requires absent effective guidance strength",
            ));
        }
        let policy = candidate["context"]["numeric_policy"]
            .as_str()
            .unwrap_or_default();
        if !policy.contains("f32 activation arithmetic")
            || !policy.contains("author BF16 intermediate rounding not reproduced")
        {
            return Err(io::Error::other(
                "paper candidate F32 numeric policy absent",
            ));
        }
        selected_records += 1;
        if selected_records != 1 {
            return Err(io::Error::other("paper candidate record duplicated"));
        }
        let fields: &[&str] = match method {
            "baseline" => &["terminal_logits", "guided_logits"],
            "fixed-logits" | "adaptive-logits" => {
                &["reference_logits", "terminal_logits", "guided_logits"]
            }
            "fixed-hidden" => &[
                "raw_reference_hidden",
                "raw_terminal_hidden",
                "contrasted_hidden",
                "normalized_hidden",
                "guided_logits",
            ],
            _ => return Err(io::Error::other("unknown oracle paper method")),
        };
        for field in fields {
            let actual = compare::read_values(&candidate, field)?;
            let reference = compare::read_values(&oracle, field)?;
            let width = if field.contains("hidden") {
                2048
            } else {
                49152
            };
            if actual.len() != width || reference.len() != width {
                return Err(io::Error::other("paper vector dimensions differ"));
            }
            comparisons.compare(
                field,
                &Vector {
                    values: actual,
                    shape: vec![width],
                    path: candidate_path.to_path_buf(),
                    line,
                    field: (*field).to_owned(),
                },
                &Vector {
                    values: reference,
                    shape: vec![width],
                    path: oracle_path.clone(),
                    line: oracle_line,
                    field: (*field).to_owned(),
                },
                method,
            )?;
        }
        for field in [
            "head_executions",
            "layer_executions",
            "branch_norm_sites",
            "loop_norm_sites",
        ] {
            if candidate[field] != oracle[field] {
                return Err(io::Error::other(format!(
                    "paper execution count {field} differs"
                )));
            }
        }
        for field in if method == "adaptive-logits" {
            &[
                "effective_strength",
                "top_probability",
                "second_probability",
                "margin",
            ][..]
        } else if method == "baseline" {
            &[][..]
        } else {
            &["effective_strength"][..]
        } {
            let actual_value = if *field == "effective_strength" {
                &candidate[*field]
            } else {
                &candidate["adaptive"][*field]
            };
            let reference_value = if *field == "effective_strength" {
                &oracle[*field]
            } else {
                &oracle["adaptive"][*field]
            };
            let actual = actual_value
                .as_f64()
                .ok_or_else(|| io::Error::other("candidate method scalar absent"))?;
            let reference = reference_value
                .as_f64()
                .ok_or_else(|| io::Error::other("reference method scalar absent"))?;
            let threshold = ABSOLUTE + RELATIVE * actual.abs().max(reference.abs());
            if !actual.is_finite() || !reference.is_finite() {
                return Err(io::Error::other("paper method scalar is nonfinite"));
            }
            if (actual - reference).abs() > threshold {
                scalar_exceeding += 1;
            }
            scalar_comparisons.emit(&json!({"field":field,"actual":actual,"reference":reference,"absolute_error":(actual-reference).abs(),"threshold":threshold,"exceeding":(actual-reference).abs()>threshold,"actual_source":{"path":candidate_path,"line":line},"reference_source":{"path":oracle_path,"line":oracle_line}}))?;
        }
        Ok(())
    })?;
    if selected_records != 1 {
        return Err(io::Error::other("matching paper candidate record absent"));
    }
    scalar_comparisons.flush()?;
    selections.flush()?;
    let mut summary = comparisons.finish()?;
    summary["method"] = json!(method);
    summary["scalar_comparisons"] = json!(scalar_comparisons.line);
    summary["scalar_exceeding"] = json!(scalar_exceeding);
    summary["selection_differences"] = json!(selection_differences);
    summary["selection_comparison_records"] = json!(selections.line);
    summary["selected_candidate_records"] = json!(selected_records);
    Ok(summary)
}

pub fn scope_controls(source: &Path, prefix: &[u32], directory: &Path) -> io::Result<Value> {
    let admitted = Archive::open(source, prefix)?;
    admitted.require_complete()?;
    let mut original = Vec::new();
    compare::visit_records(source, |_, record| {
        if record["context"]["arm"] == "baseline4"
            && ["layer_execution", "loop_boundary", "diagnostic_readouts"]
                .contains(&record["kind"].as_str().unwrap_or_default())
        {
            original.push(record);
        }
        Ok(())
    })?;
    let mut controls = JsonLines::create(&directory.join("scope-controls.jsonl"))?;
    for name in [
        "wrong_prefix",
        "wrong_checkpoint",
        "wrong_dtype",
        "missing_layer",
        "duplicate_layer",
    ] {
        let path = directory.join(format!("{name}.jsonl"));
        let mut records = JsonLines::create(&path)?;
        for (index, original_record) in original.iter().enumerate() {
            if name == "missing_layer" && index == 0 {
                continue;
            }
            let mut record = original_record.clone();
            match name {
                "wrong_prefix" => record["context"]["prefix_ids"] = json!([0, 1, 2]),
                "wrong_checkpoint" => {
                    record["context"]["checkpoint_observed_sha256"] =
                        json!("wrong-checkpoint-control")
                }
                "wrong_dtype" => {
                    record["context"]["numeric_policy"] =
                        json!("BF16 activations deliberately mislabeled control")
                }
                _ => {}
            }
            records.emit(&record)?;
            if name == "duplicate_layer" && index == 0 {
                records.emit(&record)?;
            }
        }
        records.flush()?;
        let failure = match Archive::open(&path, prefix) {
            Ok(_) => {
                return Err(io::Error::other(format!(
                    "scope control {name} unexpectedly accepted"
                )));
            }
            Err(error) => error.to_string(),
        };
        controls.emit(&json!({"kind":"scope_negative_control","name":name,"artifact":path,"rejected":true,"observed_error":failure}))?;
    }
    controls.flush()?;
    Ok(
        json!({"positive_archive_controls":1,"negative_scope_controls":5,"scope_controls_rejected":5,"unexpected_acceptances":0,"positive_boundary_vectors":admitted.vectors.len()}),
    )
}
