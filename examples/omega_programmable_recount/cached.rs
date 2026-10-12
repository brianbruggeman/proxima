use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde_json::{Value, json};

use super::{check_summary, differences, emit, number, rows, statistics};

fn required_array<'value>(row: &'value Value, field: &str) -> io::Result<&'value Vec<Value>> {
    row[field]
        .as_array()
        .ok_or_else(|| io::Error::other(format!("missing array {field}")))
}

fn recount(path: &Path, writer: &mut impl Write) -> io::Result<Vec<(usize, Value)>> {
    let records = rows(path)?;
    let frozen: Value = serde_json::from_str(include_str!(
        "../../proxima-tensor/specs/omega-programmable/frozen-prefixes.json"
    ))
    .map_err(io::Error::other)?;
    let short_prefix = required_array(&frozen["cases"][1], "prefix_ids")?;
    if short_prefix.len() != 24 {
        return Err(io::Error::other("frozen short prefix must contain 24 IDs"));
    }
    let long_prefix: Vec<_> = short_prefix.iter().cloned().cycle().take(256).collect();
    let mut cases: BTreeMap<String, Vec<(usize, &Value)>> = BTreeMap::new();
    let mut warmups = BTreeSet::new();
    let mut diagnostics = BTreeSet::new();
    let mut summaries = BTreeMap::new();
    let mut completions = 0;
    for (line, row) in &records {
        let kind = row["kind"].as_str().unwrap_or_default();
        if kind == "cached_complete" || kind == "run_complete" {
            if row["request_count"] != 12 {
                return Err(io::Error::other("completion request count differs"));
            }
            completions += 1;
            continue;
        }
        if matches!(kind, "cached_setup" | "run_context" | "setup_timing") {
            continue;
        }
        let case = row["case_id"]
            .as_str()
            .ok_or_else(|| io::Error::other("cached record case absent"))?;
        let expected_prefix = match case {
            "ball24" => short_prefix,
            "repeated-ball256" => &long_prefix,
            _ => return Err(io::Error::other("cached case is not in the frozen panel")),
        };
        if matches!(
            kind,
            "cached_diagnostic" | "cached_sample" | "cached_warmup"
        ) && required_array(row, "prefix_ids")? != expected_prefix
        {
            return Err(io::Error::other(
                "cached diagnostic/request prefix differs from its frozen case",
            ));
        }
        match kind {
            "cached_diagnostic" => {
                if !diagnostics.insert(case.to_owned()) {
                    return Err(io::Error::other("duplicate cached diagnostic"));
                }
            }
            "cached_summary" => {
                if summaries.insert(case.to_owned(), (*line, row)).is_some() {
                    return Err(io::Error::other("duplicate cached summary"));
                }
            }
            "cached_sample" | "cached_warmup" => {
                let ids = required_array(row, "generated_ids")?;
                let latencies = required_array(row, "token_latencies_ms")?;
                let prefix = required_array(row, "prefix_ids")?;
                if ids.len() != latencies.len()
                    || row["generated_count"] != ids.len()
                    || row["max_new_tokens"] != 64
                    || ids.len() > 64
                    || !matches!(prefix.len(), 24 | 256)
                    || row["error_count"] != 0
                    || row["depth"] != 4
                    || row["type_k"] != "F32"
                    || row["type_v"] != "F32"
                    || ids
                        .iter()
                        .chain(prefix)
                        .any(|token| token.as_u64().is_none_or(|token| token >= 49152))
                    || latencies
                        .iter()
                        .any(|value| number(value).map_or(true, |value| value <= 0.0))
                    || number(&row["wall_ms"])? <= 0.0
                {
                    return Err(io::Error::other(format!(
                        "invalid request payload at {}:{line}",
                        path.display()
                    )));
                }
                if row["runtime"] == "proxima" {
                    let emitted: Vec<_> = required_array(row, "events")?
                        .iter()
                        .filter(|event| event["phase"] == "Token")
                        .map(|event| event["id"].clone())
                        .collect();
                    if &emitted != ids {
                        return Err(io::Error::other(
                            "production callback payload differs from returned IDs",
                        ));
                    }
                }
                if kind == "cached_warmup" {
                    if !warmups.insert(case.to_owned()) {
                        return Err(io::Error::other("duplicate cached warmup"));
                    }
                } else {
                    cases.entry(case.to_owned()).or_default().push((*line, row));
                }
                emit(
                    writer,
                    &json!({"kind":"request_payload_recount","source":{"path":path,"line":line},
                    "case_id":case,"request_kind":kind,"repetition":row["repetition"],"prefix_ids":prefix,
                    "generated_ids":ids,"generated_count":ids.len(),"stopped_by_eos":row["stopped_by_eos"],
                    "wall_ms":row["wall_ms"],"first_token_ms":row["first_token_ms"],
                    "token_latencies_ms":latencies,"response_tokens_per_second":ids.len() as f64*1000.0/number(&row["wall_ms"])?}),
                )?;
            }
            _ => return Err(io::Error::other(format!("unexpected cached record {kind}"))),
        }
    }
    let case_names: BTreeSet<_> = cases.keys().cloned().collect();
    if cases.len() != 2
        || warmups != case_names
        || diagnostics != case_names
        || summaries.keys().cloned().collect::<BTreeSet<_>>() != case_names
        || completions != 1
    {
        return Err(io::Error::other(
            "cached cohort requires two cases, one warmup/diagnostic/summary per case and one completion",
        ));
    }
    for (case, samples) in cases {
        let repetitions: BTreeSet<_> = samples
            .iter()
            .map(|(_, row)| row["repetition"].as_u64())
            .collect();
        if samples.len() != 5 || repetitions != (0..5).map(Some).collect() {
            return Err(io::Error::other(
                "cached case requires five unique repetitions 0 through 4",
            ));
        }
        let first = samples[0].1;
        if samples.iter().any(|(_, row)| {
            row["prefix_ids"] != first["prefix_ids"] || row["runtime"] != first["runtime"]
        }) {
            return Err(io::Error::other(
                "cached timing cell input or runtime differs",
            ));
        }
        let walls: Vec<_> = samples
            .iter()
            .map(|(_, row)| number(&row["wall_ms"]))
            .collect::<io::Result<_>>()?;
        let first_tokens: Vec<_> = samples
            .iter()
            .map(|(_, row)| number(&row["first_token_ms"]))
            .collect::<io::Result<_>>()?;
        let (summary_line, summary) = summaries[&case];
        let wall = if first["runtime"] == "proxima" {
            check_summary(&walls, &summary["wall"])?
        } else {
            let derived = statistics(&walls)?;
            for field in ["mean_ms", "cov_percent"] {
                let expected = number(&derived[field])?;
                if (number(&summary["wall"][field])? - expected).abs()
                    > 128.0 * f64::EPSILON * expected.abs().max(1.0)
                {
                    return Err(io::Error::other(
                        "native oracle summary differs from independent recount",
                    ));
                }
            }
            derived
        };
        emit(
            writer,
            &json!({"kind":"cached_timing_recount","source":path,"summary_line":summary_line,
            "case_id":case,"runtime":first["runtime"],"condition":"five measured requests; one separate warmup; empty KV per request",
            "status":"derived from complete request payloads","sample_lines":samples.iter().map(|(line,_)|*line).collect::<Vec<_>>(),
            "wall":wall,"first_token":statistics(&first_tokens)?,
            "generated_counts":samples.iter().map(|(_,row)|row["generated_count"].clone()).collect::<Vec<_>>()}),
        )?;
    }
    Ok(records)
}

fn diagnostic_vector(row: &Value) -> io::Result<&Value> {
    row["logits"]
        .get(0)
        .or_else(|| row["logits_by_step"].get(0))
        .ok_or_else(|| io::Error::other("complete initial diagnostic vector absent"))
}

fn identity(records: &[(usize, Value)]) -> io::Result<(&Value, &Value)> {
    let setups: Vec<_> = records
        .iter()
        .filter(|(_, row)| row["kind"] == "cached_setup" || row["kind"] == "run_context")
        .collect();
    let [(_, setup)] = setups.as_slice() else {
        return Err(io::Error::other(
            "cached model provenance absent or duplicate",
        ));
    };
    let (model, checkpoint) = if setup["kind"] == "cached_setup" {
        (
            &setup["observed_gguf_sha256"],
            &setup["checkpoint_source_sha256"],
        )
    } else {
        (&setup["model_sha256"], &setup["checkpoint_sha256"])
    };
    if model.as_str().is_none_or(|value| value.len() != 64)
        || checkpoint.as_str().is_none_or(|value| value.len() != 64)
    {
        return Err(io::Error::other("cached SHA256 identity malformed"));
    }
    Ok((model, checkpoint))
}

fn configured_pair(reference: &[(usize, Value)], actual: &[(usize, Value)]) -> io::Result<Value> {
    let reference_setups: Vec<_> = reference
        .iter()
        .filter(|(_, row)| row["kind"] == "cached_setup")
        .collect();
    let actual_setups: Vec<_> = actual
        .iter()
        .filter(|(_, row)| row["kind"] == "cached_setup")
        .collect();
    let ([(reference_line, reference_setup)], [(actual_line, actual_setup)]) =
        (reference_setups.as_slice(), actual_setups.as_slice())
    else {
        return Err(io::Error::other(
            "configured pair requires exactly one production setup each",
        ));
    };
    if reference_setup["build_dense_float_storage"] != false
        || actual_setup["build_dense_float_storage"] != true
        || reference_setup["build_tiled_min_tokens"] != 160
        || actual_setup["build_tiled_min_tokens"] != 160
        || reference_setup["omega_config"].is_null()
        || reference_setup["omega_config"] != actual_setup["omega_config"]
        || reference_setup["settings"] != actual_setup["settings"]
        || reference_setup["omega_config"]["engine"] != "gpu"
        || reference_setup["omega_config"]["gpu_driver"] != "metal"
        || reference_setup["omega_config"]["selection"]["mode"] != "configured"
    {
        return Err(io::Error::other(
            "configured pair differs in gate, threshold, runtime or serving policy",
        ));
    }
    Ok(
        json!({"control_setup_line":reference_line,"enabled_setup_line":actual_line,
        "control_gate":reference_setup["build_dense_float_storage"],
        "enabled_gate":actual_setup["build_dense_float_storage"],
        "control_threshold":reference_setup["build_tiled_min_tokens"],
        "enabled_threshold":actual_setup["build_tiled_min_tokens"],
        "matched_omega_config":reference_setup["omega_config"],
        "matched_serving_settings":reference_setup["settings"]}),
    )
}

fn compare(
    reference: &[(usize, Value)],
    actual: &[(usize, Value)],
    reference_path: &Path,
    actual_path: &Path,
    writer: &mut impl Write,
) -> io::Result<()> {
    if identity(reference)? != identity(actual)? {
        return Err(io::Error::other(
            "paired cached model/checkpoint digests differ",
        ));
    }
    for (actual_line, actual_row) in actual.iter().filter(|(_, row)| {
        row["kind"] == "cached_diagnostic"
            || row["kind"] == "cached_sample"
            || row["kind"] == "cached_warmup"
    }) {
        let matches: Vec<_> = reference
            .iter()
            .filter(|(_, row)| {
                row["kind"] == actual_row["kind"]
                    && row["case_id"] == actual_row["case_id"]
                    && (row["kind"] == "cached_diagnostic"
                        || row["repetition"] == actual_row["repetition"])
            })
            .collect();
        let [(reference_line, reference_row)] = matches.as_slice() else {
            return Err(io::Error::other("paired cached record absent or duplicate"));
        };
        if reference_row["prefix_ids"] != actual_row["prefix_ids"] {
            return Err(io::Error::other("paired cached input IDs differ"));
        }
        let source = json!({"reference":{"path":reference_path,"line":reference_line},"actual":{"path":actual_path,"line":actual_line}});
        if actual_row["kind"] == "cached_diagnostic" {
            let reference_vector = diagnostic_vector(reference_row)?;
            let actual_vector = diagnostic_vector(actual_row)?;
            if reference_vector.as_array().map(Vec::len) != Some(49152)
                || actual_vector.as_array().map(Vec::len) != Some(49152)
            {
                return Err(io::Error::other("diagnostic vocabulary width differs"));
            }
            let changes = differences(reference_vector, actual_vector)?;
            emit(
                writer,
                &json!({"kind":"cached_diagnostic_coordinates","source":source,"case_id":actual_row["case_id"],
                "compared_coordinates":49152,"difference_count":changes.len(),"signed_differences":changes,
                "scope":"initial prefix logits only; subsequent tokens may diverge"}),
            )?;
        } else {
            if reference_row["max_new_tokens"] != actual_row["max_new_tokens"] {
                return Err(io::Error::other("paired cached token budget differs"));
            }
            emit(
                writer,
                &json!({"kind":"cached_trajectory_pair","source":source,"case_id":actual_row["case_id"],
                "request_kind":actual_row["kind"],"repetition":actual_row["repetition"],
                "reference_ids":reference_row["generated_ids"],"actual_ids":actual_row["generated_ids"],
                "ids_equal":reference_row["generated_ids"]==actual_row["generated_ids"],
                "reference_stopped_by_eos":reference_row["stopped_by_eos"],"actual_stopped_by_eos":actual_row["stopped_by_eos"],
                "reference_wall_ms":reference_row["wall_ms"],"actual_wall_ms":actual_row["wall_ms"]}),
            )?;
        }
    }
    Ok(())
}

fn matched_batches(records: &[(usize, Value)], production: bool) -> io::Result<()> {
    let kind = if production {
        "cached_setup"
    } else {
        "run_context"
    };
    let setups: Vec<_> = records
        .iter()
        .filter(|(_, row)| row["kind"] == kind)
        .collect();
    let [(_, setup)] = setups.as_slice() else {
        return Err(io::Error::other("matched batch setup absent or duplicate"));
    };
    if setup["batch_size"] != 256
        || setup["ubatch_size"] != 256
        || setup["context_size"] != 320
        || (production && setup["gpu_profile"] != true)
    {
        return Err(io::Error::other(
            "matched batch or instrumentation condition differs",
        ));
    }
    if production {
        for (_, row) in records
            .iter()
            .filter(|(_, row)| row["kind"] == "cached_sample" || row["kind"] == "cached_warmup")
        {
            if row["gpu_profile"]["dropped_before"] != 0
                || row["gpu_profile"]["dropped_after"] != 0
                || row["gpu_profile"]["outside_timer"] != true
            {
                return Err(io::Error::other(
                    "cached profile request lost records or included drains in timer",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn run(arguments: &[OsString], matched_batch: bool) -> io::Result<()> {
    let [reference_path, actual_path, oracle_path, output] = arguments else {
        return Err(io::Error::other(
            "usage: --cached CONTROL_JSONL ENABLED_JSONL LLAMA_JSONL FRESH_OUTPUT_DIR",
        ));
    };
    let reference_path = Path::new(reference_path);
    let actual_path = Path::new(actual_path);
    let oracle_path = Path::new(oracle_path);
    let output = Path::new(output);
    fs::create_dir(output)?;
    let mut timings = BufWriter::new(File::create(output.join("timing-recount.jsonl"))?);
    let reference = recount(reference_path, &mut timings)?;
    let actual = recount(actual_path, &mut timings)?;
    let oracle = recount(oracle_path, &mut timings)?;
    if matched_batch {
        matched_batches(&reference, true)?;
        matched_batches(&actual, true)?;
        matched_batches(&oracle, false)?;
    }
    timings.flush()?;
    let configured_setup = configured_pair(&reference, &actual)?;
    let duplicated_off_rejected = configured_pair(&reference, &reference).is_err();
    let mut coordinates =
        BufWriter::new(File::create(output.join("coordinate-comparisons.jsonl"))?);
    let positive = differences(&json!([1.0, 2.0]), &json!([1.0, 2.0]))?;
    let negative = differences(&json!([1.0, 2.0]), &json!([1.0, 3.0]))?;
    if !positive.is_empty()
        || negative.len() != 1
        || negative[0]["coordinate"] != 1
        || !duplicated_off_rejected
    {
        return Err(io::Error::other(
            "cached coordinate comparator controls disagree",
        ));
    }
    emit(
        &mut coordinates,
        &json!({"kind":"comparator_control","positive":positive,"negative":negative,"duplicated_off_arm_rejected":duplicated_off_rejected,
            "matched_batch_required":matched_batch,"configured_setup":configured_setup,"control_source":reference_path,"enabled_source":actual_path}),
    )?;
    compare(
        &reference,
        &actual,
        reference_path,
        actual_path,
        &mut coordinates,
    )?;
    compare(
        &reference,
        &oracle,
        reference_path,
        oracle_path,
        &mut coordinates,
    )?;
    coordinates.flush()
}
