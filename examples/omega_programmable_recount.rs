use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde_json::{Value, json};

#[path = "omega_programmable_recount/cached.rs"]
mod cached;

fn rows(path: &Path) -> io::Result<Vec<(usize, Value)>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(line, text)| {
            Ok((
                line + 1,
                serde_json::from_str(&text?).map_err(io::Error::other)?,
            ))
        })
        .collect()
}

fn emit(writer: &mut impl Write, value: &Value) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, value).map_err(io::Error::other)?;
    writer.write_all(b"\n")
}

fn number(value: &Value) -> io::Result<f64> {
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| io::Error::other("coordinate or duration is not finite"))
}

fn differences(reference: &Value, actual: &Value) -> io::Result<Vec<Value>> {
    let reference = reference
        .as_array()
        .ok_or_else(|| io::Error::other("reference vector absent"))?;
    let actual = actual
        .as_array()
        .ok_or_else(|| io::Error::other("actual vector absent"))?;
    if reference.is_empty() || reference.len() != actual.len() {
        return Err(io::Error::other("coordinate count is empty or differs"));
    }
    let mut differences = Vec::new();
    for (coordinate, (reference, actual)) in reference.iter().zip(actual).enumerate() {
        let reference = number(reference)?;
        let actual = number(actual)?;
        if actual.to_bits() != reference.to_bits() {
            differences.push(json!({"coordinate":coordinate,"reference":reference,
                "actual":actual,"signed_delta":actual-reference}));
        }
    }
    Ok(differences)
}

fn statistics(values: &[f64]) -> io::Result<Value> {
    if values.len() != 5
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(io::Error::other(
            "each timing cell requires five positive finite durations",
        ));
    }
    // translation avoids large wall offsets without changing variance.
    let origin = values[0];
    let mut mean = 0.0;
    let mut second_moment = 0.0;
    for (position, value) in values.iter().enumerate() {
        let centered = value - origin;
        let difference = centered - mean;
        mean += difference / (position + 1) as f64;
        second_moment += difference * (centered - mean);
    }
    let deviation = (second_moment / (values.len() - 1) as f64).sqrt();
    let mean = origin + mean;
    Ok(
        json!({"raw_ms":values,"mean_ms":mean,"sample_sd_ms":deviation,
        "cov_percent":100.0*deviation/mean,"count":values.len(),
        "formula":"first-sample-centered online Welford sample variance with n-1 denominator"}),
    )
}

fn check_summary(values: &[f64], summary: &Value) -> io::Result<Value> {
    let independent = statistics(values)?;
    if summary["sample_count"] != values.len() || summary["raw_durations_ms"] != json!(values) {
        return Err(io::Error::other("summary count or raw durations differ"));
    }
    for (derived, reported) in [
        ("mean_ms", "mean_ms"),
        ("sample_sd_ms", "sample_standard_deviation_ms"),
        ("cov_percent", "cov_percent"),
    ] {
        let expected = number(&independent[derived])?;
        let actual = number(&summary[reported])?;
        // different summation orders round differently; this checks summary serialization.
        let rounding_bound = 128.0 * f64::EPSILON * expected.abs().max(1.0);
        if (actual - expected).abs() > rounding_bound {
            return Err(io::Error::other(format!(
                "summary {reported} differs: reported={actual:.17e} independent={expected:.17e} difference={:.17e} rounding_bound={rounding_bound:.17e} raw={values:?}",
                (actual - expected).abs()
            )));
        }
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    for (field, rank) in [
        ("p50_ms", 2),
        ("p90_ms", 4),
        ("p95_ms", 4),
        ("p99_ms", 4),
        ("p999_ms", 4),
    ] {
        if number(&summary[field])? != ordered[rank] {
            return Err(io::Error::other(format!("summary {field} differs")));
        }
    }
    Ok(independent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recount_float_roundtrip() {
        let values = [37978.240083000004_f64, 37947.256541999996_f64];
        let serialized = serde_json::to_string(&values).expect("serialize measured input bits");
        let recovered: Vec<f64> =
            serde_json::from_str(&serialized).expect("parse measured input bits");
        assert_eq!(recovered.len(), values.len());
        for (actual, expected) in recovered.iter().zip(values) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        println!("serialized={serialized} recovered={recovered:?}");
    }

    #[test]
    fn recount_centered_variance() {
        let values = [
            37990.627208,
            37978.240083000004,
            38008.974541,
            38012.673958,
            37947.256541999996,
        ];
        let summary = json!({"sample_count":5,"raw_durations_ms":values,
            "mean_ms":37987.5544664,"sample_standard_deviation_ms":26.50438946723304,
            "cov_percent":0.069771244397099,"p50_ms":37990.627208,
            "p90_ms":38012.673958,"p95_ms":38012.673958,"p99_ms":38012.673958,"p999_ms":38012.673958});
        let serialized = serde_json::to_string(&summary).expect("serialize measured summary");
        let parsed: Value = serde_json::from_str(&serialized).expect("parse measured summary");
        let independent = check_summary(&values, &parsed)
            .expect("measured summary within unchanged rounding bound");
        let known = statistics(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("known five-point variance");
        assert_eq!(known["mean_ms"], 3.0);
        assert_eq!(known["sample_sd_ms"], 2.5_f64.sqrt());
        let mut corrupted = parsed.clone();
        corrupted["sample_standard_deviation_ms"] = json!(27.50438946723304);
        assert!(check_summary(&values, &corrupted).is_err());
        println!("independent={independent} known={known} corrupted_rejected=true");
    }
}

fn recount(directory: &Path, writer: &mut impl Write) -> io::Result<Vec<(usize, Value)>> {
    let path = directory.join("samples.jsonl");
    let samples = rows(&path)?;
    let mut durations: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    let mut ids = BTreeMap::new();
    let mut warmups = 0;
    let mut warmup_cells: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (line, sample) in &samples {
        let identifier = sample["sample_id"]
            .as_str()
            .ok_or_else(|| io::Error::other("sample id absent"))?;
        if ids.insert(identifier, *line).is_some() {
            return Err(io::Error::other("duplicate sample id"));
        }
        if sample["kind"] == "warmup" {
            warmups += 1;
            let case = sample["case_id"]
                .as_str()
                .ok_or_else(|| io::Error::other("warmup case absent"))?;
            let method = sample["method"]
                .as_str()
                .ok_or_else(|| io::Error::other("warmup method absent"))?;
            *warmup_cells
                .entry((case.to_owned(), method.to_owned()))
                .or_default() += 1;
        } else if sample["kind"] == "sample" {
            let case = sample["case_id"]
                .as_str()
                .ok_or_else(|| io::Error::other("case id absent"))?;
            let method = sample["method"]
                .as_str()
                .ok_or_else(|| io::Error::other("method absent"))?;
            durations
                .entry((case.to_owned(), method.to_owned()))
                .or_default()
                .push(number(&sample["wall_ms"])?);
        } else {
            return Err(io::Error::other("unexpected sample kind"));
        }
    }
    if samples.len() != 48 || warmups != 8 || durations.len() != 8 {
        return Err(io::Error::other(
            "cohort requires 48 rows, eight warmups and eight timing cells",
        ));
    }
    if warmup_cells.len() != 8
        || warmup_cells.values().any(|count| *count != 1)
        || warmup_cells.keys().ne(durations.keys())
    {
        return Err(io::Error::other(
            "each timing cell requires exactly one matching warmup",
        ));
    }
    let summary_path = directory.join("summary.json");
    let summary: Value =
        serde_json::from_reader(File::open(&summary_path)?).map_err(io::Error::other)?;
    let summary = summary
        .as_array()
        .filter(|rows| rows.len() == 8)
        .ok_or_else(|| io::Error::other("summary requires eight cells"))?;
    let mut seen_summaries = BTreeSet::new();
    for ((case, method), values) in durations {
        let matching: Vec<_> = summary
            .iter()
            .enumerate()
            .filter(|(_, row)| row["case_id"] == case && row["method"] == method)
            .collect();
        let [(summary_index, reported)] = matching.as_slice() else {
            return Err(io::Error::other("summary cell absent or duplicated"));
        };
        seen_summaries.insert(*summary_index);
        let measurement = check_summary(&values, &reported["cost"])?;
        emit(
            writer,
            &json!({"kind":"independent_timing_recount","source":path,
            "case_id":case,"method":method,"provenance":"derived from raw durations",
            "summary_source":summary_path,"summary_index":summary_index,
            "measurement":measurement}),
        )?;
    }
    if seen_summaries.len() != 8 {
        return Err(io::Error::other("summary cells were not fully reconciled"));
    }
    Ok(samples)
}

fn main() -> io::Result<()> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.first().is_some_and(|value| value == "--cached") {
        return cached::run(&arguments[1..], false);
    }
    if arguments
        .first()
        .is_some_and(|value| value == "--cached-matched-batch")
    {
        return cached::run(&arguments[1..], true);
    }
    let [reference, actual, oracle, output] = arguments.as_slice() else {
        return Err(io::Error::other(
            "usage: omega_programmable_recount CONTROL_DIR ENABLED_DIR LLAMA_JSONL OUTPUT_DIR",
        ));
    };
    let reference = Path::new(reference);
    let actual = Path::new(actual);
    let oracle = Path::new(oracle);
    let output = Path::new(output);
    fs::create_dir_all(output)?;
    let mut timings = BufWriter::new(File::create(output.join("timing-recount.jsonl"))?);
    let references = recount(reference, &mut timings)?;
    let actuals = recount(actual, &mut timings)?;
    let reference_map: BTreeMap<_, _> = references
        .iter()
        .map(|(line, value)| {
            Ok((
                value["sample_id"]
                    .as_str()
                    .ok_or_else(|| io::Error::other("reference id absent"))?,
                (*line, value),
            ))
        })
        .collect::<io::Result<_>>()?;
    let mut comparison = BufWriter::new(File::create(output.join("coordinate-comparisons.jsonl"))?);
    compare_micro(
        &reference
            .parent()
            .ok_or_else(|| io::Error::other("control parent absent"))?
            .join("micro-disabled"),
        &actual
            .parent()
            .ok_or_else(|| io::Error::other("enabled parent absent"))?
            .join("micro-enabled"),
        &mut timings,
        &mut comparison,
    )?;
    let positive = differences(&json!([1.0, 2.0]), &json!([1.0, 2.0]))?;
    let negative = differences(&json!([1.0, 2.0]), &json!([1.0, 3.0]))?;
    if !positive.is_empty() || negative.len() != 1 || negative[0]["coordinate"] != 1 {
        return Err(io::Error::other("coordinate comparator controls disagree"));
    }
    emit(
        &mut comparison,
        &json!({"kind":"comparator_controls","positive":positive,"negative":negative}),
    )?;
    for (actual_line, candidate) in &actuals {
        let identifier = candidate["sample_id"]
            .as_str()
            .ok_or_else(|| io::Error::other("candidate id absent"))?;
        let (reference_line, baseline) = reference_map
            .get(identifier)
            .ok_or_else(|| io::Error::other("reference sample absent"))?;
        for field in ["prefix_ids", "checkpoint_observed_sha256", "omega_config"] {
            if baseline["context"][field].is_null()
                || baseline["context"][field] != candidate["context"][field]
            {
                return Err(io::Error::other(format!("paired context differs: {field}")));
            }
        }
        for field in [
            "depth",
            "arm",
            "readout",
            "build_tiled_min_tokens",
            "retained_layer_statistic_nodes",
            "retained_loop_norm_statistic_nodes",
        ] {
            if baseline["context"][field].is_null()
                || baseline["context"][field] != candidate["context"][field]
            {
                return Err(io::Error::other(format!("paired work differs: {field}")));
            }
        }
        if baseline["context"]["build_dense_float_storage"] != false
            || candidate["context"]["build_dense_float_storage"] != true
        {
            return Err(io::Error::other(
                "paired build arms must be disabled then enabled",
            ));
        }
        let fields = baseline["payload"]
            .as_object()
            .ok_or_else(|| io::Error::other("reference payload absent"))?;
        if candidate["payload"].as_object().map(|value| value.len()) != Some(fields.len()) {
            return Err(io::Error::other("payload field counts differ"));
        }
        for (field, baseline_values) in fields {
            if candidate["payload"].get(field).is_none() {
                return Err(io::Error::other("candidate payload field absent"));
            }
            let candidate_values = &candidate["payload"][field];
            if baseline_values.is_null() && candidate_values.is_null() {
                continue;
            }
            let deltas = differences(baseline_values, candidate_values)?;
            let positive = deltas
                .iter()
                .filter(|row| {
                    row["signed_delta"]
                        .as_f64()
                        .is_some_and(|value| value > 0.0)
                })
                .count();
            let negative = deltas
                .iter()
                .filter(|row| {
                    row["signed_delta"]
                        .as_f64()
                        .is_some_and(|value| value < 0.0)
                })
                .count();
            emit(
                &mut comparison,
                &json!({"kind":"paired_payload","sample_id":identifier,"field":field,
                "reference_source":{"file":reference.join("samples.jsonl"),"line":reference_line},
                "actual_source":{"file":actual.join("samples.jsonl"),"line":actual_line},
                "coordinates":baseline_values.as_array().map(Vec::len),"positive_deltas":positive,
                "negative_deltas":negative,"all_differences":deltas,
                "reference_selected_id":baseline["selected_id"],"actual_selected_id":candidate["selected_id"],
                "argmax_changed":baseline["selected_id"]!=candidate["selected_id"]}),
            )?;
        }
    }
    let oracle_rows = rows(oracle)?;
    let context = &oracle_rows
        .first()
        .ok_or_else(|| io::Error::other("oracle context absent"))?
        .1;
    for (field, expected) in [
        ("llama_revision", "1550360e089185632c2036c908411b05fafa1119"),
        (
            "checkpoint_revision",
            "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
        ),
        (
            "checkpoint_sha256",
            "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af",
        ),
        (
            "model_sha256",
            "d3cdf276005d768c5def9cfb6679eb5600a424c5cd23b45157e65204708ae478",
        ),
    ] {
        if context[field] != expected {
            return Err(io::Error::other(format!(
                "pinned oracle identity differs: {field}"
            )));
        }
    }
    let mut oracle_durations: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut oracle_sources: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (line, row) in &oracle_rows {
        if row["kind"] != "sample" {
            continue;
        }
        let case = row["case_id"]
            .as_str()
            .ok_or_else(|| io::Error::other("oracle case absent"))?;
        let proxima = actuals
            .iter()
            .find(|(_, candidate)| candidate["case_id"] == case)
            .ok_or_else(|| io::Error::other("oracle case differs from paired cohort"))?;
        if row["prefix_ids"] != proxima.1["context"]["prefix_ids"]
            || row["depth"] != proxima.1["context"]["depth"]
        {
            return Err(io::Error::other("oracle prefix or depth differs"));
        }
        oracle_durations
            .entry(case.to_owned())
            .or_default()
            .push(number(&row["duration_ns"])? / 1_000_000.0);
        oracle_sources
            .entry(case.to_owned())
            .or_default()
            .push(*line);
    }
    if oracle_durations.len() != 2 {
        return Err(io::Error::other("oracle requires two timing cases"));
    }
    for (case, durations) in oracle_durations {
        emit(
            &mut timings,
            &json!({"kind":"llama_timing_recount","source":oracle,
            "source_lines":oracle_sources.get(&case),"case_id":case,"method":"baseline4",
            "measurement":statistics(&durations)?,"comparison_scope":"performance only; arithmetic and final-layer work differ"}),
        )?;
    }
    emit(
        &mut timings,
        &json!({"kind":"llama_oracle_register","source":oracle,
        "records":oracle_rows.len(),"context":oracle_rows.first().map(|(_,value)|value),
        "comparison_scope":"performance oracle only; arithmetic and final-layer work differ"}),
    )?;
    timings.flush()?;
    comparison.flush()
}

fn compare_micro(
    reference: &Path,
    actual: &Path,
    timings: &mut impl Write,
    output: &mut impl Write,
) -> io::Result<()> {
    let reference_rows = rows(&reference.join("microbench.jsonl"))?;
    let actual_rows = rows(&actual.join("microbench.jsonl"))?;
    for (_, setup) in actual_rows
        .iter()
        .filter(|(_, value)| value["kind"] == "projection_setup")
    {
        let (_, baseline) = reference_rows
            .iter()
            .find(|(_, value)| {
                value["kind"] == "projection_setup" && value["setup_id"] == setup["setup_id"]
            })
            .ok_or_else(|| io::Error::other("paired projection setup absent"))?;
        for field in [
            "rows",
            "columns",
            "reduction",
            "prefix_ids",
            "weight_name",
            "checkpoint_sha256",
            "weight_absolute_byte_range",
            "weight_codec",
            "activation_codec",
            "requested_output_nodes",
            "build_tiled_min_tokens",
            "config",
            "activations",
            "independent_f64_output",
            "absolute_error_bounds",
        ] {
            if setup[field].is_null() || setup[field] != baseline[field] {
                return Err(io::Error::other(format!(
                    "paired projection setup differs: {field}"
                )));
            }
        }
        if baseline["build_dense_float_storage"] != false
            || setup["build_dense_float_storage"] != true
        {
            return Err(io::Error::other(
                "projection build arms must be disabled then enabled",
            ));
        }
    }
    for (directory, records) in [(reference, &reference_rows), (actual, &actual_rows)] {
        let setup_ids = records
            .iter()
            .filter(|(_, value)| value["kind"] == "projection_setup")
            .map(|(_, value)| {
                value["setup_id"]
                    .as_str()
                    .ok_or_else(|| io::Error::other("projection setup id absent"))
            })
            .collect::<io::Result<BTreeSet<_>>>()?;
        if setup_ids.len() != 3 {
            return Err(io::Error::other("projection setup ids must be unique"));
        }
        let sample_ids = records
            .iter()
            .filter(|(_, value)| {
                matches!(
                    value["kind"].as_str(),
                    Some("projection_sample" | "projection_warmup")
                )
            })
            .map(|(_, value)| {
                value["sample_id"]
                    .as_str()
                    .ok_or_else(|| io::Error::other("projection sample id absent"))
            })
            .collect::<io::Result<BTreeSet<_>>>()?;
        if sample_ids.len() != 18 {
            return Err(io::Error::other("projection sample ids must be unique"));
        }
        for (kind, expected) in [
            ("projection_setup", 3),
            ("projection_comparator_control", 3),
            ("projection_warmup", 3),
            ("projection_sample", 15),
            ("projection_summary", 3),
        ] {
            if records
                .iter()
                .filter(|(_, value)| value["kind"] == kind)
                .count()
                != expected
            {
                return Err(io::Error::other(format!("unexpected {kind} count")));
            }
        }
        let mut durations: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
        for (line, record) in records {
            if record["kind"] != "projection_sample" {
                continue;
            }
            let extent = record["rows"]
                .as_u64()
                .ok_or_else(|| io::Error::other("projection rows absent"))?;
            durations
                .entry(extent)
                .or_default()
                .push(number(&record["wall_ms"])?);
            if record["error_bound_violation_coordinates"] != json!([]) {
                return Err(io::Error::other(format!(
                    "projection error envelope failed at {}:{line}",
                    directory.display()
                )));
            }
        }
        if durations.keys().copied().collect::<Vec<_>>() != [1, 5, 24] {
            return Err(io::Error::other("projection row extents differ"));
        }
        for (extent, values) in durations {
            let matching: Vec<_> = records
                .iter()
                .filter(|(_, row)| row["kind"] == "projection_summary" && row["rows"] == extent)
                .collect();
            let [(_, summary)] = matching.as_slice() else {
                return Err(io::Error::other("projection summary absent or duplicated"));
            };
            let measurement = check_summary(&values, &summary["cost"])?;
            emit(
                timings,
                &json!({"kind":"micro_timing_recount","source":directory.join("microbench.jsonl"),
            "rows":extent,"measurement":measurement}),
            )?;
        }
    }
    let mut comparisons = 0;
    for (line, candidate) in &actual_rows {
        if !matches!(
            candidate["kind"].as_str(),
            Some("projection_sample" | "projection_warmup")
        ) {
            continue;
        }
        let (reference_line, baseline) = reference_rows
            .iter()
            .find(|(_, value)| value["sample_id"] == candidate["sample_id"])
            .ok_or_else(|| io::Error::other("paired projection sample absent"))?;
        let deltas = differences(&baseline["actual"], &candidate["actual"])?;
        emit(
            output,
            &json!({"kind":"paired_projection","sample_id":candidate["sample_id"],
            "reference_source":{"file":reference.join("microbench.jsonl"),"line":reference_line},
            "actual_source":{"file":actual.join("microbench.jsonl"),"line":line},
            "coordinates":candidate["actual"].as_array().map(Vec::len),"all_differences":deltas}),
        )?;
        comparisons += 1;
    }
    if comparisons != 18 {
        return Err(io::Error::other(
            "projection requires eighteen paired vectors",
        ));
    }
    Ok(())
}
