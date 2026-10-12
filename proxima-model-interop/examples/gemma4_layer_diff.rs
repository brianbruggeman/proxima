//! Per-layer comparison of proxima's one-shot Metal forward against a
//! `llama-eval-callback` dump of the same prompt on the same GGUF.
//!
//! `--list-nodes <file>` writes `id kind name description` for every node of
//! the loaded forward program and stops, so a caller can pick the nodes that
//! match llama's named tensors. Otherwise `--nodes <id:label,...>` evaluates
//! those nodes through [`LoadedModel::forward_node_values_on_backend`] and sets
//! each next to the llama tensor named by its label (`l_out-3`), using the
//! tensor's `sum =` line and its printed rows.
//!
//! `cargo run --release --example gemma4_layer_diff --features std,metal -- \
//!    --model <gguf> --prompt <text> --llama-dump <stdout of llama-eval-callback> \
//!    --nodes 123:l_out-0,456:l_out-1 --out <json>`

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;

use proxima_gguf::parse_complete;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel};
use proxima_tensor::op::NodeId;
use serde_json::{Value, json};

struct LlamaTensor {
    sum: f64,
    rows: Vec<Vec<f64>>,
}

fn parse_row(line: &str) -> Vec<f64> {
    line.trim()
        .trim_start_matches('[')
        .trim_end_matches(',')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|cell| cell.trim().parse::<f64>().ok())
        .collect()
}

fn parse_llama_dump(text: &str) -> BTreeMap<String, Vec<LlamaTensor>> {
    let mut tensors: BTreeMap<String, Vec<LlamaTensor>> = BTreeMap::new();
    let mut current: Option<(String, Vec<Vec<f64>>)> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("common_debug_cb_eval:") {
            let name = rest.split('=').next().unwrap_or("").trim().to_owned();
            current = Some((name, Vec::new()));
        } else if let Some((_, rows)) = current.as_mut() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("sum =") {
                let sum = trimmed
                    .trim_start_matches("sum =")
                    .trim()
                    .parse()
                    .unwrap_or(f64::NAN);
                let (name, rows) = current.take().expect("current tensor");
                tensors
                    .entry(name)
                    .or_default()
                    .push(LlamaTensor { sum, rows });
            } else if trimmed.starts_with('[') && trimmed.len() > 2 && trimmed.contains(',') {
                rows.push(parse_row(trimmed));
            }
        }
    }
    tensors
}

fn main() {
    let mut values = BTreeMap::new();
    let mut tokens = std::env::args().skip(1);
    while let Some(flag) = tokens.next() {
        values.insert(
            flag.clone(),
            tokens
                .next()
                .unwrap_or_else(|| panic!("{flag} needs a value")),
        );
    }
    let required = |flag: &str| {
        values
            .get(flag)
            .unwrap_or_else(|| panic!("missing {flag}"))
            .clone()
    };
    if let Some(other) = values.get("--other-dump") {
        compare_dumps(&required("--llama-dump"), other, &required("--out"));
        return;
    }
    let file = File::open(required("--model")).expect("open model");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap model");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse gguf header");
    let model = LoadedModel::load(&parsed, bytes).expect("load model");

    if let Some(path) = values.get("--list-nodes") {
        let mut out = File::create(path).expect("create listing");
        for index in 0..model.node_count() {
            let node = NodeId(u32::try_from(index).expect("node id fits u32"));
            let description: String = model.node_description(node).chars().take(300).collect();
            writeln!(
                out,
                "{index}\t{}\t{:?}\t{:?}\t{description}",
                model.node_kind(node),
                model.node_name(node),
                model
                    .node_dependencies(node)
                    .iter()
                    .map(|dep| dep.0)
                    .collect::<Vec<_>>(),
            )
            .expect("write listing");
        }
        println!("listed {} nodes", model.node_count());
        return;
    }

    let prompt = required("--prompt");
    let gpu_layers = if values.get("--backend").is_some_and(|name| name == "cpu") {
        0
    } else {
        GPU_LAYERS_ALL
    };
    let llama =
        parse_llama_dump(&std::fs::read_to_string(required("--llama-dump")).expect("read dump"));
    let mut nodes = layer_output_nodes(&model);
    if let Some(extra) = values.get("--nodes") {
        nodes.extend(extra.split(',').map(|pair| {
            let (id, label) = pair.split_once(':').expect("id:label");
            (id.parse().expect("node id"), label.to_owned())
        }));
    }
    let ids: Vec<NodeId> = nodes.iter().map(|(id, _)| NodeId(*id)).collect();
    let evaluated = model
        .forward_node_values_on_backend(&prompt, &ids, gpu_layers)
        .expect("forward_node_values_on_backend");
    let records: Vec<Value> = nodes
        .iter()
        .zip(&evaluated)
        .map(|((id, label), data)| {
            compare_node(
                *id,
                label,
                data,
                llama.get(label).and_then(|entries| entries.last()),
            )
        })
        .collect();
    let rendered = serde_json::to_string_pretty(&records).expect("render");
    std::fs::write(required("--out"), &rendered).expect("write out");
    records
        .iter()
        .for_each(|record| println!("{}", summary_line(record)));
}

fn summary_line(record: &Value) -> String {
    let maximum = |key: &str| {
        record[key]
            .as_array()
            .map(|cells| cells.iter().filter_map(Value::as_f64).fold(0.0, f64::max))
    };
    format!(
        "{:<24} node {:>6} sum_rel_diff {:>12.3e} max_row_corner_abs_diff {:>12.3e} max_row_llama_scale {:>10.3e}",
        record["label"].as_str().unwrap_or("?"),
        record["node"],
        record["sum_rel_diff"].as_f64().unwrap_or(f64::NAN),
        maximum("row_corner_max_abs_diff").unwrap_or(f64::NAN),
        maximum("row_corner_llama_scale").unwrap_or(f64::NAN),
    )
}

fn layer_output_nodes(model: &LoadedModel) -> Vec<(u32, String)> {
    (0..model.node_count())
        .filter_map(|index| {
            let node = NodeId(u32::try_from(index).expect("node id fits u32"));
            let name = model.node_name(node)?;
            let layer = name
                .strip_prefix("blk.")?
                .strip_suffix(".layer_output_scale.weight")?;
            Some((node.0 + 1, format!("l_out-{layer}")))
        })
        .collect()
}

fn compare_node(id: u32, label: &str, data: &[f32], llama: Option<&LlamaTensor>) -> Value {
    let rows = llama.map_or(1, |entry| entry.rows.len().max(1));
    let width = data.len() / rows;
    if width < 6 {
        return json!({
            "node": id,
            "label": label,
            "proxima_values": data,
            "llama_rows": llama.map(|entry| entry.rows.clone()),
        });
    }
    let proxima_sum: f64 = data.iter().map(|value| f64::from(*value)).sum();
    let last_row = &data[(rows - 1) * width..];
    let proxima_last = [
        last_row[0],
        last_row[1],
        last_row[2],
        last_row[width - 3],
        last_row[width - 2],
        last_row[width - 1],
    ];
    let llama_last = llama.and_then(|entry| entry.rows.last().cloned());
    let corner_max_abs_diff = llama_last.as_ref().map(|row| {
        row.iter()
            .zip(proxima_last)
            .map(|(expected, actual)| (expected - f64::from(actual)).abs())
            .fold(0.0, f64::max)
    });
    let per_row: Option<Vec<f64>> = llama.map(|entry| {
        entry
            .rows
            .iter()
            .enumerate()
            .map(|(row, expected)| {
                let base = row * width;
                let actual = [
                    data[base],
                    data[base + 1],
                    data[base + 2],
                    data[base + width - 3],
                    data[base + width - 2],
                    data[base + width - 1],
                ];
                expected
                    .iter()
                    .zip(actual)
                    .map(|(left, right)| (left - f64::from(right)).abs())
                    .fold(0.0, f64::max)
            })
            .collect()
    });
    let row_scale: Option<Vec<f64>> = llama.map(|entry| {
        entry
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .fold(0.0_f64, |scale, value| scale.max(value.abs()))
            })
            .collect()
    });
    json!({
        "node": id,
        "label": label,
        "row_corner_max_abs_diff": per_row,
        "row_corner_llama_scale": row_scale,
        "proxima_values_if_small": (data.len() <= 4096).then(|| data.to_vec()),
        "llama_rows_if_small": llama.filter(|_| data.len() <= 4096).map(|entry| entry.rows.clone()),
        "proxima_len": data.len(),
        "proxima_sum": proxima_sum,
        "llama_sum": llama.map(|entry| entry.sum),
        "sum_rel_diff": llama.map(|entry| (proxima_sum - entry.sum) / entry.sum.abs().max(1e-9)),
        "proxima_last_row_corners": proxima_last,
        "llama_last_row_corners": llama_last,
        "last_row_corner_max_abs_diff": corner_max_abs_diff,
    })
}

fn compare_dumps(first_path: &str, second_path: &str, out_path: &str) {
    let first = parse_llama_dump(&std::fs::read_to_string(first_path).expect("read first dump"));
    let second = parse_llama_dump(&std::fs::read_to_string(second_path).expect("read second dump"));
    let records: Vec<Value> = (0..30)
        .filter_map(|layer| {
            let label = format!("l_out-{layer}");
            let (left, right) = (first.get(&label)?.last()?, second.get(&label)?.last()?);
            let corner_max_abs_diff = left
                .rows
                .last()?
                .iter()
                .zip(right.rows.last()?)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0, f64::max);
            let per_row: Vec<f64> = left
                .rows
                .iter()
                .zip(&right.rows)
                .map(|(expected, actual)| {
                    expected
                        .iter()
                        .zip(actual)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max)
                })
                .collect();
            Some(json!({
                "label": label,
                "row_corner_max_abs_diff": per_row,
                "first_sum": left.sum,
                "second_sum": right.sum,
                "sum_rel_diff": (right.sum - left.sum) / left.sum.abs().max(1e-9),
                "last_row_corner_max_abs_diff": corner_max_abs_diff,
            }))
        })
        .collect();
    let rendered = serde_json::to_string_pretty(&records).expect("render");
    std::fs::write(out_path, &rendered).expect("write out");
    println!("{rendered}");
}
