//! Op-by-op comparison of one forward node's full value against the full
//! tensor `llama-eval-callback` wrote for the same prompt (the
//! `LLAMA_FULL_DUMP_PATH` record stream: name length, name, four i64 dims,
//! then every f32 with dimension 0 fastest).
//!
//! Each `--nodes id:label[#occurrence]` is evaluated through
//! [`LoadedModel::forward_node_values_on_backend`] on `--backend cpu|metal`.
//! llama tensors of rank three can sit in a different axis order than the
//! proxima node, so every axis order is scored and the best one is reported
//! next to the identity order.
//!
//! `--other-full-dump <file> --labels a,b` instead compares two llama dumps
//! (the engine-to-engine noise floor).
//!
//! `cargo run --release --example gemma4_op_diff --features std,metal -- \
//!    --model <gguf> --prompt <text> --backend cpu --llama-full-dump <bin> \
//!    --nodes 4:inp_scaled,70:attn_norm-0 --out <json>`

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fs::File;

use proxima_gguf::parse_complete;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel};
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::recorder::Recorder;
use proxima_tensor::op::NodeId;
use serde_json::{Value, json};

const AXIS_ORDERS: [[usize; 3]; 6] = [
    [2, 1, 0],
    [2, 0, 1],
    [1, 2, 0],
    [1, 0, 2],
    [0, 2, 1],
    [0, 1, 2],
];

#[derive(Clone)]
struct Dumped {
    dims: [usize; 4],
    values: Vec<f32>,
}

fn read_dump(path: &str) -> BTreeMap<String, Vec<Dumped>> {
    let bytes = std::fs::read(path).expect("read llama full dump");
    let mut cursor = 0;
    let mut tensors: BTreeMap<String, Vec<Dumped>> = BTreeMap::new();
    while cursor < bytes.len() {
        let name_len =
            u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().expect("name len")) as usize;
        cursor += 4;
        let name = String::from_utf8(bytes[cursor..cursor + name_len].to_vec()).expect("name");
        cursor += name_len;
        let mut dims = [0_usize; 4];
        for dim in &mut dims {
            let raw = i64::from_le_bytes(bytes[cursor..cursor + 8].try_into().expect("dim"));
            *dim = usize::try_from(raw).expect("non-negative dim");
            cursor += 8;
        }
        let count: usize = dims.iter().product();
        let values = bytes[cursor..cursor + count * 4]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32")))
            .collect();
        cursor += count * 4;
        tensors.entry(name).or_default().push(Dumped { dims, values });
    }
    tensors
}

impl Dumped {
    fn view(&self, spec: &str) -> Dumped {
        match spec.strip_prefix("dims=") {
            Some(extents) => self.reshaped(extents),
            None => self.half(spec),
        }
    }

    fn reshaped(&self, extents: &str) -> Dumped {
        let mut dims = [1_usize; 4];
        extents
            .split('x')
            .zip(dims.iter_mut())
            .for_each(|(extent, dim)| *dim = extent.parse().expect("extent"));
        assert_eq!(
            dims.iter().product::<usize>(),
            self.values.len(),
            "reshape must keep the element count"
        );
        Dumped {
            dims,
            values: self.values.clone(),
        }
    }

    fn half(&self, which: &str) -> Dumped {
        let width = self.dims[0] / 2;
        let offset = match which {
            "lo" => 0,
            "hi" => width,
            other => panic!("unknown half {other}"),
        };
        let values = self
            .values
            .chunks_exact(self.dims[0])
            .flat_map(|row| row[offset..offset + width].iter().copied())
            .collect();
        Dumped {
            dims: [width, self.dims[1], self.dims[2], self.dims[3]],
            values,
        }
    }
}

fn lookup(tensors: &BTreeMap<String, Vec<Dumped>>, label: &str) -> Dumped {
    let (spec, half) = label
        .split_once('@')
        .map_or((label, None), |(spec, half)| (spec, Some(half)));
    let (name, occurrence) = spec.split_once('#').map_or((spec, None), |(name, index)| {
        (name, Some(index.parse::<usize>().expect("occurrence index")))
    });
    let entries = tensors
        .get(name)
        .unwrap_or_else(|| panic!("llama dump has no tensor named {name}"));
    let entry = occurrence.map_or_else(
        || entries.last().expect("non-empty entries"),
        |index| &entries[index],
    );
    half.map_or_else(|| entry.clone(), |view| entry.view(view))
}

struct Scored {
    max_abs_diff: f64,
    squared_diff: f64,
    worst_index: usize,
}

fn score(reference: &Dumped, actual: &[f32], order: [usize; 3]) -> Scored {
    let [d0, d1, d2, _] = reference.dims;
    let extents = [d0, d1, d2];
    let mut scored = Scored {
        max_abs_diff: 0.0,
        squared_diff: 0.0,
        worst_index: 0,
    };
    for i2 in 0..d2 {
        for i1 in 0..d1 {
            for i0 in 0..d0 {
                let coordinates = [i0, i1, i2];
                let reference_index = (i2 * d1 + i1) * d0 + i0;
                let actual_index = order
                    .iter()
                    .fold(0, |index, axis| index * extents[*axis] + coordinates[*axis]);
                let diff = f64::from(reference.values[reference_index])
                    - f64::from(actual[actual_index]);
                scored.squared_diff += diff * diff;
                if diff.abs() > scored.max_abs_diff {
                    scored.max_abs_diff = diff.abs();
                    scored.worst_index = reference_index;
                }
            }
        }
    }
    scored
}

fn compare(label: &str, reference: &Dumped, actual: &[f32]) -> Value {
    let expected: usize = reference.dims.iter().product();
    if actual.len() != expected {
        return json!({ "label": label, "error": "length mismatch",
            "llama_len": expected, "proxima_len": actual.len(), "llama_dims": reference.dims });
    }
    let squared_reference: f64 = reference
        .values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum();
    let max_abs_reference = reference
        .values
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(f64::from(*value).abs()));
    let orders = if reference.dims[3] == 1 {
        AXIS_ORDERS.to_vec()
    } else {
        vec![AXIS_ORDERS[0]]
    };
    let scored: Vec<(usize, Scored)> = orders
        .iter()
        .enumerate()
        .map(|(position, order)| (position, score(reference, actual, *order)))
        .collect();
    let identity = &scored[0].1;
    let (best_position, best) = scored
        .iter()
        .min_by(|left, right| {
            left.1
                .max_abs_diff
                .total_cmp(&right.1.max_abs_diff)
                .then_with(|| left.0.cmp(&right.0))
        })
        .expect("at least one axis order");
    let width = reference.dims[0];
    json!({
        "label": label,
        "llama_dims": reference.dims,
        "max_abs_llama": max_abs_reference,
        "identity_max_abs_diff": identity.max_abs_diff,
        "identity_rel_l2": (identity.squared_diff / squared_reference.max(1e-30)).sqrt(),
        "best_axis_order_slow_to_fast": orders[*best_position],
        "best_max_abs_diff": best.max_abs_diff,
        "best_rel_l2": (best.squared_diff / squared_reference.max(1e-30)).sqrt(),
        "worst_row": best.worst_index / width,
        "worst_col": best.worst_index % width,
        "worst_llama": reference.values[best.worst_index],
    })
}

fn summary_line(record: &Value) -> String {
    if record.get("error").is_some() {
        return format!("{record}");
    }
    format!(
        "{:<26} max_abs_llama {:>10.3e} identity: max_abs {:>10.3e} rel_l2 {:>10.3e} | best {}: max_abs {:>10.3e} rel_l2 {:>10.3e} worst(row {}, col {}) llama {:>10.4}",
        record["label"].as_str().unwrap_or("?"),
        record["max_abs_llama"].as_f64().unwrap_or(f64::NAN),
        record["identity_max_abs_diff"].as_f64().unwrap_or(f64::NAN),
        record["identity_rel_l2"].as_f64().unwrap_or(f64::NAN),
        record["best_axis_order_slow_to_fast"],
        record["best_max_abs_diff"].as_f64().unwrap_or(f64::NAN),
        record["best_rel_l2"].as_f64().unwrap_or(f64::NAN),
        record["worst_row"],
        record["worst_col"],
        record["worst_llama"].as_f64().unwrap_or(f64::NAN),
    )
}

fn finish(records: &[Value], out_path: &str) {
    std::fs::write(
        out_path,
        serde_json::to_string_pretty(records).expect("render"),
    )
    .expect("write out");
    records
        .iter()
        .for_each(|record| println!("{}", summary_line(record)));
}

fn llama_against_llama(values: &BTreeMap<String, String>) {
    let first = read_dump(&values["--llama-full-dump"]);
    let second = read_dump(&values["--other-full-dump"]);
    let records: Vec<Value> = values["--labels"]
        .split(',')
        .map(|label| {
            let other = lookup(&second, label);
            compare(label, &lookup(&first, label), &other.values)
        })
        .collect();
    finish(&records, &values["--out"]);
}

fn print_raw(model: &LoadedModel, prompt: &str, raw_nodes: &str, gpu_layers: i32) {
    let ids: Vec<NodeId> = raw_nodes
        .split(',')
        .map(|id| NodeId(id.parse().expect("node id")))
        .collect();
    let evaluated = model
        .forward_node_values_on_backend(prompt, &ids, gpu_layers)
        .expect("forward_node_values_on_backend");
    for (id, data) in ids.iter().zip(&evaluated) {
        let sum: f64 = data.iter().map(|value| f64::from(*value)).sum();
        println!(
            "node {} len {} sum {:.6} head {:?} tail {:?} stride256 {:?}",
            id.0,
            data.len(),
            sum,
            &data[..data.len().min(6)],
            &data[data.len().saturating_sub(4)..],
            data.iter().step_by(256).take(24).collect::<Vec<_>>(),
        );
    }
}

fn print_hashes(model: &LoadedModel, prompt: &str, range: &str, gpu_layers: i32) {
    let (first, last) = range.split_once('-').expect("first-last");
    let ids: Vec<NodeId> = (first.parse::<u32>().expect("first")..=last.parse().expect("last"))
        .map(NodeId)
        .collect();
    let evaluated = model
        .forward_node_values_on_backend(prompt, &ids, gpu_layers)
        .expect("forward_node_values_on_backend");
    for (id, data) in ids.iter().zip(&evaluated) {
        let hash = data.iter().fold(0xcbf2_9ce4_8422_2325_u64, |state, value| {
            (state ^ u64::from(value.to_bits())).wrapping_mul(0x0100_0000_01b3)
        });
        let sum: f64 = data.iter().map(|value| f64::from(*value)).sum();
        println!("node {} len {} sum {sum:.9} fnv {hash:016x}", id.0, data.len());
    }
}

fn print_llama(path: &str, labels: &str) {
    let tensors = read_dump(path);
    for label in labels.split(',') {
        let entry = lookup(&tensors, label);
        let sum: f64 = entry.values.iter().map(|value| f64::from(*value)).sum();
        println!(
            "llama {label} dims {:?} sum {:.6} head {:?} tail {:?} stride256 {:?}",
            entry.dims,
            sum,
            &entry.values[..entry.values.len().min(6)],
            &entry.values[entry.values.len().saturating_sub(4)..],
            entry.values.iter().step_by(256).take(24).collect::<Vec<_>>(),
        );
    }
}

fn main() {
    let mut values = BTreeMap::new();
    let mut tokens = std::env::args().skip(1);
    while let Some(flag) = tokens.next() {
        let value = tokens
            .next()
            .unwrap_or_else(|| panic!("{flag} needs a value"));
        values.insert(flag, value);
    }
    let recorder = values.get("--telemetry-log").map(|path| {
        Recorder::builder()
            .export(Exporter::file(path).format(Formatter::Text))
            .expect("file exporter")
            .install()
            .expect("recorder")
    });
    run(&values);
    if let Some(recorder) = recorder {
        while recorder.drain() > 0 {}
    }
}

fn run(values: &BTreeMap<String, String>) {
    if values.contains_key("--other-full-dump") {
        return llama_against_llama(&values);
    }
    if let Some(labels) = values.get("--print-llama") {
        return print_llama(&values["--llama-full-dump"], labels);
    }
    let file = File::open(&values["--model"]).expect("open model");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap model");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse gguf header");
    let model = LoadedModel::load(&parsed, bytes).expect("load model");
    let gpu_layers = if values.get("--backend").is_some_and(|name| name == "cpu") {
        0
    } else {
        GPU_LAYERS_ALL
    };
    if let Some(described) = values.get("--describe") {
        for id in described.split(',') {
            let node = NodeId(id.parse().expect("node id"));
            println!(
                "node {id} {} deps {:?}: {}",
                model.node_kind(node),
                model
                    .node_dependencies(node)
                    .iter()
                    .map(|dep| dep.0)
                    .collect::<Vec<_>>(),
                model.node_description(node)
            );
        }
        return;
    }
    if let Some(range) = values.get("--hash-range") {
        return print_hashes(&model, &values["--prompt"], range, gpu_layers);
    }
    if let Some(raw_nodes) = values.get("--raw-nodes") {
        return print_raw(&model, &values["--prompt"], raw_nodes, gpu_layers);
    }
    let llama = read_dump(&values["--llama-full-dump"]);
    let nodes: Vec<(u32, &str)> = values["--nodes"]
        .split(',')
        .map(|pair| {
            let (id, label) = pair.split_once(':').expect("id:label");
            (id.parse().expect("node id"), label)
        })
        .collect();
    let ids: Vec<NodeId> = nodes.iter().map(|(id, _)| NodeId(*id)).collect();
    let evaluated = model
        .forward_node_values_on_backend(&values["--prompt"], &ids, gpu_layers)
        .expect("forward_node_values_on_backend");
    let records: Vec<Value> = nodes
        .iter()
        .zip(&evaluated)
        .map(|((_, label), data)| compare(label, &lookup(&llama, label), data))
        .collect();
    finish(&records, &values["--out"]);
}
