//! Rewrites the synthetic granite `affine4` reference in mlx-lm's layout:
//! `mlx_granite_affine_header <fixtures>/granite-3.1-1b-a400m-moe`.
//!
//! Input is the vendored `plain/header.json` (the IBM checkpoint). The rule is
//! transcribed from mlx-lm `granitemoe.py` and mlx `quantize`, not from this
//! crate's export: `sanitize` splits `input_linear` into `switch_mlp.gate_proj`
//! and `switch_mlp.up_proj` along the row axis and renames `output_linear` to
//! `switch_mlp.down_proj`; every `.weight` of two or more dimensions whose last
//! dimension is a multiple of the group size becomes `weight` (U32, last
//! dimension `x bits / 32`), `scales` and `biases` (BF16, last dimension `/ 64`);
//! `quant_predicate` quantizes `router.layer` at 8 bits and the rest at 4.
//! The reference stays self-referential: it shares the rule with the export.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{read, write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

const GROUP_SIZE: u64 = 64;
const DEFAULT_BITS: u64 = 4;
const ROUTER_BITS: u64 = 8;
const ROUTER_SUFFIX: &str = "block_sparse_moe.router.layer";
const FUSED: &str = "block_sparse_moe.input_linear.weight";
const GATE: &str = "block_sparse_moe.switch_mlp.gate_proj.weight";
const UP: &str = "block_sparse_moe.switch_mlp.up_proj.weight";
const DOWN_FROM: &str = "block_sparse_moe.output_linear.weight";
const DOWN_TO: &str = "block_sparse_moe.switch_mlp.down_proj.weight";

struct Entry {
    dtype: &'static str,
    shape: Vec<u64>,
}

impl Entry {
    fn bytes(&self) -> u64 {
        let width = if self.dtype == "U32" { 4 } else { 2 };
        self.shape.iter().product::<u64>() * width
    }
}

fn shape_of(value: &Value) -> Result<Vec<u64>, Box<dyn Error>> {
    let items = value["shape"].as_array().ok_or("entry without a shape")?;
    items
        .iter()
        .map(|item| item.as_u64().ok_or_else(|| "non-integer dimension".into()))
        .collect()
}

fn split_rows(name: &str, shape: &[u64]) -> Vec<(String, Vec<u64>)> {
    let mut half = shape.to_vec();
    half[1] /= 2;
    vec![
        (name.replace(FUSED, GATE), half.clone()),
        (name.replace(FUSED, UP), half),
    ]
}

fn sanitized(name: &str, shape: Vec<u64>) -> Vec<(String, Vec<u64>)> {
    if name.ends_with(FUSED) {
        return split_rows(name, &shape);
    }
    vec![(name.replace(DOWN_FROM, DOWN_TO), shape)]
}

fn bits_of(module: &str) -> u64 {
    if module.ends_with(ROUTER_SUFFIX) {
        ROUTER_BITS
    } else {
        DEFAULT_BITS
    }
}

fn quantized(name: &str, shape: &[u64]) -> Vec<(String, Entry)> {
    let last = shape.last().copied().unwrap_or(0);
    let Some(module) = name.strip_suffix(".weight") else {
        return vec![plain(name, shape)];
    };
    if shape.len() < 2 || last % GROUP_SIZE != 0 {
        return vec![plain(name, shape)];
    }
    let lead = &shape[..shape.len() - 1];
    let with_last = |value: u64| [lead, &[value]].concat();
    vec![
        (
            format!("{module}.weight"),
            Entry {
                dtype: "U32",
                shape: with_last(last * bits_of(module) / 32),
            },
        ),
        (
            format!("{module}.scales"),
            Entry {
                dtype: "BF16",
                shape: with_last(last / GROUP_SIZE),
            },
        ),
        (
            format!("{module}.biases"),
            Entry {
                dtype: "BF16",
                shape: with_last(last / GROUP_SIZE),
            },
        ),
    ]
}

fn plain(name: &str, shape: &[u64]) -> (String, Entry) {
    (
        name.to_string(),
        Entry {
            dtype: "BF16",
            shape: shape.to_vec(),
        },
    )
}

fn header_of(plain_header: &Map<String, Value>) -> Result<Value, Box<dyn Error>> {
    let mut entries: BTreeMap<String, Entry> = BTreeMap::new();
    for (name, value) in plain_header
        .iter()
        .filter(|(name, _)| *name != "__metadata__")
    {
        for (renamed, shape) in sanitized(name, shape_of(value)?) {
            entries.extend(quantized(&renamed, &shape));
        }
    }
    let mut header = Map::new();
    header.insert("__metadata__".into(), json!({"format": "mlx"}));
    let mut offset = 0;
    for (name, entry) in &entries {
        let end = offset + entry.bytes();
        header.insert(
            name.clone(),
            json!({"dtype": entry.dtype, "shape": entry.shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    Ok(Value::Object(header))
}

fn routers(plain_header: &Map<String, Value>) -> Vec<String> {
    plain_header
        .keys()
        .filter_map(|name| name.strip_suffix(".weight"))
        .filter(|module| module.ends_with(ROUTER_SUFFIX))
        .map(str::to_string)
        .collect()
}

fn with_router_entries(config: &mut Value, modules: &[String]) {
    for key in ["quantization", "quantization_config"] {
        if let Some(block) = config[key].as_object_mut() {
            modules.iter().for_each(|module| {
                block.insert(
                    module.clone(),
                    json!({"group_size": GROUP_SIZE, "bits": ROUTER_BITS}),
                );
            });
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(root) = std::env::args().nth(1).map(PathBuf::from) else {
        return Err("usage: mlx_granite_affine_header <granite fixture directory>".into());
    };
    let plain_header: Value = serde_json::from_slice(&read(root.join("plain/header.json"))?)?;
    let plain_entries = plain_header
        .as_object()
        .ok_or("plain header is not an object")?;
    let header = header_of(plain_entries)?;
    write(
        root.join("affine4/header.json"),
        serde_json::to_vec(&header)?,
    )?;
    let config_path = root.join("affine4/config.json");
    let mut config: Value = serde_json::from_slice(&read(&config_path)?)?;
    with_router_entries(&mut config, &routers(plain_entries));
    write(&config_path, serde_json::to_vec_pretty(&config)?)?;
    report(&root, &header)
}

fn report(root: &Path, header: &Value) -> Result<(), Box<dyn Error>> {
    let count = header.as_object().ok_or("header is not an object")?.len() - 1;
    println!(
        "{}: {count} tensors",
        root.join("affine4/header.json").display()
    );
    Ok(())
}
