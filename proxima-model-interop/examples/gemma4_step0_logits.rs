//! Step-0 logits of one fixture prompt through `LoadedModel::forward_logits_on_backend`
//! on the CPU interpreter and on Metal, as log-softmax, so they can be set
//! next to llama-server's `n_probs` logprobs for the same ids
//! (`llama_answer_oracle` writes those). Reports each engine's top-5 plus the
//! logprob it assigns to every id in `--watch`.
//!
//! `cargo run --release --example gemma4_step0_logits --features std,metal -- \
//!    --model <gguf> --fixture <llama_ids.json> --prompt-index 0 --watch 8591,5279 --out <json>`

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fs::File;

use proxima_gguf::parse_complete;
use proxima_model_interop::GPU_LAYERS_ALL;
use proxima_model_interop::LoadedModel;
use serde_json::{Value, json};

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = logits
        .iter()
        .map(|logit| f64::from(*logit - maximum).exp())
        .sum();
    let normalizer = f64::from(maximum) + total.ln();
    logits
        .iter()
        .map(|logit| f64::from(*logit) - normalizer)
        .collect()
}

fn top_ids(logprobs: &[f64], count: usize) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..logprobs.len()).collect();
    ranked.sort_by(|left, right| {
        logprobs[*right]
            .total_cmp(&logprobs[*left])
            .then_with(|| left.cmp(right))
    });
    ranked.truncate(count);
    ranked
}

fn report(logits: &[f32], watch: &[usize]) -> Value {
    let logprobs = log_softmax(logits);
    let top: Vec<Value> = top_ids(&logprobs, 5)
        .into_iter()
        .map(|id| json!({ "id": id, "logit": logits[id], "logprob": logprobs[id] }))
        .collect();
    let watched: Vec<Value> = watch
        .iter()
        .map(|&id| json!({ "id": id, "logit": logits[id], "logprob": logprobs[id] }))
        .collect();
    json!({ "top5": top, "watched": watched })
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
    let fixture: Vec<Value> = serde_json::from_str(
        &std::fs::read_to_string(required("--fixture")).expect("read fixture"),
    )
    .expect("fixture json");
    let index: usize = required("--prompt-index")
        .parse()
        .expect("--prompt-index is a usize");
    let prompt = fixture[index]["prompt"]
        .as_str()
        .expect("prompt")
        .to_owned();
    let watch: Vec<usize> = required("--watch")
        .split(',')
        .map(|id| id.parse().expect("watch id"))
        .collect();

    let file = File::open(required("--model")).expect("open model");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap model");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse gguf header");
    let model = LoadedModel::load(&parsed, bytes).expect("load model");

    let backends = values
        .get("--backends")
        .cloned()
        .unwrap_or_else(|| "metal".to_owned());
    let llama_lists = llama_step0_lists(&values, index);
    let mut engines = serde_json::Map::new();
    for label in backends.split(',') {
        let gpu_layers = if label == "cpu" { 0 } else { GPU_LAYERS_ALL };
        let logits = model
            .forward_logits_on_backend(&prompt, gpu_layers)
            .unwrap_or_else(|error| panic!("{label} forward_logits_on_backend: {error:?}"));
        let mut entry = report(&logits, &watch);
        entry["against_llama"] = cross_rank(&logits, &llama_lists);
        engines.insert(label.to_owned(), entry);
    }
    let rendered = serde_json::to_string_pretty(&json!({ "prompt": prompt, "engines": engines }))
        .expect("render");
    std::fs::write(required("--out"), &rendered).expect("write out");
    println!("{rendered}");
}

fn llama_step0_lists(
    values: &BTreeMap<String, String>,
    index: usize,
) -> Vec<(String, Vec<(usize, f64)>)> {
    let Some(spec) = values.get("--llama-json") else {
        return Vec::new();
    };
    spec.split(',')
        .map(|pair| {
            let (label, path) = pair.split_once('=').expect("label=path");
            let records: Vec<Value> =
                serde_json::from_str(&std::fs::read_to_string(path).expect("read llama json"))
                    .expect("llama json");
            let step0 = &records[index]["llama_top5_per_step"][0]["top_logprobs"];
            let list = step0
                .as_array()
                .expect("top_logprobs array")
                .iter()
                .map(|entry| {
                    (
                        usize::try_from(entry["id"].as_u64().expect("id")).expect("id fits usize"),
                        entry["logprob"].as_f64().expect("logprob"),
                    )
                })
                .collect();
            (label.to_owned(), list)
        })
        .collect()
}

fn cross_rank(logits: &[f32], llama_lists: &[(String, Vec<(usize, f64)>)]) -> Value {
    let logprobs = log_softmax(logits);
    let proxima_top = top_ids(&logprobs, 5);
    let proxima_rank = |id: usize| {
        top_ids(&logprobs, usize::MAX)
            .iter()
            .position(|&candidate| candidate == id)
    };
    let ranked = llama_lists
        .iter()
        .map(|(label, list)| {
            let proxima_top_in_llama: Vec<Value> = proxima_top
                .iter()
                .map(|&id| {
                    let found = list.iter().position(|&(candidate, _)| candidate == id);
                    json!({
                        "id": id,
                        "proxima_logprob": logprobs[id],
                        "llama_rank": found,
                        "llama_logprob": found.map(|position| list[position].1),
                    })
                })
                .collect();
            let llama_top_in_proxima: Vec<Value> = list
                .iter()
                .take(5)
                .enumerate()
                .map(|(rank, &(id, logprob))| {
                    json!({
                        "id": id,
                        "llama_rank": rank,
                        "llama_logprob": logprob,
                        "proxima_rank": proxima_rank(id),
                        "proxima_logprob": logprobs[id],
                    })
                })
                .collect();
            json!({
                "llama": label,
                "llama_list_len": list.len(),
                "proxima_top5_in_llama": proxima_top_in_llama,
                "llama_top5_in_proxima": llama_top_in_proxima,
            })
        })
        .collect();
    Value::Array(ranked)
}
