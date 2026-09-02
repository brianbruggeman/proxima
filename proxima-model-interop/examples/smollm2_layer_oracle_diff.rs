// a diagnostic binary, not library surface: every `.expect()`/`.unwrap()`
// below is a setup precondition (real checkpoint present, oracle dump
// present, program builds, label present in this run's own `labels` vec)
// whose only correct response is to panic with the failing step named,
// matching this crate's own sibling examples (`any_listener.rs` and friends
// carry the identical allow for the identical reason).
#![allow(clippy::expect_used, clippy::unwrap_used)]

//! Depth-bisection of the real SmolLM2-135M-Instruct forward pass against
//! `llama.cpp`'s own per-layer residual-stream dump
//! (`l_out-<layer>`/`inp_embd`, produced OUTSIDE this repo by a small
//! `llama.h`-linked C++ probe against the real f16 GGUF, not committed
//! here -- same convention as `smollm2_logit_oracle_diff.rs`).
//!
//! [`LoadedModel::forward_node_values`]'s own doc explains the derivation:
//! rebuilding [`mistral_cached_forward_program_with_experts`] at a shorter
//! `block_count` against the SAME architecture shares every `NodeId` up to
//! where it stops (`NodeId`'s id-is-index invariant), so the last id two
//! consecutive-depth builds still agree on IS the residual-stream value
//! right after that layer -- the exact quantity `l_out-<layer>` names on
//! the oracle side.

use std::env;
use std::fs;
use std::path::PathBuf;

use proxima_model_interop::{LoadedModel, architecture_from_hf_config, parse_hf_config};
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::level::Level;
use proxima_telemetry::recorder::Recorder;
use proxima_tensor::spec::mistral_cached_forward_program_with_experts;

fn read_oracle_activation(path: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(path)
        .unwrap_or_else(|error| panic!("read oracle activation at {path:?}: {error}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// The last `NodeId` a `block_count`-deep throwaway program shares with a
/// `block_count + 1`-deep one -- the residual-stream output right after
/// layer `block_count - 1` (`block_count == 0` names the embedding lookup
/// itself, before any layer runs).
fn layer_boundary_node_id(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
) -> proxima_tensor::op::NodeId {
    if block_count == 0 {
        // `ids`, `token_embd.weight`, then the gather itself -- the
        // embedding lookup is always the 3rd op any depth of this program
        // appends, so a 1-layer throwaway build's own `NodeId(2)` names it
        // without needing a 0-layer build (`causal_mask`'s `Iota` extent
        // would be degenerate at `block_count == 0` for a MoE-shaped call,
        // an edge this diagnostic has no reason to depend on).
        return proxima_tensor::op::NodeId(2);
    }
    let (shallow, _, _) = mistral_cached_forward_program_with_experts(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        0,
        0,
    )
    .expect("build shallow throwaway program");
    let (deep, _, _) = mistral_cached_forward_program_with_experts(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count + 1,
        0,
        0,
    )
    .expect("build deep throwaway program");

    let first_diff = shallow
        .iter()
        .zip(deep.iter())
        .position(|(left, right)| left != right)
        .unwrap_or(shallow.len());
    proxima_tensor::op::NodeId((first_diff - 1) as u32)
}

fn main() {
    let model_dir = env::args().nth(1).unwrap_or_else(|| {
        "/Users/brianbruggeman/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct".to_string()
    });
    let oracle_dir = env::args().nth(2).unwrap_or_else(|| {
        "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/6cd9e134-c1a3-450a-be93-76dd95389bf4/scratchpad/diverge/dump_f16_layers".to_string()
    });
    let prompt = env::args()
        .nth(3)
        .unwrap_or_else(|| "The capital of France is the".to_string());
    let log_path = env::args()
        .nth(4)
        .unwrap_or_else(|| "smollm2_layer_diff.jsonl".to_string());

    let model_path = PathBuf::from(&model_dir).join("model.safetensors");
    let oracle_path = PathBuf::from(&oracle_dir);
    if !model_path.exists() {
        println!("skipping: no host-local smollm2 safetensors checkpoint at {model_path:?}");
        return;
    }
    if !oracle_path.exists() {
        println!("skipping: no oracle layer-activation dump directory at {oracle_path:?}");
        return;
    }

    let recorder = Recorder::builder()
        .export(Exporter::file(&log_path).format(Formatter::Text))
        .expect("file exporter")
        .install()
        .expect("recorder");

    let config_bytes =
        fs::read(PathBuf::from(&model_dir).join("config.json")).expect("read config.json");
    let hf_config = parse_hf_config(&config_bytes).expect("parse config.json");
    let architecture = architecture_from_hf_config(&hf_config);

    let tokenizer_bytes =
        fs::read(PathBuf::from(&model_dir).join("tokenizer.json")).expect("read tokenizer.json");
    let vocab =
        proxima_tokenizer::hf::vocab_from_tokenizer_json(&tokenizer_bytes, Some(1), None, None)
            .expect("build vocab from tokenizer.json");

    let file_bytes = fs::read(&model_path).expect("read model.safetensors");
    let manifest =
        proxima_safetensors::parse_complete(&file_bytes).expect("parse model.safetensors");
    let mut length_prefix = [0u8; 8];
    length_prefix.copy_from_slice(&file_bytes[..8]);
    let data_start = 8 + u64::from_le_bytes(length_prefix);

    let model =
        LoadedModel::load_from_safetensors(&manifest, &file_bytes, data_start, architecture, vocab)
            .expect("load real smollm2 checkpoint");

    let block_count = architecture.block_count;
    let mut node_ids = Vec::with_capacity(block_count as usize + 1);
    let mut labels: Vec<String> = Vec::with_capacity(block_count as usize + 1);
    node_ids.push(layer_boundary_node_id(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.query_heads,
        architecture.kv_heads,
        architecture.head_dim,
        0,
    ));
    labels.push("inp_embd".to_string());
    for layer in 0..block_count {
        node_ids.push(layer_boundary_node_id(
            architecture.vocab,
            architecture.embedding,
            architecture.feed_forward,
            architecture.query_heads,
            architecture.kv_heads,
            architecture.head_dim,
            layer + 1,
        ));
        labels.push(format!("l_out-{layer}"));
    }

    let ours = model
        .forward_node_values(&prompt, &node_ids)
        .expect("compute our own layer activations");

    println!("label node_id max_abs_diff worst_index ours_value theirs_value");
    let mut first_gross_divergence: Option<String> = None;
    for (label, values) in labels.iter().zip(ours.iter()) {
        let oracle_path = oracle_path.join(format!("{label}.f32"));
        if !oracle_path.exists() {
            println!("{label} MISSING_ORACLE_FILE");
            continue;
        }
        let theirs = read_oracle_activation(&oracle_path);
        assert_eq!(
            values.len(),
            theirs.len(),
            "{label}: element count mismatch"
        );

        let mut max_abs_diff = 0f32;
        let mut worst_index = 0usize;
        for index in 0..values.len() {
            let diff = (values[index] - theirs[index]).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
                worst_index = index;
            }
        }
        println!(
            "{label} node={} {:e} {worst_index} {:.6} {:.6}",
            node_ids[labels
                .iter()
                .position(|candidate| candidate == label)
                .unwrap()]
            .0,
            max_abs_diff,
            values[worst_index],
            theirs[worst_index]
        );
        let level = if max_abs_diff > 1e-1 {
            Level::WARN
        } else {
            Level::DEBUG
        };
        // this diagnostic runs once per layer (31 max on a real checkpoint)
        // and `tag` needs `&'static str`; leaking the per-layer label is
        // bounded and cheaper than threading an owned-string tag type
        // through the recorder for a one-shot comparison run (same
        // rationale `proxima-tensor/examples/q4k_ggml_fidelity.rs` uses).
        let layer_label: &'static str = Box::leak(label.clone().into_boxed_str());
        recorder
            .log()
            .level(level)
            .message("layer activation compared against oracle")
            .module_path(module_path!())
            .tag("layer", layer_label)
            .tag("max_abs_diff", f64::from(max_abs_diff))
            .tag("worst_index", worst_index as u64)
            .emit();

        // per-position breakdown -- always recorded (DEBUG, so a default
        // view stays quiet), never a hand-rolled env-gated dump: this is
        // what proved the RoPE-permutation root cause (position 0's angle
        // is zero for every pair regardless of pairing convention, so it
        // stays at the noise floor; position 1 onward does not, unless
        // this permutation is applied).
        if label != "inp_embd" {
            let embedding = architecture.embedding as usize;
            for position in 0..(values.len() / embedding) {
                let mut position_max = 0f32;
                for local in 0..embedding {
                    let diff = (values[position * embedding + local]
                        - theirs[position * embedding + local])
                        .abs();
                    position_max = position_max.max(diff);
                }
                recorder
                    .log()
                    .level(Level::DEBUG)
                    .message("layer activation compared against oracle, per position")
                    .module_path(module_path!())
                    .tag("layer", layer_label)
                    .tag("position", position as u64)
                    .tag("max_abs_diff", f64::from(position_max))
                    .emit();
            }
        }

        if max_abs_diff > 1e-1 && first_gross_divergence.is_none() {
            first_gross_divergence = Some(label.clone());
        }
    }
    while recorder.drain() > 0 {}

    match first_gross_divergence {
        Some(label) => println!("\nfirst gross divergence at: {label}"),
        None => println!("\nno gross divergence found at any dumped layer"),
    }
}
