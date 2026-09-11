//! Probe: does simulated `Q4_K` round-trip quantization of BGE-small's
//! weights hold retrieval-relevant fidelity? Answers the question that
//! gates the whole onnx-to-gguf direction *before* anything about a
//! bridge is built.
//!
//! Pipe question, in writing: this is a throwaway measurement binary, not
//! a library type — it wires two already-landed crates
//! (`proxima_gguf::quant::q4_k`/`policy`, `proxima_onnx::lower` +
//! `proxima_tensor::cpu::evaluate_named`) together at the call site. No
//! new struct, trait, or combinator is introduced. `proxima-gguf` is
//! pulled in as a `[dev-dependencies]` edge on `proxima-onnx` (this
//! example only) — the reverse of the production bridge direction this
//! task asks about, and dev-scoped, so it proves nothing about whether
//! `proxima-gguf` should depend on `proxima-onnx`/`proxima-tensor` (it
//! should not, and does not — see the report).
//!
//! Method: take BGE's real f32 initializers (from `proxima_onnx::lower`),
//! round-trip every weight-matrix-shaped tensor through
//! `q4_k::quantize`/`q4_k::dequantize` (padding to a `QK_K=256` multiple
//! where the tensor's own flat length isn't one, then truncating back —
//! this is a FIDELITY simulation, not the row-addressed packed format the
//! real matmul kernel needs; that structural constraint is exercised
//! separately in `bge_q4k_rate_probe.rs`), substitute the round-tripped
//! weights back into the SAME lowered program, and diff the resulting
//! sentence embeddings against the untouched f32 forward pass.
//!
//! 1-D tensors (length exactly 384, BGE's hidden size — every LayerNorm
//! gamma/beta and every attention/FFN bias) are excluded from EVERY
//! condition below, always kept f32. This mirrors real practice
//! (`PrecisionPolicy::llama_cpp_q4_k_s` never block-quantizes a norm, and
//! ggml never block-quantizes a 1-D tensor at all) rather than being an
//! arbitrary carve-out invented for this probe.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

use proxima_gguf::GgmlType;
use proxima_gguf::quant::policy::{PrecisionPolicy, TensorRole};
use proxima_gguf::quant::q4_k;
use proxima_tensor::NodeId;
use proxima_tensor::cpu::evaluate_named;

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const HIDDEN: usize = 384;

fn sentences_s8() -> [(&'static str, Vec<i64>); 3] {
    [
        (
            "the cat sat on the mat",
            vec![101, 1996, 4937, 2938, 2006, 1996, 13523, 102],
        ),
        (
            "a cat is sitting on a mat",
            vec![101, 1037, 4937, 2003, 3564, 2006, 1037, 13523, 102],
        ),
        (
            "quantum physics explains atomic energy",
            vec![101, 8559, 5584, 7607, 9593, 2943, 102],
        ),
    ]
}

/// Synthetic token stream of exactly `target_len`: `[CLS]`, the six
/// content ids of "the cat sat on the mat" tiled to fill, `[SEP]`. Not a
/// real sentence — a length-S shape probe over the SAME real weights, so
/// S=128/S=512 drift is measured on genuine forward-pass arithmetic, just
/// without a paired oracle cosine (no ground-truth long-sentence pair
/// exists for this model in this repo).
fn synthetic_tokens(target_len: usize) -> Vec<i64> {
    const MIDDLE: [i64; 6] = [1996, 4937, 2938, 2006, 1996, 13523];
    let mut tokens = vec![101i64];
    let mut index = 0usize;
    while tokens.len() < target_len - 1 {
        tokens.push(MIDDLE[index % MIDDLE.len()]);
        index += 1;
    }
    tokens.push(102);
    tokens
}

fn dynamic_inputs(tokens: &[i64]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let sequence_length = tokens.len();
    let input_ids = tokens.iter().map(|&id| id as f32).collect();
    let attention_mask = vec![1.0f32; sequence_length];
    let token_type_ids = vec![0.0f32; sequence_length];
    (input_ids, attention_mask, token_type_ids)
}

fn cls_normalize(data: &[f32]) -> Vec<f32> {
    let cls = &data[0..HIDDEN];
    let norm = cls.iter().map(|value| value * value).sum::<f32>().sqrt();
    cls.iter().map(|&value| value / norm).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let total: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&left, &right)| (left - right).abs())
        .sum();
    total / a.len() as f32
}

/// Classifies a real BGE (HuggingFace BERT ONNX export) initializer name
/// into `proxima_gguf`'s llama.cpp-shaped [`TensorRole`]. BGE's naming
/// (`encoder.layer.N.attention.self.query.weight`) has no relation to
/// llama.cpp's (`blk.N.attn_q.weight`) — [`TensorRole::classify`] cannot
/// read it directly — so this is the probe's own translation, checked
/// order-sensitively the same way `TensorRole::classify` itself is
/// (`attention.output.dense` before the bare `output.dense` fallback, the
/// same substring-superstring trap `ffn_gate_inp`/`ffn_gate` guards
/// against upstream).
fn classify_bert_role(name: &str) -> TensorRole {
    if name.contains("word_embeddings") {
        TensorRole::TokenEmbd
    } else if name.contains("attention.self.query") {
        TensorRole::AttnQ
    } else if name.contains("attention.self.key") {
        TensorRole::AttnK
    } else if name.contains("attention.self.value") {
        TensorRole::AttnV
    } else if name.contains("attention.output.dense") {
        TensorRole::AttnOutput
    } else if name.contains("attention.output.LayerNorm") {
        TensorRole::AttnNorm
    } else if name.contains("intermediate.dense") {
        TensorRole::FfnUp
    } else if name.contains("output.LayerNorm") {
        TensorRole::FfnNorm
    } else if name.contains("output.dense") {
        // attention.output.dense already matched above; this is the FFN
        // sub-block's own output.dense (the down-projection).
        TensorRole::FfnDown
    } else {
        TensorRole::Other
    }
}

/// Pads `data` up to the next `QK_K`-element multiple with zeros,
/// round-trips it through `q4_k::quantize` -> `q4_k::dequantize`, then
/// truncates back to `data.len()`. Zero padding is inert: those elements
/// never round-trip back out.
fn round_trip_q4k(data: &[f32]) -> Vec<f32> {
    let padded_len = data.len().div_ceil(q4_k::QK_K) * q4_k::QK_K;
    let mut padded = vec![0.0f32; padded_len];
    padded[..data.len()].copy_from_slice(data);
    let block_count = padded_len / q4_k::QK_K;
    let mut packed = vec![0u8; q4_k::bytes_for_blocks(block_count)];
    q4_k::quantize(&padded, &mut packed).expect("q4_k quantize");
    let mut out = vec![0.0f32; padded_len];
    q4_k::dequantize(&packed, &mut out).expect("q4_k dequantize");
    out.truncate(data.len());
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Condition {
    F32Baseline,
    AllQuantized,
    Policy,
    PolicyNoEmbed,
}

impl Condition {
    fn label(self) -> &'static str {
        match self {
            Condition::F32Baseline => "f32-baseline",
            Condition::AllQuantized => "all-tensors-q4k",
            Condition::Policy => "policy-q4k",
            Condition::PolicyNoEmbed => "policy-q4k-no-embed",
        }
    }
}

/// Applies `condition` to `initializers`, returning a new owned vector in
/// the same order. 1-D (length-384) tensors are never touched. Returns
/// the count of tensors actually quantized this call, for the N==0 check.
fn build_condition(
    initializers: &[(String, Vec<f32>)],
    policy: &PrecisionPolicy,
    condition: Condition,
    declared_names: &std::collections::BTreeSet<String>,
) -> (Vec<(String, Vec<f32>)>, usize) {
    let mut quantized_count = 0usize;
    let out = initializers
        .iter()
        .map(|(name, data)| {
            // Only tensors the ONNX graph itself declared as initializers
            // (real weights) are eligible -- `Lowered.initializers` also
            // carries pin-dependent constant-folded values (their name and
            // shape vary with the pinned sequence length), which must pass
            // through untouched or the forward pass shape-mismatches on
            // whichever length didn't build this condition's weight set.
            if data.len() == HIDDEN
                || condition == Condition::F32Baseline
                || !declared_names.contains(name)
            {
                return (name.clone(), data.clone());
            }
            let role = classify_bert_role(name);
            let quantize_this = match condition {
                Condition::F32Baseline => unreachable!(),
                Condition::AllQuantized => true,
                Condition::Policy => policy.target_for_role(role) == GgmlType::Q4_K,
                Condition::PolicyNoEmbed => {
                    role != TensorRole::TokenEmbd && policy.target_for_role(role) == GgmlType::Q4_K
                }
            };
            if quantize_this {
                quantized_count += 1;
                (name.clone(), round_trip_q4k(data))
            } else {
                (name.clone(), data.clone())
            }
        })
        .collect();
    (out, quantized_count)
}

fn run_embedding(
    program: &[proxima_tensor::op::Op],
    output: NodeId,
    initializers: &[(String, Vec<f32>)],
    graph_inputs: &[String],
    tokens: &[i64],
) -> Vec<f32> {
    let (input_ids, attention_mask, token_type_ids) = dynamic_inputs(tokens);
    let mut named: Vec<(&str, &[f32])> = initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    for name in graph_inputs {
        let data: &[f32] = match name.as_str() {
            "input_ids" => &input_ids,
            "attention_mask" => &attention_mask,
            "token_type_ids" => &token_type_ids,
            other => panic!("unexpected graph input {other:?}"),
        };
        named.push((name.as_str(), data));
    }
    let evaluated =
        evaluate_named(program, &[], &named, &[output]).expect("evaluate BGE forward pass");
    let (data, shape) = evaluated.get(output).expect("last_hidden_state present");
    assert_eq!(
        shape,
        &[1u64, tokens.len() as u64, HIDDEN as u64],
        "unexpected last_hidden_state shape"
    );
    cls_normalize(data)
}

fn main() {
    let Ok(model_path) = env::var(MODEL_PATH_ENV) else {
        eprintln!("skipping: set {MODEL_PATH_ENV} to a local BGE-small-en-v1.5 model.onnx");
        return;
    };
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: {MODEL_PATH_ENV}={model_path:?} does not exist");
        return;
    }
    let bytes = fs::read(&model_path).expect("read bge model.onnx");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse");
    let graph = model.graph.as_ref().expect("graph");

    let policy = PrecisionPolicy::llama_cpp_q4_k_s();

    println!("=== ROLE CLASSIFICATION (BGE ONNX names -> TensorRole) ===");
    let mut role_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for tensor in &graph.initializer {
        let role = classify_bert_role(tensor.name);
        *role_counts.entry(role_label(role)).or_insert(0) += 1;
    }
    for (role, count) in &role_counts {
        println!("  {role}: {count}");
    }
    let total_named = graph.initializer.len();
    assert!(total_named > 0, "N==0: no initializers found — RED");
    println!("total initializers: {total_named}\n");
    let declared_names: std::collections::BTreeSet<String> = graph
        .initializer
        .iter()
        .map(|tensor| tensor.name.to_string())
        .collect();

    let conditions = [
        Condition::AllQuantized,
        Condition::Policy,
        Condition::PolicyNoEmbed,
    ];

    for &bucket in &["S~8", "S=128", "S=512"] {
        println!("=== {bucket} ===");
        let is_short_bucket = bucket == "S~8";

        let items: Vec<(&str, Vec<i64>)> = if is_short_bucket {
            sentences_s8().to_vec()
        } else {
            let target_len = if bucket == "S=128" { 128 } else { 512 };
            vec![("synthetic", synthetic_tokens(target_len))]
        };

        // Each sentence in the short bucket has its own real token count
        // (7/8/9) -- BGE's tokenizer output, not a fixed pin -- so this
        // lowers once per DISTINCT token length actually present, exactly
        // the cache-by-length pattern `bge_eval.rs` uses for the same
        // reason.
        let mut lowered_by_length: BTreeMap<usize, proxima_onnx::lower::Lowered> = BTreeMap::new();
        for (_, tokens) in &items {
            lowered_by_length.entry(tokens.len()).or_insert_with(|| {
                let mut pins = BTreeMap::new();
                pins.insert("batch_size", 1u64);
                pins.insert("sequence_length", tokens.len() as u64);
                proxima_onnx::lower::lower_graph_pinned(graph, &pins)
                    .expect("lower BGE-small with pinned symbolic axes")
            });
        }

        let mut f32_embeddings = Vec::new();
        for (label, tokens) in &items {
            let lowered = &lowered_by_length[&tokens.len()];
            let output: NodeId = lowered
                .graph_outputs
                .first()
                .expect("last_hidden_state output")
                .1;
            let embedding = run_embedding(
                &lowered.program,
                output,
                &lowered.initializers,
                &lowered.graph_inputs,
                tokens,
            );
            println!(
                "  f32 baseline [{label}] len={}: first3={:?} finite={}",
                tokens.len(),
                &embedding[0..3],
                embedding.iter().all(|value| value.is_finite())
            );
            f32_embeddings.push(embedding);
        }

        if is_short_bucket {
            let oracle = [0.936311f32, 0.378777, 0.334176];
            let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
            println!("  --- cosine-oracle check (f32) ---");
            for (index, &(left, right)) in pairs.iter().enumerate() {
                let measured = cosine(&f32_embeddings[left], &f32_embeddings[right]);
                println!(
                    "    pair {left}-{right}: measured={measured:.6} oracle={:.6} delta={:.6}",
                    oracle[index],
                    (measured - oracle[index]).abs()
                );
            }
        }

        for &condition in &conditions {
            println!("  --- condition={} ---", condition.label());
            let mut quantized_embeddings = Vec::new();
            let mut quantized_counts = Vec::new();
            for (_, tokens) in &items {
                let lowered = &lowered_by_length[&tokens.len()];
                let (substituted, quantized_count) =
                    build_condition(&lowered.initializers, &policy, condition, &declared_names);
                quantized_counts.push(quantized_count);
                let output: NodeId = lowered
                    .graph_outputs
                    .first()
                    .expect("last_hidden_state output")
                    .1;
                let embedding = run_embedding(
                    &lowered.program,
                    output,
                    &substituted,
                    &lowered.graph_inputs,
                    tokens,
                );
                quantized_embeddings.push(embedding);
            }
            assert!(
                quantized_counts.iter().all(|&count| count > 0),
                "N==0: {} quantized zero tensors on some length — RED",
                condition.label()
            );
            println!(
                "    quantized counts per item (of {total_named} declared initializers): {quantized_counts:?}"
            );
            for (index, (label, _)) in items.iter().enumerate() {
                let f32_embedding = &f32_embeddings[index];
                let quantized_embedding = &quantized_embeddings[index];
                let drift_cosine = cosine(f32_embedding, quantized_embedding);
                let max_delta = max_abs_diff(f32_embedding, quantized_embedding);
                let mean_delta = mean_abs_diff(f32_embedding, quantized_embedding);
                println!(
                    "    [{label}] f32-vs-quantized cosine={drift_cosine:.6} max_abs_delta={max_delta:.6} mean_abs_delta={mean_delta:.6}"
                );
            }
            if is_short_bucket {
                let oracle = [0.936311f32, 0.378777, 0.334176];
                let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
                for (index, &(left, right)) in pairs.iter().enumerate() {
                    let measured =
                        cosine(&quantized_embeddings[left], &quantized_embeddings[right]);
                    println!(
                        "    quantized pair {left}-{right}: measured={measured:.6} oracle={:.6} delta={:.6}",
                        oracle[index],
                        (measured - oracle[index]).abs()
                    );
                }
            }
        }
        println!();
    }
}

fn role_label(role: TensorRole) -> &'static str {
    match role {
        TensorRole::TokenEmbd => "TokenEmbd",
        TensorRole::AttnQ => "AttnQ",
        TensorRole::AttnK => "AttnK",
        TensorRole::AttnV => "AttnV",
        TensorRole::AttnOutput => "AttnOutput",
        TensorRole::AttnNorm => "AttnNorm",
        TensorRole::FfnGateInp => "FfnGateInp",
        TensorRole::FfnGate => "FfnGate",
        TensorRole::FfnUp => "FfnUp",
        TensorRole::FfnDown => "FfnDown",
        TensorRole::FfnNorm => "FfnNorm",
        TensorRole::OutputNorm => "OutputNorm",
        TensorRole::OutputWeight => "OutputWeight",
        TensorRole::Other => "Other",
        _ => "Unknown",
    }
}
