//! Proves [`Architecture::step_inputs`] from OUTSIDE this crate: a foreign
//! architecture wraps [`DenseArch::bind`] (same delegation shape
//! `external_architecture_registry.rs` already exercises) and adds one
//! extra `Op::Input` leaf, `"aux_rows"`, that no builtin decode-loop block
//! knows how to feed -- only [`Architecture::step_inputs`] can.
//!
//! `aux_rows` gathers rows out of a second step-supplied leaf, `"aux_table"`
//! (a small constant table, re-fed identically every step -- nothing stops
//! a step input from being step-invariant, it is simply computed fresh by
//! the same hook), and the gathered row is added straight onto
//! [`BoundProgram::logits_root`] -- so a run's own output is only correct
//! if `step_inputs` actually ran and its gather actually changed the
//! result, not merely that `resolve` picked the right architecture.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrayvec::ArrayVec;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::tensor::MAX_DIMS;
use proxima_gguf::{
    GgmlType, GgufModel, MetadataArray, MetadataValue, TensorPayload, parse_complete,
    write_complete,
};
use proxima_model_interop::{
    Architecture, ArchitectureRegistry, BoundProgram, DenseArch, InteropError, LoadedModel,
    StepInput, StepInputContext, symbols,
};
use proxima_primitives::pipe::Pipe;
use proxima_tensor::spec::{elementwise, embedding_lookup, input_leaf};
use proxima_tensor::{DType, Extent, ScalarOp};
use proxima_tokenizer::byte_level::byte_to_char;

#[path = "support/mod.rs"]
#[allow(dead_code)]
mod support;

const STEP_INPUT_ARCHITECTURE_NAME: &str = "acme-step-input";
const TABLE_ROWS: u32 = 8;

fn push_tokenizer_metadata(metadata: &mut Vec<(String, MetadataValue)>) {
    let mut tokens: Vec<String> = (0..=255u8).map(|byte| String::from(byte_to_char(byte))).collect();
    tokens.push(String::from("<|endoftext|>"));
    metadata.push((
        "tokenizer.ggml.model".to_string(),
        MetadataValue::String("gpt2".to_string()),
    ));
    metadata.push((
        "tokenizer.ggml.tokens".to_string(),
        MetadataValue::Array(MetadataArray::String(tokens)),
    ));
    metadata.push((
        "tokenizer.ggml.merges".to_string(),
        MetadataValue::Array(MetadataArray::String(Vec::new())),
    ));
    metadata.push(("tokenizer.ggml.bos_token_id".to_string(), MetadataValue::U32(0)));
    metadata.push((
        "tokenizer.ggml.eos_token_id".to_string(),
        MetadataValue::U32(support::EOS_TOKEN_ID),
    ));
}

/// Byte-for-byte the same dense checkpoint layout
/// `external_architecture_registry.rs`'s own `foreign_checkpoint_bytes`
/// writes, under a different `general.architecture` name -- this test
/// proves a different seam ([`Architecture::step_inputs`]), not a second
/// tensor layout.
fn checkpoint_bytes(architecture_name: &str) -> Vec<u8> {
    let embedding = support::EMBEDDING;
    let feed_forward = support::FEED_FORWARD;
    let vocab = support::VOCAB;

    let mut buffers: Vec<Vec<u8>> = Vec::new();
    let mut specs: Vec<(String, ArrayVec<u64, MAX_DIMS>, GgmlType)> = Vec::new();

    let embed_values = vec![0.0f32; (vocab * embedding) as usize];
    buffers.push(support::encode_weights(GgmlType::F32, &embed_values));
    specs.push((
        String::from("token_embd.weight"),
        [u64::from(vocab * embedding)].into_iter().collect(),
        GgmlType::F32,
    ));

    let norm_values = vec![1.0f32; embedding as usize];
    buffers.push(support::encode_weights(GgmlType::F32, &norm_values));
    specs.push((
        String::from("blk.0.attn_norm.weight"),
        [u64::from(embedding)].into_iter().collect(),
        GgmlType::F32,
    ));
    buffers.push(support::encode_weights(GgmlType::F32, &norm_values));
    specs.push((
        String::from("blk.0.ffn_norm.weight"),
        [u64::from(embedding)].into_iter().collect(),
        GgmlType::F32,
    ));

    let kv_dim = support::KV_HEADS * support::HEAD_DIM;
    let square = vec![0.0f32; (embedding * embedding) as usize];
    let kv_projection = vec![0.0f32; (embedding * kv_dim) as usize];
    buffers.push(support::encode_weights(GgmlType::F32, &square));
    specs.push((
        String::from("blk.0.attn_q.weight"),
        [u64::from(embedding), u64::from(embedding)].into_iter().collect(),
        GgmlType::F32,
    ));
    for name in ["blk.0.attn_k.weight", "blk.0.attn_v.weight"] {
        buffers.push(support::encode_weights(GgmlType::F32, &kv_projection));
        specs.push((
            String::from(name),
            [u64::from(embedding), u64::from(kv_dim)].into_iter().collect(),
            GgmlType::F32,
        ));
    }
    buffers.push(support::encode_weights(GgmlType::F32, &square));
    specs.push((
        String::from("blk.0.attn_output.weight"),
        [u64::from(embedding), u64::from(embedding)].into_iter().collect(),
        GgmlType::F32,
    ));

    let ffn = vec![0.0f32; (embedding * feed_forward) as usize];
    for name in ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"] {
        buffers.push(support::encode_weights(GgmlType::F32, &ffn));
        specs.push((
            String::from(name),
            [u64::from(embedding), u64::from(feed_forward)].into_iter().collect(),
            GgmlType::F32,
        ));
    }
    buffers.push(support::encode_weights(GgmlType::F32, &ffn));
    specs.push((
        String::from("blk.0.ffn_down.weight"),
        [u64::from(feed_forward), u64::from(embedding)].into_iter().collect(),
        GgmlType::F32,
    ));

    buffers.push(support::encode_weights(GgmlType::F32, &norm_values));
    specs.push((
        String::from("output_norm.weight"),
        [u64::from(embedding)].into_iter().collect(),
        GgmlType::F32,
    ));

    let output_values = vec![0.0f32; (embedding * vocab) as usize];
    buffers.push(support::encode_weights(GgmlType::F32, &output_values));
    specs.push((
        String::from("output.weight"),
        [u64::from(embedding), u64::from(vocab)].into_iter().collect(),
        GgmlType::F32,
    ));

    let tensors: Vec<TensorPayload<'_>> = specs
        .iter()
        .zip(buffers.iter())
        .map(|((name, dims, ggml_type), data)| TensorPayload {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: *ggml_type,
            data: data.as_slice(),
        })
        .collect();

    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String(architecture_name.to_string()),
        ),
        (
            format!("{architecture_name}.embedding_length"),
            MetadataValue::U32(embedding),
        ),
        (
            format!("{architecture_name}.feed_forward_length"),
            MetadataValue::U32(feed_forward),
        ),
        (
            format!("{architecture_name}.attention.head_count"),
            MetadataValue::U32(support::QUERY_HEADS),
        ),
        (
            format!("{architecture_name}.attention.head_count_kv"),
            MetadataValue::U32(support::KV_HEADS),
        ),
        (format!("{architecture_name}.block_count"), MetadataValue::U32(1)),
        (
            format!("{architecture_name}.rope.dimension_count"),
            MetadataValue::U32(support::HEAD_DIM),
        ),
    ];
    push_tokenizer_metadata(&mut metadata);

    let model = GgufModel {
        version: 3,
        metadata,
        tensors,
    };
    write_complete(&model).expect("writes a well-formed synthetic checkpoint")
}

/// A foreign architecture whose forward program declares one leaf this
/// crate's decode loop knows nothing about (`"aux_rows"`), fed every step
/// by [`Architecture::step_inputs`] alone. `hash_multiplier` is the knob
/// the two tests below vary to prove the gather is LIVE (changes the
/// output), not merely present (bound but never read).
struct StepInputArch {
    name: &'static str,
    hash_multiplier: u32,
}

static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
static OBSERVED_NEW_COUNTS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

fn reset_observations() {
    CALL_COUNT.store(0, Ordering::SeqCst);
    OBSERVED_NEW_COUNTS.lock().expect("test-only mutex, never poisoned").clear();
}

impl Architecture for StepInputArch {
    fn name(&self) -> &'static str {
        self.name
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        let mut bound = DenseArch.bind(parsed, file_bytes)?;
        let table = input_leaf(
            &mut bound.program,
            DType::Float32,
            vec![Extent::Static(TABLE_ROWS), Extent::Static(support::VOCAB)],
            "aux_table",
        );
        let rows = input_leaf(
            &mut bound.program,
            DType::Int32,
            vec![Extent::Symbolic(symbols::FIRST_FREE)],
            "aux_rows",
        );
        let gathered = embedding_lookup(&mut bound.program, table, rows);
        let summed = elementwise(
            &mut bound.program,
            DType::Float32,
            ScalarOp::Add,
            &[(bound.logits_root, "sd->sd"), (gathered, "sd->sd")],
        )
        .expect("logits_root and the aux gather are both [seq, vocab]");
        bound.logits_root = summed;
        Ok(bound)
    }

    fn step_inputs(&self, context: &StepInputContext<'_>, out: &mut Vec<StepInput>) {
        CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        OBSERVED_NEW_COUNTS
            .lock()
            .expect("test-only mutex, never poisoned")
            .push(context.new_count);

        let mut rows = Vec::with_capacity(context.new_count);
        for offset in 0..context.new_count {
            let position = context.new_start + offset;
            let token_id = context.all_token_ids[position];
            let row = token_id
                .wrapping_mul(self.hash_multiplier)
                .wrapping_add(position as u32)
                % TABLE_ROWS;
            rows.push(row as f32);
        }
        out.push(StepInput {
            name: "aux_rows",
            symbol: Some((symbols::FIRST_FREE, rows.len())),
            values: rows,
        });

        // A one-hot "spike" per row rather than a uniform per-row offset:
        // every dense weight this fixture writes is zero (`checkpoint_bytes`
        // never varies a real value), so an offset that is the SAME across
        // every vocab position would never change which position argmax
        // picks -- a spike at a row-dependent column is what actually makes
        // the gathered row observable in the sampled token id.
        let table_values: Vec<f32> = (0..TABLE_ROWS)
            .flat_map(|row| {
                let spike_column = (row * 17 + 3) % (support::VOCAB - 1);
                (0..support::VOCAB).map(move |column| {
                    if column == spike_column { 1000.0 } else { 0.0 }
                })
            })
            .collect();
        out.push(StepInput {
            name: "aux_table",
            symbol: None,
            values: table_values,
        });
    }
}

static LOW_MULTIPLIER: StepInputArch = StepInputArch {
    name: STEP_INPUT_ARCHITECTURE_NAME,
    hash_multiplier: 31,
};

static HIGH_MULTIPLIER: StepInputArch = StepInputArch {
    name: STEP_INPUT_ARCHITECTURE_NAME,
    hash_multiplier: 97,
};

fn registry_with(architecture: &'static StepInputArch) -> ArchitectureRegistry {
    let mut registry = ArchitectureRegistry::with_builtin();
    registry.register(architecture);
    registry
}

/// The happy path: `step_inputs` runs once per decode step (prefill
/// included), `new_count` matches the prompt length on step 0 and `1` on
/// every step after, and a run with a different `hash_multiplier`
/// produces different token ids -- proof the gathered leaf is actually
/// read, not just bound and ignored.
#[proxima::test]
async fn a_foreign_architecture_feeds_a_token_derived_leaf_each_step() {
    reset_observations();
    let file_bytes = checkpoint_bytes(STEP_INPUT_ARCHITECTURE_NAME);
    let parsed = parse_complete(&file_bytes).expect("parses the synthetic checkpoint");
    let registry = registry_with(&LOW_MULTIPLIER);

    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("loads through the foreign architecture's own bind");

    let prompt = "abc";

    let (low_ids, _text, _stopped) = Pipe::call(&model, (prompt.to_string(), 3))
        .await
        .expect("greedy decode runs through the extra aux leaf");
    assert_eq!(low_ids.len(), 3, "max_tokens=3 produces exactly three token ids");

    let observed = OBSERVED_NEW_COUNTS.lock().expect("test-only mutex, never poisoned").clone();
    assert_eq!(
        observed.len(),
        3,
        "step_inputs runs exactly once per decode step, prefill included"
    );
    assert!(
        observed[0] >= prompt.len(),
        "step 0 (prefill) evaluates the whole (bos-prefixed) prompt, not one token"
    );
    assert_eq!(observed[1], 1, "every step after prefill evaluates exactly one new token");
    assert_eq!(observed[2], 1, "every step after prefill evaluates exactly one new token");

    reset_observations();
    let registry_high = registry_with(&HIGH_MULTIPLIER);
    let model_high = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry_high)
        .expect("loads the same checkpoint through a differently-hashed architecture instance");
    let (high_ids, _text, _stopped) = Pipe::call(&model_high, (prompt.to_string(), 3))
        .await
        .expect("greedy decode runs through the extra aux leaf with a different hash constant");

    assert_ne!(
        low_ids, high_ids,
        "a different hash_multiplier changes which aux_table row is gathered, so the \
         token-derived leaf is live, not merely bound and ignored"
    );
}

/// Sad path: a program with an [`Architecture::step_inputs`] that supplies
/// nothing leaves `aux_rows` unbound -- [`InteropError::MissingStepInput`]
/// names it rather than the decode loop silently reading garbage or the
/// generic tensor error surfacing several layers down.
struct NoStepInputArch;

impl Architecture for NoStepInputArch {
    fn name(&self) -> &'static str {
        "acme-missing-step-input"
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        let mut bound = DenseArch.bind(parsed, file_bytes)?;
        let table = input_leaf(
            &mut bound.program,
            DType::Float32,
            vec![Extent::Static(TABLE_ROWS), Extent::Static(support::VOCAB)],
            "aux_table",
        );
        let rows = input_leaf(
            &mut bound.program,
            DType::Int32,
            vec![Extent::Symbolic(symbols::FIRST_FREE)],
            "aux_rows",
        );
        let gathered = embedding_lookup(&mut bound.program, table, rows);
        let summed = elementwise(
            &mut bound.program,
            DType::Float32,
            ScalarOp::Add,
            &[(bound.logits_root, "sd->sd"), (gathered, "sd->sd")],
        )
        .expect("logits_root and the aux gather are both [seq, vocab]");
        bound.logits_root = summed;
        Ok(bound)
        // deliberately no `step_inputs` override: the default no-op
    }
}

static NO_STEP_INPUT: NoStepInputArch = NoStepInputArch;

/// Sad path: an [`Architecture::step_inputs`] that names a RESERVED symbol
/// slot ([`symbols::KV_BOUND`]) instead of its own -- the decode loop must
/// reject this before it ever evaluates, with
/// [`InteropError::ReservedSymbolSlot`] naming the exact slot, rather than
/// silently overwriting `kv_bound_extent` and resolving `aux_rows`'
/// gather axis against the KV bucket capacity (the defect this whole test
/// module exists to catch, restated as a bind-time input instead of a
/// program author's mistake).
struct ReservedSlotArch;

impl Architecture for ReservedSlotArch {
    fn name(&self) -> &'static str {
        "acme-reserved-slot"
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        let mut bound = DenseArch.bind(parsed, file_bytes)?;
        let rows = input_leaf(
            &mut bound.program,
            DType::Int32,
            vec![Extent::Symbolic(symbols::KV_BOUND)],
            "aux_rows",
        );
        let _ = rows;
        Ok(bound)
    }

    fn step_inputs(&self, context: &StepInputContext<'_>, out: &mut Vec<StepInput>) {
        let rows: Vec<f32> = (0..context.new_count).map(|_| 0.0).collect();
        out.push(StepInput {
            name: "aux_rows",
            symbol: Some((symbols::KV_BOUND, rows.len())),
            values: rows,
        });
    }
}

static RESERVED_SLOT: ReservedSlotArch = ReservedSlotArch;

#[proxima::test]
async fn a_step_input_naming_a_reserved_symbol_slot_is_rejected() {
    let name = "acme-reserved-slot";
    let file_bytes = checkpoint_bytes(name);
    let parsed = parse_complete(&file_bytes).expect("parses the synthetic checkpoint");
    let mut registry = ArchitectureRegistry::with_builtin();
    registry.register(&RESERVED_SLOT);

    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("loads: bind itself never reads aux_rows");

    match Pipe::call(&model, ("a".to_string(), 1)).await {
        Err(InteropError::ReservedSymbolSlot { slot }) => {
            assert_eq!(
                slot,
                symbols::KV_BOUND,
                "must name the exact reserved slot the architecture collided with"
            );
        }
        Ok(_) => panic!("expected ReservedSymbolSlot: aux_rows named the builtin kv_bound slot"),
        Err(other) => panic!("expected ReservedSymbolSlot, got {other}"),
    }
}

#[proxima::test]
async fn a_program_leaf_with_no_step_inputs_override_reports_missing_step_input() {
    let name = "acme-missing-step-input";
    let file_bytes = checkpoint_bytes(name);
    let parsed = parse_complete(&file_bytes).expect("parses the synthetic checkpoint");
    let mut registry = ArchitectureRegistry::with_builtin();
    registry.register(&NO_STEP_INPUT);

    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("loads: bind itself never reads aux_rows/aux_table");

    match Pipe::call(&model, ("a".to_string(), 1)).await {
        Err(InteropError::MissingStepInput { name }) => {
            assert!(
                name == "aux_rows" || name == "aux_table",
                "MissingStepInput must name one of the two leaves this bind declared, got {name:?}"
            );
        }
        Ok(_) => panic!("expected MissingStepInput: neither aux leaf was ever fed"),
        Err(other) => panic!("expected MissingStepInput, got {other}"),
    }
}
