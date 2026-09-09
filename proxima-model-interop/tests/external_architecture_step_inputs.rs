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
    Architecture, ArchitectureRegistry, BoundProgram, BoundWeights, DenseArch, InteropError,
    LoadedModel, StepInput, StepInputContext, architecture_from_metadata, bind_dense, symbols,
};
use proxima_primitives::pipe::Pipe;
use proxima_tensor::spec::{elementwise, embedding_lookup, input_leaf, reduce};
use proxima_tensor::{DType, Extent, ReduceInit, ScalarOp};
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

        // `logits_root` (`DenseArch::bind`, which this architecture wraps)
        // is now always the LAST row only -- `mistral_cached_forward_program_with_experts_and_layer_taps`'s
        // own `last_row_only: true` default for the dense decode loop, see
        // that flag's own doc -- so `aux_rows` must supply exactly one
        // gathered row, matching the last of THIS step's own new
        // positions, not one per `new_count`.
        let position = context.new_start + context.new_count - 1;
        let token_id = context.all_token_ids[position];
        let row = token_id
            .wrapping_mul(self.hash_multiplier)
            .wrapping_add(position as u32)
            % TABLE_ROWS;
        out.push(StepInput {
            name: "aux_rows",
            symbol: Some((symbols::FIRST_FREE, 1)),
            values: vec![row as f32],
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

/// The defect [`crate::generate::LoadedModel::forward_node_values_on_backend`]
/// used to carry: it built a step's named inputs with hard-coded
/// `kv_cache.{layer}.{k_even,k_odd,v}` leaf names and never ran
/// [`Architecture::step_inputs`] at all, so tapping ANY node through
/// [`LoadedModel::forward_node_values`] on a foreign architecture that
/// declares its own leaves (`aux_rows`/`aux_table` here) failed with
/// [`InteropError::UnboundInputName`] the moment `evaluate` reached a leaf
/// only `step_inputs` could feed -- before this crate's decode loop ever
/// got a chance to prove the mechanism worked. This is the same
/// [`StepInputArch`] fixture the happy-path test above already decodes
/// with, tapping [`LoadedModel::hidden_root`] (a plain interior node, not
/// the `aux`-gathered `logits_root`) through the one-shot forward path
/// instead of the decode loop.
#[proxima::test]
async fn a_forward_tap_feeds_step_inputs_the_same_way_the_decode_loop_does() {
    reset_observations();
    let file_bytes = checkpoint_bytes(STEP_INPUT_ARCHITECTURE_NAME);
    let parsed = parse_complete(&file_bytes).expect("parses the synthetic checkpoint");
    let registry = registry_with(&LOW_MULTIPLIER);

    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("loads through the foreign architecture's own bind");

    let hidden_root = model
        .hidden_root()
        .expect("DenseArch::bind (this fixture's own delegate) always names a hidden root");
    let prompt = "abc";

    let values = model
        .forward_node_values(prompt, &[hidden_root])
        .expect(
            "a one-shot forward tap must assemble this step's inputs the SAME way the decode \
             loop does -- including running Architecture::step_inputs -- so a foreign \
             architecture's own leaves (aux_rows/aux_table here) are fed, not left unbound",
        );

    assert_eq!(values.len(), 1, "one value vector per requested node");

    let observed = OBSERVED_NEW_COUNTS.lock().expect("test-only mutex, never poisoned").clone();
    assert_eq!(
        observed.len(),
        1,
        "step_inputs runs exactly once for a one-shot forward tap, same as decode step 0"
    );
    assert!(
        observed[0] >= prompt.len(),
        "step_inputs' own new_count must cover the whole (bos-prefixed) prompt in one shot, \
         got {}",
        observed[0]
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

/// Sad path (the defect 08bf6025 introduced): a foreign [`Architecture`]
/// whose `logits_root` is left at the un-gathered `[new_count, vocab]`
/// shape -- no `lm_head_row` gather, unlike every builtin program
/// (`DenseArch`/`Qwen35Arch`) which always slices to the last row before
/// this crate's decode loop ever reads `logits_root`
/// ([`BoundProgram::logits_root`]'s own doc). Before the fix this silently
/// sampled row 0 (position 0's logits) every step; after the fix the
/// decode loop rejects it with [`InteropError::LogitsShapeMismatch`]
/// instead of guessing which row was meant.
struct MultiRowLogitsArch;

impl Architecture for MultiRowLogitsArch {
    fn name(&self) -> &'static str {
        "acme-multi-row-logits"
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        let architecture = architecture_from_metadata(parsed)?;
        let mut weights = BoundWeights::new(&[]);
        bind_dense(parsed, file_bytes, "token_embd.weight".to_string(), &mut weights)?;
        bind_dense(parsed, file_bytes, "output.weight".to_string(), &mut weights)?;

        let mut program = Vec::new();
        let ids = input_leaf(&mut program, DType::Int32, vec![Extent::Symbolic(0)], "ids");
        let table = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(architecture.vocab),
                Extent::Static(architecture.embedding),
            ],
            "token_embd.weight",
        );
        // `hidden` is `[new_count, embedding]` -- every new position, not
        // sliced to the last one -- so `logits` below stays `[new_count,
        // vocab]` all the way to `logits_root`, the exact shape a builtin
        // architecture never hands the decode loop (`last_row_only: true`
        // on every path this crate ships).
        let hidden = embedding_lookup(&mut program, table, ids);
        let lm_head = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(architecture.embedding),
                Extent::Static(architecture.vocab),
            ],
            "output.weight",
        );
        let logits_product = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(hidden, "sd->sdv"), (lm_head, "dv->sdv")],
        )
        .expect("hidden [seq, embedding] times output.weight [embedding, vocab] broadcasts");
        let logits = reduce(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            logits_product,
            "sdv->sdv",
            "sv->sdv",
        )?;

        Ok(BoundProgram {
            weights,
            architecture,
            program,
            logits_root: logits,
            hidden_root: None,
            layer_roots: Vec::new(),
            moe_sites: proxima_tensor::spec::MoeSites::default(),
        })
    }
}

static MULTI_ROW_LOGITS: MultiRowLogitsArch = MultiRowLogitsArch;

/// RED before the fix: this checkpoint's `general.architecture` resolves to
/// [`MultiRowLogitsArch`], whose `logits_root` evaluates to `[new_count,
/// vocab]` on the prefill step (`new_count == prompt length > 1`) -- the
/// pre-fix decode loop indexed `logits[..vocab_size]`, position 0's row,
/// and silently decoded from it. GREEN after: the same step returns
/// [`InteropError::LogitsShapeMismatch`] naming the exact row count found.
#[proxima::test]
async fn a_multi_row_logits_root_is_rejected_instead_of_silently_sampling_row_zero() {
    let name = "acme-multi-row-logits";
    let file_bytes = checkpoint_bytes(name);
    let parsed = parse_complete(&file_bytes).expect("parses the synthetic checkpoint");
    let mut registry = ArchitectureRegistry::with_builtin();
    registry.register(&MULTI_ROW_LOGITS);

    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("loads: bind itself never evaluates the program, so the shape defect is silent");

    // "abc" tokenizes to more than one id, so the prefill step's own
    // `new_count > 1` is what makes `logits_root`'s `[new_count, vocab]`
    // shape observably wrong instead of accidentally `[1, vocab]`.
    match Pipe::call(&model, ("abc".to_string(), 1)).await {
        Err(InteropError::LogitsShapeMismatch {
            expected_rows,
            found_rows,
            vocab,
        }) => {
            assert_eq!(expected_rows, 1, "the contract is exactly one row");
            assert!(
                found_rows > 1,
                "found_rows must report the actual multi-row buffer, got {found_rows}"
            );
            assert_eq!(vocab, support::VOCAB as usize, "vocab must be this checkpoint's own vocab");
        }
        Ok(_) => panic!(
            "expected LogitsShapeMismatch: logits_root never gathers to the last row, so decode \
             must not silently sample row 0"
        ),
        Err(other) => panic!("expected LogitsShapeMismatch, got {other}"),
    }
}

/// Positive control for the test above: a foreign architecture that DOES
/// gather to one row (wrapping [`DenseArch::bind`], unmodified) decodes
/// normally through the exact same shape check -- the check rejects a
/// multi-row buffer, not every foreign architecture.
struct OneRowLogitsArch;

impl Architecture for OneRowLogitsArch {
    fn name(&self) -> &'static str {
        "acme-one-row-logits"
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        DenseArch.bind(parsed, file_bytes)
    }
}

static ONE_ROW_LOGITS: OneRowLogitsArch = OneRowLogitsArch;

#[proxima::test]
async fn a_one_row_logits_root_still_decodes_through_the_same_shape_check() {
    let name = "acme-one-row-logits";
    let file_bytes = checkpoint_bytes(name);
    let parsed = parse_complete(&file_bytes).expect("parses the synthetic checkpoint");
    let mut registry = ArchitectureRegistry::with_builtin();
    registry.register(&ONE_ROW_LOGITS);

    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("loads through the foreign architecture's own bind");

    let (ids, _text, _stopped) = Pipe::call(&model, ("abc".to_string(), 2))
        .await
        .expect("a one-row logits_root decodes without LogitsShapeMismatch");
    assert_eq!(ids.len(), 2, "max_tokens=2 produces exactly two token ids");
}
