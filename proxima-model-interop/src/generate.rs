//! Reachable text generation: bind a checkpoint's weights once, then
//! generate text repeatedly against the bound weights without re-paying
//! the load cost.
//!
//! [`LoadedModel`] is the transform pipe (`In = (String, usize), Out =
//! (Vec<u32>, String, bool)` -- `proxima_primitives::pipe::Pipe`,
//! `proxima-primitives/src/pipe/primitives.rs:91-102`'s general form,
//! since neither `In` nor `Out` is `()`). [`LoadedModel::load`] is a plain
//! constructor, not a pipe: it pays the expensive one-time cost (mmap +
//! parse + bind 226 tensors, ~4 GB / ~120 ms prefault on the real
//! openchat-3.5 checkpoint -- `crate::bind::bind_all_weights`'s own doc)
//! and hands back a value that [`Pipe::call`] is then cheap to invoke many
//! times against, one call per generation request, without rebinding.
//! That two-step shape is the direct answer to "load once, generate
//! repeatedly": a caller holds one `LoadedModel` and calls it as many
//! times as it wants, exactly the way a caller holds one bound
//! `TcpListener` and accepts many connections from it.
//!
//! `call`'s body is synchronous CPU work wrapped in `async move { .. }`
//! with no internal `.await` -- the same shape `Pipe`'s own doc's
//! `Double`/`Always`/`Discard`/`Echo` examples use. It is still the right
//! trait: the algebra's whole point is that combinators (retry, tee,
//! rate-limit, ...) compose over `Pipe` regardless of whether a given
//! impl happens to yield control anywhere inside.
//!
//! # Stopping: the model's own signal, not just the caller's budget
//!
//! `Out`'s third field is `true` exactly when decoding stopped because the
//! model emitted its own end-of-sequence token, `false` when it stopped
//! because `max_tokens` ran out first -- the two are otherwise
//! indistinguishable to a caller (`generated_ids.len() < max_tokens` is
//! not proof of an early stop if `max_tokens` itself was small). A plain
//! `bool` earns this over a new enum because this checkpoint's own
//! metadata defines exactly one stopping condition to check, confirmed by
//! reading it rather than assumed: on the real openchat-3.5-1210 fixture
//! (`~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf`),
//! `tokenizer.ggml.eos_token_id = 32000`, which is *not* the SentencePiece
//! `</s>` (id 2) -- it is `<|end_of_turn|>`, a [`proxima_tokenizer::vocab::TokenType::Control`]
//! entry, and the same id OpenChat's own `tokenizer.chat_template` emits
//! between turns. There is no separate `tokenizer.ggml.eot_token_id` (or
//! similar) key on this fixture; the GGUF writer already folded the
//! turn-boundary marker into the one `eos_token_id` slot
//! [`proxima_tokenizer::Vocab::eos_token_id`] reads. So checking a single
//! id against [`Vocab::eos_token_id`] is this fixture's whole stopping
//! condition -- a `bool` carries it exactly; an enum would be modeling a
//! multi-token-family case this checkpoint does not have.
//!
//! The stop token itself is excluded from both the returned ids and the
//! returned text (never pushed onto `generated_ids` before the loop
//! breaks) -- symmetric exclusion, not just from decoded text, because a
//! caller who re-feeds `generated_ids` as a future prompt's tokens should
//! never see a turn-boundary marker reappear as if it were generated
//! content.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;
use core::future::Future;

use proxima_gguf::GgmlType;
use proxima_gguf::pipe::ParsedGguf;
use proxima_primitives::pipe::Pipe;
use proxima_tensor::cpu::{QuantizedBlock, evaluate_quantized_named_with_scratch};
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::{CachedLayerRoots, mistral_cached_forward_program};
use proxima_tokenizer::Vocab;

use crate::bind::{BoundWeights, ModelArchitecture, architecture_from_metadata, bind_all_weights};
use crate::error::InteropError;
use crate::serving::ServingConfig;
use crate::serving::apply_serving_config;

const ROPE_FREQ_BASE: f32 = 10_000.0;
const RMS_EPSILON: f32 = 1e-5;

/// A checkpoint's weights, bound once from a caller-owned byte view, plus
/// its compiled cached forward program -- everything a generation request
/// needs that does not change between requests. Borrows `file_bytes` for
/// `'file` rather than owning it, matching the rest of this crate's
/// sans-IO discipline (this crate never opens a file itself): the caller
/// keeps its own `mmap`/`Vec<u8>` alive for as long as it holds a
/// `LoadedModel` borrowed from it.
pub struct LoadedModel<'file> {
    weights: BoundWeights<'file>,
    architecture: ModelArchitecture,
    vocab: Vocab,
    program: Vec<Op>,
    logits_root: NodeId,
    cache_roots: Vec<CachedLayerRoots>,
}

impl<'file> LoadedModel<'file> {
    /// Binds every weight the cached forward program needs out of
    /// `parsed`/`file_bytes` ([`crate::bind::bind_all_weights`]), derives
    /// [`ModelArchitecture`] from `parsed`'s own metadata
    /// ([`crate::bind::architecture_from_metadata`]), builds the vocab
    /// from the same metadata, and compiles the cached forward program
    /// once. Pays the whole load cost; every [`Pipe::call`] after reuses
    /// the result.
    ///
    /// # Errors
    ///
    /// Whatever [`crate::bind::architecture_from_metadata`],
    /// [`proxima_tokenizer::gguf::vocab_from_metadata`], or
    /// [`proxima_tensor::spec::mistral_cached_forward_program`] can fail
    /// with.
    pub fn load(parsed: &ParsedGguf, file_bytes: &'file [u8]) -> Result<Self, InteropError> {
        let architecture = architecture_from_metadata(parsed)?;
        let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)?;
        let weights = bind_all_weights(parsed, file_bytes, &architecture);
        let (program, logits_root, cache_roots) = mistral_cached_forward_program(
            architecture.vocab,
            architecture.embedding,
            architecture.feed_forward,
            architecture.query_heads,
            architecture.kv_heads,
            architecture.head_dim,
            architecture.block_count,
        )?;
        Ok(Self {
            weights,
            architecture,
            vocab,
            program,
            logits_root,
            cache_roots,
        })
    }
}

/// This call's growable per-layer key/value cache -- `F32` only:
/// [`apply_serving_config`]'s own gate rejects any other
/// `kv_cache_key_quant`/`kv_cache_value_quant` before [`LoadedModel::call`]
/// ever reaches this loop, so there is no second precision for this type
/// to carry (contrast `bind.rs`'s own `real_openchat_file::LayerCache`,
/// which still probes the rejected `Q8_0` path directly against the
/// tensor seam that gate exists to keep unreachable here).
struct LayerCache {
    k_even: Vec<f32>,
    k_odd: Vec<f32>,
    v: Vec<f32>,
}

impl LayerCache {
    fn new() -> Self {
        Self { k_even: Vec::new(), k_odd: Vec::new(), v: Vec::new() }
    }

    fn append(&mut self, even: &[f32], odd: &[f32], value: &[f32]) {
        self.k_even.extend_from_slice(even);
        self.k_odd.extend_from_slice(odd);
        self.v.extend_from_slice(value);
    }

    fn named_blocks<'cache>(
        &'cache self,
        k_even_name: &'cache str,
        k_odd_name: &'cache str,
        v_name: &'cache str,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 3] {
        [
            (k_even_name, QuantizedBlock::Float32(self.k_even.as_slice())),
            (k_odd_name, QuantizedBlock::Float32(self.k_odd.as_slice())),
            (v_name, QuantizedBlock::Float32(self.v.as_slice())),
        ]
    }
}

/// Every per-call input the cached forward program needs beyond the model
/// weights and the growing key/value cache: `ids_f32`/RoPE `cos`/`sin` for
/// only the `new` positions this call introduces, at their true absolute
/// angle (`start_position`, not 0 -- a generated token's position is
/// `cached_len`, never the start of the sequence), plus the
/// reduce-broadcast `eps` vector sized to match.
struct PositionInputs {
    ids_f32: Vec<f32>,
    epsilon: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn build_position_inputs(new_ids: &[u32], start_position: usize, head_dim: u32) -> PositionInputs {
    let new_count = new_ids.len();
    let pairs = head_dim as usize / 2;
    let ids_f32: Vec<f32> = new_ids.iter().map(|&id| id as f32).collect();
    let epsilon = alloc::vec![RMS_EPSILON; new_count];

    let mut cos = alloc::vec![0.0f32; new_count * pairs];
    let mut sin = alloc::vec![0.0f32; new_count * pairs];
    for offset in 0..new_count {
        let position = (start_position + offset) as f32;
        for pair in 0..pairs {
            let theta = position * ROPE_FREQ_BASE.powf(-((2 * pair) as f32) / (head_dim as f32));
            cos[offset * pairs + pair] = theta.cos();
            sin[offset * pairs + pair] = theta.sin();
        }
    }

    PositionInputs { ids_f32, epsilon, cos, sin }
}

/// The fully-supported [`ServingConfig`]: every knob [`apply_serving_config`]
/// accepts today, `F32` key/value cache storage (the only precision the
/// cached-attention reduce's shared `kv_heads` axis can cross -- see
/// `bind.rs`'s own `q8_0_quantized_key_value_cache_cannot_cross_the_weight_matmul_quantized_seam`
/// for the gap this sidesteps by construction rather than by luck).
fn supported_serving_config() -> ServingConfig<'static> {
    ServingConfig {
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

impl<'file> Pipe for LoadedModel<'file> {
    type In = (String, usize);
    type Out = (Vec<u32>, String, bool);
    type Err = InteropError;

    fn call(&self, input: (String, usize)) -> impl Future<Output = Result<(Vec<u32>, String, bool), InteropError>> {
        async move {
            let (prompt, max_tokens) = input;
            self.generate(&prompt, max_tokens)
        }
    }
}

/// The decode loop's termination policy, isolated from the forward pass
/// that produces each token: pulls up to `max_tokens` ids out of
/// `produce_next_token` (one call per step, `0`-indexed), appending each
/// to the result unless it is `vocab`'s end-of-sequence id, in which case
/// decoding stops immediately without appending that id. Returns the
/// accumulated ids plus whether the stop was the model's own signal
/// (`true`) rather than the budget running out (`false`).
///
/// Factored out so this policy -- the exact defect this module's
/// [`LoadedModel::generate`] fixed (a loop with no termination condition
/// besides the budget) -- is provable against a scripted token source,
/// without paying for a real forward pass per test.
fn decode_until_stop_or_budget(
    vocab: &Vocab,
    max_tokens: usize,
    mut produce_next_token: impl FnMut(usize) -> Result<u32, InteropError>,
) -> Result<(Vec<u32>, bool), InteropError> {
    let mut generated_ids = Vec::with_capacity(max_tokens);
    let mut stopped_by_eos = false;
    for step in 0..max_tokens {
        let token_id = produce_next_token(step)?;
        if vocab.eos_token_id() == Some(token_id) {
            stopped_by_eos = true;
            break;
        }
        generated_ids.push(token_id);
    }
    Ok((generated_ids, stopped_by_eos))
}

impl<'file> LoadedModel<'file> {
    /// The greedy decode loop itself: `max_tokens` steps, each one call
    /// into `evaluate_quantized_named_with_scratch` against `new_positions
    /// == 1` after the first step (`new_positions == prompt_length` on the
    /// first), growing [`LayerCache`] by one call's worth of positions
    /// every step instead of re-running the whole sequence from scratch --
    /// stopping early the moment the model emits its own end-of-sequence
    /// id (see this module's doc for what that id is on the real
    /// checkpoint), never running past `max_tokens` regardless.
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<(Vec<u32>, String, bool), InteropError> {
        let serving_config = supported_serving_config();
        let ids = proxima_tokenizer::encode_with_bos_eos(prompt, &self.vocab, true, false)?;

        let block_count = self.architecture.block_count as usize;
        let kv_cache_names: Vec<(String, String, String)> = (0..block_count)
            .map(|layer| {
                (
                    alloc::format!("kv_cache.{layer}.k_even"),
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    alloc::format!("kv_cache.{layer}.v"),
                )
            })
            .collect();
        let mut layer_caches: Vec<LayerCache> = (0..block_count).map(|_| LayerCache::new()).collect();

        let mut cached_len = 0usize;
        let mut next_ids = ids;
        let mut free_buffers: Vec<Vec<f32>> = Vec::new();
        let mut validated_weight_nodes: Option<BTreeSet<NodeId>> = None;
        let vocab_size = self.architecture.vocab as usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(&self.vocab, max_tokens, |_step| {
            let new_count = next_ids.len();
            apply_serving_config(&serving_config, cached_len + new_count);
            let inputs = build_position_inputs(&next_ids, cached_len, self.architecture.head_dim);

            let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
                self.weights.owned.len() + self.weights.packed.len() + 3 + layer_caches.len() * 3,
            );
            named_blocks.push(("ids", QuantizedBlock::Float32(inputs.ids_f32.as_slice())));
            for (name, data) in &self.weights.owned {
                named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
            }
            for (name, block) in &self.weights.packed {
                named_blocks.push((name.as_str(), *block));
            }
            named_blocks.push(("eps", QuantizedBlock::Float32(inputs.epsilon.as_slice())));
            named_blocks.push(("rope_cos", QuantizedBlock::Float32(inputs.cos.as_slice())));
            named_blocks.push(("rope_sin", QuantizedBlock::Float32(inputs.sin.as_slice())));
            for (layer, (k_even_name, k_odd_name, v_name)) in kv_cache_names.iter().enumerate() {
                named_blocks.extend(layer_caches[layer].named_blocks(k_even_name, k_odd_name, v_name));
            }

            let symbols = [new_count as u64, cached_len as u64];
            let mut roots: Vec<NodeId> = Vec::with_capacity(1 + self.cache_roots.len() * 3);
            roots.push(self.logits_root);
            for (even, odd, value) in &self.cache_roots {
                roots.push(*even);
                roots.push(*odd);
                roots.push(*value);
            }

            let evaluated = evaluate_quantized_named_with_scratch(
                &self.program,
                &symbols,
                &named_blocks,
                &roots,
                &mut free_buffers,
                &mut validated_weight_nodes,
            )?;

            for (layer, (even, odd, value)) in self.cache_roots.iter().enumerate() {
                let (even_data, _) = evaluated.get(*even).ok_or(InteropError::MissingEvaluatedNode { node: *even })?;
                let (odd_data, _) = evaluated.get(*odd).ok_or(InteropError::MissingEvaluatedNode { node: *odd })?;
                let (value_data, _) = evaluated.get(*value).ok_or(InteropError::MissingEvaluatedNode { node: *value })?;
                layer_caches[layer].append(even_data, odd_data, value_data);
            }
            cached_len += new_count;

            let (logits, _shape) = evaluated
                .get(self.logits_root)
                .ok_or(InteropError::MissingEvaluatedNode { node: self.logits_root })?;
            let last_position = &logits[(new_count - 1) * vocab_size..new_count * vocab_size];

            let token_id = proxima_tokenizer::greedy_pick(last_position).ok_or(InteropError::EmptyLogits)?;
            next_ids = alloc::vec![token_id];
            Ok(token_id)
        })?;

        let text = proxima_tokenizer::decode(&generated_ids, &self.vocab)?;
        Ok((generated_ids, text, stopped_by_eos))
    }
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::String;
    use alloc::vec::Vec;

    use proxima_tokenizer::Vocab;

    use super::decode_until_stop_or_budget;

    /// A minimal valid [`Vocab`] (every byte-level BPE vocab needs all 256
    /// base-byte tokens present or [`Vocab::new`] rejects it) plus one
    /// extra token at id `256` marked as this vocab's end-of-sequence id --
    /// enough to exercise [`decode_until_stop_or_budget`]'s stopping policy
    /// without a real checkpoint. Spells the base-byte alphabet as the
    /// SentencePiece `"<0xXX>"` fallback form directly (not through
    /// `proxima_tokenizer`'s private `byte_to_char`) since that spelling is
    /// public knowledge, not an internal detail this test needs to reach
    /// into the crate for.
    fn vocab_with_eos(eos_id: u32) -> Vocab {
        let mut tokens: Vec<String> = (0..=255u8).map(|byte| alloc::format!("<0x{byte:02X}>")).collect();
        tokens.push(String::from("<eos-marker>"));
        Vocab::new(tokens, &[], Some(0), Some(eos_id), None).expect("minimal vocab builds")
    }

    /// The defect this module exists to fix, proved directly: a scripted
    /// token source that would emit `999` on a 4th call never gets asked
    /// for it, because the 3rd call's token (`32000`, this vocab's eos id)
    /// stops the loop first. Also proves the eos id itself never lands in
    /// `generated_ids`.
    #[test]
    fn stops_early_when_eos_is_produced_and_excludes_it_from_ids() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [10u32, 20, 32_000, 999];
        let mut calls = 0usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(&vocab, 4, |step| {
            calls += 1;
            Ok(scripted_tokens[step])
        })
        .expect("scripted token source never errors");

        assert_eq!(generated_ids, alloc::vec![10, 20], "eos id must not be appended to the generated ids");
        assert!(stopped_by_eos, "must report that the stop was the model's own eos signal");
        assert_eq!(calls, 3, "must not pull a 4th token once eos is seen on the 3rd");
    }

    /// The other half of the invariant: when the model never emits eos,
    /// decoding runs the full budget and reports that distinctly from an
    /// eos stop -- `stopped_by_eos == false` is the caller's only way to
    /// tell "ran out of budget" apart from "the model finished".
    #[test]
    fn exhausts_the_budget_and_reports_it_distinctly_from_an_eos_stop() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [10u32, 20, 30, 40];

        let (generated_ids, stopped_by_eos) =
            decode_until_stop_or_budget(&vocab, scripted_tokens.len(), |step| Ok(scripted_tokens[step]))
                .expect("scripted token source never errors");

        assert_eq!(generated_ids, alloc::vec![10, 20, 30, 40], "every scripted token is a real id, none is eos");
        assert!(!stopped_by_eos, "budget exhaustion must not be reported as an eos stop");
        assert_eq!(generated_ids.len(), scripted_tokens.len(), "budget exhaustion still runs every requested step");
    }

    /// Degenerate control: if the eos comparison were broken (e.g. always
    /// `false`), this test's scripted eos-first source would run the full
    /// budget instead of stopping on step 1 -- confirming the two tests
    /// above are not passing by coincidence of never actually comparing
    /// against `vocab.eos_token_id()`.
    #[test]
    fn stops_on_the_very_first_token_when_it_is_eos() {
        let vocab = vocab_with_eos(32_000);
        let mut calls = 0usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(&vocab, 10, |_step| {
            calls += 1;
            Ok(32_000)
        })
        .expect("scripted token source never errors");

        assert!(generated_ids.is_empty(), "an immediate eos must produce zero generated ids");
        assert!(stopped_by_eos);
        assert_eq!(calls, 1, "must stop after exactly one call, not run toward the budget of 10");
    }
}
