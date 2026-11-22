//! HOBBIT: per-layer mixed precision chosen by measured sensitivity.
//!
//! The pass is model-agnostic. It derives layer prefixes from the checkpoint
//! directory (`blk.<n>.`), recodes exactly one layer at a time through
//! [`crate::ServingConfig::weight_precision`], and measures the output delta
//! against the unrecoded run on a fixed calibration set: top-1 agreement and
//! KL(reference || candidate) over the full vocabulary at every
//! teacher-forced position. The ranking is deterministic; repeating it on
//! the same checkpoint and prompts reproduces [`HobbitSensitivityReport::ranking_hash`].
//!
//! [`allocate`] then spends a byte budget on the least sensitive layers,
//! emitting one `Prefix` rule per chosen layer. A budget that cannot be met
//! is a typed refusal naming the shortfall, never an overspend.
//!
//! The candidate codec is a [`proxima_numeric::NumericFormat`]; the ggml
//! wire type and the encoder check come from [`proxima_numeric`] and
//! [`crate::target_has_encoder`]. This module adds no numeric type: the
//! quality arithmetic is `f64` over the model's own logits, because
//! `proxima-numeric` owns formats and conversion, not softmax/KL.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::collections::BTreeSet;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::types::GgmlType;
use proxima_numeric::NumericFormat;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::bind::codec_to_ggml_type;
use crate::error::InteropError;
use crate::generate::LoadedModel;
use crate::serving::{NamePattern, ServingConfig, WeightPrecisionRule};

/// A HOBBIT measurement or allocation refusal. Every variant names the input
/// that could not be met, so a caller never has to re-derive a number.
#[derive(Debug, Error)]
pub enum HobbitError {
    #[error("checkpoint has no blk.<n>. tensors, so there is no layer to rank")]
    LayerlessCheckpoint,
    #[error("calibration prompt {index} did not tokenize: {source}")]
    Tokenize {
        index: usize,
        source: proxima_tokenizer::TokenizerError,
    },
    #[error("no layer recoded to {codec}: every candidate tensor already holds it")]
    NoRecodedLayer { codec: String },
    #[error(
        "sensitivity did not separate the layers beyond noise (max mean KL {max_mean_kl} over {layers} layers)"
    )]
    NoSensitivitySignal { max_mean_kl: f64, layers: usize },
    #[error("hobbit.budget_bytes is unset; a budget is required before applying a rule")]
    BudgetUnset,
    #[error("only {available} layers can be recoded, but hobbit.min_layers is {requested}")]
    InsufficientCandidateLayers { requested: usize, available: usize },
    #[error(
        "budget {budget} bytes cannot be met: the minimum after recoding every candidate is {minimum} bytes (shortfall {shortfall})"
    )]
    BudgetInfeasible {
        budget: u64,
        minimum: u64,
        shortfall: u64,
    },
    #[error("hobbit.quality sets neither max_mean_kl nor min_top1_agreement; no bound, no apply")]
    QualityBoundMissing,
    #[error(
        "quality bound exceeded: mean_kl {mean_kl} (max {max_mean_kl:?}), top1_agreement {top1_agreement} (min {min_top1_agreement:?})"
    )]
    QualityBoundExceeded {
        mean_kl: f64,
        top1_agreement: f64,
        max_mean_kl: Option<f64>,
        min_top1_agreement: Option<f64>,
    },
    #[error("reference has {reference_rows} logits rows, candidate has {candidate_rows}")]
    QualityShapeMismatch {
        reference_rows: usize,
        candidate_rows: usize,
    },
    #[error(
        "logits row {row} has {reference_vocab} entries in the reference and {candidate_vocab} in the candidate"
    )]
    VocabularyMismatch {
        row: usize,
        reference_vocab: usize,
        candidate_vocab: usize,
    },
    #[error(transparent)]
    Model(#[from] InteropError),
}

/// One layer's measured sensitivity and the byte win a recode of that layer
/// produces. `mean_kl` is the ranking key; a higher value means the layer's
/// output moved more when it was recoded, so it is more sensitive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HobbitLayerSensitivity {
    /// The checkpoint's own prefix, e.g. `blk.7.`. Never a hard-coded list.
    pub layer: String,
    pub mean_kl: f64,
    pub top1_agreement: f64,
    pub positions: usize,
    /// Bytes of this layer's recoded tensors after the candidate encode.
    pub recoded_bytes: u64,
    /// Bytes those same tensors occupied on disk before the recode.
    pub base_bytes: u64,
    /// `base_bytes - recoded_bytes`; zero when no tensor in the layer
    /// recoded or the candidate did not shrink it.
    pub saved_bytes: u64,
    /// The exact tensor names this layer's measurement recoded, sorted.
    /// [`allocate`] emits one `Exact` rule per entry, so a norm or a
    /// rank-3 expert stack in the same layer is never swept in.
    pub tensors: Vec<String>,
}

/// The whole deterministic sensitivity result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HobbitSensitivityReport {
    pub candidate_codec: String,
    pub prompts: usize,
    pub positions: usize,
    /// Total bytes of every tensor in the checkpoint's data section.
    pub base_bytes: u64,
    /// Layers in ranking order: least sensitive first, layer name as the
    /// deterministic tie-break.
    pub layers: Vec<HobbitLayerSensitivity>,
    /// FNV-1a over the ordered ranking; equal across two runs on the same
    /// checkpoint and prompts.
    pub ranking_hash: u64,
}

/// The measured difference between an unrecoded and a recoded run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HobbitQualityDelta {
    pub positions: usize,
    pub top1_agreement: f64,
    pub mean_kl: f64,
}

/// One emitted rule: an exact tensor name and the candidate codec name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HobbitRule {
    pub tensor: String,
    pub codec: String,
}

/// A deterministic budget allocation. `assigned_bytes` is the checkpoint's
/// total tensor bytes after the chosen recodes; it is always `<= budget`
/// on success.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HobbitAllocation {
    pub budget_bytes: u64,
    pub base_bytes: u64,
    pub assigned_bytes: u64,
    pub recoded_bytes: u64,
    pub saved_bytes: u64,
    pub shortfall: u64,
    pub rules: Vec<HobbitRule>,
}

impl HobbitAllocation {
    /// Lowers the emitted rules to the bind path's own
    /// [`WeightPrecisionRule`]s. The patterns borrow this allocation's
    /// strings, so the allocation must outlive the config that carries them.
    ///
    /// # Errors
    ///
    /// [`InteropError::UnsupportedCodec`] / [`InteropError::UnsupportedWeightPrecisionTarget`]
    /// when a stored codec name no longer maps to an encodable target.
    pub fn weight_precision_rules(&self) -> Result<Vec<WeightPrecisionRule<'_>>, InteropError> {
        self.rules
            .iter()
            .map(|rule| {
                let format = NumericFormat::from_name(&rule.codec).ok_or(
                    InteropError::UnsupportedCodec {
                        codec: "hobbit candidate codec",
                    },
                )?;
                Ok(WeightPrecisionRule {
                    pattern: NamePattern::Exact(rule.tensor.as_str()),
                    target: codec_to_ggml_type(format)?,
                })
            })
            .collect()
    }
}

/// Every layer prefix the checkpoint's own tensor directory names, sorted.
/// The parser is structural (`blk.` + an integer + `.`), not a list.
#[must_use]
pub fn layer_prefixes(parsed: &ParsedGguf) -> Vec<String> {
    let mut prefixes = BTreeSet::new();
    for tensor in &parsed.tensors {
        let Some(rest) = tensor.name.strip_prefix("blk.") else {
            continue;
        };
        let Some((index, _)) = rest.split_once('.') else {
            continue;
        };
        if index.parse::<u32>().is_ok() {
            prefixes.insert(format!("blk.{index}."));
        }
    }
    prefixes.into_iter().collect()
}

/// Total data-section bytes of every tensor the directory names.
#[must_use]
pub fn checkpoint_tensor_bytes(parsed: &ParsedGguf) -> u64 {
    parsed
        .tensors
        .iter()
        .filter_map(proxima_gguf::tensor::TensorInfo::nbytes)
        .sum()
}

/// The rank-2 tensors of one layer that a recode can shrink: the program
/// binds these through `bind_dense_as`/`bind_matmul_weight_as` (rank-1 norms
/// and biases stay full precision, rank-3 expert stacks are the named
/// residual of the recode primitive), and a tensor already at `target` has
/// nothing to do.
#[must_use]
pub fn recode_candidate_tensors<'a>(
    parsed: &'a ParsedGguf,
    prefix: &str,
    target: GgmlType,
) -> Vec<&'a proxima_gguf::tensor::TensorInfo> {
    parsed
        .tensors
        .iter()
        .filter(|tensor| {
            tensor.name.starts_with(prefix)
                && tensor.dims.len() == 2
                && tensor.ggml_type != target
                && tensor.nbytes_as(target).is_some()
        })
        .collect()
}

fn model_tensor_names_that_recoded(
    prefix: &str,
    recoded_bytes: u64,
    recoded: &[(String, usize)],
) -> Vec<String> {
    if recoded_bytes == 0 {
        return Vec::new();
    }
    recoded
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .map(|(name, _)| {
            name.rsplit_once('@')
                .map_or(name.as_str(), |(base, _)| base)
                .to_string()
        })
        .collect()
}

/// Tokenizes the configured calibration prompts with the checkpoint's own
/// vocabulary. `max_tokens` truncates each id sequence without splitting a
/// multi-token prompt into nothing (a one-token prompt drops to one).
///
/// # Errors
///
/// [`HobbitError::Tokenize`] naming the prompt index.
pub fn calibration_ids(
    parsed: &ParsedGguf,
    prompts: &[String],
    max_tokens: usize,
) -> Result<Vec<Vec<u32>>, HobbitError> {
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)
        .map_err(|source| HobbitError::Tokenize { index: 0, source })?;
    let add_bos = vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some());
    prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            let mut ids = proxima_tokenizer::encode_with_bos_eos(prompt, &vocab, add_bos, false)
                .map_err(|source| HobbitError::Tokenize { index, source })?;
            if max_tokens > 0 && ids.len() > max_tokens {
                ids.truncate(max_tokens.max(2));
            }
            Ok(ids)
        })
        .collect()
}

/// Teacher-forced logits for `ids`: one row per next-token position, with
/// the first token as the prompt and the remainder forced. Comparing two
/// models' rows is the measured output delta HOBBIT ranks on.
fn logits_on(
    model: &LoadedModel<'_>,
    ids: &[u32],
    config: &ServingConfig<'_>,
) -> Result<Vec<Vec<f32>>, InteropError> {
    if ids.len() < 2 {
        return Ok(Vec::new());
    }
    model.teacher_forced_logits_from_ids(&ids[..1], &ids[1..], config)
}

/// Top-1 agreement and mean KL(reference || candidate) over aligned rows.
///
/// # Errors
///
/// [`HobbitError::QualityShapeMismatch`] or
/// [`HobbitError::VocabularyMismatch`] when the two runs did not produce the
/// same rows.
pub fn quality_delta(
    reference: &[Vec<f32>],
    candidate: &[Vec<f32>],
) -> Result<HobbitQualityDelta, HobbitError> {
    if reference.len() != candidate.len() {
        return Err(HobbitError::QualityShapeMismatch {
            reference_rows: reference.len(),
            candidate_rows: candidate.len(),
        });
    }
    if reference.is_empty() {
        return Ok(HobbitQualityDelta {
            positions: 0,
            top1_agreement: 1.0,
            mean_kl: 0.0,
        });
    }
    let mut agree = 0usize;
    let mut kl_sum = 0.0f64;
    for (row, (reference_row, candidate_row)) in reference.iter().zip(candidate.iter()).enumerate()
    {
        if reference_row.len() != candidate_row.len() {
            return Err(HobbitError::VocabularyMismatch {
                row,
                reference_vocab: reference_row.len(),
                candidate_vocab: candidate_row.len(),
            });
        }
        if argmax(reference_row) == argmax(candidate_row) {
            agree += 1;
        }
        kl_sum += kl_row(reference_row, candidate_row);
    }
    Ok(HobbitQualityDelta {
        positions: reference.len(),
        top1_agreement: agree as f64 / reference.len() as f64,
        mean_kl: kl_sum / reference.len() as f64,
    })
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in row.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    best
}

/// `KL(p || q)` for one logits row, computed in `f64` with a stable
/// log-sum-exp. `0.0` for identical rows.
fn kl_row(reference: &[f32], candidate: &[f32]) -> f64 {
    let reference_log = log_softmax(reference);
    let candidate_log = log_softmax(candidate);
    let mut kl = 0.0f64;
    for (log_p, log_q) in reference_log.iter().zip(candidate_log.iter()) {
        let probability = log_p.exp();
        kl += probability * (log_p - log_q);
    }
    kl.max(0.0)
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum = row
        .iter()
        .map(|value| ((*value as f64) - max).exp())
        .sum::<f64>();
    let log_sum = sum.ln();
    row.iter()
        .map(|value| (*value as f64) - max - log_sum)
        .collect()
}

/// FNV-1a over the canonical ranking: candidate codec, positions, then each
/// layer's name and the bit patterns of its two metrics.
#[must_use]
pub fn ranking_hash(
    candidate_codec: &str,
    positions: usize,
    layers: &[HobbitLayerSensitivity],
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(candidate_codec.as_bytes());
    feed(&(positions as u64).to_le_bytes());
    for layer in layers {
        feed(layer.layer.as_bytes());
        feed(&layer.mean_kl.to_bits().to_le_bytes());
        feed(&layer.top1_agreement.to_bits().to_le_bytes());
    }
    hash
}

/// Measures every layer's sensitivity to `candidate` on the calibration set.
///
/// `base_config` supplies every decode knob except the rules; the pass sets
/// only [`ServingConfig::weight_precision`] on a copy. Layer prefixes come
/// from [`layer_prefixes`], so a dense checkpoint and a mixture-of-experts
/// checkpoint take the identical path.
///
/// # Errors
///
/// [`HobbitError::LayerlessCheckpoint`], the tokenizer refusal, a model load
/// failure, or [`HobbitError::NoSensitivitySignal`] when every recoded layer
/// produced the same output as the unrecoded run.
pub fn measure_sensitivity(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    prompts: &[String],
    candidate: NumericFormat,
    max_calibration_tokens: usize,
    base_config: &ServingConfig<'_>,
) -> Result<HobbitSensitivityReport, HobbitError> {
    let prefixes = layer_prefixes(parsed);
    if prefixes.is_empty() {
        return Err(HobbitError::LayerlessCheckpoint);
    }
    measure_sensitivity_for_layers(
        parsed,
        file_bytes,
        prompts,
        candidate,
        max_calibration_tokens,
        base_config,
        &prefixes,
    )
}

/// [`measure_sensitivity`] over an explicit prefix list. The production pass
/// passes every prefix [`layer_prefixes`] derives; a bounded test or a
/// partial sweep passes a subset. Each prefix is measured exactly the same
/// way, so a subset's ranking is a prefix-stable slice of the full one.
///
/// # Errors
///
/// As [`measure_sensitivity`], plus [`HobbitError::LayerlessCheckpoint`]
/// when `prefixes` is empty.
pub fn measure_sensitivity_for_layers(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    prompts: &[String],
    candidate: NumericFormat,
    max_calibration_tokens: usize,
    base_config: &ServingConfig<'_>,
    prefixes: &[String],
) -> Result<HobbitSensitivityReport, HobbitError> {
    if prefixes.is_empty() {
        return Err(HobbitError::LayerlessCheckpoint);
    }
    let target = codec_to_ggml_type(candidate)?;
    let ids = calibration_ids(parsed, prompts, max_calibration_tokens)?;

    let baseline = {
        let model = LoadedModel::load_with_serving(parsed, file_bytes, base_config)?;
        let mut rows = Vec::new();
        for prompt in &ids {
            rows.extend(logits_on(&model, prompt, base_config)?);
        }
        rows
    };
    let positions = baseline.len();

    let mut layers = Vec::with_capacity(prefixes.len());
    for prefix in prefixes.iter().cloned() {
        let rules = recode_candidate_tensors(parsed, &prefix, target)
            .into_iter()
            .map(|tensor| WeightPrecisionRule {
                pattern: NamePattern::Exact(tensor.name.as_str()),
                target,
            })
            .collect::<Vec<_>>();
        let config = base_config.with_weight_precision(&rules);
        let (rows, recoded_bytes, base_bytes, tensors) = {
            let model = LoadedModel::load_with_serving(parsed, file_bytes, &config)?;
            let mut rows = Vec::new();
            for prompt in &ids {
                rows.extend(logits_on(&model, prompt, &config)?);
            }
            let recoded = model.recoded_weights();
            let recoded_bytes = recoded
                .iter()
                .filter(|(name, _)| name.starts_with(prefix.as_str()))
                .map(|(_, bytes)| *bytes as u64)
                .sum::<u64>();
            let base_bytes = recoded
                .iter()
                .filter(|(name, _)| name.starts_with(prefix.as_str()))
                .filter_map(|(name, _)| {
                    let base = name
                        .rsplit_once('@')
                        .map_or(name.as_str(), |(base, _)| base);
                    parsed
                        .tensors
                        .iter()
                        .find(|tensor| tensor.name == base)
                        .and_then(proxima_gguf::tensor::TensorInfo::nbytes)
                })
                .sum::<u64>();
            let mut tensors = model_tensor_names_that_recoded(&prefix, recoded_bytes, &recoded);
            tensors.sort();
            (rows, recoded_bytes, base_bytes, tensors)
        };
        let delta = quality_delta(&baseline, &rows)?;
        layers.push(HobbitLayerSensitivity {
            layer: prefix,
            mean_kl: delta.mean_kl,
            top1_agreement: delta.top1_agreement,
            positions: delta.positions,
            recoded_bytes,
            base_bytes,
            saved_bytes: base_bytes.saturating_sub(recoded_bytes),
            tensors,
        });
    }

    layers.sort_by(|left, right| {
        left.mean_kl
            .total_cmp(&right.mean_kl)
            .then_with(|| left.layer.cmp(&right.layer))
    });
    let max_mean_kl = layers
        .iter()
        .map(|layer| layer.mean_kl)
        .fold(0.0f64, f64::max);
    let any_recoded = layers.iter().any(|layer| layer.recoded_bytes > 0);
    if !any_recoded {
        return Err(HobbitError::NoRecodedLayer {
            codec: candidate.name().unwrap_or("unknown").to_string(),
        });
    }
    if max_mean_kl <= f64::EPSILON {
        return Err(HobbitError::NoSensitivitySignal {
            max_mean_kl,
            layers: layers.len(),
        });
    }
    let codec_name = candidate.name().unwrap_or("unknown").to_string();
    let hash = ranking_hash(&codec_name, positions, &layers);
    Ok(HobbitSensitivityReport {
        candidate_codec: codec_name,
        prompts: ids.len(),
        positions,
        base_bytes: checkpoint_tensor_bytes(parsed),
        layers,
        ranking_hash: hash,
    })
}

/// Deterministic budget allocation: spend on the least sensitive layers
/// first, honor `min_layers` even when the budget is already met, and refuse
/// with the shortfall when even recoding every candidate cannot fit.
///
/// # Errors
///
/// [`HobbitError::BudgetUnset`], [`HobbitError::InsufficientCandidateLayers`]
/// or [`HobbitError::BudgetInfeasible`].
pub fn allocate(
    report: &HobbitSensitivityReport,
    budget_bytes: Option<u64>,
    min_layers: usize,
) -> Result<HobbitAllocation, HobbitError> {
    let budget = budget_bytes.ok_or(HobbitError::BudgetUnset)?;
    let mut candidates: Vec<&HobbitLayerSensitivity> = report
        .layers
        .iter()
        .filter(|layer| layer.saved_bytes > 0)
        .collect();
    candidates.sort_by(|left, right| {
        left.mean_kl
            .total_cmp(&right.mean_kl)
            .then_with(|| left.layer.cmp(&right.layer))
    });
    if candidates.len() < min_layers {
        return Err(HobbitError::InsufficientCandidateLayers {
            requested: min_layers,
            available: candidates.len(),
        });
    }

    let total_saved = candidates
        .iter()
        .map(|layer| layer.saved_bytes)
        .sum::<u64>();
    let minimum = report.base_bytes.saturating_sub(total_saved);
    if minimum > budget {
        return Err(HobbitError::BudgetInfeasible {
            budget,
            minimum,
            shortfall: minimum - budget,
        });
    }

    let mut chosen: Vec<&HobbitLayerSensitivity> =
        candidates.iter().take(min_layers).copied().collect();
    let mut assigned = report
        .base_bytes
        .saturating_sub(chosen.iter().map(|layer| layer.saved_bytes).sum::<u64>());
    for layer in candidates.iter().skip(min_layers) {
        if assigned <= budget {
            break;
        }
        chosen.push(*layer);
        assigned = assigned.saturating_sub(layer.saved_bytes);
    }

    chosen.sort_by(|left, right| left.layer.cmp(&right.layer));
    let mut tensors = chosen
        .iter()
        .flat_map(|layer| layer.tensors.iter().cloned())
        .collect::<Vec<_>>();
    tensors.sort();
    tensors.dedup();
    let rules = tensors
        .into_iter()
        .map(|tensor| HobbitRule {
            tensor,
            codec: report.candidate_codec.clone(),
        })
        .collect::<Vec<_>>();
    let recoded_bytes = chosen.iter().map(|layer| layer.recoded_bytes).sum();
    let saved_bytes = chosen.iter().map(|layer| layer.saved_bytes).sum();
    Ok(HobbitAllocation {
        budget_bytes: budget,
        base_bytes: report.base_bytes,
        assigned_bytes: assigned,
        recoded_bytes,
        saved_bytes,
        shortfall: 0,
        rules,
    })
}

/// Applies the configured quality bound to a measured delta. A missing bound
/// is a refusal, never an implicit pass.
///
/// # Errors
///
/// [`HobbitError::QualityBoundMissing`] or
/// [`HobbitError::QualityBoundExceeded`].
pub fn check_quality_bound(
    delta: &HobbitQualityDelta,
    bound: Option<(Option<f64>, Option<f64>)>,
) -> Result<(), HobbitError> {
    let (max_mean_kl, min_top1_agreement) = bound.ok_or(HobbitError::QualityBoundMissing)?;
    let kl_holds = max_mean_kl.is_none_or(|max| delta.mean_kl <= max);
    let top1_holds = min_top1_agreement.is_none_or(|min| delta.top1_agreement >= min);
    if kl_holds && top1_holds {
        Ok(())
    } else {
        Err(HobbitError::QualityBoundExceeded {
            mean_kl: delta.mean_kl,
            top1_agreement: delta.top1_agreement,
            max_mean_kl,
            min_top1_agreement,
        })
    }
}

#[cfg(test)]
mod tests;
