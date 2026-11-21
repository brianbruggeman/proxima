//! Encoder-decoder decode policy: the prompt, the logit filters, greedy
//! selection and the end condition, as a pure function of logits plus config
//! (PROXNUM-610 S7).
//!
//! The token ids come from the checkpoint's `generation_config.json` and
//! `config.json` ([`DecodePolicy::from_checkpoint`]); the defaults follow the
//! reference implementation (the step cap is `n_text_ctx / 2 - 4`, the
//! checkpoint's non-speech `suppress_tokens` list applies only when
//! [`DecodeConfig::suppress_nst`] is set) and the checkpoint's own values are
//! opt-in overrides ([`DecodeConfig::use_checkpoint_max_length`]).
//!
//! Composes [`proxima_tokenizer::greedy_pick`] for the argmax (first maximum
//! wins ties); a filter here is a masked `-inf`, the same value the
//! decoder-only sampler's filters write. Nothing runs a model: drive
//! [`DecodePolicy::decode`] with a closure that returns one step's logits, or
//! call [`DecodePolicy::pick`] on logits you already hold.
//!
//! [`DecodeSettings`] (std) is the conflaguration/`bon` mirror of
//! [`DecodeConfig`], the same split as [`crate::PromptCacheSettings`].

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tokenizer::greedy_pick;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// What the caller chooses; everything else is read from the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default)]
pub struct DecodeConfig {
    /// Language name as it appears in the checkpoint's `lang_to_id` (`"en"`
    /// resolves `<|en|>`). Empty takes the checkpoint's `forced_decoder_ids`
    /// entry at position 1.
    pub language: String,
    /// Task name as it appears in `task_to_id`. Empty takes the checkpoint's
    /// `forced_decoder_ids` entry at position 2.
    pub task: String,
    /// Suppress the space token and the end token at the first generated step.
    pub suppress_blank: bool,
    /// Apply the checkpoint's `suppress_tokens` list at every step.
    pub suppress_nst: bool,
    /// An explicit step cap; `0` leaves the choice to the default or to
    /// [`Self::use_checkpoint_max_length`].
    pub max_steps: u32,
    /// Take the cap from `generation_config.json`'s `max_length` (a count of
    /// total tokens, so the prompt length is subtracted) instead of the
    /// default `n_text_ctx / 2 - 4`.
    pub use_checkpoint_max_length: bool,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            language: String::from("en"),
            task: String::from("transcribe"),
            suppress_blank: true,
            suppress_nst: false,
            max_steps: 0,
            use_checkpoint_max_length: false,
        }
    }
}

/// Why a policy could not be built from a checkpoint, or a step failed.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodePolicyError {
    /// A value the policy needs is absent or malformed in the checkpoint.
    #[error("checkpoint config lacks `{0}`")]
    Missing(&'static str),
    /// `lang_to_id` has no entry for the requested language.
    #[error("language `{0}` is not in lang_to_id")]
    UnknownLanguage(String),
    /// `task_to_id` has no entry for the requested task.
    #[error("task `{0}` is not in task_to_id")]
    UnknownTask(String),
    /// A step produced no logits.
    #[error("step logits are empty")]
    EmptyLogits,
}

/// The resolved policy: token ids and the step cap, no model state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodePolicy {
    prompt: Vec<u32>,
    always_suppressed: Vec<u32>,
    first_step_suppressed: Vec<u32>,
    checkpoint_suppressed: Vec<u32>,
    apply_checkpoint_suppressed: bool,
    timestamp_begin: u32,
    end_token: u32,
    step_cap: u32,
}

/// The tokens a finished decode produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    /// Generated tokens, the end token included when decoding ended on it.
    pub tokens: Vec<u32>,
    /// `true` when the end token ended decoding, `false` at the step cap.
    pub ended_on_end_token: bool,
}

fn id_at(value: &Value, key: &'static str) -> Result<u32, DecodePolicyError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or(DecodePolicyError::Missing(key))
}

fn id_list(value: &Value, key: &str) -> Vec<u32> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|ids| {
            ids.iter()
                .filter_map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                .collect()
        })
        .unwrap_or_default()
}

fn map_ids(value: &Value, key: &'static str) -> Result<Vec<(String, u32)>, DecodePolicyError> {
    let map = value
        .get(key)
        .and_then(Value::as_object)
        .ok_or(DecodePolicyError::Missing(key))?;
    Ok(map
        .iter()
        .filter_map(|(name, id)| Some((name.clone(), u32::try_from(id.as_u64()?).ok()?)))
        .collect())
}

fn forced_id(generation: &Value, position: u64) -> Option<u32> {
    generation
        .get("forced_decoder_ids")?
        .as_array()?
        .iter()
        .filter_map(Value::as_array)
        .find(|pair| pair.first().and_then(Value::as_u64) == Some(position))
        .and_then(|pair| pair.get(1)?.as_u64())
        .and_then(|id| u32::try_from(id).ok())
}

fn named_id(
    table: &[(String, u32)],
    name: &str,
    forced: Option<u32>,
    unknown: fn(String) -> DecodePolicyError,
    missing: &'static str,
) -> Result<u32, DecodePolicyError> {
    if name.is_empty() {
        return forced.ok_or(DecodePolicyError::Missing(missing));
    }
    table
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, id)| *id)
        .ok_or_else(|| unknown(String::from(name)))
}

fn sorted(mut ids: Vec<u32>) -> Vec<u32> {
    ids.sort_unstable();
    ids.dedup();
    ids
}

impl DecodePolicy {
    /// Resolve the policy from the checkpoint's `generation_config.json` and
    /// `config.json` values plus the caller's [`DecodeConfig`].
    ///
    /// The prompt is `[decoder_start, language, task, no_timestamps]`. The
    /// always-suppressed set is the start token, the task tokens, every
    /// language token, the special tokens between the task tokens and the
    /// no-timestamps token (previous-context, no-speech, start-of-LM) and the
    /// no-timestamps token itself. Every id from the first timestamp
    /// (`no_timestamps_token_id + 1`) up is also masked, at every step.
    pub fn from_checkpoint(
        generation: &Value,
        config: &Value,
        settings: &DecodeConfig,
    ) -> Result<Self, DecodePolicyError> {
        let start = id_at(generation, "decoder_start_token_id")?;
        let end_token = id_at(generation, "eos_token_id")?;
        let no_timestamps = id_at(generation, "no_timestamps_token_id")?;
        let languages = map_ids(generation, "lang_to_id")?;
        let tasks = map_ids(generation, "task_to_id")?;
        let language = named_id(
            &languages,
            &settings.language_key(),
            forced_id(generation, 1),
            DecodePolicyError::UnknownLanguage,
            "forced_decoder_ids[1]",
        )?;
        let task = named_id(
            &tasks,
            &settings.task,
            forced_id(generation, 2),
            DecodePolicyError::UnknownTask,
            "forced_decoder_ids[2]",
        )?;
        let prompt = alloc::vec![start, language, task, no_timestamps];
        let task_ceiling = tasks.iter().map(|(_, id)| *id).max().unwrap_or(task);
        let mut always = alloc::vec![start, no_timestamps];
        always.extend(tasks.iter().map(|(_, id)| *id));
        always.extend(languages.iter().map(|(_, id)| *id));
        always.extend(task_ceiling + 1..no_timestamps);
        let mut first_step = id_list(generation, "begin_suppress_tokens");
        if settings.suppress_blank {
            first_step.push(end_token);
        } else {
            first_step.clear();
        }
        let step_cap = step_cap(generation, config, settings, prompt.len())?;
        Ok(Self {
            prompt,
            always_suppressed: sorted(always),
            first_step_suppressed: sorted(first_step),
            checkpoint_suppressed: sorted(id_list(generation, "suppress_tokens")),
            apply_checkpoint_suppressed: settings.suppress_nst,
            timestamp_begin: no_timestamps + 1,
            end_token,
            step_cap,
        })
    }

    /// The prompt the decoder consumes before the first generated token.
    #[must_use]
    pub fn prompt(&self) -> &[u32] {
        &self.prompt
    }

    /// The most steps [`Self::decode`] runs.
    #[must_use]
    pub const fn step_cap(&self) -> u32 {
        self.step_cap
    }

    /// The token that ends decoding.
    #[must_use]
    pub const fn end_token(&self) -> u32 {
        self.end_token
    }

    /// The first id of the timestamp range.
    #[must_use]
    pub const fn timestamp_begin(&self) -> u32 {
        self.timestamp_begin
    }

    /// Mask `logits` for generated step `step` (0 is the first generated
    /// token) to `-inf`: every timestamp, the always-suppressed set, the
    /// checkpoint's non-speech list when enabled, and at step 0 the
    /// first-step set. Ids past the end of `logits` are ignored.
    pub fn filter(&self, logits: &mut [f32], step: usize) {
        let begin = usize::try_from(self.timestamp_begin).unwrap_or(usize::MAX);
        if begin < logits.len() {
            logits[begin..].fill(f32::NEG_INFINITY);
        }
        mask(logits, &self.always_suppressed);
        if self.apply_checkpoint_suppressed {
            mask(logits, &self.checkpoint_suppressed);
        }
        if step == 0 {
            mask(logits, &self.first_step_suppressed);
        }
    }

    /// Filter `logits` for `step`, then take the greedy pick (the first
    /// maximum wins ties).
    pub fn pick(&self, logits: &mut [f32], step: usize) -> Result<u32, DecodePolicyError> {
        self.filter(logits, step);
        greedy_pick(logits).ok_or(DecodePolicyError::EmptyLogits)
    }

    /// Run the loop: ask `next_logits` for each step's logits given the
    /// sequence so far (prompt, then generated tokens), pick, stop on the end
    /// token or after [`Self::step_cap`] steps.
    pub fn decode<E: From<DecodePolicyError>>(
        &self,
        mut next_logits: impl FnMut(&[u32]) -> Result<Vec<f32>, E>,
    ) -> Result<Decoded, E> {
        let mut sequence = self.prompt.clone();
        let mut tokens = Vec::new();
        for step in 0..self.step_cap as usize {
            let mut logits = next_logits(&sequence)?;
            let token = self.pick(&mut logits, step)?;
            sequence.push(token);
            tokens.push(token);
            if token == self.end_token {
                return Ok(Decoded {
                    tokens,
                    ended_on_end_token: true,
                });
            }
        }
        Ok(Decoded {
            tokens,
            ended_on_end_token: false,
        })
    }
}

impl DecodeConfig {
    fn language_key(&self) -> String {
        if self.language.is_empty() {
            return String::new();
        }
        format!("<|{}|>", self.language)
    }
}

fn mask(logits: &mut [f32], ids: &[u32]) {
    for &id in ids {
        if let Some(slot) = usize::try_from(id).ok().and_then(|at| logits.get_mut(at)) {
            *slot = f32::NEG_INFINITY;
        }
    }
}

fn step_cap(
    generation: &Value,
    config: &Value,
    settings: &DecodeConfig,
    prompt_len: usize,
) -> Result<u32, DecodePolicyError> {
    if settings.max_steps > 0 {
        return Ok(settings.max_steps);
    }
    if settings.use_checkpoint_max_length {
        let total = id_at(generation, "max_length")?;
        return Ok(total.saturating_sub(u32::try_from(prompt_len).unwrap_or(u32::MAX)));
    }
    let text_context = id_at(config, "max_target_positions")?;
    Ok((text_context / 2).saturating_sub(4))
}

#[cfg(feature = "std")]
mod settings {
    use alloc::string::String;

    use bon::Builder;
    use conflaguration::{Settings, Validate};
    use serde::{Deserialize, Serialize};

    use super::DecodeConfig;

    /// The conflaguration/`bon` mirror of [`DecodeConfig`]: the builder, a TOML
    /// file and `PROXIMA_DECODE_*` env vars produce the same value, and
    /// [`Self::as_decode_config`] lowers it into the plain shape
    /// [`super::DecodePolicy::from_checkpoint`] reads.
    #[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings, Validate)]
    #[serde(default)]
    #[settings(prefix = "PROXIMA_DECODE")]
    #[builder(derive(Clone, Debug))]
    pub struct DecodeSettings {
        /// See [`DecodeConfig::language`].
        #[setting(default = "en")]
        #[builder(default = String::from("en"))]
        pub language: String,
        /// See [`DecodeConfig::task`].
        #[setting(default = "transcribe")]
        #[builder(default = String::from("transcribe"))]
        pub task: String,
        /// See [`DecodeConfig::suppress_blank`].
        #[setting(default = true)]
        #[builder(default = true)]
        pub suppress_blank: bool,
        /// See [`DecodeConfig::suppress_nst`].
        #[setting(default = false)]
        #[builder(default = false)]
        pub suppress_nst: bool,
        /// See [`DecodeConfig::max_steps`].
        #[setting(default = 0)]
        #[builder(default = 0)]
        pub max_steps: u32,
        /// See [`DecodeConfig::use_checkpoint_max_length`].
        #[setting(default = false)]
        #[builder(default = false)]
        pub use_checkpoint_max_length: bool,
    }

    impl Default for DecodeSettings {
        fn default() -> Self {
            Self::builder().build()
        }
    }

    impl DecodeSettings {
        /// The plain shape the policy reads.
        #[must_use]
        pub fn as_decode_config(&self) -> DecodeConfig {
            DecodeConfig {
                language: self.language.clone(),
                task: self.task.clone(),
                suppress_blank: self.suppress_blank,
                suppress_nst: self.suppress_nst,
                max_steps: self.max_steps,
                use_checkpoint_max_length: self.use_checkpoint_max_length,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn builder_defaults_lower_to_the_plain_default_config() {
            let settings = DecodeSettings::builder().build();
            assert_eq!(settings.as_decode_config(), DecodeConfig::default());
        }

        #[test]
        fn toml_loader_matches_the_fluent_builder() {
            let text = "language = \"fr\"\nsuppress_nst = true\nmax_steps = 12\n";
            let via_toml: DecodeSettings = toml::from_str(text).expect("settings toml parses");
            let via_builder = DecodeSettings::builder()
                .language(String::from("fr"))
                .suppress_nst(true)
                .max_steps(12)
                .build();
            assert_eq!(via_toml, via_builder);
        }
    }
}

#[cfg(feature = "std")]
pub use settings::DecodeSettings;
