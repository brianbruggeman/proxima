use std::path::{Path, PathBuf};

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use proxima_core::ProximaError;
use serde::{Deserialize, Serialize};

use proxima_recording::pipe::DeferredRuntime;

use crate::capture::{Capture, ChunkGranularity};
use crate::pipe::InterceptPipe;
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};

/// Typed config surface for the `intercept` MITM pipe — the wire form the
/// [`InterceptPipeFactory`](crate::InterceptPipeFactory) deserialises in place
/// of the historical hand-parser. Only the serialisable knobs live here; the
/// runtime CA key pair (and the rustls acceptors built from it) are constructed
/// in [`Self::into_pipe`] exactly as the factory did before.
#[derive(Debug, Clone, Default, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_INTERCEPT")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct InterceptConfig {
    /// PEM-encoded CA cert path. Paired with `ca_key`: when BOTH are set the
    /// pipe loads a persistent CA, otherwise it generates an ephemeral one.
    /// example: /var/lib/proxima/intercept-ca.crt
    #[setting(default)]
    #[serde(default)]
    pub ca_cert: Option<String>,

    /// PEM-encoded CA private key path. See `ca_cert`.
    /// example: /var/lib/proxima/intercept-ca.key
    #[setting(default)]
    #[serde(default)]
    pub ca_key: Option<String>,

    /// Capture sink config. Present = record the intercepted exchanges to a
    /// durable log (disarmed until the App turns the spigot on at serve);
    /// absent = observe-and-forward only.
    #[setting(skip)]
    #[serde(default)]
    pub capture: Option<CaptureSettings>,
}

impl Validate for InterceptConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors: Vec<ValidationMessage> = Vec::new();
        // both-or-neither CA paths — matches the factory's (Some, Some) gate.
        if self.ca_cert.is_some() != self.ca_key.is_some() {
            errors.push(ValidationMessage::new(
                "ca_cert,ca_key",
                "ca_cert and ca_key must either both be set or both be empty",
            ));
        }
        if let Some(capture) = &self.capture {
            // the factory rejected `..` traversal in capture.data_path; carry
            // that guard onto the typed sub-config.
            if capture
                .data_path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
            {
                errors.push(ValidationMessage::new(
                    "capture.data_path",
                    "capture.data_path must not contain `..` components",
                ));
            }
            if let Err(conflaguration::Error::Validation { errors: nested }) = capture.validate() {
                errors.extend(nested);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl InterceptConfig {
    /// Materialise the intercept pipe: load or generate the CA, then attach the
    /// capture terminal (disarmed) when configured. Preserves the factory's
    /// decode/parse/error semantics — `validate()` runs first so the `..` and
    /// both-or-neither-CA checks surface as `ProximaError::Config`.
    pub fn into_pipe(self, spigot: DeferredRuntime) -> Result<InterceptPipe, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        let pipe = match (self.ca_cert.as_deref(), self.ca_key.as_deref()) {
            (Some(cert_path), Some(key_path)) => {
                InterceptPipe::with_ca_files(Path::new(cert_path), Path::new(key_path))?
            }
            _ => InterceptPipe::with_generated_ca()?,
        };
        match self.capture {
            Some(capture) => Ok(pipe.with_capture(capture.into_capture(spigot)?)),
            None => Ok(pipe),
        }
    }

    /// Lower straight to a [`PipeHandle`] — the form the factory hands back.
    pub fn into_handle(self, spigot: DeferredRuntime) -> Result<PipeHandle, ProximaError> {
        Ok(into_handle(self.into_pipe(spigot)?))
    }
}

#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "INTERCEPT_CAPTURE")]
#[builder(derive(Clone, Debug))]
pub struct CaptureSettings {
    /// Where the BinSink durable log lives. Index file is sibling at
    /// `<path>.idx`. Path must be writable; capture::open fails fast
    /// if the parent dir is missing or read-only.
    /// example: /var/lib/proxima/intercept.bin
    pub data_path: PathBuf,

    /// Vestigial: the per-direction broadcast ring was replaced by an owned
    /// chunk buffer, so this no longer affects capture. Retained for config
    /// back-compat (and to avoid breaking existing TOML); ignored at runtime.
    /// example: 256
    #[setting(default = 256)]
    #[serde(default = "default_tee_cap")]
    #[builder(default = default_tee_cap())]
    pub tee_cap_items: usize,

    /// How chunks become recorded events — the fidelity/throughput knob:
    /// `per_chunk` (exact-wire replay, default), `coalesced` (one event per
    /// direction — replay works, framing lost), or `discard` (structural
    /// events only; no raw bytes buffered/stored — for logging without replay).
    /// example: per_chunk
    #[serde(default)]
    #[builder(default)]
    pub chunk_granularity: ChunkGranularity,

    /// zstd level for the block compressor — the CPU/ratio lever. 3 is a good
    /// default (fast, ~14x on repetitive SSE); raise for smaller files at more
    /// CPU. Only applies to blocks over the compress threshold.
    /// example: 3
    #[setting(default = 3)]
    #[serde(default = "default_zstd_level")]
    #[builder(default = default_zstd_level())]
    pub zstd_level: i32,
}

fn default_tee_cap() -> usize {
    256
}

fn default_zstd_level() -> i32 {
    3
}

impl Validate for CaptureSettings {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors: Vec<ValidationMessage> = Vec::new();
        if self.data_path.as_os_str().is_empty() {
            errors.push(ValidationMessage::new(
                "data_path",
                "data_path must be non-empty",
            ));
        }
        if self.tee_cap_items == 0 {
            errors.push(ValidationMessage::new(
                "tee_cap_items",
                "tee_cap_items must be > 0",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl CaptureSettings {
    /// Construct from path with default tee capacity. Mirrors the fluent
    /// API `Capture::open(path)` so config-driven and API-driven setups
    /// build equivalent state.
    #[must_use]
    pub fn from_path(data_path: impl AsRef<Path>) -> Self {
        Self {
            data_path: data_path.as_ref().to_path_buf(),
            tee_cap_items: default_tee_cap(),
            chunk_granularity: ChunkGranularity::default(),
            zstd_level: default_zstd_level(),
        }
    }

    /// Build the (disarmed) capture terminal from settings + the shared
    /// spigot. Opens no file until the App arms the spigot at serve.
    pub fn into_capture(self, spigot: DeferredRuntime) -> Result<Capture, ProximaError> {
        Ok(
            Capture::open_with_level(&self.data_path, self.zstd_level, spigot)?
                .with_chunk_granularity(self.chunk_granularity),
        )
    }
}

/// Error parsing [`ChunkGranularity`] from a config/env string.
#[derive(Debug)]
pub struct ChunkGranularityParseError {
    value: String,
}

impl std::fmt::Display for ChunkGranularityParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "unknown chunk_granularity {:?} (expected per_chunk or discard)",
            self.value
        )
    }
}

impl std::error::Error for ChunkGranularityParseError {}

// lets the `Settings` derive resolve `chunk_granularity` from an env var
// (INTERCEPT_CAPTURE_CHUNK_GRANULARITY=discard), matching the serde rename.
impl conflaguration::FromEnvStr for ChunkGranularity {
    type Err = ChunkGranularityParseError;

    fn from_env_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "per_chunk" | "perchunk" => Ok(Self::PerChunk),
            "discard" => Ok(Self::Discard),
            _ => Err(ChunkGranularityParseError {
                value: value.to_string(),
            }),
        }
    }

    fn type_name() -> &'static str {
        "ChunkGranularity"
    }
}

/// Settings surface for the request compressor (C5 Cfg/API). The knobs were
/// previously hardcoded constants in compress.rs; this exposes them as a
/// conflaguration-backed config + bon builder. `into_params()` produces the
/// always-available `compress::CompressParams` the compressor actually takes,
/// so config-driven and direct-API setups converge on identical state.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "INTERCEPT_COMPRESS")]
#[builder(derive(Clone, Debug))]
pub struct CompressConfig {
    /// Lines shorter than this are never deduplicated (kept verbatim).
    /// example: 30
    #[setting(default = 30)]
    #[serde(default = "default_dedup_min_line_len")]
    #[builder(default = default_dedup_min_line_len())]
    pub dedup_min_line_len: usize,

    /// Entropy pruning operates on blocks of this many chars.
    /// example: 200
    #[setting(default = 200)]
    #[serde(default = "default_entropy_block_size")]
    #[builder(default = default_entropy_block_size())]
    pub entropy_block_size: usize,

    /// Blocks scoring below this Shannon entropy (bits/byte) are dropped.
    /// example: 2.5
    #[setting(default = 2.5)]
    #[serde(default = "default_entropy_floor")]
    #[builder(default = default_entropy_floor())]
    pub entropy_floor: f64,
}

fn default_dedup_min_line_len() -> usize {
    crate::compress::DEFAULT_DEDUP_MIN_LINE_LEN
}

fn default_entropy_block_size() -> usize {
    crate::compress::DEFAULT_ENTROPY_BLOCK_SIZE
}

fn default_entropy_floor() -> f64 {
    crate::compress::DEFAULT_ENTROPY_FLOOR
}

/// Where the intercept MITM CA cert + key live. Used by
/// [`crate::ca::load_ca`] so a binary can drive its certificate
/// material entirely from env vars or a TOML block. When neither
/// path is set, callers should fall back to `generate_ca()` (the
/// existing ephemeral path).
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "INTERCEPT_CA")]
#[builder(derive(Clone, Debug), on(PathBuf, into))]
pub struct CaConfig {
    /// PEM-encoded cert path. Read by `load_ca`. Both this and
    /// `key_path` must be set to load a persistent CA; either
    /// missing is interpreted as "use ephemeral CA via
    /// generate_ca()". example: `/var/lib/proxima/intercept-ca.crt`
    #[setting(default_str = "")]
    #[serde(default)]
    #[builder(default)]
    pub cert_path: PathBuf,

    /// PEM-encoded private key path. Must be readable by the
    /// process. example: `/var/lib/proxima/intercept-ca.key`
    #[setting(default_str = "")]
    #[serde(default)]
    #[builder(default)]
    pub key_path: PathBuf,
}

impl Default for CaConfig {
    fn default() -> Self {
        Self {
            cert_path: PathBuf::new(),
            key_path: PathBuf::new(),
        }
    }
}

impl CaConfig {
    /// Whether both cert + key paths are set (the "load persistent
    /// CA" path). When false, callers should call
    /// [`crate::ca::generate_ca`] for an ephemeral CA.
    #[must_use]
    pub fn is_persistent(&self) -> bool {
        !self.cert_path.as_os_str().is_empty() && !self.key_path.as_os_str().is_empty()
    }

    /// Load a persistent CA from the configured paths. Returns
    /// `Err` if `is_persistent()` is false or the files cannot be
    /// read.
    pub fn load(&self) -> Result<crate::ca::CaKeyPair, ProximaError> {
        if !self.is_persistent() {
            return Err(ProximaError::Config(
                "CaConfig::load: both cert_path and key_path must be non-empty".into(),
            ));
        }
        crate::ca::load_ca(&self.cert_path, &self.key_path)
    }
}

impl Validate for CaConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        // both empty is the "generate ephemeral" path; both set is
        // the "load persistent" path; exactly one set is ambiguous.
        if self.cert_path.as_os_str().is_empty() != self.key_path.as_os_str().is_empty() {
            return Err(conflaguration::Error::Validation {
                errors: vec![ValidationMessage::new(
                    "cert_path,key_path",
                    "cert_path and key_path must either both be set or both be empty",
                )],
            });
        }
        Ok(())
    }
}

impl Default for CompressConfig {
    fn default() -> Self {
        Self {
            dedup_min_line_len: default_dedup_min_line_len(),
            entropy_block_size: default_entropy_block_size(),
            entropy_floor: default_entropy_floor(),
        }
    }
}

impl Validate for CompressConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors: Vec<ValidationMessage> = Vec::new();
        if self.entropy_block_size == 0 {
            errors.push(ValidationMessage::new(
                "entropy_block_size",
                "entropy_block_size must be > 0",
            ));
        }
        if self.entropy_floor < 0.0 {
            errors.push(ValidationMessage::new(
                "entropy_floor",
                "entropy_floor must be >= 0",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl CompressConfig {
    /// Convert to the plain `CompressParams` the compressor consumes. This is
    /// the bridge that makes config-driven and direct-API compression
    /// equivalent (Cfg/API parity).
    #[must_use]
    pub fn into_params(self) -> crate::compress::CompressParams {
        crate::compress::CompressParams {
            dedup_min_line_len: self.dedup_min_line_len,
            entropy_block_size: self.entropy_block_size,
            entropy_floor: self.entropy_floor,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn validation_rejects_empty_path() {
        let settings = CaptureSettings {
            data_path: PathBuf::new(),
            tee_cap_items: 256,
            chunk_granularity: ChunkGranularity::default(),
            zstd_level: default_zstd_level(),
        };
        let result = settings.validate();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_zero_tee_cap() {
        let settings = CaptureSettings {
            data_path: PathBuf::from("/tmp/intercept.bin"),
            tee_cap_items: 0,
            chunk_granularity: ChunkGranularity::default(),
            zstd_level: default_zstd_level(),
        };
        let result = settings.validate();
        assert!(result.is_err());
    }

    #[test]
    fn validation_accepts_valid_config() {
        let settings = CaptureSettings::from_path("/tmp/intercept.bin");
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn from_path_uses_default_tee_cap() {
        let settings = CaptureSettings::from_path("/tmp/intercept.bin");
        assert_eq!(settings.tee_cap_items, default_tee_cap());
    }

    #[test]
    fn serde_round_trip_preserves_state() {
        let original = CaptureSettings {
            data_path: PathBuf::from("/var/lib/proxima/intercept.bin"),
            tee_cap_items: 1024,
            chunk_granularity: ChunkGranularity::Discard,
            zstd_level: 9,
        };
        let toml_text = toml::to_string(&original).expect("serialize");
        let restored: CaptureSettings = toml::from_str(&toml_text).expect("deserialize");
        assert_eq!(restored.data_path, original.data_path);
        assert_eq!(restored.tee_cap_items, original.tee_cap_items);
        assert_eq!(restored.chunk_granularity, original.chunk_granularity);
        assert_eq!(restored.zstd_level, original.zstd_level);
    }

    #[test]
    fn config_and_api_paths_produce_equivalent_capture_target() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let path = temp_dir.path().join("intercept.bin");
        let from_settings = CaptureSettings::from_path(&path);
        let from_builder = CaptureSettings::builder().data_path(path.clone()).build();
        assert_eq!(from_settings.data_path, from_builder.data_path);
        assert_eq!(from_settings.tee_cap_items, from_builder.tee_cap_items);
    }

    #[test]
    fn into_capture_builds_disarmed_terminal() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let path = temp_dir.path().join("intercept.bin");
        let settings = CaptureSettings::from_path(&path);
        let capture = settings
            .into_capture(proxima_recording::pipe::deferred_runtime())
            .expect("build from settings");
        // disarmed until the App arms the spigot: no file, nothing to pump.
        assert!(!capture.is_armed());
        assert!(!path.exists(), "no spigot -> no file opened");
    }

    // --- CompressConfig (C5 Cfg/API) ---

    #[test]
    fn compress_config_default_matches_compress_constants() {
        let config = CompressConfig::default();
        assert_eq!(
            config.dedup_min_line_len,
            crate::compress::DEFAULT_DEDUP_MIN_LINE_LEN
        );
        assert_eq!(
            config.entropy_block_size,
            crate::compress::DEFAULT_ENTROPY_BLOCK_SIZE
        );
        assert!((config.entropy_floor - crate::compress::DEFAULT_ENTROPY_FLOOR).abs() < 1e-9);
    }

    #[test]
    fn compress_config_and_params_defaults_agree() {
        // Cfg/API parity: a default config converted to params must equal the
        // plain CompressParams::default the compressor uses without config.
        let from_config = CompressConfig::default().into_params();
        assert_eq!(from_config, crate::compress::CompressParams::default());
    }

    #[test]
    fn compress_config_builder_and_into_params_round_trip() {
        let config = CompressConfig::builder()
            .dedup_min_line_len(50)
            .entropy_block_size(128)
            .entropy_floor(3.0)
            .build();
        let params = config.into_params();
        assert_eq!(params.dedup_min_line_len, 50);
        assert_eq!(params.entropy_block_size, 128);
        assert!((params.entropy_floor - 3.0).abs() < 1e-9);
    }

    #[test]
    fn compress_config_validation_rejects_zero_block_size() {
        let config = CompressConfig {
            dedup_min_line_len: 30,
            entropy_block_size: 0,
            entropy_floor: 2.5,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn compress_config_validation_rejects_negative_floor() {
        let config = CompressConfig {
            dedup_min_line_len: 30,
            entropy_block_size: 200,
            entropy_floor: -1.0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn compress_config_validation_accepts_default() {
        assert!(CompressConfig::default().validate().is_ok());
    }

    #[test]
    fn compress_config_toml_round_trip() {
        let original = CompressConfig::builder()
            .dedup_min_line_len(40)
            .entropy_block_size(256)
            .entropy_floor(2.0)
            .build();
        let toml_text = toml::to_string(&original).expect("serialize");
        let restored: CompressConfig = toml::from_str(&toml_text).expect("deserialize");
        assert_eq!(restored.dedup_min_line_len, original.dedup_min_line_len);
        assert_eq!(restored.entropy_block_size, original.entropy_block_size);
        assert!((restored.entropy_floor - original.entropy_floor).abs() < 1e-9);
    }

    #[test]
    fn compress_config_drives_actual_compression() {
        // a custom config must change compressor behavior: with an
        // unreachable entropy floor, every block is dropped from a long
        // low-entropy instruction.
        let body = serde_json::json!({
            "model": "model-mini",
            "instructions": "x".repeat(400),
            "input": [{"role": "user", "content": "hi"}],
        });
        let raw = serde_json::to_vec(&body).expect("serialize");
        let aggressive = CompressConfig::builder()
            .entropy_floor(8.0)
            .build()
            .into_params();
        let compressed =
            crate::compress::compress_json_messages_with(&raw, &aggressive).expect("compress");
        let result: serde_json::Value = serde_json::from_slice(&compressed).expect("parse");
        let instructions = result["instructions"].as_str().unwrap();
        assert!(
            instructions.len() < 400,
            "aggressive floor must prune the low-entropy block"
        );
    }

    #[test]
    fn ca_config_defaults_to_empty_ephemeral_path() {
        let config = CaConfig::default();
        assert!(!config.is_persistent());
        assert!(config.cert_path.as_os_str().is_empty());
        assert!(config.key_path.as_os_str().is_empty());
    }

    #[test]
    fn ca_config_with_both_paths_set_is_persistent() {
        let config = CaConfig::builder()
            .cert_path("/tmp/ca.crt")
            .key_path("/tmp/ca.key")
            .build();
        assert!(config.is_persistent());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn ca_config_with_only_one_path_set_fails_validation() {
        let only_cert = CaConfig::builder().cert_path("/tmp/ca.crt").build();
        assert!(!only_cert.is_persistent());
        assert!(only_cert.validate().is_err());

        let only_key = CaConfig::builder().key_path("/tmp/ca.key").build();
        assert!(!only_key.is_persistent());
        assert!(only_key.validate().is_err());
    }

    #[test]
    fn ca_config_load_rejects_non_persistent() {
        let outcome = CaConfig::default().load();
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    // --- InterceptConfig (top-level intercept pipe wire form) ---

    #[test]
    fn intercept_config_rejects_parent_dir_traversal_in_capture() {
        let config: InterceptConfig = serde_json::from_value(serde_json::json!({
            "capture": { "data_path": "/var/data/../../etc/recording.bin" }
        }))
        .expect("from_value");
        let outcome = config.into_pipe(proxima_recording::pipe::deferred_runtime());
        let Err(ProximaError::Config(message)) = outcome else {
            panic!("expected a `..` config error");
        };
        assert!(
            message.contains(".."),
            "error should mention `..`, got: {message}"
        );
    }

    #[test]
    fn intercept_config_missing_capture_data_path_errors() {
        let outcome: Result<InterceptConfig, _> =
            serde_json::from_value(serde_json::json!({ "capture": {} }));
        let message = format!("{}", outcome.expect_err("missing data_path must fail"));
        assert!(
            message.contains("data_path"),
            "error should mention data_path, got: {message}"
        );
    }

    #[test]
    fn intercept_config_rejects_one_sided_ca() {
        let config: InterceptConfig =
            serde_json::from_value(serde_json::json!({ "ca_cert": "/tmp/ca.crt" }))
                .expect("from_value");
        assert!(
            config.validate().is_err(),
            "lone ca_cert must fail validation"
        );
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // an equivalent intercept pipe (capture disarmed, no file opened at build).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let path = temp_dir.path().join("intercept.bin");
        let path_str = path.to_string_lossy().to_string();

        let from_value: InterceptConfig = serde_json::from_value(serde_json::json!({
            "capture": { "data_path": path_str },
        }))
        .expect("from_value");
        let from_value = from_value
            .into_pipe(proxima_recording::pipe::deferred_runtime())
            .expect("into_pipe value");

        let from_builder = InterceptConfig::builder()
            .capture(CaptureSettings::from_path(&path))
            .build()
            .into_pipe(proxima_recording::pipe::deferred_runtime())
            .expect("into_pipe builder");

        // both lower to an ephemeral-CA pipe with a disarmed capture terminal;
        // neither opens a file until the spigot is armed at serve.
        assert!(from_value.capture_is_armed() == from_builder.capture_is_armed());
        assert!(
            !from_value.capture_is_armed(),
            "capture must be disarmed at build"
        );
        assert!(!path.exists(), "no spigot -> no file opened");
    }
}
