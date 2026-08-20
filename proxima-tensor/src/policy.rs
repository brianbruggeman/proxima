//! Runtime execution policy -- the `std`-tier configuration surface over
//! [`crate::sized`]'s build-time constants.
//!
//! Composition, so a reader can trace every part back to a primitive
//! (guiding-principle 2):
//!
//! - [`crate::sized`] supplies the compiled floor. [`ExecutionPolicy::COMPILED`]
//!   is nothing but those constants collected into one value, so the
//!   `no_std + alloc` tier -- which never compiles this module, exactly like it
//!   never compiles [`crate::cpu`] -- keeps `sized` as its only configuration
//!   and loses nothing.
//! - `conflaguration::ConfigBuilder` supplies the layering: compiled value,
//!   then a TOML file, then per-key environment overrides, then validation.
//!   This is the same chain `prime/src/os/sizing.rs` runs over
//!   `prime-runtime.toml`, and the same loader `crate::spec` already uses for
//!   graph specs -- there is no second config mechanism in this crate.
//! - `bon::Builder` supplies the fluent half of guiding-principle 4:
//!   `ExecutionPolicy::builder().workers(..).build()` produces exactly the
//!   value a TOML file deserializes to, and
//!   `policy_and_builder_agree_field_for_field` asserts it.
//!
//! # Resolution happens once
//!
//! [`active`] memoizes into a `OnceLock`, so a hot path pays one acquire load
//! per read and never a file read, an environment read, or an allocation.
//! This is not a stylistic preference: `matmul_worker_count`'s history has two
//! measured regressions from getting it wrong -- `available_parallelism()` at
//! 3.53 us/call across the 1350 calls one forward makes, and a
//! `std::env::var` in the same spot allocating a `String` on each of them.
//! Every consumer in [`crate::cpu`] binds `let policy = policy::active();`
//! once per call, above any loop.
//!
//! # Setting it
//!
//! A library embedder, before the first evaluation:
//!
//! ```no_run
//! # #[cfg(feature = "config")] {
//! use core::num::NonZeroUsize;
//! use proxima_tensor::policy::{self, ExecutionPolicy};
//!
//! let policy = ExecutionPolicy::builder()
//!     .maybe_workers(NonZeroUsize::new(8))
//!     .cohort_spin_polls(2_000)
//!     .build();
//! policy::install(policy).expect("install before the first evaluation");
//! # }
//! ```
//!
//! A deployment, with no code at all -- `./proxima-tensor.toml`, or any path
//! named by `PROXIMA_TENSOR_CONFIG`:
//!
//! ```toml
//! workers = 8
//! min_macs_per_chunk = 500000
//! cohort_spin_polls = 2000
//! ```
//!
//! A one-off sweep, per invocation, which is what the two deleted ad-hoc
//! environment reads (`PROXIMA_MATMUL_WORKERS`, `PROXIMA_PREFAULT`) used to
//! do by hand: `PROXIMA_TENSOR_WORKERS=8 cargo run ...`. Every field has such
//! a key, derived mechanically as `PROXIMA_TENSOR_<FIELD>`; none of them is
//! hand-read anywhere in this crate.

use core::num::NonZeroUsize;
#[cfg(feature = "config")]
use core::str::FromStr;
use std::sync::OnceLock;

#[cfg(feature = "config")]
use bon::Builder;
#[cfg(feature = "config")]
use conflaguration::{ConfigBuilder, Settings, Validate, ValidationMessage};
#[cfg(feature = "config")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "config")]
use crate::error::TensorError;
use crate::sized;

/// Which executor runs a bound program. `Cpu` is [`crate::cpu`], the only
/// executor this crate ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "lowercase"))]
#[non_exhaustive]
pub enum Device {
    /// `cpu::evaluate` / `cpu::evaluate_parallel`.
    #[default]
    Cpu,
    /// GPU offload -- llama.cpp's `-ngl` / `--no-kv-offload` axis. Resolving
    /// this variant is a [`todo!`] on purpose (see [`ExecutionPolicy::device`]).
    Metal,
}

// `config`-gated because its error variant is: `TensorError::MalformedExtent`
// only exists in a `config` build, and this parser only has a caller there
// (`parse_device`, the `#[setting(resolve_with)]` hook).
#[cfg(feature = "config")]
impl FromStr for Device {
    type Err = TensorError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            other => Err(TensorError::MalformedExtent(alloc::format!(
                "device must be `cpu` or `metal`; got `{other}`"
            ))),
        }
    }
}

/// `#[setting(resolve_with)]` parser for [`ExecutionPolicy::device`] -- the
/// derive needs a `fn(&str) -> Result<T, E>` for any field type that is not
/// itself `FromStr`-resolved by the generated code.
#[cfg(feature = "config")]
fn parse_device(raw: &str) -> Result<Device, TensorError> {
    Device::from_str(raw)
}

/// `#[setting(resolve_with)]` parser for [`ExecutionPolicy::workers`]. `0`
/// parses to `None`, which is the "detect it" default -- the same spelling
/// llama.cpp uses for `-t 0`.
#[cfg(feature = "config")]
fn parse_workers(raw: &str) -> Result<Option<NonZeroUsize>, core::num::ParseIntError> {
    raw.parse::<usize>().map(NonZeroUsize::new)
}

/// Every runtime-settable execution knob, resolved. One type, not several:
/// each field is read by [`crate::cpu`] and by nothing else, so splitting it
/// would produce two structs with one consumer between them. Model-level
/// policy (context length, batch size, KV-cache quantization, layer offload)
/// deliberately lives in `proxima-model-interop`'s own settings instead --
/// this crate has no model, no KV cache and no generation loop to hang those
/// on, and a knob whose owner cannot read it is a lie.
///
/// Defaults are [`Self::COMPILED`], i.e. [`crate::sized`] verbatim, so an
/// untouched build executes exactly the instructions it executed before this
/// module existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(Builder, Serialize, Deserialize, Settings))]
#[cfg_attr(feature = "config", settings(prefix = "PROXIMA_TENSOR"))]
#[cfg_attr(feature = "config", serde(default))]
#[cfg_attr(feature = "config", builder(derive(Clone, Debug)))]
#[non_exhaustive]
pub struct ExecutionPolicy {
/// Below this many iteration-space elements, a nest runs the plain
/// sequential path even when `workers > 1`: `std::thread::scope`'s spawn
/// and join overhead outweighs the work for a small nest.
    /// Runtime form of [`sized::PARALLEL_THRESHOLD`].
    #[cfg_attr(feature = "config", setting(default))]
    #[cfg_attr(feature = "config", builder(default = sized::PARALLEL_THRESHOLD))]
    pub parallel_threshold: usize,

/// chunk count is `workers * OVERSUBSCRIBE`, not `workers`: equal row
/// counts do not mean equal wall-clock (measured 2.04x spread across 8
/// equal-row chunks of a 1024^3 GEMM), and one chunk per worker leaves no
/// spare chunk for a worker that finishes early to pick up — more chunks
/// than workers lets a fast worker absorb a slow chunk's slack. Only pays
/// off under `nest_pool`'s dynamic claiming (see `claim_and_run`), the
/// only chunk dispatch this module has: the puller count `run_chunks_threaded`
/// spawns caps at `workers` regardless of `OVERSUBSCRIBE`, so raising this
/// grows the number of chunks a fixed puller count can steal from, without
/// growing the number of threads touching them.
///
/// A `4` was tried on this same mechanism: more chunks than workers gives a
/// work-stealing pool room for a fast puller to absorb a slow chunk's slack,
/// which is structural and not in dispute. The comparison that measured it —
/// 274.75 vs 270.08 mean GFLOPS at 2048^3/4 workers, n=9, 5.9 sigma — never
/// recorded the ambient load it ran under, and the 8-worker cells it was
/// meant to help never cleared their own CoV gate at any sample size tried
/// (n up to 30) under the load present when those cells were measured (5-10,
/// against a stated 2.2 plateau). That is the same evidence shape that
/// produced three false readings for `SPLIT_ALIGNMENT` on this same box:
/// strong sigma inside an unvalidated run does not rule out noise correlated
/// across that run rather than random within it. Left at `1`, the original
/// value, until a re-measurement validates its own floor first — a
/// same-code-path comparison, at a size and load where the two configurations
/// provably execute identical instructions — and only then shows an
/// oversubscription effect outside it.
    /// Runtime form of [`sized::OVERSUBSCRIBE`].
    #[cfg_attr(feature = "config", setting(default))]
    #[cfg_attr(feature = "config", builder(default = sized::OVERSUBSCRIBE))]
    pub oversubscribe: usize,

    /// Chunk-count multiplier over `workers` for `matmul_rows_threaded`'s
    /// dynamic-claiming row split. Runtime form of
    /// [`sized::ROW_OVERSUBSCRIBE`], whose doc carries the measurement that
    /// picked `4`.
    #[cfg_attr(feature = "config", setting(default))]
    #[cfg_attr(feature = "config", builder(default = sized::ROW_OVERSUBSCRIBE))]
    pub row_oversubscribe: usize,

    /// Floor on multiply-adds per chunk for that same row split. Runtime form
    /// of [`sized::MIN_MACS_PER_CHUNK`], whose doc carries the per-shape
    /// measurement that picked `500_000` and the `700_000` it rejected.
    #[cfg_attr(feature = "config", setting(default))]
    #[cfg_attr(feature = "config", builder(default = sized::MIN_MACS_PER_CHUNK))]
    pub min_macs_per_chunk: usize,

/// Row-alignment applied to every non-final chunk boundary via
/// `BoundOp::split_aligned`. `1` is a no-op (see that method's doc): every
/// chunk boundary lands wherever `extent / chunk_count` puts it, which is
/// not necessarily a multiple of `TILE_ROWS` — so a chunk pays its own
/// row-remainder through the kernel's narrower fallback path independently
/// of every other chunk, and that per-chunk remainder count grows with
/// chunk count even though the total row count did not change. That
/// mechanism is structural and not in dispute; whether it moves
/// busy-per-MAC by a measurable amount is.
///
/// Four measurements of this same `1` -> `TILE_ROWS` change exist, all
/// against the column-panel blocking below (already landed and left on —
/// see `sized::COLUMN_PANEL_BUDGET_BYTES`). Three, run at system load
/// 12-31, read as 3-10% busy-per-MAC improvements. None of the three
/// established a noise floor before comparing — at that load level a
/// handful of percent between two configurations is not distinguishable
/// from scheduler contention, so those figures are retained here only as
/// unverified prior readings, not as evidence.
///
/// The fourth run validated a floor first: at load 2.49-3.07, `neither` vs
/// `panel` at 512^3 and 1024^3 — sizes where `neon_column_panel_cols`
/// provably clamps the panel to one, i.e. the two configurations execute
/// identical code — agreed to within +/-3.5%, sigma up to 3.1. That is the
/// noise floor any real effect at this load has to clear. Against it,
/// alignment (`1` -> `TILE_ROWS`) on top of the panel measured
/// +2.03% / +0.31% / -0.72% at 512^3 / 1024^3 / 2048^3, 8 threads — inside
/// the floor at every size. No measurable effect anywhere in the one
/// comparison whose noise floor is known.
///
/// Set to `1` on that basis: the only measurement with a validated floor
/// found nothing outside it, and the three load-12-31 figures were never
/// shown to clear their own (unmeasured) noise, so they carry no weight
/// against it. This changes only if a future re-measurement (a) validates
/// its own floor the same way — a same-code-path comparison at a size
/// where the panel is a no-op — and (b) then shows an alignment effect
/// outside that floor.
///
/// Provenance: the three load-12-31 figures were not independently dated
/// or sample-counted in the record available to this pass — treat them as
/// unverified, not merely old. The load-2.49-3.07 measurement is this
/// session's own, 2026-08-18; its sample count for the alignment
/// comparison specifically is not broken out beyond the three-configuration
/// grid it ran alongside.
    /// Runtime form of [`sized::SPLIT_ALIGNMENT`].
    #[cfg_attr(feature = "config", setting(default))]
    #[cfg_attr(feature = "config", builder(default = sized::SPLIT_ALIGNMENT))]
    pub split_alignment: u64,

/// Bytes of L2 budgeted for a resident `b` column panel in the tiled GEMM
/// pass below. M1 Max: 12 MiB shared L2 per performance cluster of 4 cores —
/// about 3 MiB/core once every worker in the cluster streams its own panel,
/// not 12 MiB as an 8 MiB budget implicitly assumed (one worker owning the
/// whole cluster's L2). ggml's own combined panel footprint never exceeds
/// ~2.5 MiB at any size or thread count, which is also where headroom for
/// the row-strip's `a` tile, the output tile in flight, and set-associativity
/// conflicts remains without the near-fit turning into a thrash.
///
/// Swept 8/4/3/2.5/2 MiB at 512/1024/2048^3, 1/2/4/8 threads, n=9,
/// interleaved round-robin per budget, 2026-08-18, system load 1.8-3.4
/// (mostly under 3.0, one late 8-thread cell drifted to 3.37). Only the
/// 1-thread cells stayed under the 1.5% CoV resolvability bar; every
/// 2+-thread cell exceeded it (up to 20% CoV, this session's shared-host
/// contention) and is not usable for a budget comparison. Within the
/// resolvable 1-thread cells: 512^3 and 1024^3 measured flat across every
/// budget from 8 MiB down to 2 MiB (busy-per-MAC within ~1% of each other,
/// GFLOPS parity vs ggml 89.57-90.17 for 1024^3 across 8/2.5 MiB, no
/// resolvable win despite the panel becoming numerically "active" at
/// 1024^3 below ~2.8 MiB) — the hypothesis that a lower budget would help
/// 1024^3 did NOT hold up. 2048^3/1-thread did show a real, resolvable
/// effect: busy-per-mac dropped ~1.7-2% for every budget at or below 4 MiB
/// versus the 8 MiB control (0.02238 -> ~0.0220), and GFLOPS parity vs ggml
/// rose from 0.999x to 1.026x at 2.5 MiB. 4/3/2.5/2 MiB were statistically
/// indistinguishable from each other at 2048^3/1-thread (within ~0.5%, same
/// order as the noise floor) — no single value in that range measured best.
/// 2.5 MiB is landed here because it matches ggml's own measured combined
/// footprint and never measured worse than the 8 MiB control in any
/// resolvable cell; 4 MiB or 3 MiB would be an equally defensible pick on
/// this data. checksums (135.87619/260.24106/513.10425) and the 1024^3
/// allocation shape were unchanged across every budget tested.
    /// Runtime form of [`sized::COLUMN_PANEL_BUDGET_BYTES`]. Read only by the
    /// aarch64 tiled-GEMM pass today, but the field is unconditional: a
    /// config file written on one machine has to deserialize on another, and
    /// an L2 budget in bytes is meaningful on every target whether or not a
    /// kernel currently consults it there.
    #[cfg_attr(feature = "config", setting(default))]
    #[cfg_attr(feature = "config", builder(default = sized::COLUMN_PANEL_BUDGET_BYTES))]
    pub column_panel_budget_bytes: usize,

    /// Worker count for every threaded dispatch this crate makes. `None`
    /// means detect (Apple P-cores via `sysctlbyname`, else
    /// `available_parallelism`), which is what `cpu::matmul_worker_count`
    /// does. This field replaces the ad-hoc `PROXIMA_MATMUL_WORKERS` read
    /// that used to sit inside that function: the environment key is now
    /// `PROXIMA_TENSOR_WORKERS`, resolved through the same layered chain as
    /// every other knob rather than by a hand-written `std::env::var`, and
    /// `0` still spells "detect".
    #[cfg_attr(feature = "config", setting(default, resolve_with = "parse_workers"))]
    pub workers: Option<NonZeroUsize>,

    /// Spin budget, in `core::hint::spin_loop()` polls, a cohort member burns
    /// before parking -- `prime::os::cohort::CohortConfig::spin_polls`, which
    /// `cpu::nest_cohort` used to leave at `CohortBuilder::default()`. Not a
    /// new knob and not a new type: the cohort already had the config and the
    /// builder, this crate simply never chose a value. `2_000` is measured;
    /// `200` was measurably worse.
    #[cfg_attr(feature = "config", setting(default))]
    #[cfg_attr(feature = "config", builder(default = sized::COHORT_SPIN_POLLS))]
    pub cohort_spin_polls: u32,

    /// Which executor runs the program -- llama.cpp's `-ngl` /
    /// `--no-kv-offload` axis, as far as this crate can express it.
    /// [`Device::Cpu`] resolves; [`Device::Metal`] hits a [`todo!`] in
    /// [`ExecutionPolicy::executor`] naming what is missing, because
    /// `omega::metal::execute` takes `blocks: &[&[f32]]` and so cannot accept
    /// the packed quantized weights every real model in this workspace loads.
    /// The field exists rather than being omitted so the gap is compilable
    /// and greppable instead of tribal.
    #[cfg_attr(feature = "config", setting(default, resolve_with = "parse_device"))]
    #[cfg_attr(feature = "config", builder(default))]
    pub device: Device,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self::COMPILED
    }
}

impl ExecutionPolicy {
    /// [`crate::sized`] verbatim -- the value every tier below `std` uses as
    /// its only configuration, and the base layer every higher tier overrides
    /// per key.
    pub const COMPILED: Self = Self {
        parallel_threshold: sized::PARALLEL_THRESHOLD,
        oversubscribe: sized::OVERSUBSCRIBE,
        row_oversubscribe: sized::ROW_OVERSUBSCRIBE,
        min_macs_per_chunk: sized::MIN_MACS_PER_CHUNK,
        split_alignment: sized::SPLIT_ALIGNMENT,
        column_panel_budget_bytes: sized::COLUMN_PANEL_BUDGET_BYTES,
        workers: None,
        cohort_spin_polls: sized::COHORT_SPIN_POLLS,
        device: Device::Cpu,
    };

    /// The executor this policy selects. `Device::Metal` is a [`todo!`], not
    /// an error, because the gap is an unwritten code path rather than a bad
    /// request: naming it here is what turns "we have no GPU offload" from
    /// tribal knowledge into one grep.
    pub fn executor(&self) -> Device {
        match self.device {
            Device::Cpu => Device::Cpu,
            Device::Metal => todo!(
                "gpu offload (llama.cpp `-ngl` / `--no-kv-offload`): omega::metal::execute \
                 takes `blocks: &[&[f32]]` and cannot accept the packed Q4_K/Q5_K/Q6_K \
                 blocks proxima-model-interop binds, so there is no path from a bound \
                 program to the Metal executor yet. This match arm is where the dispatch \
                 would branch once `omega::metal::execute` accepts `QuantizedBlock` \
                 operands; until then `device = \"metal\"` is a knob with nothing behind it \
                 and says so instead of silently running on the CPU."
            ),
        }
    }
}

/// The layered load, in precedence order: [`ExecutionPolicy::COMPILED`], then
/// the TOML file, then set environment keys, then validation. Missing file
/// keys and unset environment keys leave the compiled value untouched -- the
/// same per-key fallback `prime::os::sizing::Sizing::resolved` gives.
#[cfg(feature = "config")]
impl ExecutionPolicy {
    /// Load an explicit path. The entry point for an embedder that keeps its
    /// own config file rather than using discovery.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, TensorError> {
        ConfigBuilder::<Self>::new()
            .value(Self::COMPILED)
            .file(path)
            .env()
            .validate()
            .build()
            .map_err(|error| TensorError::MalformedExtent(alloc::format!("execution policy: {error}")))
    }

    /// Discovery form: `$PROXIMA_TENSOR_CONFIG`, else `./proxima-tensor.toml`
    /// when it exists, else compiled-plus-environment. Never fails -- a
    /// malformed file reports on stderr and the compiled floor stands, since
    /// this runs inside a `OnceLock` init on a path whose callers (`usize`
    /// -returning chunk math) have nowhere to return an error to. An embedder
    /// that wants the error uses [`Self::load`].
    #[must_use]
    pub fn discover() -> Self {
        let chain = ConfigBuilder::<Self>::new().value(Self::COMPILED);
        let chain = match discovered_path() {
            Some(path) => chain.file(path),
            None => chain,
        };
        match chain.env().validate().build() {
            Ok(policy) => policy,
            Err(error) => {
                // no telemetry dependency at this tier (`instrument` is
                // default-off), and a config we could not read must not
                // pass silently
                std::eprintln!(
                    "proxima-tensor: execution policy unusable ({error}); using compiled sized.rs values"
                );
                Self::COMPILED
            }
        }
    }
}

#[cfg(feature = "config")]
fn discovered_path() -> Option<std::path::PathBuf> {
    if let Ok(explicit) = std::env::var("PROXIMA_TENSOR_CONFIG") {
        return Some(std::path::PathBuf::from(explicit));
    }
    let fallback = std::path::PathBuf::from("proxima-tensor.toml");
    fallback.is_file().then_some(fallback)
}

#[cfg(feature = "config")]
impl Validate for ExecutionPolicy {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = alloc::vec::Vec::new();
        let mut require_nonzero = |name: &'static str, value: u64| {
            if value == 0 {
                errors.push(ValidationMessage::new(name, "must be non-zero"));
            }
        };
        require_nonzero("oversubscribe", self.oversubscribe as u64);
        require_nonzero("row_oversubscribe", self.row_oversubscribe as u64);
        require_nonzero("min_macs_per_chunk", self.min_macs_per_chunk as u64);
        require_nonzero("split_alignment", self.split_alignment);
        require_nonzero("column_panel_budget_bytes", self.column_panel_budget_bytes as u64);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

static ACTIVE: OnceLock<ExecutionPolicy> = OnceLock::new();

/// The process-wide policy, resolved at most once. Hot paths call this and
/// bind the result above their loops; the cost per call is one acquire load
/// on an already-initialized `OnceLock`, never a file, an environment read,
/// or an allocation.
#[must_use]
pub fn active() -> &'static ExecutionPolicy {
    ACTIVE.get_or_init(resolve)
}

/// Install a policy programmatically. Fails, returning the rejected value, if
/// [`active`] already resolved -- the policy is a process-wide constant once
/// read, so an embedder installs before the first evaluation or not at all.
/// Deliberately not a `set`-and-forget: silently ignoring a late install
/// would make a caller believe a knob took effect when the first matmul
/// already baked the old one into `cpu`'s worker cache.
pub fn install(policy: ExecutionPolicy) -> Result<(), ExecutionPolicy> {
    ACTIVE.set(policy)
}

#[cfg(feature = "config")]
fn resolve() -> ExecutionPolicy {
    ExecutionPolicy::discover()
}

/// Without the `config` feature there is no loader in the build at all -- no
/// `serde`, no `conflaguration`, no file, no environment. `std` alone runs
/// the compiled floor plus whatever an embedder installs through [`install`].
#[cfg(not(feature = "config"))]
fn resolve() -> ExecutionPolicy {
    ExecutionPolicy::COMPILED
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn compiled_policy_is_sized_verbatim_so_defaults_change_no_instruction() {
        let policy = ExecutionPolicy::COMPILED;

        assert_eq!(policy.parallel_threshold, sized::PARALLEL_THRESHOLD);
        assert_eq!(policy.oversubscribe, sized::OVERSUBSCRIBE);
        assert_eq!(policy.row_oversubscribe, sized::ROW_OVERSUBSCRIBE);
        assert_eq!(policy.min_macs_per_chunk, sized::MIN_MACS_PER_CHUNK);
        assert_eq!(policy.split_alignment, sized::SPLIT_ALIGNMENT);
        assert_eq!(policy.column_panel_budget_bytes, sized::COLUMN_PANEL_BUDGET_BYTES);
        assert_eq!(policy.cohort_spin_polls, sized::COHORT_SPIN_POLLS);
        assert_eq!(policy.workers, None);
        assert_eq!(policy.device, Device::Cpu);
    }

    #[cfg(feature = "config")]
    #[test]
    fn builder_and_config_agree_field_for_field() {
        let built = ExecutionPolicy::builder().build();
        let deserialized: ExecutionPolicy = toml::from_str("").expect("empty toml is all defaults");

        assert_eq!(built, ExecutionPolicy::COMPILED);
        assert_eq!(deserialized, ExecutionPolicy::COMPILED);
    }

    #[cfg(feature = "config")]
    #[test]
    fn a_partial_toml_overrides_only_the_keys_it_names() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("proxima-tensor.toml");
        std::fs::write(&path, "workers = 8\nmin_macs_per_chunk = 250000\n").expect("write toml");

        let policy = ExecutionPolicy::load(&path).expect("load the override file");

        assert_eq!(policy.workers, NonZeroUsize::new(8));
        assert_eq!(policy.min_macs_per_chunk, 250_000);
        assert_eq!(policy.row_oversubscribe, sized::ROW_OVERSUBSCRIBE);
        assert_eq!(policy.parallel_threshold, sized::PARALLEL_THRESHOLD);
    }

    #[cfg(feature = "config")]
    #[test]
    fn an_environment_key_overrides_one_field_and_leaves_the_rest_compiled() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("empty.toml");
        std::fs::write(&path, "").expect("write empty toml");

        let policy = temp_env::with_var("PROXIMA_TENSOR_WORKERS", Some("6"), || {
            ExecutionPolicy::load(&path).expect("load with an env override")
        });

        assert_eq!(policy.workers, NonZeroUsize::new(6));
        assert_eq!(policy.cohort_spin_polls, sized::COHORT_SPIN_POLLS);
    }

    #[cfg(feature = "config")]
    #[test]
    fn workers_zero_spells_detect_the_same_way_llama_cpp_does() {
        assert_eq!(parse_workers("0").expect("zero parses"), None);
        assert_eq!(parse_workers("8").expect("eight parses"), NonZeroUsize::new(8));
    }

    #[cfg(feature = "config")]
    #[test]
    fn a_zero_divisor_is_rejected_rather_than_dividing_by_it_in_row_chunk_count() {
        let broken = ExecutionPolicy {
            min_macs_per_chunk: 0,
            ..ExecutionPolicy::COMPILED
        };

        assert!(broken.validate().is_err());
    }
}
