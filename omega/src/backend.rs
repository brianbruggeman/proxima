//! The backend-agnostic entry point: one `plan_named`/`execute_plan_named`
//! pair that runs a named-block program on whichever [`Engine`] the caller
//! names, without that caller ever writing `proxima_tensor::cpu` or
//! `omega::metal` itself.
//!
//! # Why this is not just "call whichever one is reachable"
//!
//! `proxima_tensor::cpu` is gated on `feature = "std"` alone
//! (`proxima-tensor/src/lib.rs:204`), and BOTH of omega's own `cpu` and
//! `metal` features turn `proxima-tensor/std` on (`cpu` needs it for the
//! evaluator itself; `metal` needs it because the driver's own prepare
//! pipeline uses `Evaluated`/`QuantizedBlock`/`resolve_named_blocks`, all
//! std-gated). So in a `--features std,metal` build with omega's `cpu`
//! feature OFF, `proxima_tensor::cpu` is still importable — reachability of
//! the CPU evaluator says nothing about whether the CPU backend was meant to
//! be compiled in. Every arm below is therefore gated on OMEGA'S OWN feature
//! (`#[cfg(feature = "cpu")]`, `#[cfg(feature = "metal")]`), never on
//! whether `proxima_tensor::cpu` happens to be visible.
//!
//! # Two engines, two drivers for one of them
//!
//! There are two places an op runs: a CPU core, or a GPU. [`Engine`] names
//! exactly that, `{ Cpu, Gpu }`. `Metal`/`Wgpu`/`Vulkan`/`Cuda` were never
//! peers of `Cpu` — `Metal` and `Wgpu` are two DRIVERS reaching the same Gpu
//! engine (same `BoundOp` descriptor, same emit-then-drive split, an MSL vs
//! a WGSL emitter), which [`GpuDriver`] now names; `Vulkan`/`Npu`/`Ane` were
//! name reservations with no lowering and are deleted, not carried; `Cuda`
//! is an optional driver (`cuda-driver`) over `crate::cuda`'s source emitter,
//! and is a variant of [`GpuDriver`] only when that feature is enabled. See
//! `docs/bench-campaigns/2026-09-03-gpu-one-risc/design-2026-09-04/design-final.md`
//! §B.4 for the collapse this replaces (`Backend`'s prior seven variants,
//! three of which executed).
//!
//! [`GpuDriver::for_target`] resolves the driver ONCE per compiled target
//! from cargo features and `target_os`, mirroring the `cfg!` cascade this
//! module always ran per call. [`plan_named`] additionally takes an
//! `Option<GpuDriver>` override so a caller that has BOTH drivers compiled in
//! (a parity harness measuring Metal against wgpu on the same host) can force
//! one rather than accept whichever `for_target` prefers; production callers
//! pass `None` and get the same per-target resolution [`GpuDriver::for_target`]
//! always gave.
//!
//! # Selection is per-call, not process-wide
//!
//! [`plan_named`] takes `engine: Engine` as a plain argument — an explicit
//! choice made by the caller for THIS call, not a cached global one call
//! reads and every later call inherits. [`Engine::from_env`] exists only as
//! a convenience a caller may use to *compute* that argument (the same
//! environment resolution performed by [`proxima_tensor::cpu`]'s
//! `matmul_worker_count` for `PROXIMA_MATMUL_WORKERS`), never as
//! something `plan_named`/`execute_plan_named` consult on their own — so one
//! process can plan one program on [`Engine::Cpu`] and the next on
//! [`Engine::Gpu`] without touching an environment variable in between.
//! `OMEGA_BACKEND` still accepts the legacy `metal`/`wgpu` names for one
//! release, mapped to `Engine::Gpu` with a [`proxima_telemetry::warn!`]
//! deprecation event — see [`Engine::from_env`]'s own doc.
//!
//! # Why plan/execute is not a `Pipe`
//!
//! Adjudicated 2026-08-30 with both call sites written out: wrapping the pair
//! as a pipe passes the pipe question (it is the `AdamStep` shape — interior-
//! mutable state, `call` delegating to the free function) but fails the
//! relocation question. The call sites are line-for-line equivalent, every
//! resilience combinator requires `SendPipe` while an interior-mutable `Plan`
//! is `!Sync`, and nothing downstream consumes `Evaluated` as a pipe input —
//! so a wrapper would relocate the impl without enabling any composition. The
//! free-function pair stays.

use crate::config::{RuntimeConfig, SelectionMode};
use conflaguration::Validate;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "cuda-driver")]
use std::sync::OnceLock;
use std::time::Instant;

use proxima_tensor::cpu::ExpertSource;
use proxima_tensor::{Evaluated, NodeId, NumericPolicy, Op, QuantizedBlock, TensorError};

#[cfg(feature = "cpu")]
use crate::msl::Codec;
#[cfg(feature = "cpu")]
use proxima_tensor::config::TensorExecutionConfig;
#[cfg(feature = "cpu")]
use proxima_tensor::cpu::{
    cpu_tensor_reserved_bytes, evaluate_quantized_named_exact_with_scratch_and_experts,
    evaluate_quantized_named_with_config, evaluate_quantized_named_with_scratch_and_experts,
};
#[cfg(feature = "cpu")]
use proxima_tensor::resolve_named_blocks;

#[cfg(feature = "cuda-driver")]
use crate::cuda_driver::{CudaDriver, CudaDriverError};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::metal::{self, MetalError};
#[cfg(feature = "wgpu-backend")]
use crate::wgpu_driver::{self, WgpuError};

#[cfg(feature = "cpu")]
type CpuInputContract = (NodeId, Option<Codec>, bool, usize);

/// Where an op runs. Two variants, because there are two places: a CPU core
/// and a GPU. `Metal`/`Wgpu`/`Vulkan`/`Cuda` are DRIVERS of the Gpu engine,
/// not peers of the Cpu engine ([`GpuDriver`]), and `Npu`/`Ane` were name
/// reservations with no lowering — deleted, not carried. A caller can still
/// name `Engine::Gpu` in a build with neither GPU driver feature on and get
/// back an honest [`BackendError::NoGpuDriver`] instead of a type that does
/// not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine {
    Cpu,
    Gpu,
}

impl Engine {
    /// The name [`core::str::FromStr`] parses back into this variant — used
    /// for error messages and for [`Engine::from_env`]'s own parsing, so the
    /// two never drift on what an engine is called.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Engine::Cpu => "cpu",
            Engine::Gpu => "gpu",
        }
    }

    /// Reads `OMEGA_BACKEND` for this resolution. Parsing is cheap and keeping
    /// the read live makes test and embedding-process environment changes
    /// deterministic instead of pinning the first caller's value.
    ///
    /// Accepts `cpu`/`gpu`. For one release it also accepts the legacy
    /// `metal`/`wgpu` names, mapped to [`Engine::Gpu`] with a
    /// [`proxima_telemetry::warn!`] deprecation event — a caller that named a
    /// driver instead of an engine gets the engine, once, loudly, rather than
    /// a silent behavior change.
    ///
    /// This is a DEFAULT a caller may use to compute the `engine` argument
    /// [`plan_named`] takes; it is never read by [`plan_named`] or
    /// [`execute_plan_named`] themselves, so calling this once and then
    /// calling `plan_named` with an explicit [`Engine`] on the very next line
    /// runs that program on whichever engine was passed, not on whatever
    /// this returned.
    ///
    /// Unset or empty falls back to [`Engine::Gpu`] when
    /// [`GpuDriver::for_target`] resolves a driver, otherwise [`Engine::Cpu`].
    /// A name nothing above recognizes is an error, not a silent fallback —
    /// a typo in `OMEGA_BACKEND` must not be free to run on whatever engine
    /// happened to be compiled in instead.
    ///
    /// # Errors
    /// [`BackendError::UnknownName`] when `OMEGA_BACKEND` is set to a name
    /// this does not recognize.
    pub fn from_env() -> Result<Engine, BackendError> {
        let raw = std::env::var("OMEGA_BACKEND").unwrap_or_default();
        if raw.is_empty() {
            return Ok(Engine::default_compiled());
        }
        match raw.as_str() {
            "metal" | "wgpu" => {
                warn_deprecated_driver_name(&raw);
                Ok(Engine::Gpu)
            }
            _ => raw
                .parse::<Engine>()
                .inspect_err(|error| warn_unknown_backend_env(&raw, error)),
        }
    }

    fn default_compiled() -> Engine {
        if GpuDriver::for_target().is_some() {
            Engine::Gpu
        } else {
            Engine::Cpu
        }
    }
}

/// Which driver renders the Gpu engine on this target. Resolved ONCE at
/// plan/schedule time from the compiled features and `target_os`, never
/// carried per op — every Gpu op on a host uses the same driver unless a
/// caller of [`plan_named`] explicitly overrides it (a parity harness
/// comparing both drivers on one host).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuDriver {
    Metal,
    Wgpu,
    #[cfg(feature = "cuda-driver")]
    Cuda,
}

impl GpuDriver {
    /// The resolution this module's `cfg!` cascade always performed per
    /// call, hoisted to one place instead of once per `match backend` arm.
    /// Prefers Metal when both drivers are compiled (the driver a
    /// GPU-capable macOS caller actually wants by default).
    #[must_use]
    pub const fn for_target() -> Option<Self> {
        if cfg!(all(feature = "metal", target_os = "macos")) {
            Some(Self::Metal)
        } else if cfg!(feature = "wgpu-backend") {
            Some(Self::Wgpu)
        } else if cfg!(feature = "cuda-driver") {
            #[cfg(feature = "cuda-driver")]
            {
                Some(Self::Cuda)
            }
            #[cfg(not(feature = "cuda-driver"))]
            {
                None
            }
        } else {
            None
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            GpuDriver::Metal => "metal",
            GpuDriver::Wgpu => "wgpu",
            #[cfg(feature = "cuda-driver")]
            GpuDriver::Cuda => "cuda",
        }
    }

    /// Whether this driver renders the fused `BoundOpKind::CachedAttention`, whose bound
    /// excludes the pad rows of a bucketed KV range. A driver that runs the unfused
    /// chain (CUDA, wgpu) sees every pad row, so its KV range must be bound exact.
    #[must_use]
    pub const fn fuses_cached_attention(self) -> bool {
        matches!(self, GpuDriver::Metal)
    }
}

/// Emits the telemetry event for an unrecognized `OMEGA_BACKEND` value.
fn warn_unknown_backend_env(value: &str, error: &BackendError) {
    proxima_telemetry::warn!(%value, %error, "OMEGA_BACKEND names no known backend");
}

/// Emits the deprecation event for the legacy `metal`/`wgpu` `OMEGA_BACKEND`
/// spelling — see [`Engine::from_env`]'s own doc for the one-release grace
/// period this backs.
fn warn_deprecated_driver_name(value: &str) {
    proxima_telemetry::warn!(
        %value,
        "OMEGA_BACKEND names a GPU driver, not an engine; use `gpu` instead (this alias is removed next release)"
    );
}

impl core::str::FromStr for Engine {
    type Err = BackendError;

    fn from_str(value: &str) -> Result<Engine, BackendError> {
        match value {
            "cpu" => Ok(Engine::Cpu),
            "gpu" => Ok(Engine::Gpu),
            other => Err(BackendError::UnknownName {
                name: other.to_string(),
            }),
        }
    }
}

/// Everything the backend-agnostic wrapper can fail with: an unrecognized
/// engine name, an engine whose feature is not compiled in or that has no
/// execution arm yet, a Gpu request with no compiled driver at all, or a
/// failure the underlying evaluator itself (CPU, Metal, or wgpu) produced.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Boxed: the validation error is 104 bytes and would make every `Result<_, BackendError>` large.
    #[error("invalid runtime configuration: {0}")]
    Configuration(Box<conflaguration::Error>),
    #[error("backend `{backend}` cannot enforce `{field}`: {reason}")]
    UnsupportedConstraint {
        backend: &'static str,
        field: &'static str,
        reason: &'static str,
    },
    #[error("backend `{backend}` needs {requested} bytes for `{field}`, exceeding {limit}")]
    ResourceLimit {
        backend: &'static str,
        field: &'static str,
        requested: u64,
        limit: u64,
    },
    #[error("backend `{backend}` overflowed `{field}` accounting")]
    ResourceOverflow {
        backend: &'static str,
        field: &'static str,
    },
    #[error("input contract changed for {node:?}: {reason}")]
    InputContractChanged { node: NodeId, reason: &'static str },
    #[error("no admissible execution candidate")]
    NoAdmissibleCandidate {
        candidates: Vec<CandidateMeasurement>,
    },
    #[error("unknown engine name `{name}`; known engines: cpu, gpu")]
    UnknownName { name: String },

    /// The named engine's cargo feature is off, so nothing behind it was
    /// compiled — never a fallback to whatever IS compiled.
    #[error("backend `{backend}` needs the `{feature}` cargo feature, which is not compiled in")]
    NotCompiled {
        backend: &'static str,
        feature: &'static str,
    },

    /// [`Engine::Gpu`] was requested (directly, or via [`GpuDriver::for_target`]
    /// falling through [`Engine::from_env`]'s default) but neither the
    /// `metal` nor the `wgpu-backend` feature is compiled in.
    #[error("engine `gpu` has no compiled driver; enable the `metal` or `wgpu-backend` feature")]
    NoGpuDriver,

    /// The selected GPU driver has no per-expert buffer-table binding yet.
    /// Refusing a nonempty table prevents a residency decision from being
    /// silently replaced by the original contiguous expert stack.
    #[error("backend `{backend}` cannot bind {source_count} expert source tables")]
    ExpertSourcesUnsupported {
        backend: &'static str,
        source_count: usize,
    },

    /// The named backend's feature is on (its name is reserved,
    /// `Cargo.toml`), but `backend.rs` has no execution arm for it yet —
    /// distinct from [`BackendError::NotCompiled`] so the message tells a
    /// caller which fix applies: turn on a feature, or wait for the driver.
    #[error("backend `{backend}` is compiled in but has no execution arm implemented yet")]
    NotImplemented { backend: &'static str },

    #[error(transparent)]
    Tensor(#[from] TensorError),

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error(transparent)]
    Metal(#[from] MetalError),

    #[cfg(all(
        feature = "metal",
        target_os = "macos",
        feature = "metal-attn-variants"
    ))]
    #[error(transparent)]
    Attention(#[from] crate::EmitError),

    #[cfg(all(
        feature = "metal",
        target_os = "macos",
        feature = "metal-attn-variants"
    ))]
    #[error("attention variants require a Metal plan")]
    AttentionVariantRequiresMetal,

    #[cfg(feature = "wgpu-backend")]
    #[error(transparent)]
    Wgpu(#[from] WgpuError),

    #[cfg(feature = "cuda-driver")]
    #[error(transparent)]
    Cuda(#[from] CudaDriverError),
}

impl From<conflaguration::Error> for BackendError {
    fn from(error: conflaguration::Error) -> Self {
        Self::Configuration(Box::new(error))
    }
}

/// [`Plan::Metal`]'s own payload type. Boxed only under
/// `metal-plan-stable-buffers`: that feature's `BufferArena`/`PlanUniforms`
/// fields grow `metal::Plan` well past [`CpuPlan`]'s size, tripping
/// `clippy::large_enum_variant` on this enum regardless of which arm is
/// active. Unboxed with the feature off, matching this enum's behavior
/// before the card existed -- the indirection is the exception the arena
/// earns, not a cost every build pays.
#[cfg(all(
    feature = "metal",
    target_os = "macos",
    feature = "metal-plan-stable-buffers"
))]
type MetalPlanHandle = alloc::boxed::Box<metal::Plan>;
#[cfg(all(
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-plan-stable-buffers")
))]
type MetalPlanHandle = metal::Plan;

/// A resolved, reusable program for exactly one engine+driver — never a
/// cross-engine union. A future scheduler that wants to hold a CPU plan and
/// a Metal plan for the same program side by side holds two `Plan`s, one per
/// engine, and chooses between them per call; this type does not grow a
/// variant that mixes them. Placement of individual ops ACROSS engines in one
/// pass is a later card (design §B.4) and is not this type's job.
pub enum Plan {
    #[cfg(feature = "cpu")]
    Cpu(CpuPlan),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalPlanHandle),
    #[cfg(feature = "wgpu-backend")]
    Wgpu(wgpu_driver::WgpuPlan),
    #[cfg(feature = "cuda-driver")]
    Cuda(crate::cuda_driver::CudaPlan),
}

/// Raw cold and synchronized execution observations for one candidate.
#[derive(Debug, Clone)]
pub struct CandidateMeasurement {
    pub engine: Engine,
    pub driver: Option<GpuDriver>,
    pub device_identity: String,
    pub cold_plan_ns: Option<u128>,
    pub warmup_ns: Vec<u128>,
    pub execution_ns: Vec<u128>,
    pub execution_sum_ns: Option<u128>,
    pub selected: bool,
    pub rejection: Option<String>,
    pub host_tensor_reserved_bytes: Option<u64>,
    pub device_buffer_reserved_bytes: Option<u64>,
    pub largest_device_buffer_bytes: Option<u64>,
    pub resolved_op_count: usize,
    pub kernel_identities: Vec<String>,
}

impl CandidateMeasurement {
    fn new(engine: Engine, driver: Option<GpuDriver>) -> Self {
        Self {
            engine,
            driver,
            device_identity: driver.map_or_else(|| "cpu".into(), |value| value.name().into()),
            cold_plan_ns: None,
            warmup_ns: Vec::new(),
            execution_ns: Vec::new(),
            execution_sum_ns: None,
            selected: false,
            rejection: None,
            host_tensor_reserved_bytes: None,
            device_buffer_reserved_bytes: None,
            largest_device_buffer_bytes: None,
            resolved_op_count: 0,
            kernel_identities: Vec::new(),
        }
    }
}

/// Build an existing backend plan using the resolved runtime configuration.
///
/// The returned [`Plan`] owns reusable backend state. Configured mode performs
/// no trial executions; measured mode retains each synchronized sample in the
/// returned report and selects the smallest observed arithmetic mean.
///
/// ```
/// # #[cfg(feature = "cpu")]
/// # {
/// use omega::backend::{plan_named_with_config, execute_plan_named};
/// use omega::config::RuntimeConfig;
/// use proxima_tensor::{DType, Extent, NodeId, Op, QuantizedBlock, ScalarOp};
/// use proxima_tensor::map::{self, IndexMap};
/// let program = vec![
///     Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(2)], name: Some("input".into()) },
///     Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Negate,
///         operands: vec![(NodeId(0), IndexMap::Affine(map::projection(1, &[0])))], name: None },
/// ];
/// let input = [1.0, 2.0];
/// let named = [("input", QuantizedBlock::Float32(&input))];
/// let (mut plan, report) = plan_named_with_config(&RuntimeConfig::default(), &program, &[], &named, &[NodeId(1)])?;
/// assert!(report[0].execution_ns.is_empty());
/// assert_eq!(execute_plan_named(&mut plan, &named)?.root(), &[-1.0, -2.0]);
/// # }
/// # Ok::<(), omega::backend::BackendError>(())
/// ```
pub fn plan_named_with_config(
    config: &RuntimeConfig,
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<(Plan, Vec<CandidateMeasurement>), BackendError> {
    config.validate()?;
    if config.selection.mode == SelectionMode::Configured {
        let engine = config.engine.parse()?;
        let driver = configured_driver(config, engine)?;
        let mut measurement = CandidateMeasurement::new(engine, driver);
        let started = Instant::now();
        let plan =
            plan_configured_candidate(config, engine, driver, program, symbols, named, outputs)?;
        measurement.cold_plan_ns = Some(started.elapsed().as_nanos());
        fill_candidate_identity(
            &plan,
            config,
            program,
            symbols,
            named,
            outputs,
            &mut measurement,
        )?;
        measurement.selected = true;
        return Ok((plan, vec![measurement]));
    }
    let mut measurements: Vec<CandidateMeasurement> = Vec::new();
    let configured_engine = config.engine.parse()?;
    let preferred_driver = configured_driver(config, configured_engine);
    let ordered = ordered_candidates(config)?;
    match preferred_driver {
        Err(error) => {
            let mut rejected = CandidateMeasurement::new(configured_engine, None);
            rejected.device_identity = config.gpu_driver.clone();
            rejected.rejection = Some(error.to_string());
            measurements.push(rejected);
        }
        Ok(driver) if !ordered.contains(&(configured_engine, driver)) => {
            let mut rejected = CandidateMeasurement::new(configured_engine, driver);
            let started = Instant::now();
            rejected.rejection = plan_configured_candidate(
                config,
                configured_engine,
                driver,
                program,
                symbols,
                named,
                outputs,
            )
            .err()
            .map(|error| error.to_string());
            rejected.cold_plan_ns = Some(started.elapsed().as_nanos());
            measurements.push(rejected);
        }
        Ok(_) => {}
    }
    let mut winner: Option<(Plan, usize)> = None;
    for (engine, driver) in ordered {
        let mut measurement = CandidateMeasurement::new(engine, driver);
        let started = Instant::now();
        let planned =
            plan_configured_candidate(config, engine, driver, program, symbols, named, outputs);
        measurement.cold_plan_ns = Some(started.elapsed().as_nanos());
        match planned {
            Ok(mut plan) => {
                let sampled = fill_candidate_identity(
                    &plan,
                    config,
                    program,
                    symbols,
                    named,
                    outputs,
                    &mut measurement,
                )
                .and_then(|()| measure_candidate(&mut plan, named, config, &mut measurement));
                if let Err(error) = sampled {
                    measurement.rejection = Some(error.to_string());
                } else {
                    let improves = winner.as_ref().is_none_or(|(_, index)| {
                        candidate_is_better(&measurement, &measurements[*index])
                    });
                    if improves {
                        winner = Some((plan, measurements.len()));
                    }
                }
            }
            Err(error) => measurement.rejection = Some(error.to_string()),
        }
        measurements.push(measurement);
    }
    let Some((plan, index)) = winner else {
        return Err(BackendError::NoAdmissibleCandidate {
            candidates: measurements,
        });
    };
    measurements[index].selected = true;
    Ok((plan, measurements))
}

fn candidate_is_better(candidate: &CandidateMeasurement, incumbent: &CandidateMeasurement) -> bool {
    candidate.rejection.is_none()
        && candidate.execution_sum_ns.is_some()
        && candidate.execution_ns.len() == incumbent.execution_ns.len()
        && candidate.execution_sum_ns < incumbent.execution_sum_ns
}

#[cfg(any(test, feature = "wgpu-backend", feature = "cuda-driver"))]
pub(crate) fn checked_resource_sum(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values.into_iter().try_fold(0_u64, u64::checked_add)
}

fn prefer_candidate<T: PartialEq>(candidates: &mut Vec<T>, preferred: &T) {
    if let Some(position) = candidates
        .iter()
        .position(|candidate| candidate == preferred)
    {
        let preferred = candidates.remove(position);
        candidates.insert(0, preferred);
    }
}

fn configured_driver(
    config: &RuntimeConfig,
    engine: Engine,
) -> Result<Option<GpuDriver>, BackendError> {
    if engine == Engine::Cpu {
        return Ok(None);
    }
    match config.gpu_driver.as_str() {
        "auto" => GpuDriver::for_target()
            .map(Some)
            .ok_or(BackendError::NoGpuDriver),
        "metal" => Ok(Some(GpuDriver::Metal)),
        "wgpu" => Ok(Some(GpuDriver::Wgpu)),
        "cuda" => {
            #[cfg(feature = "cuda-driver")]
            {
                Ok(Some(GpuDriver::Cuda))
            }
            #[cfg(not(feature = "cuda-driver"))]
            {
                Err(BackendError::NotCompiled {
                    backend: "cuda",
                    feature: "cuda-driver",
                })
            }
        }
        _ => Err(BackendError::NoGpuDriver),
    }
}

fn ordered_candidates(
    config: &RuntimeConfig,
) -> Result<Vec<(Engine, Option<GpuDriver>)>, BackendError> {
    let candidates: Vec<_> = [
        (cfg!(feature = "cpu"), (Engine::Cpu, None)),
        (
            cfg!(all(feature = "metal", target_os = "macos")),
            (Engine::Gpu, Some(GpuDriver::Metal)),
        ),
        (
            cfg!(feature = "wgpu-backend"),
            (Engine::Gpu, Some(GpuDriver::Wgpu)),
        ),
    ]
    .into_iter()
    .filter_map(|(compiled, candidate)| compiled.then_some(candidate))
    .collect();
    #[cfg(feature = "cuda-driver")]
    let candidates = candidates
        .into_iter()
        .chain([(Engine::Gpu, Some(GpuDriver::Cuda))])
        .collect();
    let mut candidates: Vec<_> = candidates;
    let engine = config.engine.parse()?;
    if let Ok(driver) = configured_driver(config, engine) {
        prefer_candidate(&mut candidates, &(engine, driver));
    }
    Ok(candidates)
}

fn measure_candidate(
    plan: &mut Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    config: &RuntimeConfig,
    measurement: &mut CandidateMeasurement,
) -> Result<(), BackendError> {
    for _ in 0..config.selection.resolved_warmup_runs() {
        let started = Instant::now();
        let result = execute_plan_named(plan, named)?;
        measurement.warmup_ns.push(started.elapsed().as_nanos());
        drop(result);
    }
    let mut sum = 0_u128;
    for _ in 0..config.selection.resolved_measurement_runs() {
        let started = Instant::now();
        let result = execute_plan_named(plan, named)?;
        let elapsed = started.elapsed().as_nanos();
        measurement.execution_ns.push(elapsed);
        sum = sum
            .checked_add(elapsed)
            .ok_or(BackendError::ResourceOverflow {
                backend: "selection",
                field: "execution_sum_ns",
            })?;
        drop(result);
    }
    measurement.execution_sum_ns = Some(sum);
    Ok(())
}

fn plan_configured_candidate(
    config: &RuntimeConfig,
    engine: Engine,
    driver: Option<GpuDriver>,
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<Plan, BackendError> {
    if engine == Engine::Cpu {
        #[cfg(feature = "cpu")]
        {
            for (field, value) in [
                (
                    "max_device_buffer_bytes",
                    config.constraints.max_device_buffer_bytes,
                ),
                (
                    "max_single_device_buffer_bytes",
                    config.constraints.max_single_device_buffer_bytes,
                ),
            ] {
                if value.is_some() {
                    return Err(BackendError::UnsupportedConstraint {
                        backend: "cpu",
                        field,
                        reason: "cpu plans own no device buffers",
                    });
                }
            }
            let input_contract = capture_input_contract(program, symbols, named)?;
            let execution = TensorExecutionConfig::builder()
                .numeric(config.numeric)
                .allow_fusion(config.allow_fusion)
                .exact_activations(config.exact_activations)
                .maybe_max_host_tensor_bytes(config.constraints.max_host_tensor_bytes)
                .build();
            if let Some(limit) = execution.max_host_tensor_bytes {
                let requested =
                    cpu_tensor_reserved_bytes(program, symbols, named, outputs, &[], &execution)?;
                if requested > limit {
                    return Err(BackendError::ResourceLimit {
                        backend: "cpu",
                        field: "max_host_tensor_bytes",
                        requested,
                        limit,
                    });
                }
            }
            return Ok(Plan::Cpu(CpuPlan {
                program: program.to_vec(),
                symbols: symbols.to_vec(),
                outputs: outputs.to_vec(),
                free_buffers: Vec::new(),
                validated_weight_nodes: None,
                exact_activations: config.exact_activations,
                execution_config: Some(execution),
                input_contract,
            }));
        }
        #[cfg(not(feature = "cpu"))]
        return Err(BackendError::NotCompiled {
            backend: "cpu",
            feature: "cpu",
        });
    }
    plan_configured_gpu(config, driver, program, symbols, named, outputs)
}

fn plan_configured_gpu(
    config: &RuntimeConfig,
    driver: Option<GpuDriver>,
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<Plan, BackendError> {
    let _ = (config, program, symbols, named, outputs);
    let driver = driver.ok_or(BackendError::NoGpuDriver)?;
    match driver {
        GpuDriver::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                Ok(Plan::Metal(
                    metal::plan_named_with_config(program, symbols, named, outputs, config)
                        .map_err(map_metal_configuration_error)?
                        .into(),
                ))
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Err(BackendError::NotCompiled {
                    backend: "metal",
                    feature: "metal",
                })
            }
        }
        GpuDriver::Wgpu => {
            #[cfg(feature = "wgpu-backend")]
            {
                Ok(Plan::Wgpu(
                    wgpu_driver::plan_named_with_config(config, program, symbols, named, outputs)
                        .map_err(map_wgpu_configuration_error)?,
                ))
            }
            #[cfg(not(feature = "wgpu-backend"))]
            {
                Err(BackendError::NotCompiled {
                    backend: "wgpu",
                    feature: "wgpu-backend",
                })
            }
        }
        #[cfg(feature = "cuda-driver")]
        GpuDriver::Cuda => Ok(Plan::Cuda(
            cuda_driver()?
                .plan_with_config(program, symbols, outputs, named, config)
                .map_err(map_cuda_configuration_error)?,
        )),
    }
}

#[cfg(feature = "cuda-driver")]
fn map_cuda_configuration_error(error: CudaDriverError) -> BackendError {
    match error {
        CudaDriverError::UnsupportedConstraint { field, reason } => {
            BackendError::UnsupportedConstraint {
                backend: "cuda",
                field,
                reason,
            }
        }
        CudaDriverError::ResourceLimit {
            field,
            requested,
            limit,
        } => BackendError::ResourceLimit {
            backend: "cuda",
            field,
            requested,
            limit,
        },
        CudaDriverError::ResourceOverflow => BackendError::ResourceOverflow {
            backend: "cuda",
            field: "tensor_payload_bytes",
        },
        CudaDriverError::InputContractChanged { node, reason } => {
            BackendError::InputContractChanged { node, reason }
        }
        other => BackendError::Cuda(other),
    }
}

#[cfg(feature = "wgpu-backend")]
fn map_wgpu_configuration_error(error: WgpuError) -> BackendError {
    match error {
        WgpuError::ResourceLimit {
            field,
            requested,
            limit,
        } => BackendError::ResourceLimit {
            backend: "wgpu",
            field,
            requested,
            limit,
        },
        WgpuError::ResourceOverflow => BackendError::ResourceOverflow {
            backend: "wgpu",
            field: "tensor_payload_bytes",
        },
        WgpuError::InputContractChanged { node, reason } => {
            BackendError::InputContractChanged { node, reason }
        }
        other => BackendError::Wgpu(other),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn map_metal_configuration_error(error: MetalError) -> BackendError {
    match error {
        MetalError::UnsupportedConstraint { field, reason } => {
            BackendError::UnsupportedConstraint {
                backend: "metal",
                field,
                reason,
            }
        }
        MetalError::ResourceLimit {
            field,
            requested,
            limit,
        } => BackendError::ResourceLimit {
            backend: "metal",
            field,
            requested,
            limit,
        },
        MetalError::ResourceOverflow { field } => BackendError::ResourceOverflow {
            backend: "metal",
            field,
        },
        MetalError::InputContractChanged { node, reason } => {
            BackendError::InputContractChanged { node, reason }
        }
        other => BackendError::Metal(other),
    }
}

fn fill_candidate_identity(
    plan: &Plan,
    config: &RuntimeConfig,
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
    measurement: &mut CandidateMeasurement,
) -> Result<(), BackendError> {
    let shapes = proxima_tensor::infer(program, symbols)?;
    let resolved = if config.allow_fusion {
        proxima_tensor::bind_with_fusion(program, &shapes, outputs, true, config.numeric)?
    } else {
        proxima_tensor::bind_unfused(program, &shapes, outputs, config.numeric)?
    };
    measurement.resolved_op_count = resolved.len();
    measurement.kernel_identities = resolved
        .iter()
        .map(|bound| format!("{}:{:?}", bound.node.0, bound.kind))
        .collect();
    #[cfg(any(
        feature = "cpu",
        all(feature = "metal", target_os = "macos"),
        feature = "wgpu-backend",
        feature = "cuda-driver"
    ))]
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(cpu) => {
            if let Some(execution) = &cpu.execution_config
                && execution.max_host_tensor_bytes.is_some()
            {
                measurement.host_tensor_reserved_bytes = Some(cpu_tensor_reserved_bytes(
                    program,
                    symbols,
                    named,
                    outputs,
                    &cpu.free_buffers,
                    execution,
                )?);
            }
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal) => {
            measurement.device_identity = metal.device_identity()?;
            measurement.resolved_op_count = metal.resolved_op_count();
            measurement.kernel_identities = metal.kernel_keys()?;
            if let Some((host, total, largest)) = metal.scoped_reservation() {
                measurement.host_tensor_reserved_bytes = Some(host);
                measurement.device_buffer_reserved_bytes = Some(total);
                measurement.largest_device_buffer_bytes = Some(largest);
            }
        }
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(wgpu) => {
            measurement.device_identity = wgpu.device_identity().to_owned();
            measurement.resolved_op_count = wgpu.resolved_op_count();
            measurement.kernel_identities = wgpu.kernel_identities()?;
            let (host, total, largest) = wgpu.scoped_reservation()?;
            measurement.host_tensor_reserved_bytes = Some(host);
            measurement.device_buffer_reserved_bytes = Some(total);
            measurement.largest_device_buffer_bytes = Some(largest);
        }
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(cuda) => {
            measurement.device_identity = cuda.device_identity()?;
            measurement.resolved_op_count = cuda.resolved_op_count();
            measurement.kernel_identities = cuda.kernel_identities();
            if config.constraints.max_host_tensor_bytes.is_some()
                || config.constraints.max_device_buffer_bytes.is_some()
                || config.constraints.max_single_device_buffer_bytes.is_some()
            {
                let (host, total, largest) = cuda.scoped_reservation()?;
                measurement.host_tensor_reserved_bytes = Some(host);
                measurement.device_buffer_reserved_bytes = Some(total);
                measurement.largest_device_buffer_bytes = Some(largest);
            }
        }
    }
    let _ = (plan, named);
    Ok(())
}

#[cfg(feature = "cpu")]
fn capture_input_contract(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Vec<CpuInputContract>, BackendError> {
    let shapes = proxima_tensor::infer(program, symbols)?;
    let blocks = resolve_named_blocks(program, named)?;
    proxima_tensor::block_node_ids(program)
        .iter()
        .zip(blocks)
        .map(|(node, block)| {
            let expected = shapes
                .of(*node)
                .iter()
                .try_fold(1_usize, |count, extent| {
                    count.checked_mul(usize::try_from(*extent).ok()?)
                })
                .ok_or(BackendError::ResourceOverflow {
                    backend: "cpu",
                    field: "input_elements",
                })?;
            if block.element_count()? != expected {
                return Err(BackendError::InputContractChanged {
                    node: *node,
                    reason: "input element count does not match the planned shape",
                });
            }
            let (codec, integer) = block_codec_identity(block);
            Ok((*node, codec, integer, expected))
        })
        .collect()
}

#[cfg(feature = "cpu")]
fn block_codec_identity(block: QuantizedBlock<'_>) -> (Option<Codec>, bool) {
    match block {
        QuantizedBlock::Float32(_) => (None, false),
        QuantizedBlock::Int32(_) => (None, true),
        QuantizedBlock::Packed { codec, .. } => (Some(codec), false),
    }
}

#[cfg(feature = "cpu")]
fn validate_input_contract(
    program: &[Op],
    named: &[(&str, QuantizedBlock<'_>)],
    contract: &[CpuInputContract],
) -> Result<(), BackendError> {
    let blocks = resolve_named_blocks(program, named)?;
    for ((node, codec, integer, elements), block) in contract.iter().zip(blocks) {
        if block_codec_identity(block) != (*codec, *integer) {
            return Err(BackendError::InputContractChanged {
                node: *node,
                reason: "input codec differs from the planned codec",
            });
        }
        if block.element_count()? != *elements {
            return Err(BackendError::InputContractChanged {
                node: *node,
                reason: "input element count differs from the planned shape",
            });
        }
    }
    Ok(())
}

/// The CPU arm's plan state. `proxima_tensor::cpu` has no persistent
/// plan/execute split of its own — `evaluate_quantized_with_scratch`
/// re-runs `infer`/`bind` every call — so "planning" here is exactly the
/// caller-owned pieces that DO persist across calls: the program itself
/// (owned, so the [`Plan`] outlives the caller's borrowed slices) and the
/// reusable scratch
/// [`evaluate_quantized_named_with_scratch_and_experts`] takes, so
/// repeated [`execute_plan_named`] calls keep reusing the same buffers
/// instead of reintroducing the per-call allocation that function's own
/// `scratch` parameters exist to avoid.
#[cfg(feature = "cpu")]
pub struct CpuPlan {
    program: Vec<Op>,
    symbols: Vec<u64>,
    outputs: Vec<NodeId>,
    free_buffers: Vec<Vec<f32>>,
    validated_weight_nodes: Option<std::collections::BTreeSet<NodeId>>,
    // `false` for every existing `plan_named`/`plan_named_cpu` caller --
    // routes through the SAME `q{4,5,6}k-int8-dot` fast path this plan
    // always has. `true` only for a plan built via `plan_named_cpu_exact`
    // (`plan_named_exact`'s `Engine::Cpu` arm): the CPU reference a
    // cross-backend parity harness compares Metal's exact kernels against
    // must itself be exact -- see `proxima_tensor::cpu::evaluate_quantized_exact`'s
    // own doc for the finding this exists to fix.
    exact_activations: bool,
    execution_config: Option<TensorExecutionConfig>,
    input_contract: Vec<CpuInputContract>,
}

/// Resolves a program into a reusable [`Plan`] for `engine`, binding blocks
/// by NAME through [`resolve_named_blocks`] — the same function the CPU
/// evaluator and the Metal driver both already call, so this wrapper cannot
/// introduce a second, drifting name-to-position mapping.
///
/// `gpu_driver` is read only when `engine == Engine::Gpu`: `None` resolves
/// through [`GpuDriver::for_target`] (the production default — one driver per
/// compiled target); `Some(driver)` forces that driver regardless of which
/// `for_target` would have preferred, for a caller with both GPU features
/// compiled that wants to measure them against each other on one host. A
/// `Cpu` engine ignores `gpu_driver` entirely.
///
/// # Errors
/// [`BackendError::NotCompiled`] if the resolved engine/driver's feature is
/// off, [`BackendError::NoGpuDriver`] if `Engine::Gpu` resolves to no driver
/// at all, otherwise whatever the chosen evaluator itself rejects (unresolved
/// names, shape mismatches, unsupported dtypes).
// leading underscores: with every backend feature off (a bare `std`-only
// build), no arm below reads these -- the same "unused unless a feature
// reads it" shape `mark_resident`'s own `_resident_names` documents above.
pub fn plan_named(
    engine: Engine,
    gpu_driver: Option<GpuDriver>,
    _program: &[Op],
    _symbols: &[u64],
    _named: &[(&str, QuantizedBlock<'_>)],
    _outputs: &[NodeId],
    _numeric_policy: NumericPolicy,
) -> Result<Plan, BackendError> {
    match engine {
        Engine::Cpu => {
            #[cfg(feature = "cpu")]
            {
                plan_named_cpu(_program, _symbols, _named, _outputs)
            }
            #[cfg(not(feature = "cpu"))]
            {
                Err(BackendError::NotCompiled {
                    backend: "cpu",
                    feature: "cpu",
                })
            }
        }
        Engine::Gpu => {
            let driver = gpu_driver
                .or_else(GpuDriver::for_target)
                .ok_or(BackendError::NoGpuDriver)?;
            match driver {
                GpuDriver::Metal => {
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    {
                        plan_named_metal(_program, _symbols, _named, _outputs, _numeric_policy)
                    }
                    #[cfg(not(all(feature = "metal", target_os = "macos")))]
                    {
                        Err(BackendError::NotCompiled {
                            backend: "metal",
                            feature: "metal",
                        })
                    }
                }
                GpuDriver::Wgpu => {
                    #[cfg(feature = "wgpu-backend")]
                    {
                        plan_named_wgpu(_program, _symbols, _named, _outputs, _numeric_policy)
                    }
                    #[cfg(not(feature = "wgpu-backend"))]
                    {
                        Err(BackendError::NotCompiled {
                            backend: "wgpu",
                            feature: "wgpu-backend",
                        })
                    }
                }
                #[cfg(feature = "cuda-driver")]
                GpuDriver::Cuda => {
                    plan_named_cuda(_program, _symbols, _named, _outputs, _numeric_policy)
                }
            }
        }
    }
}

/// Runs an already-resolved [`Plan`] against fresh named block data — the
/// serving-loop entry point, called once per token with the CPU's scratch
/// (or the Metal driver's device buffers) reused from the previous call
/// rather than rebuilt.
///
/// # Errors
/// Whatever the chosen backend's own evaluator rejects (unresolved names,
/// shape mismatches, device/driver failures).
// leading underscore on `named`: with every backend feature off, `Plan` has
// no variants and the match below reduces to its never-pattern arm alone,
// which never reads it -- same shape `mark_resident`'s `_resident_names`
// documents below.
pub fn execute_plan_named(
    plan: &mut Plan,
    _named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, BackendError> {
    execute_plan_named_with_resident_names(plan, _named, None, None)
}

/// Executes a named plan with a per-step expert substitution table. CPU
/// consumes the table through `proxima-tensor`'s expert-aware evaluator;
/// Metal lowers the descriptor table directly and selects the packed decoder
/// for each routed expert, including mixed-codec HOBBIT substitutions.
///
/// # Errors
/// In addition to [`execute_plan_named`]'s errors, Metal rejects malformed
/// tables (empty, unpacked, unsupported-codec, or wrong-node) with a typed
/// error.
pub fn execute_plan_named_with_expert_sources(
    plan: &mut Plan,
    _named: &[(&str, QuantizedBlock<'_>)],
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
) -> Result<Evaluated, BackendError> {
    execute_plan_named_with_resident_names(plan, _named, None, expert_sources)
}

/// Executes a named plan while allowing immutable model inputs to remain on
/// the device between calls. `resident_names` contains only caller-owned bytes
/// whose contents are immutable for the plan's lifetime.
// leading underscore on `expert_sources`: same shape as `_named` and
// `_resident_names` above -- with every backend feature off, only the
// never-pattern arm survives cfg-stripping and it never reads the table.
pub fn execute_plan_named_with_resident_names(
    plan: &mut Plan,
    _named: &[(&str, QuantizedBlock<'_>)],
    _resident_names: Option<&BTreeSet<&str>>,
    _expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
) -> Result<Evaluated, BackendError> {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(cpu_plan) => execute_plan_named_cpu(cpu_plan, _named, _expert_sources),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal_plan) => match _expert_sources {
            Some(sources) if !sources.is_empty() => Ok(
                metal::execute_plan_named_with_expert_sources(metal_plan, _named, sources)?,
            ),
            _ => execute_plan_named_metal(metal_plan, _named),
        },
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(wgpu_plan) => {
            reject_gpu_expert_sources("wgpu", _expert_sources)?;
            execute_plan_named_wgpu(wgpu_plan, _named)
        }
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(cuda_plan) => {
            reject_gpu_expert_sources("cuda", _expert_sources)?;
            Ok(cuda_plan
                .execute_named(_named, _resident_names)
                .map_err(map_cuda_configuration_error)?)
        }
        // `Plan` is uninhabited with every backend feature off; `*plan {}`
        // is the never-pattern proof of that rather than a runtime `todo!`.
        #[cfg(not(any(
            feature = "cpu",
            all(feature = "metal", target_os = "macos"),
            feature = "wgpu-backend",
            feature = "cuda-driver"
        )))]
        _ => match *plan {},
    }
}

#[cfg(any(feature = "wgpu-backend", feature = "cuda-driver", test))]
fn reject_gpu_expert_sources(
    backend: &'static str,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
) -> Result<(), BackendError> {
    let source_count = expert_sources.map_or(0, BTreeMap::len);
    if source_count == 0 {
        return Ok(());
    }
    Err(BackendError::ExpertSourcesUnsupported {
        backend,
        source_count,
    })
}

/// Classifies every named block bound to one of `resident_names` as data
/// that never changes across calls, so the macOS `Plan::Metal` driver may cache and
/// reuse its device buffer instead of re-copying it every call — see
/// `metal::Plan::mark_resident`'s own doc for the full mechanism and the
/// soundness argument for why this needs a caller-supplied name set rather
/// than being inferred from bytes alone. A no-op on [`Plan::Cpu`]: the CPU
/// evaluator has no device buffer to cache — that no-op is the match arm
/// below (`Plan::Cpu(_) => {}`), by construction, not this parameter being
/// dropped at the signature.
// `resident_names` is genuinely read by the metal arm below and genuinely
// unread in a `cpu`-only build with `metal` cfg'd out (no such arm exists
// there at all) -- the cfg_attr states that per-build fact directly instead
// of leaving a permanent leading-underscore that reads as "always discarded".
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(
        unused_variables,
        reason = "only the metal arm below reads this in this build"
    )
)]
pub fn mark_resident(plan: &mut Plan, resident_names: &std::collections::BTreeSet<&str>) {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => {}
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal_plan) => metal_plan.mark_resident(resident_names),
        // v1's `wgpu_driver::WgpuPlan` re-uploads every block on every
        // `execute_plan` call -- see that module's own doc for why residency
        // caching is out of v1 scope.
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => {}
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(cuda_plan) => cuda_plan.mark_resident(resident_names),
        // `Plan` is uninhabited with every backend feature off; `*plan {}`
        // is the never-pattern proof of that rather than a runtime `todo!`.
        #[cfg(not(any(
            feature = "cpu",
            all(feature = "metal", target_os = "macos"),
            feature = "wgpu-backend",
            feature = "cuda-driver"
        )))]
        _ => match *plan {},
    }
}

#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(
        unused_variables,
        reason = "only the metal arm below reads this in this build"
    )
)]
pub fn mark_plan_time_constants_resident(plan: &mut Plan) {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => {}
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal_plan) => metal_plan.mark_plan_time_constants_resident(),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => {}
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => {}
        #[cfg(not(any(
            feature = "cpu",
            all(feature = "metal", target_os = "macos"),
            feature = "wgpu-backend",
            feature = "cuda-driver"
        )))]
        _ => match *plan {},
    }
}

/// Returns the physical output-slot bytes retained by a resolved plan's
/// Metal arena. Other backends have no persistent device arena and return
/// zero; the caller uses this only for an allocation census.
#[must_use]
pub fn plan_arena_allocated_bytes(plan: &Plan) -> usize {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => 0,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal_plan) => metal_plan.arena_allocated_bytes().unwrap_or(0),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => 0,
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => 0,
        #[cfg(not(any(
            feature = "cpu",
            all(feature = "metal", target_os = "macos"),
            feature = "wgpu-backend",
            feature = "cuda-driver"
        )))]
        _ => match *plan {},
    }
}

/// Narrows [`metal::MathMode`] on [`Plan::Metal`] -- see that type's own doc
/// for the measured rationale (`proxima-tensor/docs/discipline.md` ROW
/// 296/297) and [`metal::Plan::set_math_mode`]'s own doc for why this can
/// now fail (a `math_mode` needing a permission the plan's bound
/// [`NumericPolicy`] does not grant). A no-op `Ok(())` on every other arm:
/// `Plan::Cpu`'s interpreter has no `MTLCompileOptions` to set, and v1's
/// `wgpu_driver::WgpuPlan` has no analogous knob. Gated on `metal`+macos
/// (unlike [`mark_resident`]) because its own parameter,
/// [`metal::MathMode`], only exists in that build.
///
/// # Errors
/// [`metal::MetalError::NumericPolicyMismatch`] when `math_mode` needs a
/// permission the plan's bound policy does not grant.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn set_math_mode(plan: &mut Plan, math_mode: metal::MathMode) -> Result<(), metal::MetalError> {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => Ok(()),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal_plan) => metal_plan.set_math_mode(math_mode),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => Ok(()),
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => Ok(()),
        #[cfg(not(any(
            feature = "cpu",
            all(feature = "metal", target_os = "macos"),
            feature = "wgpu-backend",
            feature = "cuda-driver"
        )))]
        _ => match *plan {},
    }
}

/// Moves a cached plan to `symbols` in place when only the key range of its
/// fused cached-attention ops changes -- [`metal::Plan::refit_symbols`]'s own
/// doc has the contract. `Ok(false)` on every other arm: `Plan::Cpu` keeps
/// the symbols it was planned at and re-binds per call, and v1's
/// `wgpu_driver::WgpuPlan`/`CudaPlan` carry no refit, so the caller builds a
/// fresh plan exactly as before.
///
/// # Errors
/// [`metal::MetalError`] when shape inference or kernel resolution fails at
/// `symbols`; the plan is unchanged.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn refit_symbols(plan: &mut Plan, symbols: &[u64]) -> Result<bool, metal::MetalError> {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => Ok(false),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal_plan) => metal_plan.refit_symbols(symbols),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => Ok(false),
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => Ok(false),
        #[cfg(not(any(
            feature = "cpu",
            all(feature = "metal", target_os = "macos"),
            feature = "wgpu-backend",
            feature = "cuda-driver"
        )))]
        _ => match *plan {},
    }
}

/// Sets [`metal::DispatchType`] on [`Plan::Metal`] -- see that type's own
/// doc for the measured rationale (`proxima-tensor/docs/discipline.md` ROW
/// 311/312). A no-op on every other arm: `Plan::Cpu`'s interpreter has no
/// compute encoder to open, and v1's `wgpu_driver::WgpuPlan` has no
/// analogous knob. Gated on `metal`+macos (unlike [`mark_resident`]) because
/// its own parameter, [`metal::DispatchType`], only exists in that build.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn set_dispatch_type(plan: &mut Plan, dispatch_type: metal::DispatchType) {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => {}
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal_plan) => metal_plan.set_dispatch_type(dispatch_type),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => {}
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => {}
        #[cfg(not(any(
            feature = "cpu",
            all(feature = "metal", target_os = "macos"),
            feature = "wgpu-backend",
            feature = "cuda-driver"
        )))]
        _ => match *plan {},
    }
}

/// Applies the selected cached-attention axes to a Metal plan before its
/// attention pipelines resolve.
#[cfg(all(
    feature = "metal",
    target_os = "macos",
    feature = "metal-attn-variants"
))]
pub fn set_attention_variant(
    plan: &mut Plan,
    variant: crate::AttentionVariant,
) -> Result<(), BackendError> {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => Err(BackendError::AttentionVariantRequiresMetal),
        Plan::Metal(metal_plan) => Ok(metal_plan.set_attention_variant(variant)?),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => Err(BackendError::AttentionVariantRequiresMetal),
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => Err(BackendError::AttentionVariantRequiresMetal),
        #[cfg(not(any(feature = "cpu", feature = "wgpu-backend", feature = "cuda-driver")))]
        _ => match *plan {},
    }
}

/// Sets [`metal::Plan::command_buffer_chunks`] on [`Plan::Metal`] --
/// `ServingConfig::command_buffer_chunks`'s own doc for the config-sourced
/// default this feeds `execute_plan_with_placements`'s chunked-submission
/// loop. A no-op on every other arm, same scoping as [`set_dispatch_type`].
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn set_command_buffer_chunks(plan: &mut Plan, chunks: u32, decode_shaped: bool) {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => {}
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal_plan) => metal_plan.set_command_buffer_chunks(chunks, decode_shaped),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => {}
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => {}
        #[cfg(not(any(
            feature = "cpu",
            all(feature = "metal", target_os = "macos"),
            feature = "wgpu-backend",
            feature = "cuda-driver"
        )))]
        _ => match *plan {},
    }
}

/// Registers the page-aligned, process-lifetime mapping backing a loaded
/// checkpoint's tensor bytes -- see `metal::register_checkpoint_mapping`'s
/// own doc for the mechanism this feeds. A no-op unless the Metal backend is
/// compiled in: the CPU evaluator has no device buffer to address by offset,
/// and v1's wgpu driver re-uploads every block every call regardless (see
/// [`mark_resident`]'s own doc for that same scoping).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn register_checkpoint_mapping(bytes: &[u8]) {
    metal::register_checkpoint_mapping(bytes);
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn register_checkpoint_mapping(_bytes: &[u8]) {}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn register_expert_mapping(bytes: &[u8]) {
    metal::register_expert_mapping(bytes);
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn register_expert_mapping(_bytes: &[u8]) {}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn clear_expert_source_cache() {
    metal::clear_expert_source_cache();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn clear_expert_source_cache() {}

/// Evicts a dropped checkpoint's own resident weight names from the Metal
/// driver's device-buffer caches -- see [`metal::release_resident_names`]'s
/// own doc for the mechanism and why it evicts by name rather than
/// clearing every cached entry. A no-op unless the Metal backend is
/// compiled in, matching every other Metal-only knob in this module
/// ([`mark_resident`], [`register_checkpoint_mapping`]).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn release_resident_names<'name>(names: impl IntoIterator<Item = &'name str>) {
    metal::release_resident_names(names);
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[cfg_attr(
    not(feature = "cuda-driver"),
    allow(
        unused_variables,
        reason = "only the cuda arm below reads this in this build"
    )
)]
pub fn release_resident_names<'name>(names: impl IntoIterator<Item = &'name str>) {
    #[cfg(feature = "cuda-driver")]
    if let Some(Ok(driver)) = CUDA_DRIVER.get() {
        driver.release_resident_names(names);
    }
}

/// Unregisters the checkpoint mapping [`register_checkpoint_mapping`]
/// installed for `bytes` -- see [`metal::unregister_checkpoint_mapping`]'s
/// own doc for why this is safe to call unconditionally at drop time even
/// when a later model already superseded the mapping. A no-op unless the
/// Metal backend is compiled in, matching [`register_checkpoint_mapping`]
/// above.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn unregister_checkpoint_mapping(bytes: &[u8]) {
    metal::unregister_checkpoint_mapping(bytes);
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn unregister_checkpoint_mapping(_bytes: &[u8]) {}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn unregister_expert_mapping(bytes: &[u8]) {
    metal::unregister_expert_mapping(bytes);
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn unregister_expert_mapping(_bytes: &[u8]) {}

#[cfg(feature = "cpu")]
fn plan_named_cpu(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<Plan, BackendError> {
    // fails fast on an unresolvable name here, mirroring `metal::plan_named`'s
    // own eager check, even though the CPU evaluator itself re-resolves names
    // on every `execute_plan_named` call (it has no persistent bind step to
    // cache into).
    for node in proxima_tensor::block_node_ids(program) {
        let name = program[node.0 as usize]
            .name()
            .ok_or(proxima_tensor::error::TensorError::UnnamedInput(node))?;
        if !name.contains("_exps.weight") && !named.iter().any(|(candidate, _)| *candidate == name)
        {
            return Err(BackendError::Tensor(
                proxima_tensor::error::TensorError::UnboundInputName(name.to_owned()),
            ));
        }
    }
    Ok(Plan::Cpu(CpuPlan {
        program: program.to_vec(),
        symbols: symbols.to_vec(),
        outputs: outputs.to_vec(),
        free_buffers: Vec::new(),
        validated_weight_nodes: None,
        exact_activations: false,
        execution_config: None,
        input_contract: Vec::new(),
    }))
}

/// [`plan_named_cpu`]'s exact-activation counterpart -- the ONLY
/// difference is [`CpuPlan::exact_activations`], read back by
/// [`execute_plan_named_cpu`] to pick
/// [`evaluate_quantized_named_exact_with_scratch_and_experts`] over
/// [`evaluate_quantized_named_with_scratch_and_experts`]. A sibling function
/// rather
/// than a parameter on [`plan_named_cpu`] itself so every one of that
/// function's existing callers (`plan_named`'s `Engine::Cpu` arm) is
/// untouched.
#[cfg(feature = "cpu")]
fn plan_named_cpu_exact(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<Plan, BackendError> {
    resolve_named_blocks(program, named)?;
    Ok(Plan::Cpu(CpuPlan {
        program: program.to_vec(),
        symbols: symbols.to_vec(),
        outputs: outputs.to_vec(),
        free_buffers: Vec::new(),
        validated_weight_nodes: None,
        exact_activations: true,
        execution_config: None,
        input_contract: Vec::new(),
    }))
}

/// [`plan_named`]'s exact-CPU-activation counterpart: identical for every
/// `Engine::Gpu` driver (Metal's own kernels are already exact -- see
/// `proxima_tensor::cpu::evaluate_quantized_exact`'s own doc), and routes
/// `Engine::Cpu` through `plan_named_cpu_exact` instead of
/// `plan_named_cpu`. The seam a cross-backend parity harness
/// (`proxima-model-interop`'s quality harness) uses to build a CPU
/// reference that carries the SAME zero activation-quantization error
/// Metal's kernels do, instead of the `q{4,5,6}k-int8-dot` fast path's own
/// ~1e-3 relative error.
///
/// # Errors
/// Same as [`plan_named`].
pub fn plan_named_exact(
    engine: Engine,
    gpu_driver: Option<GpuDriver>,
    _program: &[Op],
    _symbols: &[u64],
    _named: &[(&str, QuantizedBlock<'_>)],
    _outputs: &[NodeId],
    _numeric_policy: NumericPolicy,
) -> Result<Plan, BackendError> {
    match engine {
        Engine::Cpu => {
            #[cfg(feature = "cpu")]
            {
                plan_named_cpu_exact(_program, _symbols, _named, _outputs)
            }
            #[cfg(not(feature = "cpu"))]
            {
                Err(BackendError::NotCompiled {
                    backend: "cpu",
                    feature: "cpu",
                })
            }
        }
        Engine::Gpu => plan_named(
            engine,
            gpu_driver,
            _program,
            _symbols,
            _named,
            _outputs,
            _numeric_policy,
        ),
    }
}

#[cfg(feature = "cpu")]
fn execute_plan_named_cpu(
    plan: &mut CpuPlan,
    named: &[(&str, QuantizedBlock<'_>)],
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
) -> Result<Evaluated, BackendError> {
    let evaluated = if let Some(config) = &plan.execution_config {
        validate_input_contract(&plan.program, named, &plan.input_contract)?;
        evaluate_quantized_named_with_config(
            &plan.program,
            &plan.symbols,
            named,
            &plan.outputs,
            (&mut plan.free_buffers, &mut plan.validated_weight_nodes),
            expert_sources,
            config,
        )?
    } else if plan.exact_activations {
        evaluate_quantized_named_exact_with_scratch_and_experts(
            &plan.program,
            &plan.symbols,
            named,
            &plan.outputs,
            &mut plan.free_buffers,
            &mut plan.validated_weight_nodes,
            expert_sources,
        )?
    } else {
        evaluate_quantized_named_with_scratch_and_experts(
            &plan.program,
            &plan.symbols,
            named,
            &plan.outputs,
            &mut plan.free_buffers,
            &mut plan.validated_weight_nodes,
            expert_sources,
        )?
    };
    Ok(evaluated)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn plan_named_metal(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Plan, BackendError> {
    let plan = metal::plan_named(program, symbols, named, outputs, numeric_policy)?;
    // `.into()` covers both `MetalPlanHandle` shapes: identity when it is
    // `metal::Plan` itself, `Box::from` (`impl<T> From<T> for Box<T>`) when
    // `metal-plan-stable-buffers` makes it `Box<metal::Plan>`.
    Ok(Plan::Metal(plan.into()))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn execute_plan_named_metal(
    plan: &metal::Plan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, BackendError> {
    let evaluated = metal::execute_plan_named_with_placements(plan, named, &[], &[])
        .map_err(map_metal_configuration_error)?;
    Ok(evaluated)
}

#[cfg(feature = "wgpu-backend")]
fn plan_named_wgpu(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Plan, BackendError> {
    let plan =
        wgpu_driver::plan_named_with_policy(program, symbols, named, outputs, numeric_policy)?;
    Ok(Plan::Wgpu(plan))
}

#[cfg(feature = "wgpu-backend")]
fn execute_plan_named_wgpu(
    plan: &mut wgpu_driver::WgpuPlan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, BackendError> {
    let evaluated =
        wgpu_driver::execute_plan_named(plan, named).map_err(map_wgpu_configuration_error)?;
    Ok(evaluated)
}

#[cfg(feature = "cuda-driver")]
fn plan_named_cuda(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Plan, BackendError> {
    Ok(Plan::Cuda(cuda_driver()?.plan(
        program,
        symbols,
        outputs,
        numeric_policy,
        named,
    )?))
}

#[cfg(feature = "cuda-driver")]
static CUDA_DRIVER: OnceLock<Result<CudaDriver, String>> = OnceLock::new();

#[cfg(feature = "cuda-driver")]
fn cuda_driver() -> Result<&'static CudaDriver, BackendError> {
    let driver = match CUDA_DRIVER
        .get_or_init(|| crate::cuda_driver::CudaDriver::new(0).map_err(|error| error.to_string()))
    {
        Ok(driver) => driver,
        Err(error) => {
            return Err(BackendError::Cuda(CudaDriverError::Driver(error.clone())));
        }
    };
    Ok(driver)
}

/// Diagnostic counterpart of [`execute_plan_named`], reachable only when a
/// caller already holds a [`Plan::Metal`] -- see [`metal::execute_plan_op_timed`]'s
/// own doc for why this must never replace [`execute_plan_named`] on the
/// serving loop. Returns [`BackendError::NotImplemented`] for any other
/// `Plan` variant rather than panicking: a caller asking this of a CPU plan
/// asked the wrong question, not an unreachable one.
///
/// # Errors
/// Propagates the Metal driver's per-op timing failures; reports a non-Metal
/// plan as [`BackendError::NotImplemented`].
#[cfg(all(feature = "metal", target_os = "macos", feature = "instrument"))]
pub fn execute_plan_named_metal_op_timed(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    cpu_reference: Option<&std::collections::BTreeMap<NodeId, Vec<f32>>>,
) -> Result<(Evaluated, Vec<metal::OpGpuTiming>), BackendError> {
    match plan {
        Plan::Metal(metal_plan) => Ok(metal::execute_plan_named_op_timed(
            metal_plan,
            named,
            cpu_reference,
        )?),
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => Err(BackendError::NotImplemented { backend: "cpu" }),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => Err(BackendError::NotImplemented { backend: "wgpu" }),
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => Err(BackendError::NotImplemented { backend: "cuda" }),
    }
}

/// Routed-expert counterpart of [`execute_plan_named_metal_op_timed`].
/// The diagnostic preserves the plan's mixed-codec expert substitutions
/// while timing each bound operation independently.
///
/// # Errors
/// Propagates the Metal driver's name-resolution, expert-source, and timing
/// failures; reports a non-Metal plan as [`BackendError::NotImplemented`].
#[cfg(all(feature = "metal", target_os = "macos", feature = "instrument"))]
pub fn execute_plan_named_metal_op_timed_with_expert_sources(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    expert_sources: &std::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
    cpu_reference: Option<&std::collections::BTreeMap<NodeId, Vec<f32>>>,
) -> Result<(Evaluated, Vec<metal::OpGpuTiming>), BackendError> {
    match plan {
        Plan::Metal(metal_plan) => Ok(metal::execute_plan_named_op_timed_with_expert_sources(
            metal_plan,
            named,
            expert_sources,
            cpu_reference,
        )?),
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => Err(BackendError::NotImplemented { backend: "cpu" }),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => Err(BackendError::NotImplemented { backend: "wgpu" }),
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => Err(BackendError::NotImplemented { backend: "cuda" }),
    }
}

#[cfg(all(test, feature = "cpu"))]
// hand-built fixtures: an expect failure IS the test failing, same as `mod tests` below.
#[allow(clippy::expect_used)]
mod programmable_tests;

#[cfg(test)]
// test fixtures below are hand-built to succeed; an expect/unwrap failure IS
// the test failing, same convention as every `omega/tests/*.rs` file.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use proxima_tensor::NodeId;
    use proxima_tensor::cpu::ExpertSource;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use proxima_tensor::{
        DType, Extent, IndexMap, NumericPolicy, Op, QuantizedBlock, ScalarOp, append, projection,
    };

    use super::{BackendError, Engine, GpuDriver};
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use super::MetalError;

    static ENVIRONMENT_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn only_the_metal_driver_renders_the_fused_cached_attention_bound() {
        assert!(GpuDriver::Metal.fuses_cached_attention());
        assert!(!GpuDriver::Wgpu.fuses_cached_attention());
        #[cfg(feature = "cuda-driver")]
        assert!(!GpuDriver::Cuda.fuses_cached_attention());
    }

    #[test]
    fn gpu_expert_source_gate_rejects_a_nonempty_substitution_table() {
        let mut sources = BTreeMap::new();
        sources.insert(NodeId(7), ExpertSource::new(&[]));

        assert!(super::reject_gpu_expert_sources("metal", None).is_ok());
        assert!(super::reject_gpu_expert_sources("metal", Some(&BTreeMap::new())).is_ok());
        assert!(matches!(
            super::reject_gpu_expert_sources("metal", Some(&sources)),
            Err(BackendError::ExpertSourcesUnsupported {
                backend: "metal",
                source_count: 1,
            })
        ));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_plan_rejects_malformed_expert_sources_at_the_public_execution_boundary() {
        let mut program = Vec::new();
        let input = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(1)],
                name: Some("input".into()),
            },
        );
        let output = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(input, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        let values = [1.0f32];
        let named = [("input", QuantizedBlock::Float32(&values))];
        let mut plan = super::plan_named(
            Engine::Gpu,
            Some(GpuDriver::Metal),
            &program,
            &[],
            &named,
            &[output],
            NumericPolicy::default(),
        )
        .expect("plans the Metal identity program");
        let mut sources = BTreeMap::new();
        sources.insert(input, ExpertSource::new(&[]));

        let error =
            super::execute_plan_named_with_expert_sources(&mut plan, &named, Some(&sources))
                .expect_err("Metal must not silently discard a malformed expert source table");

        assert!(matches!(
            error,
            BackendError::Metal(MetalError::ExpertSourceUnsupported { node, .. }) if node == input
        ));
    }

    #[test]
    fn every_engine_name_round_trips_through_from_str() {
        for engine in [Engine::Cpu, Engine::Gpu] {
            let parsed: Engine = engine
                .name()
                .parse()
                .expect("every engine's own name parses back");
            assert_eq!(parsed, engine);
        }
    }

    #[test]
    fn an_unknown_engine_name_lists_the_known_ones() {
        let error = "quantum"
            .parse::<Engine>()
            .expect_err("quantum names no engine");
        let message = error.to_string();
        assert!(message.contains("quantum"));
        for known in ["cpu", "gpu"] {
            assert!(
                message.contains(known),
                "error should name {known}: {message}"
            );
        }
    }

    // `super::Plan` deliberately carries no `Debug` (its Metal variant wraps
    // device-buffer handles it does not implement `Debug` for either), so
    // `expect_err`/`unwrap_err` cannot be called on `Result<Plan, _>`
    // directly -- this pulls the error out by hand instead.
    // only feature-gated tests below call this, and no single cargo feature
    // combination compiles all three of them at once.
    #[allow(dead_code)]
    fn expect_plan_err(result: Result<super::Plan, BackendError>, message: &str) -> BackendError {
        match result {
            Ok(_) => panic!("{message}"),
            Err(error) => error,
        }
    }

    #[cfg(not(feature = "cpu"))]
    #[test]
    fn requesting_cpu_without_the_feature_errors_naming_it() {
        let error = expect_plan_err(
            super::plan_named(
                Engine::Cpu,
                None,
                &[],
                &[],
                &[],
                &[],
                proxima_tensor::NumericPolicy::default(),
            ),
            "cpu engine must not be selectable when its feature is off",
        );
        assert!(matches!(
            error,
            BackendError::NotCompiled {
                backend: "cpu",
                feature: "cpu"
            }
        ));
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    #[test]
    fn requesting_metal_without_the_feature_errors_naming_it() {
        let error = expect_plan_err(
            super::plan_named(
                Engine::Gpu,
                Some(super::GpuDriver::Metal),
                &[],
                &[],
                &[],
                &[],
                proxima_tensor::NumericPolicy::default(),
            ),
            "metal driver must not be selectable when its feature is off",
        );
        assert!(matches!(
            error,
            BackendError::NotCompiled {
                backend: "metal",
                feature: "metal"
            }
        ));
    }

    // Environment parsing is intentionally live so these tests remain
    // deterministic under both nextest and plain cargo test.
    #[test]
    fn from_env_with_a_known_name_resolves_to_its_variant() {
        let _guard = ENVIRONMENT_LOCK
            .lock()
            .expect("environment lock is not poisoned");
        // SAFETY: this test owns `OMEGA_BACKEND` for its own process; nextest
        // runs each test in a separate process, so no concurrent reader.
        unsafe {
            std::env::set_var("OMEGA_BACKEND", "cpu");
        }
        let engine = super::Engine::from_env().expect("cpu is a known engine name");
        assert_eq!(engine, super::Engine::Cpu);
    }

    #[test]
    fn from_env_with_an_unknown_name_errors_without_falling_back() {
        let _guard = ENVIRONMENT_LOCK
            .lock()
            .expect("environment lock is not poisoned");
        // SAFETY: see `from_env_with_a_known_name_resolves_to_its_variant`.
        unsafe {
            std::env::set_var("OMEGA_BACKEND", "quantum");
        }
        let error = super::Engine::from_env().expect_err("quantum names no engine");
        assert!(matches!(
            error,
            BackendError::UnknownName { name } if name == "quantum"
        ));
    }

    #[test]
    fn from_env_unset_falls_back_to_the_compiled_default() {
        let _guard = ENVIRONMENT_LOCK
            .lock()
            .expect("environment lock is not poisoned");
        // SAFETY: this test owns `OMEGA_BACKEND` for its own process; remove
        // an inherited shell value so the fallback assertion is deterministic.
        unsafe {
            std::env::remove_var("OMEGA_BACKEND");
        }
        let engine = super::Engine::from_env().expect("unset falls back, never errors");
        assert_eq!(engine, super::Engine::default_compiled());
    }

    #[cfg(not(any(
        all(feature = "metal", target_os = "macos"),
        feature = "wgpu-backend",
        feature = "cuda-driver"
    )))]
    #[test]
    fn requesting_gpu_with_no_compiled_driver_never_falls_back_to_cpu() {
        let error = expect_plan_err(
            super::plan_named(
                Engine::Gpu,
                None,
                &[],
                &[],
                &[],
                &[],
                proxima_tensor::NumericPolicy::default(),
            ),
            "gpu with no driver compiled must never silently run on cpu",
        );
        assert!(matches!(error, BackendError::NoGpuDriver));
    }
}
