//! The backend-agnostic entry point: one `plan_named`/`execute_plan_named`
//! pair that runs a named-block program on whichever [`Backend`] the caller
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
//! # Six backends, two implemented
//!
//! [`Backend`] carries a variant for every backend this crate expects to
//! grow into (`Cpu`, `Metal`, `Vulkan`, `Cuda`, `Npu`, `Ane`), not just the
//! two implemented today — naming, parsing (`FromStr`) and error reporting
//! all work for a backend with no driver behind it yet, so adding the next
//! one is a variant, a feature, and one match arm, never a rewrite of the
//! selection mechanism. It stays a plain discriminated enum matched at the
//! dispatch point: the set of backends is closed and known ahead of time,
//! which is exactly the case the workspace's box-free rule reserves for an
//! enum instead of a `dyn Trait` (`dyn` is for an open, unbounded set; this
//! is neither open nor unbounded).
//!
//! # Selection is per-call, not process-wide
//!
//! [`plan_named`] takes `backend: Backend` as a plain argument — an explicit
//! choice made by the caller for THIS call, not a cached global one call
//! reads and every later call inherits. [`Backend::from_env`] exists only as
//! a convenience a caller may use to *compute* that argument (the same
//! env-var-into-`OnceLock` idiom [`proxima_tensor::cpu`]'s own
//! `matmul_worker_count` uses for `PROXIMA_MATMUL_WORKERS`), never as
//! something `plan_named`/`execute_plan_named` consult on their own — so one
//! process can plan one program on [`Backend::Cpu`] and the next on
//! [`Backend::Metal`] without touching an environment variable in between.

use std::sync::OnceLock;

use proxima_tensor::{NodeId, Op, QuantizedBlock, TensorError, Evaluated};

#[cfg(feature = "cpu")]
use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
#[cfg(feature = "cpu")]
use proxima_tensor::resolve_named_blocks;

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::metal::{self, MetalError};
#[cfg(feature = "wgpu-backend")]
use crate::wgpu_driver::{self, WgpuError};

/// Every backend `omega` expects to support, whether or not this build was
/// compiled with the feature behind it. Both fields of a call
/// (`plan_named`'s `backend` argument) and the compiled reality (which
/// cargo features are on) are independent axes on purpose: a caller can
/// still NAME `Backend::Vulkan` in a build that never turned the `vulkan`
/// feature on, and get back an honest [`BackendError`] instead of a type
/// that does not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
    /// The portable `wgpu`/WGSL driver (`crate::wgpu_driver`) — one
    /// abstraction layer over [`Backend::Metal`]: same `BoundOp` descriptor,
    /// same emit-then-drive split, a WGSL emitter and a `wgpu::Device`
    /// instead of an MSL emitter and an `objc2-metal` device. See
    /// `crate::wgsl`'s own doc for what its v1 op set covers.
    Wgpu,
    Vulkan,
    Cuda,
    Npu,
    Ane,
}

impl Backend {
    /// The name [`core::str::FromStr`] parses back into this variant — used
    /// for error messages and for [`Backend::from_env`]'s own parsing, so
    /// the two never drift on what a backend is called.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Metal => "metal",
            Backend::Wgpu => "wgpu",
            Backend::Vulkan => "vulkan",
            Backend::Cuda => "cuda",
            Backend::Npu => "npu",
            Backend::Ane => "ane",
        }
    }

    /// Reads `OMEGA_BACKEND` once per process into a `OnceLock`, mirroring
    /// `proxima_tensor::cpu::matmul_worker_count`'s own idiom for
    /// `PROXIMA_MATMUL_WORKERS` — a per-call `std::env::var` would allocate a
    /// `String` on every plan for a value that cannot change once the
    /// process has started.
    ///
    /// This is a DEFAULT a caller may use to compute the `backend` argument
    /// [`plan_named`] takes; it is never read by [`plan_named`] or
    /// [`execute_plan_named`] themselves, so calling this once and then
    /// calling `plan_named` with an explicit [`Backend`] on the very next
    /// line runs that program on whichever backend was passed, not on
    /// whatever this returned.
    ///
    /// Unset, empty, or a name [`core::str::FromStr`] does not recognize all
    /// fall back to whichever of [`Backend::Metal`]/[`Backend::Cpu`] is
    /// actually compiled in, preferring Metal when both are (the backend a
    /// GPU-capable caller actually wants by default).
    #[must_use]
    pub fn from_env() -> Backend {
        static SELECTED: OnceLock<Backend> = OnceLock::new();
        *SELECTED.get_or_init(|| {
            std::env::var("OMEGA_BACKEND")
                .ok()
                .and_then(|value| value.parse::<Backend>().ok())
                .unwrap_or_else(Backend::default_compiled)
        })
    }

    fn default_compiled() -> Backend {
        if cfg!(all(feature = "metal", target_os = "macos")) {
            Backend::Metal
        } else {
            Backend::Cpu
        }
    }
}

impl core::str::FromStr for Backend {
    type Err = BackendError;

    fn from_str(value: &str) -> Result<Backend, BackendError> {
        match value {
            "cpu" => Ok(Backend::Cpu),
            "metal" => Ok(Backend::Metal),
            "wgpu" => Ok(Backend::Wgpu),
            "vulkan" => Ok(Backend::Vulkan),
            "cuda" => Ok(Backend::Cuda),
            "npu" => Ok(Backend::Npu),
            "ane" => Ok(Backend::Ane),
            other => Err(BackendError::UnknownName {
                name: other.to_string(),
            }),
        }
    }
}

/// Everything the backend-agnostic wrapper can fail with: an unrecognized
/// backend name, a named backend whose feature is not compiled in or that
/// has no execution arm yet, or a failure the underlying evaluator itself
/// (CPU or Metal) produced.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("unknown backend name `{name}`; known backends: cpu, metal, wgpu, vulkan, cuda, npu, ane")]
    UnknownName { name: String },

    /// The named backend's cargo feature is off, so nothing behind it was
    /// compiled — never a fallback to whatever IS compiled.
    #[error("backend `{backend}` needs the `{feature}` cargo feature, which is not compiled in")]
    NotCompiled {
        backend: &'static str,
        feature: &'static str,
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

    #[cfg(feature = "wgpu-backend")]
    #[error(transparent)]
    Wgpu(#[from] WgpuError),
}

/// A resolved, reusable program for exactly one [`Backend`] — never a
/// cross-backend union. A future scheduler that wants to hold a CPU plan and
/// a Metal plan for the same program side by side holds two `Plan`s, one per
/// backend, and chooses between them per call; this type does not grow a
/// variant that mixes them.
pub enum Plan {
    #[cfg(feature = "cpu")]
    Cpu(CpuPlan),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(metal::Plan),
    #[cfg(feature = "wgpu-backend")]
    Wgpu(wgpu_driver::WgpuPlan),
}

/// The CPU arm's plan state. `proxima_tensor::cpu` has no persistent
/// plan/execute split of its own — `evaluate_quantized_with_scratch`
/// re-runs `infer`/`bind` every call — so "planning" here is exactly the
/// caller-owned pieces that DO persist across calls: the program itself
/// (owned, so the [`Plan`] outlives the caller's borrowed slices) and the
/// reusable scratch [`evaluate_quantized_named_with_scratch`] takes, so
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
}

/// Resolves a program into a reusable [`Plan`] for `backend`, binding
/// blocks by NAME through [`resolve_named_blocks`] — the same function the
/// CPU evaluator and the Metal driver both already call, so this wrapper
/// cannot introduce a second, drifting name-to-position mapping.
///
/// # Errors
/// [`BackendError::NotCompiled`] if `backend`'s feature is off,
/// [`BackendError::NotImplemented`] if `backend` is reserved but has no
/// driver yet, otherwise whatever the chosen evaluator itself rejects
/// (unresolved names, shape mismatches, unsupported dtypes).
pub fn plan_named(
    backend: Backend,
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<Plan, BackendError> {
    match backend {
        Backend::Cpu => {
            #[cfg(feature = "cpu")]
            {
                plan_named_cpu(program, symbols, named, outputs)
            }
            #[cfg(not(feature = "cpu"))]
            {
                Err(BackendError::NotCompiled {
                    backend: "cpu",
                    feature: "cpu",
                })
            }
        }
        Backend::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                plan_named_metal(program, symbols, named, outputs)
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Err(BackendError::NotCompiled {
                    backend: "metal",
                    feature: "metal",
                })
            }
        }
        Backend::Wgpu => {
            #[cfg(feature = "wgpu-backend")]
            {
                plan_named_wgpu(program, symbols, named, outputs)
            }
            #[cfg(not(feature = "wgpu-backend"))]
            {
                Err(BackendError::NotCompiled {
                    backend: "wgpu",
                    feature: "wgpu-backend",
                })
            }
        }
        Backend::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                Err(BackendError::NotImplemented { backend: "vulkan" })
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err(BackendError::NotCompiled {
                    backend: "vulkan",
                    feature: "vulkan",
                })
            }
        }
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Err(BackendError::NotImplemented { backend: "cuda" })
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(BackendError::NotCompiled {
                    backend: "cuda",
                    feature: "cuda",
                })
            }
        }
        Backend::Npu => {
            #[cfg(feature = "npu")]
            {
                Err(BackendError::NotImplemented { backend: "npu" })
            }
            #[cfg(not(feature = "npu"))]
            {
                Err(BackendError::NotCompiled {
                    backend: "npu",
                    feature: "npu",
                })
            }
        }
        Backend::Ane => {
            #[cfg(feature = "ane")]
            {
                Err(BackendError::NotImplemented { backend: "ane" })
            }
            #[cfg(not(feature = "ane"))]
            {
                Err(BackendError::NotCompiled {
                    backend: "ane",
                    feature: "ane",
                })
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
pub fn execute_plan_named(
    plan: &mut Plan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, BackendError> {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(cpu_plan) => execute_plan_named_cpu(cpu_plan, named),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal_plan) => execute_plan_named_metal(metal_plan, named),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(wgpu_plan) => execute_plan_named_wgpu(wgpu_plan, named),
    }
}

/// Classifies every named block bound to one of `resident_names` as data
/// that never changes across calls, so [`Plan::Metal`]'s driver may cache and
/// reuse its device buffer instead of re-copying it every call — see
/// [`metal::Plan::mark_resident`]'s own doc for the full mechanism and the
/// soundness argument for why this needs a caller-supplied name set rather
/// than being inferred from bytes alone. A no-op on [`Plan::Cpu`]: the CPU
/// evaluator has no device buffer to cache.
// leading underscore: only the metal arm below reads this, so a Linux
// `cpu`-only build (metal cfg'd out) would otherwise warn on an unused
// parameter -- the binding is still fully used wherever the metal arm exists.
pub fn mark_resident(plan: &mut Plan, _resident_names: &std::collections::BTreeSet<&str>) {
    match plan {
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => {}
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(metal_plan) => metal_plan.mark_resident(_resident_names),
        // v1's `wgpu_driver::WgpuPlan` re-uploads every block on every
        // `execute_plan` call -- see that module's own doc for why residency
        // caching is out of v1 scope.
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => {}
    }
}

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
    resolve_named_blocks(program, named)?;
    Ok(Plan::Cpu(CpuPlan {
        program: program.to_vec(),
        symbols: symbols.to_vec(),
        outputs: outputs.to_vec(),
        free_buffers: Vec::new(),
        validated_weight_nodes: None,
    }))
}

#[cfg(feature = "cpu")]
fn execute_plan_named_cpu(
    plan: &mut CpuPlan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, BackendError> {
    let evaluated = evaluate_quantized_named_with_scratch(
        &plan.program,
        &plan.symbols,
        named,
        &plan.outputs,
        &mut plan.free_buffers,
        &mut plan.validated_weight_nodes,
    )?;
    Ok(evaluated)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn plan_named_metal(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<Plan, BackendError> {
    let plan = metal::plan_named(program, symbols, named, outputs)?;
    Ok(Plan::Metal(plan))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn execute_plan_named_metal(
    plan: &metal::Plan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, BackendError> {
    let evaluated = metal::execute_plan_named(plan, named)?;
    Ok(evaluated)
}

#[cfg(feature = "wgpu-backend")]
fn plan_named_wgpu(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<Plan, BackendError> {
    let plan = wgpu_driver::plan_named(program, symbols, named, outputs)?;
    Ok(Plan::Wgpu(plan))
}

#[cfg(feature = "wgpu-backend")]
fn execute_plan_named_wgpu(
    plan: &mut wgpu_driver::WgpuPlan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, BackendError> {
    let evaluated = wgpu_driver::execute_plan_named(plan, named)?;
    Ok(evaluated)
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
) -> Result<(Evaluated, Vec<metal::OpGpuTiming>), BackendError> {
    match plan {
        Plan::Metal(metal_plan) => Ok(metal::execute_plan_named_op_timed(metal_plan, named)?),
        #[cfg(feature = "cpu")]
        Plan::Cpu(_) => Err(BackendError::NotImplemented { backend: "cpu" }),
    }
}

#[cfg(test)]
// test fixtures below are hand-built to succeed; an expect/unwrap failure IS
// the test failing, same convention as every `omega/tests/*.rs` file.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{Backend, BackendError};

    #[test]
    fn every_backend_name_round_trips_through_from_str() {
        for backend in [
            Backend::Cpu,
            Backend::Metal,
            Backend::Wgpu,
            Backend::Vulkan,
            Backend::Cuda,
            Backend::Npu,
            Backend::Ane,
        ] {
            let parsed: Backend = backend.name().parse().expect("every backend's own name parses back");
            assert_eq!(parsed, backend);
        }
    }

    #[test]
    fn an_unknown_backend_name_lists_the_known_ones() {
        let error = "quantum".parse::<Backend>().expect_err("quantum names no backend");
        let message = error.to_string();
        assert!(message.contains("quantum"));
        for known in ["cpu", "metal", "wgpu", "vulkan", "cuda", "npu", "ane"] {
            assert!(message.contains(known), "error should name {known}: {message}");
        }
    }

    // `super::Plan` deliberately carries no `Debug` (its Metal variant wraps
    // device-buffer handles it does not implement `Debug` for either), so
    // `expect_err`/`unwrap_err` cannot be called on `Result<Plan, _>`
    // directly -- this pulls the error out by hand instead.
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
            super::plan_named(Backend::Cpu, &[], &[], &[], &[]),
            "cpu backend must not be selectable when its feature is off",
        );
        assert!(matches!(
            error,
            BackendError::NotCompiled { backend: "cpu", feature: "cpu" }
        ));
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    #[test]
    fn requesting_metal_without_the_feature_errors_naming_it() {
        let error = expect_plan_err(
            super::plan_named(Backend::Metal, &[], &[], &[], &[]),
            "metal backend must not be selectable when its feature is off",
        );
        assert!(matches!(
            error,
            BackendError::NotCompiled { backend: "metal", feature: "metal" }
        ));
    }

    #[test]
    fn requesting_vulkan_never_falls_back_to_another_backend() {
        let error = expect_plan_err(
            super::plan_named(Backend::Vulkan, &[], &[], &[], &[]),
            "vulkan has no driver yet, compiled feature or not",
        );
        assert!(matches!(
            error,
            BackendError::NotCompiled { backend: "vulkan", .. } | BackendError::NotImplemented { backend: "vulkan" }
        ));
    }
}
