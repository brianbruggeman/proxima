// P16 — at-creation sampling gate. Consulted before record allocation;
// records returning Decision::Drop are never constructed — zero allocation,
// zero ring push.

// SamplerSpec lives in `config` (std tier); the spec→sampler glue rides with it.
#[cfg(feature = "std")]
use crate::config::SamplerSpec;
use crate::id::TraceId;
use crate::tag::Tag;

/// Pre-allocation sampling gate.
///
/// Consulted by `SpanBuilderWired::start`, `LogEmitBuilder::emit`, and metric
/// emit paths BEFORE record state is allocated. Records returning
/// `Decision::Drop` are never constructed — zero allocation, zero ring push.
///
/// Implementations must be cheap (target ≤10 ns per call). Hot-path code.
///
/// `Box<dyn Sampler>` is the one documented deviation from "no Box<dyn Trait>":
/// samplers need hot-swap (via `Recorder::swap_sampler`) and cross-core sharing,
/// the same justification as `Arc<dyn Exporter>` in C9. User-facing builders
/// and span types remain concrete/generic.
pub trait Sampler: Send + Sync + 'static {
    fn should_sample(&self, ctx: SamplingContext<'_>) -> Decision;
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingContext<'a> {
    pub kind: RecordKind,
    pub name: &'static str,
    pub trace_id: Option<TraceId>,
    pub parent_sampled: Option<bool>,
    pub attrs: &'a [Tag],
}

impl<'a> SamplingContext<'a> {
    pub fn span(name: &'static str) -> Self {
        Self {
            kind: RecordKind::Span,
            name,
            trace_id: None,
            parent_sampled: None,
            attrs: &[],
        }
    }

    pub fn log(name: &'static str) -> Self {
        Self {
            kind: RecordKind::Log,
            name,
            trace_id: None,
            parent_sampled: None,
            attrs: &[],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Keep,
    Drop,
}

impl Decision {
    /// True if the record should be kept. Used by the emit filter, which reuses
    /// this pre-allocation keep/drop decision rather than defining its own.
    #[must_use]
    pub const fn is_keep(self) -> bool {
        matches!(self, Self::Keep)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    Span,
    Event,
    Log,
    Metric,
    Link,
}

/// Keep every record. Default when no sampler configured.
pub struct AlwaysOn;

impl Sampler for AlwaysOn {
    #[inline]
    fn should_sample(&self, _: SamplingContext<'_>) -> Decision {
        Decision::Keep
    }
}

/// Drop every record. Useful for testing or temporary disable.
pub struct AlwaysOff;

impl Sampler for AlwaysOff {
    #[inline]
    fn should_sample(&self, _: SamplingContext<'_>) -> Decision {
        Decision::Drop
    }
}

/// Deterministic ratio sampling using trace_id bytes.
///
/// Same trace_id always produces the same decision. Without a trace_id,
/// falls back to `fastrand::u64(..)`.
pub struct TraceIdRatioBased {
    #[allow(dead_code)]
    p: f64,
    threshold: u64,
}

impl TraceIdRatioBased {
    pub fn new(p: f64) -> Self {
        let clamped = p.clamp(0.0, 1.0);
        let threshold = (clamped * u64::MAX as f64) as u64;
        Self {
            p: clamped,
            threshold,
        }
    }
}

impl Sampler for TraceIdRatioBased {
    #[inline]
    fn should_sample(&self, ctx: SamplingContext<'_>) -> Decision {
        let value = match ctx.trace_id {
            Some(tid) => {
                let bytes = tid.to_bytes();
                u64::from_le_bytes([
                    bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                    bytes[15],
                ])
            }
            None => fastrand::u64(..),
        };
        if value < self.threshold {
            Decision::Keep
        } else {
            Decision::Drop
        }
    }
}

/// Honors upstream sample decision via traceparent.
///
/// Falls back to one of three child samplers depending on context:
/// - `root` when no parent present (no traceparent)
/// - `sampled` when upstream indicated sampled=true
/// - `not_sampled` when upstream indicated sampled=false
pub struct ParentBased {
    root: alloc::boxed::Box<dyn Sampler>,
    sampled: alloc::boxed::Box<dyn Sampler>,
    not_sampled: alloc::boxed::Box<dyn Sampler>,
}

impl ParentBased {
    pub fn new(
        root: alloc::boxed::Box<dyn Sampler>,
        sampled: alloc::boxed::Box<dyn Sampler>,
        not_sampled: alloc::boxed::Box<dyn Sampler>,
    ) -> Self {
        Self {
            root,
            sampled,
            not_sampled,
        }
    }
}

impl Sampler for ParentBased {
    fn should_sample(&self, ctx: SamplingContext<'_>) -> Decision {
        match ctx.parent_sampled {
            None => self.root.should_sample(ctx),
            Some(true) => self.sampled.should_sample(ctx),
            Some(false) => self.not_sampled.should_sample(ctx),
        }
    }
}

/// Convert a `SamplerSpec` into a concrete `Box<dyn Sampler>`.
///
/// Called from `config.rs::Recorder::from_config`; avoids inline `use` inside function body.
#[cfg(feature = "std")]
pub fn spec_to_box(spec: &SamplerSpec) -> alloc::boxed::Box<dyn Sampler> {
    alloc::boxed::Box::<dyn Sampler>::from(spec)
}

#[cfg(feature = "std")]
impl From<&SamplerSpec> for alloc::boxed::Box<dyn Sampler> {
    fn from(spec: &SamplerSpec) -> Self {
        match spec {
            SamplerSpec::AlwaysOn => alloc::boxed::Box::new(AlwaysOn),
            SamplerSpec::AlwaysOff => alloc::boxed::Box::new(AlwaysOff),
            SamplerSpec::TraceIdRatioBased { p } => {
                alloc::boxed::Box::new(TraceIdRatioBased::new(*p))
            }
            SamplerSpec::ParentBased {
                root,
                sampled,
                not_sampled,
            } => alloc::boxed::Box::new(ParentBased::new(
                alloc::boxed::Box::<dyn Sampler>::from(root.as_ref()),
                alloc::boxed::Box::<dyn Sampler>::from(sampled.as_ref()),
                alloc::boxed::Box::<dyn Sampler>::from(not_sampled.as_ref()),
            )),
        }
    }
}

impl Default for alloc::boxed::Box<dyn Sampler> {
    fn default() -> Self {
        alloc::boxed::Box::new(AlwaysOn)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    extern crate std;

    use super::*;
    use crate::id::TraceId;

    fn ctx_span() -> SamplingContext<'static> {
        SamplingContext::span("op")
    }

    fn ctx_span_with_trace(tid: TraceId) -> SamplingContext<'static> {
        SamplingContext {
            kind: RecordKind::Span,
            name: "op",
            trace_id: Some(tid),
            parent_sampled: None,
            attrs: &[],
        }
    }

    #[test]
    fn always_on_keeps_all() {
        let sampler = AlwaysOn;
        for _ in 0..100 {
            assert_eq!(sampler.should_sample(ctx_span()), Decision::Keep);
        }
    }

    #[test]
    fn always_off_drops_all() {
        let sampler = AlwaysOff;
        for _ in 0..100 {
            assert_eq!(sampler.should_sample(ctx_span()), Decision::Drop);
        }
    }

    #[test]
    fn ratio_zero_drops_all() {
        let sampler = TraceIdRatioBased::new(0.0);
        let tid = TraceId::from_bytes([
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10,
        ]);
        for _ in 0..100 {
            assert_eq!(
                sampler.should_sample(ctx_span_with_trace(tid)),
                Decision::Drop
            );
        }
    }

    #[test]
    fn ratio_one_keeps_all() {
        let sampler = TraceIdRatioBased::new(1.0);
        let tid = TraceId::from_bytes([
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10,
        ]);
        for _ in 0..100 {
            assert_eq!(
                sampler.should_sample(ctx_span_with_trace(tid)),
                Decision::Keep
            );
        }
    }

    #[test]
    fn ratio_half_approximately_half() {
        // sampler hashes bytes[8..16] of the trace_id as little-endian u64.
        // bytes[0] = 0x01 keeps the trace_id non-zero (OTel-valid) without
        // pinning any of the hash bytes. fastrand fills bytes[8..16] with
        // uniform random — the high byte must vary for the threshold
        // comparison to actually split the u64 range.
        let sampler = TraceIdRatioBased::new(0.5);
        let total = 10_000usize;
        let kept = (0..total)
            .filter(|_| {
                let mut bytes = [0u8; 16];
                bytes[0] = 0x01;
                for byte in &mut bytes[8..16] {
                    *byte = fastrand::u8(..);
                }
                let tid = TraceId::from_bytes(bytes);
                sampler.should_sample(ctx_span_with_trace(tid)) == Decision::Keep
            })
            .count();
        let pct = kept as f64 / total as f64;
        assert!(
            (0.45..=0.55).contains(&pct),
            "expected ~50% kept, got {:.1}%",
            pct * 100.0
        );
    }

    #[test]
    fn ratio_deterministic_for_same_trace_id() {
        let sampler = TraceIdRatioBased::new(0.5);
        let tid = TraceId::from_bytes([
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
            0x11, 0x22,
        ]);
        let first = sampler.should_sample(ctx_span_with_trace(tid));
        for _ in 0..20 {
            assert_eq!(sampler.should_sample(ctx_span_with_trace(tid)), first);
        }
    }

    #[test]
    fn parent_based_respects_upstream_sampled_true() {
        let sampler = ParentBased::new(
            alloc::boxed::Box::new(AlwaysOff),
            alloc::boxed::Box::new(AlwaysOn),
            alloc::boxed::Box::new(AlwaysOff),
        );
        let ctx = SamplingContext {
            kind: RecordKind::Span,
            name: "op",
            trace_id: None,
            parent_sampled: Some(true),
            attrs: &[],
        };
        assert_eq!(sampler.should_sample(ctx), Decision::Keep);
    }

    #[test]
    fn parent_based_respects_upstream_sampled_false() {
        let sampler = ParentBased::new(
            alloc::boxed::Box::new(AlwaysOn),
            alloc::boxed::Box::new(AlwaysOn),
            alloc::boxed::Box::new(AlwaysOff),
        );
        let ctx = SamplingContext {
            kind: RecordKind::Span,
            name: "op",
            trace_id: None,
            parent_sampled: Some(false),
            attrs: &[],
        };
        assert_eq!(sampler.should_sample(ctx), Decision::Drop);
    }

    #[test]
    fn parent_based_falls_back_to_root_when_no_parent() {
        let sampler = ParentBased::new(
            alloc::boxed::Box::new(AlwaysOff),
            alloc::boxed::Box::new(AlwaysOn),
            alloc::boxed::Box::new(AlwaysOn),
        );
        let ctx = SamplingContext {
            kind: RecordKind::Span,
            name: "op",
            trace_id: None,
            parent_sampled: None,
            attrs: &[],
        };
        assert_eq!(sampler.should_sample(ctx), Decision::Drop);
    }
}
