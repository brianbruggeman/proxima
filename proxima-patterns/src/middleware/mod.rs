//! Pipe-graph middleware operators — the dep-heavy HTTP leaf.
//!
//! Folded from the former `proxima-middleware` crate. Generic policy
//! (retry, filter, delay, transform, validate, rate_limit, chaos,
//! mutate) lives in `proxima-primitives::pipe`; this module retains only the
//! submodules that pull proxima-auth / [`crate::kv`] / proxima-config's schema module:
//! `auth`, `client_auth`, `context_inject`, and `write_back`.
//!
//! Re-exports from `proxima-primitives::pipe` keep the old `proxima_middleware::*` /
//! `proxima_patterns::middleware::*` paths working for downstream code
//! that has not migrated yet.

#[cfg(feature = "std")]
pub mod auth;
#[cfg(feature = "std")]
pub mod client_auth;
#[cfg(feature = "std")]
pub mod context_inject;
#[cfg(feature = "std")]
pub mod write_back;

/// `base` plus one more pair. `auth` and `write_back` each carried their own
/// copy of this under a different name (`with_extra` / `context_labels_with`)
/// after the fold.
#[cfg(feature = "std")]
pub(crate) fn labels_with(
    base: &proxima_primitives::pipe::telemetry_surface::Labels,
    key: &str,
    value: &str,
) -> proxima_primitives::pipe::telemetry_surface::Labels {
    let mut pairs: Vec<(&str, &str)> = base
        .entries()
        .iter()
        .map(|(name, existing)| (name.as_str(), existing.as_str()))
        .collect();
    pairs.push((key, value));
    proxima_primitives::pipe::telemetry_surface::Labels::from_pairs(&pairs)
}

#[cfg(feature = "std")]
pub use auth::{Auth, AuthFactory};
#[cfg(feature = "std")]
pub use client_auth::{
    ClientAuthFactory, ClientAuthPipe, DigestAuthPipe, OauthAuthPipe, SigV4AuthPipe,
};
#[cfg(feature = "std")]
pub use context_inject::ContextInjector;

// ── re-exports: generic policy moved to proxima-primitives::pipe ────────────────────────────────────
#[cfg(feature = "std")]
pub use proxima_primitives::pipe::capabilities::BytePayload;
#[cfg(feature = "std")]
pub use proxima_primitives::pipe::{
    ChaosBuilder, ChaosConfig, Delay, DelayConfig, DelayFactory, Dist, FilterConfig, FilterFactory,
    KeyExtractor, LatencyFault, MutateOp, Mutation, Predicate, RateLimit, RateLimitCaps,
    RateLimitFactory, RejectMode, RequestOp, ResponseOp, Retry, RetryBudget, RetryFactory,
    RetryPredicate, TokenBucketConfig, Transform, TransformFactory, Validate, ValidateFactory,
    ValidateOp, chaos,
};
#[cfg(feature = "std")]
pub use proxima_primitives::pipe::{ExceededAction, KeyOf, When};
