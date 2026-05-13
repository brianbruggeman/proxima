//! Middleware [`Pipe`](crate::Pipe) implementations — each wraps an
//! inner `Pipe` and transforms request, response, or call decision.
//! Top-down reading order = request execution order; the response
//! unwinds in reverse.
//!
//! ```ignore
//! use proxima::{App, BearerAuth, Composable, HttpUpstream, MountTarget};
//! use proxima::settings::RateLimit;
//! use proxima::middlewares::retry::Retry;
//! use std::time::Duration;
//!
//! let mut app = App::new()?;
//! app.pipe(
//!     "api",
//!     BearerAuth::allow_tokens(["t-1"])           // auth fires first
//!         .then(RateLimit::token_bucket(100, 50)) // then rate limit
//!         .then(Retry::up_to(3))                  // then retry the upstream
//!         .then(HttpUpstream::builder()           // leaf upstream
//!             .url("https://backend.internal")
//!             .timeout(Duration::from_secs(5))
//!             .build()),
//! ).await?;
//! ```
//!
//! Order is not commutative — `Auth` after `Retry` would retry
//! unauthenticated requests; `Retry` after `Auth` retries only
//! authenticated work. Pick deliberately.
//!
//! Each module below is a single middleware. Cross-cutting
//! composition primitives ([`Tee`](crate::Tee),
//! [`Diff`](crate::Diff), [`Isolate`](crate::Isolate),
//! [`Causal`](crate::Causal), [`SwappablePipe`](crate::SwappablePipe),
//! [`WriteBack`](crate::WriteBack)) live alongside as their own
//! top-level modules; see the [`crate::pipe`] module rustdoc for the
//! full menu (substrate primitives, recording-as-Pipe, serving).

pub use proxima_primitives::pipe::diff;
pub use proxima_primitives::pipe::isolate;
pub use proxima_patterns::middleware::auth;
pub use proxima_patterns::middleware::client_auth;
pub use proxima_patterns::middleware::write_back;
pub use proxima_primitives::pipe::rate_limit;
pub use proxima_primitives::pipe::retry;
pub use proxima_primitives::pipe::transform;
pub use proxima_primitives::pipe::validate;

pub use proxima_primitives::pipe::diff::{Diff, diff_handle};
pub use proxima_primitives::pipe::isolate::{Isolate, IsolateFactory};
pub use proxima_patterns::middleware::auth::{Auth, AuthFactory};
pub use proxima_patterns::middleware::client_auth::{ClientAuthFactory, ClientAuthPipe};
pub use proxima_patterns::middleware::write_back::{WriteBack, WriteBackTarget};
pub use proxima_primitives::pipe::rate_limit::{
    ExceededAction, KeyExtractor, KeyOf, RateLimit, RateLimitCaps, RateLimitFactory,
    TokenBucketConfig,
};
pub use proxima_primitives::pipe::retry::{Retry, RetryBudget, RetryFactory, RetryPredicate};
pub use proxima_primitives::pipe::transform::{RequestOp, ResponseOp, Transform, TransformFactory};
pub use proxima_primitives::pipe::validate::{Validate, ValidateFactory};
