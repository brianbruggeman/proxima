//! `proxima::sync::oneshot` — single-value channel, shape-compatible
//! with `tokio::sync::oneshot`. Backed by `futures::channel::oneshot`
//! (no runtime coupling). The receiver's `await` resolves to
//! `Result<T, Canceled>` rather than tokio's `RecvError`; callers that
//! only care about success/failure (`.ok()`, `.is_err()`) port
//! verbatim. Code that explicitly matches on `RecvError` must adapt to
//! [`Canceled`].
//!
//! # Non-coverage
//!
//! - `Sender::closed().await` — sender-side wait for the receiver to
//!   drop. `futures::channel::oneshot::Sender::cancellation()`
//!   provides an equivalent if a caller needs it; not re-exported
//!   because no internal caller uses it today.
//! - `Sender::is_closed()` IS available via the futures crate's
//!   `poll_canceled` but not exposed here as a sync probe; if needed,
//!   wrap with `futures::future::FutureExt::now_or_never`.
//! - `Receiver::try_recv() -> Result<Option<T>, TryRecvError>` —
//!   non-blocking probe. Compose with `futures::poll!` if needed.
//! - `Receiver::close()` — explicit close from the receiver side.
//!   The futures version drops are sufficient for proxima's usage.

pub use futures::channel::oneshot::{Canceled, Receiver, Sender, channel};
