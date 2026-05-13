//! `proxima::sync::OnceCell` — async-initialized cell, shape-
//! compatible with `tokio::sync::OnceCell`. Backed by
//! `async_lock::OnceCell`. Direct re-export — `get_or_init`,
//! `get_or_try_init`, `get`, `get_mut`, `take`, `into_inner`,
//! `set`, and `wait` are all present and signature-compatible
//! with the tokio variants used in proxima today.
//!
//! # Non-coverage / minor naming drift
//!
//! - `is_initialized()` vs tokio's `initialized()` — same semantic,
//!   different name. Sites that called `initialized()` will need
//!   one-character rename; documented at the call site if it ever
//!   matters.
//! - `wait()` vs tokio's `wait_initialized()` — same semantic,
//!   different name. Same deal.
//! - `set()` returns `Result<&T, T>` (async-lock); tokio returns
//!   `Result<(), SetError<T>>` (richer error). No proxima caller
//!   uses `set()` today, so the minor divergence stays uncovered.
//!
//! Note: this codebase has multiple `OnceCell`-shaped sites
//! using `std::sync::OnceLock` (terminal-init, sync) rather than
//! `tokio::sync::OnceCell` (async init). Only `client/handle.rs`
//! uses the async-init form; this module is for that site.

pub use async_lock::OnceCell;
