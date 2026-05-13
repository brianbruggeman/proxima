//! Tiny library face — RISC by design. The only thing `Client` does
//! is hold a pipe spec and dispatch `(method, path) -> Response`.
//!
//! There are no `cache()` / `retry()` / `timeout()` builder methods.
//! If you want any of those, put them in the spec — same shape that
//! goes in `proxima.toml`. Sugar exists separately (`proxima::desugar`)
//! and is a pure transformation: feed it a sugary spec, read the
//! desugared output to learn how the primitives compose.
//!
//! ```ignore
//! // plain http
//! let client = proxima::Client::from_value(serde_json::json!({
//!     "http": "https://api.example.com",
//! }))?;
//! let resp = client.call("GET", "/v1/items").send().await?;
//! let body: serde_json::Value = resp.json().await?;
//!
//! // cached http (sugar form — see proxima::desugar for the expansion)
//! let client = proxima::Client::from_sugar(serde_json::json!({
//!     "http": "https://api.example.com",
//!     "cache": true,
//! }))?;
//! ```

pub mod handle;
pub mod request;
pub mod response;

pub use handle::{Client, ClientProtocol, Transport};
pub use request::RequestBuilder;
pub use response::Response;
