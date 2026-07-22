//! `#[proxima::main]` end-to-end on the prime runtime.
//!
//! Proves the macro boots the prime per-core runtime and drives an async
//! body returning ANY error type, not just `ProximaError` — here a `?`
//! propagates a `ProximaError` into the declared
//! `Box<dyn std::error::Error + Send + Sync>`, the general idiom for a
//! `#[proxima::main]`-driven binary. `Send` is load-bearing on this backend:
//! the prime driver moves the body's output across a driver-core channel, so
//! a bare, non-`Send` `Box<dyn std::error::Error>` does not compile here —
//! that form only works under `runtime = "tokio"`, see
//! `examples/proxima_main_tokio_bare_error_demo.rs`. Run with the prime
//! runtime features (the default build has them via `serve-prime`):
//!
//! ```sh
//! cargo run --example proxima_main_demo
//! ```
//!
//! The sibling tokio proof is `examples/proxima_main_tokio_demo.rs`.

use proxima::ProximaError;

#[proxima::main(runtime = "prime")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let computed = compute().await?;
    assert_eq!(computed, 42);
    println!("proxima::main(prime): computed {computed}");
    Ok(())
}

async fn compute() -> Result<u32, ProximaError> {
    let half = async { 21u32 }.await;
    Ok(half * 2)
}
