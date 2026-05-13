//! `#[proxima::main]` end-to-end on the prime runtime.
//!
//! Proves the macro boots the prime per-core runtime and drives an async
//! body to completion, returning its `Result`. Run with the prime runtime
//! features (the default build has them via `serve-prime`):
//!
//! ```sh
//! cargo run --example proxima_main_demo
//! ```
//!
//! The sibling tokio proof is `examples/proxima_main_tokio_demo.rs`.

use proxima::ProximaError;

#[proxima::main(runtime = "prime")]
async fn main() -> Result<(), ProximaError> {
    let computed = compute().await;
    assert_eq!(computed, 42);
    println!("proxima::main(prime): computed {computed}");
    Ok(())
}

async fn compute() -> u32 {
    let half = async { 21u32 }.await;
    half * 2
}
