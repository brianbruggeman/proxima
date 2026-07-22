//! `#[proxima::main]` on tokio with a BARE, non-`Send` `Box<dyn
//! std::error::Error>`.
//!
//! Proves the achievable boundary of "any error type": `run_tokio` only
//! bounds `F: Future` (no `F::Output: Send`), so a bare `Box<dyn Error>`
//! compiles and runs here. The prime backend cannot accept this same form —
//! it moves the body's output across a driver-core `std::sync::mpsc`
//! channel, which requires `F::Output: Send + 'static` — see
//! `examples/proxima_main_demo.rs` for the `Send + Sync` form that works on
//! both backends.
//!
//! ```sh
//! cargo run --example proxima_main_tokio_bare_error_demo --features tokio
//! ```

use proxima::ProximaError;

#[proxima::main(runtime = "tokio")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let computed = compute().await?;
    assert_eq!(computed, 42);
    println!("proxima::main(tokio, bare Box<dyn Error>): computed {computed}");
    Ok(())
}

async fn compute() -> Result<u32, ProximaError> {
    let half = async { 21u32 }.await;
    Ok(half * 2)
}
