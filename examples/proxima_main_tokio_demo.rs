//! `#[proxima::main]` end-to-end on the tokio runtime.
//!
//! Proves the macro boots a tokio multi-thread runtime and drives an async
//! body returning `std::process::ExitCode` to completion:
//!
//! ```sh
//! cargo run --example proxima_main_tokio_demo
//! ```
//!
//! The sibling prime proof is `examples/proxima_main_demo.rs`.

use std::process::ExitCode;

#[proxima::main(runtime = "tokio", flavor = "multi_thread")]
async fn main() -> ExitCode {
    let ready = tokio::task::yield_now();
    ready.await;
    let computed = async { 6u32 * 7 }.await;
    assert_eq!(computed, 42);
    println!("proxima::main(tokio): computed {computed}");
    ExitCode::SUCCESS
}
