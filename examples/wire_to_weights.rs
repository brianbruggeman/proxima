//! wire_to_weights — one algebra from the NIC to the weights: an HTTP
//! request in, a real `mnist.onnx` forward pass, and a response out,
//! composed end to end as [`Pipe`](proxima::pipe::Pipe)s over
//! `serve_http`. No Python, no serialization seam beyond the wire itself —
//! see `examples/support/wire_to_weights_pipeline.rs` for the composition.
//!
//! Mirrors `examples/h1_native_prime_round_trip.rs` for the serve side
//! (tokio-free h1 over the prime runtime, `PrimeServeExt::serve_http`) and
//! `proxima-onnx/tests/real_mnist_accuracy.rs` for the model side
//! (`parse_complete` -> `lower_graph` -> `evaluate_named`, parsed and
//! lowered exactly once at startup).
//!
//! Presence-guarded: exits cleanly (not an error) when the real
//! `mnist.onnx` checkout is absent, the same convention
//! `real_mnist_accuracy.rs::checkpoint_present` uses for its own
//! host-local fixture.
//!
//!   cargo run --release --example wire_to_weights --features wire-to-weights-demo -- [addr]
//!
//! Request shape: `POST /classify` with exactly `28*28*4 = 3136` bytes of
//! raw little-endian `f32` pixel data (normalized the same way
//! `real_mnist_accuracy.rs` normalizes: `(pixel/255 - 0.1307)/0.3081`).
//! Response: `{"log_probs":[...10 floats...],"digit":N}`.

#[path = "support/wire_to_weights_pipeline.rs"]
mod pipeline;

use std::error::Error;
use std::path::Path;
use std::sync::Arc;

use proxima::prime::PrimeRuntime;
use proxima::runtime::PrimeServeExt;

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";

fn main() -> Result<(), Box<dyn Error>> {
    let Some(model) = pipeline::load_model(Path::new(MODEL_PATH))? else {
        eprintln!("wire_to_weights: skipping, no host-local mnist.onnx checkout at {MODEL_PATH}");
        return Ok(());
    };

    let handler = pipeline::build_handler(Arc::new(model));

    let runtime = Arc::new(PrimeRuntime::builder().cores(1).background_inline().build()?);
    let addr = std::env::args().nth(1).unwrap_or_else(|| "127.0.0.1:0".to_string()).parse()?;
    let handle = runtime.serve_http(addr, handler)?;
    let bound = handle.bind_addr().ok_or("listener did not report a bound address")?;

    // LISTENING line is the smoke test's synchronization point: it starts
    // this binary as a child process and blocks on stdout until this
    // exact line appears, so it never races the bind.
    println!("LISTENING {bound}");
    use std::io::Write as _;
    std::io::stdout().flush()?;

    loop {
        std::thread::park();
    }
}
