use burn_import::onnx::{ModelGen, RecordType};

// same mnist.onnx path `proxima-onnx/benches/mnist_f32_lane.rs` reads at
// runtime -- one file, two harnesses, no divergence possible.
const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";

fn main() {
    if cfg!(feature = "embedded-model") {
        ModelGen::new()
            .input(MODEL_PATH)
            .out_dir("model/")
            .record_type(RecordType::Bincode)
            .embed_states(true)
            .run_from_script();
    } else {
        ModelGen::new()
            .input(MODEL_PATH)
            .out_dir("model/")
            .run_from_script();
    }
}
