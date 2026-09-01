# proxima-onnx op lowering coverage

Generated from `src/lower.rs`'s own `lower_node` match arms by `examples/generate_op_coverage_doc.rs` -- do not hand-edit. Regenerate with `cargo run -p proxima-onnx --example generate_op_coverage_doc`. `tests/op_coverage_doc_drift.rs` fails the build if this file falls out of sync.

48 ops lower today; any ONNX op not in this list hits `LowerError::UnsupportedOp`.

| onnx op |
| --- |
| Add |
| Sub |
| Mul |
| Div |
| Relu |
| Sigmoid |
| Tanh |
| Exp |
| Log |
| Sqrt |
| Neg |
| Reciprocal |
| Identity |
| Erf |
| Max |
| Min |
| Greater |
| Equal |
| MatMul |
| Gemm |
| Softmax |
| LogSoftmax |
| Transpose |
| Gather |
| Unsqueeze |
| Constant |
| Where |
| If |
| Scan |
| Loop |
| ReduceSum |
| ReduceMax |
| ReduceMin |
| ReduceProd |
| ReduceMean |
| Reshape |
| Flatten |
| Concat |
| Conv |
| ConvTranspose |
| MaxPool |
| AveragePool |
| BatchNormalization |
| Cast |
| Shape |
| Pow |
| Slice |
| Dropout |
