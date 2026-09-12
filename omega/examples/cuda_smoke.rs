//! Exercise CUDA context creation, precompiled PTX launch, and (when present)
//! NVRTC launch, synchronization, and readback. A host without a visible
//! CUDA device reports the typed driver error and exits successfully so this
//! remains useful in CPU CI.

use omega::CudaDriver;
use proxima_tensor::{
    DType, Extent, IndexMap, NodeId, NumericPolicy, Op, QuantizedBlock, ScalarOp, append, map,
};

fn main() {
    match CudaDriver::new(0) {
        Ok(driver) => {
            let (free_bytes, total_bytes) = driver
                .memory_info()
                .expect("CUDA memory query should execute");
            let precompiled = driver
                .smoke_add_one_precompiled(&[1.0, 2.0, 3.0])
                .expect("precompiled CUDA smoke kernel should execute");
            assert_eq!(precompiled, [2.0, 3.0, 4.0]);
            let mut graph_summary = String::from("nvrtc=unavailable");
            let run_nvrtc = std::env::var_os("PROXIMA_CUDA_NVRTC").is_some();
            let mut program = Vec::new();
            let input = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: vec![Extent::Static(4)],
                    name: Some("x".into()),
                },
            );
            let one = append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: vec![Extent::Static(4)],
                    value: 1.0,
                },
            );
            let output_node = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    operands: vec![
                        (input, IndexMap::Affine(map::projection(1, &[0]))),
                        (one, IndexMap::Affine(map::projection(1, &[0]))),
                    ],
                    name: None,
                },
            );
            let mut plan = driver
                .plan(
                    &program,
                    &[],
                    &[output_node],
                    NumericPolicy::default(),
                    &[(&"x", QuantizedBlock::Float32(&[1.0, 2.0, 3.0, 4.0]))],
                )
                .expect("planned CUDA graph should bind");
            if run_nvrtc {
                let output = driver
                    .smoke_add_one(&[1.0, 2.0, 3.0])
                    .expect("CUDA NVRTC smoke kernel should execute");
                assert_eq!(output, [2.0, 3.0, 4.0]);
                let (chained, allocations) = driver
                    .smoke_add_one_persistent(&[1.0, 2.0, 3.0])
                    .expect("persistent CUDA smoke chain should execute");
                assert_eq!(chained, [3.0, 4.0, 5.0]);
                assert_eq!(allocations, 3);
                let evaluated = plan
                    .execute_named(
                        &[("x", QuantizedBlock::Float32(&[1.0, 2.0, 3.0, 4.0]))],
                        None,
                    )
                    .expect("planned CUDA graph should execute");
                let (values, shape) = evaluated
                    .get(NodeId(output_node.0))
                    .expect("planned graph output should be present");
                assert_eq!(values, [2.0, 3.0, 4.0, 5.0]);
                assert_eq!(shape, [4]);
                graph_summary = format!("nvrtc=ok chained={chained:?} allocations={allocations}");
            }
            println!(
                "cuda_smoke: precompiled={precompiled:?} {graph_summary} graph_allocations={} free_bytes={free_bytes} total_bytes={total_bytes} kernel_compilations={}",
                plan.allocations(),
                driver.kernel_compilations()
            );
        }
        Err(error) => {
            println!("cuda_smoke: unavailable: {error}");
        }
    }
}
