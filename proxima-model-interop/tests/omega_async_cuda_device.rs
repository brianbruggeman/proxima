#![cfg(all(feature = "cuda", feature = "interop-bgpool"))]

use core::future::{Future, poll_fn};
use core::pin::pin;
use core::task::Poll;
use std::string::String;

use omega::CudaDriver;
use proxima_model_interop::cuda_async::{CudaGpuService, OwnedInputError};
use proxima_primitives::pipe::CreditAcquireError;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock, ScalarOp};

fn required_driver() -> CudaDriver {
    CudaDriver::new(0).expect("AC13 requires a visible CUDA device at ordinal 0")
}

fn negate_program() -> Vec<Op> {
    vec![
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Symbolic(0)],
            name: Some(String::from("values")),
        },
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: vec![(NodeId(0), IndexMap::Affine(map::projection(1, &[0])))],
            name: Some(String::from("negated")),
        },
    ]
}

#[proxima::test]
async fn cuda_device_name_and_literal_sync_async_output_match() {
    let driver = required_driver();
    let synchronous_device = driver
        .device_snapshot()
        .expect("device name should resolve");
    assert_eq!(synchronous_device.ordinal, 0);
    assert!(!synchronous_device.name.is_empty());

    let service = CudaGpuService::start().expect("CUDA owner should start");
    let owner_device = service
        .device_snapshot()
        .await
        .expect("owner device should resolve");
    assert_eq!(
        (owner_device.ordinal, &owner_device.name),
        (0, &synchronous_device.name)
    );
    let program = negate_program();
    let values = [3.0_f32, -4.0];
    let named = [("values", QuantizedBlock::Float32(&values))];
    let mut plan = driver
        .plan(
            &program,
            &[2],
            &[NodeId(1)],
            NumericPolicy::default(),
            &named,
        )
        .expect("synchronous graph should plan");
    let synchronous = plan
        .execute_named(&named, None)
        .expect("synchronous graph should execute");
    let owned = service
        .copy_named_blocks(&named)
        .await
        .expect("source should copy");
    let lease = service
        .register_graph(
            &program,
            &[2],
            &[NodeId(1)],
            NumericPolicy::default(),
            true,
            None,
            &owned,
        )
        .await
        .expect("owner graph should register");
    let receipt = service
        .execute_cuda_recorded(&lease, owned)
        .await
        .expect("owner graph should execute");
    let asynchronous = &receipt.evaluated;
    assert_eq!(synchronous.root(), &[-3.0, 4.0]);
    assert_eq!(asynchronous.root(), &[-3.0, 4.0]);
    assert_eq!(asynchronous.root(), synchronous.root());
    assert_eq!(asynchronous.shape(), synchronous.shape());
    let activity = service
        .device_snapshot()
        .await
        .expect("owner activity should resolve");
    assert!(activity.graph_uploads > 0);
    assert!(activity.graph_launches > 0);
    eprintln!(
        "{}",
        serde_json::json!({
            "device_ordinal": activity.ordinal, "device_name": activity.name,
            "job": "literal-parity", "generation": lease.generation(),
            "submission_index": 0, "owner_completion_sequence": receipt.completion_sequence, "status": "ok",
            "sync_output": synchronous.root(), "async_output": asynchronous.root(),
            "graph_uploads": activity.graph_uploads, "graph_launches": activity.graph_launches,
        })
    );
    service.shutdown().await.expect("owner should drain");
}

#[proxima::test]
async fn cuda_device_completions_follow_submission_order() {
    let driver = required_driver();
    let service = CudaGpuService::start().expect("CUDA owner should start");
    let device = service
        .device_snapshot()
        .await
        .expect("owner device should resolve");
    assert_eq!(
        device.name,
        driver.device_snapshot().expect("device name").name
    );
    let program = negate_program();
    let first = [1.0_f32, 2.0];
    let second = [3.0_f32, 4.0];
    let third = [5.0_f32, 6.0];
    let first_owned = service
        .copy_named_blocks(&[("values", QuantizedBlock::Float32(&first))])
        .await
        .expect("first copy");
    let second_owned = service
        .copy_named_blocks(&[("values", QuantizedBlock::Float32(&second))])
        .await
        .expect("second copy");
    let third_owned = service
        .copy_named_blocks(&[("values", QuantizedBlock::Float32(&third))])
        .await
        .expect("third copy");
    let lease = service
        .register_graph(
            &program,
            &[2],
            &[NodeId(1)],
            NumericPolicy::default(),
            true,
            None,
            &first_owned,
        )
        .await
        .expect("owner graph should register");
    let outputs = {
        let mut first_call = pin!(service.execute_cuda_recorded(&lease, first_owned));
        let mut second_call = pin!(service.execute_cuda_recorded(&lease, second_owned));
        let mut third_call = pin!(service.execute_cuda_recorded(&lease, third_owned));
        let mut outputs: [Option<(Vec<f32>, u64)>; 3] = std::array::from_fn(|_| None);
        poll_fn(|context| {
            if outputs[0].is_none() {
                if let Poll::Ready(result) = first_call.as_mut().poll(context) {
                    let receipt = result.expect("first execution");
                    outputs[0] = Some((
                        receipt.evaluated.root().to_vec(),
                        receipt.completion_sequence,
                    ));
                }
            }
            if outputs[1].is_none() {
                if let Poll::Ready(result) = second_call.as_mut().poll(context) {
                    let receipt = result.expect("second execution");
                    outputs[1] = Some((
                        receipt.evaluated.root().to_vec(),
                        receipt.completion_sequence,
                    ));
                }
            }
            if outputs[2].is_none() {
                if let Poll::Ready(result) = third_call.as_mut().poll(context) {
                    let receipt = result.expect("third execution");
                    outputs[2] = Some((
                        receipt.evaluated.root().to_vec(),
                        receipt.completion_sequence,
                    ));
                }
            }
            if outputs.iter().all(Option::is_some) {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        })
        .await;
        outputs
    };
    assert_eq!(
        outputs,
        [
            Some((vec![-1.0, -2.0], 0)),
            Some((vec![-3.0, -4.0], 1)),
            Some((vec![-5.0, -6.0], 2))
        ]
    );
    let activity = service
        .device_snapshot()
        .await
        .expect("owner activity should resolve");
    assert!(activity.graph_launches >= 3);
    for (submission_index, output) in outputs.iter().enumerate() {
        let (values, owner_completion_sequence) = output.as_ref().expect("all jobs completed");
        eprintln!(
            "{}",
            serde_json::json!({
                "device_ordinal": activity.ordinal, "device_name": activity.name,
                "job": format!("ordered-{submission_index}"), "generation": lease.generation(),
                "submission_index": submission_index, "owner_completion_sequence": owner_completion_sequence,
                "status": "ok", "output": values,
                "graph_uploads_total": activity.graph_uploads, "graph_launches_total": activity.graph_launches,
            })
        );
    }
    service.shutdown().await.expect("owner should drain");
}

#[proxima::test]
async fn cuda_device_over_budget_refuses_before_upload_or_launch() {
    let driver = required_driver();
    let service = CudaGpuService::start_with_limits(1, 4).expect("bounded owner should start");
    let before = service
        .device_snapshot()
        .await
        .expect("owner device should resolve");
    assert_eq!(
        before.name,
        driver.device_snapshot().expect("device name").name
    );
    assert_eq!((before.graph_uploads, before.graph_launches), (0, 0));
    let values = [1.0_f32, 2.0];
    let rejected = service
        .copy_named_blocks(&[("values", QuantizedBlock::Float32(&values))])
        .await;
    assert!(matches!(
        rejected,
        Err(OwnedInputError::Budget(CreditAcquireError::OverCapacity))
    ));
    let after = service
        .device_snapshot()
        .await
        .expect("owner activity should resolve");
    assert_eq!((after.graph_uploads, after.graph_launches), (0, 0));
    eprintln!(
        "{}",
        serde_json::json!({
            "device_ordinal": after.ordinal, "device_name": after.name,
            "job": "over-budget", "generation": null,
            "submission_index": null, "completion_index": null,
            "status": "refused-over-capacity", "source_bytes": 14, "byte_limit": 4,
            "graph_uploads": after.graph_uploads, "graph_launches": after.graph_launches,
        })
    );
    service.shutdown().await.expect("owner should drain");
}
