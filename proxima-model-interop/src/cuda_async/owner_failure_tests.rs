use core::future::{Future, poll_fn};
use core::pin::pin;
use core::task::Poll;
use std::time::Duration;

use proxima_core::ProximaError;
use proxima_primitives::pipe::CreditAcquireError;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, Op};
use std::sync::Arc;

use super::{
    CudaExecutionError, CudaGpuService, CudaPlanLease, CudaPlanOptions, DEFAULT_OWNED_HOST_BYTES,
    GpuAdmissionError, GpuPipeError, OwnedBlock, OwnedGpuInputs, OwnedInputError,
};

async fn failing_inputs<Owner>(service: &CudaGpuService<Owner>) -> OwnedGpuInputs
where
    Owner: Future<Output = Result<(), ProximaError>> + Send,
{
    let values = [3.0_f32];
    service
        .copy_named_blocks(&[("values", QuantizedBlock::Float32(&values))])
        .await
        .expect("one malformed graph input should fit the byte budget")
}

async fn failing_fake_plan<Owner>(service: &CudaGpuService<Owner>) -> CudaPlanLease
where
    Owner: Future<Output = Result<(), ProximaError>> + Send,
{
    let graph = [Op::Input {
        dtype: DType::FLOAT32,
        shape: vec![Extent::Static(2)],
        name: Some(String::from("values")),
    }];
    let inputs = failing_inputs(service).await;
    service
        .register_graph(
            &graph,
            &[],
            &[NodeId(0)],
            CudaPlanOptions {
                numeric_policy: NumericPolicy::bit_exact(),
                exact_activations: false,
                omega_runtime: None,
            },
            &inputs,
        )
        .await
        .expect("fake owner should register the graph before execution")
}

#[proxima::test]
async fn test_omega_async_owner_failure_preserves_active_error_and_closes_queued_job() {
    let (service, control) =
        CudaGpuService::start_fake_held().expect("held fake owner should start");
    let lease = failing_fake_plan(&service).await;
    let initial_jobs = service.job_gate.available();
    let initial_bytes = service.available_owned_host_bytes();
    let (active_pending, accepted, queued_pending, release, active_result, queued_result) = {
        let first_inputs = failing_inputs(&service).await;
        let second_inputs = failing_inputs(&service).await;
        let mut active = pin!(service.execute_cuda(&lease, first_inputs));
        let active_pending = matches!(
            poll_fn(|context| Poll::Ready(active.as_mut().poll(context))).await,
            Poll::Pending
        );
        let accepted = control.accepted.recv_timeout(Duration::from_secs(2));
        let mut queued = pin!(service.execute_cuda(&lease, second_inputs));
        let queued_pending = matches!(
            poll_fn(|context| Poll::Ready(queued.as_mut().poll(context))).await,
            Poll::Pending
        );
        let release = control.release.send(());
        let active_result = active.await;
        let queued_result = queued.await;
        (
            active_pending,
            accepted,
            queued_pending,
            release,
            active_result,
            queued_result,
        )
    };
    let executed = service.fake_execute_count();
    let remaining_jobs = service.job_gate.available();
    let remaining_bytes = service.available_owned_host_bytes();
    let shutdown = service.shutdown().await;

    accepted.expect("fake owner should hold the active job");
    release.expect("held fake owner should accept release");
    assert!(active_pending);
    assert!(queued_pending);
    assert!(matches!(
        active_result,
        Err(GpuPipeError::Execution(CudaExecutionError::Driver(
            omega::CudaDriverError::UnsupportedGraph {
                node: NodeId(0),
                reason: "fake graph evaluation failed"
            }
        )))
    ));
    assert!(matches!(
        queued_result,
        Err(GpuPipeError::Execution(CudaExecutionError::ServiceClosed))
    ));
    assert_eq!(executed, 1);
    assert_eq!(remaining_jobs, initial_jobs);
    assert_eq!(remaining_bytes, initial_bytes);
    shutdown.expect("owner should join after draining the queued job");
}

#[proxima::test]
async fn test_omega_async_owner_failure_returns_later_input_and_closes_copy_admission() {
    let service = CudaGpuService::start_fake().expect("fake owner should start");
    let lease = failing_fake_plan(&service).await;
    let initial_jobs = service.job_gate.available();
    let late_inputs = failing_inputs(&service).await;
    let active_result = service
        .execute_cuda(&lease, failing_inputs(&service).await)
        .await;
    let late_result = service.execute_cuda(&lease, late_inputs).await;
    let returned = match late_result {
        Err(GpuPipeError::Admission(GpuAdmissionError::Closed(job))) => job,
        Err(GpuPipeError::Execution(CudaExecutionError::ServiceClosedWithInput(job))) => job,
        other => panic!("failed service should return the rejected input: {other:?}"),
    };
    let (returned_lease, returned_inputs) = returned.into_parts();
    let returned_bytes = returned_inputs.owned_bytes;
    let returned_values = match &returned_inputs.blocks[0].block {
        OwnedBlock::Float32(values) => values.to_vec(),
        _ => panic!("rejected input should retain its float payload"),
    };
    drop(returned_inputs);
    let copy_after_failure = service
        .copy_named_blocks(&[("values", QuantizedBlock::Float32(&[5.0]))])
        .await;
    let executed = service.fake_execute_count();
    let job_gate = Arc::clone(&service.job_gate);
    let byte_budget = Arc::clone(&service.byte_budget);
    let shutdown = service.shutdown().await;
    let remaining_jobs = job_gate.available();
    let remaining_bytes = byte_budget.available_bytes();

    assert!(matches!(
        active_result,
        Err(GpuPipeError::Execution(CudaExecutionError::Driver(_)))
    ));
    assert_eq!(returned_lease.generation(), lease.generation());
    assert_eq!(returned_bytes, 10);
    assert_eq!(returned_values, [3.0]);
    assert!(matches!(
        copy_after_failure,
        Err(OwnedInputError::Budget(CreditAcquireError::Closed))
    ));
    assert_eq!(executed, 1);
    assert_eq!(remaining_jobs, initial_jobs);
    assert_eq!(remaining_bytes, DEFAULT_OWNED_HOST_BYTES);
    shutdown.expect("failed owner should still join");
}

#[proxima::test]
async fn test_omega_async_owner_failure_drains_every_accepted_job_and_restores_credits() {
    let (service, control) =
        CudaGpuService::start_fake_held().expect("held fake owner should start");
    let lease = failing_fake_plan(&service).await;
    let initial_jobs = service.job_gate.available();
    let initial_bytes = service.available_owned_host_bytes();
    let (
        active_pending,
        accepted,
        first_pending,
        second_pending,
        reserved_jobs,
        reserved_bytes,
        release,
        active_result,
        first_result,
        second_result,
    ) = {
        let mut active = pin!(service.execute_cuda(&lease, failing_inputs(&service).await));
        let active_pending = matches!(
            poll_fn(|context| Poll::Ready(active.as_mut().poll(context))).await,
            Poll::Pending
        );
        let accepted = control.accepted.recv_timeout(Duration::from_secs(2));
        let mut queued_first = pin!(service.execute_cuda(&lease, failing_inputs(&service).await));
        let first_pending = matches!(
            poll_fn(|context| Poll::Ready(queued_first.as_mut().poll(context))).await,
            Poll::Pending
        );
        let mut queued_second = pin!(service.execute_cuda(&lease, failing_inputs(&service).await));
        let second_pending = matches!(
            poll_fn(|context| Poll::Ready(queued_second.as_mut().poll(context))).await,
            Poll::Pending
        );
        let reserved_jobs = service.job_gate.available();
        let reserved_bytes = service.available_owned_host_bytes();
        let release = control.release.send(());
        let active_result = active.await;
        let first_result = queued_first.await;
        let second_result = queued_second.await;
        (
            active_pending,
            accepted,
            first_pending,
            second_pending,
            reserved_jobs,
            reserved_bytes,
            release,
            active_result,
            first_result,
            second_result,
        )
    };
    let executed = service.fake_execute_count();
    let remaining_jobs = service.job_gate.available();
    let remaining_bytes = service.available_owned_host_bytes();
    let shutdown = service.shutdown().await;

    accepted.expect("fake owner should hold the active job");
    release.expect("held fake owner should accept release");
    assert!(active_pending && first_pending && second_pending);
    assert_eq!(reserved_jobs, initial_jobs - 3);
    assert_eq!(reserved_bytes, initial_bytes - 30);
    assert!(matches!(
        active_result,
        Err(GpuPipeError::Execution(CudaExecutionError::Driver(_)))
    ));
    assert!(matches!(
        first_result,
        Err(GpuPipeError::Execution(CudaExecutionError::ServiceClosed))
    ));
    assert!(matches!(
        second_result,
        Err(GpuPipeError::Execution(CudaExecutionError::ServiceClosed))
    ));
    assert_eq!(executed, 1);
    assert_eq!(remaining_jobs, initial_jobs);
    assert_eq!(remaining_bytes, initial_bytes);
    shutdown.expect("owner should join after draining all accepted jobs");
}
