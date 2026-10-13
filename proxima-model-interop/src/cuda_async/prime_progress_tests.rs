use core::future::{Future, poll_fn};
use core::pin::pin;
use core::task::Poll;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use prime::os::core_shard::current_core;
use prime::os::runtime::PrimeRuntime;
use proxima::CoreId;
use proxima_primitives::pipe::{CreditGate, Pipe, PipeExt};
use proxima_primitives::sync::mpsc::{Receiver as CommandReceiver, channel as command_channel};
use proxima_tensor::{Evaluated, NodeId};

use super::{
    AdmittedGpuJob, CompletedGpuJob, CudaExecutionPipe, CudaPlanLease, GpuCapacityGate,
    OwnedByteBudget, OwnedGpuJob, OwnerCommand, ReservationSink,
};

async fn owned_gpu_job() -> OwnedGpuJob {
    let budget = OwnedByteBudget::new(1).expect("one byte of accounting capacity");
    let inputs = budget
        .copy_named_blocks(&[])
        .await
        .expect("empty owned inputs");
    OwnedGpuJob {
        lease: CudaPlanLease {
            service: Arc::new(()),
            slotindex: 0,
            generation: 0,
        },
        inputs,
    }
}

type PrimePipeCase = (
    PrimeRuntime,
    CommandReceiver<OwnerCommand>,
    Receiver<Option<CoreId>>,
    Receiver<(Option<CoreId>, Vec<f32>)>,
);

fn prime_pipe_case() -> PrimePipeCase {
    let runtime = PrimeRuntime::new(1).expect("one prime core should start");
    let (task_commands, commands) = command_channel(1);
    let task_commands = Arc::new(task_commands);
    let (submitted_sender, submitted_receiver) = mpsc::channel();
    let (completed_sender, completed_receiver) = mpsc::channel();

    runtime
        .spawn_typed_on_core(CoreId(0), async move {
            let gate = CreditGate::new(ReservationSink, 1).expect("one job credit");
            let pipeline = GpuCapacityGate { gate: &gate }.and_then(CudaExecutionPipe {
                commands: task_commands,
            });
            let mut call = pin!(pipeline.call(owned_gpu_job().await));
            let first_poll = poll_fn(|context| Poll::Ready(call.as_mut().poll(context))).await;
            assert!(matches!(first_poll, Poll::Pending));
            submitted_sender
                .send(current_core())
                .expect("driver should retain submission receiver");
            let evaluated = call.await.expect("controlled owner should return a result");
            completed_sender
                .send((current_core(), evaluated.root().to_vec()))
                .expect("driver should retain completion receiver");
        })
        .expect("task should enter core zero");

    (runtime, commands, submitted_receiver, completed_receiver)
}

fn publish_owner_result(commands: &CommandReceiver<OwnerCommand>, value: f32) {
    let command = commands
        .recv_blocking()
        .expect("a polled pipe should enqueue execution");
    let OwnerCommand::ExecutePlan { job, reply } = command else {
        panic!("the controlled queue should contain one execution");
    };
    let AdmittedGpuJob { job, credit } = job;
    let evaluated = Evaluated::from_parts(NodeId(0), vec![(NodeId(0), vec![1], vec![value])], None);
    assert!(
        reply
            .send(CompletedGpuJob {
                result: Ok(evaluated),
                completion_sequence: 0,
                _credit: credit
            })
            .is_ok()
    );
    drop(job);
}

#[test]
fn test_omega_async_prime_progress_unrelated_task_runs_before_gpu_completion() {
    let (runtime, commands, submitted, completed) = prime_pipe_case();
    assert_eq!(
        submitted
            .recv_timeout(Duration::from_secs(2))
            .expect("pipe should submit"),
        Some(CoreId(0))
    );
    let (progress_sender, progress_receiver) = mpsc::channel();
    runtime
        .spawn_typed_on_core(CoreId(0), async move {
            progress_sender
                .send(current_core())
                .expect("progress receiver should live");
        })
        .expect("unrelated task should enter core zero");

    assert_eq!(
        progress_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("unrelated task should advance"),
        Some(CoreId(0))
    );
    assert!(matches!(
        completed.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    publish_owner_result(&commands, 41.0);
    assert_eq!(
        completed
            .recv_timeout(Duration::from_secs(2))
            .expect("pipe should complete"),
        (Some(CoreId(0)), vec![41.0])
    );
}

#[test]
fn test_omega_async_prime_progress_owner_completion_wakes_original_task() {
    let (_runtime, commands, submitted, completed) = prime_pipe_case();
    assert_eq!(
        submitted
            .recv_timeout(Duration::from_secs(2))
            .expect("pipe should submit"),
        Some(CoreId(0))
    );
    assert!(matches!(
        completed.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    publish_owner_result(&commands, -7.5);
    assert_eq!(
        completed
            .recv_timeout(Duration::from_secs(2))
            .expect("owner reply should wake caller"),
        (Some(CoreId(0)), vec![-7.5])
    );
}

#[test]
fn test_omega_async_prime_progress_blocking_control_detects_one_violation() {
    let blocking_runtime = PrimeRuntime::new(1).expect("one prime core should start");
    let (entered_sender, entered_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = mpsc::channel::<()>();
    let (blocking_progress_sender, blocking_progress_receiver) = mpsc::channel();
    let blocking_progress = Arc::new(AtomicUsize::new(0));
    let task_progress = Arc::clone(&blocking_progress);
    blocking_runtime
        .spawn_typed_on_core(CoreId(0), async move {
            entered_sender
                .send(current_core())
                .expect("control receiver should live");
            release_receiver.recv().expect("control should be released");
        })
        .expect("blocking control should enter core zero");
    assert_eq!(
        entered_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("control should enter"),
        Some(CoreId(0))
    );
    blocking_runtime
        .spawn_typed_on_core(CoreId(0), async move {
            task_progress.fetch_add(1, Ordering::AcqRel);
            blocking_progress_sender
                .send(current_core())
                .expect("progress receiver should live");
        })
        .expect("progress task should be queued");
    let blocking_violations = usize::from(blocking_progress.load(Ordering::Acquire) == 0);
    release_sender
        .send(())
        .expect("control should retain release receiver");
    assert_eq!(
        blocking_progress_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("released core should progress"),
        Some(CoreId(0))
    );

    let (runtime, commands, submitted, completed) = prime_pipe_case();
    assert_eq!(
        submitted
            .recv_timeout(Duration::from_secs(2))
            .expect("pipe should submit"),
        Some(CoreId(0))
    );
    let (progress_sender, progress_receiver) = mpsc::channel();
    runtime
        .spawn_typed_on_core(CoreId(0), async move {
            progress_sender
                .send(current_core())
                .expect("progress receiver should live");
        })
        .expect("progress task should be queued");
    let production_violations = usize::from(
        progress_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("pipe should yield before owner reply")
            != Some(CoreId(0)),
    );
    assert!(matches!(
        completed.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    publish_owner_result(&commands, 13.0);
    assert_eq!(
        completed
            .recv_timeout(Duration::from_secs(2))
            .expect("pipe should wake"),
        (Some(CoreId(0)), vec![13.0])
    );
    assert_eq!((blocking_violations, production_violations), (1, 0));
}
