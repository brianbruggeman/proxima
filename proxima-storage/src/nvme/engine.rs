use core::future::{Future, poll_fn};
use core::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use core::task::Poll;

use proxima_primitives::pipe::{Pipe, SendPipe};
use proxima_protocols::nvme::{
    CommandBuilder, CompletionEntry, CompletionRing, StatusField, SubmissionRing, command,
    completion,
};

use crate::nvme::backend::QueueBackend;
use crate::nvme::error::NvmeError;

const SQE_LEN: usize = command::ENTRY_LEN;
const CQE_LEN: usize = completion::ENTRY_LEN;

/// An owned, decoded completion — the [`Pipe`] output. `Copy`, no borrow into
/// backend memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completion {
    /// Command identifier echoed from the originating SQE.
    pub command_id: u16,
    /// The decoded status field (success, phase, retry, etc.).
    pub status: StatusField,
    /// Command-specific result dword (dword 0).
    pub command_specific: u32,
    /// Controller's submission-queue head — how far it has consumed the SQ.
    pub sq_head: u16,
}

/// A prime-native NVMe queue pair. Drives the sans-IO codec + ring FSM over a
/// [`QueueBackend`] and exposes "submit a command, await its completion" as a
/// proxima [`Pipe`] (per-core, `!Send` root) and, when the backend crosses
/// cores, a [`SendPipe`].
///
/// The ring cursors live in atomics so the pair is `Sync` (required by
/// `SendPipe`) without an async lock; the sealed [`SubmissionRing`] /
/// [`CompletionRing`] logic is reconstituted per operation via their `resume`
/// constructors. The submission ring is single-producer by NVMe design — the
/// `SendPipe` form lets the handle migrate across cores, not be hammered
/// concurrently from many cores against one pair (NVMe uses one pair per core).
#[derive(Debug)]
pub struct QueuePair<Backend> {
    backend: Backend,
    depth: u32,
    submission_tail: AtomicU16,
    completion_head: AtomicU16,
    completion_phase: AtomicBool,
}

impl<Backend: QueueBackend> QueuePair<Backend> {
    /// Build a queue pair over `backend` with `depth` entries per ring. `depth`
    /// is validated against the NVMe-legal range by the codec.
    pub fn new(backend: Backend, depth: u32) -> Result<Self, NvmeError> {
        // validate depth once through the sealed ring constructor.
        SubmissionRing::new(depth)?;
        Ok(Self {
            backend,
            depth,
            submission_tail: AtomicU16::new(0),
            completion_head: AtomicU16::new(0),
            completion_phase: AtomicBool::new(true),
        })
    }

    /// The backend, for inspection or teardown.
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    fn submit(&self, command: &CommandBuilder) -> Result<(), NvmeError> {
        let mut ring =
            SubmissionRing::resume(self.depth, self.submission_tail.load(Ordering::Acquire))?;
        let mut sqe = [0u8; SQE_LEN];
        command.write(&mut sqe)?;
        self.backend.write_submission(ring.slot(), &sqe);
        let tail = ring.advance();
        self.submission_tail.store(tail, Ordering::Release);
        self.backend.ring_submit_doorbell(tail);
        Ok(())
    }

    fn try_reap(&self, command_id: u16) -> Result<Option<Completion>, NvmeError> {
        let mut ring = CompletionRing::resume(
            self.depth,
            self.completion_head.load(Ordering::Acquire),
            self.completion_phase.load(Ordering::Acquire),
        )?;
        let cqe = self.backend.read_completion(ring.slot());
        if !ring.is_ready(phase_tag(&cqe)) {
            return Ok(None);
        }
        let entry = CompletionEntry::parse(&cqe)?;
        let (reaped_id, status) = entry.command_id_and_status();
        let completion = Completion {
            command_id: reaped_id,
            status,
            command_specific: entry.command_specific(),
            sq_head: entry.sq_head(),
        };
        let head = ring.advance();
        self.completion_head.store(head, Ordering::Release);
        self.completion_phase
            .store(ring.expected_phase(), Ordering::Release);
        self.backend.ring_complete_doorbell(head);
        // a completion for a different in-flight command is consumed but not
        // returned here; the caller keeps polling for its own id.
        Ok((reaped_id == command_id).then_some(completion))
    }

    /// Submit `command`, then cooperatively poll the completion ring until the
    /// matching completion appears. A poll-mode NVMe queue has no fd to register
    /// with the reactor, so — exactly like a dpdk poll-mode driver — each poll
    /// that finds nothing re-arms its own waker and yields, letting the executor
    /// run other tasks before re-driving this one. No spin loop monopolises the
    /// core; cooperation, not a private busy-wait, is the difference from the
    /// earlier draft.
    async fn complete(&self, command: CommandBuilder) -> Result<Completion, NvmeError> {
        let command_id = command.command_id();
        self.submit(&command)?;
        poll_fn(move |context| match self.try_reap(command_id) {
            Ok(Some(completion)) => Poll::Ready(Ok(completion)),
            Ok(None) => {
                // no fresh completion this turn: re-arm and re-poll next turn.
                context.waker().wake_by_ref();
                Poll::Pending
            }
            Err(error) => Poll::Ready(Err(error)),
        })
        .await
    }
}

#[inline]
fn phase_tag(cqe: &[u8; CQE_LEN]) -> bool {
    cqe[14] & 0b1 != 0
}

impl<Backend: QueueBackend> Pipe for QueuePair<Backend> {
    type In = CommandBuilder;
    type Out = Completion;
    type Err = NvmeError;

    fn call(&self, input: CommandBuilder) -> impl Future<Output = Result<Completion, NvmeError>> {
        self.complete(input)
    }
}

impl<Backend: QueueBackend + Send + Sync + 'static> SendPipe for QueuePair<Backend> {
    type In = CommandBuilder;
    type Out = Completion;
    type Err = NvmeError;

    // the same `complete` future; it is `Send` whenever the backend is, so one
    // body serves both forms (mirrors primitives's `AndThen`).
    fn call(
        &self,
        input: CommandBuilder,
    ) -> impl Future<Output = Result<Completion, NvmeError>> + Send {
        self.complete(input)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proxima_protocols::nvme::write_completion;
    use std::sync::Mutex;
    use std::vec;
    use std::vec::Vec;

    const OPC_READ: u8 = 0x02;
    const SUCCESS_PHASE_1: u16 = 0x0001;

    /// In-memory NVMe controller — the storage analog of net-dpdk's `net_tap`
    /// loopback. On the submit doorbell it consumes the new SQEs and posts a
    /// success completion for each, flipping the phase tag on wrap exactly as a
    /// real controller does. `Mutex` makes it `Send + Sync` so it can drive the
    /// `SendPipe` form too.
    struct Loopback {
        state: Mutex<LoopbackState>,
    }

    struct LoopbackState {
        depth: u32,
        submission: Vec<[u8; SQE_LEN]>,
        completion: Vec<[u8; CQE_LEN]>,
        controller_sq_head: u16,
        submitted_tail: u16,
        controller_cq_tail: u16,
        controller_phase: bool,
        /// 0 = post completions the instant a command is submitted; N = post
        /// only after the host has polled the completion ring N times, so the
        /// future is forced to pend (and yield) N-1 times first.
        complete_after: u32,
        reads: u32,
    }

    impl LoopbackState {
        /// Process every submitted-but-unhandled SQE: post a success completion
        /// for each, flipping the phase tag on wrap exactly as a real controller.
        fn process(&mut self) {
            while self.controller_sq_head != self.submitted_tail {
                let head = self.controller_sq_head as usize;
                let command_id =
                    u16::from_le_bytes([self.submission[head][2], self.submission[head][3]]);
                let sq_head_after = ((u32::from(self.controller_sq_head) + 1) % self.depth) as u16;

                let status = StatusField::from_bits(if self.controller_phase {
                    SUCCESS_PHASE_1
                } else {
                    0x0000
                });
                let cq_slot = self.controller_cq_tail as usize;
                write_completion(
                    &mut self.completion[cq_slot],
                    0,
                    sq_head_after,
                    0,
                    command_id,
                    status,
                )
                .expect("16-byte cqe slot");

                let next_cq = ((u32::from(self.controller_cq_tail) + 1) % self.depth) as u16;
                if next_cq == 0 {
                    self.controller_phase = !self.controller_phase;
                }
                self.controller_cq_tail = next_cq;
                self.controller_sq_head = sq_head_after;
            }
        }
    }

    impl Loopback {
        fn build(depth: u32, complete_after: u32) -> Self {
            Self {
                state: Mutex::new(LoopbackState {
                    depth,
                    submission: vec![[0u8; SQE_LEN]; depth as usize],
                    completion: vec![[0u8; CQE_LEN]; depth as usize],
                    controller_sq_head: 0,
                    submitted_tail: 0,
                    controller_cq_tail: 0,
                    controller_phase: true,
                    complete_after,
                    reads: 0,
                }),
            }
        }

        /// A controller that posts completions the instant a command is submitted.
        fn new(depth: u32) -> Self {
            Self::build(depth, 0)
        }

        /// A controller that posts only after the host has polled the completion
        /// ring `after` times — forcing the completion future to pend (and yield
        /// the core) in between.
        fn delayed(depth: u32, after: u32) -> Self {
            Self::build(depth, after)
        }
    }

    impl QueueBackend for Loopback {
        fn write_submission(&self, slot: u16, entry: &[u8; SQE_LEN]) {
            self.state.lock().expect("loopback poisoned").submission[slot as usize] = *entry;
        }

        fn ring_submit_doorbell(&self, tail: u16) {
            let mut state = self.state.lock().expect("loopback poisoned");
            state.submitted_tail = tail;
            if state.complete_after == 0 {
                state.process();
            }
        }

        fn read_completion(&self, slot: u16) -> [u8; CQE_LEN] {
            let mut state = self.state.lock().expect("loopback poisoned");
            if state.complete_after != 0 {
                state.reads += 1;
                if state.reads >= state.complete_after {
                    state.process();
                }
            }
            state.completion[slot as usize]
        }

        fn ring_complete_doorbell(&self, _head: u16) {}
    }

    /// Minimal driver for the cooperative future when an auto-completing backend
    /// makes it ready on the first poll. The real-executor tests below use a
    /// prime `LocalExecutor` instead, where the self-rewake actually yields.
    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    fn read_command(command_id: u16) -> CommandBuilder {
        CommandBuilder::new(OPC_READ, command_id)
            .namespace_id(1)
            .command_dword(0, 0x1000)
    }

    #[test]
    fn pipe_call_round_trips_a_completion() {
        let queue = QueuePair::new(Loopback::new(8), 8).expect("legal depth");
        let completion = block_on(Pipe::call(&queue, read_command(0x0007))).expect("completes");
        assert_eq!(completion.command_id, 0x0007);
        assert!(completion.status.is_success());
        assert_eq!(completion.sq_head, 1);
    }

    fn assert_send<Type: Send>(_: &Type) {}

    #[test]
    fn send_pipe_call_round_trips_with_a_send_future() {
        let queue = QueuePair::new(Loopback::new(8), 8).expect("legal depth");
        let future = SendPipe::call(&queue, read_command(0x00ab));
        assert_send(&future);
        let completion = block_on(future).expect("completes");
        assert_eq!(completion.command_id, 0x00ab);
        assert!(completion.status.is_success());
    }

    #[test]
    fn many_commands_wrap_the_rings_with_correct_phase() {
        // depth 4 -> submit 11 commands forces two full completion-ring wraps,
        // proving the engine's phase expectation stays in lockstep with the
        // controller's phase flips across laps.
        let queue = QueuePair::new(Loopback::new(4), 4).expect("legal depth");
        for sequence in 0..11u16 {
            let command_id = 0x0100 + sequence;
            let completion =
                block_on(Pipe::call(&queue, read_command(command_id))).expect("completes");
            assert_eq!(
                completion.command_id, command_id,
                "completion matches the submitted id"
            );
            assert!(
                completion.status.is_success(),
                "every loopback completion succeeds"
            );
        }
    }

    #[test]
    fn the_queue_pair_is_send_and_sync() {
        fn assert_send_sync<Type: Send + Sync>() {}
        assert_send_sync::<QueuePair<Loopback>>();
    }

    // --- C3: the cooperative future runs on a real per-core prime worker ---

    use prime::core::local_executor::LocalExecutor;
    use std::cell::Cell;
    use std::rc::Rc;

    /// Yield to the executor exactly once, then resume — the minimal cooperative
    /// sibling used to observe interleaving.
    async fn yield_once() {
        let mut yielded = false;
        poll_fn(move |context| {
            if yielded {
                Poll::Ready(())
            } else {
                yielded = true;
                context.waker().wake_by_ref();
                Poll::Pending
            }
        })
        .await;
    }

    #[test]
    fn completion_drives_to_done_on_a_prime_local_executor() {
        let queue = Rc::new(QueuePair::new(Loopback::new(8), 8).expect("legal depth"));
        let executor = LocalExecutor::new();
        let task_queue = queue.clone();
        let completion =
            executor.block_on(async move { Pipe::call(&*task_queue, read_command(0x0007)).await });
        let completion = completion.expect("completes on the prime worker");
        assert_eq!(completion.command_id, 0x0007);
        assert!(completion.status.is_success());
    }

    #[test]
    fn pending_completion_yields_so_a_sibling_task_makes_progress() {
        // a controller that posts only after the 6th completion poll forces the
        // nvme future to pend (and self-rewake) five times. That is the whole
        // point of C3: the core is NOT monopolised, so a co-scheduled sibling
        // interleaves on every pend instead of being starved by a spin loop.
        let queue = Rc::new(QueuePair::new(Loopback::delayed(8, 6), 8).expect("legal depth"));
        let executor = LocalExecutor::new();

        let sibling_ticks = Rc::new(Cell::new(0u32));
        let sibling_counter = sibling_ticks.clone();
        executor.spawn_local(async move {
            for _ in 0..4 {
                sibling_counter.set(sibling_counter.get() + 1);
                yield_once().await;
            }
        });

        // run the nvme completion to done; the executor interleaves the sibling
        // while it pends.
        let queue_task = queue.clone();
        let completion = executor
            .block_on(async move { Pipe::call(&*queue_task, read_command(0x0007)).await })
            .expect("completes once the controller posts");

        assert_eq!(completion.command_id, 0x0007);
        assert!(completion.status.is_success());
        assert!(
            sibling_ticks.get() >= 2,
            "the sibling made progress while the nvme completion was pending (got {})",
            sibling_ticks.get()
        );
    }
}
