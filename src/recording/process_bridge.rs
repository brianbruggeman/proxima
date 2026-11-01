use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use proxima_primitives::sync::task::JoinSet;
use proxima_telemetry::error;
use time::OffsetDateTime;
use tokio::sync::oneshot;

use crate::error::ProximaError;
use crate::log_buffer::LiveTailReceiver;
use crate::recording::event::{InteractionId, ProcessEvent, ProtocolEvent, RecordingEvent};
use crate::recording::sink::DynRecordingSink;
use crate::upstreams::ProcessUpstream;

/// Bridge that maps a `ProcessUpstream`'s stdout/stderr `LogBuffer`s and
/// terminal exit signal into universal `RecordingEvent` envelopes
/// carrying `ProtocolEvent::Process` payloads. The bridge stamps each
/// event with `stage_id` and a `parent` of the owning pipeline (or
/// other coordinator), so downstream tools can walk the parent edge to
/// reconstruct the DAG.
///
/// **Subscription race**: `LogBuffer::subscribe` only delivers lines
/// pushed AFTER subscription, and the child can write the moment it is
/// spawned. `attach` therefore takes receivers the caller subscribed
/// BEFORE `ProcessUpstream::spawn_with_buffers`, so no line can be pushed
/// before a subscriber exists. The standard usage pattern below shows
/// the order.
///
/// Standard usage:
/// ```no_run
/// # use std::sync::Arc;
/// # use proxima::error::ProximaError;
/// # use proxima::recording::{DynRecordingSink, InteractionId, ProcessEventBridge};
/// # use proxima::upstreams::{ProcessSpec, ProcessUpstream};
/// # use proxima::{DEFAULT_LOG_BUFFER_CAPACITY, LogBuffer};
/// # async fn run_stage(
/// #     spec: ProcessSpec,
/// #     stage_id: InteractionId,
/// #     parent: Option<InteractionId>,
/// #     sink: DynRecordingSink,
/// # ) -> Result<Option<i32>, ProximaError> {
/// let stdout_buf = Arc::new(LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY));
/// let stderr_buf = Arc::new(LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY));
/// let stdout_rx = stdout_buf.subscribe();
/// let stderr_rx = stderr_buf.subscribe();
/// let upstream = Arc::new(
///     ProcessUpstream::spawn_with_buffers(
///         "stage-1", spec, Some(stdout_buf), Some(stderr_buf), None,
///     ).await?,
/// );
/// let handle = ProcessEventBridge::attach(
///     upstream.clone(), stdout_rx, stderr_rx, stage_id, parent, sink,
/// ).await?;
/// Ok(handle.wait_for_exit().await)
/// # }
/// ```
pub struct ProcessEventBridge;

/// Handle to a running bridge. Dropping aborts the drain + exit-watch
/// tasks; callers should typically `wait_for_exit()` first.
pub struct BridgeHandle {
    exit_rx: oneshot::Receiver<Option<i32>>,
    _tasks: JoinSet<()>,
}

impl BridgeHandle {
    /// Returns the recorded exit code once the bridge has emitted the
    /// terminal `Exited` event. `None` means the process was signaled
    /// (no exit code) OR the supervisor terminated without observing an
    /// exit (rare; relaunch failed without a child).
    pub async fn wait_for_exit(self) -> Option<i32> {
        self.exit_rx.await.ok().flatten()
    }
}

impl ProcessEventBridge {
    pub async fn attach(
        upstream: Arc<ProcessUpstream>,
        stdout_rx: LiveTailReceiver,
        stderr_rx: LiveTailReceiver,
        stage_id: InteractionId,
        parent: Option<InteractionId>,
        sink: DynRecordingSink,
    ) -> Result<BridgeHandle, ProximaError> {
        let started = Instant::now();
        let started_ts = OffsetDateTime::now_utc();

        let spec = upstream.spec().clone();
        sink.append(RecordingEvent {
            id: stage_id,
            ts_ms: 0,
            parent,
            event: ProtocolEvent::Process(ProcessEvent::Started {
                ts: started_ts,
                command: spec.command,
                args: spec.args,
                env: spec.env,
                cwd: spec.working_dir,
            }),
        })
        .await?;

        let mut tasks = JoinSet::new();
        {
            let sink = sink.clone();
            tasks.spawn(async move {
                while let Some(line) = stdout_rx.recv().await {
                    let event = RecordingEvent {
                        id: stage_id,
                        ts_ms: elapsed_ms(started),
                        parent,
                        event: ProtocolEvent::Process(ProcessEvent::Stdout(Bytes::from(line))),
                    };
                    if let Err(error) = sink.append(event).await {
                        error!(?error, "process bridge: stdout sink append failed");
                    }
                }
            });
        }
        {
            let sink = sink.clone();
            tasks.spawn(async move {
                while let Some(line) = stderr_rx.recv().await {
                    let event = RecordingEvent {
                        id: stage_id,
                        ts_ms: elapsed_ms(started),
                        parent,
                        event: ProtocolEvent::Process(ProcessEvent::Stderr(Bytes::from(line))),
                    };
                    if let Err(error) = sink.append(event).await {
                        error!(?error, "process bridge: stderr sink append failed");
                    }
                }
            });
        }
        let (exit_tx, exit_rx) = oneshot::channel();
        {
            let sink = sink.clone();
            let upstream = upstream.clone();
            tasks.spawn(async move {
                let exit_status = upstream.wait_for_exit().await;
                let exit_code = exit_status.and_then(|status| status.code());
                let event = RecordingEvent {
                    id: stage_id,
                    ts_ms: elapsed_ms(started),
                    parent,
                    event: ProtocolEvent::Process(ProcessEvent::Exited { exit_code }),
                };
                if let Err(error) = sink.append(event).await {
                    error!(?error, "process bridge: exited sink append failed");
                }
                let _ = exit_tx.send(exit_code);
            });
        }

        Ok(BridgeHandle {
            exit_rx,
            _tasks: tasks,
        })
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::log_buffer::{DEFAULT_LOG_BUFFER_CAPACITY, LogBuffer};
    use crate::recording::event::{ProcessEvent, ProtocolEvent, RecordingEvent};
    use crate::recording::sink::{AppendFuture, DynRecordingSink, RecordingSink};
    use crate::upstreams::{
        ProcessSpec, ProcessUpstream, ReadyProbe, RestartPolicy, ShutdownSignal,
    };
    use std::collections::BTreeMap;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;
    use tokio::sync::Notify;

    #[derive(Default)]
    struct MemorySink {
        events: StdMutex<Vec<RecordingEvent>>,
        appended: Notify,
    }

    impl RecordingSink for MemorySink {
        fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
            Box::pin(async move {
                self.events.lock().expect("memory sink lock").push(event);
                self.appended.notify_one();
                Ok(())
            })
        }

        fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
            Box::pin(async { Ok(()) })
        }
    }

    fn shell_spec(script: &str) -> ProcessSpec {
        let (cmd, flag) = if cfg!(windows) {
            ("cmd", "/c")
        } else {
            ("/bin/sh", "-c")
        };
        ProcessSpec {
            command: cmd.into(),
            args: vec![flag.into(), script.into()],
            working_dir: None,
            env: BTreeMap::new(),
            restart: RestartPolicy::Never,
            restart_delay_ms: 0,
            max_restart_attempts: 0,
            ready_probe: ReadyProbe::None,
            shutdown_signal: ShutdownSignal::Term,
            shutdown_timeout_ms: 5_000,
        }
    }

    async fn run_stage(script: &str) -> (Vec<RecordingEvent>, Option<i32>) {
        run_stage_attaching(script, false).await
    }

    async fn run_stage_attaching(
        script: &str,
        attach_after_exit: bool,
    ) -> (Vec<RecordingEvent>, Option<i32>) {
        let sink: Arc<MemorySink> = Arc::new(MemorySink::default());
        let dyn_sink: DynRecordingSink = sink.clone();
        let stdout_buf = Arc::new(LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY));
        let stderr_buf = Arc::new(LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY));
        let stdout_rx = stdout_buf.subscribe();
        let stderr_rx = stderr_buf.subscribe();
        let upstream = Arc::new(
            ProcessUpstream::spawn_with_buffers(
                "stage",
                shell_spec(script),
                Some(stdout_buf.clone()),
                Some(stderr_buf.clone()),
                None,
            )
            .await
            .expect("spawn"),
        );
        if attach_after_exit {
            upstream.wait_for_exit().await;
        }
        let stage_id = InteractionId::new();
        let pipeline_id = InteractionId::new();
        let handle = ProcessEventBridge::attach(
            upstream.clone(),
            stdout_rx,
            stderr_rx,
            stage_id,
            Some(pipeline_id),
            dyn_sink,
        )
        .await
        .expect("attach");
        let exit_code = handle.wait_for_exit().await;
        // the supervisor joins both drains before firing exit, so the buffers
        // hold every line the child wrote; started + exited + one event per
        // line is the exact count the forwarding tasks still owe the sink.
        let expected_events = 2 + stdout_buf.len() + stderr_buf.len();
        let delivered = tokio::time::timeout(Duration::from_secs(5), async {
            while sink.events.lock().expect("lock").len() < expected_events {
                sink.appended.notified().await;
            }
        })
        .await;
        let events = sink.events.lock().expect("lock").clone();
        assert!(
            delivered.is_ok(),
            "forwarding tasks must deliver every buffered line before drop: recorded {} of {} events; \
             stdout missing {:?}; stderr missing {:?}",
            events.len(),
            expected_events,
            missing_lines(&stdout_buf.snapshot(None), &stream_lines(&events, Stream::Stdout)),
            missing_lines(&stderr_buf.snapshot(None), &stream_lines(&events, Stream::Stderr)),
        );
        drop(upstream);
        (events, exit_code)
    }

    #[derive(Clone, Copy)]
    enum Stream {
        Stdout,
        Stderr,
    }

    fn stream_lines(events: &[RecordingEvent], stream: Stream) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match (&event.event, stream) {
                (ProtocolEvent::Process(ProcessEvent::Stdout(data)), Stream::Stdout)
                | (ProtocolEvent::Process(ProcessEvent::Stderr(data)), Stream::Stderr) => {
                    Some(String::from_utf8_lossy(data).into_owned())
                }
                _ => None,
            })
            .collect()
    }

    fn missing_lines(buffered: &[String], recorded: &[String]) -> Vec<String> {
        buffered
            .iter()
            .filter(|line| !recorded.contains(line))
            .cloned()
            .collect()
    }

    #[proxima::test]
    async fn bridge_records_every_line_of_a_burst_on_both_streams_exactly_once_in_order() {
        let lines = 200;
        let script = "for i in $(seq 1 200); do echo out$i; echo err$i 1>&2; done";
        let (events, exit_code) = run_stage_attaching(script, true).await;
        assert_eq!(exit_code, Some(0));
        let expected_stdout: Vec<String> = (1..=lines).map(|index| format!("out{index}")).collect();
        let expected_stderr: Vec<String> = (1..=lines).map(|index| format!("err{index}")).collect();
        assert_eq!(stream_lines(&events, Stream::Stdout), expected_stdout);
        assert_eq!(stream_lines(&events, Stream::Stderr), expected_stderr);
    }

    #[proxima::test]
    async fn bridge_emits_started_stdout_exited_for_simple_command() {
        let (events, exit_code) = run_stage("echo hello; exit 0").await;
        assert_eq!(exit_code, Some(0));
        assert!(matches!(
            events[0].event,
            ProtocolEvent::Process(ProcessEvent::Started { .. })
        ));
        let stdout_lines: Vec<String> = events
            .iter()
            .filter_map(|event| match &event.event {
                ProtocolEvent::Process(ProcessEvent::Stdout(data)) => {
                    Some(String::from_utf8_lossy(data).into_owned())
                }
                _ => None,
            })
            .collect();
        assert_eq!(stdout_lines, vec!["hello".to_string()]);
        let terminal = events.last().expect("at least one event");
        assert!(matches!(
            terminal.event,
            ProtocolEvent::Process(ProcessEvent::Exited { exit_code: Some(0) })
        ));
    }

    #[proxima::test]
    async fn bridge_captures_stderr_separately_from_stdout() {
        let (events, _exit_code) = run_stage("echo out; echo err 1>&2; exit 0").await;
        let stdout_lines: Vec<String> = events
            .iter()
            .filter_map(|event| match &event.event {
                ProtocolEvent::Process(ProcessEvent::Stdout(data)) => {
                    Some(String::from_utf8_lossy(data).into_owned())
                }
                _ => None,
            })
            .collect();
        let stderr_lines: Vec<String> = events
            .iter()
            .filter_map(|event| match &event.event {
                ProtocolEvent::Process(ProcessEvent::Stderr(data)) => {
                    Some(String::from_utf8_lossy(data).into_owned())
                }
                _ => None,
            })
            .collect();
        assert_eq!(stdout_lines, vec!["out".to_string()]);
        assert_eq!(stderr_lines, vec!["err".to_string()]);
    }

    #[proxima::test]
    async fn bridge_surfaces_nonzero_exit_code() {
        let (events, exit_code) = run_stage("exit 3").await;
        assert_eq!(exit_code, Some(3));
        let terminal = events
            .iter()
            .rev()
            .find(|event| {
                matches!(
                    event.event,
                    ProtocolEvent::Process(ProcessEvent::Exited { .. })
                )
            })
            .expect("exited event");
        assert!(matches!(
            terminal.event,
            ProtocolEvent::Process(ProcessEvent::Exited { exit_code: Some(3) })
        ));
    }

    #[proxima::test]
    async fn bridge_stamps_parent_pipeline_id_on_every_event() {
        let (events, _exit_code) = run_stage("echo a; echo b; exit 0").await;
        let mut parents = std::collections::HashSet::new();
        for event in &events {
            if let Some(parent) = event.parent {
                parents.insert(parent);
            }
        }
        assert_eq!(parents.len(), 1, "every event must share one parent id");
    }

    // intentionally tight: validates the bridge does not hang on a
    // process that exits immediately (no stdout, no stderr).
    #[proxima::test]
    async fn bridge_handles_silent_immediate_exit_without_hanging() {
        let result = tokio::time::timeout(Duration::from_secs(5), run_stage("exit 0")).await;
        let (events, exit_code) = result.expect("must not hang");
        assert_eq!(exit_code, Some(0));
        // at minimum: Started + Exited
        assert!(events.len() >= 2);
    }
}
