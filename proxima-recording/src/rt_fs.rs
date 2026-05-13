//! Runtime-agnostic blocking-I/O offload for the recording sinks/sources.
//!
//! File I/O is inherently blocking — there is no portable async file syscall;
//! `tokio::fs` just hides a blocking thread pool. To keep the recording
//! substrate runtime-swappable (prime first, tokio one injected backend) the
//! on-disk formats run their `std::fs` work through
//! [`proxima_runtime::Runtime::spawn_background_blocking`] — the same offload
//! pool `proxima-pgwire` routes SCRAM-KDF through
//! (`proxima-pgwire/src/connection.rs:1160`) — and await the `Send` handle.
//! Awaiting yields the per-core caller (the prime serve task) instead of
//! stalling its reactor; the blocking syscall runs on the pool
//! (`ProximaBackgroundPool` under prime, `spawn_blocking` under tokio).
//!
//! The caller injects the runtime, so the same code path drives either
//! backend — no `tokio::fs`, no per-crate runtime selection.

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::sync::Arc;
use core::any::Any;

use proxima_core::ProximaError;
use proxima_runtime::Runtime;

/// Run a blocking closure on the runtime's background pool and await its
/// result, yielding the calling core meanwhile.
///
/// Mirrors the `spawn_background_blocking` usage in `proxima-pgwire`: the
/// closure is boxed as `Box<dyn Any + Send>`, the `Send` handle is awaited,
/// and the result is downcast back. The downcast cannot fail — the closure
/// boxes exactly `Out` — but a mismatch is surfaced as an error rather than a
/// panic, per the no-panic discipline.
pub async fn offload<Work, Out>(runtime: &Arc<dyn Runtime>, work: Work) -> Result<Out, ProximaError>
where
    Work: FnOnce() -> Result<Out, ProximaError> + Send + 'static,
    Out: Send + 'static,
{
    let erased: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>, ProximaError> + Send> =
        Box::new(move || work().map(|value| Box::new(value) as Box<dyn Any + Send>));
    let result = runtime.spawn_background_blocking(erased).await?;
    result
        .downcast::<Out>()
        .map(|boxed| *boxed)
        .map_err(|_| ProximaError::Record("rt_fs offload returned an unexpected type".to_string()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::path::Path;

    use prime::os::runtime::PrimeRuntime;
    use proxima_runtime::tokio::TokioPerCoreRuntime;

    // a real frame the BinSink path would hand the filesystem, not an AAAA stub
    const PAYLOAD: &[u8] = b"recording-runtime-swap seam: [u32 len][postcard BinEnvelope] block";

    // offload a std::fs write then a std::fs read through the injected runtime;
    // returns the bytes that round-tripped through the pool-driven file I/O.
    async fn write_then_read(runtime: &Arc<dyn Runtime>, path: &Path) -> Vec<u8> {
        let write_path = path.to_path_buf();
        offload(runtime, move || {
            std::fs::write(&write_path, PAYLOAD)
                .map_err(|err| ProximaError::Record(err.to_string()))
        })
        .await
        .expect("write offload");

        let read_path = path.to_path_buf();
        offload(runtime, move || {
            std::fs::read(&read_path).map_err(|err| ProximaError::Record(err.to_string()))
        })
        .await
        .expect("read offload")
    }

    // prime is the default backend: drive the seam on ProximaBackgroundPool.
    #[test]
    fn offload_round_trips_on_prime_backend() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("seam.bin");
        let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1).expect("prime runtime"));
        let bytes = futures::executor::block_on(write_then_read(&runtime, &path));
        assert_eq!(
            bytes, PAYLOAD,
            "prime-driven offload must round-trip the bytes"
        );
    }

    // tokio is one swappable backend: the SAME seam on spawn_blocking.
    #[test]
    fn offload_round_trips_on_tokio_backend() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("seam.bin");
        let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let bytes = tokio_runtime.block_on(async {
            let runtime: Arc<dyn Runtime> =
                Arc::new(TokioPerCoreRuntime::new(1).expect("tokio runtime"));
            write_then_read(&runtime, &path).await
        });
        assert_eq!(
            bytes, PAYLOAD,
            "tokio-driven offload must round-trip the bytes"
        );
    }

    // the parity claim: both backends produce the identical result from the
    // identical offloaded file work — the seam is genuinely runtime-swappable.
    #[test]
    fn prime_and_tokio_backends_agree() {
        let dir = tempfile::tempdir().expect("tempdir");

        let prime_path = dir.path().join("prime.bin");
        let prime_runtime: Arc<dyn Runtime> =
            Arc::new(PrimeRuntime::new(1).expect("prime runtime"));
        let prime_bytes = futures::executor::block_on(write_then_read(&prime_runtime, &prime_path));

        let tokio_path = dir.path().join("tokio.bin");
        let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let tokio_bytes = tokio_runtime.block_on(async {
            let runtime: Arc<dyn Runtime> =
                Arc::new(TokioPerCoreRuntime::new(1).expect("tokio runtime"));
            write_then_read(&runtime, &tokio_path).await
        });

        assert_eq!(
            prime_bytes, tokio_bytes,
            "prime and tokio backends must agree"
        );
    }
}
