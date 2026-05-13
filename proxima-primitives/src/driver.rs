//! the no-runtime drive primitive: [`block_on`].

/// Drive `future` to completion on the calling thread with no runtime, no
/// reactor, and no allocator — a `core`-only `Waker::noop` poll loop (stable
/// since Rust 1.85). This is the floor every other `block_on` in the workspace
/// points down to: the runtime-holding `proxima_runtime::block_on(&dyn Runtime,
/// ..)` and the edge `run*` drivers add a runtime ON TOP of the same verb.
///
/// Use it for a sync boundary, a `no_std`/bare-metal caller, or a bench where
/// the future resolves without ever parking (nothing wakes the noop waker, so a
/// future that genuinely suspends would spin here). Never call it from inside
/// async code — it busy-loops the calling thread instead of yielding.
pub fn block_on<Fut: core::future::Future>(future: Fut) -> Fut::Output {
    let mut future = core::pin::pin!(future);
    let mut context = core::task::Context::from_waker(core::task::Waker::noop());
    loop {
        if let core::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
    }
}
