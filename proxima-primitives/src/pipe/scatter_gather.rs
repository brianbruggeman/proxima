//! `ScatterGather<Source, Policy>` — scatter one query to N source
//! [`SendPipe`]s, gather all responses into a `Vec`.
//!
//! One input is broadcast to N sources concurrently (`futures::future::
//! join_all`, wait-all, no cancellation); each source's `Out` is collected,
//! ordered by source index. `Policy` governs errors: [`AllOrNothing`] returns
//! the first error; [`crate::pipe::BestEffort`] surfaces the first error after
//! gathering; [`crate::pipe::IgnoreErrors`] drops errors and returns the partial
//! `Vec`.
//!
//! # Not fan-in
//!
//! True fan-in (N continuous producers merged into one consumer) is
//! [`crate::pipe::FanIn`]. `ScatterGather` is request/response: one call → N
//! calls → one `Vec`. Different shapes; neither replaces the other.
//!
//! # Cancellation contract
//!
//! Requires `Source: DropSafe` for parity with [`crate::pipe::Race`] and to allow
//! future cancelling policies — the in-flight futures are abandoned if the
//! returned future is dropped. The marker is cheap for the common cases
//! (pure computation, detached-blocking sources).

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::marker::PhantomData;

use futures::future::join_all;
use proxima_core::markers::DropSafe;
use crate::pipe::SendPipe;

use crate::pipe::fanout::{BestEffort, FanPolicy};

/// Scatter one query to N drop-safe sources, gather their responses.
pub struct ScatterGather<Source, Policy = BestEffort> {
    sources: Arc<Vec<Source>>,
    policy: PhantomData<fn() -> Policy>,
}

impl<Source, Policy> Clone for ScatterGather<Source, Policy> {
    fn clone(&self) -> Self {
        Self {
            sources: Arc::clone(&self.sources),
            policy: PhantomData,
        }
    }
}

impl<Source, Policy> ScatterGather<Source, Policy> {
    /// Scatter to `sources` (empty is allowed — it gathers an empty `Vec`).
    #[must_use]
    pub fn new(sources: Vec<Source>) -> Self {
        Self {
            sources: Arc::new(sources),
            policy: PhantomData,
        }
    }

    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

impl<Source, Policy> SendPipe for ScatterGather<Source, Policy>
where
    Source: SendPipe + DropSafe,
    Source::In: Clone + Send,
    Source::Out: Send,
    Policy: FanPolicy,
{
    type In = Source::In;
    type Out = Vec<Source::Out>;
    type Err = Source::Err;

    fn call(
        &self,
        query: Source::In,
    ) -> impl Future<Output = Result<Vec<Source::Out>, Source::Err>> + Send {
        let sources = Arc::clone(&self.sources);
        async move {
            let live: Vec<_> = sources
                .iter()
                .map(|source| Box::pin(source.call(query.clone())))
                .collect();
            let outcomes = join_all(live).await;

            let mut gathered: Vec<Source::Out> = Vec::with_capacity(outcomes.len());
            let mut first_err: Option<Source::Err> = None;
            for outcome in outcomes {
                match outcome {
                    Ok(output) => gathered.push(output),
                    Err(err) => {
                        if Policy::SHORT_CIRCUIT {
                            return Err(err);
                        }
                        if !Policy::IGNORE_ERRORS {
                            first_err.get_or_insert(err);
                        }
                    }
                }
            }
            match first_err {
                Some(err) => Err(err),
                None => Ok(gathered),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::pipe::{AllOrNothing, IgnoreErrors};
    use alloc::sync::Arc;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use futures::executor::block_on;

    #[derive(Debug, PartialEq)]
    struct SrcErr(u32);

    struct Source {
        calls: Arc<AtomicUsize>,
        fail: bool,
        reply: u32,
    }

    impl SendPipe for Source {
        type In = u32;
        type Out = u32;
        type Err = SrcErr;

        fn call(&self, input: u32) -> impl Future<Output = Result<u32, SrcErr>> + Send {
            let calls = Arc::clone(&self.calls);
            let fail = self.fail;
            let reply = self.reply;
            async move {
                calls.fetch_add(1, Ordering::Relaxed);
                if fail { Err(SrcErr(input)) } else { Ok(reply) }
            }
        }
    }

    impl DropSafe for Source {}

    fn src(calls: &Arc<AtomicUsize>, reply: u32) -> Source {
        Source {
            calls: Arc::clone(calls),
            fail: false,
            reply,
        }
    }
    fn bad(calls: &Arc<AtomicUsize>) -> Source {
        Source {
            calls: Arc::clone(calls),
            fail: true,
            reply: 0,
        }
    }

    #[test]
    fn gathers_all_responses_in_source_order() {
        let calls = Arc::new(AtomicUsize::new(0));
        let gather = ScatterGather::<_, AllOrNothing>::new(vec![
            src(&calls, 10),
            src(&calls, 20),
            src(&calls, 30),
        ]);
        let out = block_on(gather.call(1)).unwrap();
        assert_eq!(out, vec![10, 20, 30]);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn all_or_nothing_surfaces_an_error() {
        let calls = Arc::new(AtomicUsize::new(0));
        let gather = ScatterGather::<_, AllOrNothing>::new(vec![src(&calls, 1), bad(&calls)]);
        assert_eq!(block_on(gather.call(9)), Err(SrcErr(9)));
    }

    #[test]
    fn ignore_errors_returns_partial_gather() {
        let calls = Arc::new(AtomicUsize::new(0));
        let gather = ScatterGather::<_, IgnoreErrors>::new(vec![
            src(&calls, 5),
            bad(&calls),
            src(&calls, 7),
        ]);
        let out = block_on(gather.call(2)).unwrap();
        assert_eq!(out, vec![5, 7], "failed source dropped, oks gathered");
        assert_eq!(calls.load(Ordering::Relaxed), 3, "every source attempted");
    }

    #[test]
    fn empty_sources_gathers_empty() {
        let gather = ScatterGather::<Source, BestEffort>::new(vec![]);
        assert_eq!(block_on(gather.call(0)).unwrap(), Vec::<u32>::new());
    }
}
