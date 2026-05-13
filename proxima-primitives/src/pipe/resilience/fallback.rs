use crate::pipe::primitives::Pipe;
use core::future::Future;

/// A [`Pipe`] combinator that routes to `secondary` when `primary` fails.
///
/// `primary.call(input.clone())` is tried first; on any error,
/// `secondary.call(input)` is called with the original input. On success,
/// `secondary` is never invoked.
///
/// Requires `P::In: Clone` so the input can be replayed.
#[derive(Debug, Clone)]
pub struct Fallback<P, S> {
    pub primary: P,
    pub secondary: S,
}

impl<P, S> Pipe for Fallback<P, S>
where
    P: Pipe,
    S: Pipe<In = P::In, Out = P::Out, Err = P::Err>,
    P::In: Clone,
{
    type In = P::In;
    type Out = P::Out;
    type Err = P::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            match self.primary.call(input.clone()).await {
                Ok(out) => Ok(out),
                Err(_) => self.secondary.call(input).await,
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use core::future::Future;
    use core::task::Poll;

    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct AlwaysFail;

    impl Pipe for AlwaysFail {
        type In = u32;
        type Out = u32;
        type Err = &'static str;

        fn call(&self, _input: u32) -> impl Future<Output = Result<u32, &'static str>> {
            async { Err("primary failed") }
        }
    }

    #[derive(Debug, Clone)]
    struct AlwaysOk(u32);

    impl Pipe for AlwaysOk {
        type In = u32;
        type Out = u32;
        type Err = &'static str;

        fn call(&self, _input: u32) -> impl Future<Output = Result<u32, &'static str>> {
            let value = self.0;
            async move { Ok(value) }
        }
    }

    #[derive(Debug, Clone)]
    struct DoubleInput;

    impl Pipe for DoubleInput {
        type In = u32;
        type Out = u32;
        type Err = &'static str;

        fn call(&self, input: u32) -> impl Future<Output = Result<u32, &'static str>> {
            async move { Ok(input * 2) }
        }
    }

    /// secondary is AlwaysFail but primary succeeds — secondary must never be called
    #[test]
    fn fallback_uses_primary_when_it_succeeds() {
        let fallback = Fallback {
            primary: AlwaysOk(10),
            secondary: AlwaysFail,
        };
        let result = block_on(Pipe::call(&fallback, 0));
        assert_eq!(result, Ok(10), "primary result returned unchanged");
    }

    #[test]
    fn fallback_uses_secondary_when_primary_fails() {
        let fallback = Fallback {
            primary: AlwaysFail,
            secondary: AlwaysOk(42),
        };
        let result = block_on(Pipe::call(&fallback, 0));
        assert_eq!(result, Ok(42), "secondary result after primary error");
    }

    #[test]
    fn fallback_passes_original_input_to_secondary() {
        let fallback = Fallback {
            primary: AlwaysFail,
            secondary: DoubleInput,
        };
        let result = block_on(Pipe::call(&fallback, 7));
        assert_eq!(result, Ok(14), "secondary receives original input");
    }

    #[test]
    fn fallback_does_not_call_secondary_when_primary_ok() {
        // secondary would return Err; getting Ok proves secondary was skipped
        struct SuccessOrBust(u32);
        impl Pipe for SuccessOrBust {
            type In = u32;
            type Out = u32;
            type Err = &'static str;
            fn call(&self, _input: u32) -> impl Future<Output = Result<u32, &'static str>> {
                let value = self.0;
                async move { Ok(value) }
            }
        }
        let fallback = Fallback {
            primary: SuccessOrBust(99),
            secondary: AlwaysFail,
        };
        let result = block_on(Pipe::call(&fallback, 0));
        assert_eq!(result, Ok(99));
    }
}
