//! [`KeyedLiveFilter`] — a live, reconfigurable id-set predicate over
//! `Request<Bytes>`, itself a `SendPipe<In = Out = Request<Bytes>>` decision
//! that drops straight into `.and_then(inner)`.
//!
//! It composes two primitives that already exist, rather than inventing a
//! third extraction mechanism:
//!
//! - [`KeyOf`] / [`KeyExtractor`](crate::pipe::rate_limit::KeyExtractor) — the same
//!   seam `RateLimit` uses to pull a byte key out of a request (constant,
//!   header, or path+method).
//! - [`LiveFilter`] / [`IdSet`] / [`FilterControl`] — the live-swappable
//!   id-set membership predicate from `crate::pipe::live_filter`.
//!
//! `KeyedLiveFilter` extracts the request key via [`KeyOf::rate_key`] and
//! delegates membership to the inner [`LiveFilter`]'s synchronous
//! [`LiveFilter::contains`] — HTTP-specific, so it sits alongside `filter.rs`'s
//! own `Predicate`/`FilterConfig` (which decide generically over any `In`)
//! as a distinct decision type.

use core::future::Future;
use core::str;

use bytes::Bytes;

use crate::pipe::SendPipe;
use crate::pipe::capabilities::KeyOf;
use crate::pipe::live_filter::{FilterControl, IdSet, LiveFilter, live_filter_ids};

use crate::pipe::rate_limit::KeyExtractor;
use crate::pipe::request::Request;
use proxima_core::ProximaError;

/// A `SendPipe<In = Out = Request<Bytes>>` decision whose matched id-set is
/// live-reconfigurable through the paired [`FilterControl`]. Cheap clone: an
/// `Arc` bump for the live cell plus the extractor's own (typically small)
/// clone cost.
#[derive(Clone)]
pub struct KeyedLiveFilter<Extractor = KeyExtractor> {
    extractor: Extractor,
    filter: LiveFilter<IdSet<String>>,
}

impl<Extractor> KeyedLiveFilter<Extractor> {
    /// Pair an extractor with an already-split [`LiveFilter`] half — for
    /// sharing one live id-set across more than one `KeyedLiveFilter` (e.g.
    /// distinct extractors gated by the same subscription set).
    #[must_use]
    pub fn new(extractor: Extractor, filter: LiveFilter<IdSet<String>>) -> Self {
        Self { extractor, filter }
    }
}

/// Split a [`KeyedLiveFilter`] seeded from `ids`, keyed by `extractor` (e.g.
/// `KeyExtractor::Header("x-correlation-id".into())`). Equivalent to pairing
/// [`live_filter_ids`] with [`KeyedLiveFilter::new`].
#[must_use]
pub fn keyed_live_filter_ids<Extractor>(
    extractor: Extractor,
    ids: impl IntoIterator<Item = String>,
) -> (KeyedLiveFilter<Extractor>, FilterControl<IdSet<String>>) {
    let (filter, control) = live_filter_ids(ids);
    (KeyedLiveFilter::new(extractor, filter), control)
}

impl<Extractor> KeyedLiveFilter<Extractor>
where
    Request<Bytes>: KeyOf<Extractor>,
{
    fn admits(&self, input: &Request<Bytes>) -> bool {
        let key = input.rate_key(&self.extractor);
        // the id-set stores owned `String` members, so a borrowed byte key
        // must become one to look up membership. `KeyOf` already allocates
        // for its synthesized `PathAndMethod` case, so this is not a new
        // cost class, just the same tradeoff paid on every extractor variant.
        let Ok(key) = str::from_utf8(key.as_ref()) else {
            return false;
        };
        self.filter.contains(&key.to_string())
    }
}

impl<Extractor> SendPipe for KeyedLiveFilter<Extractor>
where
    Extractor: Send + Sync + 'static,
    Request<Bytes>: KeyOf<Extractor>,
{
    type In = Request<Bytes>;
    type Out = Request<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        input: Request<Bytes>,
    ) -> impl Future<Output = Result<Request<Bytes>, ProximaError>> + Send {
        let admitted = self.admits(&input);
        async move {
            if admitted {
                Ok(input)
            } else {
                Err(ProximaError::Forbidden("forbidden".into()))
            }
        }
    }
}

// `#[proxima::test]` pulls in the `proxima` dev-dependency, which the
// loom build keeps out of the graph (see
// `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::future::Future;

    use bytes::Bytes;
    use proxima_core::ProximaError;
    use crate::pipe::SendPipe;

    use super::*;
    use crate::pipe::handler::{PipeHandle, into_handle};
    use crate::pipe::request::Response;

    fn echo_pipe() -> PipeHandle {
        struct EchoPipe;
        impl SendPipe for EchoPipe {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
                async move {
                    let (_, body) = request.body_bytes().await?;
                    Ok(Response::new(200).with_body(body))
                }
            }
        }
        into_handle(EchoPipe)
    }

    fn order_request(correlation_id: &str) -> Request<Bytes> {
        Request::builder()
            .method("POST")
            .path("/orders")
            .header("x-correlation-id", correlation_id)
            .body(Bytes::from_static(b"{\"sku\":\"widget-42\"}"))
            .build()
            .expect("builder")
    }

    #[proxima::test]
    async fn keyed_live_filter_gates_on_header_and_reconfigures_live() {
        let subscribed = "corr-7f3a9c2e1b4d".to_string();
        let unsubscribed = "corr-0f43d9c0aa18".to_string();

        let (keyed_filter, control) = keyed_live_filter_ids(
            KeyExtractor::Header("x-correlation-id".to_string()),
            [subscribed.clone()],
        );
        let stack = keyed_filter.and_then(echo_pipe());

        let admitted = SendPipe::call(&stack, order_request(&subscribed))
            .await
            .expect("call");
        assert_eq!(
            admitted.status, 200,
            "subscribed correlation id reaches the inner pipe"
        );

        let rejected = SendPipe::call(&stack, order_request(&unsubscribed)).await;
        assert!(
            matches!(rejected, Err(ProximaError::Forbidden(_))),
            "unsubscribed correlation id is rejected"
        );

        control.add(unsubscribed.clone());

        let now_admitted = SendPipe::call(&stack, order_request(&unsubscribed))
            .await
            .expect("call");
        assert_eq!(
            now_admitted.status, 200,
            "control.add makes a previously-rejected correlation id pass"
        );
    }
}
