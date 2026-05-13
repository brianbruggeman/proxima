//! WebSocket replay: reconstruct a recorded WS session as a *real* upgrade
//! flow, not an HTTP body-chunk list.
//!
//! `ReplayUpstream` (the HTTP path) reconstructs a 101 recording as a
//! `Response { status: 101, body: <frame bytes> }` — bytes, not a working
//! WebSocket. `WsReplayUpstream` instead completes the RFC 6455 handshake
//! against the *replaying* client (recomputing `Sec-WebSocket-Accept` from
//! that client's `Sec-WebSocket-Key` — the recorded accept is bound to the
//! original client's key and would be rejected) and then streams the recorded
//! server frames back over the upgraded socket.
//!
//! The recorded `ResponseChunk` bytes for a WS interaction are server→client
//! frames, which per RFC 6455 §5.1 are never masked — so they replay verbatim.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use bytes::Bytes;
use futures::AsyncWriteExt;

use crate::event::{HttpEvent, InteractionId, ProtocolEvent, RecordingEvent};
use crate::source::RecordingSource;
use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::upgrade::UpgradeHandler;

use crate::replay::keying::{match_key_from_recording, match_key_from_request};

// Sec-WebSocket-Accept derivation is the shared handshake primitive; re-exported
// so the `proxima_recording::replay::ws::compute_accept_key` path callers use still holds.
pub use proxima_protocols::websocket_handshake::compute_accept_key;

/// One recorded WebSocket session: the ordered server→client frame byte
/// blocks captured during the original interaction.
#[derive(Debug, Clone, Default)]
struct WsRecording {
    frames: Vec<Bytes>,
}

/// Replays recorded WebSocket sessions as genuine upgrades. Indexes only the
/// 101 interactions in a recording (HTTP interactions are `ReplayUpstream`'s
/// job). On a matching upgrade request it completes the handshake and streams
/// the recorded frames.
pub struct WsReplayUpstream {
    label: String,
    by_match_key: HashMap<String, WsRecording>,
}

impl WsReplayUpstream {
    pub async fn from_source(
        source: std::sync::Arc<dyn RecordingSource>,
        label: impl Into<String>,
    ) -> Result<Self, ProximaError> {
        let by_match_key = index_ws_recording(source.as_ref()).await?;
        Ok(Self {
            label: label.into(),
            by_match_key,
        })
    }

    /// This replay upstream's label, set at construction. Carries no
    /// runtime behaviour beyond identification (TARGET 3 — served-Pipe
    /// naming now lives at the mount-site label, not the handle).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[must_use]
    pub fn known_keys(&self) -> Vec<String> {
        self.by_match_key.keys().cloned().collect()
    }

    /// Number of recorded frames for a given match key, or None if the key
    /// was not a recorded WS session. Exposed for tests/inspection.
    #[must_use]
    pub fn frame_count(&self, match_key: &str) -> Option<usize> {
        self.by_match_key.get(match_key).map(|rec| rec.frames.len())
    }
}

impl SendPipe for WsReplayUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let key = match_key_from_request(&request);
        let recording = self.by_match_key.get(&key).cloned();
        let client_key = request
            .metadata
            .get_str("sec-websocket-key")
            .map(str::to_string);
        async move {
            let Some(recording) = recording else {
                return Err(ProximaError::ReplayMiss { fingerprint: key });
            };
            let Some(client_key) = client_key else {
                return Err(ProximaError::Body(
                    "ws replay: request missing Sec-WebSocket-Key".to_string(),
                ));
            };

            let accept = compute_accept_key(&client_key);
            let frames = recording.frames;
            let handler = UpgradeHandler::new(move |socket| write_frames(socket, frames));

            Ok(Response::new(101)
                .with_header("upgrade", "websocket")
                .with_header("connection", "Upgrade")
                .with_header("sec-websocket-accept", accept)
                .with_upgrade(handler))
        }
    }
}


fn write_frames(
    mut socket: proxima_primitives::pipe::upgrade::HijackedSocket,
    frames: Vec<Bytes>,
) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send>> {
    Box::pin(async move {
        for frame in &frames {
            socket
                .stream
                .write_all(frame)
                .await
                .map_err(|err| ProximaError::Body(format!("ws replay write: {err}")))?;
        }
        socket
            .stream
            .flush()
            .await
            .map_err(|err| ProximaError::Body(format!("ws replay flush: {err}")))?;
        Ok(())
    })
}

async fn index_ws_recording(
    source: &dyn RecordingSource,
) -> Result<HashMap<String, WsRecording>, ProximaError> {
    use futures::StreamExt as _;

    struct Draft {
        match_key: String,
        is_ws: bool,
        frames: Vec<Bytes>,
    }

    let mut in_flight: HashMap<InteractionId, Draft> = HashMap::new();
    let mut output: HashMap<String, WsRecording> = HashMap::new();
    let mut events = source.events();

    while let Some(event) = events.next().await {
        let RecordingEvent { id, event, .. } = event?;
        match event {
            ProtocolEvent::Http(HttpEvent::Started { request, .. }) => {
                in_flight.insert(
                    id,
                    Draft {
                        match_key: match_key_from_recording(&request),
                        is_ws: false,
                        frames: Vec::new(),
                    },
                );
            }
            ProtocolEvent::Http(HttpEvent::ResponseStarted { status, .. }) => {
                if let Some(draft) = in_flight.get_mut(&id) {
                    // a 101 response is the upgrade-completion signal
                    draft.is_ws = status == 101;
                }
            }
            ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. }) => {
                if let Some(draft) = in_flight.get_mut(&id) {
                    draft.frames.push(data);
                }
            }
            ProtocolEvent::Http(HttpEvent::Ended { .. }) => {
                if let Some(draft) = in_flight.remove(&id)
                    && draft.is_ws
                {
                    output.insert(
                        draft.match_key,
                        WsRecording {
                            frames: draft.frames,
                        },
                    );
                }
            }
            _ => {}
        }
    }

    Ok(output)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn accept_key_matches_rfc6455_test_vector() {
        // RFC 6455 §1.3: key "dGhlIHNhbXBsZSBub25jZQ==" must yield
        // accept "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=".
        assert_eq!(
            compute_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn accept_key_is_deterministic_and_key_dependent() {
        let a = compute_accept_key("AAAAAAAAAAAAAAAAAAAAAA==");
        let b = compute_accept_key("BBBBBBBBBBBBBBBBBBBBBB==");
        assert_eq!(a, compute_accept_key("AAAAAAAAAAAAAAAAAAAAAA=="));
        assert_ne!(a, b, "different client keys must yield different accepts");
    }
}
