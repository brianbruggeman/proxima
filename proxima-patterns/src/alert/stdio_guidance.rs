//! C5 — `StdioGuidancePipe`: duplex sync stdin/stdout for guidance round-trip.
//!
//! Receives a [`GuidanceQuestion`] as `request.payload`, writes the prompt
//! to stdout, reads one line from stdin, and returns a [`GuidanceAnswer`] as
//! `response.payload`. No serialization; purely in-process typed pipe.
//!
//! Markers:
//! - NOT `WithoutFilesystem` (stdin/stdout are fs-shaped via the kernel
//!   pseudo-device).
//! - NOT `IdempotentSideEffectFree` (reading stdin twice → different lines).
//! - `WithoutNetwork`, `WithoutSpawn`, `WithoutTime`, `WithoutRandom`.

use std::sync::Arc;

use proxima_core::markers::{WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime};
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Response;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use ulid::Ulid;

use crate::alert::event::{AnswerString, GuidanceAnswer, GuidanceRequestId, ResponderString};
use crate::alert::methods;
use crate::alert::pipes::{GuidanceRequest, GuidanceResponse};

/// Async reader trait for stdin injection.
pub trait GuidanceReader: tokio::io::AsyncBufRead + Send + Sync + Unpin + 'static {}
impl<T: tokio::io::AsyncBufRead + Send + Sync + Unpin + 'static> GuidanceReader for T {}

/// Async writer trait for stdout injection.
pub trait GuidanceWriter: tokio::io::AsyncWrite + Send + Sync + Unpin + 'static {}
impl<T: tokio::io::AsyncWrite + Send + Sync + Unpin + 'static> GuidanceWriter for T {}

/// Duplex sync stdin/stdout guidance pipe.
pub struct StdioGuidancePipe {
    reader: Arc<Mutex<Box<dyn GuidanceReader>>>,
    writer: Arc<Mutex<Box<dyn GuidanceWriter>>>,
    responder_label: String,
    prompt_prefix: String,
}

impl Default for StdioGuidancePipe {
    fn default() -> Self {
        let stdin = BufReader::new(tokio::io::stdin());
        let stdout = tokio::io::stdout();
        Self {
            reader: Arc::new(Mutex::new(Box::new(stdin))),
            writer: Arc::new(Mutex::new(Box::new(stdout))),
            responder_label: "stdin".to_string(),
            prompt_prefix: "[ask] ".to_string(),
        }
    }
}

impl StdioGuidancePipe {
    /// Build the production version that reads stdin and writes stdout.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inject specific reader/writer pair — used by tests.
    #[must_use]
    pub fn with_io<R: GuidanceReader, W: GuidanceWriter>(reader: R, writer: W) -> Self {
        Self {
            reader: Arc::new(Mutex::new(Box::new(reader))),
            writer: Arc::new(Mutex::new(Box::new(writer))),
            responder_label: "stdin".to_string(),
            prompt_prefix: "[ask] ".to_string(),
        }
    }

    /// Fluent builder entry point (principle 4).
    #[must_use]
    pub fn builder() -> StdioGuidancePipeBuilder {
        StdioGuidancePipeBuilder::default()
    }
}

impl SendPipe for StdioGuidancePipe {
    type In = GuidanceRequest;
    type Out = GuidanceResponse;
    type Err = ProximaError;

    fn call(
        &self,
        request: GuidanceRequest,
    ) -> impl std::future::Future<Output = Result<GuidanceResponse, ProximaError>> + Send {
        let method_known = request.method.as_bytes() == methods::GUIDANCE_QUESTION;
        let question = request.payload;
        let reader = self.reader.clone();
        let writer = self.writer.clone();
        let responder_label = self.responder_label.clone();
        let prompt_prefix = self.prompt_prefix.clone();
        async move {
            if !method_known {
                return Ok(Response::typed(405, make_empty_answer()));
            }

            {
                let mut writer_guard = writer.lock().await;
                writer_guard
                    .write_all(prompt_prefix.as_bytes())
                    .await
                    .map_err(ProximaError::Io)?;
                writer_guard
                    .write_all(question.question.as_str().as_bytes())
                    .await
                    .map_err(ProximaError::Io)?;
                writer_guard
                    .write_all(b"\n")
                    .await
                    .map_err(ProximaError::Io)?;
                writer_guard.flush().await.map_err(ProximaError::Io)?;
            }

            let mut line = String::new();
            {
                let mut reader_guard = reader.lock().await;
                let _ = reader_guard
                    .read_line(&mut line)
                    .await
                    .map_err(ProximaError::Io)?;
            }
            while line.ends_with('\n') || line.ends_with('\r') {
                line.pop();
            }

            let content = truncate_to_answer(&line);
            let responder = ResponderString::try_from(responder_label.as_str()).unwrap_or_default();

            let answer = GuidanceAnswer {
                request_id: question.id,
                content,
                responder,
                responded_at_micros: now_micros(),
            };
            Ok(Response::typed(200, answer))
        }
    }
}

impl WithoutNetwork for StdioGuidancePipe {}
impl WithoutSpawn for StdioGuidancePipe {}
impl WithoutTime for StdioGuidancePipe {}
impl WithoutRandom for StdioGuidancePipe {}

fn make_empty_answer() -> GuidanceAnswer {
    GuidanceAnswer {
        request_id: GuidanceRequestId(Ulid::nil()),
        content: AnswerString::new(),
        responder: ResponderString::new(),
        responded_at_micros: 0,
    }
}

fn truncate_to_answer(value: &str) -> AnswerString {
    let max = crate::alert::event::sized::GUIDANCE_ANSWER_MAX;
    let truncated = if value.len() > max {
        &value[..max]
    } else {
        value
    };
    AnswerString::try_from(truncated).unwrap_or_default()
}

fn now_micros() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_micros()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Helper: construct a `GuidanceRequestId` from a Ulid for callers that
/// don't depend on the proto crate directly.
#[must_use]
pub fn new_request_id() -> GuidanceRequestId {
    GuidanceRequestId(Ulid::new())
}

/// Builder for [`StdioGuidancePipe`] (principle 4).
#[derive(Default)]
pub struct StdioGuidancePipeBuilder {
    responder_label: Option<String>,
    prompt_prefix: Option<String>,
}

impl StdioGuidancePipeBuilder {
    /// Override the default responder label (`"stdin"`).
    #[must_use]
    pub fn responder_label(mut self, value: impl Into<String>) -> Self {
        self.responder_label = Some(value.into());
        self
    }

    /// Override the default prompt prefix (`"[ask] "`).
    #[must_use]
    pub fn prompt_prefix(mut self, value: impl Into<String>) -> Self {
        self.prompt_prefix = Some(value.into());
        self
    }

    /// Build the immutable [`StdioGuidancePipe`] with real stdin/stdout.
    #[must_use]
    pub fn build(self) -> StdioGuidancePipe {
        let mut pipe = StdioGuidancePipe::default();
        if let Some(label) = self.responder_label {
            pipe.responder_label = label;
        }
        if let Some(prefix) = self.prompt_prefix {
            pipe.prompt_prefix = prefix;
        }
        pipe
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use bytes::Bytes;
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::header_list::HeaderList;
    use proxima_primitives::pipe::request::{Request, RequestContext};
    use tokio::io::BufReader;

    use crate::alert::event::{ContextBytes, GuidanceAnswer, GuidanceQuestion, QuestionString};

    use super::*;

    fn make_question(text: &str) -> GuidanceQuestion {
        GuidanceQuestion {
            id: new_request_id(),
            agent_id: crate::alert::event::AgentId(Ulid::nil()),
            parent_id: None,
            question: QuestionString::try_from(text).unwrap(),
            context: ContextBytes::new(),
            asked_at_micros: 0,
            timeout_micros: 0,
        }
    }

    fn make_guidance_request(question: GuidanceQuestion) -> GuidanceRequest {
        Request {
            method: methods::guidance_question_method(),
            path: Bytes::from_static(b"/guidance"),
            query: HeaderList::new(),
            metadata: HeaderList::new(),
            payload: question,
            stream: None,
            context: RequestContext::default(),
        }
    }

    #[proxima::test]
    async fn unknown_method_returns_405() {
        let pipe = StdioGuidancePipe::with_io(BufReader::new(&b""[..]), Vec::<u8>::new());
        let question = make_question("irrelevant");
        let request = Request {
            method: proxima_primitives::pipe::method::Method::from_bytes(b"UNKNOWN"),
            path: Bytes::from_static(b"/"),
            query: HeaderList::new(),
            metadata: HeaderList::new(),
            payload: question,
            stream: None,
            context: RequestContext::default(),
        };
        let response = pipe.call(request).await.expect("call should not error");
        assert_eq!(response.status, 405);
    }

    #[proxima::test]
    async fn full_question_answer_roundtrip_via_injected_io() {
        let stdin_tape = b"Push the parser change first.\n".to_vec();
        let reader = BufReader::new(std::io::Cursor::new(stdin_tape));
        let writer = Vec::<u8>::new();
        let pipe = StdioGuidancePipe::with_io(reader, writer);

        let question = make_question("Should I push refactor or land parser?");
        let question_id = question.id;
        let request = make_guidance_request(question);

        let response = pipe.call(request).await.expect("call should succeed");
        assert_eq!(response.status, 200);
        let answer: &GuidanceAnswer = &response.payload;
        assert_eq!(answer.request_id, question_id);
        assert_eq!(answer.content.as_str(), "Push the parser change first.");
        assert_eq!(answer.responder.as_str(), "stdin");
        assert!(answer.responded_at_micros > 0);
    }

    #[proxima::test]
    async fn builder_sets_responder_label_and_prompt_prefix() {
        let pipe = StdioGuidancePipe::builder()
            .responder_label("test-responder")
            .prompt_prefix("?> ")
            .build();
        assert_eq!(pipe.responder_label, "test-responder");
        assert_eq!(pipe.prompt_prefix, "?> ");
    }
}
