use bytes::Bytes;
use serde::de::DeserializeOwned;

use crate::body::ResponseStream;
use crate::error::ProximaError;
use crate::header_list::HeaderList;
use crate::request::Response as ProximaResponse;

/// Wrapper around the inner proxima `Response` that exposes
/// `requests`-shaped consumers (`text`, `json`, `bytes`).
pub struct Response {
    inner: ProximaResponse<Bytes>,
}

impl Response {
    pub(crate) fn from_proxima(inner: ProximaResponse<Bytes>) -> Self {
        Self { inner }
    }

    #[must_use]
    pub fn status(&self) -> u16 {
        self.inner.status
    }

    #[must_use]
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.inner.status)
    }

    #[must_use]
    pub fn headers(&self) -> &HeaderList {
        &self.inner.metadata
    }

    pub async fn bytes(self) -> Result<Bytes, ProximaError> {
        self.inner.collect_body().await
    }

    /// Take the streamed-body view: a present stream, else a one-chunk
    /// stream of the buffered bytes.
    #[must_use]
    pub fn into_body(self) -> ResponseStream {
        ResponseStream::from_chunk_stream(self.inner.into_chunk_stream())
    }

    pub async fn text(self) -> Result<String, ProximaError> {
        let bytes = self.inner.collect_body().await?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub async fn json<T: DeserializeOwned>(self) -> Result<T, ProximaError> {
        let bytes = self.inner.collect_body().await?;
        serde_json::from_slice(&bytes)
            .map_err(|err| ProximaError::Decode(format!("client json: {err}")))
    }

    /// Convert a non-2xx status into a typed error so callers can use `?`
    /// without juggling the status check separately. Mirrors
    /// `reqwest::Response::error_for_status`.
    pub fn error_for_status(self) -> Result<Self, ProximaError> {
        if self.ok() {
            Ok(self)
        } else {
            Err(ProximaError::Upstream(format!(
                "status {}",
                self.inner.status
            )))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make(status: u16, body: &str) -> Response {
        Response {
            inner: ProximaResponse::new(status).with_body(Bytes::from(body.to_string())),
        }
    }

    #[proxima::test]
    async fn text_collects_body_as_utf8_string() {
        let response = make(200, "hello");
        assert_eq!(response.text().await.expect("text"), "hello");
    }

    #[proxima::test]
    async fn json_deserializes_response_body() {
        let response = make(200, "{\"x\":1}");
        let parsed: serde_json::Value = response.json().await.expect("json");
        assert_eq!(parsed["x"], 1);
    }

    #[proxima::test]
    async fn error_for_status_passes_2xx_and_fails_5xx() {
        let ok = make(200, "ok");
        assert!(ok.error_for_status().is_ok());
        let err = make(503, "fail");
        assert!(err.error_for_status().is_err());
    }

    #[test]
    fn ok_helper_predicates_2xx() {
        assert!(make(200, "").ok());
        assert!(make(299, "").ok());
        assert!(!make(199, "").ok());
        assert!(!make(300, "").ok());
    }
}
