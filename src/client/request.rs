use std::collections::BTreeMap;

use bytes::Bytes;
use serde::Serialize;

use crate::client::Response;
use crate::client::handle::Client;
use crate::error::ProximaError;
use crate::request::Request as ProximaRequest;

pub struct RequestBuilder {
    client: Client,
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    query: BTreeMap<String, String>,
    body: Option<Bytes>,
}

impl RequestBuilder {
    pub(crate) fn new(client: Client, method: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            client,
            method: method.into(),
            path: path.into(),
            headers: BTreeMap::new(),
            query: BTreeMap::new(),
            body: None,
        }
    }

    #[must_use]
    pub fn method(&self) -> &str {
        &self.method
    }

    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    #[must_use]
    pub fn query(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.insert(name.into(), value.into());
        self
    }

    #[must_use]
    pub fn body(mut self, body: impl Into<Bytes>) -> Self {
        self.body = Some(body.into());
        self
    }

    pub fn json<T: Serialize>(mut self, value: &T) -> Result<Self, ProximaError> {
        let bytes = serde_json::to_vec(value)
            .map_err(|err| ProximaError::Encode(format!("client json: {err}")))?;
        self.headers
            .entry("content-type".into())
            .or_insert_with(|| "application/json".into());
        self.body = Some(Bytes::from(bytes));
        Ok(self)
    }

    #[must_use]
    pub fn text(mut self, body: impl Into<String>) -> Self {
        let text: String = body.into();
        self.headers
            .entry("content-type".into())
            .or_insert_with(|| "text/plain; charset=utf-8".into());
        self.body = Some(Bytes::from(text));
        self
    }

    pub async fn send(self) -> Result<Response, ProximaError> {
        let mut builder = ProximaRequest::builder()
            .method(self.method.as_str())
            .path(self.path);
        for (name, value) in self.headers {
            builder = builder.header(name, value);
        }
        for (name, value) in self.query {
            builder = builder.query_param(name, value);
        }
        if let Some(body) = self.body {
            builder = builder.body(body);
        }
        let request = builder.build()?;

        // one dispatch seam (shared with `impl Pipe for Client`): on-worker calls
        // the handle directly; off-worker hops onto the client's runtime.
        let response = self.client.dispatch(request).await?;
        Ok(Response::from_proxima(response))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn synth_client() -> Client {
        Client::from_value(json!({
            "synth": { "status": 200, "body": "ok" },
        }))
        .expect("build")
    }

    #[proxima::test]
    async fn header_chaining_propagates_to_outgoing_request() {
        let request = synth_client()
            .call("GET", "/x")
            .header("x-trace", "abc")
            .query("a", "1");
        assert_eq!(request.method(), "GET");
        assert_eq!(request.path(), "/x");
        assert_eq!(request.headers.get("x-trace"), Some(&"abc".to_string()));
        assert_eq!(request.query.get("a"), Some(&"1".to_string()));
    }

    #[proxima::test]
    async fn json_sets_content_type_and_body() {
        let request = synth_client()
            .call("POST", "/")
            .json(&json!({ "x": 1 }))
            .expect("json");
        assert_eq!(
            request.headers.get("content-type"),
            Some(&"application/json".to_string()),
        );
        assert!(request.body.is_some());
    }

    #[proxima::test]
    async fn send_dispatches_through_pipe_and_returns_response() {
        let response = synth_client().call("GET", "/").send().await.expect("send");
        assert_eq!(response.status(), 200);
        let text = response.text().await.expect("text");
        assert_eq!(text, "ok");
    }
}
