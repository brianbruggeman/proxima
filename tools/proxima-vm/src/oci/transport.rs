use bytes::Bytes;
use http::{HeaderName, HeaderValue, Request, Response};
use proxima_http::http1::PrimeHttpPipeFactory;
use proxima_primitives::pipe::{Method, PipeFactory, Request as PipeRequest, SendPipe};
use serde_json::json;

use crate::oci::error::OciError;

/// The registry transport pipe, in the shape `registry::pull` takes: one request, any host.
/// Must run on a prime worker (`oci::pull_blocking` provides one).
pub async fn fetch(request: Request<Bytes>) -> Result<Response<Bytes>, OciError> {
    let url = request.uri().to_string();
    let failure = |reason: String| OciError::Transport { url: url.clone(), reason };
    let scheme = request.uri().scheme_str().ok_or_else(|| failure("the url has no scheme".into()))?;
    let authority = request.uri().authority().ok_or_else(|| failure("the url has no host".into()))?;
    let origin = format!("{scheme}://{authority}");
    let pipe = PrimeHttpPipeFactory::new().build(&json!({ "url": origin }), None).await.map_err(|error| failure(format!("dial setup: {error}")))?;
    let mut builder = PipeRequest::builder().method(Method::Get).path(request.uri().path().to_string());
    for pair in request.uri().query().into_iter().flat_map(|query| query.split('&')) {
        let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
        builder = builder.query_param(name.to_string(), value.to_string());
    }
    for (name, value) in request.headers() {
        builder = builder.header(name.as_str().to_string(), value.as_bytes().to_vec());
    }
    let sent = builder.build().map_err(|error| failure(format!("request: {error}")))?;
    let reply = pipe.call(sent).await.map_err(|error| failure(error.to_string()))?;
    let (status, metadata) = (reply.status, reply.metadata.clone());
    let payload = reply.collect_body().await.map_err(|error| failure(format!("response body: {error}")))?;
    let mut response = Response::builder().status(status);
    for (name, value) in &metadata {
        if let (Ok(name), Ok(value)) = (HeaderName::from_bytes(name), HeaderValue::from_bytes(value)) {
            response = response.header(name, value);
        }
    }
    response.body(payload).map_err(|error| failure(format!("response: {error}")))
}
