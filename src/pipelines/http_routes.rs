use std::future::Future;

use bytes::Bytes;
use futures::StreamExt;
use serde::Serialize;

use crate::body::ResponseStream;
use proxima_primitives::pipe::SendPipe;

use crate::error::ProximaError;
use crate::pipelines::control_plane::{
    DynPipelineControlPlane, EventFilter, ListFilter, PipelineSubmission,
};
use crate::pipelines::spec::{PipelineSpec, StageSpec};
use crate::recording::event::InteractionId;
use crate::recording::jsonl::encode_jsonl_line;
use crate::request::{Request, Response};
use std::collections::BTreeMap;

/// HTTP edge for `PipelineControlPlane`. Translates the daemon's
/// REST-shaped routes into trait calls. Mirrors the existing
/// `ControlPlanePipe` (control_plane.rs:217-345) so the daemon's HTTP
/// listener can mount this beside it.
///
/// Routes:
///
/// | method | path                                  | maps to              |
/// |--------|---------------------------------------|----------------------|
/// | POST   | /pipelines/submit                     | submit               |
/// | GET    | /pipelines                            | list                 |
/// | GET    | /pipelines/resolve?q=…                | resolve              |
/// | GET    | /pipelines/<id>                       | inspect              |
/// | GET    | /pipelines/<id>/tail                  | subscribe (chunked)  |
/// | GET    | /events                               | subscribe (chunked)  |
/// | GET    | /pipelines/<id>/explain?stage=…       | (501, G7)            |
/// | POST   | /pipelines/<id>/replay                | (501, G8)            |
/// | GET    | /pipelines/<id>/artifact?stage=&path= | (501, G9)            |
pub struct PipelineControlPlanePipe {
    plane: DynPipelineControlPlane,
    label: String,
}

impl PipelineControlPlanePipe {
    #[must_use]
    pub fn new(plane: DynPipelineControlPlane) -> Self {
        Self {
            plane,
            label: "pipeline_control_plane".into(),
        }
    }

    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
}

impl SendPipe for PipelineControlPlanePipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let plane = self.plane.clone();
        async move { route(plane, request).await }
    }
}


async fn route(
    plane: DynPipelineControlPlane,
    request: Request<Bytes>,
) -> Result<Response<Bytes>, ProximaError> {
    let method_string = String::from_utf8_lossy(request.method.as_bytes()).to_ascii_uppercase();
    let path_string = String::from_utf8_lossy(&request.path).into_owned();
    let method = method_string.as_str();
    let path = path_string.as_str();

    match (method, path) {
        ("POST", "/pipelines/submit") => handle_submit(plane, request).await,
        ("GET", "/pipelines") => handle_list(plane, request).await,
        ("GET", "/pipelines/resolve") => handle_resolve(plane, request).await,
        ("GET", "/events") => handle_events(plane).await,
        ("GET", path) if let Some(id) = strip_tail_suffix(path) => handle_tail(plane, id).await,
        ("GET", path) if let Some(id) = strip_explain_suffix(path) => {
            // pull `stage` out before the await so the future doesn't
            // hold `&Request` across the suspension point — `Request`
            // owns a `dyn Stream` (not Sync), and the returned future
            // must be Send.
            let stage = request.query.get_str("stage").map(str::to_string);
            handle_explain(plane, id, stage).await
        }
        ("POST", path) if let Some(id) = strip_replay_suffix(path) => {
            handle_replay(plane, id, request).await
        }
        ("GET", path) if let Some(id) = strip_artifact_suffix(path) => {
            let stage = request.query.get_str("stage").map(str::to_string);
            let artifact_path = request.query.get_str("path").map(str::to_string);
            handle_artifact(plane, id, stage, artifact_path).await
        }
        ("GET", path) if let Some(id) = strip_pipelines_id_prefix(path) => {
            handle_inspect(plane, id).await
        }
        _ => Ok(Response::not_found().with_body(format!(
            "unknown pipeline control-plane route: {method} {path}"
        ))),
    }
}

async fn handle_submit(
    plane: DynPipelineControlPlane,
    request: Request<Bytes>,
) -> Result<Response<Bytes>, ProximaError> {
    let content_type = request
        .metadata
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(b"content-type"))
        .map(|(_, value)| String::from_utf8_lossy(value).into_owned())
        .unwrap_or_default();
    let (_meta, body_bytes) = request.body_bytes().await?;
    if body_bytes.is_empty() {
        return Ok(Response::new(400)
            .with_body("submit requires a TOML or JSON body containing a PipelineSpec"));
    }
    let spec: PipelineSpec = if content_type.contains("application/json") {
        serde_json::from_slice(&body_bytes)
            .map_err(|err| ProximaError::Config(format!("submit body (json): {err}")))?
    } else {
        let text = std::str::from_utf8(&body_bytes)
            .map_err(|err| ProximaError::Config(format!("submit body (toml) not utf-8: {err}")))?;
        toml::from_str(text)
            .map_err(|err| ProximaError::Config(format!("submit body (toml): {err}")))?
    };
    let submission: PipelineSubmission = plane.submit(spec).await?;
    json_response(201, &submission)
}

async fn handle_list(
    plane: DynPipelineControlPlane,
    request: Request<Bytes>,
) -> Result<Response<Bytes>, ProximaError> {
    let filter = ListFilter {
        name: request.query.get_str("name").map(str::to_string),
        spec_hash_hex: request.query.get_str("spec_hash_hex").map(str::to_string),
    };
    let summaries = plane.list(filter).await?;
    json_response(200, &summaries)
}

async fn handle_resolve(
    plane: DynPipelineControlPlane,
    request: Request<Bytes>,
) -> Result<Response<Bytes>, ProximaError> {
    let query = request
        .query
        .get_str("q")
        .ok_or_else(|| ProximaError::Config("resolve requires query string `q`".into()))?;
    let id = plane.resolve(query).await?;
    json_response(200, &ResolveResponse { pipeline_id: id })
}

async fn handle_artifact(
    plane: DynPipelineControlPlane,
    id_text: &str,
    stage: Option<String>,
    artifact_path: Option<String>,
) -> Result<Response<Bytes>, ProximaError> {
    let id = parse_interaction_id(id_text)?;
    let stage = stage
        .ok_or_else(|| ProximaError::Config("artifact requires query string `stage`".into()))?;
    let relative = artifact_path
        .ok_or_else(|| ProximaError::Config("artifact requires query string `path`".into()))?;
    let resolved = match plane
        .artifact_path(id, &stage, std::path::Path::new(&relative))
        .await
    {
        Ok(path) => path,
        Err(ProximaError::NotFound(message)) => {
            return Ok(Response::not_found().with_body(message));
        }
        Err(ProximaError::Config(message)) => {
            return Ok(Response::new(400).with_body(message));
        }
        Err(other) => return Err(other),
    };
    let metadata = tokio::fs::metadata(&resolved).await.map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!(
            "stat artifact {resolved:?}: {err}"
        )))
    })?;
    if !metadata.is_file() {
        return Ok(
            Response::new(400).with_body(format!("artifact {resolved:?} is not a regular file"))
        );
    }
    // stream the file as chunked. a streamed body emits chunked because
    // there's no known Content-Length until the read completes.
    let file = tokio::fs::File::open(&resolved).await.map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!(
            "open artifact {resolved:?}: {err}"
        )))
    })?;
    let stream = tokio_util::io::ReaderStream::new(file).map(|item| {
        item.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("stream artifact: {err}")))
        })
    });
    Ok(Response::new(200)
        .with_header("content-type", "application/octet-stream")
        .with_stream(ResponseStream::new(stream)))
}

async fn handle_replay(
    plane: DynPipelineControlPlane,
    id_text: &str,
    request: Request<Bytes>,
) -> Result<Response<Bytes>, ProximaError> {
    let id = parse_interaction_id(id_text)?;
    let (_meta, body_bytes) = request.body_bytes().await?;
    let substitutes: BTreeMap<String, StageSpec> = if body_bytes.is_empty() {
        BTreeMap::new()
    } else {
        serde_json::from_slice(&body_bytes).map_err(|err| {
            ProximaError::Config(format!(
                "replay body must be a JSON object mapping stage-name to StageSpec: {err}"
            ))
        })?
    };
    match plane.replay(id, substitutes).await {
        Ok(submission) => json_response(201, &submission),
        Err(ProximaError::NotFound(message)) => Ok(Response::not_found().with_body(message)),
        Err(other) => Err(other),
    }
}

async fn handle_explain(
    plane: DynPipelineControlPlane,
    id_text: &str,
    stage: Option<String>,
) -> Result<Response<Bytes>, ProximaError> {
    let id = parse_interaction_id(id_text)?;
    let stage = stage
        .ok_or_else(|| ProximaError::Config("explain requires query string `stage`".into()))?;
    match plane.explain(id, &stage).await {
        Ok(chain) => json_response(200, &chain),
        Err(ProximaError::NotFound(message)) => Ok(Response::not_found().with_body(message)),
        Err(other) => Err(other),
    }
}

async fn handle_inspect(
    plane: DynPipelineControlPlane,
    id_text: &str,
) -> Result<Response<Bytes>, ProximaError> {
    let id = parse_interaction_id(id_text)?;
    match plane.inspect(id).await {
        Ok(record) => json_response(200, &record),
        Err(ProximaError::NotFound(message)) => Ok(Response::not_found().with_body(message)),
        Err(other) => Err(other),
    }
}

async fn handle_tail(
    plane: DynPipelineControlPlane,
    id_text: &str,
) -> Result<Response<Bytes>, ProximaError> {
    let id = parse_interaction_id(id_text)?;
    let event_stream = plane.subscribe_events(EventFilter::Pipeline(id)).await?;
    let chunk_stream = event_stream.map(|event| {
        let mut bytes = encode_jsonl_line(event)?;
        bytes.push(b'\n');
        Ok::<Bytes, ProximaError>(Bytes::from(bytes))
    });
    Ok(Response::new(200)
        .with_header("content-type", "application/x-ndjson")
        .with_stream(ResponseStream::new(chunk_stream)))
}

async fn handle_events(plane: DynPipelineControlPlane) -> Result<Response<Bytes>, ProximaError> {
    let event_stream = plane.subscribe_events(EventFilter::AllEvents).await?;
    let chunk_stream = event_stream.map(|event| {
        let mut bytes = encode_jsonl_line(event)?;
        bytes.push(b'\n');
        Ok::<Bytes, ProximaError>(Bytes::from(bytes))
    });
    Ok(Response::new(200)
        .with_header("content-type", "application/x-ndjson")
        .with_stream(ResponseStream::new(chunk_stream)))
}

fn parse_interaction_id(text: &str) -> Result<InteractionId, ProximaError> {
    let ulid: ulid::Ulid = text
        .parse()
        .map_err(|err| ProximaError::Config(format!("invalid pipeline id `{text}`: {err}")))?;
    Ok(InteractionId::from_ulid(ulid))
}

fn json_response<T: Serialize>(status: u16, value: &T) -> Result<Response<Bytes>, ProximaError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|err| ProximaError::Encode(format!("pipeline control plane json: {err}")))?;
    Ok(Response::new(status)
        .with_header("content-type", "application/json")
        .with_body(Bytes::from(bytes)))
}

#[derive(Debug, Serialize)]
struct ResolveResponse {
    pipeline_id: InteractionId,
}

fn strip_pipelines_id_prefix(path: &str) -> Option<&str> {
    let remainder = path.strip_prefix("/pipelines/")?;
    if remainder.is_empty() || remainder.contains('/') {
        return None;
    }
    Some(remainder)
}

fn strip_tail_suffix(path: &str) -> Option<&str> {
    let remainder = path.strip_prefix("/pipelines/")?;
    let prefix = remainder.strip_suffix("/tail")?;
    if prefix.is_empty() || prefix.contains('/') {
        return None;
    }
    Some(prefix)
}

fn strip_explain_suffix(path: &str) -> Option<&str> {
    let remainder = path.strip_prefix("/pipelines/")?;
    let prefix = remainder.strip_suffix("/explain")?;
    if prefix.is_empty() || prefix.contains('/') {
        return None;
    }
    Some(prefix)
}

fn strip_replay_suffix(path: &str) -> Option<&str> {
    let remainder = path.strip_prefix("/pipelines/")?;
    let prefix = remainder.strip_suffix("/replay")?;
    if prefix.is_empty() || prefix.contains('/') {
        return None;
    }
    Some(prefix)
}

fn strip_artifact_suffix(path: &str) -> Option<&str> {
    let remainder = path.strip_prefix("/pipelines/")?;
    let prefix = remainder.strip_suffix("/artifact")?;
    if prefix.is_empty() || prefix.contains('/') {
        return None;
    }
    Some(prefix)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipelines::control_plane::InMemoryPipelineControlPlane;
    use crate::pipelines::spec::{PipelineSpec, StageSpec};
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::Duration;

    fn shell_stage(name: &str, script: &str, deps: &[&str]) -> StageSpec {
        let (cmd, flag) = if cfg!(windows) {
            ("cmd", "/c")
        } else {
            ("/bin/sh", "-c")
        };
        StageSpec {
            name: name.into(),
            command: cmd.into(),
            args: vec![flag.into(), script.into()],
            env: BTreeMap::new(),
            cwd: None,
            depends_on: deps.iter().map(|raw| (*raw).into()).collect(),
        }
    }

    fn pipe() -> PipelineControlPlanePipe {
        let plane: DynPipelineControlPlane = Arc::new(InMemoryPipelineControlPlane::new());
        PipelineControlPlanePipe::new(plane)
    }

    async fn submit_json(pipe_ref: &PipelineControlPlanePipe, name: &str) -> InteractionId {
        let spec = PipelineSpec {
            name: Some(name.into()),
            stages: vec![shell_stage("only", "exit 0", &[])],
        };
        let body = serde_json::to_vec(&spec).expect("serialize spec");
        let request = Request::builder()
            .method("POST")
            .path("/pipelines/submit")
            .header("content-type", "application/json")
            .body(body)
            .build()
            .expect("build request");
        let response = SendPipe::call(pipe_ref, request).await.expect("call");
        assert_eq!(response.status, 201);
        let body_bytes = response.collect_body().await.expect("collect");
        let submission: PipelineSubmission =
            serde_json::from_slice(&body_bytes).expect("decode submission");
        submission.pipeline_id
    }

    async fn wait_for_terminal(pipe_ref: &PipelineControlPlanePipe, id: InteractionId) {
        for _ in 0..500 {
            let request = Request::builder()
                .method("GET")
                .path(format!("/pipelines/{id}"))
                .build()
                .expect("build inspect request");
            let response = SendPipe::call(pipe_ref, request).await.expect("call");
            if response.status == 200 {
                let body_bytes = response.collect_body().await.expect("collect");
                let value: serde_json::Value =
                    serde_json::from_slice(&body_bytes).expect("decode inspect");
                let status = value["summary"]["status"].as_str().unwrap_or("");
                if status != "running" {
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("pipeline did not reach terminal state within 5 seconds");
    }

    #[proxima::test]
    async fn submit_returns_201_with_pipeline_id() {
        let pipe_ref = pipe();
        let id = submit_json(&pipe_ref, "alpha").await;
        let _ = id;
    }

    #[proxima::test]
    async fn list_returns_submissions_newest_first() {
        let pipe_ref = pipe();
        let _first = submit_json(&pipe_ref, "alpha").await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        let second = submit_json(&pipe_ref, "beta").await;
        wait_for_terminal(&pipe_ref, second).await;
        let request = Request::builder()
            .method("GET")
            .path("/pipelines")
            .build()
            .expect("build");
        let response = SendPipe::call(&pipe_ref, request).await.expect("call");
        assert_eq!(response.status, 200);
        let body_bytes = response.collect_body().await.expect("collect");
        let summaries: Vec<serde_json::Value> =
            serde_json::from_slice(&body_bytes).expect("decode");
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0]["name"], "beta", "newest first");
    }

    #[proxima::test]
    async fn resolve_by_name_returns_canonical_id() {
        let pipe_ref = pipe();
        let id = submit_json(&pipe_ref, "named").await;
        // RequestBuilder::path stores the raw path bytes; the HTTP listener
        // parses ?query=... into request.query in production. Mirror that
        // here via query_param.
        let request = Request::builder()
            .method("GET")
            .path("/pipelines/resolve")
            .query_param("q", "named")
            .build()
            .expect("build");
        let response = SendPipe::call(&pipe_ref, request).await.expect("call");
        assert_eq!(response.status, 200);
        let body_bytes = response.collect_body().await.expect("collect");
        let value: serde_json::Value = serde_json::from_slice(&body_bytes).expect("decode");
        assert_eq!(value["pipeline_id"].as_str().unwrap(), id.to_string());
    }

    #[proxima::test]
    async fn inspect_returns_record_for_known_id() {
        let pipe_ref = pipe();
        let id = submit_json(&pipe_ref, "inspect-me").await;
        wait_for_terminal(&pipe_ref, id).await;
        let request = Request::builder()
            .method("GET")
            .path(format!("/pipelines/{id}"))
            .build()
            .expect("build");
        let response = SendPipe::call(&pipe_ref, request).await.expect("call");
        assert_eq!(response.status, 200);
    }

    #[proxima::test]
    async fn inspect_returns_404_for_unknown_id() {
        let pipe_ref = pipe();
        let bogus = InteractionId::new();
        let request = Request::builder()
            .method("GET")
            .path(format!("/pipelines/{bogus}"))
            .build()
            .expect("build");
        let response = SendPipe::call(&pipe_ref, request).await.expect("call");
        assert_eq!(response.status, 404);
    }

    #[proxima::test]
    async fn unknown_route_returns_404_with_body() {
        let pipe_ref = pipe();
        let request = Request::builder()
            .method("GET")
            .path("/nope")
            .build()
            .expect("build");
        let response = SendPipe::call(&pipe_ref, request).await.expect("call");
        assert_eq!(response.status, 404);
    }

    #[proxima::test]
    async fn artifact_on_in_memory_plane_returns_not_found() {
        // InMemoryPipelineControlPlane has no on-disk workspace, so
        // artifact_path's default impl returns NotFound → 404.
        let pipe_ref = pipe();
        let id = submit_json(&pipe_ref, "no-artifacts").await;
        let request = Request::builder()
            .method("GET")
            .path(format!("/pipelines/{id}/artifact"))
            .query_param("stage", "anything")
            .query_param("path", "file.txt")
            .build()
            .expect("build");
        let response = SendPipe::call(&pipe_ref, request).await.expect("call");
        assert_eq!(response.status, 404);
    }

    #[proxima::test]
    async fn explain_returns_chain_for_known_stage() {
        let pipe_ref = pipe();
        // submit a 2-stage pipeline so we have a `depends_on` edge
        let spec = PipelineSpec {
            name: Some("explain-chain".into()),
            stages: vec![
                shell_stage("fetch", "exit 0", &[]),
                shell_stage("build", "exit 0", &["fetch"]),
            ],
        };
        let body = serde_json::to_vec(&spec).expect("serialize spec");
        let submit_request = Request::builder()
            .method("POST")
            .path("/pipelines/submit")
            .header("content-type", "application/json")
            .body(body)
            .build()
            .expect("build");
        let submit_response = SendPipe::call(&pipe_ref, submit_request)
            .await
            .expect("call");
        let submission: PipelineSubmission =
            serde_json::from_slice(&submit_response.collect_body().await.expect("collect"))
                .expect("decode submission");
        let id = submission.pipeline_id;
        let explain_request = Request::builder()
            .method("GET")
            .path(format!("/pipelines/{id}/explain"))
            .query_param("stage", "build")
            .build()
            .expect("build");
        let response = SendPipe::call(&pipe_ref, explain_request)
            .await
            .expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("collect");
        let chain: Vec<serde_json::Value> = serde_json::from_slice(&body).expect("decode");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0]["stage"], "build");
        assert_eq!(chain[0]["depth"], 0);
        assert_eq!(chain[1]["stage"], "fetch");
        assert_eq!(chain[1]["depth"], 1);
    }

    #[proxima::test]
    async fn tail_returns_ndjson_stream() {
        let pipe_ref = pipe();
        let id = submit_json(&pipe_ref, "tail-me").await;
        wait_for_terminal(&pipe_ref, id).await;
        let request = Request::builder()
            .method("GET")
            .path(format!("/pipelines/{id}/tail"))
            .build()
            .expect("build");
        let response = SendPipe::call(&pipe_ref, request).await.expect("call");
        assert_eq!(response.status, 200);
        let content_type = response
            .metadata
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(b"content-type"))
            .map(|(_, value)| String::from_utf8_lossy(value).into_owned())
            .unwrap_or_default();
        assert_eq!(content_type, "application/x-ndjson");
        let body_bytes = response.collect_body().await.expect("collect");
        let text = String::from_utf8_lossy(&body_bytes);
        let lines: Vec<&str> = text.lines().filter(|line| !line.is_empty()).collect();
        assert!(
            !lines.is_empty(),
            "tail must return at least one event for a terminal pipeline"
        );
    }
}
