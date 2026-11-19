//! A `pipeline` key in a pipe spec builds a pipe from config: the loader
//! builds the named `pipes` through `build_pipe`, validates the stage DAG at
//! load, and serves the `output` stage. The stub pipe kinds below stand in for
//! the vault, github and llm leaves; they live here so `src/` adds no
//! `PipeFactory` impl.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde_json::Value;

const REVIEW_TICKET: &str = include_str!("fixtures/pipeline_pipe/review_ticket.json");

fn fixture() -> Value {
    serde_json::from_str(REVIEW_TICKET).expect("review_ticket fixture is valid json")
}

#[cfg(feature = "tokio")]
mod with_tokio {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bytes::Bytes;
    use proxima::pipe::into_handle;
    use proxima::pipelines::{PipelineSpec, topological_order};
    use proxima::request::{Request, Response};
    use proxima::{LoadContext, PipeFactory, PipeHandle, ProximaError, Spec, load};
    use proxima_primitives::pipe::SendPipe;
    use serde_json::{Value, json};

    use super::fixture;

    const STATE: &str = r#"{"repo":"brianbruggeman/proxima","pr":42}"#;
    const NOTES: &str = r#"["design.md"]"#;
    const PULL: &str = r#"{"title":"split stage body","body":"adds StageBody"}"#;
    const FILES: &str = r#"[{"filename":"src/pipelines/spec.rs"}]"#;
    const REVIEW: &str = r#"{"content":[{"type":"text","text":"LGTM with one nit"}]}"#;
    const PULL_PATH: &str = "/repos/brianbruggeman/proxima/pulls/42";

    #[derive(Clone)]
    struct StubPipe {
        routes: &'static [(&'static str, &'static str)],
        fail_path: Option<&'static str>,
        calls: Arc<AtomicUsize>,
    }

    impl StubPipe {
        fn respond(&self, path: &[u8]) -> Response<Bytes> {
            if self.fail_path.is_some_and(|fail| fail.as_bytes() == path) {
                return Response::new(500);
            }
            match self
                .routes
                .iter()
                .find(|(route, _)| route.as_bytes() == path)
            {
                Some((_, reply)) => Response::ok(Bytes::from_static(reply.as_bytes())),
                None => Response::new(404),
            }
        }
    }

    impl SendPipe for StubPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let response = self.respond(&request.path);
            async move { Ok(response) }
        }
    }

    struct StubFactory {
        kind: &'static str,
        stub: StubPipe,
    }

    impl PipeFactory for StubFactory {
        fn name(&self) -> &str {
            self.kind
        }

        fn build(
            &self,
            _spec: &Value,
            _inner: Option<PipeHandle>,
        ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
            Box::pin(async move { Ok(into_handle(self.stub.clone())) })
        }
    }

    struct Rig {
        context: LoadContext,
        llm_calls: Arc<AtomicUsize>,
    }

    fn register(context: &LoadContext, kind: &'static str, stub: StubPipe) {
        let factory = StubFactory { kind, stub };
        context
            .registry
            .register(Arc::new(factory))
            .expect("register stub kind");
    }

    fn stub(
        routes: &'static [(&'static str, &'static str)],
        fail_path: Option<&'static str>,
    ) -> StubPipe {
        StubPipe {
            routes,
            fail_path,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn rig(github_fail_path: Option<&'static str>) -> Rig {
        let context = LoadContext::with_default_registry().expect("default load context");
        let llm = stub(&[("/v1/messages", REVIEW)], None);
        let llm_calls = llm.calls.clone();
        register(
            &context,
            "vault",
            stub(
                &[
                    ("/PROXIMA-24/state.json", STATE),
                    ("/PROXIMA-24/notes", NOTES),
                ],
                None,
            ),
        );
        register(
            &context,
            "github",
            stub(
                &[
                    (PULL_PATH, PULL),
                    ("/repos/brianbruggeman/proxima/pulls/42/files", FILES),
                ],
                github_fail_path,
            ),
        );
        register(&context, "llm", llm);
        Rig { context, llm_calls }
    }

    fn stage_mut<'spec>(spec: &'spec mut Value, name: &str) -> &'spec mut Value {
        spec["pipeline"]["stages"]
            .as_array_mut()
            .and_then(|stages| stages.iter_mut().find(|stage| stage["name"] == name))
            .expect("fixture stage exists")
    }

    fn unknown_pipe() -> Value {
        let mut spec = fixture();
        stage_mut(&mut spec, "pr")["pipe"] = json!("gitlab");
        spec
    }

    fn unknown_dependency() -> Value {
        let mut spec = fixture();
        stage_mut(&mut spec, "review")["depends_on"]
            .as_array_mut()
            .expect("review depends_on")
            .push(json!("summary"));
        spec
    }

    fn cycle() -> Value {
        let mut spec = fixture();
        stage_mut(&mut spec, "pr")["depends_on"]
            .as_array_mut()
            .expect("pr depends_on")
            .push(json!("review"));
        spec
    }

    struct Expect {
        required: &'static [&'static str],
        absent: &'static [&'static str],
    }

    const UNKNOWN_PIPE: Expect = Expect {
        required: &["stage `pr`", "`gitlab`"],
        absent: &["`notes`"],
    };
    const UNKNOWN_DEPENDENCY: Expect = Expect {
        required: &["stage `review` depends on unknown stage `summary`"],
        absent: &[],
    };
    const CYCLE: Expect = Expect {
        required: &[
            "pipeline DAG has a cycle involving stages:",
            "\"pr\"",
            "\"review\"",
        ],
        absent: &["\"ticket\""],
    };

    fn check_refusal(error: &str, expect: &Expect) -> Result<(), String> {
        let missing = expect.required.iter().find(|text| !error.contains(**text));
        let present = expect.absent.iter().find(|text| error.contains(**text));
        match (missing, present) {
            (None, None) => Ok(()),
            (Some(text), _) => Err(format!("`{text}` missing from: {error}")),
            (None, Some(text)) => Err(format!("bystander `{text}` present in: {error}")),
        }
    }

    async fn load_error(rig: &Rig, spec: Value) -> String {
        match load(Spec::Inline(spec), &rig.context).await {
            Ok(_) => "loaded without error".to_string(),
            Err(error) => error.to_string(),
        }
    }

    fn incumbent_validate_error(spec: &Value) -> String {
        let parsed = parse_stages(spec);
        parsed
            .validate()
            .expect_err("incumbent validate refuses")
            .to_string()
    }

    fn incumbent_topological_error(spec: &Value) -> String {
        let parsed = parse_stages(spec);
        topological_order(&parsed)
            .expect_err("incumbent topo sort refuses")
            .to_string()
    }

    fn parse_stages(spec: &Value) -> PipelineSpec {
        serde_json::from_value(json!({ "stages": spec["pipeline"]["stages"] }))
            .expect("fixture stages parse")
    }

    async fn run_review(rig: &Rig) -> Response<Bytes> {
        let handle = load(Spec::Inline(fixture()), &rig.context)
            .await
            .expect("load review_ticket");
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::from_static(br#"{"ticket":"PROXIMA-24"}"#))
            .build()
            .expect("request");
        SendPipe::call(&handle, request)
            .await
            .expect("pipeline call")
    }

    #[proxima::test]
    async fn pipeline_pipe_failure_names_the_failing_stage_and_skips_the_llm() {
        let rig = rig(Some(PULL_PATH));

        let response = run_review(&rig).await;

        let status = response.status;
        let body =
            String::from_utf8(response.collect_body().await.expect("body").to_vec()).expect("utf8");
        assert_eq!(status, 502, "body: {body}");
        assert!(body.contains("stage `pr`"), "body: {body}");
        assert_eq!(rig.llm_calls.load(Ordering::SeqCst), 0);
    }

    #[proxima::test]
    async fn pipeline_pipe_load_refuses_a_stage_with_an_unknown_pipe() {
        let error = load_error(&rig(None), unknown_pipe()).await;

        assert_eq!(check_refusal(&error, &UNKNOWN_PIPE), Ok(()));
    }

    #[proxima::test]
    async fn pipeline_pipe_load_refuses_a_dependency_on_an_unknown_stage() {
        let error = load_error(&rig(None), unknown_dependency()).await;

        assert_eq!(check_refusal(&error, &UNKNOWN_DEPENDENCY), Ok(()));
    }

    #[proxima::test]
    async fn pipeline_pipe_load_refuses_a_cycle() {
        let error = load_error(&rig(None), cycle()).await;

        assert_eq!(check_refusal(&error, &CYCLE), Ok(()));
    }

    #[test]
    fn ctl_load_accepts_the_incumbent_validate_error_for_an_unknown_dependency() {
        let error = incumbent_validate_error(&unknown_dependency());

        assert_eq!(check_refusal(&error, &UNKNOWN_DEPENDENCY), Ok(()));
    }

    #[test]
    fn ctl_load_accepts_the_incumbent_topological_error_for_a_cycle() {
        let error = incumbent_topological_error(&cycle());

        assert_eq!(check_refusal(&error, &CYCLE), Ok(()));
    }

    #[test]
    fn ctl_load_rejects_an_error_that_names_nothing() {
        assert!(check_refusal("invalid pipeline", &UNKNOWN_PIPE).is_err());
    }

    #[test]
    fn ctl_load_rejects_an_error_that_also_names_the_bystander() {
        let error = "stage `pr` uses unknown pipe `gitlab`; check `notes` too";

        assert!(check_refusal(error, &UNKNOWN_PIPE).is_err());
    }

    #[test]
    fn ctl_load_accepts_an_error_that_names_exactly_the_offender() {
        let error = "stage `pr` uses unknown pipe `gitlab`";

        assert_eq!(check_refusal(error, &UNKNOWN_PIPE), Ok(()));
    }

    #[test]
    fn ctl_load_rejects_a_cycle_error_that_also_names_the_bystander() {
        let error = "pipeline DAG has a cycle involving stages: [\"pr\", \"review\", \"ticket\"]";

        assert!(check_refusal(error, &CYCLE).is_err());
    }
}

#[cfg(not(feature = "tokio"))]
mod without_tokio {
    use proxima::{LoadContext, Spec, load};

    use super::fixture;

    const REFUSAL: &[&str] = &["`pipeline` requires the `tokio` feature", "review_ticket"];

    fn check_tokio_refusal(error: &str) -> Result<(), String> {
        match REFUSAL.iter().find(|text| !error.contains(**text)) {
            None => Ok(()),
            Some(text) => Err(format!("`{text}` missing from: {error}")),
        }
    }

    #[proxima::test]
    async fn pipeline_pipe_refused_without_tokio() {
        let context = LoadContext::with_default_registry().expect("default load context");

        let error = load(Spec::Inline(fixture()), &context)
            .await
            .err()
            .expect("load refuses");

        assert_eq!(check_tokio_refusal(&error.to_string()), Ok(()));
        let names = context.registry.names();
        assert!(
            !names
                .iter()
                .any(|name| ["vault", "github", "llm"].contains(&name.as_str()))
        );
    }

    #[test]
    fn ctl_tokio_accepts_the_named_refusal() {
        let error = "`pipeline` requires the `tokio` feature (pipe `review_ticket`)";

        assert_eq!(check_tokio_refusal(error), Ok(()));
    }

    #[test]
    fn ctl_tokio_rejects_the_incumbent_fallthrough() {
        let error = "spec must include `upstreams`, `http`, `kv`, or a `type` discriminator";

        assert!(check_tokio_refusal(error).is_err());
    }
}
