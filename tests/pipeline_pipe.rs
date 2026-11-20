//! A `pipeline` key in a pipe spec builds a pipe from config: the loader
//! builds the named `pipes` through `build_pipe`, validates the stage DAG at
//! load, and serves the `output` stage. The stub pipe kinds below stand in for
//! the vault, github and llm leaves; they live here so `src/` adds no
//! `PipeFactory` impl.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde_json::Value;

const REVIEW_TICKET: &str = include_str!("fixtures/pipeline_pipe/review_ticket.json");
#[cfg(feature = "tokio")]
const TRIAGE: &str = include_str!("fixtures/pipeline_pipe/triage.json");

fn fixture() -> Value {
    serde_json::from_str(REVIEW_TICKET).expect("review_ticket fixture is valid json")
}

#[cfg(feature = "tokio")]
fn triage_fixture() -> Value {
    serde_json::from_str(TRIAGE).expect("triage fixture is valid json")
}

#[cfg(feature = "tokio")]
mod with_tokio {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bytes::Bytes;
    use proxima::pipe::into_handle;
    use proxima::pipelines::{PipelineSpec, topological_order};
    use proxima::request::{Request, Response};
    use proxima::{LoadContext, PipeFactory, PipeHandle, ProximaError, Spec, load};
    use proxima_primitives::pipe::SendPipe;
    use serde_json::{Value, json};

    use super::{fixture, triage_fixture};

    const STATE: &str = r#"{"repo":"brianbruggeman/proxima","pr":42}"#;
    const NOTES: &str = r#"["design.md"]"#;
    const PULL: &str = r#"{"title":"split stage body","body":"adds StageBody"}"#;
    const FILES: &str = r#"[{"filename":"src/pipelines/spec.rs"}]"#;
    const REVIEW: &str = r#"{"content":[{"type":"text","text":"LGTM with one nit"}]}"#;
    const PULL_PATH: &str = "/repos/brianbruggeman/proxima/pulls/42";
    const FILES_PATH: &str = "/repos/brianbruggeman/proxima/pulls/42/files";
    const REVIEW_BODY: &[u8] = br#"{"content":[{"text":"LGTM with one nit","type":"text"}]}"#;
    const LLM_REQUEST_PARTS: &[&str] = &["adds StageBody", "src/pipelines/spec.rs", "design.md"];

    #[derive(Clone)]
    struct StubPipe {
        routes: &'static [(&'static str, &'static str)],
        fail_path: Option<&'static str>,
        calls: Arc<AtomicUsize>,
        seen: Arc<OnceLock<Bytes>>,
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
            let _ = self.seen.set(request.payload.clone());
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
        vault: StubPipe,
        github: StubPipe,
        llm: StubPipe,
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
            seen: Arc::new(OnceLock::new()),
        }
    }

    fn rig(github_fail_path: Option<&'static str>) -> Rig {
        let context = LoadContext::with_default_registry().expect("default load context");
        let vault = stub(
            &[
                ("/PROXIMA-24/state.json", STATE),
                ("/PROXIMA-24/notes", NOTES),
            ],
            None,
        );
        let github = stub(&[(PULL_PATH, PULL), (FILES_PATH, FILES)], github_fail_path);
        let llm = stub(&[("/v1/messages", REVIEW)], None);
        register(&context, "vault", vault.clone());
        register(&context, "github", github.clone());
        register(&context, "llm", llm.clone());
        Rig {
            context,
            vault,
            github,
            llm,
        }
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

    #[derive(Debug, Default)]
    struct Run {
        status: u16,
        body: Bytes,
        vault: usize,
        github: usize,
        llm: usize,
        llm_request: String,
    }

    fn observe(rig: &Rig, status: u16, body: Bytes) -> Run {
        let llm_request = rig.llm.seen.get().map(|seen| seen.to_vec());
        Run {
            status,
            body,
            vault: rig.vault.calls.load(Ordering::SeqCst),
            github: rig.github.calls.load(Ordering::SeqCst),
            llm: rig.llm.calls.load(Ordering::SeqCst),
            llm_request: String::from_utf8_lossy(&llm_request.unwrap_or_default()).into_owned(),
        }
    }

    async fn run_review(rig: &Rig) -> Run {
        let handle = load(Spec::Inline(fixture()), &rig.context)
            .await
            .expect("load review_ticket");
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::from_static(br#"{"ticket":"PROXIMA-24"}"#))
            .build()
            .expect("request");
        let response = SendPipe::call(&handle, request)
            .await
            .expect("pipeline call");
        let status = response.status;
        let body = response.collect_body().await.expect("body");
        observe(rig, status, body)
    }

    async fn fetch(stub: &StubPipe, path: &str, body: &str) -> Bytes {
        let request = Request::builder()
            .method("GET")
            .path(path)
            .body(Bytes::from(body.to_string()))
            .build()
            .expect("request");
        let response = SendPipe::call(stub, request).await.expect("stub call");
        response.collect_body().await.expect("stub body")
    }

    async fn wired_llm_reply(rig: &Rig) -> Bytes {
        let state = fetch(&rig.vault, "/PROXIMA-24/state.json", "").await;
        let pull = fetch(&rig.github, PULL_PATH, "").await;
        let files = fetch(&rig.github, FILES_PATH, "").await;
        let notes = fetch(&rig.vault, "/PROXIMA-24/notes", "").await;
        assert_eq!(state, STATE.as_bytes());
        let prompt = format!(
            "pull request: {}\nfiles: {}\nnotes: {}",
            String::from_utf8_lossy(&pull),
            String::from_utf8_lossy(&files),
            String::from_utf8_lossy(&notes),
        );
        fetch(&rig.llm, "/v1/messages", &prompt).await
    }

    fn reserialize(raw: &[u8]) -> Bytes {
        let parsed: Value = serde_json::from_slice(raw).expect("llm reply is json");
        Bytes::from(serde_json::to_vec(&parsed).expect("serialize"))
    }

    fn check_counts(run: &Run) -> Result<(), String> {
        match (run.vault, run.github, run.llm) {
            (2, 2, 1) => Ok(()),
            (vault, github, llm) => Err(format!("calls vault {vault} github {github} llm {llm}")),
        }
    }

    fn check_response(run: &Run) -> Result<(), String> {
        if run.status != 200 {
            return Err(format!("status {}", run.status));
        }
        if run.body != REVIEW_BODY {
            return Err(format!("body {}", String::from_utf8_lossy(&run.body)));
        }
        let parsed: Value = serde_json::from_slice(&run.body).map_err(|error| error.to_string())?;
        let want = json!({"content":[{"type":"text","text":"LGTM with one nit"}]});
        (parsed == want)
            .then_some(())
            .ok_or_else(|| format!("parsed {parsed}"))
    }

    fn check_review(run: &Run) -> Result<(), String> {
        check_counts(run)?;
        check_response(run)?;
        match LLM_REQUEST_PARTS
            .iter()
            .find(|part| !run.llm_request.contains(**part))
        {
            None => Ok(()),
            Some(part) => Err(format!("`{part}` missing from: {}", run.llm_request)),
        }
    }

    fn check_failure(run: &Run) -> Result<(), String> {
        let body = String::from_utf8_lossy(&run.body);
        if run.status != 502 || !body.contains("stage `pr`") || run.llm != 0 {
            return Err(format!(
                "status {} llm {} body: {body}",
                run.status, run.llm
            ));
        }
        Ok(())
    }

    #[proxima::test]
    async fn pipeline_pipe_review_ticket() {
        let run = run_review(&rig(None)).await;

        assert_eq!(check_review(&run), Ok(()));
    }

    #[proxima::test]
    async fn ctl_review_accepts_a_hand_wired_sequence_with_the_parsed_llm_reply() {
        let rig = rig(None);
        let reply = wired_llm_reply(&rig).await;

        let run = observe(&rig, 200, reserialize(&reply));

        assert_eq!(check_review(&run), Ok(()));
    }

    #[proxima::test]
    async fn ctl_review_rejects_the_raw_unparsed_llm_bytes() {
        let rig = rig(None);
        let reply = wired_llm_reply(&rig).await;

        let run = observe(&rig, 200, reply);

        assert!(check_review(&run).is_err());
    }

    #[proxima::test]
    async fn ctl_review_rejects_a_run_where_the_llm_was_never_called() {
        let rig = rig(None);
        fetch(&rig.vault, "/PROXIMA-24/state.json", "").await;

        let run = observe(&rig, 200, Bytes::from_static(REVIEW_BODY));

        assert!(check_review(&run).is_err());
    }

    #[proxima::test]
    async fn pipeline_pipe_failure_names_the_failing_stage_and_skips_the_llm() {
        let run = run_review(&rig(Some(PULL_PATH))).await;

        assert_eq!(check_failure(&run), Ok(()));
    }

    #[test]
    fn ctl_failure_accepts_a_hand_built_502_naming_the_pr_stage() {
        let body = Bytes::from_static(b"stage `pr` failed: status 500");
        let run = Run {
            status: 502,
            body,
            ..Run::default()
        };

        assert_eq!(check_failure(&run), Ok(()));
    }

    #[test]
    fn ctl_failure_rejects_a_200_success_response() {
        let run = Run {
            status: 200,
            body: Bytes::from_static(REVIEW_BODY),
            ..Run::default()
        };

        assert!(check_failure(&run).is_err());
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

    const TICKET_WITH_PR: &str = r#"{"repo":"brianbruggeman/proxima","pr":42}"#;
    const TICKET_WITHOUT_PR: &str = r#"{"repo":"brianbruggeman/proxima","pr":null}"#;
    const TRIAGE_PULL: &str = r#"{"title":"split stage body","body":"adds StageBody"}"#;
    const TRIAGE_REVIEW: &str = r#"{"content":[{"type":"text","text":"LGTM with one nit"}]}"#;
    const TRIAGE_BRIEF: &str = r#"{"content":[{"type":"text","text":"no PR yet; ticket only"}]}"#;
    const REVIEW_SORTED: &[u8] = br#"{"content":[{"text":"LGTM with one nit","type":"text"}]}"#;
    const BRIEF_SORTED: &[u8] = br#"{"content":[{"text":"no PR yet; ticket only","type":"text"}]}"#;
    const NO_OUTPUT_RAN: &[u8] = b"no output stage ran: review, summary";
    const PR_NOT_FOUND: &[u8] = b"stage `pr` failed: stage `pr` pipe `github`: returned status 404";

    struct TriageRig {
        context: LoadContext,
        stubs: [StubPipe; 4],
    }

    #[derive(Debug)]
    struct TriageExpect {
        status: u16,
        body: &'static [u8],
        calls: [usize; 4],
    }

    fn triage_rig() -> TriageRig {
        let context = LoadContext::with_default_registry().expect("default load context");
        let tracker = stub(
            &[
                ("/tickets/PROXIMA-24", TICKET_WITH_PR),
                ("/tickets/PROXIMA-25", TICKET_WITHOUT_PR),
            ],
            None,
        );
        let github = stub(
            &[("/repos/brianbruggeman/proxima/pulls/42", TRIAGE_PULL)],
            None,
        );
        let llm = stub(&[("/v1/messages", TRIAGE_REVIEW)], None);
        let brief = stub(&[("/v1/summary", TRIAGE_BRIEF)], None);
        register(&context, "tracker", tracker.clone());
        register(&context, "github", github.clone());
        register(&context, "llm", llm.clone());
        register(&context, "brief", brief.clone());
        TriageRig {
            context,
            stubs: [tracker, github, llm, brief],
        }
    }

    fn triage_calls(rig: &TriageRig) -> [usize; 4] {
        rig.stubs
            .clone()
            .map(|stub| stub.calls.load(Ordering::SeqCst))
    }

    fn check_triage(
        response: (u16, &[u8]),
        calls: [usize; 4],
        expected: &TriageExpect,
    ) -> Result<(), String> {
        let (status, body) = response;
        if status != expected.status || body != expected.body {
            return Err(format!(
                "status {status} body {}; want {} {}",
                String::from_utf8_lossy(body),
                expected.status,
                String::from_utf8_lossy(expected.body)
            ));
        }
        (calls == expected.calls)
            .then_some(())
            .ok_or_else(|| format!("calls {calls:?}; want {:?}", expected.calls))
    }

    async fn call_triage(handle: &PipeHandle, ticket: &str) -> (u16, Bytes) {
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::from(format!(r#"{{"ticket":"{ticket}"}}"#)))
            .build()
            .expect("request");
        let response = SendPipe::call(handle, request)
            .await
            .expect("pipeline call");
        let status = response.status;
        (status, response.collect_body().await.expect("body"))
    }

    async fn load_triage(rig: &TriageRig, spec: Value) -> PipeHandle {
        load(Spec::Inline(spec), &rig.context)
            .await
            .expect("load triage")
    }

    async fn load_error_triage(rig: &TriageRig, spec: Value) -> String {
        match load(Spec::Inline(spec), &rig.context).await {
            Ok(_) => "loaded without error".to_string(),
            Err(error) => error.to_string(),
        }
    }

    async fn triage_once(spec: Value, ticket: &str) -> (u16, Bytes) {
        let rig = triage_rig();
        let handle = load_triage(&rig, spec).await;
        call_triage(&handle, ticket).await
    }

    fn without_when(spec: &mut Value, stage: &str) {
        stage_mut(spec, stage)
            .as_object_mut()
            .expect("stage is an object")
            .remove("when");
    }

    const BRANCH_PR: TriageExpect = TriageExpect {
        status: 200,
        body: REVIEW_SORTED,
        calls: [1, 1, 1, 0],
    };
    const BRANCH_SUMMARY: TriageExpect = TriageExpect {
        status: 200,
        body: BRIEF_SORTED,
        calls: [2, 1, 1, 1],
    };

    #[proxima::test]
    async fn pipeline_guard_refuses_an_unparseable_when_naming_the_stage() {
        let rig = triage_rig();
        let mut spec = triage_fixture();
        stage_mut(&mut spec, "pr")["when"] = json!("ticket.pr !=");

        let error = load_error_triage(&rig, spec).await;

        assert!(
            error.contains("stage `pr`") && error.contains("when"),
            "{error}"
        );
        assert_eq!(triage_calls(&rig), [0, 0, 0, 0]);
    }

    #[proxima::test]
    async fn pipeline_guard_refuses_an_output_naming_no_stage() {
        let rig = triage_rig();
        let mut spec = triage_fixture();
        spec["pipeline"]["output"] = json!(["review", "sumary"]);

        let error = load_error_triage(&rig, spec).await;

        assert!(error.contains("sumary"), "{error}");
        assert_eq!(triage_calls(&rig), [0, 0, 0, 0]);
    }

    #[proxima::test]
    async fn pipeline_guard_branch_one_pipe_takes_each_branch_on_its_own_input() {
        let rig = triage_rig();
        let handle = load_triage(&rig, triage_fixture()).await;

        let first = call_triage(&handle, "PROXIMA-24").await;
        assert_eq!(
            check_triage((first.0, &first.1), triage_calls(&rig), &BRANCH_PR),
            Ok(())
        );
        let second = call_triage(&handle, "PROXIMA-25").await;

        assert_eq!(
            check_triage((second.0, &second.1), triage_calls(&rig), &BRANCH_SUMMARY),
            Ok(())
        );
    }

    #[proxima::test]
    async fn pipeline_guard_output_answers_502_when_no_listed_stage_ran() {
        let mut spec = triage_fixture();
        stage_mut(&mut spec, "summary")["when"] = json!("false");

        let (status, body) = triage_once(spec, "PROXIMA-25").await;

        assert_eq!((status, body.as_ref()), (502, NO_OUTPUT_RAN));
    }

    #[proxima::test]
    async fn pipeline_guard_output_prefers_the_first_listed_stage_that_ran() {
        let mut spec = triage_fixture();
        without_when(&mut spec, "summary");

        let (status, body) = triage_once(spec, "PROXIMA-24").await;

        assert_eq!((status, body.as_ref()), (200, REVIEW_SORTED));
    }

    #[proxima::test]
    async fn pipeline_guard_output_follows_list_order_not_stage_order() {
        let mut spec = triage_fixture();
        without_when(&mut spec, "summary");
        spec["pipeline"]["output"] = json!(["summary", "review"]);

        let (status, body) = triage_once(spec, "PROXIMA-24").await;

        assert_eq!((status, body.as_ref()), (200, BRIEF_SORTED));
    }

    #[proxima::test]
    async fn ctl_branch_unguarded_triage_fails_the_second_call_and_the_check_notices() {
        let rig = triage_rig();
        let mut spec = triage_fixture();
        without_when(&mut spec, "pr");
        without_when(&mut spec, "summary");
        let handle = load_triage(&rig, spec).await;

        call_triage(&handle, "PROXIMA-24").await;
        let second = call_triage(&handle, "PROXIMA-25").await;

        let calls = triage_calls(&rig);
        let observed = TriageExpect {
            status: 502,
            body: PR_NOT_FOUND,
            calls: [2, 2, 1, 2],
        };
        assert_eq!(
            check_triage((second.0, &second.1), calls, &observed),
            Ok(())
        );
        assert!(check_triage((second.0, &second.1), calls, &BRANCH_SUMMARY).is_err());
    }

    #[test]
    fn ctl_branch_accepts_a_hand_built_summary_branch_response() {
        let response = (200, BRIEF_SORTED);

        assert_eq!(
            check_triage(response, [2, 1, 1, 1], &BRANCH_SUMMARY),
            Ok(())
        );
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
