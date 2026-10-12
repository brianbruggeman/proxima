use core::ops::ControlFlow;
use std::sync::Arc;

use proxima_telemetry::emit::{EnvFilter, global};
use proxima_telemetry::export::set_default_recorder;
use proxima_telemetry::pipes::InMemoryPipe;
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::tag::{ScalarValue, Tag};

use super::chunked_prefill_tests::dense_checkpoint;
use super::draft_simple_tests::{armed_target, draft_simple_config, write_draft_checkpoint};
use super::{SpeculativeDecodeStats, TokenEvent};

const REQUEST_EVENT: &str =
    "draft_n_accepted: request-level speculative decode draft/accept totals";

fn request_fields(recorder: &Recorder, pipe: &InMemoryPipe) -> Vec<Vec<(&'static str, u64)>> {
    while recorder.drain() > 0 {}
    let records = pipe
        .logs()
        .iter()
        .filter(|log| log.body == proxima_telemetry::log::LogBody::Text(REQUEST_EVENT))
        .map(|log| {
            log.attrs
                .iter()
                .filter_map(|tag| match tag {
                    Tag::Scalar {
                        key,
                        value: ScalarValue::U64(value),
                    } => Some((*key, *value)),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect();
    pipe.clear();
    records
}

fn field(fields: &[(&str, u64)], name: &str) -> Option<u64> {
    fields
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| *value)
}

#[test]
fn a_draft_simple_request_reports_its_own_totals_apart_from_the_ngram_families() {
    let checkpoint = dense_checkpoint();
    let parsed = proxima_gguf::parse_complete(&checkpoint)
        .unwrap_or_else(|error| panic!("dense fixture parses: {error:?}"));
    let model = armed_target(&parsed, &checkpoint);
    let draft = write_draft_checkpoint(&checkpoint);
    let draft_path = draft
        .path()
        .to_str()
        .unwrap_or_else(|| panic!("draft path is UTF-8"));
    let config = draft_simple_config(draft_path);
    global::install(EnvFilter::parse(
        "proxima_model_interop::generate::decode=info",
    ));
    let pipe = InMemoryPipe::new();
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(pipe.clone())
            .core_count(1)
            .start()
            .unwrap_or_else(|error| panic!("request recorder starts: {error:?}")),
    );
    set_default_recorder(Arc::clone(&recorder));
    let mut stats = SpeculativeDecodeStats::default();
    let mut callback = |_: TokenEvent<'_>| ControlFlow::Continue(());

    model
        .generate_streaming_with_speculative_stats(
            "123",
            8,
            config,
            &mut callback,
            &mut stats,
            None,
        )
        .unwrap_or_else(|error| panic!("speculative generation: {error:?}"));

    let requests = request_fields(&recorder, &pipe);
    assert_eq!(requests.len(), 1, "one request record");
    assert_eq!(field(&requests[0], "draft_n"), Some(stats.drafted_total));
    assert_eq!(
        field(&requests[0], "draft_simple_drafted"),
        Some(stats.draft_simple.drafted)
    );
    assert_eq!(
        field(&requests[0], "draft_simple_accepted"),
        Some(stats.draft_simple.accepted)
    );
    assert!(stats.draft_simple.drafted > 0, "the draft proposed rows");
    assert_eq!(field(&requests[0], "ngram_simple_drafted"), Some(0));
}
