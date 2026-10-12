use core::ops::ControlFlow;
use std::sync::Arc;

use proxima_telemetry::emit::{EnvFilter, global};
use proxima_telemetry::export::set_default_recorder;
use proxima_telemetry::log::LogBody;
use proxima_telemetry::pipes::InMemoryPipe;
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::tag::{ScalarValue, Tag};

use super::chunked_prefill_tests::dense_checkpoint;
use super::draft_simple_tests::{
    armed_target, draft_simple_config, serving_config, write_draft_checkpoint,
};
use super::{LoadedModel, PrefixState, SpeculativeDecodeStats, TokenEvent};
use crate::serving::SpeculativeConfig;
use crate::{InteropError, ServingConfig};

const REQUEST_EVENT: &str =
    "draft_n_accepted: request-level speculative decode draft/accept totals";
const TOKEN_BUDGET: usize = 8;
const REQUEST_DRAFTED: u64 = 5;
const REQUEST_ACCEPTED: u64 = 5;

fn generate(
    api: &str,
    model: &LoadedModel<'_>,
    prefix: &PrefixState,
    config: &ServingConfig<'_>,
    budget: usize,
    stats: &mut SpeculativeDecodeStats,
) -> Result<Vec<u32>, InteropError> {
    let mut callback = |_: TokenEvent<'_>| ControlFlow::Continue(());
    let result = match api {
        "plain" => model.generate_with_serving_config("123", budget, *config),
        "streaming" => model.generate_streaming("123", budget, *config, &mut callback),
        "streaming_stats" => model.generate_streaming_with_speculative_stats(
            "123",
            budget,
            *config,
            &mut callback,
            stats,
            None,
        ),
        "prefix" => model.generate_from_prefix(prefix, "23", budget, config, &mut callback),
        "prefix_stats" => model.generate_from_prefix_with_speculative_stats(
            prefix,
            "23",
            budget,
            *config,
            &mut callback,
            stats,
            None,
        ),
        "ids" => model.generate_from_ids(&[49, 50, 51], budget, config, &mut callback),
        "ids_stats" => model.generate_from_ids_with_speculative_stats(
            &[49, 50, 51],
            budget,
            config,
            &mut callback,
            stats,
            None,
        ),
        "turns" => model.generate_from_ids_with_turn_ends(
            &[49, 50, 51],
            &[1],
            budget,
            config,
            &mut callback,
        ),
        _ => panic!("unknown public API fixture {api}"),
    };
    result.map(|(tokens, _, _)| tokens)
}

fn request_records(recorder: &Recorder, pipe: &InMemoryPipe) -> Vec<Vec<(&'static str, u64)>> {
    while recorder.drain() > 0 {}
    let records = pipe
        .logs()
        .iter()
        .filter(|log| log.body == LogBody::Text(REQUEST_EVENT))
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

fn assert_request_matches(fields: &[(&str, u64)], api: &str) {
    for (key, value) in [
        ("draft_n", REQUEST_DRAFTED),
        ("draft_n_accepted", REQUEST_ACCEPTED),
        ("draft_simple_drafted", REQUEST_DRAFTED),
        ("draft_simple_accepted", REQUEST_ACCEPTED),
        ("ngram_simple_drafted", 0),
        ("ngram_map_k_drafted", 0),
        ("ngram_map_k4v_drafted", 0),
        ("ngram_mod_drafted", 0),
        ("ngram_cache_drafted", 0),
    ] {
        assert_eq!(
            field(fields, key),
            Some(value),
            "{api}: request field {key}"
        );
    }
}

fn install_request_recorder() -> (Arc<Recorder>, InMemoryPipe) {
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
    (recorder, pipe)
}

fn check_public_api(api: &str, gpu_layers: i32) {
    let checkpoint = dense_checkpoint();
    let parsed = proxima_gguf::parse_complete(&checkpoint)
        .unwrap_or_else(|error| panic!("dense fixture parses: {error:?}"));
    let model = armed_target(&parsed, &checkpoint);
    let draft = write_draft_checkpoint(&checkpoint);
    let draft_path = draft
        .path()
        .to_str()
        .unwrap_or_else(|| panic!("draft path is UTF-8"));
    let mut config = draft_simple_config(draft_path);
    config.gpu_layers = gpu_layers;
    let mut plain_config = serving_config(SpeculativeConfig::none());
    plain_config.gpu_layers = gpu_layers;
    let prefix = model
        .prefill_prefix("1", &plain_config)
        .unwrap_or_else(|error| panic!("prefix prefills: {error:?}"));
    let (recorder, pipe) = install_request_recorder();
    let mut stats = SpeculativeDecodeStats::default();

    let plain = generate(
        api,
        &model,
        &prefix,
        &plain_config,
        TOKEN_BUDGET,
        &mut stats,
    )
    .unwrap_or_else(|error| panic!("{api} ordinary generation: {error:?}"));
    assert_eq!(
        plain.len(),
        TOKEN_BUDGET,
        "{api}: full generated token budget"
    );
    assert!(
        request_records(&recorder, &pipe).is_empty(),
        "{api}: speculation off emits no request"
    );

    let generated = generate(api, &model, &prefix, &config, TOKEN_BUDGET, &mut stats)
        .unwrap_or_else(|error| panic!("{api} speculative generation: {error:?}"));
    assert_eq!(generated, plain, "{api}: target token IDs");
    let records = request_records(&recorder, &pipe);
    assert_eq!(records.len(), 1, "{api}: one request record");
    assert_request_matches(&records[0], api);

    if api.ends_with("stats") {
        assert_eq!(
            stats.drafted_total, REQUEST_DRAFTED,
            "{api}: caller accumulator"
        );
        generate(api, &model, &prefix, &config, TOKEN_BUDGET, &mut stats)
            .unwrap_or_else(|error| panic!("{api} reused accumulator: {error:?}"));
        assert_eq!(
            stats.drafted_total,
            2 * REQUEST_DRAFTED,
            "{api}: the accumulator keeps growing across requests"
        );
        let records = request_records(&recorder, &pipe);
        assert_eq!(records.len(), 1, "{api}: reused stats emits once");
        assert_request_matches(&records[0], api);
    }
}

#[proxima::test]
#[case::generate_with_serving_config("plain")]
#[case::generate_streaming("streaming")]
#[case::generate_streaming_with_speculative_stats("streaming_stats")]
#[case::generate_from_prefix("prefix")]
#[case::generate_from_prefix_with_speculative_stats("prefix_stats")]
#[case::generate_from_ids("ids")]
#[case::generate_from_ids_with_speculative_stats("ids_stats")]
#[case::generate_from_ids_with_turn_ends("turns")]
async fn a_speculative_request_emits_one_request_event_from_every_public_api(#[case] api: &str) {
    check_public_api(api, 0);
}

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
#[proxima::test]
#[case::generate_with_serving_config("plain")]
#[case::generate_streaming("streaming")]
#[case::generate_streaming_with_speculative_stats("streaming_stats")]
#[case::generate_from_ids("ids")]
#[case::generate_from_ids_with_turn_ends("turns")]
async fn a_speculative_request_on_the_placed_metal_path_emits_one_request_event(#[case] api: &str) {
    check_public_api(api, crate::GPU_LAYERS_ALL);
}
