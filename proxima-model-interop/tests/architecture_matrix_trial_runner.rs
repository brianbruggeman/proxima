#![cfg(feature = "std")]

mod support;

use proxima_gguf::GgmlType;
use serde_json::Value;
use support::trial::{TrialRequest, run_trial, verify_invocation_evidence};

const FIXTURE_REVISION: &str = "0000000000000000000000000000000000000001";
const CONFIG_BYTES: &[u8] =
    b"{\"architecture\":\"llama\",\"embedding_length\":256,\"block_count\":2}";
const RED_PPM: &[u8] = b"P6\n1 1\n255\n\xff\x00\x00";
const RED_RGB: &[u8] = &[255, 0, 0];
const TEXT_WEIGHTS_SHA256: &str =
    "713330d3bb20ff3fe9bf58aea8dd34a3ac1e6a98d022e1d8adef901c735373cc";
const IMAGE_WEIGHTS_SHA256: &str =
    "431e17a390b0e0b41996d995d18e379043b7e4c19b1a3e4bbe48d904c3ec84c7";
const TEXT_REQUEST_SHA256: &str =
    "12cdb4f9705851e798c005831a9cc4691758a702bd7beac939254a3b53a2fff9";
const IMAGE_REQUEST_SHA256: &str =
    "74c5ccab36b8b76d4f18048760fc3d1eb51bc3ed8e8ce5a40e76bc9f462d2af5";
const TEXT_OUTPUT_SHA256: &str = "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d";

fn invocation_hash(record: &Value) -> &str {
    record["proxima"]["invocation"]["request_sha256"]
        .as_str()
        .expect("trial invocation records a request hash")
}

#[proxima::test]
async fn architecture_matrix_trial_runner() {
    let checkpoint = support::checkpoint_bytes(GgmlType::F32);
    let modal_checkpoint = support::checkpoint_bytes_moe(
        GgmlType::F32,
        support::EXPERT_COUNT,
        support::EXPERT_USED_COUNT,
    );
    let text = run_trial(
        "fixture_text",
        "fixture/synthetic-llama",
        FIXTURE_REVISION,
        CONFIG_BYTES,
        &checkpoint,
        TrialRequest::Text {
            prompt: "The capital of France is",
            expected_substring: "Paris",
            max_tokens: 1,
        },
    )
    .await;
    let image = run_trial(
        "fixture_image",
        "fixture/synthetic-llama",
        FIXTURE_REVISION,
        CONFIG_BYTES,
        &modal_checkpoint,
        TrialRequest::Image {
            payload: RED_PPM,
            format: "ppm-p6",
            decoded_descriptor: "1x1 RGB red pixel",
            decoded_bytes: RED_RGB,
        },
    )
    .await;

    assert_eq!(text["proxima"]["invocation_status"], "completed");
    assert_eq!(text["result"]["generated_ids"], serde_json::json!([0]));
    assert_eq!(text["semantic"]["status"], "incoherent");
    assert_eq!(image["proxima"]["invocation_status"], "failed");
    assert_eq!(image["result"]["code"], "unsupported_modality");
    assert_eq!(image["result"]["stage"], "invocation");
    assert_eq!(image["proxima"]["invocation"]["output_sha256"], Value::Null);
    assert_eq!(text["checkpoint"]["weights_sha256"], TEXT_WEIGHTS_SHA256);
    assert_eq!(image["checkpoint"]["weights_sha256"], IMAGE_WEIGHTS_SHA256);
    assert_eq!(invocation_hash(&text), TEXT_REQUEST_SHA256);
    assert_eq!(invocation_hash(&image), IMAGE_REQUEST_SHA256);
    assert_eq!(
        text["proxima"]["invocation"]["output_sha256"],
        TEXT_OUTPUT_SHA256
    );
    assert!(verify_invocation_evidence(&text, invocation_hash(&text)));
    assert!(verify_invocation_evidence(&image, invocation_hash(&image)));

    let mut fabricated = image.clone();
    fabricated["proxima"]["invocation_status"] = serde_json::json!("completed");
    assert!(!verify_invocation_evidence(
        &fabricated,
        invocation_hash(&image),
    ));

    println!("text_record={text}");
    println!("image_record={image}");
    println!("fixture_requests=2 invocation_records=2 false_invocations_rejected=1");
}
