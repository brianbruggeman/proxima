use proxima_gguf::parse_complete;
use proxima_model_interop::LoadedModel;
use proxima_primitives::pipe::Pipe;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub enum TrialRequest<'payload> {
    Text {
        prompt: &'payload str,
        expected_substring: &'payload str,
        max_tokens: usize,
    },
    Image {
        payload: &'payload [u8],
        format: &'payload str,
        decoded_descriptor: &'payload str,
        decoded_bytes: &'payload [u8],
    },
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn verify_invocation_evidence(record: &Value, request_sha256: &str) -> bool {
    let Some(proxima) = record.get("proxima") else {
        return false;
    };
    let Some(invocation) = proxima.get("invocation") else {
        return false;
    };
    let Some(result) = record.get("result") else {
        return false;
    };
    if invocation.get("request_sha256").and_then(Value::as_str) != Some(request_sha256)
        || invocation.get("call_id").and_then(Value::as_str).is_none()
    {
        return false;
    }
    match proxima.get("invocation_status").and_then(Value::as_str) {
        Some("completed") => {
            let Some(output_sha256) = invocation.get("output_sha256").and_then(Value::as_str)
            else {
                return false;
            };
            result.get("status").and_then(Value::as_str) == Some("completed")
                && result.get("output_utf8_sha256").and_then(Value::as_str) == Some(output_sha256)
                && result
                    .get("generated_ids")
                    .and_then(Value::as_array)
                    .is_some_and(|ids| !ids.is_empty())
        }
        Some("failed") => {
            invocation.get("output_sha256") == Some(&Value::Null)
                && result.get("status").and_then(Value::as_str) == Some("failed")
                && result.get("stage").and_then(Value::as_str) == Some("invocation")
        }
        _ => false,
    }
}

pub async fn run_trial(
    case: &str,
    checkpoint_repo: &str,
    checkpoint_revision: &str,
    config_bytes: &[u8],
    weight_bytes: &[u8],
    request: TrialRequest<'_>,
) -> Value {
    let parsed = parse_complete(weight_bytes).expect("synthetic GGUF fixture parses");
    let input = match &request {
        TrialRequest::Text { prompt, .. } => {
            let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
                .expect("synthetic GGUF fixture has tokenizer metadata");
            let token_ids = proxima_tokenizer::encode_with_bos_eos(
                prompt,
                &vocab,
                vocab
                    .add_bos_token()
                    .unwrap_or_else(|| vocab.bos_token_id().is_some()),
                vocab.add_eos_token().unwrap_or(false),
            )
            .expect("fixture prompt tokenizes");
            let token_bytes = serde_json::to_vec(&token_ids).expect("token IDs serialize");
            json!({
                "kind": "text",
                "prompt_utf8_sha256": sha256_hex(prompt.as_bytes()),
                "token_ids": token_ids,
                "token_ids_sha256": sha256_hex(&token_bytes),
            })
        }
        TrialRequest::Image {
            payload,
            format,
            decoded_descriptor,
            decoded_bytes,
        } => json!({
            "kind": "image",
            "payload_sha256": sha256_hex(payload),
            "format": format,
            "decoded_input_descriptor": decoded_descriptor,
            "decoded_input_sha256": sha256_hex(decoded_bytes),
        }),
    };
    let budget = match &request {
        TrialRequest::Text { max_tokens, .. } => Some(*max_tokens),
        TrialRequest::Image { .. } => None,
    };
    let request_payload = serde_json::to_vec(&json!({
        "case": case,
        "input": input,
        "max_tokens": budget,
    }))
    .expect("trial request serializes");
    let request_sha256 = sha256_hex(&request_payload);
    let invocation = json!({
        "call_id": format!("fixture-{}", &request_sha256[..16]),
        "request_sha256": request_sha256,
        "output_sha256": null,
    });
    let checkpoint = json!({
        "repo": checkpoint_repo,
        "revision": checkpoint_revision,
        "config_sha256": sha256_hex(config_bytes),
        "weights_sha256": sha256_hex(weight_bytes),
    });
    let loaded = LoadedModel::load(&parsed, weight_bytes);
    let (load_status, bind_status, invocation_status, invocation, result, semantic) = match loaded {
        Err(error) => (
            "failed",
            "not_attempted",
            "not_attempted",
            Value::Null,
            json!({"status":"failed","stage":"load","code":"model_load_failed","message":error.to_string()}),
            json!({"status":"unavailable","rubric":"fixture-output-v1","reason":"model load failed"}),
        ),
        Ok(model) => match request {
            TrialRequest::Text {
                prompt,
                expected_substring,
                max_tokens,
            } => match Pipe::call(&model, (prompt.to_owned(), max_tokens)).await {
                Ok((generated_ids, output, _stopped_by_eos)) => {
                    let output_sha256 = sha256_hex(output.as_bytes());
                    let mut completed_invocation = invocation;
                    completed_invocation["output_sha256"] = json!(output_sha256);
                    let coherent = output.contains(expected_substring);
                    (
                        "completed",
                        "completed",
                        "completed",
                        completed_invocation,
                        json!({"status":"completed","generated_ids":generated_ids,"output_utf8_sha256":output_sha256}),
                        json!({"status":if coherent {"coherent"} else {"incoherent"},"rubric":format!("contains:{expected_substring}"),"evidence_sha256":sha256_hex(output.as_bytes())}),
                    )
                }
                Err(error) => (
                    "completed",
                    "completed",
                    "failed",
                    invocation,
                    json!({"status":"failed","stage":"invocation","code":"model_call_failed","message":error.to_string()}),
                    json!({"status":"unavailable","rubric":"fixture-output-v1","reason":"model call failed"}),
                ),
            },
            TrialRequest::Image { .. } => (
                "completed",
                "completed",
                "failed",
                invocation,
                json!({"status":"failed","stage":"invocation","code":"unsupported_modality","message":"LoadedModel Pipe accepts (String, usize); image input is unsupported"}),
                json!({"status":"unavailable","rubric":"fixture-image-v1","reason":"image input is unsupported by LoadedModel Pipe"}),
            ),
        },
    };
    json!({
        "version": 1,
        "case": case,
        "checkpoint": checkpoint,
        "input": input,
        "proxima": {
            "load_status": load_status,
            "bind_status": bind_status,
            "invocation_status": invocation_status,
            "invocation": invocation,
        },
        "result": result,
        "semantic": semantic,
        "numeric": {"status":"unavailable","boundary":"fixture-output","reason":"no independent reference for synthetic checkpoint"},
    })
}
