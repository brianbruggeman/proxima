use serde_json::Value;

/// Default knobs the compressor shipped with before they were made
/// configurable. `CompressParams::default()` and (under `intercept-config`)
/// `CompressConfig::default()` both resolve to these — `config.rs` parity
/// test guards the drift.
pub const DEFAULT_DEDUP_MIN_LINE_LEN: usize = 30;
pub const DEFAULT_ENTROPY_BLOCK_SIZE: usize = 200;
pub const DEFAULT_ENTROPY_FLOOR: f64 = 2.5;

/// Plain (always-available) tuning surface for the compressor. The fluent
/// Rust API: construct directly or via `CompressParams::default()`. The
/// Settings-backed config surface lives in `config::CompressConfig` (behind
/// the `intercept-config` feature) and produces one of these.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressParams {
    /// Lines shorter than this are never deduplicated (kept verbatim).
    pub dedup_min_line_len: usize,
    /// Entropy pruning operates on blocks of this many chars.
    pub entropy_block_size: usize,
    /// Blocks scoring below this Shannon entropy (bits/byte) are dropped.
    pub entropy_floor: f64,
}

impl Default for CompressParams {
    fn default() -> Self {
        Self {
            dedup_min_line_len: DEFAULT_DEDUP_MIN_LINE_LEN,
            entropy_block_size: DEFAULT_ENTROPY_BLOCK_SIZE,
            entropy_floor: DEFAULT_ENTROPY_FLOOR,
        }
    }
}

fn shannon_entropy(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    let mut total = 0u32;
    for byte in text.bytes() {
        counts[byte as usize] += 1;
        total += 1;
    }
    let total_f = total as f64;
    counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let prob = count as f64 / total_f;
            -prob * prob.log2()
        })
        .sum()
}

fn entropy_prune(text: &str, block_size: usize, floor: f64) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= block_size {
        return text.to_string();
    }

    let mut kept = String::with_capacity(text.len());
    let mut offset = 0;

    while offset < chars.len() {
        let end = (offset + block_size).min(chars.len());
        let block: String = chars[offset..end].iter().collect();

        if end - offset < 60 || shannon_entropy(&block) >= floor {
            kept.push_str(&block);
        }

        offset = end;
    }

    kept
}

fn dedup_messages(messages: &[Value], min_line_len: usize) -> Vec<Value> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut result = Vec::with_capacity(messages.len());

    for message in messages {
        let Some(content) = message.get("content").and_then(Value::as_str) else {
            result.push(message.clone());
            continue;
        };

        let kept: Vec<&str> = content
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                if trimmed.len() < min_line_len {
                    return true;
                }
                seen.insert(trimmed.to_lowercase())
            })
            .collect();

        let deduped_content = kept.join("\n");
        if deduped_content.trim().is_empty() {
            continue;
        }

        let mut msg = message.clone();
        if let Some(obj) = msg.as_object_mut() {
            obj.insert("content".into(), Value::String(deduped_content));
        }
        result.push(msg);
    }

    result
}

pub fn compress_json_messages(body: &[u8]) -> Option<Vec<u8>> {
    compress_json_messages_with(body, &CompressParams::default())
}

pub fn compress_json_messages_with(body: &[u8], params: &CompressParams) -> Option<Vec<u8>> {
    let mut parsed: Value = serde_json::from_slice(body).ok()?;

    let compressed = if parsed.get("messages").is_some() {
        compress_messages_array(&mut parsed, "messages", params)
    } else if parsed.get("input").is_some() {
        compress_responses_api(&mut parsed, params)
    } else {
        false
    };

    if compressed {
        serde_json::to_vec(&parsed).ok()
    } else {
        None
    }
}

fn compress_messages_array(parsed: &mut Value, key: &str, params: &CompressParams) -> bool {
    let Some(messages) = parsed.get_mut(key).and_then(Value::as_array_mut) else {
        return false;
    };

    let deduped = dedup_messages(messages, params.dedup_min_line_len);
    let pruned: Vec<Value> = deduped
        .into_iter()
        .map(|mut msg| {
            prune_content_fields(&mut msg, params);
            msg
        })
        .collect();

    *messages = pruned;
    true
}

fn compress_responses_api(parsed: &mut Value, params: &CompressParams) -> bool {
    let mut changed = false;

    if let Some(instructions) = parsed.get("instructions").and_then(Value::as_str) {
        let pruned = entropy_prune(
            instructions,
            params.entropy_block_size,
            params.entropy_floor,
        );
        if pruned.len() < instructions.len()
            && let Some(obj) = parsed.as_object_mut()
        {
            obj.insert("instructions".into(), Value::String(pruned));
            changed = true;
        }
    }

    if let Some(input) = parsed.get_mut("input").and_then(Value::as_array_mut) {
        let deduped = dedup_messages(input, params.dedup_min_line_len);
        let pruned: Vec<Value> = deduped
            .into_iter()
            .map(|mut msg| {
                prune_content_fields(&mut msg, params);
                msg
            })
            .collect();
        *input = pruned;
        changed = true;
    }

    changed
}

fn prune_content_fields(msg: &mut Value, params: &CompressParams) {
    if let Some(content) = msg.get("content").and_then(Value::as_str) {
        let pruned = entropy_prune(content, params.entropy_block_size, params.entropy_floor);
        if let Some(obj) = msg.as_object_mut() {
            obj.insert("content".into(), Value::String(pruned));
        }
    }

    if let Some(content_arr) = msg.get_mut("content").and_then(Value::as_array_mut) {
        for item in content_arr.iter_mut() {
            if let Some(text) = item.get("text").and_then(Value::as_str) {
                let pruned = entropy_prune(text, params.entropy_block_size, params.entropy_floor);
                if let Some(obj) = item.as_object_mut() {
                    obj.insert("text".into(), Value::String(pruned));
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn compress_dedup_across_messages() {
        let body = serde_json::json!({
            "model": "model-a",
            "messages": [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "the weather is quite nice today in the downtown area\nwhat do you think"},
                {"role": "assistant", "content": "the weather is quite nice today in the downtown area\nyes it is great"},
            ]
        });
        let raw = serde_json::to_vec(&body).expect("serialize");
        let compressed = compress_json_messages(&raw).expect("compress");
        let result: Value = serde_json::from_slice(&compressed).expect("parse");
        let messages = result["messages"].as_array().expect("messages");

        let combined: String = messages
            .iter()
            .filter_map(|msg| msg["content"].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let count = combined
            .matches("the weather is quite nice today in the downtown area")
            .count();
        assert_eq!(
            count, 1,
            "duplicate line should appear once across messages"
        );
    }

    #[test]
    fn compress_preserves_non_message_fields() {
        let body = serde_json::json!({
            "model": "model-a",
            "temperature": 0.7,
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        });
        let raw = serde_json::to_vec(&body).expect("serialize");
        let compressed = compress_json_messages(&raw).expect("compress");
        let result: Value = serde_json::from_slice(&compressed).expect("parse");
        assert_eq!(result["model"], "model-a");
        assert_eq!(result["temperature"], 0.7);
    }

    #[test]
    fn compress_returns_none_for_non_json() {
        assert!(compress_json_messages(b"not json").is_none());
    }

    #[test]
    fn compress_returns_none_for_no_messages() {
        let body = serde_json::json!({"model": "model-a"});
        let raw = serde_json::to_vec(&body).expect("serialize");
        assert!(compress_json_messages(&raw).is_none());
    }

    #[test]
    fn shannon_entropy_zero_for_empty_string() {
        assert!((shannon_entropy("") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn shannon_entropy_zero_for_single_byte_run() {
        // entropy of "aaaa...a" = 0 (probability of a is 1, -1*log2(1)=0)
        assert!((shannon_entropy("aaaaaa") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn shannon_entropy_one_for_two_balanced_bytes() {
        // entropy of "ab" alternating = 1 bit/symbol exactly
        let entropy = shannon_entropy("abab");
        assert!((entropy - 1.0).abs() < 1e-9, "got {entropy}");
    }

    #[test]
    fn shannon_entropy_high_for_diverse_text() {
        // english prose tends to fall in the 3.5-5 bit/byte range
        let entropy = shannon_entropy("the quick brown fox jumps over the lazy dog");
        assert!(
            entropy > 3.5,
            "diverse english must score above 3.5 bits/byte, got {entropy}"
        );
        assert!(entropy < 5.0);
    }

    #[test]
    fn entropy_prune_passes_short_text_unchanged() {
        // text shorter than block_size returns unchanged
        let kept = entropy_prune("short", 200, 2.5);
        assert_eq!(kept, "short");
    }

    #[test]
    fn entropy_prune_keeps_high_entropy_blocks_drops_low() {
        // build a string with two blocks: one diverse (high entropy) and one
        // all-the-same (low entropy). prune at 2.5 bit floor should keep only
        // the diverse block.
        let diverse_block: String = (0..200)
            .map(|index| (b'a' + (index % 26) as u8) as char)
            .collect();
        let low_entropy_block: String = "x".repeat(200);
        let input = format!("{diverse_block}{low_entropy_block}");
        let kept = entropy_prune(&input, 200, 2.5);
        // the low-entropy x-block has entropy 0 (single symbol) and must drop
        assert!(
            kept.contains(&diverse_block[..50]),
            "diverse block must survive"
        );
        assert!(
            !kept.contains(&low_entropy_block[..]),
            "all-x block must drop"
        );
    }

    #[test]
    fn entropy_prune_always_keeps_short_tail_block_below_60() {
        // when the last block is < 60 chars long, the bypass keeps it regardless
        // of entropy (matches the existing `if end - offset < 60` check at the
        // block boundary).
        let body: String = "x".repeat(200);
        let tail = "abc"; // 3 chars, well below 60
        let input = format!("{body}{tail}");
        let kept = entropy_prune(&input, 200, 2.5);
        assert!(
            kept.ends_with(tail),
            "short tail must survive the floor check"
        );
    }

    #[test]
    fn dedup_messages_keeps_lines_below_min_line_len() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "ok"}),
            serde_json::json!({"role": "user", "content": "ok"}),
        ];
        let deduped = dedup_messages(&messages, 30);
        // each "ok" is 2 chars < min_line_len 30, so dedup must NOT drop them
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn dedup_messages_strips_duplicate_long_lines_case_insensitive() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "this is a long system instruction line that exceeds thirty characters"}),
            serde_json::json!({"role": "user", "content": "THIS IS A LONG SYSTEM INSTRUCTION LINE THAT EXCEEDS THIRTY CHARACTERS\nuser query"}),
        ];
        let deduped = dedup_messages(&messages, 30);
        // second message must lose the duplicate (case-insensitive) and keep "user query"
        let second_content = deduped[1]["content"].as_str().unwrap();
        assert!(second_content.contains("user query"));
        assert!(
            !second_content
                .to_lowercase()
                .contains("this is a long system instruction")
        );
    }

    #[test]
    fn dedup_messages_elides_message_whose_content_becomes_empty() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "shared shared shared shared shared shared content"}),
            serde_json::json!({"role": "user", "content": "shared shared shared shared shared shared content"}),
        ];
        let deduped = dedup_messages(&messages, 30);
        // second message becomes empty after dedup and must be elided entirely
        assert_eq!(deduped.len(), 1);
    }

    #[test]
    fn compress_returns_none_for_no_input_and_no_messages() {
        let body = serde_json::json!({"foo": "bar"});
        let raw = serde_json::to_vec(&body).expect("serialize");
        assert!(compress_json_messages(&raw).is_none());
    }

    #[test]
    fn compress_handles_responses_api_input_array_shape() {
        let body = serde_json::json!({
            "model": "model-mini",
            "instructions": "system prompt",
            "input": [
                {"role": "user", "content": "ping"},
            ],
        });
        let raw = serde_json::to_vec(&body).expect("serialize");
        let compressed = compress_json_messages(&raw).expect("compress");
        let result: Value = serde_json::from_slice(&compressed).expect("parse");
        assert_eq!(result["model"], "model-mini");
        assert!(result["input"].is_array());
    }

    #[test]
    fn compress_prunes_instructions_when_low_entropy_block_present() {
        // build instructions with a mix: a real prompt + a long low-entropy
        // block that will be pruned out
        let real_prompt = "you are a helpful assistant who answers concisely. ".repeat(8);
        let low_entropy = "x".repeat(200);
        let instructions = format!("{real_prompt}{low_entropy}");
        let body = serde_json::json!({
            "model": "model-mini",
            "instructions": instructions,
            "input": [{"role": "user", "content": "hi"}],
        });
        let raw = serde_json::to_vec(&body).expect("serialize");
        let compressed = compress_json_messages(&raw).expect("compress");
        let result: Value = serde_json::from_slice(&compressed).expect("parse");
        let pruned_instructions = result["instructions"].as_str().unwrap();
        assert!(
            pruned_instructions.len() < instructions.len(),
            "instructions must shrink when a low-entropy block is present"
        );
    }
}
