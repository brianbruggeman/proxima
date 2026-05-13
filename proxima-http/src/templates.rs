//! Tiny string-template expansion: `{{var}}` substitution with
//! path-param, query, and time placeholders. Folded in from the
//! former `proxima-templates` crate — used by [`crate::http1::client`]
//! and [`crate::http1::upstream`] for dynamic header injection.

use std::env;
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use xxhash_rust::xxh3::Xxh3;

#[derive(Debug, Default)]
pub struct TemplateContext<'request> {
    pub request_id: Option<&'request str>,
    pub trace_id: Option<&'request str>,
    pub pipe: Option<&'request str>,
    /// Parsed request body, used for `{{body.X.Y}}` lookups. None when the
    /// caller hasn't materialized the body or it isn't JSON-shaped.
    pub body: Option<&'request Value>,
}

#[must_use]
pub fn expand(input: &str, context: &TemplateContext<'_>) -> String {
    if !input.contains("{{") {
        return input.to_string();
    }
    let mut output = String::with_capacity(input.len());
    let mut chars = input.char_indices().peekable();
    while let Some((index, character)) = chars.next() {
        if character == '{' && chars.peek().map(|(_, next)| *next) == Some('{') {
            chars.next();
            let start = index + 2;
            let mut end = start;
            let mut found = false;
            while let Some((position, next)) = chars.next() {
                if next == '}' && chars.peek().map(|(_, peeked)| *peeked) == Some('}') {
                    chars.next();
                    end = position;
                    found = true;
                    break;
                }
            }
            if !found {
                output.push_str(&input[index..]);
                return output;
            }
            let key = input[start..end].trim();
            output.push_str(&resolve(key, context));
            continue;
        }
        output.push(character);
    }
    output
}

fn resolve(key: &str, context: &TemplateContext<'_>) -> String {
    if let Some(path) = key.strip_prefix("body.") {
        return resolve_body_path(context.body, path);
    }
    if key == "body" {
        return context.body.map(value_to_display).unwrap_or_default();
    }
    if let Some(value) = resolve_platform(key) {
        return value;
    }
    match key {
        "request.id" => context.request_id.unwrap_or("").to_string(),
        "request.trace_id" | "trace_id" => context.trace_id.unwrap_or("").to_string(),
        "pipe" | "request.pipe" => context.pipe.unwrap_or("").to_string(),
        other => format!("{{{{{other}}}}}"),
    }
}

fn resolve_platform(key: &str) -> Option<String> {
    if let Some(env_name) = key.strip_prefix("env.") {
        return Some(resolve_env(env_name));
    }
    match key {
        "uuid" => Some(generate_uuid_v4_like()),
        "timestamp" => Some(unix_seconds().to_string()),
        "timestamp.ms" => Some(unix_millis().to_string()),
        _ => None,
    }
}

fn resolve_env(name: &str) -> String {
    env::var(name).unwrap_or_default()
}

fn resolve_body_path(root: Option<&Value>, path: &str) -> String {
    let Some(mut current) = root else {
        return String::new();
    };
    for segment in path.split('.') {
        match current {
            Value::Object(map) => match map.get(segment) {
                Some(next) => current = next,
                None => return String::new(),
            },
            Value::Array(items) => match segment.parse::<usize>() {
                Ok(index) => match items.get(index) {
                    Some(next) => current = next,
                    None => return String::new(),
                },
                Err(_) => return String::new(),
            },
            _ => return String::new(),
        }
    }
    value_to_display(current)
}

fn value_to_display(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number.to_string(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or(0)
}

fn generate_uuid_v4_like() -> String {
    let mut hasher = Xxh3::new();
    hasher.update(&unix_millis().to_le_bytes());
    hasher.update(&process::id().to_le_bytes());
    let value = hasher.digest128();
    let high = (value >> 64) as u64;
    let low = value as u64;
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        (high >> 32) as u32,
        ((high >> 16) & 0xffff) as u16,
        (high & 0xffff) as u16,
        (low >> 48) as u16,
        low & 0xffff_ffff_ffff,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_without_braces() {
        let context = TemplateContext::default();
        assert_eq!(expand("plain text", &context), "plain text");
    }

    #[test]
    fn env_substitution() {
        // SAFETY: tests run single-threaded by default for env mutation; this test sets and reads
        unsafe {
            std::env::set_var("PROXIMA_TEMPLATE_TEST", "hello");
        }
        let context = TemplateContext::default();
        assert_eq!(
            expand("X={{env.PROXIMA_TEMPLATE_TEST}}", &context),
            "X=hello"
        );
        unsafe {
            std::env::remove_var("PROXIMA_TEMPLATE_TEST");
        }
    }

    #[test]
    fn missing_env_yields_empty() {
        let context = TemplateContext::default();
        assert_eq!(expand("X={{env.NOT_SET_VARIABLE_4242}}", &context), "X=");
    }

    #[test]
    fn request_id_substitution() {
        let context = TemplateContext {
            request_id: Some("req-42"),
            ..Default::default()
        };
        assert_eq!(expand("trace={{request.id}}", &context), "trace=req-42");
    }

    #[test]
    fn unknown_key_passes_through_braced() {
        let context = TemplateContext::default();
        assert_eq!(expand("X={{unknown}}", &context), "X={{unknown}}");
    }

    #[test]
    fn uuid_substitution_produces_uuid_shaped_string() {
        let context = TemplateContext::default();
        let expanded = expand("id={{uuid}}", &context);
        let id = expanded.strip_prefix("id=").expect("prefix");
        assert_eq!(id.len(), 36, "uuid should be 36 chars: {id}");
        assert_eq!(id.chars().filter(|character| *character == '-').count(), 4);
    }

    #[test]
    fn unmatched_open_brace_is_preserved() {
        let context = TemplateContext::default();
        let outcome = expand("trailing {{", &context);
        assert!(
            outcome.contains("{{"),
            "unmatched open should be preserved: {outcome}"
        );
    }

    #[test]
    fn timestamp_substitution_is_numeric() {
        let context = TemplateContext::default();
        let expanded = expand("ts={{timestamp}}", &context);
        let stamp = expanded.strip_prefix("ts=").expect("prefix");
        assert!(
            stamp.parse::<u64>().is_ok(),
            "timestamp should parse as u64: {stamp}"
        );
    }
}
