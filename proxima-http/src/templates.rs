//! Tiny `{{var}}` string-template expansion with path-param, query, and
//! time placeholders. Folded in from the former `proxima-templates`
//! crate — used by `crate::http1::client` and `crate::http1::upstream`
//! for dynamic header injection, so it rides `http1`/`http1-stream-client`.

use std::convert::Infallible;
use std::env;
use std::iter::Peekable;
use std::process;
use std::str::CharIndices;
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
    let Ok(output) = expand_with(input, |key| Ok::<_, Infallible>(resolve(key, context)));
    output
}

/// Like [`expand`], but a `{{body.<path>}}` whose path is absent from
/// `context.body` returns `Err(<trimmed placeholder key>)`.
#[must_use = "the Err names the first body path that did not resolve"]
pub fn expand_strict(input: &str, context: &TemplateContext<'_>) -> Result<String, String> {
    expand_with(input, |key| resolve_strict(key, context))
}

fn expand_with<E>(
    input: &str,
    resolver: impl FnMut(&str) -> Result<String, E>,
) -> Result<String, E> {
    if !input.contains("{{") {
        return Ok(input.to_string());
    }
    expand_placeholders(input, resolver)
}

fn expand_placeholders<E>(
    input: &str,
    mut resolver: impl FnMut(&str) -> Result<String, E>,
) -> Result<String, E> {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.char_indices().peekable();
    while let Some((index, character)) = chars.next() {
        if !opens_placeholder(character, &mut chars) {
            output.push(character);
            continue;
        }
        let Some(end) = find_close(&mut chars) else {
            output.push_str(&input[index..]);
            break;
        };
        output.push_str(&resolver(input[index + 2..end].trim())?);
    }
    Ok(output)
}

fn opens_placeholder(character: char, chars: &mut Peekable<CharIndices<'_>>) -> bool {
    let opens = character == '{' && chars.peek().map(|(_, next)| *next) == Some('{');
    if opens {
        chars.next();
    }
    opens
}

fn find_close(chars: &mut Peekable<CharIndices<'_>>) -> Option<usize> {
    while let Some((position, next)) = chars.next() {
        if next == '}' && chars.peek().map(|(_, peeked)| *peeked) == Some('}') {
            chars.next();
            return Some(position);
        }
    }
    None
}

fn resolve_strict(key: &str, context: &TemplateContext<'_>) -> Result<String, String> {
    match key.strip_prefix("body.") {
        Some(path) => lookup_body_path(context.body, path).ok_or_else(|| key.to_string()),
        None => Ok(resolve(key, context)),
    }
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
    lookup_body_path(root, path).unwrap_or_default()
}

fn lookup_body_path(root: Option<&Value>, path: &str) -> Option<String> {
    let mut current = root?;
    for segment in path.split('.') {
        current = match current {
            Value::Object(map) => map.get(segment)?,
            Value::Array(items) => items.get(segment.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(value_to_display(current))
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

    fn ticket_scope() -> Value {
        serde_json::json!({"in":{"ticket":"PROXIMA-24"},"ticket":{"repo":"brianbruggeman/proxima","pr":42}})
    }

    fn strict_matches(result: &Result<String, String>, expected: &Result<String, String>) -> bool {
        result == expected
    }

    #[test]
    fn expand_strict_resolves_single_body_path() {
        let scope = ticket_scope();
        let context = TemplateContext {
            body: Some(&scope),
            ..Default::default()
        };
        let outcome = expand_strict("{{body.in.ticket}}", &context);
        assert!(strict_matches(&outcome, &Ok("PROXIMA-24".to_string())));
    }

    #[test]
    fn expand_strict_resolves_paths_inside_a_url() {
        let scope = ticket_scope();
        let context = TemplateContext {
            body: Some(&scope),
            ..Default::default()
        };
        let outcome = expand_strict(
            "/repos/{{body.ticket.repo}}/pulls/{{body.ticket.pr}}",
            &context,
        );
        let expected = Ok("/repos/brianbruggeman/proxima/pulls/42".to_string());
        assert!(strict_matches(&outcome, &expected));
    }

    #[test]
    fn expand_strict_names_the_missing_body_path() {
        let scope = ticket_scope();
        let context = TemplateContext {
            body: Some(&scope),
            ..Default::default()
        };
        let outcome = expand_strict("{{body.ticket.missing}}", &context);
        assert!(strict_matches(
            &outcome,
            &Err("body.ticket.missing".to_string())
        ));
    }

    #[test]
    fn ctl_expand_matches_strict_on_present_paths() {
        let scope = ticket_scope();
        let context = TemplateContext {
            body: Some(&scope),
            ..Default::default()
        };
        let ticket = expand("{{body.in.ticket}}", &context);
        let url = expand(
            "/repos/{{body.ticket.repo}}/pulls/{{body.ticket.pr}}",
            &context,
        );
        assert_eq!(ticket, "PROXIMA-24");
        assert_eq!(url, "/repos/brianbruggeman/proxima/pulls/42");
    }

    #[test]
    fn ctl_expand_lenient_empty_is_rejected() {
        let scope = ticket_scope();
        let context = TemplateContext {
            body: Some(&scope),
            ..Default::default()
        };
        let lenient = Ok(expand("{{body.ticket.missing}}", &context));
        let expected = Err("body.ticket.missing".to_string());
        assert!(!strict_matches(&lenient, &expected));
    }
}
