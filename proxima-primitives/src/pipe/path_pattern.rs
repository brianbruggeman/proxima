#![cfg(feature = "alloc")]

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct PathPattern {
    segments: Vec<Segment>,
    raw: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment {
    Literal(String),
    Param(String),
    Wildcard(String),
}

impl PathPattern {
    pub fn parse(pattern: &str) -> Self {
        let mut segments = Vec::new();
        for raw in pattern.trim_start_matches('/').split('/') {
            if raw.is_empty() {
                continue;
            }
            if let Some(name) = raw
                .strip_prefix('{')
                .and_then(|inner| inner.strip_suffix('}'))
            {
                if let Some(rest) = name.strip_prefix('*') {
                    segments.push(Segment::Wildcard(rest.to_string()));
                } else {
                    segments.push(Segment::Param(name.to_string()));
                }
                continue;
            }
            segments.push(Segment::Literal(raw.to_string()));
        }
        Self {
            segments,
            raw: pattern.to_string(),
        }
    }

    #[must_use]
    pub fn raw(&self) -> &str {
        &self.raw
    }

    #[must_use]
    pub fn matches(&self, path: &str) -> Option<BTreeMap<String, String>> {
        let mut params = BTreeMap::new();
        let segments_in: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|segment| !segment.is_empty())
            .collect();
        let mut input_index = 0usize;
        for (pattern_index, segment) in self.segments.iter().enumerate() {
            match segment {
                Segment::Literal(literal) => {
                    if input_index >= segments_in.len() || segments_in[input_index] != literal {
                        return None;
                    }
                    input_index += 1;
                }
                Segment::Param(name) => {
                    if input_index >= segments_in.len() {
                        return None;
                    }
                    params.insert(name.clone(), segments_in[input_index].to_string());
                    input_index += 1;
                }
                Segment::Wildcard(name) => {
                    let remainder = segments_in[input_index..].join("/");
                    let key = if name.is_empty() {
                        "wildcard".to_string()
                    } else {
                        name.clone()
                    };
                    params.insert(key, remainder);
                    input_index = segments_in.len();
                    if pattern_index != self.segments.len() - 1 {
                        return None;
                    }
                }
            }
        }
        if input_index != segments_in.len() {
            return None;
        }
        Some(params)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::root("/", "/", true)]
    #[case::literal_match("/users", "/users", true)]
    #[case::trailing_slash("/users", "/users/", true)]
    #[case::literal_mismatch("/users", "/posts", false)]
    #[case::extra_segment("/users", "/users/42", false)]
    fn literal_paths(#[case] pattern: &str, #[case] input: &str, #[case] expected: bool) {
        let parsed = PathPattern::parse(pattern);
        assert_eq!(parsed.matches(input).is_some(), expected);
    }

    #[test]
    fn param_segment_captures_value() {
        let pattern = PathPattern::parse("/users/{id}");
        let params = pattern.matches("/users/42").expect("pattern should match");
        assert_eq!(params.get("id"), Some(&"42".into()));
    }

    #[test]
    fn multiple_params_capture_all() {
        let pattern = PathPattern::parse("/users/{user}/posts/{post}");
        let params = pattern
            .matches("/users/alice/posts/12")
            .expect("pattern should match");
        assert_eq!(params.get("user"), Some(&"alice".into()));
        assert_eq!(params.get("post"), Some(&"12".into()));
    }

    #[test]
    fn named_wildcard_captures_remainder() {
        let pattern = PathPattern::parse("/files/{*rest}");
        let params = pattern
            .matches("/files/2026/05/03/notes.md")
            .expect("pattern should match");
        assert_eq!(params.get("rest"), Some(&"2026/05/03/notes.md".into()));
    }

    #[test]
    fn wildcard_only_at_tail() {
        let pattern = PathPattern::parse("/files/{*rest}/extra");
        assert!(pattern.matches("/files/anything/extra").is_none());
    }

    #[test]
    fn param_must_consume_all_input() {
        let pattern = PathPattern::parse("/users/{id}");
        assert!(pattern.matches("/users/42/extra").is_none());
    }
}
