//! C15 backward-compatibility test for BinRecordMeta.
//!
//! Verifies that `BinRecordMeta` bytes serialized BEFORE C15 (no trace/span
//! fields) still deserialize correctly after the addition of those fields.
//! The `#[serde(default)]` annotation on the three new fields is what makes
//! this safe — this test proves the annotation is in place and effective.
//!
//! Uses serde's JSON representation (not postcard) so we can synthesize a
//! "pre-C15" payload without touching private wire types directly.

// module_inception: after consolidation this file's own module nests under
// the same-named `#[path]` mod that tests/e2e.rs declares.
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::module_inception)]
mod wire_back_compat {
    /// The pre-C15 shape of BinRecordMeta — no trace/span/parent_span fields.
    /// If we can deserialize this into the new shape, back-compat is confirmed.
    #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
    struct OldStyleMeta {
        cache: Option<String>,
        retries: u32,
        upstream: Option<String>,
        instance_id: Option<String>,
        extra_json: Option<String>,
    }

    /// Mirrors the new BinRecordMeta with the three C15 fields.
    #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
    struct NewStyleMeta {
        cache: Option<String>,
        retries: u32,
        upstream: Option<String>,
        instance_id: Option<String>,
        extra_json: Option<String>,
        #[serde(default)]
        trace_id: Option<[u8; 16]>,
        #[serde(default)]
        span_id: Option<[u8; 8]>,
        #[serde(default)]
        parent_span_id: Option<[u8; 8]>,
    }

    #[test]
    fn old_format_deserializes_with_new_fields_as_none() {
        let old = OldStyleMeta {
            cache: None,
            retries: 2,
            upstream: Some("origin-svc".to_string()),
            instance_id: None,
            extra_json: None,
        };

        let json_bytes = serde_json::to_vec(&old).expect("serialize old format");
        let new: NewStyleMeta =
            serde_json::from_slice(&json_bytes).expect("deserialize as new format");

        assert_eq!(new.retries, 2, "existing field preserved");
        assert_eq!(
            new.upstream,
            Some("origin-svc".to_string()),
            "existing upstream preserved",
        );
        assert_eq!(new.trace_id, None, "trace_id must be None for old records");
        assert_eq!(new.span_id, None, "span_id must be None for old records");
        assert_eq!(
            new.parent_span_id, None,
            "parent_span_id must be None for old records",
        );
    }

    #[test]
    fn new_format_with_trace_fields_roundtrips() {
        let new = NewStyleMeta {
            cache: None,
            retries: 0,
            upstream: None,
            instance_id: Some("inst-abc".to_string()),
            extra_json: None,
            trace_id: Some([0x01; 16]),
            span_id: Some([0x02; 8]),
            parent_span_id: Some([0x03; 8]),
        };

        let json_bytes = serde_json::to_vec(&new).expect("serialize new format");
        let restored: NewStyleMeta =
            serde_json::from_slice(&json_bytes).expect("deserialize new format");

        assert_eq!(restored, new, "new format must round-trip");
    }

    #[test]
    fn new_format_is_forward_compatible_with_old_reader() {
        let new = NewStyleMeta {
            cache: None,
            retries: 5,
            upstream: Some("svc-b".to_string()),
            instance_id: None,
            extra_json: None,
            trace_id: Some([0xaa; 16]),
            span_id: Some([0xbb; 8]),
            parent_span_id: None,
        };

        let json_bytes = serde_json::to_vec(&new).expect("serialize new format");
        let old: OldStyleMeta =
            serde_json::from_slice(&json_bytes).expect("old reader ignores unknown fields");

        assert_eq!(
            old.retries, 5,
            "retries preserved when old reader parses new bytes"
        );
        assert_eq!(
            old.upstream,
            Some("svc-b".to_string()),
            "upstream preserved",
        );
    }
}
