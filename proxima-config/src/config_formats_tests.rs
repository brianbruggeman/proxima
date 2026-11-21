#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(feature = "std")]
mod std_controls {
    use crate::default_config_format_registry;
    use serde_json::Value;

    const DOC_EXPECTED: &str = include_str!("../tests/fixtures/formats/doc.expected.json");
    const DOC_XML_EXPECTED: &str = include_str!("../tests/fixtures/formats/doc.xml.expected.json");

    fn expected(text: &str) -> Value {
        serde_json::from_str(text).expect("hand-written expected json")
    }

    fn parse_ext(ext: &str, raw: &str) -> Value {
        default_config_format_registry()
            .expect("registry")
            .get_by_extension(ext)
            .expect("format")
            .parse(raw)
            .expect("parse")
    }

    #[test]
    fn ctl_format_doc_toml() {
        let raw = include_str!("../tests/fixtures/formats/doc.toml");
        assert_eq!(parse_ext("toml", raw), expected(DOC_EXPECTED));
    }

    #[test]
    fn ctl_format_doc_json() {
        let raw = include_str!("../tests/fixtures/formats/doc.json");
        assert_eq!(parse_ext("json", raw), expected(DOC_EXPECTED));
    }

    #[test]
    fn ctl_format_doc_yaml() {
        let raw = include_str!("../tests/fixtures/formats/doc.yaml");
        assert_eq!(parse_ext("yaml", raw), expected(DOC_EXPECTED));
    }

    #[test]
    fn ctl_format_doc_json5() {
        let raw = include_str!("../tests/fixtures/formats/doc.json5");
        assert_eq!(parse_ext("json5", raw), expected(DOC_EXPECTED));
    }

    #[test]
    fn ctl_format_doc_ron() {
        let raw = include_str!("../tests/fixtures/formats/doc.ron");
        assert_eq!(parse_ext("ron", raw), expected(DOC_EXPECTED));
    }

    #[test]
    fn ctl_format_doc_xml() {
        let raw = include_str!("../tests/fixtures/formats/doc.xml");
        let tree = conflaguration::from_xml_str::<conflaguration::Tree>(raw).expect("xml tree");
        assert_eq!(Value::from(tree), expected(DOC_XML_EXPECTED));
    }

    #[test]
    fn ctl_format_diff_datetime() {
        let raw = include_str!("../tests/fixtures/formats/diff_datetime.toml");
        let want = serde_json::json!({"when": "1979-05-27T07:32:00Z"});
        assert_eq!(parse_ext("toml", raw), want);
    }

    #[test]
    fn ctl_format_diff_nonfinite() {
        let raw = include_str!("../tests/fixtures/formats/diff_nonfinite.toml");
        let want = serde_json::json!({"x": null, "y": null});
        assert_eq!(parse_ext("toml", raw), want);
    }

    #[test]
    fn ctl_format_reject_yaml() {
        let raw = include_str!("../tests/fixtures/formats/reject.yaml");
        let registry = default_config_format_registry().expect("registry");
        let err = registry
            .get_by_extension("yaml")
            .expect("format")
            .parse(raw);
        assert_eq!(
            err.unwrap_err().to_string(),
            "config: yaml: top-level must be a mapping or sequence"
        );
    }

    #[test]
    fn ctl_format_reject_xml() {
        let raw = include_str!("../tests/fixtures/formats/reject.xml");
        let registry = default_config_format_registry().expect("registry");
        let err = registry.get_by_extension("xml").expect("format").parse(raw);
        assert_eq!(
            err.unwrap_err().to_string(),
            "config: xml: top-level must be an element"
        );
    }

    #[test]
    fn ctl_format_unknown_ext() {
        let registry = default_config_format_registry().expect("registry");
        let err = registry.get_by_extension("cfg").err().expect("unknown ext");
        assert_eq!(
            err.to_string(),
            "registry: no config format for extension `.cfg`"
        );
    }
}

#[cfg(feature = "schema-std")]
mod schema_controls {
    use crate::schema::{FieldFlags, Schema, StructField};

    fn bool_seq(min_items: Option<usize>, max_items: Option<usize>) -> Schema {
        Schema::Seq {
            items: Box::new(Schema::Bool),
            min_items,
            max_items,
        }
    }

    fn flagged(optional: bool, deprecated: bool) -> Schema {
        Schema::Struct {
            name: "S".into(),
            fields: vec![StructField {
                name: "f".into(),
                schema: Schema::Bool,
                flags: FieldFlags {
                    optional,
                    deprecated,
                    description: None,
                },
            }],
        }
    }

    fn string_with(min_len: Option<usize>, max_len: Option<usize>) -> Schema {
        Schema::String {
            pattern: None,
            format: None,
            min_len,
            max_len,
        }
    }

    fn check(fixture: &str, want: Schema) {
        let got = toml::from_str::<Schema>(fixture).expect("schema toml");
        assert_eq!(got, want);
    }

    #[test]
    fn ctl_format_buffered_schema_int_min() {
        let fixture = include_str!("../tests/fixtures/schema_fields/int_min.toml");
        check(
            fixture,
            Schema::Int {
                min: Some(-5),
                max: None,
            },
        );
    }

    #[test]
    fn ctl_format_buffered_schema_int_max() {
        let fixture = include_str!("../tests/fixtures/schema_fields/int_max.toml");
        check(
            fixture,
            Schema::Int {
                min: None,
                max: Some(100),
            },
        );
    }

    #[test]
    fn ctl_format_buffered_schema_uint_min() {
        let fixture = include_str!("../tests/fixtures/schema_fields/uint_min.toml");
        check(
            fixture,
            Schema::UInt {
                min: Some(3),
                max: None,
            },
        );
    }

    #[test]
    fn ctl_format_buffered_schema_uint_max() {
        let fixture = include_str!("../tests/fixtures/schema_fields/uint_max.toml");
        check(
            fixture,
            Schema::UInt {
                min: None,
                max: Some(4_000_000_000),
            },
        );
    }

    #[test]
    fn ctl_format_buffered_schema_float_finite() {
        let fixture = include_str!("../tests/fixtures/schema_fields/float_finite.toml");
        check(fixture, Schema::Float { finite: true });
    }

    #[test]
    fn ctl_format_buffered_schema_string_min_len() {
        let fixture = include_str!("../tests/fixtures/schema_fields/string_min_len.toml");
        check(fixture, string_with(Some(2), None));
    }

    #[test]
    fn ctl_format_buffered_schema_string_max_len() {
        let fixture = include_str!("../tests/fixtures/schema_fields/string_max_len.toml");
        check(fixture, string_with(None, Some(64)));
    }

    #[test]
    fn ctl_format_buffered_schema_bytes_max_len() {
        let fixture = include_str!("../tests/fixtures/schema_fields/bytes_max_len.toml");
        check(
            fixture,
            Schema::Bytes {
                max_len: Some(1024),
            },
        );
    }

    #[test]
    fn ctl_format_buffered_schema_seq_min_items() {
        let fixture = include_str!("../tests/fixtures/schema_fields/seq_min_items.toml");
        check(fixture, bool_seq(Some(1), None));
    }

    #[test]
    fn ctl_format_buffered_schema_seq_max_items() {
        let fixture = include_str!("../tests/fixtures/schema_fields/seq_max_items.toml");
        check(fixture, bool_seq(None, Some(8)));
    }

    #[test]
    fn ctl_format_buffered_schema_flags_optional() {
        let fixture = include_str!("../tests/fixtures/schema_fields/flags_optional.toml");
        check(fixture, flagged(true, false));
    }

    #[test]
    fn ctl_format_buffered_schema_flags_deprecated() {
        let fixture = include_str!("../tests/fixtures/schema_fields/flags_deprecated.toml");
        check(fixture, flagged(false, true));
    }
}
