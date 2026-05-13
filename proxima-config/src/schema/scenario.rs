//! `Schema` IR entries for [`crate::scenarios`] types. Built manually
//! and registered in `SchemaRegistry` so:
//!
//! 1. `proxima describe --format json-schema` can emit them for
//!    downstream binding generators (Python pydantic, TypeScript zod,
//!    Go structs).
//! 2. The smoke fixtures and any hand-written scenarios can be
//!    validated against the schema at CI time, catching drift between
//!    the serde structs in [`crate::scenarios::spec`] and the schema
//!    description here.
//!
//! Schema names registered:
//!
//! - `scenario`            — top-level [`Scenario`]
//! - `workload_spec`       — [`WorkloadSpec`]
//! - `expectation`         — [`Expectation`]
//! - `duration_spec`       — [`DurationSpec`]
//! - `profile_step`        — [`ProfileStep`]
//! - `orchestration_mode`  — [`OrchestrationMode`]
//! - `compare_op`          — [`CompareOp`]
//! - `scenario_pipe_spec`  — [`ScenarioPipeSpec`]
//!
//! Drift between this module and `scenarios::spec` is caught by the
//! round-trip test in `proxima/rust/tests/scenario_schema.rs`.
//!
//! [`Scenario`]: crate::scenarios::Scenario
//! [`WorkloadSpec`]: crate::scenarios::WorkloadSpec
//! [`Expectation`]: crate::scenarios::Expectation
//! [`DurationSpec`]: crate::scenarios::DurationSpec
//! [`ProfileStep`]: crate::scenarios::ProfileStep
//! [`OrchestrationMode`]: crate::scenarios::OrchestrationMode
//! [`CompareOp`]: crate::scenarios::CompareOp
//! [`ScenarioPipeSpec`]: crate::scenarios::ScenarioPipeSpec

use crate::schema::{EnumVariant, FieldFlags, Schema, SchemaRegistry, StructField};
use proxima_core::ProximaError;

/// Register every scenario-related schema in `registry`. Idempotent
/// across `LoadContext` clones because `SchemaRegistry::register`
/// returns an error on duplicate names — callers should swallow that
/// when bulk-registering.
pub fn register_scenario_schemas(registry: &SchemaRegistry) -> Result<(), ProximaError> {
    registry.register("duration_spec", duration_spec_schema())?;
    registry.register("profile_step", profile_step_schema())?;
    registry.register("compare_op", compare_op_schema())?;
    registry.register("orchestration_mode", orchestration_mode_schema())?;
    registry.register("expectation", expectation_schema())?;
    registry.register("workload_spec", workload_spec_schema())?;
    registry.register("scenario_pipe_spec", scenario_pipe_spec_schema())?;
    registry.register("scenario", scenario_schema())?;
    Ok(())
}

/// `DurationSpec` Deserialize accepts string forms (`"30s"` / `"5m"`)
/// AND the canonical `{secs: u64}` table — but Serialize emits only
/// the struct form, so the schema describes that shape. TOML callers
/// can still use the string form on input via the custom deserializer.
fn duration_spec_schema() -> Schema {
    Schema::Struct {
        name: "DurationSpec".into(),
        fields: vec![field(
            "secs",
            Schema::UInt {
                min: None,
                max: None,
            },
            "duration in whole seconds",
        )],
    }
}

fn profile_step_schema() -> Schema {
    Schema::Struct {
        name: "ProfileStep".into(),
        fields: vec![
            field(
                "rate",
                Schema::UInt {
                    min: Some(1),
                    max: None,
                },
                "requests-per-second target during this step",
            ),
            field(
                "duration",
                Schema::Ref("duration_spec".into()),
                "how long this step holds `rate`",
            ),
        ],
    }
}

fn compare_op_schema() -> Schema {
    Schema::Enum {
        name: "CompareOp".into(),
        variants: vec![unit_variant("eq"), unit_variant("ge"), unit_variant("le")],
    }
}

fn orchestration_mode_schema() -> Schema {
    Schema::Enum {
        name: "OrchestrationMode".into(),
        variants: vec![unit_variant("in_process"), unit_variant("isolated")],
    }
}

/// Expectation uses serde's internal-tagged form (`{ "kind": "...",
/// ...payload }`), but the Schema IR's `Enum` validates the
/// external-tagged form (`{ "variant": payload }`) only. Until the IR
/// grows internal-tagging support, the schema is modelled as a Struct
/// whose `kind` field constrains the discriminator and whose payload
/// fields are all optional. Cross-variant validation (rejecting
/// `ratio` when `kind=cel`) is lost; basic shape + binding generation
/// works.
fn expectation_schema() -> Schema {
    Schema::Struct {
        name: "Expectation".into(),
        fields: vec![
            field(
                "kind",
                Schema::Enum {
                    name: "ExpectationKind".into(),
                    variants: vec![
                        unit_variant("counter"),
                        unit_variant("histogram_p99_le_ms"),
                        unit_variant("success_rate_ge"),
                        unit_variant("cel"),
                        unit_variant("diff"),
                    ],
                },
                "discriminator: which kind of expectation",
            ),
            // Counter / HistogramP99LeMs shared fields:
            optional_field(
                "metric",
                Schema::String {
                    pattern: None,
                    format: None,
                    min_len: Some(1),
                    max_len: None,
                },
                "metric name (counter / histogram_p99_le_ms)",
            ),
            optional_field(
                "labels",
                labels_map_schema(),
                "optional label filter (counter / histogram_p99_le_ms)",
            ),
            // Counter:
            optional_field(
                "op",
                Schema::Ref("compare_op".into()),
                "comparison operator (counter only)",
            ),
            optional_field(
                "expected",
                Schema::UInt {
                    min: None,
                    max: None,
                },
                "expected counter value (counter only)",
            ),
            // HistogramP99LeMs:
            optional_field(
                "max_ms",
                Schema::Float { finite: true },
                "maximum allowed p99 in milliseconds (histogram_p99_le_ms only)",
            ),
            // SuccessRateGe:
            optional_field(
                "ratio",
                Schema::Float { finite: true },
                "minimum success ratio in [0, 1] (success_rate_ge only)",
            ),
            // Cel:
            optional_field(
                "expression",
                Schema::String {
                    pattern: None,
                    format: None,
                    min_len: Some(1),
                    max_len: None,
                },
                "CEL expression (cel only)",
            ),
            // Diff:
            optional_field(
                "identical",
                Schema::Bool,
                "require identical=true or false (diff only)",
            ),
            optional_field(
                "max_first_diff_offset",
                Schema::UInt {
                    min: None,
                    max: None,
                },
                "reserved (diff only)",
            ),
        ],
    }
}

fn workload_spec_schema() -> Schema {
    Schema::Struct {
        name: "WorkloadSpec".into(),
        fields: vec![
            field(
                "target_pipe",
                Schema::String {
                    pattern: None,
                    format: None,
                    min_len: Some(1),
                    max_len: None,
                },
                "name of the registered pipe to drive",
            ),
            optional_field(
                "method",
                Schema::String {
                    pattern: None,
                    format: None,
                    min_len: Some(1),
                    max_len: None,
                },
                "HTTP method; defaults to GET",
            ),
            optional_field(
                "path",
                Schema::String {
                    pattern: None,
                    format: None,
                    min_len: Some(1),
                    max_len: None,
                },
                "request path; defaults to /",
            ),
            optional_field(
                "headers",
                header_or_query_map_schema(),
                "per-request header overrides",
            ),
            optional_field(
                "query",
                header_or_query_map_schema(),
                "per-request query-string overrides",
            ),
            optional_field(
                "body",
                Schema::Optional(Box::new(Schema::String {
                    pattern: None,
                    format: None,
                    min_len: Some(1),
                    max_len: None,
                })),
                "request body as a utf-8 string; null when absent",
            ),
            optional_field(
                "requests",
                Schema::UInt {
                    min: None,
                    max: None,
                },
                "closed-loop total request count; >0 means closed-loop",
            ),
            optional_field(
                "concurrency",
                Schema::UInt {
                    min: Some(1),
                    max: None,
                },
                "in-flight cap; defaults to 1",
            ),
            optional_field(
                "target_rps",
                Schema::UInt {
                    min: Some(1),
                    max: None,
                },
                "open-loop requests-per-second target",
            ),
            optional_field(
                "duration",
                Schema::Ref("duration_spec".into()),
                "open-loop run duration (paired with target_rps)",
            ),
            optional_field(
                "profile",
                Schema::Seq {
                    items: Box::new(Schema::Ref("profile_step".into())),
                    min_items: None,
                    max_items: None,
                },
                "open-loop ramp/ladder profile; overrides (target_rps, duration) when non-empty",
            ),
        ],
    }
}

fn scenario_pipe_spec_schema() -> Schema {
    Schema::Struct {
        name: "ScenarioPipeSpec".into(),
        fields: vec![
            field(
                "name",
                Schema::String {
                    pattern: None,
                    format: None,
                    min_len: Some(1),
                    max_len: None,
                },
                "pipe name referenced from the workload's target_pipe",
            ),
            // `spec` is `#[serde(flatten)]` over an arbitrary Value;
            // schema describes it as Any so any factory's spec shape
            // validates. Per-factory schemas would land separately.
        ],
    }
}

fn scenario_schema() -> Schema {
    Schema::Struct {
        name: "Scenario".into(),
        fields: vec![
            optional_field(
                "mode",
                Schema::Ref("orchestration_mode".into()),
                "in_process (default) or isolated",
            ),
            optional_field(
                "pipe",
                Schema::Seq {
                    items: Box::new(Schema::Ref("scenario_pipe_spec".into())),
                    min_items: None,
                    max_items: None,
                },
                "pipe definitions referenced by the workload",
            ),
            field(
                "workload",
                Schema::Ref("workload_spec".into()),
                "what to drive and how",
            ),
            optional_field(
                "expect",
                Schema::Seq {
                    items: Box::new(Schema::Ref("expectation".into())),
                    min_items: None,
                    max_items: None,
                },
                "assertions evaluated against the final metrics snapshot",
            ),
        ],
    }
}

fn labels_map_schema() -> Schema {
    Schema::Map {
        keys: Box::new(Schema::String {
            pattern: None,
            format: None,
            min_len: Some(1),
            max_len: None,
        }),
        values: Box::new(Schema::String {
            pattern: None,
            format: None,
            min_len: None,
            max_len: None,
        }),
    }
}

fn header_or_query_map_schema() -> Schema {
    Schema::Map {
        keys: Box::new(Schema::String {
            pattern: None,
            format: None,
            min_len: Some(1),
            max_len: None,
        }),
        values: Box::new(Schema::String {
            pattern: None,
            format: None,
            min_len: None,
            max_len: None,
        }),
    }
}

fn field(name: &str, schema: Schema, description: &str) -> StructField {
    StructField {
        name: name.into(),
        schema,
        flags: FieldFlags {
            optional: false,
            deprecated: false,
            description: Some(description.into()),
        },
    }
}

fn optional_field(name: &str, schema: Schema, description: &str) -> StructField {
    StructField {
        name: name.into(),
        schema,
        flags: FieldFlags {
            optional: true,
            deprecated: false,
            description: Some(description.into()),
        },
    }
}

fn unit_variant(name: &str) -> EnumVariant {
    EnumVariant {
        name: name.into(),
        payload: None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::schema::{EmptyResolver, SchemaRegistry};

    #[test]
    fn register_scenario_schemas_registers_every_named_entry() {
        let registry = SchemaRegistry::new();
        register_scenario_schemas(&registry).expect("register");
        let mut names = registry.names();
        names.sort();
        let mut expected = vec![
            "compare_op",
            "duration_spec",
            "expectation",
            "orchestration_mode",
            "profile_step",
            "scenario",
            "scenario_pipe_spec",
            "workload_spec",
        ];
        expected.sort();
        assert_eq!(names, expected);
    }

    #[test]
    fn duration_spec_schema_accepts_canonical_struct_form() {
        let schema = duration_spec_schema();
        let resolver = EmptyResolver;
        let value = serde_json::json!({ "secs": 30 });
        schema
            .validate(&value, &resolver)
            .expect("canonical {secs: N} form must validate");
    }

    #[test]
    fn duration_spec_schema_rejects_string_form() {
        // Schema describes the post-Serialize wire form; string-form
        // is a TOML deserializer affordance, NOT in scope of the
        // emitted schema. Document the boundary.
        let schema = duration_spec_schema();
        let resolver = EmptyResolver;
        let value = serde_json::Value::String("30s".into());
        assert!(
            schema.validate(&value, &resolver).is_err(),
            "schema describes serialized struct form; bare strings should fail"
        );
    }

    #[test]
    fn workload_spec_schema_validates_minimal_open_loop_workload() {
        let registry = SchemaRegistry::new();
        register_scenario_schemas(&registry).expect("register");
        let schema = registry.get("workload_spec").expect("workload_spec");
        let value = serde_json::json!({
            "target_pipe": "echo",
            "target_rps": 100,
            "duration": { "secs": 5 },
            "concurrency": 16,
        });
        schema
            .validate(&value, &registry)
            .expect("minimal open-loop workload validates");
    }

    #[test]
    fn scenario_schema_validates_closed_loop_round_trip() {
        let registry = SchemaRegistry::new();
        register_scenario_schemas(&registry).expect("register");
        let schema = registry.get("scenario").expect("scenario");
        let value = serde_json::json!({
            "workload": {
                "target_pipe": "echo",
                "requests": 50,
                "concurrency": 4,
            },
            "expect": [
                { "kind": "success_rate_ge", "ratio": 0.99 }
            ],
        });
        schema
            .validate(&value, &registry)
            .expect("closed-loop scenario validates");
    }
}
