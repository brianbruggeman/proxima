//! Emitters turn the `Schema` IR into format-specific contract documents.

pub mod json_schema;
pub mod openapi;
// the `toml` crate has no no_std support at all (it uses the std lib
// unconditionally throughout), so this emitter is std-only; json_schema and
// openapi stay no_std + alloc.
#[cfg(feature = "schema-std")]
pub mod toml_schema;

pub use json_schema::emit as emit_json_schema;
pub use openapi::emit as emit_openapi;
#[cfg(feature = "schema-std")]
pub use toml_schema::emit as emit_toml_schema;
