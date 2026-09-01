//! Build-time constants -- the no_std+alloc floor's only configuration
//! surface (conflaguration's tier-2 pattern: constants ARE the config
//! below `std`; see the workspace `conflag` skill and
//! `proxima-gguf/src/sized.rs`, the pattern this mirrors).
//!
//! [`HEADER_LEN_BYTES`] is a format-identity fact (the spec's own
//! length-prefix width) and can never be runtime config at any tier, the
//! same treatment `proxima-gguf::sized::MAX_NAME_LEN` gets.
//! [`MAX_HEADER_BYTES`] is policy: at `std`,
//! [`crate::config::SafetensorsParserConfig`] seeds its runtime default
//! from this and [`crate::parser::SafetensorsParser::with_config`] can
//! override it per-process
//! (`defaults_track_the_sized_floor` in `config.rs` pins the invariant).

/// Width of the little-endian header-length prefix (`N` in the spec).
/// Format-identity, not policy -- the wire layout is fixed at 8 bytes by
/// the published safetensors spec, so there is no tier at which this
/// could vary.
pub const HEADER_LEN_BYTES: usize = 8;

/// DOS-prevention cap on the declared header length, matching the
/// reference `huggingface/safetensors` crate's own `MAX_HEADER_SIZE`
/// (`safetensors/src/tensor.rs` on `main`, checked 2026-08-18): "there's a
/// limit on the size of the header of 100MB to prevent parsing extremely
/// large JSON." Policy: at `std`,
/// `SafetensorsParserConfig::max_header_bytes` seeds from this and can be
/// lowered per-process (a caller with a known-small model corpus may want
/// a tighter cap than the reference implementation's own ceiling).
pub const MAX_HEADER_BYTES: u64 = 100_000_000;

/// `__metadata__` key [`crate::writer::write_complete`] always stamps and
/// [`crate::parser::Manifest::format_version`] reads back -- format-
/// identity, not policy, since a reader older than the writer must be able
/// to find it under a fixed name at any tier. See `crate::version` for the
/// accept/reject table.
pub const FORMAT_VERSION_KEY: &str = "proxima_format_version";

/// Bumped for a layout or semantic break this reader cannot interpret
/// without a migration -- e.g. a different `data_offsets` convention, or a
/// tensor directory field changing meaning. A file whose stamped major
/// exceeds this is rejected by
/// [`crate::parser::Manifest::format_version`] rather than silently
/// mis-read.
pub const FORMAT_VERSION_MAJOR: u16 = 1;

/// Bumped for an additive change a reader built against an older minor can
/// safely ignore -- e.g. a new advisory `__metadata__` key this crate
/// starts writing. Never gates acceptance: any minor at a supported major
/// is accepted regardless of whether it is newer than
/// [`FORMAT_VERSION_MINOR`].
pub const FORMAT_VERSION_MINOR: u16 = 0;
