//! Build-time constants -- the no_std+alloc floor's only configuration
//! surface (conflaguration's tier-2 pattern: constants ARE the config
//! below `std`). At `std`, `OnnxParserConfig` (std-only) seeds its
//! runtime default from this, never re-declaring it, so the two can never
//! drift apart (`defaults_track_the_sized_floor` in `config.rs` pins the
//! invariant).

/// Sanity cap on a length-delimited field's declared length -- not an ONNX
/// spec limit, just a guard against a corrupt file whose length prefix
/// claims an absurd size and would otherwise make the FSM buffer forever.
/// Comfortably above any real model's single embedded field (multi-gigabyte
/// `raw_data` tensors included). Policy, not format identity: at `std`,
/// `OnnxParserConfig::max_len_delimited_field` seeds from this and can be
/// overridden per-process.
pub const MAX_LEN_DELIMITED_FIELD: u64 = 1 << 40;
