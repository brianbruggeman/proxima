//! QPACK header compression per [RFC 9204].
//!
//! Three submodules:
//!
//! - [`integer`] — QPACK / HPACK integer encoding (RFC 7541 §5.1).
//!   Distinct from QUIC varints — uses a per-call prefix-length N
//!   ranging from 1..=8 bits.
//! - [`static_table`] — RFC 9204 Appendix A 99-entry static table.
//!   Lookup by index (decoder) + linear scan by (name, value) pair
//!   (encoder hint). Pure const data; tier-3.
//! - [`encoder`] — full encoded-field-section emit with the static
//!   table (dynamic table support is gated on `alloc` per principle 3).
//! - [`decoder`] — split by surface: `FieldSink` + `decode_into` are
//!   tier-3 (borrowing engine, no heap); `DecodedField` +
//!   `decode_bounded` / `decode` are tier-1 (alloc convenience
//!   wrapper over the same engine).
//!
//! [RFC 9204]: https://www.rfc-editor.org/rfc/rfc9204
//!
//! # Tier
//!
//! `integer` + `static_table` are tier-3. `encoder` is tier-1 (alloc)
//! — it writes into a caller-owned `&mut Vec<u8>`. `decoder` is
//! PROMOTED to a split tier: its borrowing engine (`FieldSink`,
//! `decode_into`) builds under `--no-default-features --features
//! no-alloc`; only the owned-`Vec` convenience surface
//! (`DecodedField`, `decode_bounded`, `decode`) requires `alloc`. See
//! `decoder`'s module docs.

pub mod integer;
pub mod static_table;

pub mod decoder;
#[cfg(feature = "http3_codec-alloc")]
pub mod encoder;
/// Adapter proving `decoder`'s `FieldSink`/`decode_into` engine IS a
/// `proxima_primitives::pipe::part::PartSource` — see the module's own docs +
/// `docs/proxima-pipe/part-source-sink-design.md` step 1. Default-off
/// (feature `part-source`).
#[cfg(feature = "http3_codec-part-source")]
pub mod part_source;
