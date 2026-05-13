//! HPACK header compression (RFC 7541).
//!
//! Layers, each in its own submodule:
//!
//! - [`integer`]: variable-length unsigned integer codec (§5.1).
//! - [`huffman`]: 256-symbol prefix code (Appendix B).
//! - [`static_table`]: 61 predefined (name, value) entries (Appendix A).
//! - `dynamic_table`: bounded LIFO with size-based eviction
//!   (§2.3.2 + §4) — requires a heap.
//! - `encoder` / `decoder`: orchestrate the layers above — require a heap.
//!
//! # Tier
//!
//! `--no-default-features --features hpack` builds tier-1 (`core::*` +
//! `alloc::*`, every module). `--no-default-features --features
//! hpack-no-alloc` builds tier-3 (`core::*` only) — exposes just
//! [`huffman`], [`integer`], and [`static_table`], none of which ever
//! allocate.

#[cfg(all(feature = "hpack-codec-trait", not(feature = "hpack-no-alloc")))]
pub mod codec_trait;
#[cfg(not(feature = "hpack-no-alloc"))]
pub mod decoder;
#[cfg(not(feature = "hpack-no-alloc"))]
pub mod dynamic_table;
#[cfg(not(feature = "hpack-no-alloc"))]
pub mod encoder;
pub mod huffman;
pub mod integer;
pub mod static_table;

#[cfg(all(feature = "hpack-codec-trait", not(feature = "hpack-no-alloc")))]
pub use codec_trait::{DEFAULT_DYNAMIC_TABLE_SIZE, HpackCodec, HpackDecoder, HpackEncoder};

#[cfg(not(feature = "hpack-no-alloc"))]
pub use decoder::{DecodeError, FieldSink, decode as decode_block, decode_into};
#[cfg(not(feature = "hpack-no-alloc"))]
pub use dynamic_table::{DynamicEntry, DynamicTable, ENTRY_OVERHEAD, STATIC_TABLE_LAST_INDEX};
#[cfg(not(feature = "hpack-no-alloc"))]
pub use encoder::encode as encode_block;
pub use huffman::{
    HuffmanError, decode as huffman_decode, encode as huffman_encode,
    encoded_len as huffman_encoded_len,
};
pub use integer::{HpackError, decode_integer, encode_integer};
pub use static_table::{
    STATIC_TABLE, entry as static_entry, lookup as static_lookup, lookup_name as static_lookup_name,
};
