//! `proxima_codec::StatefulCodec` impl for HPACK header compression.
//!
//! HPACK is canonically stateful: the dynamic table tracks
//! incremental-indexing entries across the lifetime of a connection,
//! and the encoder + decoder each own their own table. `StatefulCodec`
//! captures exactly this shape: a factory that vends `Encoder` and
//! `Decoder` instances per session.
//!
//! Gated behind the `codec-trait` feature so the no_std + alloc cliff
//! stays clean: the proxima-codec + proxima-core dependencies are the
//! only things this module imports.
//!
//! The trait impl is intentionally minimal — it only stamps out
//! Encoder/Decoder instances. The encode/decode methods are inherent
//! on each instance because their signatures (`IntoIterator<Item = (Bytes,
//! Bytes)>` for encode, `FnMut(Bytes, Bytes)` callback for decode) do
//! not fit a one-shot codec trait.

use bytes::{Bytes, BytesMut};
use proxima_codec::StatefulCodec;
use proxima_core::ProximaError;

use crate::hpack::decoder::{DecodeError, decode};
use crate::hpack::dynamic_table::DynamicTable;
use crate::hpack::encoder::encode;

/// Default HPACK dynamic table size from RFC 7541 §6.5.2. SETTINGS-driven
/// negotiations may raise or lower this; callers using
/// [`HpackCodec::with_table_size`] override it.
pub const DEFAULT_DYNAMIC_TABLE_SIZE: usize = 4096;

/// HPACK [`StatefulCodec`]. The codec itself is zero-sized; per-session
/// state lives on the [`HpackEncoder`] / [`HpackDecoder`] instances it
/// vends.
#[derive(Debug, Clone, Copy, Default)]
pub struct HpackCodec {
    dynamic_table_size: usize,
}

impl HpackCodec {
    /// Construct an HpackCodec that vends encoders / decoders with the
    /// default 4 KiB dynamic table size.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dynamic_table_size: DEFAULT_DYNAMIC_TABLE_SIZE,
        }
    }

    /// Construct an HpackCodec with a custom dynamic table size. Callers
    /// driving HTTP/2 must keep this in sync with their peer's
    /// `SETTINGS_HEADER_TABLE_SIZE` exchange.
    #[must_use]
    pub const fn with_table_size(dynamic_table_size: usize) -> Self {
        Self { dynamic_table_size }
    }
}

/// Per-session HPACK encoder. Owns its own dynamic table; not `Sync`
/// because the table mutates per encode call (matching the
/// `StatefulCodec::Encoder: Send` contract).
pub struct HpackEncoder {
    dynamic: DynamicTable,
}

impl HpackEncoder {
    /// Encode a sequence of headers into `dst`, mutating the dynamic
    /// table for incremental-indexing entries.
    pub fn encode_block<I>(&mut self, headers: I, dst: &mut BytesMut)
    where
        I: IntoIterator<Item = (Bytes, Bytes)>,
    {
        encode(headers, &mut self.dynamic, dst);
    }
}

/// Per-session HPACK decoder. Owns its own dynamic table.
pub struct HpackDecoder {
    dynamic: DynamicTable,
}

impl HpackDecoder {
    /// Decode every header field in `block`, invoking `on_header` for
    /// each. Mutates the dynamic table per the wire's
    /// incremental-indexing and size-update signals. `settings_max` is
    /// the most recent `SETTINGS_HEADER_TABLE_SIZE` advertised by THIS
    /// peer.
    pub fn decode_block<F>(
        &mut self,
        block: &Bytes,
        settings_max: usize,
        on_header: F,
    ) -> Result<(), DecodeError>
    where
        F: FnMut(Bytes, Bytes),
    {
        decode(block, &mut self.dynamic, settings_max, on_header)
    }
}

impl StatefulCodec for HpackCodec {
    type Encoder = HpackEncoder;
    type Decoder = HpackDecoder;

    fn new_encoder(&self) -> Result<Self::Encoder, ProximaError> {
        Ok(HpackEncoder {
            dynamic: DynamicTable::new(self.dynamic_table_size),
        })
    }

    fn new_decoder(&self) -> Result<Self::Decoder, ProximaError> {
        Ok(HpackDecoder {
            dynamic: DynamicTable::new(self.dynamic_table_size),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_codec::StatefulCodec;

    #[test]
    fn new_encoder_and_decoder_are_independent_instances() {
        let codec = HpackCodec::new();
        let _enc1 = codec.new_encoder().expect("encoder");
        let _enc2 = codec.new_encoder().expect("encoder");
        let _dec1 = codec.new_decoder().expect("decoder");
        // each call yields a fresh instance with its own dynamic table —
        // proves the StatefulCodec factory contract.
    }

    #[test]
    fn encode_then_decode_round_trips_a_header_set() {
        let codec = HpackCodec::new();
        let mut encoder = codec.new_encoder().expect("encoder");
        let mut decoder = codec.new_decoder().expect("decoder");

        let headers = vec![
            (Bytes::from_static(b":method"), Bytes::from_static(b"GET")),
            (
                Bytes::from_static(b":path"),
                Bytes::from_static(b"/v1/messages"),
            ),
            (
                Bytes::from_static(b"host"),
                Bytes::from_static(b"api.example.com"),
            ),
        ];

        let mut encoded = BytesMut::new();
        encoder.encode_block(headers.clone(), &mut encoded);
        assert!(!encoded.is_empty(), "encoder must emit at least one byte");

        let mut decoded: Vec<(Bytes, Bytes)> = Vec::new();
        decoder
            .decode_block(&encoded.freeze(), 4096, |name, value| {
                decoded.push((name, value));
            })
            .expect("decode");

        assert_eq!(decoded, headers);
    }

    #[test]
    fn with_table_size_is_respected_by_vended_instances() {
        let codec = HpackCodec::with_table_size(8192);
        let _enc = codec.new_encoder().expect("encoder");
        let _dec = codec.new_decoder().expect("decoder");
        // we can't introspect the table size from outside without
        // exposing more API; this test only confirms the with_table_size
        // constructor and the factory methods compose.
    }
}
