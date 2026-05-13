//! Teaching surface for C1 — RFC 9000 §16 variable-length integers.
//!
//! Run with `cargo run --example varint_walkthrough -p proxima-quic-proto`.
//!
//! Prints the encoding of representative values across the four length
//! classes, with the 2-bit prefix called out and the remaining bits shown
//! big-endian. Read alongside RFC 9000 §16 + Appendix A.1.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use proxima_protocols::quic::varint;

fn main() {
    println!("RFC 9000 §16 variable-length integer encoding");
    println!();
    println!("The top 2 bits of the first byte select the length class:");
    println!("  0b00 -> 1 byte  (value <=                          63)");
    println!("  0b01 -> 2 bytes (value <=                      16,383)");
    println!("  0b10 -> 4 bytes (value <=               1,073,741,823)");
    println!("  0b11 -> 8 bytes (value <=   4,611,686,018,427,387,903)");
    println!();
    println!("The remaining bits hold the value, big-endian unsigned.");
    println!();

    // RFC 9000 §A.1 canonical vectors plus a couple of boundary values.
    let cases: &[(u64, &str)] = &[
        (37, "RFC §A.1 1-byte form"),
        (63, "boundary: last 1-byte"),
        (64, "boundary: first 2-byte"),
        (15_293, "RFC §A.1 2-byte form"),
        (16_383, "boundary: last 2-byte"),
        (16_384, "boundary: first 4-byte"),
        (494_878_333, "RFC §A.1 4-byte form"),
        (1_073_741_823, "boundary: last 4-byte"),
        (1_073_741_824, "boundary: first 8-byte"),
        (151_288_809_941_952_652, "RFC §A.1 8-byte form"),
        (varint::MAX_VALUE, "boundary: maximum value (2^62 - 1)"),
    ];

    for (value, label) in cases {
        let mut output = [0u8; varint::MAX_ENCODED_LEN];
        let written = varint::encode(*value, &mut output).expect("encode");
        let bytes = &output[..written];
        let prefix = bytes[0] >> 6;

        println!("value: {value:>22}   ({label})");
        println!(
            "  encoded ({} byte{}): {}",
            written,
            if written == 1 { "" } else { "s" },
            format_bytes(bytes),
        );
        println!(
            "  prefix tag: 0b{prefix:02b}   payload bits: {} bit{}",
            written * 8 - 2,
            if written * 8 - 2 == 1 { "" } else { "s" },
        );

        let (decoded, consumed) = varint::decode(bytes).expect("decode");
        assert_eq!(decoded, *value);
        assert_eq!(consumed, written);
        println!("  round-trip: OK");
        println!();
    }

    // demonstrate the long-form decode (RFC says decoders accept any valid
    // form; value 37 encoded as 2 bytes is legal even though the canonical
    // form is 1 byte)
    let long_form_37: [u8; 2] = [0x40, 0x25];
    let (value, consumed) = varint::decode(&long_form_37).expect("decode");
    println!("non-canonical long form (legal on the wire per RFC §16):");
    println!(
        "  input bytes: {}   decoded value: {value}   consumed: {consumed}",
        format_bytes(&long_form_37),
    );
}

fn format_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 5);
    for (index, byte) in bytes.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(&format!("0x{byte:02x}"));
    }
    out
}
