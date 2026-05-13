//! HKDF-Expand-Label per TLS 1.3 ([RFC 8446 §7.1]), used by QUIC for all
//! key-schedule derivations ([RFC 9001 §5]).
//!
//! ```text
//! HKDF-Expand-Label(Secret, Label, Context, Length) =
//!     HKDF-Expand(Secret, HkdfLabel, Length)
//!
//! struct {
//!     uint16 length = Length;                       // 2 bytes BE
//!     opaque label<7..255> = "tls13 " + Label;      // 1-byte len + bytes
//!     opaque context<0..255> = Context;             // 1-byte len + bytes
//! } HkdfLabel;
//! ```
//!
//! The "tls13 " prefix is applied to every label per TLS 1.3 conventions;
//! QUIC uses the same prefix because the QUIC key schedule sits directly
//! on the TLS 1.3 key schedule (RFC 9001 §5.1).
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). The `info` blob is constructed in a
//! caller-supplied stack array; the HKDF state is the `hkdf::Hkdf` type
//! which is stack-only.
//!
//! # Composability
//!
//! Composes [`hkdf::Hkdf`] over [`sha2::Sha256`]. Pure RustCrypto stack;
//! no FFI; portable across every Rust target including `thumbv7m-none-eabi`.
//!
//! [RFC 8446 §7.1]: https://www.rfc-editor.org/rfc/rfc8446#section-7.1
//! [RFC 9001 §5]: https://www.rfc-editor.org/rfc/rfc9001#section-5

use hkdf::Hkdf;
use sha2::Sha256;

/// The label prefix per TLS 1.3 §7.1.
const TLS13_PREFIX: &[u8] = b"tls13 ";

/// Output of [`crate::quic::crypto::initial_keys::derive`] — also the natural
/// upper bound for a single HKDF-Expand-Label call against SHA-256.
pub const SHA256_OUTPUT_LEN: usize = 32;

/// Maximum encoded length of an HkdfLabel info blob:
/// 2 (length) + 1 (label-len) + 6 ("tls13 ") + 255 (label) + 1 (ctx-len) + 255 (ctx).
const MAX_INFO_LEN: usize = 2 + 1 + TLS13_PREFIX.len() + 255 + 1 + 255;

/// Errors that the expand-label step can surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExpandError {
    /// Output buffer length exceeded HKDF-SHA256's limit (255 × HashLen = 8160 bytes).
    OutputTooLong,
    /// Label longer than 255 - 6 bytes (TLS 1.3 §7.1).
    LabelTooLong,
    /// Context longer than 255 bytes (TLS 1.3 §7.1).
    ContextTooLong,
}

/// Run `HKDF-Expand-Label(secret, label, context, output.len())` and write
/// the result into `output`.
///
/// # Errors
///
/// See [`ExpandError`].
pub fn expand_label_sha256(
    secret: &[u8; SHA256_OUTPUT_LEN],
    label: &[u8],
    context: &[u8],
    output: &mut [u8],
) -> Result<(), ExpandError> {
    let prk = Hkdf::<Sha256>::from_prk(secret).map_err(|_| ExpandError::OutputTooLong)?;
    expand_label_inner(&prk, label, context, output)
}

/// Variant of [`expand_label_sha256`] that takes an already-constructed
/// [`Hkdf`] PRK — useful when the same secret is expanded into multiple
/// labels (avoids re-running HKDF-Extract for each one).
pub fn expand_label_from_prk(
    prk: &Hkdf<Sha256>,
    label: &[u8],
    context: &[u8],
    output: &mut [u8],
) -> Result<(), ExpandError> {
    expand_label_inner(prk, label, context, output)
}

fn expand_label_inner(
    prk: &Hkdf<Sha256>,
    label: &[u8],
    context: &[u8],
    output: &mut [u8],
) -> Result<(), ExpandError> {
    if label.len() > 255 - TLS13_PREFIX.len() {
        return Err(ExpandError::LabelTooLong);
    }
    if context.len() > 255 {
        return Err(ExpandError::ContextTooLong);
    }
    if output.len() > u16::MAX as usize {
        return Err(ExpandError::OutputTooLong);
    }

    let mut info_buffer = [0u8; MAX_INFO_LEN];
    let info_len = encode_hkdf_label(&mut info_buffer, output.len() as u16, label, context);
    prk.expand(&info_buffer[..info_len], output)
        .map_err(|_| ExpandError::OutputTooLong)
}

/// Encode an HkdfLabel struct into `out`, returning the bytes written.
fn encode_hkdf_label(out: &mut [u8], length: u16, label: &[u8], context: &[u8]) -> usize {
    let mut cursor = 0;
    out[cursor..cursor + 2].copy_from_slice(&length.to_be_bytes());
    cursor += 2;
    out[cursor] = (TLS13_PREFIX.len() + label.len()) as u8;
    cursor += 1;
    out[cursor..cursor + TLS13_PREFIX.len()].copy_from_slice(TLS13_PREFIX);
    cursor += TLS13_PREFIX.len();
    out[cursor..cursor + label.len()].copy_from_slice(label);
    cursor += label.len();
    out[cursor] = context.len() as u8;
    cursor += 1;
    out[cursor..cursor + context.len()].copy_from_slice(context);
    cursor += context.len();
    cursor
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn hkdf_label_struct_shape_matches_rfc() {
        // expand a 32-byte output from a known PRK and label.
        // RFC 9001 §A.1 fixture: client_initial_secret comes from
        // expand_label(initial_secret, "client in", "", 32). We test the
        // info-blob encoding shape directly here; the end-to-end value is
        // validated in initial_keys::tests.
        let mut info = [0u8; MAX_INFO_LEN];
        let written = encode_hkdf_label(&mut info, 32, b"client in", b"");
        // expected layout:
        //   00 20         (length = 32, BE u16)
        //   0f            (label_len = 6 + 9 = 15)
        //   74 6c 73 31 33 20    ("tls13 ")
        //   63 6c 69 65 6e 74 20 69 6e   ("client in")
        //   00            (context_len = 0)
        let expected: &[u8] = &[
            0x00, 0x20, 0x0f, 0x74, 0x6c, 0x73, 0x31, 0x33, 0x20, 0x63, 0x6c, 0x69, 0x65, 0x6e,
            0x74, 0x20, 0x69, 0x6e, 0x00,
        ];
        assert_eq!(&info[..written], expected);
    }

    #[test]
    fn label_too_long_rejected() {
        let secret = [0u8; SHA256_OUTPUT_LEN];
        let label = [0xab; 250]; // 250 > 255 - 6
        let mut output = [0u8; 16];
        assert_eq!(
            expand_label_sha256(&secret, &label, b"", &mut output),
            Err(ExpandError::LabelTooLong),
        );
    }

    #[test]
    fn context_too_long_rejected() {
        let secret = [0u8; SHA256_OUTPUT_LEN];
        let context = [0xab; 256];
        let mut output = [0u8; 16];
        assert_eq!(
            expand_label_sha256(&secret, b"quic key", &context, &mut output),
            Err(ExpandError::ContextTooLong),
        );
    }

    #[test]
    fn empty_context_and_empty_label_round_trip() {
        let secret = [0xcd; SHA256_OUTPUT_LEN];
        let mut output_a = [0u8; 16];
        let mut output_b = [0u8; 16];
        expand_label_sha256(&secret, b"x", b"", &mut output_a).unwrap();
        expand_label_sha256(&secret, b"x", b"", &mut output_b).unwrap();
        assert_eq!(output_a, output_b, "deterministic output");
        assert_ne!(output_a, [0u8; 16], "non-zero output");
    }
}
