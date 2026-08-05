/// Failure modes of the sans-IO cryptographic core.
///
/// Every variant carries a fixed-size payload. There is no `String`, no
/// `Vec`, and no boxed source: this type has to construct on a target with
/// no allocator, and an error path that allocates is an error path that can
/// fail while reporting a failure. Context that is genuinely dynamic is
/// carried as `&'static str` (a pointer and a length, no heap) or as the
/// integers that describe the mismatch.
///
/// `thiserror` with `default-features = false` derives `core::error::Error`,
/// which is the same trait std re-exports as `std::error::Error` — so the
/// bare-metal tier and the std tier get one implementation, not two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CentauriError {
    /// The entropy source cannot produce bytes right now — a hardware TRNG
    /// failing its health test, a syscall-backed source refusing before the
    /// pool is initialised. A handshake cannot proceed and must not fall
    /// back to a weaker source.
    #[error("entropy unavailable: {0}")]
    EntropyUnavailable(&'static str),
    /// A finite entropy source ran out of material. Reaching this in
    /// production means the source was mis-sized; reaching it in a test
    /// means the scripted sequence was shorter than the number of draws the
    /// state machine makes.
    #[error("entropy exhausted: drew {drawn} of {available}")]
    EntropyExhausted { drawn: usize, available: usize },
    /// A caller-provided output buffer cannot hold what the state machine
    /// needs to write. The no-alloc analogue of a failed `Vec` growth: the
    /// caller sizes the buffer, so the caller is told what it should have
    /// been.
    #[error("buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall { needed: usize, available: usize },
    /// A wire message could not be parsed. The `&'static str` names which
    /// field, not what the bytes were — error text must never carry
    /// attacker-supplied or secret material.
    #[error("invalid message: {0}")]
    InvalidMessage(&'static str),
    /// A payload exceeded `[esp].max_payload_bytes`. The cap exists so a
    /// no-alloc deployment can size packet buffers statically; exceeding it is
    /// reported rather than truncated.
    #[error("payload too large: {len} bytes, max {max}")]
    PayloadTooLarge { len: usize, max: usize },
    /// Key agreement produced the all-zero shared secret, which every
    /// low-order X25519 point yields. The peer's DH value is degenerate —
    /// substituted by an attacker, or a broken implementation — and continuing
    /// would give an ephemeral that contributes nothing.
    #[error("degenerate key agreement")]
    DegenerateKeyAgreement,
    /// The SA's sequence space is exhausted. The sequence is the nonce, so
    /// continuing would repeat one; the SA must be rekeyed or torn down.
    #[error("sequence space exhausted")]
    SequenceExhausted,
    /// AEAD sealing failed. Only reachable on a buffer-shape violation the
    /// caller-side length checks should already have caught.
    #[error("encryption failed")]
    EncryptionFailed,
    /// An AEAD tag did not verify. Carries nothing on purpose: a decrypt
    /// failure must not tell an attacker which part was wrong.
    #[error("authentication failed")]
    AuthenticationFailed,
    /// A packet's sequence number was already seen or is older than the replay
    /// window.
    #[error("replay detected: seq {0}")]
    ReplayDetected(u64),
    /// A step was attempted that the current state does not permit.
    #[error("invalid transition: expected {expected}, found {found}")]
    InvalidTransition {
        expected: &'static str,
        found: &'static str,
    },
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use core::fmt::Write;

    use super::CentauriError;

    #[test]
    fn error_is_copy_and_stays_pointer_sized() {
        let error = CentauriError::BufferTooSmall {
            needed: 64,
            available: 32,
        };
        let copied = error;

        assert_eq!(error, copied, "Copy proves the type owns no heap");

        // the widest variant is two fat pointers plus a discriminant; this
        // bound is here to fail loudly if a variant ever starts carrying an
        // inline buffer, which is how a no-alloc error type gets fat.
        let widest_variant = 2 * size_of::<&'static str>() + size_of::<usize>();
        assert!(
            size_of::<CentauriError>() <= widest_variant,
            "error grew past two fat pointers: {}",
            size_of::<CentauriError>()
        );
    }

    #[test]
    fn the_error_trait_is_implemented_at_the_bare_tier_too() {
        // the hand-rolled impl this replaced was `#[cfg(feature = "std")]`, so
        // a no-alloc caller could not treat a failure as an error at all.
        fn requires_error_trait<Failure: core::error::Error>(_: &Failure) {}

        requires_error_trait(&CentauriError::AuthenticationFailed);
    }

    #[test]
    fn display_names_the_shortfall() {
        let mut buffer = crate::test_support::Buffer::new();
        let error = CentauriError::EntropyExhausted {
            drawn: 3,
            available: 2,
        };

        write!(buffer, "{error}").expect("the message fits the buffer");

        assert_eq!(buffer.as_str(), "entropy exhausted: drew 3 of 2");
    }
}
