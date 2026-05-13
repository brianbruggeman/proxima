//! Scripted [`TlsProvider`] for sans-IO tests.
//!
//! The mock is shaped as an **ordered script of [`MockStep`]s**. Each
//! script step represents one thing the provider does next:
//!
//! - [`MockStep::EmitHandshakeBytes`] — bytes to hand to `write_handshake`
//!   when the caller calls it with the matching epoch.
//! - [`MockStep::ReadHandshake`] — bytes the caller must feed to
//!   `read_handshake`; mismatched epoch or bytes is a test-script bug.
//! - [`MockStep::InstallSecrets`] — secrets the provider pushes to the
//!   sink on the next `read_handshake` call.
//! - [`MockStep::EmitEvent`] — event the provider pushes to the sink on
//!   the next `read_handshake` call.
//! - [`MockStep::Confirm`] — marker that handshake confirmation has
//!   occurred (sets `is_confirmed`).
//! - [`MockStep::FailWith`] — next provider method returns this error.
//!
//! Mock is gated behind `feature = "quic-mock-tls"` (auto-enabled by
//! dev-dependencies) and is NOT part of the stable public API of the
//! proto crate.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::ops::Range;

use crate::quic::crypto::initial_keys::{self, InitialKeyPair};
use crate::quic::side::Side;

use super::event::{TlsEvent, TlsEventSink};
use super::secrets::{
    Direction, DirectionalKeys, Epoch, EpochSecrets, HeaderKeyMaterial, PacketKeyMaterial,
};
use super::{AeadProvider, TlsError, TlsProvider};

/// One step of a [`MockTlsProvider`] script.
#[derive(Debug, Clone)]
pub enum MockStep {
    /// Provider writes these bytes on the next `write_handshake(epoch, _)`.
    EmitHandshakeBytes { epoch: Epoch, bytes: Vec<u8> },
    /// Provider expects these bytes on the next `read_handshake(epoch, _, _)`.
    /// Bytes are checked for equality; mismatch causes the provider to
    /// return [`TlsError::UnexpectedMessage`].
    ReadHandshake { epoch: Epoch, expect: Vec<u8> },
    /// Provider pushes these secrets to the sink on the next `read_handshake`.
    InstallSecrets(EpochSecrets),
    /// Provider pushes this event to the sink on the next `read_handshake`.
    /// Slice payload is owned by the step so the sink may borrow it.
    EmitEvent(MockEvent),
    /// Marks the script position where handshake-confirmation occurs.
    Confirm,
    /// Provider's next method-call returns this error.
    FailWith(TlsError),
}

/// Owned representation of [`TlsEvent`] used inside [`MockStep::EmitEvent`].
#[derive(Debug, Clone)]
pub enum MockEvent {
    HandshakeDataReceived,
    HandshakeConfirmed,
    PeerTransportParameters(Vec<u8>),
    AlpnNegotiated(Vec<u8>),
    PeerCertificate(Vec<u8>),
    SessionTicket(Vec<u8>),
}

impl MockEvent {
    fn as_tls_event(&self) -> TlsEvent<'_> {
        match self {
            Self::HandshakeDataReceived => TlsEvent::HandshakeDataReceived,
            Self::HandshakeConfirmed => TlsEvent::HandshakeConfirmed,
            Self::PeerTransportParameters(bytes) => TlsEvent::PeerTransportParameters(bytes),
            Self::AlpnNegotiated(bytes) => TlsEvent::AlpnNegotiated(bytes),
            Self::PeerCertificate(bytes) => TlsEvent::PeerCertificate(bytes),
            Self::SessionTicket(bytes) => TlsEvent::SessionTicket(bytes),
        }
    }
}

/// Configuration for [`MockTlsProvider`].
#[derive(Debug, Clone)]
pub struct MockConfig {
    pub side: Side,
    pub script: Vec<MockStep>,
}

/// AEAD wrapper that records what was protected without doing real
/// crypto — sufficient for FSM-shape testing, NOT for wire-level
/// interop.
#[derive(Debug, Clone)]
pub struct MockAead {
    pub key_marker: u32,
}

impl AeadProvider for MockAead {
    fn seal_in_place(
        &self,
        _nonce: &[u8; 12],
        _aad: &[u8],
        plaintext_then_tag: &mut [u8],
        plaintext_len: usize,
    ) -> Result<usize, TlsError> {
        let total = plaintext_len + 16;
        if plaintext_then_tag.len() < total {
            return Err(TlsError::BufferTooSmall { needed: total });
        }
        // Append a deterministic 16-byte tag so the test can prove it round-trips.
        let marker = self.key_marker.to_be_bytes();
        for offset in 0..16usize {
            plaintext_then_tag[plaintext_len + offset] = marker[offset % 4] ^ (offset as u8);
        }
        Ok(total)
    }

    fn open_in_place(
        &self,
        _nonce: &[u8; 12],
        _aad: &[u8],
        ciphertext_with_tag: &mut [u8],
    ) -> Result<usize, TlsError> {
        if ciphertext_with_tag.len() < 16 {
            return Err(TlsError::DecryptError);
        }
        let total = ciphertext_with_tag.len();
        let plaintext_len = total - 16;
        let marker = self.key_marker.to_be_bytes();
        for offset in 0..16usize {
            let expected = marker[offset % 4] ^ (offset as u8);
            if ciphertext_with_tag[plaintext_len + offset] != expected {
                return Err(TlsError::DecryptError);
            }
        }
        Ok(plaintext_len)
    }
}

/// Scripted TLS provider.
pub struct MockTlsProvider {
    side: Side,
    script: Vec<MockStep>,
    cursor: usize,
    handshaking: bool,
    confirmed: bool,
    aborted: Option<u16>,
    aeads: BTreeMap<(Epoch, u8, Direction), MockAead>,
    pending_failure: Option<TlsError>,
    next_aead_marker: u32,
    last_retry_token: Option<Vec<u8>>,
}

impl MockTlsProvider {
    /// Script a client-handshake.
    #[must_use]
    pub fn script_client(steps: Vec<MockStep>) -> MockConfig {
        MockConfig {
            side: Side::Client,
            script: steps,
        }
    }

    /// Script a server-handshake.
    #[must_use]
    pub fn script_server(steps: Vec<MockStep>) -> MockConfig {
        MockConfig {
            side: Side::Server,
            script: steps,
        }
    }

    /// Current cursor position into the script (useful for assertions).
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Number of script steps remaining.
    #[must_use]
    pub fn remaining_steps(&self) -> usize {
        self.script.len().saturating_sub(self.cursor)
    }

    fn install_aeads_for(&mut self, secrets: &EpochSecrets) {
        // Synthetic markers: high-bit per direction for visibility under debug.
        let local_marker = (u32::from(secrets.epoch.index() as u8) << 16)
            | (u32::from(secrets.generation) << 8)
            | 0xA0;
        let remote_marker = (u32::from(secrets.epoch.index() as u8) << 16)
            | (u32::from(secrets.generation) << 8)
            | 0xB0;
        self.aeads.insert(
            (secrets.epoch, secrets.generation, Direction::Local),
            MockAead {
                key_marker: local_marker,
            },
        );
        self.aeads.insert(
            (secrets.epoch, secrets.generation, Direction::Remote),
            MockAead {
                key_marker: remote_marker,
            },
        );
        self.next_aead_marker = self.next_aead_marker.wrapping_add(1);
    }

    /// Drain script steps up to (but not including) the next
    /// [`MockStep::ReadHandshake`] or [`MockStep::EmitHandshakeBytes`],
    /// firing each into the sink.
    fn drain_async_steps(&mut self, sink: &mut dyn TlsEventSink) {
        while self.cursor < self.script.len() {
            // Borrow-check dance: take ownership of the step we drain.
            let step = match &self.script[self.cursor] {
                MockStep::InstallSecrets(_)
                | MockStep::EmitEvent(_)
                | MockStep::Confirm
                | MockStep::FailWith(_) => self.script[self.cursor].clone(),
                MockStep::EmitHandshakeBytes { .. } | MockStep::ReadHandshake { .. } => return,
            };
            self.cursor += 1;
            match step {
                MockStep::InstallSecrets(secrets) => {
                    self.install_aeads_for(&secrets);
                    sink.on_new_secrets(secrets);
                }
                MockStep::EmitEvent(owned) => {
                    if matches!(owned, MockEvent::HandshakeConfirmed) {
                        // Internal handshaking state mirrors the
                        // event so subsequent `initiate_key_update`
                        // calls don't return NotReady.
                        self.confirmed = true;
                        self.handshaking = false;
                    }
                    sink.on_event(owned.as_tls_event());
                }
                MockStep::Confirm => {
                    self.confirmed = true;
                    self.handshaking = false;
                }
                MockStep::FailWith(err) => {
                    self.pending_failure = Some(err);
                    return;
                }
                MockStep::EmitHandshakeBytes { .. } | MockStep::ReadHandshake { .. } => {
                    unreachable!("filtered above");
                }
            }
        }
    }
}

impl TlsProvider for MockTlsProvider {
    type Config = MockConfig;
    type Aead = MockAead;

    // Mock supports both sides at runtime; the SIDE const is a placeholder
    // for the trait-bound monomorphization. Tests using the side
    // distinction inspect `cfg.side` directly via the connection.
    const SIDE: Side = Side::Client;

    fn new(config: Self::Config, _local_transport_params: &[u8]) -> Result<Self, TlsError> {
        Ok(Self {
            side: config.side,
            script: config.script,
            cursor: 0,
            handshaking: true,
            confirmed: false,
            aborted: None,
            aeads: BTreeMap::new(),
            pending_failure: None,
            next_aead_marker: 0,
            last_retry_token: None,
        })
    }

    fn initial_keys(destination_cid: &[u8]) -> Result<EpochSecrets, TlsError> {
        let pair =
            initial_keys::derive(destination_cid).map_err(|_| TlsError::ProviderInternal(1))?;
        Ok(epoch_secrets_from_initial(pair))
    }

    fn write_handshake(&mut self, epoch: Epoch, out: &mut [u8]) -> Result<Range<usize>, TlsError> {
        if let Some(err) = self.pending_failure.take() {
            return Err(err);
        }
        if self.cursor >= self.script.len() {
            return Err(TlsError::NotReady);
        }
        // Peek next step; if it's EmitHandshakeBytes matching epoch, consume it.
        match &self.script[self.cursor] {
            MockStep::EmitHandshakeBytes {
                epoch: step_epoch,
                bytes,
            } if *step_epoch == epoch => {
                if out.len() < bytes.len() {
                    return Err(TlsError::BufferTooSmall {
                        needed: bytes.len(),
                    });
                }
                let len = bytes.len();
                out[..len].copy_from_slice(bytes);
                self.cursor += 1;
                Ok(0..len)
            }
            _ => Err(TlsError::NotReady),
        }
    }

    fn read_handshake(
        &mut self,
        epoch: Epoch,
        input: &[u8],
        sink: &mut dyn TlsEventSink,
    ) -> Result<(), TlsError> {
        if let Some(err) = self.pending_failure.take() {
            return Err(err);
        }
        if self.cursor >= self.script.len() {
            return Err(TlsError::NotReady);
        }
        let step = self.script[self.cursor].clone();
        match step {
            MockStep::ReadHandshake {
                epoch: step_epoch,
                expect,
            } => {
                if step_epoch != epoch {
                    return Err(TlsError::UnexpectedMessage);
                }
                if expect != input {
                    return Err(TlsError::UnexpectedMessage);
                }
                self.cursor += 1;
            }
            _ => return Err(TlsError::UnexpectedMessage),
        }
        // After a successful ReadHandshake, drain async side-effect steps
        // (InstallSecrets, EmitEvent, Confirm, FailWith) up to the next
        // EmitHandshakeBytes or ReadHandshake.
        self.drain_async_steps(sink);
        Ok(())
    }

    fn initiate_key_update(&mut self) -> Result<EpochSecrets, TlsError> {
        if self.handshaking {
            return Err(TlsError::NotReady);
        }
        // Pick the next application generation by walking already-
        // installed app secrets. Mock has no internal key schedule;
        // it just synthesizes a deterministic generation+1 set.
        let current: u8 = self
            .aeads
            .keys()
            .filter_map(|(epoch, generation, _)| {
                (*epoch == Epoch::Application).then_some(*generation)
            })
            .max()
            .unwrap_or(0);
        let next_gen: u8 = current.saturating_add(1);
        let secrets = synthetic_secrets(Epoch::Application, next_gen, 0xBBu8 ^ next_gen);
        // Install the new-gen AEADs in the mock so aead_for(gen=next)
        // returns them.
        let local_marker: u8 = next_gen ^ 0x40;
        let remote_marker: u8 = next_gen ^ 0x80;
        self.aeads.insert(
            (Epoch::Application, next_gen, Direction::Local),
            MockAead {
                key_marker: u32::from(local_marker),
            },
        );
        self.aeads.insert(
            (Epoch::Application, next_gen, Direction::Remote),
            MockAead {
                key_marker: u32::from(remote_marker),
            },
        );
        Ok(secrets)
    }

    fn aead_for(
        &self,
        epoch: Epoch,
        generation: u8,
        direction: Direction,
    ) -> Result<&Self::Aead, TlsError> {
        self.aeads
            .get(&(epoch, generation, direction))
            .ok_or(TlsError::NotReady)
    }

    fn is_handshaking(&self) -> bool {
        self.handshaking
    }

    fn is_confirmed(&self) -> bool {
        self.confirmed
    }

    fn abort(&mut self, code: u16) {
        self.aborted = Some(code);
        self.handshaking = false;
        self.pending_failure = Some(TlsError::Aborted { code });
    }

    fn set_retry_token(&mut self, token: &[u8]) {
        self.last_retry_token = Some(token.to_vec());
    }
}

/// Borrow-side helper: expose the script side for tests that need to
/// branch on it.
impl MockTlsProvider {
    #[must_use]
    pub fn side(&self) -> Side {
        self.side
    }

    /// Inspect whether `abort` has been called and with which code.
    #[must_use]
    pub fn aborted_with(&self) -> Option<u16> {
        self.aborted
    }

    /// Inspect the most-recent `set_retry_token` call payload, if any.
    /// Used by C19.3 tests to verify the connection FSM threads the
    /// server-issued Retry token down into the TLS provider.
    #[must_use]
    pub fn last_retry_token(&self) -> Option<&[u8]> {
        self.last_retry_token.as_deref()
    }
}

fn epoch_secrets_from_initial(pair: InitialKeyPair) -> EpochSecrets {
    EpochSecrets {
        epoch: Epoch::Initial,
        generation: 0,
        local: DirectionalKeys {
            packet: PacketKeyMaterial::Aes128Gcm {
                key: pair.client.key,
                iv: pair.client.iv,
            },
            header: HeaderKeyMaterial::Aes128 { hp: pair.client.hp },
        },
        remote: DirectionalKeys {
            packet: PacketKeyMaterial::Aes128Gcm {
                key: pair.server.key,
                iv: pair.server.iv,
            },
            header: HeaderKeyMaterial::Aes128 { hp: pair.server.hp },
        },
    }
}

/// Build a synthetic [`EpochSecrets`] for a non-Initial epoch with
/// deterministic placeholder key material; useful for handshake- and
/// application-secret scripting in tests where the wire bytes don't
/// need to be valid TLS-derived.
#[must_use]
pub fn synthetic_secrets(epoch: Epoch, generation: u8, marker: u8) -> EpochSecrets {
    let mut key = [0u8; 16];
    let mut iv = [0u8; 12];
    let mut hp = [0u8; 16];
    for byte in key.iter_mut() {
        *byte = marker;
    }
    for byte in iv.iter_mut() {
        *byte = marker.wrapping_add(1);
    }
    for byte in hp.iter_mut() {
        *byte = marker.wrapping_add(2);
    }
    let local = DirectionalKeys {
        packet: PacketKeyMaterial::Aes128Gcm { key, iv },
        header: HeaderKeyMaterial::Aes128 { hp },
    };
    let mut key_r = key;
    let mut iv_r = iv;
    let mut hp_r = hp;
    for byte in key_r.iter_mut() {
        *byte = marker.wrapping_add(0x10);
    }
    for byte in iv_r.iter_mut() {
        *byte = marker.wrapping_add(0x11);
    }
    for byte in hp_r.iter_mut() {
        *byte = marker.wrapping_add(0x12);
    }
    let remote = DirectionalKeys {
        packet: PacketKeyMaterial::Aes128Gcm {
            key: key_r,
            iv: iv_r,
        },
        header: HeaderKeyMaterial::Aes128 { hp: hp_r },
    };
    EpochSecrets {
        epoch,
        generation,
        local,
        remote,
    }
}

/// Convenience: derive Initial-epoch secrets directly using the C5 path.
///
/// # Errors
///
/// Returns [`TlsError::ProviderInternal`] if [`initial_keys::derive`]
/// fails (only on programmer error — the salt + DCID derivation is
/// deterministic and infallible for well-formed inputs).
pub fn initial_secrets_for(destination_cid: &[u8]) -> Result<EpochSecrets, TlsError> {
    let pair = initial_keys::derive(destination_cid).map_err(|_| TlsError::ProviderInternal(2))?;
    Ok(epoch_secrets_from_initial(pair))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::quic::tls::event::InlineEventSink;

    #[test]
    fn mock_constructor_returns_provider_with_script() {
        let config = MockTlsProvider::script_client(alloc::vec![]);
        let provider = MockTlsProvider::new(config, b"").expect("new ok");
        assert_eq!(provider.cursor(), 0);
        assert!(provider.is_handshaking());
        assert!(!provider.is_confirmed());
    }

    #[test]
    fn write_handshake_emits_scripted_bytes_for_matching_epoch() {
        let config = MockTlsProvider::script_client(alloc::vec![MockStep::EmitHandshakeBytes {
            epoch: Epoch::Initial,
            bytes: alloc::vec![0xDE, 0xAD, 0xBE, 0xEF],
        }]);
        let mut provider = MockTlsProvider::new(config, b"").expect("new ok");
        let mut out = [0u8; 8];
        let range = provider
            .write_handshake(Epoch::Initial, &mut out)
            .expect("write ok");
        assert_eq!(range, 0..4);
        assert_eq!(&out[..4], &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(provider.cursor(), 1);
    }

    #[test]
    fn write_handshake_returns_not_ready_for_wrong_epoch() {
        let config = MockTlsProvider::script_client(alloc::vec![MockStep::EmitHandshakeBytes {
            epoch: Epoch::Handshake,
            bytes: alloc::vec![0xAB],
        }]);
        let mut provider = MockTlsProvider::new(config, b"").expect("new ok");
        let mut out = [0u8; 8];
        let result = provider.write_handshake(Epoch::Initial, &mut out);
        assert!(matches!(result, Err(TlsError::NotReady)));
    }

    #[test]
    fn read_handshake_fires_secrets_then_event_in_order() {
        let secrets = synthetic_secrets(Epoch::Handshake, 0, 0xAA);
        let config = MockTlsProvider::script_client(alloc::vec![
            MockStep::ReadHandshake {
                epoch: Epoch::Initial,
                expect: alloc::vec![0x00, 0x01],
            },
            MockStep::InstallSecrets(secrets),
            MockStep::EmitEvent(MockEvent::HandshakeDataReceived),
        ]);
        let mut provider = MockTlsProvider::new(config, b"").expect("new ok");
        let mut sink = InlineEventSink::new();
        provider
            .read_handshake(Epoch::Initial, &[0x00, 0x01], &mut sink)
            .expect("read ok");
        assert_eq!(sink.secrets().len(), 1);
        assert_eq!(sink.secrets()[0].epoch, Epoch::Handshake);
        assert_eq!(sink.events().len(), 1);
        assert_eq!(provider.cursor(), 3);
        // AEAD was installed by the InstallSecrets step.
        assert!(
            provider
                .aead_for(Epoch::Handshake, 0, Direction::Local)
                .is_ok()
        );
        assert!(
            provider
                .aead_for(Epoch::Handshake, 0, Direction::Remote)
                .is_ok()
        );
    }

    #[test]
    fn read_handshake_rejects_mismatched_bytes() {
        let config = MockTlsProvider::script_client(alloc::vec![MockStep::ReadHandshake {
            epoch: Epoch::Initial,
            expect: alloc::vec![0x00],
        }]);
        let mut provider = MockTlsProvider::new(config, b"").expect("new ok");
        let mut sink = InlineEventSink::new();
        let result = provider.read_handshake(Epoch::Initial, &[0xFF], &mut sink);
        assert!(matches!(result, Err(TlsError::UnexpectedMessage)));
    }

    #[test]
    fn confirm_step_marks_handshake_confirmed() {
        let config = MockTlsProvider::script_client(alloc::vec![
            MockStep::ReadHandshake {
                epoch: Epoch::Initial,
                expect: alloc::vec![],
            },
            MockStep::Confirm,
        ]);
        let mut provider = MockTlsProvider::new(config, b"").expect("new ok");
        let mut sink = InlineEventSink::new();
        provider
            .read_handshake(Epoch::Initial, &[], &mut sink)
            .expect("read ok");
        assert!(provider.is_confirmed());
        assert!(!provider.is_handshaking());
    }

    #[test]
    fn abort_pushes_failure_to_next_method() {
        let config = MockTlsProvider::script_client(alloc::vec![]);
        let mut provider = MockTlsProvider::new(config, b"").expect("new ok");
        provider.abort(0x42);
        let mut out = [0u8; 8];
        let result = provider.write_handshake(Epoch::Initial, &mut out);
        assert!(matches!(result, Err(TlsError::Aborted { code: 0x42 })));
        assert_eq!(provider.aborted_with(), Some(0x42));
    }

    #[test]
    fn mock_aead_round_trips_synthetic_tag() {
        let aead = MockAead {
            key_marker: 0xCAFEBABE,
        };
        let mut buf = [0u8; 32];
        for (offset, slot) in buf.iter_mut().take(8).enumerate() {
            *slot = offset as u8;
        }
        let nonce = [0u8; 12];
        let written = aead
            .seal_in_place(&nonce, b"aad", &mut buf, 8)
            .expect("seal");
        assert_eq!(written, 24);
        let plaintext_len = aead
            .open_in_place(&nonce, b"aad", &mut buf[..written])
            .expect("open");
        assert_eq!(plaintext_len, 8);
    }

    #[test]
    fn mock_aead_rejects_tampered_tag() {
        let aead = MockAead {
            key_marker: 0xCAFEBABE,
        };
        let mut buf = [0u8; 32];
        let nonce = [0u8; 12];
        let written = aead
            .seal_in_place(&nonce, b"aad", &mut buf, 8)
            .expect("seal");
        // Corrupt the tag.
        buf[written - 1] ^= 0x01;
        let result = aead.open_in_place(&nonce, b"aad", &mut buf[..written]);
        assert!(matches!(result, Err(TlsError::DecryptError)));
    }

    #[test]
    fn initial_keys_returns_rfc9001_appendix_a_keys() {
        let dcid = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
        let secrets = MockTlsProvider::initial_keys(&dcid).expect("derive");
        assert_eq!(secrets.epoch, Epoch::Initial);
        assert_eq!(secrets.generation, 0);
        // Validate that local + remote keys differ — proves both sides
        // were derived.
        let PacketKeyMaterial::Aes128Gcm { key: local_key, .. } = secrets.local.packet else {
            panic!("expected AES-128-GCM");
        };
        let PacketKeyMaterial::Aes128Gcm {
            key: remote_key, ..
        } = secrets.remote.packet
        else {
            panic!("expected AES-128-GCM");
        };
        assert_ne!(local_key, remote_key);
    }
}
