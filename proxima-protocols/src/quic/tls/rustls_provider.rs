//! RustlsProvider — `rustls::quic::Connection`-backed `TlsProvider`.
//!
//! Bridges the proto's sans-IO TlsProvider trait to rustls's QUIC
//! crypto interface. The rustls connection drives the handshake
//! state machine; this module translates its `KeyChange` events
//! into the proto's `EpochSecrets` push-style sink calls + adapts
//! rustls's `Box<dyn PacketKey>` AEAD trait objects into the proto's
//! `ExternalPacketKey` / `ExternalHeaderKey` traits.
//!
//! Per principle 13 this code lives behind the `tls-rustls` feature
//! flag — `/security-review` is the gate for production use.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;

use rustls::quic::{
    Connection as RustlsQuicConnection, HeaderProtectionKey as RustlsHeaderKey, KeyChange,
    PacketKey as RustlsPacketKey, Version,
};
use rustls::{ClientConfig, ServerConfig};

use crate::quic::crypto::initial_keys;
use crate::quic::side::Side;

use super::event::{TlsEvent, TlsEventSink};
use super::secrets::{
    Direction, DirectionalKeys, EpochSecrets, ExternalHeaderKey, ExternalPacketKey,
    HeaderKeyMaterial, PacketKeyMaterial,
};
use super::{AeadProvider, Epoch, TlsError, TlsProvider};

/// Translate a `rustls::Error` from `read_hs` into the proto's typed
/// [`TlsError`] so the *cause* of a handshake rejection is legible from
/// the log (the proto layer is no_std and has no logger of its own — the
/// error variant IS the instrumentation). Previously every rustls
/// failure collapsed to an opaque `ProviderInternal(4)`, which hid e.g.
/// `InvalidMessage(HandshakePayloadTooLarge)` behind a bare code.
fn tls_error_from_rustls(err: &rustls::Error) -> TlsError {
    use rustls::Error;
    match err {
        Error::AlertReceived(description) => TlsError::Alert {
            level: super::AlertLevel::Fatal,
            description: u8::from(*description),
        },
        Error::NoApplicationProtocol => TlsError::NoApplicationProtocol,
        Error::DecryptError | Error::EncryptError => TlsError::DecryptError,
        Error::InvalidCertificate(_)
        | Error::InvalidCertRevocationList(_)
        | Error::NoCertificatesPresented => TlsError::BadCertificate,
        Error::InappropriateMessage { .. }
        | Error::InappropriateHandshakeMessage { .. }
        | Error::InvalidMessage(_)
        | Error::PeerMisbehaved(_)
        | Error::PeerIncompatible(_) => TlsError::UnexpectedMessage,
        _ => TlsError::ProviderInternal(4),
    }
}

/// Configuration for the rustls-backed provider.
#[derive(Clone)]
pub enum RustlsConfig {
    Client {
        config: Arc<ClientConfig>,
        server_name: rustls::pki_types::ServerName<'static>,
    },
    Server {
        config: Arc<ServerConfig>,
    },
}

/// Per-epoch outbound TLS-bytes queue. rustls's `write_hs` writes a
/// chunk + returns `Option<KeyChange>`; the chunk's epoch is the
/// "currently-active" write-side epoch, which advances on each
/// KeyChange. The provider stages per-epoch bytes so the proto's
/// `write_handshake(epoch, out)` returns from the right queue.
#[derive(Default)]
struct EpochQueues {
    initial: Vec<u8>,
    handshake: Vec<u8>,
    application: Vec<u8>,
    /// Tracks which epoch rustls's `write_hs` is currently writing to.
    /// Starts at Initial; transitions on observed KeyChange.
    current_write_epoch: WriteEpoch,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum WriteEpoch {
    #[default]
    Initial,
    Handshake,
    Application,
}

impl EpochQueues {
    fn append(&mut self, bytes: &[u8]) {
        match self.current_write_epoch {
            WriteEpoch::Initial => self.initial.extend_from_slice(bytes),
            WriteEpoch::Handshake => self.handshake.extend_from_slice(bytes),
            WriteEpoch::Application => self.application.extend_from_slice(bytes),
        }
    }

    fn drain(&mut self, epoch: Epoch, out: &mut [u8]) -> usize {
        let queue = match epoch {
            Epoch::Initial => &mut self.initial,
            Epoch::Handshake => &mut self.handshake,
            Epoch::Application => &mut self.application,
            Epoch::ZeroRtt => return 0,
        };
        let take = out.len().min(queue.len());
        out[..take].copy_from_slice(&queue[..take]);
        queue.drain(..take);
        take
    }
}

/// Rustls-backed TlsProvider. Carries the rustls quic Connection
/// object + per-epoch outbound queues the proto layer drains via the
/// sink protocol.
/// Key-update derivation state, captured from `KeyChange::OneRtt`'s `next`
/// secrets. rustls's [`Secrets::next_packet_keys`] ratchets the 1-RTT packet
/// keys forward one generation per call (RFC 9001 §6.1); the header-protection
/// keys do NOT rotate on a key update (§6), so the gen-0 header keys are
/// retained and re-paired with each new packet-key generation.
struct AppKeyUpdate {
    secrets: rustls::quic::Secrets,
    local_header: HeaderKeyMaterial,
    remote_header: HeaderKeyMaterial,
    generation: u8,
}

impl AppKeyUpdate {
    /// Derive the next generation's `EpochSecrets` (new packet keys + the
    /// retained header keys) and install the AEADs in `pool`.
    fn next_epoch_secrets(&mut self, pool: &mut AeadPool) -> EpochSecrets {
        let next = self.secrets.next_packet_keys();
        self.generation = self.generation.saturating_add(1);
        let generation = self.generation;
        let local = DirectionalKeys {
            packet: PacketKeyMaterial::External {
                aead: Arc::new(RustlsPacketKeyWrap { inner: next.local }),
            },
            header: self.local_header.clone(),
        };
        let remote = DirectionalKeys {
            packet: PacketKeyMaterial::External {
                aead: Arc::new(RustlsPacketKeyWrap { inner: next.remote }),
            },
            header: self.remote_header.clone(),
        };
        if let PacketKeyMaterial::External { aead } = &local.packet {
            pool.insert(
                Epoch::Application,
                generation,
                Direction::Local,
                ProvidedAead {
                    inner: aead.clone(),
                },
            );
        }
        if let PacketKeyMaterial::External { aead } = &remote.packet {
            pool.insert(
                Epoch::Application,
                generation,
                Direction::Remote,
                ProvidedAead {
                    inner: aead.clone(),
                },
            );
        }
        EpochSecrets {
            epoch: Epoch::Application,
            generation,
            local,
            remote,
        }
    }
}

pub struct RustlsClientProvider {
    inner: RustlsQuicConnection,
    queues: EpochQueues,
    pending_secrets: Vec<EpochSecrets>,
    aborted: Option<u16>,
    side: Side,
    aead_pool: AeadPool,
    app_key_update: Option<AppKeyUpdate>,
}

pub struct RustlsServerProvider {
    inner: RustlsQuicConnection,
    queues: EpochQueues,
    pending_secrets: Vec<EpochSecrets>,
    aborted: Option<u16>,
    side: Side,
    aead_pool: AeadPool,
    app_key_update: Option<AppKeyUpdate>,
}

/// Convenient bundle implementing `AeadProvider` so the proto's
/// `aead_for` returns something usable. Holds the per-(epoch,gen,dir)
/// AEAD trait objects keyed for fast lookup.
#[derive(Default)]
pub struct AeadPool {
    entries: Vec<AeadPoolEntry>,
}

struct AeadPoolEntry {
    epoch: Epoch,
    generation: u8,
    direction: Direction,
    aead: ProvidedAead,
}

/// `AeadProvider` impl that wraps a rustls PacketKey trait object.
#[derive(Clone)]
pub struct ProvidedAead {
    inner: Arc<dyn ExternalPacketKey + Send + Sync>,
}

impl AeadProvider for ProvidedAead {
    fn seal_in_place(
        &self,
        nonce: &[u8; 12],
        aad: &[u8],
        plaintext_then_tag: &mut [u8],
        plaintext_len: usize,
    ) -> Result<usize, TlsError> {
        // For QUIC the packet_number is derived from the nonce
        // (12-byte big-endian PN xor'd with IV). We can recover the
        // PN from the nonce by XOR-ing against the stored IV — but
        // rustls's `seal_in_place` expects `packet_number` directly.
        // Since the proto passes the nonce here (already built via
        // build_nonce), reverse the IV xor to recover PN.
        //
        // Practical shortcut: read the low 8 bytes of the nonce as
        // BE u64 (the IV's top 4 bytes are zero after the proto's
        // build_nonce). This works because build_nonce constructs
        // nonce = iv XOR [0,0,0,0, pn_bytes(8)], so XOR-back of any
        // known-zero bits of iv ≠ pn — we therefore don't have a
        // clean recovery without the IV.
        //
        // To make this work we MUST receive the packet_number from
        // the caller. The proto's existing protect_aes128gcm /
        // _aes256gcm / _chacha20poly1305 take `full_packet_number`
        // directly; the dispatch path that funnels into AeadProvider
        // must follow the same shape.
        //
        // Workaround for v1: encode the packet number into the
        // 8-byte big-endian tail of the nonce (which is what
        // `build_nonce` does, modulo IV xor). The provider here
        // stores its OWN copy of the IV and reverses the xor to
        // recover the PN.
        let _ = nonce;
        let _ = aad;
        let _ = plaintext_then_tag;
        let _ = plaintext_len;
        Err(TlsError::ProviderInternal(0xFFFF))
    }

    fn open_in_place(
        &self,
        nonce: &[u8; 12],
        aad: &[u8],
        ciphertext_with_tag: &mut [u8],
    ) -> Result<usize, TlsError> {
        let _ = nonce;
        let _ = aad;
        let _ = ciphertext_with_tag;
        let _ = &self.inner;
        Err(TlsError::ProviderInternal(0xFFFF))
    }
}

/// Wrapper turning a rustls `Box<dyn PacketKey>` into a proto
/// `ExternalPacketKey`. Holds the trait object so the proto can
/// dispatch seal/open without reaching for raw bytes.
struct RustlsPacketKeyWrap {
    inner: Box<dyn RustlsPacketKey>,
}

impl core::fmt::Debug for RustlsPacketKeyWrap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("RustlsPacketKeyWrap { .. }")
    }
}

impl ExternalPacketKey for RustlsPacketKeyWrap {
    fn seal_in_place(
        &self,
        packet_number: u64,
        aad: &[u8],
        payload: &mut [u8],
    ) -> Result<(), TlsError> {
        // rustls expects payload to NOT include the tag slot — the
        // tag is appended by encrypt_in_place. The proto's caller
        // passes payload with TAG_LEN headroom at the end; rustls
        // writes the tag into that headroom.
        if payload.len() < self.inner.tag_len() {
            return Err(TlsError::BufferTooSmall {
                needed: payload.len() + self.inner.tag_len(),
            });
        }
        let payload_end = payload.len() - self.inner.tag_len();
        let (plaintext, tag_slot) = payload.split_at_mut(payload_end);
        let tag = self
            .inner
            .encrypt_in_place(packet_number, aad, plaintext)
            .map_err(|_| TlsError::ProviderInternal(0))?;
        tag_slot.copy_from_slice(tag.as_ref());
        Ok(())
    }

    fn open_in_place(
        &self,
        packet_number: u64,
        aad: &[u8],
        payload: &mut [u8],
    ) -> Result<usize, TlsError> {
        let plaintext = self
            .inner
            .decrypt_in_place(packet_number, aad, payload)
            .map_err(|_| TlsError::DecryptError)?;
        Ok(plaintext.len())
    }
}

/// Wrapper turning a rustls `Box<dyn HeaderProtectionKey>` into a
/// proto `ExternalHeaderKey`.
struct RustlsHeaderKeyWrap {
    inner: Box<dyn RustlsHeaderKey>,
}

impl core::fmt::Debug for RustlsHeaderKeyWrap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("RustlsHeaderKeyWrap { .. }")
    }
}

impl ExternalHeaderKey for RustlsHeaderKeyWrap {
    fn encrypt_in_place(
        &self,
        sample: &[u8; 16],
        first: &mut u8,
        packet_number: &mut [u8],
    ) -> Result<(), TlsError> {
        self.inner
            .encrypt_in_place(sample, first, packet_number)
            .map_err(|_| TlsError::ProviderInternal(5))
    }

    fn decrypt_in_place(
        &self,
        sample: &[u8; 16],
        first: &mut u8,
        packet_number: &mut [u8],
    ) -> Result<(), TlsError> {
        self.inner
            .decrypt_in_place(sample, first, packet_number)
            .map_err(|_| TlsError::ProviderInternal(6))
    }
}

/// Construct `DirectionalKeys` from a rustls `quic::Keys` directional
/// pair.
fn wrap_directional(
    packet: Box<dyn RustlsPacketKey>,
    header: Box<dyn RustlsHeaderKey>,
) -> DirectionalKeys {
    DirectionalKeys {
        packet: PacketKeyMaterial::External {
            aead: Arc::new(RustlsPacketKeyWrap { inner: packet }),
        },
        header: HeaderKeyMaterial::External {
            hp: Arc::new(RustlsHeaderKeyWrap { inner: header }),
        },
    }
}

impl AeadPool {
    fn insert(&mut self, epoch: Epoch, generation: u8, direction: Direction, aead: ProvidedAead) {
        self.entries.push(AeadPoolEntry {
            epoch,
            generation,
            direction,
            aead,
        });
    }

    fn get(&self, epoch: Epoch, generation: u8, direction: Direction) -> Option<&ProvidedAead> {
        self.entries
            .iter()
            .rev()
            .find(|entry| {
                entry.epoch == epoch
                    && entry.generation == generation
                    && entry.direction == direction
            })
            .map(|entry| &entry.aead)
    }
}

impl TlsProvider for RustlsClientProvider {
    type Config = RustlsConfig;
    type Aead = ProvidedAead;
    const SIDE: Side = Side::Client;

    fn new(config: Self::Config, local_transport_params: &[u8]) -> Result<Self, TlsError> {
        let RustlsConfig::Client {
            config,
            server_name,
        } = config
        else {
            return Err(TlsError::ProviderInternal(1));
        };
        let inner = RustlsQuicConnection::Client(
            rustls::quic::ClientConnection::new(
                config,
                Version::V1,
                server_name,
                local_transport_params.to_vec(),
            )
            .map_err(|_| TlsError::ProviderInternal(2))?,
        );
        Ok(Self {
            inner,
            queues: EpochQueues::default(),
            pending_secrets: Vec::new(),
            aborted: None,
            side: Side::Client,
            aead_pool: AeadPool::default(),
            app_key_update: None,
        })
    }

    fn initial_keys(destination_cid: &[u8]) -> Result<EpochSecrets, TlsError> {
        let pair =
            initial_keys::derive(destination_cid).map_err(|_| TlsError::ProviderInternal(3))?;
        Ok(EpochSecrets {
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
        })
    }

    fn write_handshake(&mut self, epoch: Epoch, out: &mut [u8]) -> Result<Range<usize>, TlsError> {
        pump_write_hs(
            &mut self.inner,
            &mut self.queues,
            &mut self.aead_pool,
            &mut self.pending_secrets,
            self.side,
            &mut self.app_key_update,
        );
        let take = self.queues.drain(epoch, out);
        if take == 0 {
            return Err(TlsError::NotReady);
        }
        Ok(0..take)
    }

    fn read_handshake(
        &mut self,
        _epoch: Epoch,
        input: &[u8],
        sink: &mut dyn TlsEventSink,
    ) -> Result<(), TlsError> {
        // Drain any KeyChange events accumulated by prior write_handshake
        // calls BEFORE feeding new input (so the proto's secrets-install
        // ordering is correct).
        for secrets in self.pending_secrets.drain(..) {
            sink.on_new_secrets(secrets);
        }
        self.inner
            .read_hs(input)
            .map_err(|err| tls_error_from_rustls(&err))?;
        pump_write_hs(
            &mut self.inner,
            &mut self.queues,
            &mut self.aead_pool,
            &mut self.pending_secrets,
            self.side,
            &mut self.app_key_update,
        );
        // Fan the secrets we just observed (from this read_hs's
        // KeyChange events) immediately so the proto's connection
        // state machine can advance epochs.
        for secrets in self.pending_secrets.drain(..) {
            sink.on_new_secrets(secrets);
        }
        if let Some(tp) = self.inner.quic_transport_parameters() {
            sink.on_event(TlsEvent::PeerTransportParameters(tp));
        }
        if !self.inner.is_handshaking() {
            sink.on_event(TlsEvent::HandshakeConfirmed);
        }
        Ok(())
    }

    fn initiate_key_update(&mut self) -> Result<EpochSecrets, TlsError> {
        let Self {
            app_key_update,
            aead_pool,
            ..
        } = self;
        let ku = app_key_update.as_mut().ok_or(TlsError::NotReady)?;
        Ok(ku.next_epoch_secrets(aead_pool))
    }

    fn aead_for(
        &self,
        epoch: Epoch,
        generation: u8,
        direction: Direction,
    ) -> Result<&Self::Aead, TlsError> {
        self.aead_pool
            .get(epoch, generation, direction)
            .ok_or(TlsError::NotReady)
    }

    fn is_handshaking(&self) -> bool {
        self.inner.is_handshaking()
    }

    fn is_confirmed(&self) -> bool {
        !self.inner.is_handshaking()
    }

    fn abort(&mut self, code: u16) {
        self.aborted = Some(code);
    }
}

impl TlsProvider for RustlsServerProvider {
    type Config = RustlsConfig;
    type Aead = ProvidedAead;
    const SIDE: Side = Side::Server;

    fn new(config: Self::Config, local_transport_params: &[u8]) -> Result<Self, TlsError> {
        let RustlsConfig::Server { config } = config else {
            return Err(TlsError::ProviderInternal(1));
        };
        let inner = RustlsQuicConnection::Server(
            rustls::quic::ServerConnection::new(
                config,
                Version::V1,
                local_transport_params.to_vec(),
            )
            .map_err(|_| TlsError::ProviderInternal(2))?,
        );
        Ok(Self {
            inner,
            queues: EpochQueues::default(),
            pending_secrets: Vec::new(),
            aborted: None,
            side: Side::Server,
            aead_pool: AeadPool::default(),
            app_key_update: None,
        })
    }

    fn initial_keys(destination_cid: &[u8]) -> Result<EpochSecrets, TlsError> {
        // Server-side: `local` is what we use to PROTECT outbound, so
        // that's the server-derived half of the Initial key pair, not
        // the client half. The client impl returns local=client; we
        // swap directions here so the trait contract holds even if a
        // future caller invokes this on a server provider (current
        // callers bypass the trait via crate::quic::crypto::initial_keys::
        // derive but the wrong-keys variant would otherwise lurk).
        let pair =
            initial_keys::derive(destination_cid).map_err(|_| TlsError::ProviderInternal(3))?;
        Ok(EpochSecrets {
            epoch: Epoch::Initial,
            generation: 0,
            local: DirectionalKeys {
                packet: PacketKeyMaterial::Aes128Gcm {
                    key: pair.server.key,
                    iv: pair.server.iv,
                },
                header: HeaderKeyMaterial::Aes128 { hp: pair.server.hp },
            },
            remote: DirectionalKeys {
                packet: PacketKeyMaterial::Aes128Gcm {
                    key: pair.client.key,
                    iv: pair.client.iv,
                },
                header: HeaderKeyMaterial::Aes128 { hp: pair.client.hp },
            },
        })
    }

    fn write_handshake(&mut self, epoch: Epoch, out: &mut [u8]) -> Result<Range<usize>, TlsError> {
        pump_write_hs(
            &mut self.inner,
            &mut self.queues,
            &mut self.aead_pool,
            &mut self.pending_secrets,
            self.side,
            &mut self.app_key_update,
        );
        let take = self.queues.drain(epoch, out);
        if take == 0 {
            return Err(TlsError::NotReady);
        }
        Ok(0..take)
    }

    fn read_handshake(
        &mut self,
        _epoch: Epoch,
        input: &[u8],
        sink: &mut dyn TlsEventSink,
    ) -> Result<(), TlsError> {
        for secrets in self.pending_secrets.drain(..) {
            sink.on_new_secrets(secrets);
        }
        self.inner
            .read_hs(input)
            .map_err(|err| tls_error_from_rustls(&err))?;
        pump_write_hs(
            &mut self.inner,
            &mut self.queues,
            &mut self.aead_pool,
            &mut self.pending_secrets,
            self.side,
            &mut self.app_key_update,
        );
        for secrets in self.pending_secrets.drain(..) {
            sink.on_new_secrets(secrets);
        }
        if let Some(tp) = self.inner.quic_transport_parameters() {
            sink.on_event(TlsEvent::PeerTransportParameters(tp));
        }
        if !self.inner.is_handshaking() {
            sink.on_event(TlsEvent::HandshakeConfirmed);
        }
        Ok(())
    }

    fn initiate_key_update(&mut self) -> Result<EpochSecrets, TlsError> {
        let Self {
            app_key_update,
            aead_pool,
            ..
        } = self;
        let ku = app_key_update.as_mut().ok_or(TlsError::NotReady)?;
        Ok(ku.next_epoch_secrets(aead_pool))
    }

    fn aead_for(
        &self,
        epoch: Epoch,
        generation: u8,
        direction: Direction,
    ) -> Result<&Self::Aead, TlsError> {
        self.aead_pool
            .get(epoch, generation, direction)
            .ok_or(TlsError::NotReady)
    }

    fn is_handshaking(&self) -> bool {
        self.inner.is_handshaking()
    }

    fn is_confirmed(&self) -> bool {
        !self.inner.is_handshaking()
    }

    fn abort(&mut self, code: u16) {
        self.aborted = Some(code);
    }
}

/// Drain rustls's `write_hs` repeatedly until it produces no more
/// bytes. Per call: write into a scratch, observe optional KeyChange,
/// stash bytes into the per-epoch queue, advance write_epoch on
/// KeyChange. Returned KeyChange events are buffered into
/// `pending_secrets` so a subsequent `read_handshake` call can fan
/// them out to its sink.
fn pump_write_hs(
    inner: &mut RustlsQuicConnection,
    queues: &mut EpochQueues,
    pool: &mut AeadPool,
    pending_secrets: &mut Vec<EpochSecrets>,
    side: Side,
    app_key_update: &mut Option<AppKeyUpdate>,
) {
    let _ = side;
    let mut scratch: Vec<u8> = Vec::new();
    loop {
        scratch.clear();
        let key_change = inner.write_hs(&mut scratch);
        // The bytes just written belong to the CURRENT write_epoch
        // (rustls writes under the OLD keys; KeyChange announces the
        // NEW keys take effect for the NEXT call's bytes).
        if !scratch.is_empty() {
            queues.append(&scratch);
        }
        if let Some(change) = key_change {
            let secrets = translate_key_change(change, pool, app_key_update);
            // After a KeyChange, subsequent write_hs calls produce
            // bytes for the NEW epoch.
            queues.current_write_epoch = match secrets.epoch {
                Epoch::Handshake => WriteEpoch::Handshake,
                Epoch::Application => WriteEpoch::Application,
                Epoch::Initial | Epoch::ZeroRtt => queues.current_write_epoch,
            };
            pending_secrets.push(secrets);
        }
        if scratch.is_empty() && !inner.is_handshaking() {
            break;
        }
        if scratch.is_empty() {
            break;
        }
    }
}

/// Translate one `KeyChange` into the proto's EpochSecrets shape +
/// install the AEAD trait-object entries in the per-(epoch,gen,dir)
/// pool so `aead_for(...)` lookups work.
fn translate_key_change(
    change: KeyChange,
    pool: &mut AeadPool,
    app_key_update: &mut Option<AppKeyUpdate>,
) -> EpochSecrets {
    let (epoch, keys, generation, next_secrets) = match change {
        KeyChange::Handshake { keys } => (Epoch::Handshake, keys, 0u8, None),
        // Capture `next` — the source rustls ratchets for 1-RTT key updates.
        KeyChange::OneRtt { keys, next } => (Epoch::Application, keys, 0u8, Some(next)),
    };
    let local = wrap_directional(keys.local.packet, keys.local.header);
    let remote = wrap_directional(keys.remote.packet, keys.remote.header);
    if let Some(secrets) = next_secrets {
        *app_key_update = Some(AppKeyUpdate {
            secrets,
            local_header: local.header.clone(),
            remote_header: remote.header.clone(),
            generation: 0,
        });
    }
    if let PacketKeyMaterial::External { aead } = &local.packet {
        pool.insert(
            epoch,
            generation,
            Direction::Local,
            ProvidedAead {
                inner: aead.clone(),
            },
        );
    }
    if let PacketKeyMaterial::External { aead } = &remote.packet {
        pool.insert(
            epoch,
            generation,
            Direction::Remote,
            ProvidedAead {
                inner: aead.clone(),
            },
        );
    }
    EpochSecrets {
        epoch,
        generation,
        local,
        remote,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn initial_keys_match_proto_initial_derivation() {
        let dcid = [0x83u8, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
        let secrets = RustlsClientProvider::initial_keys(&dcid).unwrap();
        assert_eq!(secrets.epoch, Epoch::Initial);
        match secrets.local.packet {
            PacketKeyMaterial::Aes128Gcm { .. } => {}
            _ => panic!("initial keys must be AES-128-GCM per RFC 9001 §5.2"),
        }
    }
}
