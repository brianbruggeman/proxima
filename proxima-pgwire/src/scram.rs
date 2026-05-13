//! Sans-IO SCRAM-SHA-256 server state machine (RFC 5802 + RFC 7677).
//!
//! Bytes in, bytes out — no I/O, exactly like the codec. The connection
//! driver owns the wire (AuthenticationSASL / SASLContinue / SASLFinal);
//! this type owns the crypto and the two-step exchange.
//!
//! We hold the user's plaintext password (via `PasswordVerifier`), so we
//! mint a *fresh random salt per authentication* rather than persisting a
//! verifier — there is no stored `(salt, StoredKey, ServerKey)` to reuse,
//! and a per-auth salt is strictly safe here. Channel binding is not
//! offered: `SCRAM-SHA-256` (not `-PLUS`) means the client must send gs2
//! `n` (no binding); a `y`/`p` gs2 header is rejected.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use hmac::{Hmac, Mac};
use rand::RngExt as _;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

type HmacSha256 = Hmac<Sha256>;

const SALT_LEN: usize = 16;
const NONCE_BYTES: usize = 18;
const ITERATIONS: u32 = 4096;
// client-side DoS guard: reject a server demanding an absurd PBKDF2 cost
const MAX_SCRAM_ITERATIONS: u32 = 1_000_000;

/// SCRAM exchange failures. All map to a fatal `28P01` on the wire; the
/// variant carries the cause for logs without leaking secret material.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScramError {
    #[error("malformed client-first message")]
    MalformedClientFirst,
    #[error("stored server password fails saslprep")]
    InvalidStoredPassword,
    #[error("malformed client-final message")]
    MalformedClientFinal,
    #[error("channel binding requested but not supported")]
    ChannelBindingUnsupported,
    #[error("client nonce does not match server-issued nonce")]
    NonceMismatch,
    #[error("client proof is malformed")]
    MalformedProof,
    #[error("client proof verification failed")]
    ProofMismatch,
    #[error("scram exchange used out of order")]
    OutOfOrder,
    #[error("malformed server-first message")]
    MalformedServerFirst,
    #[error("malformed server-final message")]
    MalformedServerFinal,
    #[error("server signature verification failed")]
    ServerSignatureMismatch,
}

#[expect(
    clippy::expect_used,
    reason = "hmac keys any byte length, so KeyInit is infallible here; a diagnosable panic beats a hang on the impossible arm"
)]
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(message);
    mac.finalize().into_bytes().into()
}

fn sha256(input: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(input);
    hasher.finalize().into()
}

fn xor32(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut out = [0_u8; 32];
    for index in 0..32 {
        out[index] = left[index] ^ right[index];
    }
    out
}

#[expect(
    clippy::expect_used,
    reason = "pbkdf2 only errors when output length exceeds the prf limit; 32 bytes is always valid"
)]
fn pbkdf2_sha256(password: &[u8], salt: &[u8], iterations: u32) -> [u8; 32] {
    let mut output = [0_u8; 32];
    pbkdf2::pbkdf2::<HmacSha256>(password, salt, iterations, &mut output)
        .expect("pbkdf2 output length of 32 bytes is within the prf limit");
    output
}

/// The proof + server-signature a verifier expects, derived purely from
/// the inputs. Isolated so the RFC 7677 worked example can drive it.
#[derive(Debug)]
pub struct ScramKeys {
    pub stored_key: [u8; 32],
    pub client_signature: [u8; 32],
    pub server_signature: [u8; 32],
}

/// Pure crypto core: given the SASLprepped password, salt, iteration
/// count, and the three message pieces of `AuthMessage`, derive the
/// StoredKey, ClientSignature, and ServerSignature per RFC 5802 §3.
#[must_use]
pub fn scram_keys(
    salted_password: &[u8; 32],
    client_first_bare: &[u8],
    server_first: &[u8],
    client_final_without_proof: &[u8],
) -> ScramKeys {
    let client_key = Zeroizing::new(hmac_sha256(salted_password, b"Client Key"));
    let stored_key = sha256(&*client_key);
    let server_key = Zeroizing::new(hmac_sha256(salted_password, b"Server Key"));

    let mut auth_message = Vec::with_capacity(
        client_first_bare.len() + server_first.len() + client_final_without_proof.len() + 2,
    );
    auth_message.extend_from_slice(client_first_bare);
    auth_message.push(b',');
    auth_message.extend_from_slice(server_first);
    auth_message.push(b',');
    auth_message.extend_from_slice(client_final_without_proof);

    let client_signature = hmac_sha256(&stored_key, &auth_message);
    let server_signature = hmac_sha256(&*server_key, &auth_message);
    ScramKeys {
        stored_key,
        client_signature,
        server_signature,
    }
}

/// Derives SaltedPassword = PBKDF2-HMAC-SHA256(SASLprep(password), salt).
///
/// # Errors
/// [`ScramError::InvalidStoredPassword`] when SASLprep rejects the password
/// (prohibited code points / bidi rule).
pub fn salted_password(
    password: &str,
    salt: &[u8],
    iterations: u32,
) -> Result<[u8; 32], ScramError> {
    let prepped = stringprep::saslprep(password).map_err(|_| ScramError::InvalidStoredPassword)?;
    Ok(pbkdf2_sha256(prepped.as_bytes(), salt, iterations))
}

/// SCRAM-SHA-256 client side of the SASL exchange — the inverse of
/// [`ScramServer`], for proxima acting as a PostgreSQL *client* (e.g. the
/// real-PG parity harness). Sans-IO: bytes in, bytes out; the driver owns the
/// SASL wire framing. Composes the same crypto core as the server
/// ([`scram_keys`], [`salted_password`]) — no second implementation of the
/// math (principle 1). Channel binding is not used: gs2 header is `n,,`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScramPhase {
    Start,
    AwaitServerFirst,
    AwaitServerFinal,
    Done,
}

#[derive(Debug)]
pub struct ScramClient {
    password: Zeroizing<String>,
    client_nonce: Vec<u8>,
    client_first_bare: Vec<u8>,
    server_signature: Option<[u8; 32]>,
    phase: ScramPhase,
}

impl ScramClient {
    /// Builds a client for `password`, minting a fresh printable client nonce.
    #[must_use]
    pub fn new(password: &str) -> Self {
        let raw = random_bytes::<NONCE_BYTES>();
        // base64 is printable ASCII with no comma, so it is a valid SCRAM nonce
        let client_nonce = BASE64.encode(raw).into_bytes();
        Self {
            password: Zeroizing::new(password.to_string()),
            client_nonce,
            client_first_bare: Vec::new(),
            server_signature: None,
            phase: ScramPhase::Start,
        }
    }

    /// The client-first message `n,,n=,r=<nonce>`. The SCRAM username (`n=`) is
    /// empty because PostgreSQL carries the role in the StartupMessage.
    pub fn client_first(&mut self) -> Vec<u8> {
        let mut bare = Vec::with_capacity(self.client_nonce.len() + 5);
        bare.extend_from_slice(b"n=,r=");
        bare.extend_from_slice(&self.client_nonce);
        self.client_first_bare = bare.clone();

        let mut message = Vec::with_capacity(bare.len() + 3);
        message.extend_from_slice(b"n,,");
        message.extend_from_slice(&bare);
        message
    }

    /// Consumes the server-first `r=...,s=...,i=...`, returns the client-final
    /// `c=biws,r=...,p=<proof>`, and stashes the expected server signature for
    /// [`Self::verify_server_final`].
    ///
    /// # Errors
    /// [`ScramError`] on a malformed server-first, a nonce that does not extend
    /// our client nonce, or a password that fails SASLprep.
    pub fn client_final(&mut self, server_first: &[u8]) -> Result<Vec<u8>, ScramError> {
        let (mut nonce, mut salt_b64, mut iters) = (None, None, None);
        for field in server_first.split(|byte| *byte == b',') {
            match field.first() {
                Some(b'r') => nonce = field.get(2..),
                Some(b's') => salt_b64 = field.get(2..),
                Some(b'i') => iters = field.get(2..),
                _ => {}
            }
        }
        let nonce = nonce.ok_or(ScramError::MalformedServerFirst)?;
        let salt_b64 = salt_b64.ok_or(ScramError::MalformedServerFirst)?;
        let iters = iters.ok_or(ScramError::MalformedServerFirst)?;

        if !nonce.starts_with(&self.client_nonce) {
            return Err(ScramError::NonceMismatch);
        }
        let salt = BASE64
            .decode(salt_b64)
            .map_err(|_| ScramError::MalformedServerFirst)?;
        let iterations: u32 = core::str::from_utf8(iters)
            .ok()
            .and_then(|text| text.parse().ok())
            .ok_or(ScramError::MalformedServerFirst)?;
        // a hostile server could demand a huge iteration count to make our
        // PBKDF2 a self-inflicted DoS; cap it well above any real deployment
        // (PostgreSQL uses 4096; hardened servers stay well under this).
        if iterations == 0 || iterations > MAX_SCRAM_ITERATIONS {
            return Err(ScramError::MalformedServerFirst);
        }

        let salted = salted_password(self.password.as_str(), &salt, iterations)?;

        let mut without_proof = Vec::with_capacity(nonce.len() + 9);
        without_proof.extend_from_slice(b"c=biws,r=");
        without_proof.extend_from_slice(nonce);

        let keys = scram_keys(
            &salted,
            &self.client_first_bare,
            server_first,
            &without_proof,
        );
        let client_key = Zeroizing::new(hmac_sha256(&salted, b"Client Key"));
        let proof = xor32(&client_key, &keys.client_signature);
        self.server_signature = Some(keys.server_signature);

        let mut message = without_proof;
        message.extend_from_slice(b",p=");
        message.extend_from_slice(BASE64.encode(proof).as_bytes());
        Ok(message)
    }

    /// Verifies the server-final `v=<ServerSignature>` in constant time.
    ///
    /// # Errors
    /// [`ScramError::ServerSignatureMismatch`] when the signature does not
    /// match, [`ScramError::OutOfOrder`] if called before `client_final`.
    pub fn verify_server_final(&self, server_final: &[u8]) -> Result<(), ScramError> {
        let expected = self.server_signature.ok_or(ScramError::OutOfOrder)?;
        let encoded = server_final
            .strip_prefix(b"v=")
            .ok_or(ScramError::MalformedServerFinal)?;
        let signature = BASE64
            .decode(encoded)
            .map_err(|_| ScramError::MalformedServerFinal)?;
        if signature.ct_eq(&expected).into() {
            Ok(())
        } else {
            Err(ScramError::ServerSignatureMismatch)
        }
    }
}

/// pgwire's SCRAM is auth form #4 — a challenge/response handshake. Satisfying
/// the generic [`proxima_auth::Handshake`] makes it one instance of the auth
/// axis: the pgwire ClientSession drives it over the SASL wire edge, exactly as
/// an HTTP edge would drive Digest/Kerberos. The crypto core is unchanged; this
/// only sequences the two server rounds (server-first → client-final, then
/// server-final → verify).
impl proxima_auth::Handshake for ScramClient {
    type Error = ScramError;

    fn first(&mut self) -> Vec<u8> {
        self.phase = ScramPhase::AwaitServerFirst;
        self.client_first()
    }

    fn step(&mut self, server: &[u8]) -> Result<Option<Vec<u8>>, ScramError> {
        match self.phase {
            ScramPhase::AwaitServerFirst => {
                let client_final = self.client_final(server)?;
                self.phase = ScramPhase::AwaitServerFinal;
                Ok(Some(client_final))
            }
            ScramPhase::AwaitServerFinal => {
                self.verify_server_final(server)?;
                self.phase = ScramPhase::Done;
                Ok(None)
            }
            ScramPhase::Start | ScramPhase::Done => Err(ScramError::OutOfOrder),
        }
    }
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut buffer = [0_u8; N];
    rand::rng().fill(&mut buffer[..]);
    buffer
}

/// Splits the gs2 channel-binding header from the client-first-bare slice,
/// returning `(gs2_header, bare)`. The header is the exact prefix the
/// client echoes (base64) in client-final's `c=`, so it is captured
/// verbatim (it may carry an `a=authzid`, e.g. `n,a=admin,`). Accepts `n,`
/// (no binding) and `n,,`; rejects `y`/`p` (we do not offer `-PLUS`).
fn split_gs2(client_first: &[u8]) -> Result<(&[u8], &[u8]), ScramError> {
    let mut rest = match client_first.first() {
        Some(b'n') => client_first
            .get(1..)
            .ok_or(ScramError::MalformedClientFirst)?,
        Some(b'y' | b'p') => return Err(ScramError::ChannelBindingUnsupported),
        _ => return Err(ScramError::MalformedClientFirst),
    };
    if rest.first() != Some(&b',') {
        return Err(ScramError::MalformedClientFirst);
    }
    rest = rest.get(1..).ok_or(ScramError::MalformedClientFirst)?;
    // authzid section up to the next comma (may be empty)
    let comma = rest
        .iter()
        .position(|byte| *byte == b',')
        .ok_or(ScramError::MalformedClientFirst)?;
    let header_len = client_first.len() - rest.len() + comma + 1;
    let header = client_first
        .get(..header_len)
        .ok_or(ScramError::MalformedClientFirst)?;
    let bare = client_first
        .get(header_len..)
        .ok_or(ScramError::MalformedClientFirst)?;
    Ok((header, bare))
}

/// Finds the value of an attribute `name=` at the head of `input`,
/// returning `(value, remainder_after_comma_or_end)`.
fn take_attr(input: &[u8], name: u8) -> Result<&[u8], ()> {
    if input.first() != Some(&name) || input.get(1) != Some(&b'=') {
        return Err(());
    }
    let value = input.get(2..).ok_or(())?;
    Ok(value)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    AwaitClientFirst,
    AwaitClientFinal,
    Done,
}

/// SCRAM-SHA-256 server side of the SASL exchange for one connection.
#[derive(Debug)]
pub struct ScramServer {
    // PBKDF2 output is the SCRAM verifier equivalent — wiped on drop.
    salted_password: Zeroizing<[u8; 32]>,
    salt: [u8; SALT_LEN],
    iterations: u32,
    step: Step,
    gs2_header: Vec<u8>,
    client_first_bare: Vec<u8>,
    server_first: Vec<u8>,
    full_nonce: Vec<u8>,
}

impl ScramServer {
    /// Builds a server that authenticates against `password`, minting a
    /// fresh random salt for this exchange.
    ///
    /// # Errors
    /// [`ScramError::MalformedClientFirst`] when SASLprep rejects the
    /// stored password.
    pub fn new(password: &str) -> Result<Self, ScramError> {
        let salt = random_bytes::<SALT_LEN>();
        let salted_password = Zeroizing::new(salted_password(password, &salt, ITERATIONS)?);
        Ok(Self {
            salted_password,
            salt,
            iterations: ITERATIONS,
            step: Step::AwaitClientFirst,
            gs2_header: Vec::new(),
            client_first_bare: Vec::new(),
            server_first: Vec::new(),
            full_nonce: Vec::new(),
        })
    }

    /// Parses `n,,n=user,r=cnonce`, appends a server nonce, and emits the
    /// `r=...,s=...,i=...` server-first message.
    ///
    /// # Errors
    /// [`ScramError`] on a malformed header / message or a channel-binding
    /// request.
    pub fn handle_client_first(&mut self, client_first: &[u8]) -> Result<Vec<u8>, ScramError> {
        if self.step != Step::AwaitClientFirst {
            return Err(ScramError::OutOfOrder);
        }
        let (gs2_header, bare) = split_gs2(client_first)?;
        self.gs2_header = gs2_header.to_vec();
        self.client_first_bare = bare.to_vec();

        let mut fields = bare.split(|byte| *byte == b',');
        let user_field = fields.next().ok_or(ScramError::MalformedClientFirst)?;
        let nonce_field = fields.next().ok_or(ScramError::MalformedClientFirst)?;
        // n=... (username, ignored: identity rides the startup packet)
        take_attr(user_field, b'n').map_err(|()| ScramError::MalformedClientFirst)?;
        let client_nonce =
            take_attr(nonce_field, b'r').map_err(|()| ScramError::MalformedClientFirst)?;
        if client_nonce.is_empty() {
            return Err(ScramError::MalformedClientFirst);
        }

        let server_nonce = BASE64.encode(random_bytes::<NONCE_BYTES>());
        let mut full_nonce = Vec::with_capacity(client_nonce.len() + server_nonce.len());
        full_nonce.extend_from_slice(client_nonce);
        full_nonce.extend_from_slice(server_nonce.as_bytes());
        self.full_nonce = full_nonce;

        let salt_b64 = BASE64.encode(self.salt);
        let mut server_first = Vec::new();
        server_first.extend_from_slice(b"r=");
        server_first.extend_from_slice(&self.full_nonce);
        server_first.extend_from_slice(b",s=");
        server_first.extend_from_slice(salt_b64.as_bytes());
        server_first.extend_from_slice(b",i=");
        server_first.extend_from_slice(self.iterations.to_string().as_bytes());

        self.server_first = server_first.clone();
        self.step = Step::AwaitClientFinal;
        Ok(server_first)
    }

    /// Parses `c=<gs2>,r=...,p=proof` (where `<gs2>` is base64 of the exact
    /// gs2 header sent in client-first), verifies the client proof in
    /// constant time, and emits `v=base64(ServerSignature)`.
    ///
    /// # Errors
    /// [`ScramError`] on a malformed message, a nonce mismatch, or a proof
    /// that does not verify against the stored password.
    pub fn handle_client_final(&mut self, client_final: &[u8]) -> Result<Vec<u8>, ScramError> {
        if self.step != Step::AwaitClientFinal {
            return Err(ScramError::OutOfOrder);
        }

        let proof_marker = b",p=";
        let proof_at =
            find_subslice(client_final, proof_marker).ok_or(ScramError::MalformedClientFinal)?;
        let without_proof = &client_final[..proof_at];
        let proof_b64 = client_final
            .get(proof_at + proof_marker.len()..)
            .ok_or(ScramError::MalformedClientFinal)?;

        let mut fields = without_proof.split(|byte| *byte == b',');
        let channel_field = fields.next().ok_or(ScramError::MalformedClientFinal)?;
        let nonce_field = fields.next().ok_or(ScramError::MalformedClientFinal)?;
        let channel =
            take_attr(channel_field, b'c').map_err(|()| ScramError::MalformedClientFinal)?;
        // RFC 5802 §5.1: the client echoes base64 of the EXACT gs2 header it
        // sent in client-first (which may carry an `a=authzid`), not always
        // base64("n,,") = "biws".
        let expected_channel = BASE64.encode(&self.gs2_header);
        if channel != expected_channel.as_bytes() {
            return Err(ScramError::ChannelBindingUnsupported);
        }
        let nonce = take_attr(nonce_field, b'r').map_err(|()| ScramError::MalformedClientFinal)?;
        if nonce.ct_eq(self.full_nonce.as_slice()).unwrap_u8() == 0 {
            return Err(ScramError::NonceMismatch);
        }

        let client_proof = BASE64
            .decode(proof_b64)
            .ok()
            .and_then(|bytes| <[u8; 32]>::try_from(bytes).ok())
            .ok_or(ScramError::MalformedProof)?;

        let keys = scram_keys(
            &self.salted_password,
            &self.client_first_bare,
            &self.server_first,
            without_proof,
        );

        let recovered_client_key = xor32(&client_proof, &keys.client_signature);
        let recovered_stored_key = sha256(&recovered_client_key);
        if recovered_stored_key.ct_eq(&keys.stored_key).unwrap_u8() == 0 {
            return Err(ScramError::ProofMismatch);
        }

        self.step = Step::Done;
        let mut server_final = Vec::with_capacity(2 + 44);
        server_final.extend_from_slice(b"v=");
        server_final.extend_from_slice(BASE64.encode(keys.server_signature).as_bytes());
        Ok(server_final)
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn rfc7677_worked_example_reproduces_proof_and_server_signature() {
        // RFC 7677 §3: user="user", password="pencil",
        // client-nonce="rOprNGfwEbeRWgbNEkqO", server appends nonce, i=4096
        let salt = BASE64.decode("W22ZaJ0SNY7soEsUEjb6gQ==").expect("salt b64");
        let salted = salted_password("pencil", &salt, 4096).expect("saltedpassword");

        let client_first_bare = b"n=user,r=rOprNGfwEbeRWgbNEkqO";
        let server_first =
            b"r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096";
        let client_final_without_proof =
            b"c=biws,r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0";

        let keys = scram_keys(
            &salted,
            client_first_bare,
            server_first,
            client_final_without_proof,
        );

        let client_key = hmac_sha256(&salted, b"Client Key");
        let client_proof = xor32(&client_key, &keys.client_signature);
        assert_eq!(
            BASE64.encode(client_proof),
            "dHzbZapWIk4jUhN+Ute9ytag9zjfMHgsqmmiz7AndVQ=",
            "client proof must match RFC 7677 bit-exact"
        );
        assert_eq!(
            BASE64.encode(keys.server_signature),
            "6rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4=",
            "server signature must match RFC 7677 bit-exact"
        );
    }

    fn run_exchange(server_password: &str, client_password: &str) -> Result<Vec<u8>, ScramError> {
        // a minimal SASLprep-free client that drives a real ScramServer
        let mut server = ScramServer::new(server_password).expect("server");
        let client_nonce = "fyko+d2lbbFgONRv9qkxdawL";
        let client_first = format!("n,,n=user,r={client_nonce}");
        let server_first = server.handle_client_first(client_first.as_bytes())?;

        let server_first_str = String::from_utf8(server_first).expect("utf8");
        let full_nonce = server_first_str
            .strip_prefix("r=")
            .and_then(|rest| rest.split(",s=").next())
            .expect("nonce");
        let salt_b64 = server_first_str
            .split(",s=")
            .nth(1)
            .and_then(|rest| rest.split(",i=").next())
            .expect("salt");
        let salt = BASE64.decode(salt_b64).expect("salt b64");

        let client_final_without_proof = format!("c=biws,r={full_nonce}");
        let salted = salted_password(client_password, &salt, 4096).expect("client salted");
        let keys = scram_keys(
            &salted,
            client_first.strip_prefix("n,,").expect("bare").as_bytes(),
            server_first_str.as_bytes(),
            client_final_without_proof.as_bytes(),
        );
        let client_key = hmac_sha256(&salted, b"Client Key");
        let client_proof = xor32(&client_key, &keys.client_signature);
        let client_final = format!(
            "{client_final_without_proof},p={}",
            BASE64.encode(client_proof)
        );
        server.handle_client_final(client_final.as_bytes())
    }

    #[test]
    fn full_exchange_with_correct_password_succeeds() {
        let server_final = run_exchange("hunter2", "hunter2").expect("auth should succeed");
        let text = String::from_utf8(server_final).expect("utf8");

        assert!(
            text.starts_with("v="),
            "server-final must carry the verifier"
        );
    }

    #[test]
    fn full_exchange_with_wrong_password_fails_proof() {
        let result = run_exchange("hunter2", "wrongpw");

        assert_eq!(
            result,
            Err(ScramError::ProofMismatch),
            "wrong password must fail proof verify"
        );
    }

    #[test]
    fn malformed_client_first_missing_nonce_is_rejected() {
        let mut server = ScramServer::new("pw").expect("server");

        let result = server.handle_client_first(b"n,,n=user");

        assert_eq!(result, Err(ScramError::MalformedClientFirst));
    }

    #[test]
    fn channel_binding_request_is_rejected() {
        let mut server = ScramServer::new("pw").expect("server");

        let result = server.handle_client_first(b"y,,n=user,r=abc");

        assert_eq!(result, Err(ScramError::ChannelBindingUnsupported));
    }

    #[test]
    fn empty_client_first_is_rejected() {
        let mut server = ScramServer::new("pw").expect("server");

        let result = server.handle_client_first(b"");

        assert_eq!(result, Err(ScramError::MalformedClientFirst));
    }

    #[test]
    fn client_final_out_of_order_is_rejected() {
        let mut server = ScramServer::new("pw").expect("server");

        let result = server.handle_client_final(b"c=biws,r=x,p=y");

        assert_eq!(result, Err(ScramError::OutOfOrder));
    }

    fn run_exchange_with_gs2(gs2_header: &str) -> Result<Vec<u8>, ScramError> {
        let mut server = ScramServer::new("hunter2").expect("server");
        let client_nonce = "fyko+d2lbbFgONRv9qkxdawL";
        let client_first = format!("{gs2_header}n=user,r={client_nonce}");
        let server_first = server.handle_client_first(client_first.as_bytes())?;

        let server_first_str = String::from_utf8(server_first).expect("utf8");
        let full_nonce = server_first_str
            .strip_prefix("r=")
            .and_then(|rest| rest.split(",s=").next())
            .expect("nonce");
        let salt_b64 = server_first_str
            .split(",s=")
            .nth(1)
            .and_then(|rest| rest.split(",i=").next())
            .expect("salt");
        let salt = BASE64.decode(salt_b64).expect("salt b64");

        let channel = BASE64.encode(gs2_header);
        let client_final_without_proof = format!("c={channel},r={full_nonce}");
        let salted = salted_password("hunter2", &salt, 4096).expect("salted");
        let bare = format!("n=user,r={client_nonce}");
        let keys = scram_keys(
            &salted,
            bare.as_bytes(),
            server_first_str.as_bytes(),
            client_final_without_proof.as_bytes(),
        );
        let client_key = hmac_sha256(&salted, b"Client Key");
        let client_proof = xor32(&client_key, &keys.client_signature);
        let client_final = format!(
            "{client_final_without_proof},p={}",
            BASE64.encode(client_proof)
        );
        server.handle_client_final(client_final.as_bytes())
    }

    #[test]
    fn authzid_in_gs2_header_round_trips_through_client_final() {
        let server_final =
            run_exchange_with_gs2("n,a=admin,").expect("authzid auth should succeed");
        let text = String::from_utf8(server_final).expect("utf8");

        assert!(
            text.starts_with("v="),
            "server-final must carry the verifier"
        );
    }

    #[test]
    fn no_authzid_biws_path_still_works() {
        let server_final = run_exchange_with_gs2("n,,").expect("no-authzid auth should succeed");
        let text = String::from_utf8(server_final).expect("utf8");

        assert!(
            text.starts_with("v="),
            "server-final must carry the verifier"
        );
    }

    #[test]
    fn mismatched_channel_field_is_rejected() {
        // client sent `n,a=admin,` in client-first but echoes base64("n,,")
        let mut server = ScramServer::new("hunter2").expect("server");
        let client_first = b"n,a=admin,n=user,r=fyko+d2lbbFgONRv9qkxdawL";
        let server_first = server.handle_client_first(client_first).expect("first");
        let server_first_str = String::from_utf8(server_first).expect("utf8");
        let full_nonce = server_first_str
            .strip_prefix("r=")
            .and_then(|rest| rest.split(",s=").next())
            .expect("nonce");

        let result = server.handle_client_final(format!("c=biws,r={full_nonce},p=AAAA").as_bytes());

        assert_eq!(result, Err(ScramError::ChannelBindingUnsupported));
    }

    #[test]
    fn scram_client_and_server_complete_a_full_exchange() {
        let mut server = ScramServer::new("hunter2").expect("server");
        let mut client = ScramClient::new("hunter2");

        let client_first = client.client_first();
        let server_first = server
            .handle_client_first(&client_first)
            .expect("server-first");
        let client_final = client.client_final(&server_first).expect("client-final");
        let server_final = server
            .handle_client_final(&client_final)
            .expect("server accepts proof");

        client
            .verify_server_final(&server_final)
            .expect("client accepts server signature");
    }

    #[test]
    fn scram_server_rejects_client_with_wrong_password() {
        let mut server = ScramServer::new("hunter2").expect("server");
        let mut client = ScramClient::new("wrong-password");

        let client_first = client.client_first();
        let server_first = server
            .handle_client_first(&client_first)
            .expect("server-first");
        let client_final = client.client_final(&server_first).expect("client-final");

        assert_eq!(
            server.handle_client_final(&client_final),
            Err(ScramError::ProofMismatch)
        );
    }

    #[test]
    fn scram_client_rejects_a_forged_server_signature() {
        let mut server = ScramServer::new("hunter2").expect("server");
        let mut client = ScramClient::new("hunter2");

        let client_first = client.client_first();
        let server_first = server
            .handle_client_first(&client_first)
            .expect("server-first");
        let _client_final = client.client_final(&server_first).expect("client-final");

        let forged = format!("v={}", BASE64.encode([0_u8; 32]));
        assert_eq!(
            client.verify_server_final(forged.as_bytes()),
            Err(ScramError::ServerSignatureMismatch)
        );
    }

    #[test]
    fn scram_client_rejects_nonce_not_extending_its_own() {
        let mut client = ScramClient::new("hunter2");
        let _first = client.client_first();
        // a server-first whose r= does not start with the client nonce
        let server_first = b"r=totally-different-nonce,s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096";

        assert_eq!(
            client.client_final(server_first),
            Err(ScramError::NonceMismatch)
        );
    }

    #[test]
    fn scram_client_rejects_absurd_iteration_count() {
        let mut client = ScramClient::new("hunter2");
        let first = client.client_first();
        let nonce = core::str::from_utf8(&first)
            .expect("utf8")
            .strip_prefix("n,,n=,r=")
            .expect("nonce")
            .to_string();
        // a hostile server demanding a PBKDF2 cost far beyond any real server
        let server_first =
            format!("r={nonce}server,s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4000000000").into_bytes();

        assert_eq!(
            client.client_final(&server_first),
            Err(ScramError::MalformedServerFirst)
        );
    }

    #[test]
    fn scram_client_drives_through_the_generic_handshake_trait() {
        use proxima_auth::Handshake;

        let mut server = ScramServer::new("hunter2").expect("server");
        let mut client = ScramClient::new("hunter2");

        // pgwire SCRAM as auth form #4, through the protocol-agnostic trait:
        let client_first = Handshake::first(&mut client);
        let server_first = server
            .handle_client_first(&client_first)
            .expect("server-first");
        let client_final = Handshake::step(&mut client, &server_first)
            .expect("step ok")
            .expect("client-final produced");
        let server_final = server
            .handle_client_final(&client_final)
            .expect("server accepts proof");
        assert!(
            Handshake::step(&mut client, &server_final)
                .expect("verify ok")
                .is_none(),
            "handshake complete after server-final verification"
        );
    }
}
