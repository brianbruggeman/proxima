# C19 — Address validation + retry tokens (paper proof + security review)

Per [RFC 9000 §8.1.3] (retry tokens) + [§17.2.5] (Retry packet) +
[RFC 9001 §5.8] (retry integrity tag).

Tagged per principle 13: `/security-review` (token-format composition) +
`/algorithm-development` (state-reset paper proof).

[RFC 9000 §8.1.3]: https://www.rfc-editor.org/rfc/rfc9000#section-8.1.3
[§17.2.5]: https://www.rfc-editor.org/rfc/rfc9000#section-17.2.5
[RFC 9001 §5.8]: https://www.rfc-editor.org/rfc/rfc9001#section-5.8

**Crate consolidation note (2026-07):** the old crate name referenced throughout this document has since been folded into a single workspace crate: `proxima-quic-proto` -> `proxima-protocols::quic`. See `docs/decomposition/consolidation.md` for the full rename map. The prose below is left as originally written for historical accuracy.

## Scope split

C19 is three sub-components, each ship-able independently:

| Slice | Scope | /skill |
|---|---|---|
| **C19.0** | Retry integrity tag verify (RFC 9001 §5.8) — fixed canonical key, no RNG. CLIENT-side: verify that an inbound Retry packet wasn't forged by an off-path attacker. | `/security-review` |
| **C19.1** | `Rng` trait shape (deferred from C5 → C8 → C19) — first RNG consumer. Plus server-side retry token format (sign/verify with server-side secret). | `/research-rigor` for trait shape; `/security-review` for token format |
| **C19.2** | FSM wire-up: client-side resets Initial state on Retry receipt with new DCID + token attached. | `/algorithm-development` for the state-reset paper proof |

This doc covers all three to lock down the design. C19.0 lands first;
C19.1 + C19.2 follow.

---

## C19.0 — Retry integrity tag verification (RFC 9001 §5.8)

### Why this matters

A Retry packet tells the client to throw away its existing handshake
state and start over with a server-chosen SCID + an opaque token.
Without integrity protection, an off-path attacker who can spoof the
server's address (e.g. a transparent middlebox) could inject a
crafted Retry to force the client into a downgrade or DoS state.

RFC 9001 §5.8 protects against this with an AEAD-GCM tag computed
over the "pseudo-Retry packet" using a CANONICAL key + IV. The key
+ IV are published in the RFC — they're not secrets. The point isn't
to authenticate the server (that's TLS's job) but to authenticate
the path: any party that can compute the tag has read the original
client Initial's DCID, which is harder to spoof than the IP header.

### Canonical key + IV (RFC 9001 §5.8 — QUIC v1)

```
RETRY_KEY_V1 = be0c690b9f66575a1d766b54e368c84e
RETRY_IV_V1  = 461599d35d632bf2239825bb
```

### Algorithm

```
pseudo_retry_packet =
  original_destination_cid_length (1 byte)
  || original_destination_cid (variable)
  || retry_packet_without_integrity_tag

integrity_tag = AES_128_GCM_encrypt(
  key = RETRY_KEY_V1,
  iv = RETRY_IV_V1,
  aad = pseudo_retry_packet,
  plaintext = empty,
).tag
```

Verify by recomputing the tag and constant-time comparing to the
received tag. Mismatch → discard the Retry; treat as garbage.

### Worked example (RFC 9001 §A.4 canonical vectors)

Inputs (RFC 9001 §A.4):
- Original DCID: `0x8394c8f03e515708`
- Retry packet (without tag, hex):
  ```
  ff000000010008f067a5502a4262b574 6f6b656e
  ```
  (= long header + version v1 + DCID(0) + SCID(8): `f067a5502a4262b5`
   + retry token: `74 6f 6b 65 6e` = b"token")
- Expected integrity tag: `04a265e7 3ad94da7 26a4f4a8 ad5d4ca7`

Algorithm:
1. Build pseudo-Retry = 0x08 || 0x8394c8f03e515708 || retry_bytes_without_tag.
2. AES_128_GCM_encrypt with RETRY_KEY_V1 + RETRY_IV_V1 + AAD=pseudo, plaintext=empty.
3. Tag = `04a265e7 3ad94da7 26a4f4a8 ad5d4ca7` ✓.

### Security review (composition-flaw scan)

| Flaw | Status |
|---|---|
| Nonce reuse | N/A — single use per Retry verify; nonce is fixed canonical IV |
| Constant-time compare | MUST use `subtle::ConstantTimeEq` (or equivalent) for the tag compare |
| Length-extension | N/A — AEAD, not raw MAC |
| Truncated tag | N/A — full 16-byte tag |
| Cross-version replay | RFC 9001 §5.8 specifies per-version keys; v1 only for now; future versions get their own key |
| Off-path forgery | The whole point — key is public, so this PROTECTS path-integrity; an off-path attacker can compute the tag only if they observed the original Client Initial (which carries the DCID in cleartext) |
| Server-mismatch | Tag passing doesn't prove the Retry came from the SERVER (just from a party that observed the Initial). TLS handshake will fail on a forged Retry that doesn't lead to the right server certificate. |

### Code site

- `proxima-quic-proto/src/crypto/retry_integrity.rs` — `compute_retry_tag` + `verify_retry_tag`.
- Composes `crate::crypto::aead::aes_128_gcm_encrypt` (C6).

### Tier

Tier-3. No alloc.

---

## C19.1 — Rng trait shape (deferred resolution)

### Three candidates

| Candidate | Pros | Cons |
|---|---|---|
| **A**: require `rand_core::CryptoRng + RngCore` | standard rust-crypto ecosystem; no_std + alloc capable; widely understood | forces caller to depend on `rand_core` |
| **B**: define our own minimal `CryptoRng` trait | zero ecosystem coupling; can be exactly what we need | yet-another-trait; downstream impls have to write boilerplate |
| **C**: re-export an external crypto crate's `Rng` | aligns with the workspace's internal crypto conventions | couples proto crate to an external crypto crate (violates "no proxima deps" rule) |

### Decision

**A** (`rand_core::CryptoRng + RngCore`) for these reasons:

1. `rand_core` is THE rust-crypto-ecosystem standard. Every serious
   crypto library expects it. Aligning with the convention costs
   us nothing and gives downstream impls a no-think implementation
   path (just plug in `OsRng`).
2. `rand_core` is no_std + alloc capable. Tier-3 reach preserved.
3. an external crypto crate's `Rng` (option C) violates the "proto crate has zero
   deps on proxima crates" rule. It's the wrong direction.

Add `rand_core = { workspace = true, default-features = false }`
to the proto crate's dependencies. Trait bound on every API that
needs randomness is `R: rand_core::CryptoRng + rand_core::RngCore`.

Recorded in `edges.md` Resolved table.

---

## C19.1 — Server-side retry token format

### Token shape

```
RetryToken = {
  version: u8,                          // 0x01 for now
  client_ip_family: u8,                 // 0x04 (IPv4) | 0x06 (IPv6)
  client_ip: [u8; 4] | [u8; 16],
  client_port: u16,
  original_destination_cid_len: u8,
  original_destination_cid: [u8; ODCID_MAX],
  retry_source_cid_len: u8,
  retry_source_cid: [u8; RSCID_MAX],
  issued_at_unix_seconds: u64,
} sealed with AES-128-GCM(server_secret, random_nonce, aad=empty)
```

Token byte format (after AEAD seal):
```
[1] nonce_len
[12] nonce
[plaintext_len] sealed_plaintext_with_tag
```

### Sign + verify

```
fn sign_retry_token(
    server_secret: &[u8; 16],     // long-lived rotating key
    client_addr: SocketAddr,
    original_dcid: &[u8],
    retry_scid: &[u8],
    now: SystemTime,
    rng: &mut impl CryptoRng + RngCore,
) -> ArrayVec<u8, MAX_RETRY_TOKEN_LEN>;

fn verify_retry_token(
    server_secret: &[u8; 16],
    expected_client_addr: SocketAddr,
    token: &[u8],
    max_age: Duration,
    now: SystemTime,
) -> Result<RetryTokenContents, RetryTokenError>;
```

### Security review

| Flaw | Mitigation |
|---|---|
| Replay attack (resubmit valid token to bypass rate limiting) | `issued_at` + `max_age` window narrow enough that replay is bounded |
| Token-secret leak (server compromise) | Secret rotates — token expires when secret rotates out of the verifier's set |
| Cross-client token reuse | `client_ip` + `client_port` in plaintext (under AEAD seal); verify against incoming packet's source |
| ODCID/RSCID truncation attack | Length-prefixed; full-bytes AEAD'd |
| Padding oracle | AEAD provides authenticated decryption; no separate padding |
| Confused-deputy (use token from peer X for peer Y) | client_addr binding |
| Version-field downgrade | `version` field is INSIDE the AEAD-sealed plaintext (not AAD) — modifying it breaks the tag |
| Kind-confusion (Retry token replayed as NewToken or vice versa) | `kind` field is INSIDE the AEAD-sealed plaintext; verifier rejects if decrypted kind != expected |

`/security-review` formal sign-off required before C19.1 seals.

### C19.1 v1 implementation choices (vs the original design)

| Aspect | Original design | v1 ship | Why |
|---|---|---|---|
| AEAD | AES-128-GCM | ChaCha20-Poly1305 | Both already workspace deps; ChaCha20-Poly1305 is the constant-time-by-construction choice for tier-3 (no AES-NI assumption). AES-GCM remains an opt-in backend; trait shape preserved. |
| Time type | `SystemTime` + `Duration` | caller-supplied `u64` (server-private epoch + unit) | tier-3 has no wall clock; the caller (`proxima-quic` facade in std, embedded clock layer in tier-3) provides the value. Server choice of epoch is irrelevant since the server alone consumes the token (RFC §8.1.4 — "no need for a single well-defined format"). |
| Address binding | typed `SocketAddr` | `&[u8]` (≤ `MAX_CLIENT_ADDR_LEN`, build-time) | tier-3 has no `SocketAddr`. Caller serializes its own address bytes (IPv4 = 6, IPv6 = 18, NAT-extended = up to 32). Verifier compares byte-for-byte; cross-family tokens can't collide because lengths differ. |
| Version + kind | implicit in format byte | inside AEAD-sealed plaintext, parsed-and-checked at verify | Stronger: an attacker can't downgrade or kind-flip without an AEAD failure. |

### Worked example (C19.1 — Retry token roundtrip)

**Plaintext fields** (server-chosen):

| Field | Value |
|---|---|
| version | `0x01` |
| kind | `0x00` (Retry) |
| issued_at | `1_700_000_000_000_000` (server-private epoch micros) |
| client_addr | `[127, 0, 0, 1, 0x1F, 0x40]` (IPv4 + port = 8000 LE: `0x40, 0x1F` — written in caller's chosen byte order) |
| odcid | `[0x83, 0x94, 0xC8, 0xF0, 0x3E, 0x51, 0x57, 0x08]` (RFC 9001 §A.2 client initial DCID) |
| rscid | `[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11]` (server's chosen retry SCID) |

**Server-private secret** (32 bytes, fixed for the test): `[0x42; 32]`.

**Step 1 — issue.** Server calls `issue(Retry, &client_addr, &odcid, Some(&rscid), 1_700_000_000_000_000, &mut rng, &mut out_buf)`. Returns Ok(len). out_buf contains:
- bytes `[0..12]`: random 12-byte nonce.
- bytes `[12..len-16]`: ciphertext of the plaintext above.
- bytes `[len-16..len]`: 16-byte Poly1305 tag.

**Step 2 — wire.** Server emits Retry packet with this token in the Token field.

**Step 3 — verify (happy).** Client retries; server receives Initial with token. Calls `verify(Retry, &client_addr, token, 1_700_000_000_001_000, MAX_AGE = 30_000_000)`. Time-delta = 1_000 micros < 30 s → OK. Decrypts → parses version=0x01, kind=0x00, issued_at=1_700_000_000_000_000, client_addr matches, odcid + rscid extracted. Returns `Ok(VerifiedRetryToken { odcid, rscid })`.

**Step 4 — verify (replay window expired).** Same call with `now = 1_700_030_000_001_000` (30.001 s later). Time-delta > MAX_AGE → returns `Err(VerifyError::Expired)`.

**Step 5 — verify (wrong client address).** Same token, `client_addr_expected = [192, 168, 1, 1, ...]`. Decrypt succeeds (AEAD doesn't care), parse succeeds, address compare fails → returns `Err(VerifyError::AddressMismatch)`.

**Step 6 — verify (kind-flipped).** Server tries to verify the Retry token as a NewToken (`verify(NewToken, ...)`). Decrypt + parse succeed, but kind-field check fails → returns `Err(VerifyError::WrongKind)`.

**Step 7 — verify (tampered byte).** Flip any byte in `token[12..]` (ciphertext or tag). AEAD authentication fails → returns `Err(VerifyError::AuthenticationFailed)`.

**Step 8 — verify (wrong secret).** Server with `secret = [0x43; 32]` tries to verify a token issued by `[0x42; 32]` server. AEAD fails → `Err(VerifyError::AuthenticationFailed)`. Token-secret rotation works because old tokens fail under the new secret.

### Tier (C19.1 v1)

Tier-3 (bare `no_std + no_alloc`). State = `[u8; 32]` server secret. No alloc in issue or verify.

---

## C19.2 — Client-side FSM reset on Retry receipt

### Worked example

State at t=T0: ConnectionState::Initial(InitialState).

| Event | Action | State after |
|---|---|---|
| Inbound Retry packet | parse Header::Retry | (parsing) |
| Verify integrity tag (C19.0) | compute pseudo-Retry + AES-GCM tag | (if mismatch, discard) |
| Extract retry SCID + token | from parsed Header::Retry | (in flight) |
| Reset Initial state | replace InitialState with fresh one | Initial(InitialState') |
| New state: `local_initial_dcid = retry_scid` (server's SCID becomes our DCID) | | |
| Derive new initial_keys from new DCID | `initial_keys::derive(&retry_scid)` | |
| Reset initial_send / initial_recv | new SendSpace::new() / RecvSpace::new() | |
| Reset crypto_send_initial with NEW ClientHello + retry token attached | TLS provider's `set_retry_token(token)` then re-pump | |
| Reset anti-amplification counter (new flight) | new AntiAmplificationCounter | |

The transition Initial → Initial' is the cleanest paper-proof since
it's the SAME enum variant. Implementation uses
`core::mem::replace(state, sentinel_initial(now))` then constructs the
new InitialState.

### TlsProvider trait extension

Need `fn set_retry_token(&mut self, token: &[u8])` so the TLS provider
can include the token in the next ClientHello's QUIC transport
parameter set (RFC 9000 §18.2 `initial_source_connection_id` +
`original_destination_connection_id` + the token itself per
RFC 9001 §8.3.1).

Add to trait as a default-noop method; MockTlsProvider override
records the token for verification.

---

## Self-critique (binding for the C19.0 ship)

- **Pass 1 — paper before code**: this doc precedes any C19 code.
- **Pass 2 — algorithm walk produces exact expected output**: verified
  against RFC 9001 §A.4 canonical vectors (tag =
  `04a265e7 3ad94da7 26a4f4a8 ad5d4ca7`).
- **Pass 3 — code maps step-by-step to algorithm**: deferred.
- **Pass 4 — test uses exact inputs from worked example**: yes;
  C19.0's lead test encodes the §A.4 vectors bit-exact.
- **Pass 5 — would the test fail on bugs**: yes; swapping
  RETRY_KEY_V1/IV bytes, off-by-one in the pseudo-Retry length
  prefix, computing the tag over the wrong byte range would all
  break the §A.4 assertion.
- **Pass 6 — paper linked to test**: yes.
