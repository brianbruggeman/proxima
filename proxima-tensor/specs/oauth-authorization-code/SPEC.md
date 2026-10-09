# oauth-authorization-code

status: implemented (acceptance execution recorded)
owner: brian
created: 2026-10-09

## problem

Proxima's existing `client-auth` OAuth option only uses the client-credentials grant; applications that need a person to authorize access cannot run an authorization-code flow with per-transaction PKCE and optional browser launch, measurable by the auth FSM's success, rejection, and configuration test counts.

## refutation condition

If callers cannot complete a provider authorization-code transaction using the existing `proxima-auth` sans-IO boundary and a user-controlled consent page, or if enabling browser launch is required for manual URL-based authorization, this is the wrong shape.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | The auth FSM creates a fresh transaction-specific PKCE verifier, S256 challenge, and state, and exposes an authorization request for `response_type=code`; it never places the verifier in that request. | yes |
| R2 | The core accepts one callback code from the configured redirect, rejects missing/mismatched/replayed state, duplicate response fields, provider errors, malformed or over-16-KiB callbacks, and refuses token exchange unless a code and matching transaction exist. | yes |
| R3 | The token exchange request carries the authorization code, exact redirect URI, client ID, and verifier; successful token results can enter the existing credential lifecycle without exposing credentials, verifier, or code through debug output. | yes |
| R4 | Browser launch is an optional std-tier `conflaguration` setting. Disabled/manual mode returns the authorization URL without launching a browser; enabled mode delegates opening to an injected browser edge, never to the sans-IO FSM. | yes |
| R5 | Existing client-credentials OAuth behavior and feature tiers remain available; authorization-code support is independently selectable and has a no_std + alloc core build. | yes |
| R6 | Callback URI validation and PKCE use RFC 9700 / RFC 7636 security requirements: the client sends the exact registered redirect URI, uses S256 only and transaction-specific verifier/state, consumes the authorization code once, and validates RFC 9207 `iss` when an expected issuer is configured. | yes |

## architecture

Add an authorization-code transaction FSM to `proxima-auth`, with explicit states for awaiting callback, exchanging, completed, denied, and failed. The core accepts two independent caller-provided 32-byte CSPRNG draws (one for the verifier and one for state) and pins the redirect/client/authorization/token endpoint values for the transaction; it performs no I/O or clock reads. The FSM yields a typed authorization request and then a one-use typed token request. The token request consumes itself when producing its wiping form body; code and verifier are not exposed through getters. The core parses the full callback URI, binds its destination to the exact configured redirect prefix and delimiter, and rejects malformed encoding and duplicate recognized response fields. Provider consent is performed by the provider's authorization page; the user's accept/deny action arrives as the redirect callback. A network failure after the exchange request is issued consumes the authorization code and does not permit automatic replay. This API is a public-client flow and does not claim confidential-client authentication.

The optional browser integration belongs at the std-tier consumer boundary. A `conflaguration` settings value selects manual URL presentation or browser launch. Browser opening is injected as a caller-owned edge so tests and non-desktop hosts do not require ambient OS behavior. The callback parser caps the full URI at 16 KiB before decoding and does not decode unrecognized parameter values. The HTTP edge must POST the wiping form body over authenticated TLS, avoid forwarding it across origins, and avoid retrying an exchange after an ambiguous result. The core remains usable without browser, networking, or `conflaguration` dependencies.

The existing `ClientAuthScheme::Oauth` remains the client-credentials grant. The new user-mediated grant gets a distinct scheme/API to avoid changing existing configuration semantics. Long-lived token refresh uses the existing `TokenLifecycle`; the authorization transaction itself is separate and one-shot.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| OAuth role | authorization-code client flow | the provider owns the authorization and consent page; Proxima receives the code after the user accepts |
| PKCE placement | verifier is retained only in the local transaction and sent only in token exchange | placing the verifier in the authorization URL defeats code binding |
| exchange dispatch | token request extraction consumes the transaction into an `Exchanging` state | leaving a ready state after producing the request would allow the same one-time code to be dispatched repeatedly |
| browser ownership | optional injected std-tier edge | the sans-IO crate has no host/browser authority, and callers need a manual URL path |
| relation to existing OAuth | new grant, preserving client credentials | a user's browser consent cannot be represented by the existing background token fetch |
| secret diagnostics | credential, transaction, and token-request debug output redacts secret values | derived `Debug` on `Credential` currently prints bearer/signature strings and propagates them through token lifecycle types |
| issuer binding | optional expected issuer is checked against a unique callback `iss` field | RFC 9700 requires issuer identification or distinct redirect URIs when multiple authorization servers are supported |

### admission

The revised version received ADMIT after adding expected-issuer binding, bounded callback parsing, and the consuming token-body API. Implementation and acceptance-command execution remain in progress.

## acceptance criteria

Each AC is a command and its expected output, including the count. Each names the requirement it discharges.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo nextest run -p proxima-auth --features oauth-authorization-code rfc_7636_appendix_b_verifier_and_challenge_match` | 1 test compares the actual verifier/challenge with RFC 7636 Appendix B's static pair using the reference assertion; actual request uses S256 and excludes the verifier |
| AC2 | R1, R2 | `cargo nextest run -p proxima-auth --features oauth-authorization-code authorization_code::` | at least 10 authorization-code unit tests pass, 0 fail; actual authorization fields are compared with static RFC 6749 field expectations, with callback one-use, exchange failure, and lifecycle handoff assertions |
| AC3 | R2 | `cargo nextest run -p proxima-auth --features oauth-authorization-code callback_rejects` | 1 test compares parsed RFC 6749 §4.1.2 callback/error fields and actual outcomes against fixed expectations for wrong state/destination, denial, missing code, duplicates, malformed encoding, and over-limit input |
| AC4 | R6 | `cargo nextest run -p proxima-auth --features oauth-authorization-code configured_issuer` | 1 test compares the actual callback `iss` against the RFC 9207 expected-issuer fixture and rejects missing, mismatched, or duplicate issuer values |
| AC5 | R4 | `cargo nextest run -p proxima-auth --features oauth-browser browser_cases` | 2 browser tests compare both static mode rows `(launch=false, calls=0)` and `(launch=true, calls=1)` against the actual injected opener observations |
| AC6 | R5 | `cargo nextest run -p proxima-patterns --features middleware oauth` | 4 existing client-credentials OAuth middleware cases pass, 0 fail; captured incumbent bearer-injection outputs satisfy their existing assertions |
| AC7 | R5 | `cargo nextest run -p proxima-auth --no-default-features --features alloc,oauth-authorization-code authorization_code::` | at least 10 authorization-code tests pass, 0 fail with no std/browser/config dependency; the same RFC 7636 and RFC 6749 static references are compared against actual outputs at this tier |
| AC8 | R1 | `cargo nextest run -p proxima-auth --features oauth-authorization-code distinct_entropy_produces_distinct` | 1 test compares deterministic verifier and state values against static encoded rows and confirms the actual S256 challenges differ |
| AC9 | R3 | `cargo nextest run -p proxima-auth --features oauth-authorization-code debug_output_redacts` | 1 test compares actual credential debug strings with the fixed redaction format and confirms lifecycle, transaction, authorization request/response, and token request outputs omit secret values |
| AC10 | R5 | `bash scripts/proxima-auth-gate.sh` | 22 gate cells pass, 0 fail; includes the four existing client-credentials OAuth middleware tests as a regression control |

## out of scope

- Implementing an OAuth authorization server or consent UI; the provider renders consent and returns an authorization response.
- OpenID Connect ID-token parsing/validation, refresh-token rotation, device authorization, implicit flow, password grant, and provider discovery.
- A built-in OS-specific browser opener or redirect listener; those remain consumer-provided I/O edges.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| random source and transaction ownership are underspecified | medium | predictable verifier/state or cross-session mix-up | make randomness and transaction identity explicit in the API and test distinct generated transactions |
| automatic browser launch is unavailable in headless or restricted hosts | high | interactive setup cannot start automatically | manual mode is first-class; launch is injected and optional |
| grant APIs get confused with existing client credentials | medium | existing config can change behavior | preserve the current `oauth` scheme and name the new grant distinctly |

## context

- `proxima-auth/src/lib.rs:1-25` defines the sans-IO FSM-in-the-middle boundary and caller-owned I/O edges.
- `proxima-auth/src/token.rs:118-217` provides the reusable token lifecycle, including refresh-ahead and single-flight behavior.
- `proxima-patterns/src/middleware/client_auth.rs:61-100,540-587` defines existing OAuth as a distinct client-credentials config and pipe.
- `proxima-auth/Cargo.toml:22-37` keeps auth forms default-off and supports no_std + alloc.
- RFC 7636 §§4.1-4.6 and Appendix B define PKCE verifier/challenge construction, S256, and an independent concrete vector; RFC 9700 §§2.1.1 and 4.5 require PKCE protection for authorization codes and recommend S256.
- RFC 6749 §§4.1 and 10.6 defines authorization-code exchange and one-use authorization-code handling.
