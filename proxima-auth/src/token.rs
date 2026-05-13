//! Sans-IO token-lifecycle FSM (auth form #3): exchange credentials for a
//! short-lived token, refresh before it expires, coalesce concurrent refreshes.
//!
//! FSM in the middle, Pipe at the edges. This owns validity / expiry /
//! single-flight; the driver performs the token-endpoint call (the exchange
//! edge — itself a Pipe) and the per-request attach (the inject edge). The two
//! methods separate those concerns so refresh-ahead never stalls a caller:
//!
//! - [`TokenLifecycle::needs_fetch`] — single-flight trigger. Returns `true` to
//!   exactly ONE caller when a (re)fetch should start (no token, expired, or
//!   inside the refresh-ahead window); concurrent callers get `false`. The
//!   winner performs the exchange and calls [`TokenLifecycle::set_token`].
//! - [`TokenLifecycle::poll`] — per-request decision: `Use` a still-valid token
//!   (including the old one *during* a refresh-ahead fetch), else `Await`.

use alloc::string::String;

use zeroize::{Zeroize, Zeroizing};

/// Caller-owned monotonic time in milliseconds. Sans-IO never reads the clock
/// (mirrors `proxima_protocols::quic::Instant`) — the edge stamps `now`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct AuthTime(pub u64);

/// Credential material the inject edge attaches to an outbound request. The
/// edge maps it to the protocol form (`Authorization: Bearer …`, a pgwire
/// password message, …).
///
/// The inner secret material zeroizes on drop (F1): a fetched bearer token and
/// a computed signature are sensitive, so they must not linger in freed heap.
/// Because [`Drop`] forbids moving out of a field, read the inner value via
/// [`Credential::secret`] (a clone) rather than destructuring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Credential {
    /// a bearer/access token value
    Bearer(String),
    /// a computed signature credential (e.g. an AWS `SigV4` `Authorization`
    /// value the request-signing edge derived from the request + key)
    Signature(String),
}

impl Credential {
    /// The inner credential string, wrapped in `Zeroizing` so the returned copy
    /// also wipes on drop (the value zeroizes on drop, and you cannot move out
    /// of a `Drop` type, so this is a clone — but a self-wiping one, audit H1).
    #[must_use]
    pub fn secret(&self) -> Zeroizing<String> {
        match self {
            Self::Bearer(value) | Self::Signature(value) => Zeroizing::new(value.clone()),
        }
    }
}

impl Drop for Credential {
    fn drop(&mut self) {
        match self {
            Self::Bearer(secret) | Self::Signature(secret) => zeroize_secret(secret),
        }
    }
}

/// The exact in-place wipe `Credential::drop` performs: overwrite the buffer
/// with zeros (kept tested via `zeroize_clears_secret_in_place` without the
/// hazard of reading freed memory).
fn zeroize_secret(secret: &mut String) {
    secret.zeroize();
}

/// What the inject edge does for one request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenStep {
    /// a valid token is available — attach it and proceed
    Use(Credential),
    /// no usable token right now — park until a fetch completes (single-flight)
    Await,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fetch {
    Idle,
    InFlight,
}

/// Default cool-down a failed fetch imposes before the next caller may retry.
/// Bounds the thundering herd when a token endpoint is down (audit M5): every
/// parked caller wakes on the failure, but only after this window may one of
/// them re-claim the single-flight slot, so a dead IdP is hit at most once per
/// window instead of as fast as the executor reschedules.
pub const DEFAULT_RETRY_BACKOFF_MS: u64 = 1_000;

/// The token-lifecycle state machine.
#[derive(Clone, Debug)]
pub struct TokenLifecycle {
    token: Option<Credential>,
    expires_at: AuthTime,
    refresh_ahead_ms: u64,
    fetch: Fetch,
    retry_backoff_ms: u64,
    /// earliest time a new fetch may start after a failure; `0` = no cool-down.
    next_retry_at: AuthTime,
}

impl TokenLifecycle {
    /// Builds an empty lifecycle. `refresh_ahead_ms` starts a refresh that many
    /// milliseconds before hard expiry so the still-valid token covers the
    /// fetch latency (set `0` to refresh only on expiry). Uses
    /// [`DEFAULT_RETRY_BACKOFF_MS`] for the post-failure cool-down.
    #[must_use]
    pub fn new(refresh_ahead_ms: u64) -> Self {
        Self::with_backoff(refresh_ahead_ms, DEFAULT_RETRY_BACKOFF_MS)
    }

    /// Like [`Self::new`] but with an explicit failed-fetch cool-down. A `0`
    /// backoff restores the retry-immediately behaviour (used in tests that
    /// assert the single-flight slot frees without a wall-clock wait).
    #[must_use]
    pub fn with_backoff(refresh_ahead_ms: u64, retry_backoff_ms: u64) -> Self {
        Self {
            token: None,
            expires_at: AuthTime(0),
            refresh_ahead_ms,
            fetch: Fetch::Idle,
            retry_backoff_ms,
            next_retry_at: AuthTime(0),
        }
    }

    /// Single-flight trigger: `true` for exactly one caller when a (re)fetch
    /// should begin (no token, hard-expired, or within the refresh-ahead
    /// window), none is in flight, AND any post-failure cool-down has elapsed.
    /// The winner performs the exchange edge and then calls [`Self::set_token`]
    /// (or [`Self::fetch_failed`]).
    pub fn needs_fetch(&mut self, now: AuthTime) -> bool {
        if self.fetch == Fetch::InFlight || now < self.next_retry_at {
            return false;
        }
        if self.should_refresh(now) {
            self.fetch = Fetch::InFlight;
            true
        } else {
            false
        }
    }

    /// Per-request decision at `now`: use a still-valid token (the old one is
    /// served during a refresh-ahead fetch), else await one.
    #[must_use]
    pub fn poll(&self, now: AuthTime) -> TokenStep {
        match &self.token {
            Some(credential) if now < self.expires_at => TokenStep::Use(credential.clone()),
            _ => TokenStep::Await,
        }
    }

    /// Record a freshly fetched token; clears the in-flight flag and any
    /// outstanding failure cool-down.
    pub fn set_token(&mut self, token: Credential, expires_at: AuthTime) {
        self.token = Some(token);
        self.expires_at = expires_at;
        self.fetch = Fetch::Idle;
        self.next_retry_at = AuthTime(0);
    }

    /// Mark the in-flight fetch as failed at `now`: frees the single-flight slot
    /// but opens a cool-down so the next retry is deferred by the backoff window
    /// (audit M5). Any previously valid token is left intact.
    pub fn fetch_failed(&mut self, now: AuthTime) {
        self.fetch = Fetch::Idle;
        self.next_retry_at = AuthTime(now.0.saturating_add(self.retry_backoff_ms));
    }

    /// Whether a fetch is currently claimed (in flight). A parked caller uses
    /// this to tell "wait for the winner to publish" from "no winner is coming
    /// this round" (e.g. the previous fetch failed and a cool-down is open), so
    /// it does not park forever waiting for a notify that will not arrive.
    #[must_use]
    pub fn fetch_in_flight(&self) -> bool {
        self.fetch == Fetch::InFlight
    }

    fn should_refresh(&self, now: AuthTime) -> bool {
        match &self.token {
            None => true,
            Some(_) => now.0.saturating_add(self.refresh_ahead_ms) >= self.expires_at.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn bearer(value: &str) -> Credential {
        Credential::Bearer(value.to_string())
    }

    /// F1: `Credential::drop` zeroizes the secret. This tests the exact wipe
    /// function `Drop` invokes — `String::zeroize` overwrites the buffer bytes
    /// with zeros and truncates length to 0 — on a buffer we still own, so no
    /// freed memory is read. (Reading the buffer after the allocation is freed
    /// would be UB and flaky under miri/ASAN; this proves the same code path.)
    #[test]
    fn zeroize_clears_secret_in_place() {
        let mut secret = "super-secret-token-value".to_string();
        let capacity_before = secret.capacity();
        let pointer = secret.as_ptr();
        zeroize_secret(&mut secret);
        assert!(secret.is_empty(), "zeroize truncates the secret");
        assert_eq!(
            secret.capacity(),
            capacity_before,
            "wipe is in place, buffer not reallocated"
        );
        // SAFETY: still own the (now-emptied) allocation; capacity unchanged so
        // the bytes we wrote zeros into are within the live allocation.
        let observed = unsafe { core::slice::from_raw_parts(pointer, capacity_before) };
        assert!(
            observed.iter().all(|byte| *byte == 0),
            "every byte was overwritten with zero"
        );
    }

    #[test]
    fn empty_awaits_and_triggers_one_fetch() {
        let mut flow = TokenLifecycle::new(0);
        assert_eq!(flow.poll(AuthTime(0)), TokenStep::Await, "no token yet");
        assert!(flow.needs_fetch(AuthTime(0)), "first caller wins the fetch");
        assert!(
            !flow.needs_fetch(AuthTime(0)),
            "single-flight: second caller does not"
        );
    }

    #[test]
    fn set_token_serves_until_expiry() {
        let mut flow = TokenLifecycle::new(0);
        let _ = flow.needs_fetch(AuthTime(0));
        flow.set_token(bearer("t1"), AuthTime(1000));
        assert_eq!(flow.poll(AuthTime(500)), TokenStep::Use(bearer("t1")));
        assert_eq!(flow.poll(AuthTime(999)), TokenStep::Use(bearer("t1")));
        assert_eq!(
            flow.poll(AuthTime(1000)),
            TokenStep::Await,
            "expired at the boundary"
        );
    }

    #[test]
    fn expired_triggers_a_new_single_flight_fetch() {
        let mut flow = TokenLifecycle::new(0);
        let _ = flow.needs_fetch(AuthTime(0));
        flow.set_token(bearer("t1"), AuthTime(1000));
        assert!(!flow.needs_fetch(AuthTime(500)), "still fresh, no refetch");
        assert!(flow.needs_fetch(AuthTime(1000)), "expired -> refetch");
        assert!(
            !flow.needs_fetch(AuthTime(1000)),
            "single-flight holds during refetch"
        );
        flow.set_token(bearer("t2"), AuthTime(2000));
        assert_eq!(flow.poll(AuthTime(1500)), TokenStep::Use(bearer("t2")));
    }

    #[test]
    fn refresh_ahead_serves_old_token_during_the_refetch() {
        // refresh 200ms before expiry; token valid until 1000.
        let mut flow = TokenLifecycle::new(200);
        let _ = flow.needs_fetch(AuthTime(0));
        flow.set_token(bearer("t1"), AuthTime(1000));

        assert!(
            !flow.needs_fetch(AuthTime(700)),
            "outside the refresh-ahead window"
        );
        // at 850, within [800,1000): a refresh is triggered...
        assert!(
            flow.needs_fetch(AuthTime(850)),
            "refresh-ahead opens the window"
        );
        // ...but the still-valid old token is served meanwhile — no stall.
        assert_eq!(
            flow.poll(AuthTime(850)),
            TokenStep::Use(bearer("t1")),
            "old token covers the refresh latency"
        );
        assert!(
            !flow.needs_fetch(AuthTime(860)),
            "single-flight: only one refresh"
        );
    }

    #[test]
    fn fetch_failure_lets_the_next_caller_retry() {
        // zero backoff: the single-flight slot frees for an immediate retry.
        let mut flow = TokenLifecycle::with_backoff(0, 0);
        assert!(flow.needs_fetch(AuthTime(0)));
        assert!(!flow.needs_fetch(AuthTime(0)), "in flight");
        flow.fetch_failed(AuthTime(0));
        assert!(
            flow.needs_fetch(AuthTime(0)),
            "failed fetch frees the single-flight slot"
        );
    }

    #[test]
    fn fetch_failure_imposes_a_cool_down_before_the_next_retry() {
        // audit M5: with a backoff, a failed fetch defers the next retry so a
        // dead endpoint is not hammered as fast as callers re-loop.
        let mut flow = TokenLifecycle::with_backoff(0, 500);
        assert!(
            flow.needs_fetch(AuthTime(0)),
            "first caller claims the fetch"
        );
        flow.fetch_failed(AuthTime(100));
        assert!(
            !flow.needs_fetch(AuthTime(200)),
            "inside the cool-down: no retry yet"
        );
        assert!(!flow.needs_fetch(AuthTime(599)), "still cooling down");
        assert!(
            flow.needs_fetch(AuthTime(600)),
            "cool-down elapsed: retry allowed"
        );
    }

    #[test]
    fn a_successful_fetch_clears_a_pending_cool_down() {
        let mut flow = TokenLifecycle::with_backoff(0, 1_000);
        let _ = flow.needs_fetch(AuthTime(0));
        flow.fetch_failed(AuthTime(0));
        let _ = flow.needs_fetch(AuthTime(1_000));
        flow.set_token(bearer("t1"), AuthTime(5_000));
        // the next refresh decision is governed by expiry, not a stale cool-down.
        assert!(
            !flow.needs_fetch(AuthTime(1_001)),
            "fresh token, no cool-down lingering"
        );
    }
}
