//! sans-io oauth 2.0 authorization-code client transaction with pkce s256.
//!
//! the caller supplies independent cryptographic random draws, presents the
//! returned authorization URL, receives the provider redirect, and sends the
//! one-use token request through its chosen HTTP edge. The provider owns the
//! consent page; this module never opens a browser or performs network I/O.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use core::str;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::{Digest as _, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;
use zeroize::{Zeroize, Zeroizing};

use crate::{AuthTime, Credential, TokenLifecycle};

/// maximum callback uri size accepted by the parser, including the destination.
pub const MAX_CALLBACK_URI_BYTES: usize = 16 * 1024;

/// oauth client and endpoint values pinned for a single authorization attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorizationCodeConfig {
    /// provider authorization endpoint, such as `https://id.example/authorize`.
    pub authorization_endpoint: String,
    /// provider token endpoint, such as `https://id.example/token`.
    pub token_endpoint: String,
    /// registered oauth client identifier, such as `desktop-console`.
    pub client_id: String,
    /// expected authorization-server issuer; set when a client supports multiple providers.
    pub issuer: Option<String>,
    /// exact registered callback uri, such as `http://127.0.0.1:49152/callback`.
    pub redirect_uri: String,
    /// granted permission names, such as `openid` and `profile`.
    pub scopes: Vec<String>,
}

/// failures that prevent an authorization transaction from advancing.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum AuthorizationError {
    /// required client or endpoint configuration is empty or malformed.
    #[error("authorization configuration is invalid")]
    InvalidConfiguration,
    /// the authorization endpoint query contains an oauth-reserved or encoded key.
    #[error("authorization endpoint query contains an OAuth-reserved or encoded key")]
    AuthorizationEndpointQuery,
    /// the callback destination does not match the configured redirect uri.
    #[error("callback destination does not match the registered redirect URI")]
    CallbackDestination,
    /// the callback uri exceeds the parser's bounded input size.
    #[error("callback URI exceeds the maximum accepted size")]
    CallbackTooLarge,
    /// the encoded token request length cannot fit in the platform address space.
    #[error("encoded token request length overflows the platform address space")]
    TokenRequestTooLarge,
    /// the callback state is missing or does not belong to this transaction.
    #[error("callback state does not match this transaction")]
    CallbackState,
    /// the callback issuer is missing or differs from the configured issuer.
    #[error("callback issuer does not match this transaction")]
    CallbackIssuer,
    /// the callback contains duplicate or contradictory oauth response fields.
    #[error("callback contains duplicate or contradictory response fields")]
    MalformedCallback,
    /// the callback has neither an authorization code nor a provider error.
    #[error("callback does not contain an authorization code")]
    MissingCode,
    /// the user declined the provider's consent request.
    #[error("provider authorization was denied")]
    AccessDenied,
    /// the provider returned an oauth error other than `access_denied`.
    #[error("provider returned an authorization error")]
    ProviderError,
    /// the transaction is not in a state that permits the requested operation.
    #[error("authorization transaction is in an invalid state")]
    InvalidTransition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TransactionState {
    AwaitingCallback,
    Exchanging,
    Completed,
    Denied,
    Failed,
}

/// the stateful, one-use part of an oauth authorization-code flow.
pub struct AuthorizationCodeTransaction {
    config: AuthorizationCodeConfig,
    state: String,
    challenge: String,
    verifier: Option<Zeroizing<String>>,
    transaction_state: TransactionState,
}

impl fmt::Debug for AuthorizationCodeTransaction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthorizationCodeTransaction")
            .field("config", &self.config)
            .field("state", &"[REDACTED]")
            .field("challenge", &"[REDACTED]")
            .field("verifier", &"[REDACTED]")
            .field("transaction_state", &self.transaction_state)
            .finish()
    }
}

impl AuthorizationCodeTransaction {
    /// starts a transaction from separate 32-byte csprng draws for pkce and state.
    ///
    /// the entropy values must come from a cryptographically secure random
    /// source. The verifier draw is base64url encoded into the RFC 7636 length
    /// range; the state draw is separately encoded to prevent transaction reuse.
    ///
    /// # Errors
    /// returns [`AuthorizationError::InvalidConfiguration`] for empty client,
    /// endpoint, redirect, or scope values, and
    /// [`AuthorizationError::AuthorizationEndpointQuery`] when the authorization
    /// endpoint query is malformed or contains an OAuth-reserved key.
    #[must_use = "handle invalid authorization configuration"]
    pub fn new(
        config: AuthorizationCodeConfig,
        verifier_entropy: [u8; 32],
        state_entropy: [u8; 32],
    ) -> Result<Self, AuthorizationError> {
        validate_config(&config)?;

        let verifier_entropy = Zeroizing::new(verifier_entropy);
        let verifier = Zeroizing::new(URL_SAFE_NO_PAD.encode(&verifier_entropy[..]));
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        let state_entropy = Zeroizing::new(state_entropy);
        let state = URL_SAFE_NO_PAD.encode(&state_entropy[..]);

        Ok(Self {
            config,
            state,
            challenge,
            verifier: Some(verifier),
            transaction_state: TransactionState::AwaitingCallback,
        })
    }

    /// builds the provider authorization url with response type `code` and pkce s256.
    #[must_use]
    pub fn authorization_request(&self) -> AuthorizationRequest {
        let mut url = self.config.authorization_endpoint.clone();

        append_query_parameter(&mut url, "response_type", "code");
        append_query_parameter(&mut url, "client_id", &self.config.client_id);
        append_query_parameter(&mut url, "redirect_uri", &self.config.redirect_uri);
        if !self.config.scopes.is_empty() {
            let scope_value = self.config.scopes.join(" ");
            append_query_parameter(&mut url, "scope", &scope_value);
        }
        append_query_parameter(&mut url, "state", &self.state);
        append_query_parameter(&mut url, "code_challenge", &self.challenge);
        append_query_parameter(&mut url, "code_challenge_method", "S256");

        AuthorizationRequest {
            url,
            state: self.state.clone(),
            code_challenge: self.challenge.clone(),
        }
    }

    /// validates a callback and consumes the transaction into a token request.
    ///
    /// the callback edge parses the response query into `AuthorizationResponse`
    /// and supplies the fixed callback destination without OAuth response fields.
    /// the destination must equal the registered uri exactly. a successful call
    /// can happen once; it moves the FSM to `Exchanging` before returning the
    /// request so the same authorization code cannot be dispatched twice.
    ///
    /// # Errors
    /// returns an error for an invalid destination, state, response shape,
    /// provider denial, missing code, or repeated callback.
    #[must_use = "handle invalid or replayed authorization callbacks"]
    pub fn accept_callback(
        &mut self,
        response: AuthorizationResponse,
    ) -> Result<TokenRequest, AuthorizationError> {
        if self.transaction_state != TransactionState::AwaitingCallback {
            return Err(AuthorizationError::InvalidTransition);
        }
        if response.redirect_uri != self.config.redirect_uri {
            return Err(AuthorizationError::CallbackDestination);
        }
        if response.duplicate_fields {
            return Err(AuthorizationError::MalformedCallback);
        }
        let presented_state = response
            .state
            .as_deref()
            .ok_or(AuthorizationError::CallbackState)?;
        if !constant_time_equal(presented_state.as_bytes(), self.state.as_bytes()) {
            return Err(AuthorizationError::CallbackState);
        }
        if let Some(expected_issuer) = &self.config.issuer
            && response.issuer.as_deref() != Some(expected_issuer.as_str())
        {
            return Err(AuthorizationError::CallbackIssuer);
        }
        if response.error.is_some() && response.code.is_some() {
            return Err(AuthorizationError::MalformedCallback);
        }
        if let Some(provider_error) = response.error {
            self.verifier = None;
            self.transaction_state = if provider_error == "access_denied" {
                TransactionState::Denied
            } else {
                TransactionState::Failed
            };
            return Err(if provider_error == "access_denied" {
                AuthorizationError::AccessDenied
            } else {
                AuthorizationError::ProviderError
            });
        }

        let code = response.code.ok_or(AuthorizationError::MissingCode)?;
        if code.is_empty() {
            return Err(AuthorizationError::MissingCode);
        }

        let verifier = self
            .verifier
            .take()
            .ok_or(AuthorizationError::InvalidTransition)?;
        self.transaction_state = TransactionState::Exchanging;

        Ok(TokenRequest {
            endpoint: self.config.token_endpoint.clone(),
            client_id: self.config.client_id.clone(),
            redirect_uri: self.config.redirect_uri.clone(),
            code,
            verifier,
        })
    }

    /// stores a token result after an exchange request has been issued.
    ///
    /// # Errors
    /// returns [`AuthorizationError::InvalidTransition`] unless the transaction
    /// is currently exchanging a code.
    #[must_use = "handle invalid authorization transaction state"]
    pub fn complete(
        &mut self,
        lifecycle: &mut TokenLifecycle,
        credential: Credential,
        expires_at: AuthTime,
    ) -> Result<(), AuthorizationError> {
        if self.transaction_state != TransactionState::Exchanging {
            return Err(AuthorizationError::InvalidTransition);
        }
        lifecycle.set_token(credential, expires_at);
        self.transaction_state = TransactionState::Completed;
        Ok(())
    }

    /// marks an issued code exchange as unusable after transport or token failure.
    ///
    /// # Errors
    /// returns [`AuthorizationError::InvalidTransition`] unless exchange began.
    #[must_use = "handle invalid authorization transaction state"]
    pub fn fail_exchange(&mut self) -> Result<(), AuthorizationError> {
        if self.transaction_state != TransactionState::Exchanging {
            return Err(AuthorizationError::InvalidTransition);
        }
        self.transaction_state = TransactionState::Failed;
        Ok(())
    }
}

/// a parsed callback response bound to the exact registered redirect uri.
pub struct AuthorizationResponse {
    redirect_uri: String,
    state: Option<String>,
    code: Option<Zeroizing<String>>,
    error: Option<String>,
    issuer: Option<String>,
    duplicate_fields: bool,
}

impl AuthorizationResponse {
    /// parses a complete callback uri and binds it to the configured redirect.
    ///
    /// oauth fields are parsed from the query appended to the exact registered
    /// redirect URI. Duplicate `state`, `code`, or `error` fields and malformed
    /// percent escapes are rejected by the transaction.
    ///
    /// # Errors
    /// returns an error when the callback uri does not begin with the exact
    /// configured redirect URI, uses the wrong response delimiter, or contains
    /// a fragment or malformed percent encoding.
    #[must_use = "handle malformed or oversized authorization callbacks"]
    pub fn parse(
        callback_uri: &str,
        registered_redirect_uri: &str,
    ) -> Result<Self, AuthorizationError> {
        if callback_uri.len() > MAX_CALLBACK_URI_BYTES {
            return Err(AuthorizationError::CallbackTooLarge);
        }
        if callback_uri.contains('#') {
            return Err(AuthorizationError::MalformedCallback);
        }
        let response_delimiter = match registered_redirect_uri.split_once('?') {
            None => "?",
            Some((_, query)) if query.is_empty() || query.ends_with('&') => "",
            Some(_) => "&",
        };
        let callback_suffix = callback_uri
            .strip_prefix(registered_redirect_uri)
            .ok_or(AuthorizationError::CallbackDestination)?;
        let callback_query = callback_suffix
            .strip_prefix(response_delimiter)
            .filter(|query| !query.is_empty())
            .ok_or(AuthorizationError::CallbackDestination)?;

        let mut response = Self {
            redirect_uri: registered_redirect_uri.to_string(),
            state: None,
            code: None,
            error: None,
            issuer: None,
            duplicate_fields: false,
        };

        for parameter in callback_query.split('&') {
            if parameter.is_empty() {
                return Err(AuthorizationError::MalformedCallback);
            }
            let (encoded_name, encoded_value) = match parameter.split_once('=') {
                Some(parts) => parts,
                None => (parameter, ""),
            };
            let name = decode_query_component(encoded_name)?;
            match name.as_str() {
                "state" => {
                    let value = Zeroizing::new(decode_query_component(encoded_value)?);
                    if response.state.replace(value.to_string()).is_some() {
                        response.duplicate_fields = true;
                    }
                }
                "code" => {
                    let value = Zeroizing::new(decode_query_component(encoded_value)?);
                    if response.code.replace(value).is_some() {
                        response.duplicate_fields = true;
                    }
                }
                "error" => {
                    let value = Zeroizing::new(decode_query_component(encoded_value)?);
                    if response.error.replace(value.to_string()).is_some() {
                        response.duplicate_fields = true;
                    }
                }
                "iss" => {
                    let value = Zeroizing::new(decode_query_component(encoded_value)?);
                    if response.issuer.replace(value.to_string()).is_some() {
                        response.duplicate_fields = true;
                    }
                }
                _ => {}
            }
        }

        Ok(response)
    }
}

impl fmt::Debug for AuthorizationResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthorizationResponse")
            .field("redirect_uri", &self.redirect_uri)
            .field("state", &"[REDACTED]")
            .field("code", &"[REDACTED]")
            .field("error", &self.error.as_deref())
            .field("issuer", &self.issuer.as_deref())
            .field("duplicate_fields", &self.duplicate_fields)
            .finish()
    }
}

/// a prepared authorization url. debug output omits its state-bearing url.
pub struct AuthorizationRequest {
    url: String,
    state: String,
    code_challenge: String,
}

impl AuthorizationRequest {
    /// returns the url to present to the user or open through a browser edge.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// returns the transaction state for binding the callback to its user session.
    #[must_use]
    pub fn state(&self) -> &str {
        &self.state
    }

    /// returns the public s256 challenge sent to the authorization endpoint.
    #[must_use]
    pub fn code_challenge(&self) -> &str {
        &self.code_challenge
    }

    /// returns the pkce transformation name sent to the authorization endpoint.
    #[must_use]
    pub const fn code_challenge_method(&self) -> &'static str {
        "S256"
    }
}

impl fmt::Debug for AuthorizationRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthorizationRequest")
            .field("url", &"[REDACTED]")
            .field("state", &"[REDACTED]")
            .field("code_challenge", &"[REDACTED]")
            .finish()
    }
}

/// one-use oauth token request; code and verifier are wiped when dropped.
///
/// the caller-owned http edge must post [`Self::form_body`] over authenticated
/// tls with `application/x-www-form-urlencoded`, and must not forward the body
/// across origins or automatically retry it after an ambiguous exchange.
pub struct TokenRequest {
    endpoint: String,
    client_id: String,
    redirect_uri: String,
    code: Zeroizing<String>,
    verifier: Zeroizing<String>,
}

impl TokenRequest {
    /// returns the configured token endpoint pinned before browser authorization.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// returns the configured oauth client identifier.
    #[must_use]
    pub fn client_id(&self) -> &str {
        &self.client_id
    }

    /// returns the exact redirect uri included in the authorization request.
    #[must_use]
    pub fn redirect_uri(&self) -> &str {
        &self.redirect_uri
    }

    /// encodes the rfc 6749 authorization-code form body in a wiping buffer.
    ///
    /// # Errors
    /// returns [`AuthorizationError::TokenRequestTooLarge`] if the encoded
    /// request length overflows the platform address space.
    #[must_use = "handle token request encoding failure"]
    pub fn form_body(self) -> Result<Zeroizing<String>, AuthorizationError> {
        let parameters = [
            ("grant_type", "authorization_code"),
            ("code", self.code.as_str()),
            ("client_id", self.client_id.as_str()),
            ("redirect_uri", self.redirect_uri.as_str()),
            ("code_verifier", self.verifier.as_str()),
        ];
        let capacity = parameters
            .iter()
            .try_fold(0_usize, |capacity, (name, value)| {
                let name_length = encoded_component_length(name)?;
                let value_length = encoded_component_length(value)?;
                capacity
                    .checked_add(name_length)?
                    .checked_add(1)?
                    .checked_add(value_length)
            })
            .and_then(|length| length.checked_add(parameters.len() - 1))
            .ok_or(AuthorizationError::TokenRequestTooLarge)?;
        let mut body = Zeroizing::new(String::with_capacity(capacity));
        append_form_parameter(&mut body, "grant_type", "authorization_code");
        append_form_parameter(&mut body, "code", &self.code);
        append_form_parameter(&mut body, "client_id", &self.client_id);
        append_form_parameter(&mut body, "redirect_uri", &self.redirect_uri);
        append_form_parameter(&mut body, "code_verifier", &self.verifier);
        Ok(body)
    }
}

impl fmt::Debug for TokenRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TokenRequest")
            .field("endpoint", &self.endpoint)
            .field("client_id", &self.client_id)
            .field("redirect_uri", &self.redirect_uri)
            .field("code", &"[REDACTED]")
            .field("verifier", &"[REDACTED]")
            .finish()
    }
}

fn validate_config(config: &AuthorizationCodeConfig) -> Result<(), AuthorizationError> {
    if config.authorization_endpoint.is_empty()
        || config.token_endpoint.is_empty()
        || config.client_id.is_empty()
        || config.issuer.as_ref().is_some_and(|issuer| {
            issuer.is_empty()
                || issuer.chars().any(char::is_whitespace)
                || !valid_absolute_uri(issuer)
        })
        || config.redirect_uri.is_empty()
        || config.scopes.iter().any(|scope| !valid_scope(scope))
    {
        return Err(AuthorizationError::InvalidConfiguration);
    }
    if !valid_https_endpoint(&config.authorization_endpoint)
        || !valid_https_endpoint(&config.token_endpoint)
    {
        return Err(AuthorizationError::InvalidConfiguration);
    }
    if let Some((_, query)) = config.authorization_endpoint.split_once('?') {
        let reserved = [
            "response_type",
            "client_id",
            "redirect_uri",
            "scope",
            "state",
            "code_challenge",
            "code_challenge_method",
        ];
        for parameter in query.split('&') {
            let name = match parameter.split('=').next() {
                Some(name) => name,
                None => return Err(AuthorizationError::AuthorizationEndpointQuery),
            };
            if name.contains('%') || reserved.contains(&name) {
                return Err(AuthorizationError::AuthorizationEndpointQuery);
            }
        }
    }
    if config.authorization_endpoint.contains('#') {
        return Err(AuthorizationError::AuthorizationEndpointQuery);
    }
    if !config.authorization_endpoint.contains('?') && config.authorization_endpoint.ends_with('&')
    {
        return Err(AuthorizationError::AuthorizationEndpointQuery);
    }
    if config.redirect_uri.contains('#')
        || config.redirect_uri.chars().any(char::is_whitespace)
        || !valid_absolute_uri(&config.redirect_uri)
    {
        return Err(AuthorizationError::InvalidConfiguration);
    }
    Ok(())
}

fn valid_https_endpoint(endpoint: &str) -> bool {
    let Some(authority_and_path) = endpoint.strip_prefix("https://") else {
        return false;
    };
    let authority = match authority_and_path.split(['/', '?', '#']).next() {
        Some(authority) => authority,
        None => return false,
    };
    !authority.is_empty()
        && !authority.contains('@')
        && !authority.chars().any(char::is_whitespace)
        && !endpoint.contains('#')
}

fn valid_scope(scope: &str) -> bool {
    !scope.is_empty()
        && scope.bytes().all(|byte| {
            byte == 0x21 || (0x23..=0x5b).contains(&byte) || (0x5d..=0x7e).contains(&byte)
        })
}

fn valid_absolute_uri(uri: &str) -> bool {
    let Some((scheme, remainder)) = uri.split_once(':') else {
        return false;
    };
    let mut scheme_bytes = scheme.bytes();
    let Some(first_byte) = scheme_bytes.next() else {
        return false;
    };
    first_byte.is_ascii_alphabetic()
        && scheme_bytes
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'))
        && !remainder.is_empty()
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    bool::from(left.ct_eq(right))
}

fn append_query_parameter(url: &mut String, name: &str, value: &str) {
    let separator = match url.split_once('?') {
        None => Some('?'),
        Some((_, query)) if query.is_empty() || query.ends_with('&') => None,
        Some(_) => Some('&'),
    };
    if let Some(separator) = separator {
        url.push(separator);
    }
    append_encoded(url, name, value, false);
}

fn append_form_parameter(body: &mut String, name: &str, value: &str) {
    if !body.is_empty() {
        body.push('&');
    }
    append_encoded(body, name, value, true);
}

fn append_encoded(output: &mut String, name: &str, value: &str, form: bool) {
    append_encoded_component(output, name.as_bytes(), form);
    output.push('=');
    append_encoded_component(output, value.as_bytes(), form);
}

fn append_encoded_component(output: &mut String, bytes: &[u8], form: bool) {
    for byte in bytes {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                output.push(char::from(*byte));
            }
            b' ' if form => output.push('+'),
            _ => {
                output.push('%');
                output.push(hex_digit(byte >> 4));
                output.push(hex_digit(byte & 0x0f));
            }
        }
    }
}

fn encoded_component_length(value: &str) -> Option<usize> {
    value.bytes().try_fold(0_usize, |length, byte| {
        let encoded_length = match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b' ' => 1,
            _ => 3,
        };
        length.checked_add(encoded_length)
    })
}

fn decode_query_component(encoded: &str) -> Result<String, AuthorizationError> {
    let encoded_bytes = encoded.as_bytes();
    let mut decoded_bytes = Zeroizing::new(Vec::with_capacity(encoded_bytes.len()));
    let mut byte_index = 0;
    while byte_index < encoded_bytes.len() {
        match encoded_bytes[byte_index] {
            b'+' => decoded_bytes.push(b' '),
            b'%' => {
                if byte_index + 2 >= encoded_bytes.len() {
                    return Err(AuthorizationError::MalformedCallback);
                }
                let high_nibble = hex_value(encoded_bytes[byte_index + 1])
                    .ok_or(AuthorizationError::MalformedCallback)?;
                let low_nibble = hex_value(encoded_bytes[byte_index + 2])
                    .ok_or(AuthorizationError::MalformedCallback)?;
                decoded_bytes.push((high_nibble << 4) | low_nibble);
                byte_index += 2;
            }
            byte => decoded_bytes.push(byte),
        }
        byte_index += 1;
    }
    let decoded_value = match str::from_utf8(&decoded_bytes) {
        Ok(value) => Ok(value.to_string()),
        Err(_) => Err(AuthorizationError::MalformedCallback),
    };
    decoded_bytes.zeroize();
    decoded_value
}

fn hex_value(character: u8) -> Option<u8> {
    match character {
        b'0'..=b'9' => Some(character - b'0'),
        b'a'..=b'f' => Some(character - b'a' + 10),
        b'A'..=b'F' => Some(character - b'A' + 10),
        _ => None,
    }
}

fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => char::from(b'0' + nibble),
        _ => char::from(b'A' + (nibble - 10)),
    }
}

#[cfg(test)]
#[path = "authorization_code_test_cases.rs"]
mod test_cases;
