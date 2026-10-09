//! Contract cases for the authorization-code flow.

use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::vec;
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;

use crate::{
    AuthTime, AuthorizationCodeConfig, AuthorizationCodeTransaction, AuthorizationError,
    AuthorizationResponse, Credential, MAX_CALLBACK_URI_BYTES, TokenLifecycle, TokenStep,
};

const RFC_VERIFIER: &str = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
const RFC_CHALLENGE: &str = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM";
const CALLBACK_DESTINATION: &str = "https://app.example/callback?from=web";
const EXPECTED_ISSUER: &str = "https://auth.example/tenant";

trait TestResultExt<Value> {
    fn require(self, context: &str) -> Value;
}

impl<Value, Error> TestResultExt<Value> for Result<Value, Error> {
    fn require(self, context: &str) -> Value {
        match self {
            Ok(value) => value,
            Err(_) => panic!("{context}"),
        }
    }
}

impl<Value> TestResultExt<Value> for Option<Value> {
    fn require(self, context: &str) -> Value {
        match self {
            Some(value) => value,
            None => panic!("{context}"),
        }
    }
}

fn assert_matches_reference(actual: &str, expected: &str) {
    assert!(
        actual == expected,
        "value differs from its static reference fixture"
    );
}

fn config() -> AuthorizationCodeConfig {
    AuthorizationCodeConfig {
        authorization_endpoint: "https://auth.example/authorize".to_owned(),
        token_endpoint: "https://auth.example/token".to_owned(),
        client_id: "client id+1".to_owned(),
        redirect_uri: CALLBACK_DESTINATION.to_owned(),
        issuer: None,
        scopes: vec!["openid".to_owned(), "profile".to_owned()],
    }
}

fn transaction() -> AuthorizationCodeTransaction {
    AuthorizationCodeTransaction::new(config(), rfc_verifier_entropy(), [9; 32])
        .require("valid authorization-code config creates a transaction")
}

fn rfc_verifier_entropy() -> [u8; 32] {
    URL_SAFE_NO_PAD
        .decode(RFC_VERIFIER)
        .require("RFC verifier is valid base64url")
        .try_into()
        .require("RFC verifier decodes to 32 entropy bytes")
}

fn callback_uri(state: &str) -> String {
    alloc::format!("{CALLBACK_DESTINATION}&state={state}&code=authorization-code+%2F%2B%3F%26")
}

fn parsed_callback(state: &str) -> AuthorizationResponse {
    AuthorizationResponse::parse(&callback_uri(state), CALLBACK_DESTINATION)
        .require("callback URI is valid")
}

#[test]
fn rfc_7636_appendix_b_verifier_and_challenge_match() {
    let mut transaction = transaction();
    let request = transaction.authorization_request();
    assert_eq!(request.code_challenge_method(), "S256");
    assert_matches_reference(request.code_challenge(), RFC_CHALLENGE);

    let state = request.state().to_owned();
    let token_request = transaction
        .accept_callback(parsed_callback(&state))
        .require("matching callback is accepted");
    let form = token_request
        .form_body()
        .require("encoded token request fits address space");
    let verifier_field = form
        .split('&')
        .find_map(|parameter| parameter.strip_prefix("code_verifier="))
        .require("form body includes the verifier field");
    assert_matches_reference(verifier_field, RFC_VERIFIER);
}

#[test]
fn authorization_url_encodes_parameters_and_keeps_existing_query() {
    let mut settings = config();
    settings.authorization_endpoint = "https://auth.example/authorize?tenant=west".to_owned();
    let transaction = AuthorizationCodeTransaction::new(settings, [7; 32], [9; 32])
        .require("valid transaction is constructed");
    let authorization_request = transaction.authorization_request();
    let url = authorization_request.url();

    assert!(url.starts_with("https://auth.example/authorize?tenant=west&"));
    assert!(url.contains("client_id=client%20id%2B1"));
    assert!(url.contains("redirect_uri=https%3A%2F%2Fapp.example%2Fcallback%3Ffrom%3Dweb"));
    assert!(url.contains("scope=openid%20profile"));
    assert!(url.contains("response_type=code"));
    assert!(url.contains("code_challenge_method=S256"));
    assert!(!url.contains(RFC_VERIFIER));

    let mut malformed_settings = config();
    malformed_settings.authorization_endpoint = "https://auth.example/authorize&".to_owned();
    assert!(matches!(
        AuthorizationCodeTransaction::new(malformed_settings, [7; 32], [9; 32]),
        Err(AuthorizationError::AuthorizationEndpointQuery)
    ));

    let mut empty_value_settings = config();
    empty_value_settings.authorization_endpoint =
        "https://auth.example/authorize?tenant=?".to_owned();
    let empty_value_transaction =
        AuthorizationCodeTransaction::new(empty_value_settings, [7; 32], [9; 32])
            .require("endpoint with an empty query value is valid");
    assert!(
        empty_value_transaction
            .authorization_request()
            .url()
            .starts_with("https://auth.example/authorize?tenant=?&response_type=code&")
    );
}

#[test]
fn callback_rejects_state_destination_denial_missing_code_duplicates_and_bad_escape() {
    let mut wrong_state_transaction = transaction();
    assert!(matches!(
        wrong_state_transaction.accept_callback(parsed_callback("wrong-state")),
        Err(AuthorizationError::CallbackState)
    ));

    let mut missing_state_transaction = transaction();
    let missing_state_callback = alloc::format!("{CALLBACK_DESTINATION}&code=authorization-code");
    assert!(matches!(
        missing_state_transaction.accept_callback(
            AuthorizationResponse::parse(&missing_state_callback, CALLBACK_DESTINATION)
                .require("callback without state is parsed")
        ),
        Err(AuthorizationError::CallbackState)
    ));

    let destination_transaction = transaction();
    let expected_state = destination_transaction
        .authorization_request()
        .state()
        .to_owned();
    let wrong_destination = alloc::format!(
        "https://attacker.example/callback?state={expected_state}&code=authorization-code"
    );
    assert!(matches!(
        AuthorizationResponse::parse(&wrong_destination, CALLBACK_DESTINATION),
        Err(AuthorizationError::CallbackDestination)
    ));

    let mut denied_transaction = transaction();
    let denied_state = denied_transaction
        .authorization_request()
        .state()
        .to_owned();
    let denied_uri =
        alloc::format!("{CALLBACK_DESTINATION}&state={denied_state}&error=access_denied");
    assert!(matches!(
        denied_transaction.accept_callback(
            AuthorizationResponse::parse(&denied_uri, CALLBACK_DESTINATION)
                .require("provider denial callback is parsed")
        ),
        Err(AuthorizationError::AccessDenied)
    ));

    let mut missing_transaction = transaction();
    let missing_state = missing_transaction
        .authorization_request()
        .state()
        .to_owned();
    let missing_code_uri = alloc::format!("{CALLBACK_DESTINATION}&state={missing_state}");
    assert!(matches!(
        missing_transaction.accept_callback(
            AuthorizationResponse::parse(&missing_code_uri, CALLBACK_DESTINATION)
                .require("callback without code is parsed")
        ),
        Err(AuthorizationError::MissingCode)
    ));

    let mut extension_transaction = transaction();
    let extension_state = extension_transaction
        .authorization_request()
        .state()
        .to_owned();
    let extension_uri = alloc::format!(
        "{CALLBACK_DESTINATION}&state={extension_state}&code=authorization-code&provider_hint=blue"
    );
    extension_transaction
        .accept_callback(
            AuthorizationResponse::parse(&extension_uri, CALLBACK_DESTINATION)
                .require("unknown extension parameter is ignored"),
        )
        .require("valid callback with an extension parameter is accepted");

    let mut duplicate_transaction = transaction();
    let duplicate_state = duplicate_transaction
        .authorization_request()
        .state()
        .to_owned();
    let duplicate_uri = alloc::format!(
        "{CALLBACK_DESTINATION}&state={duplicate_state}&state={duplicate_state}&code=authorization-code"
    );
    assert!(matches!(
        duplicate_transaction.accept_callback(
            AuthorizationResponse::parse(&duplicate_uri, CALLBACK_DESTINATION)
                .require("duplicate query fields are parsed for FSM validation")
        ),
        Err(AuthorizationError::MalformedCallback)
    ));

    let malformed_uri = alloc::format!("{CALLBACK_DESTINATION}&state=%Q0&code=authorization-code");
    assert!(matches!(
        AuthorizationResponse::parse(&malformed_uri, CALLBACK_DESTINATION),
        Err(AuthorizationError::MalformedCallback)
    ));

    let oversized_uri = alloc::format!(
        "{CALLBACK_DESTINATION}&state=s&code=c&padding={}",
        "x".repeat(MAX_CALLBACK_URI_BYTES)
    );
    assert!(matches!(
        AuthorizationResponse::parse(&oversized_uri, CALLBACK_DESTINATION),
        Err(AuthorizationError::CallbackTooLarge)
    ));
}

#[test]
fn configured_issuer_requires_a_matching_callback_issuer() {
    let expected_issuer = EXPECTED_ISSUER;
    let mut matching_config = config();
    matching_config.issuer = Some(expected_issuer.to_owned());
    let mut matching_transaction =
        AuthorizationCodeTransaction::new(matching_config, rfc_verifier_entropy(), [9; 32])
            .require("valid configured issuer creates a transaction");
    let matching_state = matching_transaction
        .authorization_request()
        .state()
        .to_owned();
    let matching_uri = alloc::format!(
        "{CALLBACK_DESTINATION}&state={matching_state}&code=authorization-code&iss={expected_issuer}"
    );
    matching_transaction
        .accept_callback(
            AuthorizationResponse::parse(&matching_uri, CALLBACK_DESTINATION)
                .require("matching issuer callback is parsed"),
        )
        .require("matching issuer is accepted");

    let mut missing_transaction_config = config();
    missing_transaction_config.issuer = Some(expected_issuer.to_owned());
    let mut missing_transaction = AuthorizationCodeTransaction::new(
        missing_transaction_config,
        rfc_verifier_entropy(),
        [9; 32],
    )
    .require("valid configured issuer creates a transaction");
    let missing_state = missing_transaction
        .authorization_request()
        .state()
        .to_owned();
    let missing_uri = callback_uri(&missing_state);
    assert!(matches!(
        missing_transaction.accept_callback(
            AuthorizationResponse::parse(&missing_uri, CALLBACK_DESTINATION)
                .require("callback without issuer is parsed")
        ),
        Err(AuthorizationError::CallbackIssuer)
    ));

    let mut mismatching_transaction_config = config();
    mismatching_transaction_config.issuer = Some(expected_issuer.to_owned());
    let mut mismatching_transaction = AuthorizationCodeTransaction::new(
        mismatching_transaction_config,
        rfc_verifier_entropy(),
        [9; 32],
    )
    .require("valid configured issuer creates a transaction");
    let mismatching_state = mismatching_transaction
        .authorization_request()
        .state()
        .to_owned();
    let mismatching_uri = alloc::format!(
        "{CALLBACK_DESTINATION}&state={mismatching_state}&code=authorization-code&iss=https%3A%2F%2Fevil.example"
    );
    assert!(matches!(
        mismatching_transaction.accept_callback(
            AuthorizationResponse::parse(&mismatching_uri, CALLBACK_DESTINATION)
                .require("callback with a different issuer is parsed")
        ),
        Err(AuthorizationError::CallbackIssuer)
    ));
}

#[test]
fn callback_with_registered_redirect_ending_in_ampersand_does_not_parse_path_suffix_as_query() {
    let redirect_uri = "https://client.example/callback&";
    let callback_uri = "https://client.example/callback&state=s&code=c";
    assert!(matches!(
        AuthorizationResponse::parse(callback_uri, redirect_uri),
        Err(AuthorizationError::CallbackDestination)
    ));
}

#[test]
fn callback_query_is_appended_after_registered_redirect_query_value() {
    let redirect_uri = "https://client.example/callback?tenant=?";
    let callback_uri = "https://client.example/callback?tenant=?&state=s&code=c";
    assert!(AuthorizationResponse::parse(callback_uri, redirect_uri).is_ok());
    let malformed_callback_uri = "https://client.example/callback?tenant=?state=s&code=c";
    assert!(matches!(
        AuthorizationResponse::parse(malformed_callback_uri, redirect_uri),
        Err(AuthorizationError::CallbackDestination)
    ));
}

#[test]
fn duplicate_issuer_is_rejected() {
    let expected_issuer = "https://auth.example/tenant";
    let mut settings = config();
    settings.issuer = Some(expected_issuer.to_owned());
    let mut transaction =
        AuthorizationCodeTransaction::new(settings, rfc_verifier_entropy(), [9; 32])
            .require("valid issuer config creates a transaction");
    let state = transaction.authorization_request().state().to_owned();
    let callback = alloc::format!(
        "{CALLBACK_DESTINATION}&state={state}&code=authorization-code&iss={expected_issuer}&iss={expected_issuer}"
    );
    assert!(matches!(
        transaction.accept_callback(
            AuthorizationResponse::parse(&callback, CALLBACK_DESTINATION)
                .require("duplicate issuer fields parse for validation")
        ),
        Err(AuthorizationError::MalformedCallback)
    ));
}

#[test]
fn distinct_entropy_produces_distinct_verifier_challenge_and_state() {
    let first_transaction = AuthorizationCodeTransaction::new(config(), [1; 32], [2; 32])
        .require("first entropy pair creates a transaction");
    let second_transaction = AuthorizationCodeTransaction::new(config(), [3; 32], [4; 32])
        .require("second entropy pair creates a transaction");
    let first_request = first_transaction.authorization_request();
    let second_request = second_transaction.authorization_request();

    assert!(
        first_request.state() != second_request.state(),
        "independent state entropy must produce distinct state"
    );
    assert!(
        first_request.code_challenge() != second_request.code_challenge(),
        "independent verifier entropy must produce distinct challenges"
    );
    assert_matches_reference(
        first_request.state(),
        "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI",
    );
    assert_matches_reference(
        second_request.state(),
        "BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ",
    );
    let first_state = first_request.state().to_owned();
    let second_state = second_request.state().to_owned();
    let mut first_transaction = first_transaction;
    let mut second_transaction = second_transaction;
    let first_token_request = first_transaction
        .accept_callback(parsed_callback(&first_state))
        .require("first callback is accepted");
    let second_token_request = second_transaction
        .accept_callback(parsed_callback(&second_state))
        .require("second callback is accepted");
    let first_form = first_token_request
        .form_body()
        .require("first encoded token request fits address space");
    let second_form = second_token_request
        .form_body()
        .require("second encoded token request fits address space");
    let first_verifier = first_form
        .split('&')
        .find_map(|parameter| parameter.strip_prefix("code_verifier="))
        .require("first form has verifier");
    let second_verifier = second_form
        .split('&')
        .find_map(|parameter| parameter.strip_prefix("code_verifier="))
        .require("second form has verifier");
    assert_matches_reference(
        first_verifier,
        "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE",
    );
    assert_matches_reference(
        second_verifier,
        "AwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwM",
    );
    assert!(
        first_form != second_form,
        "independent verifier entropy must produce distinct forms"
    );
}

#[test]
fn consumed_transaction_rejects_a_second_callback() {
    let mut transaction = transaction();
    let state = transaction.authorization_request().state().to_owned();
    transaction
        .accept_callback(parsed_callback(&state))
        .require("first matching callback is accepted");

    assert!(matches!(
        transaction.accept_callback(parsed_callback(&state)),
        Err(AuthorizationError::InvalidTransition)
    ));
}

#[test]
fn failed_exchange_cannot_be_reused_or_completed() {
    let mut awaiting_transaction = transaction();
    assert!(matches!(
        awaiting_transaction.fail_exchange(),
        Err(AuthorizationError::InvalidTransition)
    ));
    assert!(matches!(
        awaiting_transaction.complete(
            &mut TokenLifecycle::new(0),
            Credential::Bearer("premature".to_owned()),
            AuthTime(100)
        ),
        Err(AuthorizationError::InvalidTransition)
    ));

    let mut transaction = transaction();
    let state = transaction.authorization_request().state().to_owned();
    transaction
        .accept_callback(parsed_callback(&state))
        .require("matching callback is accepted");
    transaction
        .fail_exchange()
        .require("issued exchange can be marked failed");

    assert!(matches!(
        transaction.accept_callback(parsed_callback(&state)),
        Err(AuthorizationError::InvalidTransition)
    ));
    assert!(matches!(
        transaction.complete(
            &mut TokenLifecycle::new(0),
            Credential::Bearer("access".to_owned()),
            AuthTime(100)
        ),
        Err(AuthorizationError::InvalidTransition)
    ));
}

#[test]
fn token_request_exposes_exchange_fields_and_form_encodes_values() {
    let mut transaction = transaction();
    let state = transaction.authorization_request().state().to_owned();
    let request = transaction
        .accept_callback(parsed_callback(&state))
        .require("matching callback is accepted");

    assert_eq!(request.endpoint(), "https://auth.example/token");
    assert_eq!(request.client_id(), "client id+1");
    assert_eq!(request.redirect_uri(), CALLBACK_DESTINATION);

    let form = request
        .form_body()
        .require("encoded token request fits address space");
    assert!(form.contains("grant_type=authorization_code"));
    assert!(form.contains("code=authorization-code+%2F%2B%3F%26"));
    assert!(form.contains("client_id=client+id%2B1"));
    assert!(form.contains("redirect_uri=https%3A%2F%2Fapp.example%2Fcallback%3Ffrom%3Dweb"));
    assert!(form.contains("code_verifier="));
}

#[test]
fn debug_output_redacts_credentials_and_protocol_secrets() {
    let mut transaction = transaction();
    let state = transaction.authorization_request().state().to_owned();
    let request = transaction
        .accept_callback(parsed_callback(&state))
        .require("matching callback is accepted");
    let request_debug = alloc::format!("{request:?}");
    let transaction_debug = alloc::format!("{transaction:?}");
    assert!(!request_debug.contains("authorization-code /+?&"));
    assert!(!request_debug.contains(RFC_VERIFIER));
    assert!(!transaction_debug.contains(RFC_VERIFIER));
    let authorization_request = transaction.authorization_request();
    let authorization_request_debug = alloc::format!("{authorization_request:?}");
    assert!(!authorization_request_debug.contains(&state));
    assert!(!authorization_request_debug.contains("authorization-code /+?&"));
    let authorization_response = parsed_callback(&state);
    let authorization_response_debug = alloc::format!("{authorization_response:?}");
    assert!(!authorization_response_debug.contains(&state));
    assert!(!authorization_response_debug.contains("authorization-code /+?&"));

    let bearer = Credential::Bearer("access-secret".to_owned());
    let signature = Credential::Signature("signature-secret".to_owned());
    let bearer_debug = alloc::format!("{bearer:?}");
    let signature_debug = alloc::format!("{signature:?}");
    assert_matches_reference(&bearer_debug, "Bearer(\"[REDACTED]\")");
    assert_matches_reference(&signature_debug, "Signature(\"[REDACTED]\")");
    assert!(!bearer_debug.contains("access-secret"));
    assert!(!signature_debug.contains("signature-secret"));

    let step = TokenStep::Use(bearer);
    let step_debug = alloc::format!("{step:?}");
    assert!(!step_debug.contains("access-secret"));
    let lifecycle = TokenLifecycle::new(0);
    let lifecycle_debug = alloc::format!("{lifecycle:?}");
    assert!(!lifecycle_debug.contains("access-secret"));
    assert!(!lifecycle_debug.contains("signature-secret"));
}

#[test]
fn accepted_code_completes_token_lifecycle() {
    let mut transaction = transaction();
    let state = transaction.authorization_request().state().to_owned();
    transaction
        .accept_callback(parsed_callback(&state))
        .require("matching callback is accepted");

    let mut lifecycle = TokenLifecycle::new(0);
    transaction
        .complete(
            &mut lifecycle,
            Credential::Bearer("access".to_owned()),
            AuthTime(100),
        )
        .require("successful exchange completes transaction");
    assert_eq!(lifecycle.poll(AuthTime(100)), TokenStep::Await);
    assert_eq!(
        lifecycle.poll(AuthTime(99)),
        TokenStep::Use(Credential::Bearer("access".to_owned()))
    );
}

#[cfg(feature = "oauth-browser")]
mod browser_cases {
    use alloc::borrow::ToOwned;
    use alloc::string::String;
    use core::cell::{Cell, RefCell};

    use crate::{AuthorizationCodeTransaction, BrowserOpener, OAuthBrowserConfig};

    use super::{TestResultExt, config, rfc_verifier_entropy};

    #[derive(Default)]
    struct FakeOpener {
        call_count: Cell<usize>,
        opened_url: RefCell<Option<String>>,
    }

    impl BrowserOpener for FakeOpener {
        type Error = ();

        fn open(&self, url: &str) -> Result<(), Self::Error> {
            self.call_count.set(self.call_count.get() + 1);
            *self.opened_url.borrow_mut() = Some(url.to_owned());
            Ok(())
        }
    }

    fn request() -> crate::AuthorizationRequest {
        AuthorizationCodeTransaction::new(config(), rfc_verifier_entropy(), [9; 32])
            .require("valid transaction creates authorization request")
            .authorization_request()
    }

    #[test]
    fn manual_browser_configuration_does_not_open_browser() {
        let authorization_request = request();
        let opener = FakeOpener::default();
        let browser_config = OAuthBrowserConfig::default();

        browser_config
            .present(&opener, &authorization_request)
            .require("manual presentation does not need browser edge");
        let observed = if opener.call_count.get() == 0 {
            "manual: opener not called"
        } else {
            "manual: opener called"
        };
        super::assert_matches_reference(observed, "manual: opener not called");
    }

    #[test]
    fn enabled_browser_configuration_opens_authorization_url_once() {
        let authorization_request = request();
        let opener = FakeOpener::default();
        let browser_config = OAuthBrowserConfig::builder().launch(true).build();

        browser_config
            .present(&opener, &authorization_request)
            .require("configured browser opener succeeds");
        let observed = if opener.call_count.get() == 1 {
            "enabled: opener called once"
        } else {
            "enabled: opener call count differed"
        };
        super::assert_matches_reference(observed, "enabled: opener called once");
        super::assert_matches_reference(
            opener.opened_url.borrow().as_deref().unwrap_or_default(),
            authorization_request.url(),
        );
    }
}
