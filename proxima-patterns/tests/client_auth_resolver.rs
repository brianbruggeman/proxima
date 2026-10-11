use std::future::Future;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use proxima_auth::Credential;
use proxima_core::ProximaError;
use proxima_patterns::middleware::ClientAuthConfig;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::pipe::request::{Request, Response};

struct Capture {
    calls: Arc<Mutex<usize>>,
    seen: Arc<Mutex<Option<String>>>,
}

impl SendPipe for Capture {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let calls = self.calls.clone();
        let seen = self.seen.clone();
        let value = request
            .metadata
            .get_str("authorization")
            .map(str::to_string);
        async move {
            increment(&calls)?;
            *seen.lock().map_err(poisoned)? = value;
            Ok(Response::new(200))
        }
    }
}

#[derive(Clone)]
struct CountingResolver {
    calls: Arc<Mutex<usize>>,
    credential: Option<Credential>,
}

impl SendPipe for CountingResolver {
    type In = String;
    type Out = Credential;
    type Err = ProximaError;

    fn call(
        &self,
        _credential_ref: String,
    ) -> impl Future<Output = Result<Credential, ProximaError>> + Send {
        let calls = self.calls.clone();
        let credential = self.credential.clone();
        async move {
            increment(&calls)?;
            credential.ok_or_else(|| ProximaError::Upstream("resolver failed".into()))
        }
    }
}

fn poisoned<Guard>(err: std::sync::PoisonError<Guard>) -> ProximaError {
    ProximaError::Upstream(format!("test fixture mutex poisoned: {err}"))
}

fn increment(counter: &Mutex<usize>) -> Result<(), ProximaError> {
    *counter.lock().map_err(poisoned)? += 1;
    Ok(())
}

fn request() -> Result<Request<Bytes>, ProximaError> {
    Request::builder().method("GET").path("/").build()
}

#[test]
fn resolver_backed_client_auth_injects_ephemeral_credential() {
    let calls = Arc::new(Mutex::new(0));
    let inner_calls = Arc::new(Mutex::new(0));
    let seen = Arc::new(Mutex::new(None));
    let inner = into_handle(Capture {
        calls: inner_calls.clone(),
        seen: seen.clone(),
    });
    let resolver = CountingResolver {
        calls: calls.clone(),
        credential: Some(Credential::Bearer("ephemeral-secret".into())),
    };
    let config = ClientAuthConfig::resolver("provider/api-key");
    let serialized = serde_json::to_string(&config).expect("serialize resolver config");
    assert!(serialized.contains("provider/api-key"));
    assert!(!serialized.contains("ephemeral-secret"));
    let pipe = config
        .into_resolver_pipe(inner, resolver)
        .expect("build resolver-backed auth");

    let first_request = request().expect("build request");
    futures::executor::block_on(async { pipe.call(first_request).await.expect("call") });

    assert_eq!(*calls.lock().expect("resolver call count"), 1);
    assert_eq!(*inner_calls.lock().expect("inner call count"), 1);
    assert_eq!(
        seen.lock().expect("capture header").as_deref(),
        Some("Bearer ephemeral-secret")
    );
}

#[test]
fn resolver_backed_client_auth_failure_does_not_dispatch_or_serialize_secret() {
    let calls = Arc::new(Mutex::new(0));
    let inner_calls = Arc::new(Mutex::new(0));
    let seen = Arc::new(Mutex::new(None));
    let inner = into_handle(Capture {
        calls: inner_calls.clone(),
        seen: seen.clone(),
    });
    let resolver = CountingResolver {
        calls: calls.clone(),
        credential: None,
    };
    let config = ClientAuthConfig::resolver("provider/api-key");
    let serialized = serde_json::to_string(&config).expect("serialize resolver config");
    assert!(!serialized.contains("ephemeral-secret"));
    let pipe = config
        .into_resolver_pipe(inner, resolver)
        .expect("build resolver-backed auth");

    let failing_request = request().expect("build request");
    let outcome = futures::executor::block_on(async { pipe.call(failing_request).await });

    assert!(outcome.is_err());
    let diagnostic = format!("{outcome:?}");
    assert!(!diagnostic.contains("ephemeral-secret"));
    assert_eq!(*calls.lock().expect("resolver call count"), 1);
    assert_eq!(*inner_calls.lock().expect("inner call count"), 0);
    assert!(seen.lock().expect("capture header").is_none());
}
