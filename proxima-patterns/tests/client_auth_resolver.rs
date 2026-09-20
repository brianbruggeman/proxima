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
            *calls.lock().expect("capture call count") += 1;
            *seen.lock().expect("capture header") = value;
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
            *calls.lock().expect("resolver call count") += 1;
            credential.ok_or_else(|| ProximaError::Upstream("resolver failed".into()))
        }
    }
}

fn request() -> Request<Bytes> {
    Request::builder()
        .method("GET")
        .path("/")
        .build()
        .expect("request")
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

    futures::executor::block_on(async { pipe.call(request()).await.expect("call") });

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

    let outcome = futures::executor::block_on(async { pipe.call(request()).await });

    assert!(outcome.is_err());
    let diagnostic = format!("{outcome:?}");
    assert!(!diagnostic.contains("ephemeral-secret"));
    assert_eq!(*calls.lock().expect("resolver call count"), 1);
    assert_eq!(*inner_calls.lock().expect("inner call count"), 0);
    assert!(seen.lock().expect("capture header").is_none());
}
