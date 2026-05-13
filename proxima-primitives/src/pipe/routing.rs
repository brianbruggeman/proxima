use bytes::Bytes;
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::pipe::handler::PipeHandle;
use crate::pipe::path_pattern::PathPattern;
use crate::pipe::request::Request;

/// Mount entry: pattern + handle + filters. Generic over the handle so
/// `Mount<PipeHandle>` (default) carries a cross-thread pipe while
/// `Mount<ThreadLocalPipeHandle>` carries a per-thread pipe.
#[derive(Clone)]
pub struct Mount<Handle = PipeHandle> {
    pub path: PathPattern,
    pub pipe: Handle,
    pub methods: MethodFilter,
    pub host: HostFilter,
    pub label: Arc<[u8]>,
    /// Mount-site label — set via [`Mount::named`], `App::mount*`'s
    /// `MountTarget::Named(name)` arm, or defaulted to `"anonymous"`. This
    /// is the mount-site's own name, distinct from `label` (the raw pattern
    /// bytes): mounting the same handle at two paths gives it two
    /// `pipe_name`s, one per mount-site.
    pub pipe_name: Arc<str>,
}

// host matcher: optional set of host header values that must match.
// nginx-style `server_name`. case-insensitive; `None` matches any.
// missing Host header matches `None` only.
#[derive(Debug, Clone, Default)]
pub struct HostFilter {
    allowed: Option<Vec<String>>,
}

impl HostFilter {
    #[must_use]
    pub fn any() -> Self {
        Self { allowed: None }
    }

    #[must_use]
    pub fn only<I: IntoIterator<Item = String>>(hosts: I) -> Self {
        let collected: Vec<String> = hosts
            .into_iter()
            .map(|host| host.to_ascii_lowercase())
            .collect();
        if collected.is_empty() {
            Self::any()
        } else {
            Self {
                allowed: Some(collected),
            }
        }
    }

    #[must_use]
    pub fn matches(&self, host_header: Option<&str>) -> bool {
        let Some(allowed) = self.allowed.as_ref() else {
            return true;
        };
        let Some(host) = host_header else {
            return false;
        };
        // strip port suffix per RFC: "example.com:8080" -> "example.com"
        let host_no_port = host
            .split_once(':')
            .map_or(host, |(name, _)| name)
            .to_ascii_lowercase();
        allowed.iter().any(|candidate| candidate == &host_no_port)
    }
}

#[derive(Debug, Clone)]
pub struct MethodFilter {
    allowed: Option<BTreeSet<String>>,
}

impl MethodFilter {
    #[must_use]
    pub fn any() -> Self {
        Self { allowed: None }
    }

    #[must_use]
    pub fn only<I: IntoIterator<Item = String>>(methods: I) -> Self {
        Self {
            allowed: Some(
                methods
                    .into_iter()
                    .map(|method| method.to_uppercase())
                    .collect(),
            ),
        }
    }

    #[must_use]
    pub fn matches(&self, method: &[u8]) -> bool {
        match &self.allowed {
            Some(set) => set
                .iter()
                .any(|allowed| allowed.as_bytes().eq_ignore_ascii_case(method)),
            None => true,
        }
    }
}

// generic over `Handle`: `pipe_name` (not the erased handle) carries the
// printable identity now, so the Debug impl no longer needs a concrete,
// erasure-specific handle type.
impl<Handle> std::fmt::Debug for Mount<Handle> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Mount")
            .field("path", &self.path.raw())
            .field("pipe", &self.pipe_name)
            .field("methods", &self.methods)
            .field("label", &self.label)
            .finish()
    }
}

impl<Handle> std::fmt::Debug for Router<Handle> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Router")
            .field("mounts", &self.mounts)
            .finish()
    }
}

impl<Handle> Mount<Handle> {
    #[must_use]
    pub fn new(pattern: &str, pipe: Handle) -> Self {
        Self {
            path: PathPattern::parse(pattern),
            pipe,
            methods: MethodFilter::any(),
            host: HostFilter::any(),
            label: Arc::from(pattern.as_bytes()),
            pipe_name: Arc::from("anonymous"),
        }
    }

    #[must_use]
    pub fn with_methods(mut self, methods: MethodFilter) -> Self {
        self.methods = methods;
        self
    }

    #[must_use]
    pub fn with_host(mut self, host: HostFilter) -> Self {
        self.host = host;
        self
    }

    /// Set the mount-site label (default `"anonymous"`). `App::mount*`'s
    /// `MountTarget::Named(name)` arm calls this; a `MountTarget::Handle(_)`
    /// mount leaves the default.
    #[must_use]
    pub fn named(mut self, name: impl Into<Arc<str>>) -> Self {
        self.pipe_name = name.into();
        self
    }

    #[must_use]
    pub fn matches(&self, request: &Request<Bytes>) -> bool {
        self.matches_with_params(request).is_some()
    }

    /// Like `matches` but returns the extracted path-pattern parameters
    /// when the mount accepts this request.
    #[must_use]
    pub fn matches_with_params(
        &self,
        request: &Request<Bytes>,
    ) -> Option<std::collections::BTreeMap<String, String>> {
        if !self.methods.matches(request.method.as_bytes()) {
            return None;
        }
        let host_header = request
            .metadata
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(b"host"))
            .and_then(|(_, value)| std::str::from_utf8(value).ok());
        if !self.host.matches(host_header) {
            return None;
        }
        let path_view = std::str::from_utf8(&request.path).unwrap_or("");
        self.path.matches(path_view)
    }
}

#[derive(Clone)]
pub struct Router<Handle = PipeHandle> {
    mounts: Vec<Mount<Handle>>,
}

impl<Handle> Default for Router<Handle> {
    fn default() -> Self {
        Self { mounts: Vec::new() }
    }
}

impl<Handle> Router<Handle> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            mounts: Vec::with_capacity(capacity),
        }
    }

    pub fn add(&mut self, mount: Mount<Handle>) {
        self.mounts.push(mount);
    }

    pub fn remove(&mut self, pattern: &str) -> usize {
        let before = self.mounts.len();
        let needle = pattern.as_bytes();
        self.mounts.retain(|mount| mount.label.as_ref() != needle);
        before - self.mounts.len()
    }

    pub fn replace(&mut self, pattern: &str, mount: Mount<Handle>) -> bool {
        let removed = self.remove(pattern);
        self.mounts.push(mount);
        removed > 0
    }

    #[must_use]
    pub fn route(&self, request: &Request<Bytes>) -> Option<&Mount<Handle>> {
        self.mounts.iter().find(|mount| mount.matches(request))
    }

    /// Find a matching mount and return the extracted path params.
    pub fn route_with_params(
        &self,
        request: &Request<Bytes>,
    ) -> Option<(&Mount<Handle>, std::collections::BTreeMap<String, String>)> {
        for mount in &self.mounts {
            if let Some(params) = mount.matches_with_params(request) {
                return Some((mount, params));
            }
        }
        None
    }

    #[must_use]
    pub fn mounts(&self) -> &[Mount<Handle>] {
        &self.mounts
    }
}

use core::future::Future;

/// Pattern-routed dispatcher. Generic over the handle:
/// `RoutingPipe<PipeHandle>` impls [`Handler`](crate::pipe::handler::Handler);
/// `RoutingPipe<ThreadLocalPipeHandle>` impls
/// [`ThreadLocalHandler`](crate::pipe::handler::ThreadLocalHandler).
pub struct RoutingPipe<Handle = PipeHandle> {
    router: Router<Handle>,
    label: String,
    fallback: Option<Handle>,
}

impl<Handle> RoutingPipe<Handle> {
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            router: Router::new(),
            label: label.into(),
            fallback: None,
        }
    }

    #[must_use]
    pub fn route(mut self, pattern: &str, pipe: Handle) -> Self {
        self.router.add(Mount::new(pattern, pipe));
        self
    }

    #[must_use]
    pub fn route_with_methods(
        mut self,
        pattern: &str,
        pipe: Handle,
        methods: MethodFilter,
    ) -> Self {
        self.router
            .add(Mount::new(pattern, pipe).with_methods(methods));
        self
    }

    #[must_use]
    pub fn fallback(mut self, pipe: Handle) -> Self {
        self.fallback = Some(pipe);
        self
    }

    /// The routing pipe's own label (set at construction, not a mount-site
    /// name — this labels the `RoutingPipe` node itself, e.g. for tracing).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl<Handle> crate::pipe::primitives::SendPipe for RoutingPipe<Handle>
where
    Handle: crate::pipe::handler::Handler + Clone + Send + Sync + 'static,
{
    type In = Request<Bytes>;
    type Out = crate::pipe::request::Response<Bytes>;
    type Err = crate::pipe::ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<crate::pipe::request::Response<Bytes>, crate::pipe::ProximaError>> + Send
    {
        let routed = self.router.route(&request).map(|mount| mount.pipe.clone());
        let fallback = self.fallback.clone();
        async move {
            if let Some(pipe) = routed {
                return crate::pipe::primitives::SendPipe::call(&pipe, request).await;
            }
            if let Some(pipe) = fallback {
                return crate::pipe::primitives::SendPipe::call(&pipe, request).await;
            }
            Ok(crate::pipe::request::Response::not_found().with_body("no route matched"))
        }
    }
}

impl crate::pipe::primitives::Pipe for RoutingPipe<crate::pipe::handler::ThreadLocalPipeHandle> {
    type In = Request<Bytes>;
    type Out = crate::pipe::request::Response<Bytes>;
    type Err = crate::pipe::ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<crate::pipe::request::Response<Bytes>, crate::pipe::ProximaError>> {
        let routed = self.router.route(&request).map(|mount| mount.pipe.clone());
        let fallback = self.fallback.clone();
        async move {
            if let Some(pipe) = routed {
                return crate::pipe::primitives::Pipe::call(&pipe, request).await;
            }
            if let Some(pipe) = fallback {
                return crate::pipe::primitives::Pipe::call(&pipe, request).await;
            }
            Ok(crate::pipe::request::Response::not_found().with_body("no route matched"))
        }
    }
}

// `#[proxima::test]` pulls in the `proxima` dev-dependency, which the
// loom build keeps out of the graph (see
// `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod routing_pipe_tests {
    use super::*;
    use crate::pipe::SendPipe;
    use crate::pipe::handler::into_handle;
    use crate::pipe::request::Response;

    struct Static(&'static str);

    impl SendPipe for Static {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = crate::pipe::ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, crate::pipe::ProximaError>> + Send {
            let body = self.0;
            async move { Ok(Response::ok(bytes::Bytes::from_static(body.as_bytes()))) }
        }
    }

    fn build_request(method: &str, path: &str) -> Request<Bytes> {
        Request::builder()
            .method(method)
            .path(path)
            .build()
            .expect("builder")
    }

    #[proxima::test]
    async fn route_dispatches_to_matching_pipe() {
        let svc = RoutingPipe::new("api")
            .route("/users/{id}", into_handle(Static("users")))
            .route("/posts/{id}", into_handle(Static("posts")));

        let response = SendPipe::call(&svc, build_request("GET", "/users/42"))
            .await
            .expect("call");
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"users");
    }

    #[proxima::test]
    async fn unmatched_path_falls_through_to_fallback() {
        let svc = RoutingPipe::new("api")
            .route("/known", into_handle(Static("known")))
            .fallback(into_handle(Static("fallback")));
        let response = SendPipe::call(&svc, build_request("GET", "/anything-else"))
            .await
            .expect("call");
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"fallback");
    }

    #[proxima::test]
    async fn unmatched_path_without_fallback_returns_404() {
        let svc = RoutingPipe::new("api").route("/known", into_handle(Static("known")));
        let response = SendPipe::call(&svc, build_request("GET", "/missing"))
            .await
            .expect("call");
        assert_eq!(response.status, 404);
    }

    #[proxima::test]
    async fn method_filter_excludes_unmatched_method() {
        let svc = RoutingPipe::new("api").route_with_methods(
            "/users",
            into_handle(Static("users")),
            MethodFilter::only(["GET".into()]),
        );
        let response = SendPipe::call(&svc, build_request("DELETE", "/users"))
            .await
            .expect("call");
        assert_eq!(response.status, 404);
    }
}

// `#[proxima::test]` pulls in the `proxima` dev-dependency, which the
// loom build keeps out of the graph (see
// `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipe::ProximaError;
    use crate::pipe::SendPipe;
    use crate::pipe::handler::into_handle;
    use crate::pipe::request::Response;
    use std::future::Future;

    struct StubPipe(&'static str);

    impl SendPipe for StubPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let body = self.0;
            async move { Ok(Response::ok(body)) }
        }
    }

    fn build_request(method: &str, path: &str) -> Request<Bytes> {
        Request::builder()
            .method(method)
            .path(path)
            .build()
            .expect("builder should succeed")
    }

    #[test]
    fn router_returns_first_matching_mount() {
        let mut router = Router::new();
        router.add(Mount::new("/users/{id}", into_handle(StubPipe("users"))).named("users"));
        router.add(Mount::new("/posts/{id}", into_handle(StubPipe("posts"))).named("posts"));
        let request = build_request("GET", "/users/42");
        let mount = router.route(&request).expect("should match");
        assert_eq!(mount.pipe_name.as_ref(), "users");
    }

    #[test]
    fn method_filter_excludes_unlisted() {
        let mut router = Router::new();
        let mount = Mount::new("/users", into_handle(StubPipe("users")))
            .with_methods(MethodFilter::only(["GET".into(), "POST".into()]));
        router.add(mount);
        assert!(router.route(&build_request("GET", "/users")).is_some());
        assert!(router.route(&build_request("POST", "/users")).is_some());
        assert!(router.route(&build_request("DELETE", "/users")).is_none());
    }

    #[test]
    fn no_match_returns_none() {
        let router: Router = Router::new();
        assert!(router.route(&build_request("GET", "/missing")).is_none());
    }

    #[test]
    fn unnamed_mount_defaults_to_anonymous() {
        let mount = Mount::new("/x", into_handle(StubPipe("x")));
        assert_eq!(mount.pipe_name.as_ref(), "anonymous");
    }
}
