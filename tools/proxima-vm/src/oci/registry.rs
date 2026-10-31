use std::path::Path;

use bytes::Bytes;
use http::{Method, Request, Response, StatusCode, Uri, header};
use serde::Deserialize;

use crate::oci::cache;
use crate::oci::digest::Digest;
use crate::oci::error::OciError;
use crate::oci::manifest::{self, RunConfig};
use crate::oci::reference::{ImageRef, Pin};

const MAX_REDIRECTS: usize = 5;
const ERROR_BODY_EXCERPT: usize = 256;

#[derive(Clone, Debug)]
pub struct Layer {
    pub digest: Digest,
    pub media_type: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct PulledImage {
    pub root: Digest,
    pub manifest: Digest,
    pub run: RunConfig,
    pub layers: Vec<Layer>,
}

#[derive(Deserialize)]
struct TokenReply {
    token: Option<String>,
    access_token: Option<String>,
}

/// `fetch` is the registry's transport pipe: one request in, one response out, any host.
pub async fn pull<Fetch>(fetch: &Fetch, cache_root: &Path, image: &ImageRef, os: &str, architecture: &str) -> Result<PulledImage, OciError>
where
    Fetch: AsyncFn(Request<Bytes>) -> Result<Response<Bytes>, OciError>,
{
    let mut token = None;
    let (root, root_bytes) = resolve_root(fetch, cache_root, image, &mut token).await?;
    let document = manifest::parse_document(&root_bytes)?;
    let (manifest_digest, document) = if document.is_index() {
        let chosen = document.select(os, architecture)?;
        let bytes = content(fetch, cache_root, image, "manifests", &chosen, &mut token).await?;
        (chosen, manifest::parse_document(&bytes)?)
    } else {
        (root.clone(), document)
    };
    let config_descriptor = document.config.as_ref().ok_or_else(|| OciError::Json {
        what: "manifest".into(),
        source: serde::de::Error::missing_field("config"),
    })?;
    let config_digest = Digest::parse(&config_descriptor.digest)?;
    let config_bytes = content(fetch, cache_root, image, "blobs", &config_digest, &mut token).await?;
    let run = manifest::parse_image_config(&config_bytes)?.config;
    let mut layers = Vec::with_capacity(document.layers.len());
    for descriptor in &document.layers {
        let digest = Digest::parse(&descriptor.digest)?;
        let bytes = content(fetch, cache_root, image, "blobs", &digest, &mut token).await?;
        layers.push(Layer { digest, media_type: descriptor.media_type.clone(), bytes });
    }
    Ok(PulledImage { root, manifest: manifest_digest, run, layers })
}

async fn resolve_root<Fetch>(fetch: &Fetch, cache_root: &Path, image: &ImageRef, token: &mut Option<String>) -> Result<(Digest, Vec<u8>), OciError>
where
    Fetch: AsyncFn(Request<Bytes>) -> Result<Response<Bytes>, OciError>,
{
    match &image.pin {
        Pin::Digest(digest) => Ok((digest.clone(), content(fetch, cache_root, image, "manifests", digest, token).await?)),
        Pin::Tag(tag) => {
            let url = endpoint(image, "manifests", tag);
            let body = get(fetch, image, &url, Some(manifest::ACCEPT), token).await?;
            let digest = Digest::of(&body);
            cache::write(cache_root, &digest, &body)?;
            Ok((digest, body.to_vec()))
        }
    }
}

async fn content<Fetch>(fetch: &Fetch, cache_root: &Path, image: &ImageRef, kind: &str, digest: &Digest, token: &mut Option<String>) -> Result<Vec<u8>, OciError>
where
    Fetch: AsyncFn(Request<Bytes>) -> Result<Response<Bytes>, OciError>,
{
    if let Some(cached) = cache::read(cache_root, digest)? {
        return Ok(cached);
    }
    let accept = (kind == "manifests").then_some(manifest::ACCEPT);
    let body = get(fetch, image, &endpoint(image, kind, digest.as_str()), accept, token).await?;
    digest.verify(&format!("{kind} {digest} from {}", image.registry), &body)?;
    cache::write(cache_root, digest, &body)?;
    Ok(body.to_vec())
}

fn endpoint(image: &ImageRef, kind: &str, reference: &str) -> String {
    let scheme = if image.registry.starts_with("localhost") || image.registry.starts_with("127.") { "http" } else { "https" };
    format!("{scheme}://{}/v2/{}/{kind}/{reference}", image.registry, image.repository)
}

async fn get<Fetch>(fetch: &Fetch, image: &ImageRef, start: &str, accept: Option<&str>, token: &mut Option<String>) -> Result<Bytes, OciError>
where
    Fetch: AsyncFn(Request<Bytes>) -> Result<Response<Bytes>, OciError>,
{
    let mut url = start.to_string();
    let mut authenticated = false;
    let mut redirects = 0;
    loop {
        let bearer = if url == start { token.as_deref() } else { None };
        let response = fetch(request(&url, accept, bearer)?).await?;
        let status = response.status();
        if status.is_success() {
            return Ok(response.into_body());
        }
        if status == StatusCode::UNAUTHORIZED && !authenticated && url == start {
            *token = Some(authenticate(fetch, image, &url, &response).await?);
            authenticated = true;
            continue;
        }
        if status.is_redirection() && redirects < MAX_REDIRECTS {
            url = redirect_target(&url, &response)?;
            redirects += 1;
            continue;
        }
        return Err(OciError::Status { url, status: status.as_u16(), body: excerpt(response.body()) });
    }
}

fn request(url: &str, accept: Option<&str>, bearer: Option<&str>) -> Result<Request<Bytes>, OciError> {
    let mut builder = Request::builder().method(Method::GET).uri(url);
    if let Some(accept) = accept {
        builder = builder.header(header::ACCEPT, accept);
    }
    if let Some(bearer) = bearer {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {bearer}"));
    }
    builder.body(Bytes::new()).map_err(|error| OciError::Transport { url: url.to_string(), reason: error.to_string() })
}

fn redirect_target(url: &str, response: &Response<Bytes>) -> Result<String, OciError> {
    let location = response
        .headers()
        .get(header::LOCATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| OciError::Transport { url: url.to_string(), reason: format!("{} redirect carries no Location", response.status()) })?;
    if location.contains("://") {
        return Ok(location.to_string());
    }
    let base: Uri = url.parse().map_err(|error: http::uri::InvalidUri| OciError::Transport { url: url.to_string(), reason: error.to_string() })?;
    let authority = base.authority().map_or("", |authority| authority.as_str());
    let scheme = base.scheme_str().unwrap_or("https");
    Ok(format!("{scheme}://{authority}{location}"))
}

async fn authenticate<Fetch>(fetch: &Fetch, image: &ImageRef, url: &str, challenge: &Response<Bytes>) -> Result<String, OciError>
where
    Fetch: AsyncFn(Request<Bytes>) -> Result<Response<Bytes>, OciError>,
{
    let failure = |reason: String| OciError::Auth { url: url.to_string(), reason };
    let header_value = challenge
        .headers()
        .get(header::WWW_AUTHENTICATE)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| failure("401 without a WWW-Authenticate challenge".into()))?;
    let realm = challenge_field(header_value, "realm").ok_or_else(|| failure(format!("challenge `{header_value}` has no realm")))?;
    let service = challenge_field(header_value, "service").unwrap_or_default();
    let scope = challenge_field(header_value, "scope").unwrap_or_else(|| format!("repository:{}:pull", image.repository));
    let token_url = format!("{realm}?service={service}&scope={scope}");
    let response = fetch(request(&token_url, None, None)?).await?;
    if !response.status().is_success() {
        return Err(failure(format!("token endpoint answered {}: {}", response.status(), excerpt(response.body()))));
    }
    let reply: TokenReply = serde_json::from_slice(response.body()).map_err(|error| failure(format!("token reply is not json: {error}")))?;
    reply.token.or(reply.access_token).ok_or_else(|| failure("token reply carries neither `token` nor `access_token`".into()))
}

fn challenge_field(header_value: &str, name: &str) -> Option<String> {
    let marker = format!("{name}=\"");
    let start = header_value.find(&marker)? + marker.len();
    let length = header_value[start..].find('"')?;
    Some(header_value[start..start + length].to_string())
}

fn excerpt(body: &Bytes) -> String {
    let end = body.len().min(ERROR_BODY_EXCERPT);
    String::from_utf8_lossy(&body[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use super::*;

    const LAYER: &[u8] = b"layer-bytes-standing-in-for-a-gzip-tar";
    const CONFIG: &str = r#"{"config":{"Entrypoint":["/whoami"],"Env":["PATH=/usr/bin"]}}"#;

    struct Registry {
        blobs: BTreeMap<String, Vec<u8>>,
        manifest: Digest,
        index: Digest,
        requests: RefCell<Vec<String>>,
    }

    fn registry() -> Registry {
        let config_digest = Digest::of(CONFIG.as_bytes());
        let layer_digest = Digest::of(LAYER);
        let manifest = format!(
            r#"{{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json","config":{{"mediaType":"application/vnd.oci.image.config.v1+json","digest":"{config_digest}","size":1}},"layers":[{{"mediaType":"application/vnd.oci.image.layer.v1.tar+gzip","digest":"{layer_digest}","size":1}}]}}"#
        );
        let manifest_digest = Digest::of(manifest.as_bytes());
        let index = format!(
            r#"{{"schemaVersion":2,"manifests":[{{"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"{manifest_digest}","size":1,"platform":{{"architecture":"amd64","os":"linux"}}}}]}}"#
        );
        let index_digest = Digest::of(index.as_bytes());
        let mut blobs = BTreeMap::new();
        blobs.insert(config_digest.to_string(), CONFIG.as_bytes().to_vec());
        blobs.insert(layer_digest.to_string(), LAYER.to_vec());
        blobs.insert(manifest_digest.to_string(), manifest.into_bytes());
        blobs.insert(index_digest.to_string(), index.into_bytes());
        Registry { blobs, manifest: manifest_digest, index: index_digest, requests: RefCell::new(Vec::new()) }
    }

    impl Registry {
        async fn serve(&self, request: Request<Bytes>) -> Result<Response<Bytes>, OciError> {
            let url = request.uri().to_string();
            self.requests.borrow_mut().push(url.clone());
            let reply = |status: u16, body: Vec<u8>| Response::builder().status(status).body(Bytes::from(body)).expect("response");
            if url.starts_with("https://auth.test/token") {
                return Ok(reply(200, br#"{"token":"abc"}"#.to_vec()));
            }
            if request.headers().get(header::AUTHORIZATION).is_none() {
                return Ok(Response::builder()
                    .status(401)
                    .header(header::WWW_AUTHENTICATE, r#"Bearer realm="https://auth.test/token",service="registry.test""#)
                    .body(Bytes::new())
                    .expect("challenge"));
            }
            if url.contains("/manifests/v1") {
                return Ok(reply(200, self.blobs[self.index.as_str()].clone()));
            }
            let key = url.rsplit('/').next().expect("path tail");
            match self.blobs.get(key) {
                Some(body) => Ok(reply(200, body.clone())),
                None => Ok(reply(404, b"unknown".to_vec())),
            }
        }
    }

    fn image(text: &str) -> ImageRef {
        ImageRef::parse(text).expect("reference")
    }

    #[proxima::test]
    async fn a_tag_resolves_through_the_index_to_config_and_layers() {
        let registry = registry();
        let cache_root = tempfile::tempdir().expect("tempdir");

        let pulled = pull(&async |request| registry.serve(request).await, cache_root.path(), &image("registry.test/team/app:v1"), "linux", "amd64")
            .await
            .expect("pull");

        assert_eq!(pulled.root, registry.index);
        assert_eq!(pulled.manifest, registry.manifest);
        assert_eq!(pulled.run.argv().expect("argv"), ["/whoami"]);
        assert_eq!(pulled.layers.len(), 1);
        assert_eq!(pulled.layers[0].bytes, LAYER);
    }

    #[proxima::test]
    async fn the_second_pull_by_digest_makes_zero_requests() {
        let registry = registry();
        let cache_root = tempfile::tempdir().expect("tempdir");
        let by_tag = image("registry.test/team/app:v1");
        pull(&async |request| registry.serve(request).await, cache_root.path(), &by_tag, "linux", "amd64").await.expect("first pull");
        let first_run_requests = registry.requests.borrow().len();
        let by_digest = by_tag.with_digest(registry.index.clone());

        let pulled = pull(&async |request| registry.serve(request).await, cache_root.path(), &by_digest, "linux", "amd64").await.expect("cached pull");

        assert!(first_run_requests > 0);
        assert_eq!(registry.requests.borrow().len(), first_run_requests, "a cache hit must not touch the network");
        assert_eq!(pulled.layers[0].bytes, LAYER);
    }

    #[proxima::test]
    async fn a_layer_served_with_the_wrong_bytes_is_rejected_with_expected_and_actual() {
        let mut registry = registry();
        let layer_digest = Digest::of(LAYER);
        registry.blobs.insert(layer_digest.to_string(), b"tampered layer".to_vec());
        let cache_root = tempfile::tempdir().expect("tempdir");

        let error = pull(&async |request| registry.serve(request).await, cache_root.path(), &image("registry.test/team/app:v1"), "linux", "amd64")
            .await
            .expect_err("tampered layer");

        let message = error.to_string();
        assert!(message.contains(&format!("expected {layer_digest}")), "{message}");
        assert!(message.contains(&format!("actual {}", Digest::of(b"tampered layer"))), "{message}");
        assert!(cache::read(cache_root.path(), &layer_digest).expect("cache read").is_none(), "a rejected blob must not be cached");
    }

    #[proxima::test]
    async fn a_redirected_blob_is_fetched_from_the_new_host_without_the_registry_token() {
        let registry = registry();
        let layer_digest = Digest::of(LAYER);
        let cache_root = tempfile::tempdir().expect("tempdir");
        let seen = RefCell::new(Vec::new());
        let transport = async |request: Request<Bytes>| {
            let url = request.uri().to_string();
            seen.borrow_mut().push((url.clone(), request.headers().contains_key(header::AUTHORIZATION)));
            if url.ends_with(layer_digest.as_str()) && url.starts_with("https://registry.test") {
                return Ok(Response::builder().status(307).header(header::LOCATION, "https://cdn.test/blob").body(Bytes::new()).expect("redirect"));
            }
            if url == "https://cdn.test/blob" {
                return Ok(Response::new(Bytes::from_static(LAYER)));
            }
            registry.serve(request).await
        };

        let pulled = pull(&transport, cache_root.path(), &image("registry.test/team/app:v1"), "linux", "amd64").await.expect("pull");

        assert_eq!(pulled.layers[0].bytes, LAYER);
        let cdn_hop = seen.borrow().iter().find(|(url, _)| url == "https://cdn.test/blob").cloned().expect("cdn was dialed");
        assert!(!cdn_hop.1, "the pre-signed cdn url must not receive the registry bearer token");
    }

    #[proxima::test]
    async fn a_404_names_the_url_and_status() {
        let registry = registry();
        let cache_root = tempfile::tempdir().expect("tempdir");
        let missing = Digest::of(b"nothing serves this");

        let error = pull(&async |request| registry.serve(request).await, cache_root.path(), &image("registry.test/team/app").with_digest(missing.clone()), "linux", "amd64")
            .await
            .expect_err("unknown manifest");

        assert!(matches!(error, OciError::Status { status: 404, .. }), "{error}");
        assert!(error.to_string().contains(missing.as_str()), "{error}");
    }
}
