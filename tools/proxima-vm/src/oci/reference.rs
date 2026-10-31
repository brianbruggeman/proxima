use std::fmt;

use crate::oci::digest::Digest;
use crate::oci::error::OciError;

const DOCKER_HUB_ALIAS: &str = "docker.io";
const DOCKER_HUB_REGISTRY: &str = "registry-1.docker.io";
const DOCKER_HUB_NAMESPACE: &str = "library";
const DEFAULT_TAG: &str = "latest";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pin {
    Tag(String),
    Digest(Digest),
}

/// `traefik/whoami:v1.10.2` or `ghcr.io/org/app@sha256:...`; a digest wins over a tag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageRef {
    pub registry: String,
    pub repository: String,
    pub pin: Pin,
}

impl ImageRef {
    /// Docker hub names resolve to `registry-1.docker.io`, and a bare official name gains `library/`.
    ///
    /// ```
    /// use proxima_vm::oci::reference::{ImageRef, Pin};
    ///
    /// let whoami = ImageRef::parse("traefik/whoami:v1.10.2").expect("a hub reference");
    /// assert_eq!(whoami.registry, "registry-1.docker.io");
    /// assert_eq!(whoami.repository, "traefik/whoami");
    /// assert_eq!(whoami.pin, Pin::Tag("v1.10.2".to_string()));
    /// assert_eq!(ImageRef::parse("alpine").expect("official image").repository, "library/alpine");
    /// ```
    pub fn parse(text: &str) -> Result<Self, OciError> {
        let invalid = |reason: &str| OciError::Reference { reference: text.to_string(), reason: reason.to_string() };
        let (name, digest) = match text.split_once('@') {
            Some((name, digest)) => (name, Some(Digest::parse(digest)?)),
            None => (text, None),
        };
        let (path, tag) = split_tag(name);
        let (registry, repository) = split_registry(path);
        if repository.is_empty() || repository.split('/').any(str::is_empty) {
            return Err(invalid("the repository path has an empty component"));
        }
        let pin = digest.map_or_else(|| Pin::Tag(tag.unwrap_or(DEFAULT_TAG).to_string()), Pin::Digest);
        Ok(Self { registry, repository, pin })
    }

    #[must_use]
    pub fn reference(&self) -> String {
        match &self.pin {
            Pin::Tag(tag) => tag.clone(),
            Pin::Digest(digest) => digest.to_string(),
        }
    }

    #[must_use]
    pub fn with_digest(&self, digest: Digest) -> Self {
        Self { pin: Pin::Digest(digest), ..self.clone() }
    }
}

impl fmt::Display for ImageRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.pin {
            Pin::Tag(tag) => write!(formatter, "{}/{}:{tag}", self.registry, self.repository),
            Pin::Digest(digest) => write!(formatter, "{}/{}@{digest}", self.registry, self.repository),
        }
    }
}

fn split_tag(name: &str) -> (&str, Option<&str>) {
    let last_component_start = name.rfind('/').map_or(0, |index| index + 1);
    match name[last_component_start..].rfind(':') {
        Some(offset) => (&name[..last_component_start + offset], Some(&name[last_component_start + offset + 1..])),
        None => (name, None),
    }
}

fn split_registry(path: &str) -> (String, String) {
    let (first, rest) = path.split_once('/').unwrap_or(("", path));
    let names_registry = first.contains('.') || first.contains(':') || first == "localhost";
    match (names_registry, first == DOCKER_HUB_ALIAS) {
        (true, true) => (DOCKER_HUB_REGISTRY.to_string(), hub_repository(rest)),
        (true, false) => (first.to_string(), rest.to_string()),
        (false, _) => (DOCKER_HUB_REGISTRY.to_string(), hub_repository(path)),
    }
}

fn hub_repository(path: &str) -> String {
    if path.contains('/') { path.to_string() } else { format!("{DOCKER_HUB_NAMESPACE}/{path}") }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const WHOAMI_DIGEST: &str = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    #[proxima::test]
    #[case::hub_namespaced_tag("traefik/whoami:v1.10.2", "registry-1.docker.io", "traefik/whoami", "v1.10.2")]
    #[case::hub_official_image_gets_library("alpine:3.20", "registry-1.docker.io", "library/alpine", "3.20")]
    #[case::bare_name_defaults_to_latest("alpine", "registry-1.docker.io", "library/alpine", "latest")]
    #[case::explicit_docker_io("docker.io/traefik/whoami:v1", "registry-1.docker.io", "traefik/whoami", "v1")]
    #[case::other_registry_with_port("localhost:5000/team/app:dev", "localhost:5000", "team/app", "dev")]
    #[case::other_registry_dotted("ghcr.io/org/app:1.2", "ghcr.io", "org/app", "1.2")]
    async fn tags_resolve_to_registry_repository_and_tag(
        #[case] text: &str,
        #[case] registry: &str,
        #[case] repository: &str,
        #[case] tag: &str,
    ) {
        let parsed = ImageRef::parse(text).expect("valid reference");

        assert_eq!(parsed.registry, registry);
        assert_eq!(parsed.repository, repository);
        assert_eq!(parsed.pin, Pin::Tag(tag.to_string()));
    }

    #[test]
    fn a_digest_wins_over_a_tag() {
        let parsed = ImageRef::parse(&format!("traefik/whoami:v1.10.2@{WHOAMI_DIGEST}")).expect("valid reference");

        assert_eq!(parsed.pin, Pin::Digest(Digest::parse(WHOAMI_DIGEST).expect("digest")));
        assert_eq!(parsed.to_string(), format!("registry-1.docker.io/traefik/whoami@{WHOAMI_DIGEST}"));
    }

    #[proxima::test]
    #[case::empty_component("traefik//whoami:v1")]
    #[case::trailing_slash("traefik/:v1")]
    #[case::bad_digest("traefik/whoami@sha256:zz")]
    async fn malformed_references_are_rejected(#[case] text: &str) {
        assert!(ImageRef::parse(text).is_err(), "{text} must be rejected");
    }
}
