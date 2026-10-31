use serde::Deserialize;

use crate::oci::digest::Digest;
use crate::oci::error::OciError;

pub const OCI_INDEX: &str = "application/vnd.oci.image.index.v1+json";
pub const OCI_MANIFEST: &str = "application/vnd.oci.image.manifest.v1+json";
pub const DOCKER_LIST: &str = "application/vnd.docker.distribution.manifest.list.v2+json";
pub const DOCKER_MANIFEST: &str = "application/vnd.docker.distribution.manifest.v2+json";
pub const ACCEPT: &str = "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json";

#[derive(Clone, Debug, Deserialize)]
pub struct Platform {
    pub architecture: String,
    pub os: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Descriptor {
    pub media_type: String,
    pub digest: String,
    pub platform: Option<Platform>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Document {
    pub media_type: Option<String>,
    pub config: Option<Descriptor>,
    #[serde(default)]
    pub layers: Vec<Descriptor>,
    #[serde(default)]
    pub manifests: Vec<Descriptor>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub struct RunConfig {
    #[serde(default)]
    pub entrypoint: Option<Vec<String>>,
    #[serde(default)]
    pub cmd: Option<Vec<String>>,
    #[serde(default)]
    pub env: Option<Vec<String>>,
    #[serde(default)]
    pub working_dir: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ImageConfig {
    #[serde(default)]
    pub config: RunConfig,
}

pub fn parse_document(bytes: &[u8]) -> Result<Document, OciError> {
    serde_json::from_slice(bytes).map_err(|source| OciError::Json { what: "manifest".into(), source })
}

pub fn parse_image_config(bytes: &[u8]) -> Result<ImageConfig, OciError> {
    serde_json::from_slice(bytes).map_err(|source| OciError::Json { what: "image config".into(), source })
}

impl Document {
    #[must_use]
    pub fn is_index(&self) -> bool {
        !self.manifests.is_empty()
    }

    pub fn select(&self, os: &str, architecture: &str) -> Result<Digest, OciError> {
        let chosen = self
            .manifests
            .iter()
            .find(|entry| entry.platform.as_ref().is_some_and(|platform| platform.os == os && platform.architecture == architecture));
        match chosen {
            Some(entry) => Digest::parse(&entry.digest),
            None => Err(OciError::NoPlatform {
                os: os.to_string(),
                architecture: architecture.to_string(),
                available: self
                    .manifests
                    .iter()
                    .filter_map(|entry| entry.platform.as_ref())
                    .map(|platform| format!("{}/{}", platform.os, platform.architecture))
                    .collect::<Vec<_>>()
                    .join(", "),
            }),
        }
    }
}

impl RunConfig {
    pub fn argv(&self) -> Result<Vec<String>, OciError> {
        let argv: Vec<String> = self.entrypoint.iter().chain(self.cmd.iter()).flatten().cloned().collect();
        if argv.is_empty() { Err(OciError::NothingToRun) } else { Ok(argv) }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const INDEX: &str = r#"{"schemaVersion":2,"mediaType":"application/vnd.oci.image.index.v1+json","manifests":[
        {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:1111111111111111111111111111111111111111111111111111111111111111","size":1,"platform":{"architecture":"arm64","os":"linux"}},
        {"mediaType":"application/vnd.oci.image.manifest.v1+json","digest":"sha256:2222222222222222222222222222222222222222222222222222222222222222","size":1,"platform":{"architecture":"amd64","os":"linux"}}]}"#;

    #[test]
    fn an_index_selects_the_linux_amd64_manifest() {
        let document = parse_document(INDEX.as_bytes()).expect("index parses");

        let chosen = document.select("linux", "amd64").expect("amd64 is listed");

        assert!(document.is_index());
        assert_eq!(chosen.as_str(), format!("sha256:{}", "2".repeat(64)));
    }

    #[test]
    fn a_missing_platform_lists_what_the_index_does_offer() {
        let document = parse_document(INDEX.as_bytes()).expect("index parses");

        let error = document.select("linux", "riscv64").expect_err("not listed");

        assert!(error.to_string().contains("linux/arm64, linux/amd64"), "{error}");
    }

    #[test]
    fn entrypoint_then_cmd_make_the_argv() {
        let config = parse_image_config(br#"{"config":{"Entrypoint":["/whoami"],"Cmd":["--port","8080"],"Env":["A=1"],"WorkingDir":"/app"}}"#).expect("config parses");

        assert_eq!(config.config.argv().expect("has argv"), ["/whoami", "--port", "8080"]);
        assert_eq!(config.config.working_dir.as_deref(), Some("/app"));
    }

    #[test]
    fn an_image_with_neither_entrypoint_nor_cmd_has_nothing_to_run() {
        let config = parse_image_config(br#"{"config":{"Env":["A=1"]}}"#).expect("config parses");

        assert!(matches!(config.config.argv(), Err(OciError::NothingToRun)));
    }
}
