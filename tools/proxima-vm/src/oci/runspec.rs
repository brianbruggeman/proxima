use serde::{Deserialize, Serialize};

use crate::oci::error::OciError;
use crate::oci::manifest::RunConfig;

pub const RUN_SPEC_PATH: &str = ".proxima/run.json";
const DEFAULT_PATH: &str = "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
const ROOT: &str = "/";

/// What `oci-init` execs: the image's Entrypoint then Cmd, its Env, and its WorkingDir.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunSpec {
    pub argv: Vec<String>,
    pub env: Vec<String>,
    pub cwd: String,
}

impl RunSpec {
    pub fn from_config(config: &RunConfig) -> Result<Self, OciError> {
        let mut env = config.env.clone().unwrap_or_default();
        if !env.iter().any(|entry| entry.starts_with("PATH=")) {
            env.push(DEFAULT_PATH.to_string());
        }
        let cwd = config.working_dir.as_deref().filter(|directory| !directory.is_empty()).unwrap_or(ROOT).to_string();
        Ok(Self { argv: config.argv()?, env, cwd })
    }

    pub fn to_json(&self) -> Result<Vec<u8>, OciError> {
        serde_json::to_vec(self).map_err(|source| OciError::Json { what: "run spec".into(), source })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn the_whoami_config_becomes_its_entrypoint_env_and_root_cwd() {
        let config = RunConfig { entrypoint: Some(vec!["/whoami".into()]), env: Some(vec!["PATH=/bin".into()]), ..RunConfig::default() };

        let spec = RunSpec::from_config(&config).expect("spec");

        assert_eq!(spec, RunSpec { argv: vec!["/whoami".into()], env: vec!["PATH=/bin".into()], cwd: "/".into() });
    }

    #[test]
    fn a_missing_path_is_supplied_and_the_working_dir_is_honoured() {
        let config = RunConfig { cmd: Some(vec!["/app".into(), "--serve".into()]), working_dir: Some("/srv".into()), ..RunConfig::default() };

        let spec = RunSpec::from_config(&config).expect("spec");

        assert_eq!(spec.argv, ["/app", "--serve"]);
        assert_eq!(spec.cwd, "/srv");
        assert!(spec.env.iter().any(|entry| entry.starts_with("PATH=")));
    }

    #[test]
    fn json_round_trips() {
        let spec = RunSpec { argv: vec!["/a".into()], env: vec!["K=V".into()], cwd: "/".into() };

        let decoded: RunSpec = serde_json::from_slice(&spec.to_json().expect("json")).expect("decode");

        assert_eq!(decoded, spec);
    }
}
