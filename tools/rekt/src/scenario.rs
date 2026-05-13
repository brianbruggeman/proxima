use std::time::Duration;

use serde::Deserialize;
use serde_json::Value as JsonValue;
use toml::Value as TomlValue;

use crate::error::Error;

#[derive(Debug, Deserialize)]
struct ScenarioFile {
    target: TomlValue,
    #[serde(default)]
    request: RequestFile,
    #[serde(default)]
    stage: Vec<StageSpec>,
    #[serde(default)]
    threshold: ThresholdSpec,
}

#[derive(Debug, Default, Deserialize)]
struct RequestFile {
    method: Option<String>,
    path: Option<String>,
    body: Option<String>,
    #[serde(default)]
    headers: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    query: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct StageSpec {
    rate: String,
    duration: String,
}

#[derive(Debug, Default, Deserialize)]
struct ThresholdSpec {
    p99: Option<String>,
    error_rate: Option<String>,
}

/// a parsed run: where to hit, the stages to drive, and the pass/fail gates.
pub struct Scenario {
    pub target_url: String,
    pub client_spec: JsonValue,
    pub request: RequestSpec,
    pub stages: Vec<Stage>,
    pub thresholds: Thresholds,
}

#[derive(Clone)]
pub struct RequestSpec {
    pub method: String,
    pub path: String,
    pub body: Option<String>,
    pub headers: std::collections::BTreeMap<String, String>,
    pub query: std::collections::BTreeMap<String, String>,
}

pub struct Stage {
    pub rate_per_sec: f64,
    pub duration: Duration,
}

pub struct Thresholds {
    pub p99: Option<Duration>,
    pub error_rate: Option<f64>,
}

impl Scenario {
    pub fn from_toml(text: &str) -> Result<Scenario, Error> {
        let file: ScenarioFile = toml::from_str(text)?;
        let (target_url, client_spec) = parse_target(file.target)?;
        let mut stages = Vec::with_capacity(file.stage.len());
        for s in &file.stage {
            stages.push(Stage {
                rate_per_sec: parse_rate(&s.rate)?,
                duration: parse_duration(&s.duration)?,
            });
        }
        let thresholds = Thresholds {
            p99: match &file.threshold.p99 {
                Some(s) => Some(parse_duration(s)?),
                None => None,
            },
            error_rate: match &file.threshold.error_rate {
                Some(s) => Some(parse_percent(s)?),
                None => None,
            },
        };
        Ok(Scenario {
            target_url,
            client_spec,
            request: RequestSpec::from(file.request),
            stages,
            thresholds,
        })
    }
}

impl From<RequestFile> for RequestSpec {
    fn from(file: RequestFile) -> Self {
        Self {
            method: file.method.unwrap_or_else(|| "GET".to_string()),
            path: file.path.unwrap_or_else(|| "/".to_string()),
            body: file.body,
            headers: file.headers,
            query: file.query,
        }
    }
}

fn parse_target(value: TomlValue) -> Result<(String, JsonValue), Error> {
    let json = toml_to_json(value);
    let object = json
        .as_object()
        .ok_or_else(|| Error::Config("[target] must be a TOML table".into()))?;

    if let Some(client) = object.get("client") {
        return Ok((target_label(client), client.clone()));
    }

    if object.len() == 1
        && let Some(url) = object.get("url").and_then(JsonValue::as_str)
    {
        return Ok((url.to_string(), serde_json::json!({ "http": url })));
    }

    Ok((target_label(&json), json))
}

fn target_label(spec: &JsonValue) -> String {
    spec.get("http")
        .or_else(|| spec.get("grpc"))
        .or_else(|| spec.get("url"))
        .and_then(JsonValue::as_str)
        .map(str::to_string)
        .or_else(|| {
            spec.get("type")
                .and_then(JsonValue::as_str)
                .map(|kind| format!("client:{kind}"))
        })
        .or_else(|| {
            spec.as_object().and_then(|object| {
                object
                    .keys()
                    .next()
                    .map(|key| format!("client:{key}"))
            })
        })
        .unwrap_or_else(|| "client".to_string())
}

fn toml_to_json(value: TomlValue) -> JsonValue {
    match value {
        TomlValue::String(value) => JsonValue::String(value),
        TomlValue::Integer(value) => JsonValue::Number(value.into()),
        TomlValue::Float(value) => serde_json::Number::from_f64(value)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null),
        TomlValue::Boolean(value) => JsonValue::Bool(value),
        TomlValue::Datetime(value) => JsonValue::String(value.to_string()),
        TomlValue::Array(values) => JsonValue::Array(values.into_iter().map(toml_to_json).collect()),
        TomlValue::Table(table) => JsonValue::Object(
            table
                .into_iter()
                .map(|(key, value)| (key, toml_to_json(value)))
                .collect(),
        ),
    }
}

fn split_num_unit(s: &str) -> Result<(&str, &str), Error> {
    let idx = s
        .find(|c: char| c.is_ascii_alphabetic() || c == '%')
        .ok_or_else(|| Error::Config(format!("missing unit in {s:?}")))?;
    Ok((&s[..idx], &s[idx..]))
}

fn parse_duration(s: &str) -> Result<Duration, Error> {
    let s = s.trim();
    let (num, unit) = split_num_unit(s)?;
    let v: f64 = num
        .trim()
        .parse()
        .map_err(|_| Error::Config(format!("bad duration {s:?}")))?;
    let secs = match unit {
        "ms" => v / 1000.0,
        "s" => v,
        "m" => v * 60.0,
        "h" => v * 3600.0,
        other => return Err(Error::Config(format!("bad duration unit {other:?}"))),
    };
    Ok(Duration::from_secs_f64(secs))
}

fn parse_rate(s: &str) -> Result<f64, Error> {
    let s = s.trim();
    let (num, per) = s
        .split_once('/')
        .ok_or_else(|| Error::Config(format!("bad rate {s:?}, want N/s")))?;
    let n: f64 = num
        .trim()
        .parse()
        .map_err(|_| Error::Config(format!("bad rate {s:?}")))?;
    let secs = match per.trim() {
        "s" => 1.0,
        "m" => 60.0,
        "h" => 3600.0,
        other => return Err(Error::Config(format!("bad rate unit {other:?}"))),
    };
    Ok(n / secs)
}

fn parse_percent(s: &str) -> Result<f64, Error> {
    let v: f64 = s
        .trim()
        .trim_end_matches('%')
        .parse()
        .map_err(|_| Error::Config(format!("bad percent {s:?}")))?;
    Ok(v / 100.0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn parses_a_scenario() {
        let text = r#"
[target]
url = "https://api.internal"

[[stage]]
rate = "100/s"
duration = "30s"

[[stage]]
rate = "1000/s"
duration = "2m"

[threshold]
p99 = "250ms"
error_rate = "0.1%"
"#;
        let s = Scenario::from_toml(text).expect("parses");
        assert_eq!(s.target_url, "https://api.internal");
        assert_eq!(s.client_spec, serde_json::json!({ "http": "https://api.internal" }));
        assert_eq!(s.request.method, "GET");
        assert_eq!(s.request.path, "/");
        assert_eq!(s.stages.len(), 2);
        assert!((s.stages[0].rate_per_sec - 100.0).abs() < f64::EPSILON);
        assert_eq!(s.stages[1].duration, Duration::from_secs(120));
        assert_eq!(s.thresholds.p99, Some(Duration::from_millis(250)));
        assert_eq!(s.thresholds.error_rate, Some(0.001));
    }

    #[test]
    fn target_client_spec_reaches_any_client_protocol_shape() {
        let text = r#"
[target.client]
synth = { status = 201, body = "ok" }

[request]
method = "POST"
path = "/anything"
body = "payload"

[request.headers]
x-test = "yes"

[request.query]
a = "1"

[[stage]]
rate = "1/s"
duration = "1s"
"#;
        let s = Scenario::from_toml(text).expect("parses");
        assert_eq!(s.target_url, "client:synth");
        assert_eq!(s.client_spec, serde_json::json!({ "synth": { "status": 201, "body": "ok" } }));
        assert_eq!(s.request.method, "POST");
        assert_eq!(s.request.path, "/anything");
        assert_eq!(s.request.body.as_deref(), Some("payload"));
        assert_eq!(
            s.request
                .headers
                .get("x-test")
                .map(String::as_str),
            Some("yes")
        );
        assert_eq!(s.request.query.get("a").map(String::as_str), Some("1"));
    }
}
