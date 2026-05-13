use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use arc_swap::ArcSwap;
use bytes::Bytes;
use serde_json::Value;

use proxima_core::ProximaError;

/// Type-erased codec for plugin-supplied wire formats (protobuf, cbor,
/// msgpack, …). Both directions are bytes ⇄ `serde_json::Value`; the
/// typed `MessageCodec` trait (in `lib.rs`) stays for Rust callers.
pub trait DynCodec: Send + Sync + 'static {
    fn name(&self) -> &str;

    fn content_type(&self) -> &str {
        "application/octet-stream"
    }

    fn decode_to_json(&self, bytes: &[u8]) -> Result<Value, ProximaError>;

    fn encode_from_json(&self, value: &Value) -> Result<Bytes, ProximaError>;
}

pub type DynCodecHandle = Arc<dyn DynCodec>;

pub type CodecBuildFuture<'lifetime> =
    Pin<Box<dyn Future<Output = Result<DynCodecHandle, ProximaError>> + Send + 'lifetime>>;

pub trait CodecFactory: Send + Sync + 'static {
    fn name(&self) -> &str;

    fn build<'lifetime>(&'lifetime self, spec: &'lifetime Value) -> CodecBuildFuture<'lifetime>;
}

pub type DynCodecFactory = Arc<dyn CodecFactory>;

pub struct CodecRegistry {
    factories: ArcSwap<BTreeMap<String, DynCodecFactory>>,
}

impl Default for CodecRegistry {
    fn default() -> Self {
        Self {
            factories: ArcSwap::from_pointee(BTreeMap::new()),
        }
    }
}

impl CodecRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, factory: DynCodecFactory) -> Result<(), ProximaError> {
        let name = factory.name().to_string();
        loop {
            let current = self.factories.load_full();
            if current.contains_key(&name) {
                return Err(ProximaError::Registry(format!(
                    "codec factory `{name}` already registered"
                )));
            }
            let mut next: BTreeMap<String, DynCodecFactory> = (*current).clone();
            next.insert(name.clone(), factory.clone());
            let prev = self.factories.compare_and_swap(&current, Arc::new(next));
            if Arc::ptr_eq(&prev, &current) {
                return Ok(());
            }
        }
    }

    pub fn get(&self, name: &str) -> Result<DynCodecFactory, ProximaError> {
        self.factories
            .load_full()
            .get(name)
            .cloned()
            .ok_or_else(|| ProximaError::Registry(format!("no codec factory `{name}`")))
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.factories.load_full().keys().cloned().collect()
    }

    pub async fn resolve(&self, spec: &Value) -> Result<DynCodecHandle, ProximaError> {
        let kind = spec
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| ProximaError::Config("codec spec requires `type`".into()))?;
        let factory = self.get(kind)?;
        factory.build(spec).await
    }
}

/// JSON via simd-json on the hot path; recording/config paths keep
/// vanilla serde_json.
pub struct JsonDynCodec;

impl DynCodec for JsonDynCodec {
    fn name(&self) -> &str {
        "json"
    }

    fn content_type(&self) -> &str {
        "application/json"
    }

    fn decode_to_json(&self, bytes: &[u8]) -> Result<Value, ProximaError> {
        // simd-json mutates its input — own a per-thread scratch Vec
        // so each decode reuses one allocation per worker.
        thread_local! {
            static SCRATCH: std::cell::RefCell<Vec<u8>> = const {
                std::cell::RefCell::new(Vec::new())
            };
        }
        SCRATCH.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.clear();
            buf.extend_from_slice(bytes);
            simd_json::serde::from_slice(&mut buf)
                .map_err(|err| ProximaError::Decode(format!("json codec: {err}")))
        })
    }

    fn encode_from_json(&self, value: &Value) -> Result<Bytes, ProximaError> {
        simd_json::serde::to_vec(value)
            .map(Bytes::from)
            .map_err(|err| ProximaError::Encode(format!("json codec: {err}")))
    }
}

pub struct JsonCodecFactory;

impl CodecFactory for JsonCodecFactory {
    fn name(&self) -> &str {
        "json"
    }

    fn build<'lifetime>(&'lifetime self, _spec: &'lifetime Value) -> CodecBuildFuture<'lifetime> {
        Box::pin(async move {
            let codec: DynCodecHandle = Arc::new(JsonDynCodec);
            Ok(codec)
        })
    }
}

/// Base64-wraps raw bytes for JSON-routed transports. True binary
/// passthrough should bypass the codec layer entirely.
pub struct BytesPassthroughDynCodec;

impl DynCodec for BytesPassthroughDynCodec {
    fn name(&self) -> &str {
        "bytes"
    }

    fn content_type(&self) -> &str {
        "application/octet-stream"
    }

    fn decode_to_json(&self, bytes: &[u8]) -> Result<Value, ProximaError> {
        use base64::Engine as _;
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        Ok(Value::String(encoded))
    }

    fn encode_from_json(&self, value: &Value) -> Result<Bytes, ProximaError> {
        use base64::Engine as _;
        let raw = value.as_str().ok_or_else(|| {
            ProximaError::Encode("bytes codec encode requires a base64 string value".into())
        })?;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(raw)
            .map_err(|err| ProximaError::Encode(format!("bytes codec base64: {err}")))?;
        Ok(Bytes::from(decoded))
    }
}

pub struct BytesPassthroughCodecFactory;

impl CodecFactory for BytesPassthroughCodecFactory {
    fn name(&self) -> &str {
        "bytes"
    }

    fn build<'lifetime>(&'lifetime self, _spec: &'lifetime Value) -> CodecBuildFuture<'lifetime> {
        Box::pin(async move {
            let codec: DynCodecHandle = Arc::new(BytesPassthroughDynCodec);
            Ok(codec)
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[proxima::test]
    async fn json_codec_round_trips_value_through_bytes() {
        let codec = JsonDynCodec;
        let bytes = codec.encode_from_json(&json!({"a": 1})).expect("encode");
        let decoded = codec.decode_to_json(&bytes).expect("decode");
        assert_eq!(decoded, json!({"a": 1}));
    }

    #[proxima::test]
    async fn bytes_codec_round_trips_raw_through_base64() {
        let codec = BytesPassthroughDynCodec;
        let raw = b"\x00\xff\xab\xcd binary";
        let value = codec.decode_to_json(raw).expect("decode");
        assert!(matches!(value, Value::String(_)));
        let bytes = codec.encode_from_json(&value).expect("encode");
        assert_eq!(&bytes[..], raw);
    }

    #[proxima::test]
    async fn registry_resolves_via_type_field() {
        let registry = CodecRegistry::new();
        registry
            .register(Arc::new(JsonCodecFactory))
            .expect("register");
        let codec = registry
            .resolve(&json!({"type": "json"}))
            .await
            .expect("resolve");
        assert_eq!(codec.name(), "json");
        assert_eq!(codec.content_type(), "application/json");
    }

    #[proxima::test]
    async fn registry_unknown_type_returns_registry_error() {
        let registry = CodecRegistry::new();
        let outcome = registry.resolve(&json!({"type": "nope"})).await;
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }

    #[proxima::test]
    async fn registry_missing_type_returns_config_error() {
        let registry = CodecRegistry::new();
        let outcome = registry.resolve(&json!({})).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[proxima::test]
    async fn duplicate_register_returns_registry_error() {
        let registry = CodecRegistry::new();
        registry
            .register(Arc::new(JsonCodecFactory))
            .expect("first");
        let outcome = registry.register(Arc::new(JsonCodecFactory));
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }
}
