use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use base64::Engine as _;
use bytes::Bytes;
use serde_json::Value;

use proxima_core::ProximaError;
use proxima_core::factory::Named;

use crate::decode_through_scratch;

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

    /// Build the codec a `codec = { type = "..." }` config row names.
    ///
    /// Boxed, not RPITIT: the only consumer is `Arc<dyn CodecFactory>` in a
    /// [`CodecRegistry`], and an `impl Future` return type is not
    /// dyn-compatible.
    fn build<'lifetime>(&'lifetime self, spec: &'lifetime Value) -> CodecBuildFuture<'lifetime>;
}

// bridge `CodecFactory` into the generic factory registry without touching any
// existing `impl CodecFactory` — the registry only needs the factory's name.
impl Named for dyn CodecFactory {
    fn name(&self) -> &str {
        CodecFactory::name(self)
    }
}

pub type DynCodecFactory = Arc<dyn CodecFactory>;

/// The codec registry is the generic [`proxima_core::FactoryRegistry`]
/// specialized to `dyn CodecFactory` — the shape `proxima-core`'s own module
/// doc names a codec table as a consumer of. The surface (`new` / `register` /
/// `get` / `names` / `with`) is unchanged; only the implementation is now
/// shared, and a duplicate or absent name reports the typed
/// [`proxima_core::RegistryError`] rather than a hand-formatted string.
///
/// ```
/// use std::sync::Arc;
///
/// use proxima_codec::factory::{CodecFactory, CodecRegistry, JsonCodecFactory};
///
/// let registry = CodecRegistry::new().with(Arc::new(JsonCodecFactory))?;
///
/// assert_eq!(registry.names(), vec!["json".to_string()]);
/// assert_eq!(registry.get("json")?.name(), "json");
/// assert!(registry.get("cbor").is_err());
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
pub type CodecRegistry = proxima_core::FactoryRegistry<dyn CodecFactory>;

/// Build the codec named by a `codec = { type = "..." }` config row.
///
/// A free function, not an inherent method: [`CodecRegistry`] is an alias for a
/// type this crate does not own, so an `impl` block here would not compile.
pub async fn resolve(
    registry: &CodecRegistry,
    spec: &Value,
) -> Result<DynCodecHandle, ProximaError> {
    let kind = spec
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| ProximaError::Config("codec spec requires `type`".into()))?;
    registry.get(kind)?.build(spec).await
}

/// JSON via simd-json on the hot path; recording/config paths keep
/// vanilla serde_json.
///
/// ```
/// use proxima_codec::factory::{DynCodec, JsonDynCodec};
/// use serde_json::json;
///
/// let codec = JsonDynCodec;
/// let wire = codec.encode_from_json(&json!({"op": "ping", "seq": 7}))?;
///
/// assert_eq!(codec.content_type(), "application/json");
/// assert_eq!(codec.decode_to_json(&wire)?, json!({"op": "ping", "seq": 7}));
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
pub struct JsonDynCodec;

impl DynCodec for JsonDynCodec {
    fn name(&self) -> &str {
        "json"
    }

    fn content_type(&self) -> &str {
        "application/json"
    }

    fn decode_to_json(&self, bytes: &[u8]) -> Result<Value, ProximaError> {
        decode_through_scratch(bytes)
    }

    fn encode_from_json(&self, value: &Value) -> Result<Bytes, ProximaError> {
        simd_json::serde::to_vec(value)
            .map(Bytes::from)
            .map_err(|err| ProximaError::Encode(format!("json: {err}")))
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
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        Ok(Value::String(encoded))
    }

    fn encode_from_json(&self, value: &Value) -> Result<Bytes, ProximaError> {
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
    use proxima_core::RegistryError;
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
        let codec = resolve(&registry, &json!({"type": "json"}))
            .await
            .expect("resolve");
        assert_eq!(codec.name(), "json");
        assert_eq!(codec.content_type(), "application/json");
    }

    #[proxima::test]
    async fn registry_unknown_type_returns_registry_error() {
        let registry = CodecRegistry::new();
        let outcome = resolve(&registry, &json!({"type": "nope"})).await;
        assert!(matches!(
            outcome,
            Err(ProximaError::RegistryKind(
                RegistryError::NotRegistered { .. }
            ))
        ));
    }

    #[proxima::test]
    async fn registry_missing_type_returns_config_error() {
        let registry = CodecRegistry::new();
        let outcome = resolve(&registry, &json!({})).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[proxima::test]
    async fn duplicate_register_returns_registry_error() {
        let registry = CodecRegistry::new();
        registry
            .register(Arc::new(JsonCodecFactory))
            .expect("first");
        let outcome = registry.register(Arc::new(JsonCodecFactory));
        assert!(matches!(
            outcome,
            Err(ProximaError::RegistryKind(
                RegistryError::AlreadyRegistered { .. }
            ))
        ));
    }
}
