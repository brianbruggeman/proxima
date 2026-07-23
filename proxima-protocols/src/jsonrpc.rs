//! JSON-RPC 2.0 envelope types. Parametric over `params` / `result`
//! payloads via `serde_json::Value`, so the same envelope serves any
//! JSON-RPC protocol (MCP, A2A, AG-UI, custom).
//!
//! Tier: no_std + alloc. Pure sans-IO data types — caller owns the
//! transport (HTTP / WebSocket / stdio / custom).


use alloc::format;
use alloc::string::{String, ToString};

use serde::{Deserialize, Serialize};
use serde_json::Value;

const JSONRPC_VERSION: &str = "2.0";

/// A JSON-RPC 2.0 request envelope. `id` is `None` for a notification —
/// the spec's wire form omits the `id` member entirely, and the sender
/// expects no response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<RequestId>,
}

impl JsonRpcRequest {
    #[must_use]
    pub fn new(method: impl Into<String>, params: Option<Value>, id: RequestId) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
            id: Some(id),
        }
    }

    /// A fire-and-forget request: dispatched but never answered.
    #[must_use]
    pub fn notification(method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
            id: None,
        }
    }

    #[must_use]
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

/// A JSON-RPC 2.0 response envelope. Exactly one of `result` / `error`
/// is populated. `id` is `null` when the request's id could not be
/// determined (e.g. a parse error before the envelope was readable).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    pub id: Option<RequestId>,
}

impl JsonRpcResponse {
    #[must_use]
    pub fn success(id: RequestId, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: Some(result),
            error: None,
            id: Some(id),
        }
    }

    #[must_use]
    pub fn failure(id: Option<RequestId>, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: None,
            error: Some(error),
            id,
        }
    }
}

/// JSON-RPC error payload. Codes mirror the standard registry; `-32601`
/// is method-not-found, `-32602` invalid-params, `-32603` internal-error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    #[must_use]
    pub fn parse_error(detail: impl Into<String>) -> Self {
        Self {
            code: -32_700,
            message: detail.into(),
            data: None,
        }
    }

    #[must_use]
    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32_601,
            message: format!("method not found: {method}"),
            data: None,
        }
    }

    #[must_use]
    pub fn invalid_params(detail: impl Into<String>) -> Self {
        Self {
            code: -32_602,
            message: detail.into(),
            data: None,
        }
    }

    #[must_use]
    pub fn internal(detail: impl Into<String>) -> Self {
        Self {
            code: -32_603,
            message: detail.into(),
            data: None,
        }
    }
}

/// JSON-RPC request id. The spec allows string, integer, or null for a
/// request id; proxima represents "no id" (notification, or null) as
/// `JsonRpcRequest::id: Option<RequestId>` rather than folding null into
/// this enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    Number(i64),
    String(String),
}

impl From<i64> for RequestId {
    fn from(value: i64) -> Self {
        Self::Number(value)
    }
}

impl From<String> for RequestId {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for RequestId {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_round_trip_with_params() {
        let request = JsonRpcRequest::new(
            "message/send",
            Some(json!({ "message": { "parts": [{ "text": "hello" }] } })),
            RequestId::from(7),
        );
        let serialized = serde_json::to_value(&request).expect("serialize");
        assert_eq!(serialized["jsonrpc"], "2.0");
        assert_eq!(serialized["method"], "message/send");
        assert_eq!(serialized["id"], 7);
        let parsed: JsonRpcRequest = serde_json::from_value(serialized).expect("parse");
        assert_eq!(parsed, request);
    }

    #[test]
    fn success_response_carries_result_only() {
        let response = JsonRpcResponse::success(RequestId::from("abc"), json!({"ok": true}));
        let serialized = serde_json::to_value(&response).expect("serialize");
        assert_eq!(serialized["result"], json!({"ok": true}));
        assert!(serialized.get("error").is_none());
    }

    #[test]
    fn failure_response_carries_error_only() {
        let error = JsonRpcError::method_not_found("missing/method");
        let response = JsonRpcResponse::failure(Some(RequestId::from(1)), error.clone());
        let serialized = serde_json::to_value(&response).expect("serialize");
        assert!(serialized.get("result").is_none());
        assert_eq!(serialized["error"]["code"], -32_601);
        let parsed: JsonRpcResponse = serde_json::from_value(serialized).expect("parse");
        assert_eq!(parsed.error, Some(error));
    }

    #[test]
    fn notification_omits_id_on_the_wire() {
        let request = JsonRpcRequest::notification("notifications/initialized", None);
        assert!(request.is_notification());
        let serialized = serde_json::to_value(&request).expect("serialize");
        assert!(serialized.get("id").is_none(), "got: {serialized}");
        let parsed: JsonRpcRequest = serde_json::from_value(serialized).expect("parse");
        assert!(parsed.is_notification());
    }

    #[test]
    fn failure_response_with_unknown_id_serializes_null() {
        let response =
            JsonRpcResponse::failure(None, JsonRpcError::parse_error("parse error: eof"));
        let serialized = serde_json::to_value(&response).expect("serialize");
        assert_eq!(serialized["id"], Value::Null);
        assert_eq!(serialized["error"]["code"], -32_700);
    }
}
