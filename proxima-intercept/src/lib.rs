//! Generic MITM intercept pipeline: TLS-terminating CONNECT proxy that forwards,
//! captures, and replays traffic for any host, with no vendor knowledge.

pub mod ca;
#[cfg(feature = "intercept-capture")]
pub mod capture;
pub mod compress;
#[cfg(feature = "intercept-config")]
pub mod config;
pub mod interceptor;
mod pipe;
#[cfg(feature = "quic-intercept")]
pub mod quic_relay;
pub mod session;
pub mod shutdown;
pub mod swap;
#[cfg(feature = "delta-tee")]
pub mod tee;

pub use interceptor::{HostPolicy, Interception, Interceptor};
pub use pipe::{InterceptPipe, InterceptPipeFactory, factory_arc};
pub use pipe::{
    ResponsePumpReport, WsPumpReport, decode_response_body, is_telemetry_host,
    parse_connect_target, parse_request_model, pump_streaming_response, pump_ws_client_to_upstream,
    pump_ws_upstream_to_client, request_body_tail,
};
pub use swap::{StreamFramer, SwapSurface, Turn, pump_synthesized_sse};
