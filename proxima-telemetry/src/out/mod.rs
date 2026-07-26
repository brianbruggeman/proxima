#[cfg(feature = "otlp-http")]
pub mod otlp_http;

#[cfg(feature = "otlp-grpc")]
pub mod otlp_grpc;

// generic over any `TelemetryPipeHandle` factory, so it covers both
// otlp-http and otlp-grpc transports from one sink.
#[cfg(feature = "otlp-http")]
pub mod resilient;

pub mod native;
