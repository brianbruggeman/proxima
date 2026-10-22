//! Compatibility path for the shared structural scalar codec.
//!
//! The codec lives in `proxima-primitives` so format frontends such as GGUF and safetensors use
//! the same layout-driven conversion code.

pub use proxima_primitives::float_layout_codec::{decode, encode, encode_counting};
