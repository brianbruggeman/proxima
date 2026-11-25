use alloc::string::String;

use proxima_tensor::NodeId;

/// Errors from loading the ROCm runtime, compiling HIP C++, or executing.
#[derive(Debug, thiserror::Error)]
pub enum HipDriverError {
    #[error("ROCm runtime is not installed; tried: {tried}")]
    RuntimeNotInstalled { tried: String },
    #[error("hipRTC is not installed; tried: {tried}")]
    RtcNotInstalled { tried: String },
    #[error("ROCm library `{library}` does not export `{symbol}`")]
    MissingSymbol {
        library: &'static str,
        symbol: &'static str,
    },
    #[error("HIP `{call}` failed with code {code}: {message}")]
    Runtime {
        call: &'static str,
        code: i32,
        message: String,
    },
    #[error("hipRTC `{call}` failed with code {code}: {message}")]
    Rtc {
        call: &'static str,
        code: i32,
        message: String,
    },
    #[error("hipRTC rejected the emitted source:\n{log}")]
    CompileLog { log: String },
    #[error("kernel binding contract is not inputs, output, uniforms")]
    BindingContract,
    #[error("launch argument count does not match the kernel's bindings")]
    ArgumentCountMismatch,
    #[error("device ordinal {ordinal} is out of range; this host exposes {count}")]
    DeviceOutOfRange { ordinal: i32, count: i32 },
    #[error("hip device ordinal {0} does not fit HIP's integer device index")]
    DeviceOrdinalTooLarge(u32),
    #[error("a launch of {threads} threads needs more blocks than the launch index holds")]
    GridTooLarge { threads: u64 },
    #[error("refused to launch an idle grid: an op with no live output dispatches nothing")]
    IdleLaunch,
    #[error("smoke input is too large for the launch ABI: {0} elements")]
    InputTooLarge(usize),
    #[error("hip wave width {0} is not supported; expected 32 or 64")]
    UnsupportedWaveWidth(u32),
    #[error("configured hip wave width {configured} does not match the device's warpSize {device}")]
    WaveMismatch { configured: u32, device: u32 },
    #[error("hip graph path rejected the graph: {reason}")]
    UnsupportedGraph { reason: &'static str },
    #[error("hip graph node {node:?} could not be emitted: {error}")]
    Emit { node: NodeId, error: String },
    #[error("hip graph buffer for node {0:?} is missing")]
    MissingBuffer(NodeId),
    #[error("a {len}-byte view at offset {offset} passes the end of a {bytes}-byte allocation")]
    ViewOutOfRange {
        offset: usize,
        len: usize,
        bytes: usize,
    },
    #[error("hip events are not recorded by this runtime, so launch timing is skipped")]
    EventsUnsupported,
    #[error("this hip runtime does not export hipMemGetInfo")]
    MemoryInfoUnavailable,
}
