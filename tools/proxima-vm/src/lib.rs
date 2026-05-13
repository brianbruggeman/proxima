//! Minimal VM ground for proving that a guest can emit bytes through a
//! Proxima [`Pipe`].
//!
//! [`ScratchVm`] does not boot an operating system. It synthesizes a tiny
//! architecture-native guest program whose only effect is emitting its
//! configured bytes and halting. The platform leaf is KVM on Linux x86_64
//! and Hypervisor.framework on Apple silicon.
//!
//! # Parity with proxima-process libc-shim
//!
//! Per `proxima.decision.libc_shim_vm_parity`, proxima-vm and the
//! proxima-process libc-shim share a dispatch contract:
//! [`proxima_protocols::process::ChildRequest`] /
//! [`proxima_protocols::process::ChildResponse`]. The parity-side
//! handler shape (the trait proxima-vm impls when guests start
//! emitting protocol traffic) lives in [`dispatch`].

#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(feature = "alloc")]
extern crate alloc;

pub mod dispatch;

#[cfg(feature = "std")]
use core::future::Future;

use bytes::Bytes;
use proxima_core::ProximaError;
#[cfg(feature = "std")]
use proxima_primitives::pipe::SendPipe;
#[cfg(feature = "std")]
use proxima_primitives::pipe::{Request, Response};

const HELLO_BYTES: &[u8] = b"hello from proxima-vm\n";

/// An empty-machine guest whose only declared output is `output`.
#[derive(Clone, Debug)]
pub struct ScratchVm {
    output: Bytes,
}

impl ScratchVm {
    /// Create a scratch guest that emits `output` and then stops.
    #[must_use]
    pub fn new(output: impl Into<Bytes>) -> Self {
        Self {
            output: output.into(),
        }
    }

    /// Create the first proof guest.
    #[must_use]
    pub fn hello() -> Self {
        Self::new(Bytes::from_static(HELLO_BYTES))
    }

    /// Run the guest and return bytes observed from its exit channel.
    pub fn run(&self) -> Result<Bytes, ProximaError> {
        platform::run(&self.output)
    }
}

#[cfg(feature = "std")]
impl SendPipe for ScratchVm {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let guest = self.clone();
        async move { guest.run().map(Response::ok) }
    }
}

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
mod platform {
    use std::ffi::CStr;
    use std::os::raw::c_char;

    use bytes::Bytes;
    use proxima_core::ProximaError;

    const ERROR_CAPACITY: usize = 512;

    unsafe extern "C" {
        fn proxima_vm_scratch_run(
            message: *const u8,
            message_length: usize,
            output: *mut u8,
            output_capacity: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;
    }

    pub fn run(message: &[u8]) -> Result<Bytes, ProximaError> {
        let mut output = vec![0_u8; message.len()];
        let mut error_buffer = [0_i8; ERROR_CAPACITY];
        let status = unsafe {
            proxima_vm_scratch_run(
                message.as_ptr(),
                message.len(),
                output.as_mut_ptr(),
                output.len(),
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if status != 0 {
            let message = unsafe { CStr::from_ptr(error_buffer.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(ProximaError::Upstream(message));
        }
        Ok(Bytes::from(output))
    }
}

#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
mod platform {
    use bytes::Bytes;
    use proxima_core::ProximaError;

    pub fn run(_message: &[u8]) -> Result<Bytes, ProximaError> {
        Err(ProximaError::Config(
            "scratch-vm supports linux/x86_64 KVM and macos/aarch64 Hypervisor.framework only"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use super::*;

    #[test]
    fn hello_guest_declares_expected_output() {
        assert_eq!(ScratchVm::hello().output, Bytes::from_static(HELLO_BYTES));
    }
}
