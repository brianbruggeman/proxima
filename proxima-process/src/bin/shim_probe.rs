//! Tiny harness used by the libc-shim smoke test. Calls
//! `gethostname(3)` via libc and prints whatever comes back; the
//! test spawns this binary with and without the interpose shim
//! attached to verify that loading the shim flips the answer.

use std::ffi::CStr;

fn main() {
    let mut buffer = [0u8; 256];
    // SAFETY: buffer is a fixed-size stack array; we hand libc its
    // pointer and capacity, then read back the NUL-terminated
    // string it wrote. Any non-zero return means "leave the
    // buffer alone" — we handle that by printing the empty
    // string.
    let result =
        unsafe { libc::gethostname(buffer.as_mut_ptr().cast::<libc::c_char>(), buffer.len()) };
    if result != 0 {
        println!("gethostname-error");
        return;
    }
    let cstr = CStr::from_bytes_until_nul(&buffer).unwrap_or(c"");
    let text = cstr.to_str().unwrap_or("non-utf8");
    println!("{text}");
}
