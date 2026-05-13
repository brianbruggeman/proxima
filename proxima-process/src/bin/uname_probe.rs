//! Probe binary used by the uname interpose smoke test.
//! Calls `uname(2)` via libc and prints the five utsname fields
//! pipe-separated so the test can parse them deterministically.

use std::ffi::CStr;

fn read_field(field: &[i8]) -> String {
    // SAFETY: utsname fields are NUL-terminated C strings within
    // their fixed-size char arrays.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(field.as_ptr().cast::<u8>(), field.len()) };
    let cstr = CStr::from_bytes_until_nul(bytes).unwrap_or(c"");
    cstr.to_str().unwrap_or("non-utf8").to_string()
}

fn main() {
    // SAFETY: uts is a stack-allocated POD struct; uname writes
    // into it via the pointer. Initialize to zeros so any field
    // the shim doesn't touch reads as empty.
    let mut uts: libc::utsname = unsafe { std::mem::zeroed() };
    let result = unsafe { libc::uname(&mut uts as *mut _) };
    if result != 0 {
        println!("uname-error");
        return;
    }
    // Output format: sysname|nodename|release|version|machine
    println!(
        "{}|{}|{}|{}|{}",
        read_field(&uts.sysname),
        read_field(&uts.nodename),
        read_field(&uts.release),
        read_field(&uts.version),
        read_field(&uts.machine),
    );
}
