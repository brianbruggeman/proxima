use std::env;
use std::path::PathBuf;

const ELF_BINS: [&str; 1] = ["uppercase"];

const FULL_BINS: [&str; 4] = [
    "proxima-vm-guest-lambda",
    "proxima-vm-guest-lambda-psci",
    "proxima-vm-guest-lambda-gic",
    "proxima-vm-guest-lambda-pl011",
];

// cargo itself guarantees these are set for every build script invocation;
// the only failure mode is a cargo bug, which expect() surfaces just as well
// as a hand-rolled message would.
#[allow(clippy::expect_used)]
fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    for (bins, script) in [(&FULL_BINS[..], "link.ld"), (&ELF_BINS[..], "link-elf.ld")] {
        let linker_script = manifest_dir.join(script);
        for bin in bins {
            println!("cargo:rustc-link-arg-bin={bin}=-T{}", linker_script.display());
        }
        println!("cargo:rerun-if-changed={script}");
    }

    // rust-lld defaults x86_64-unknown-none to a static-PIE layout; the guest
    // ABI needs an absolute, non-relocatable load address like aarch64 already gets.
    let target = env::var("TARGET").expect("cargo always sets TARGET");
    if target.contains("x86_64") {
        println!("cargo:rustc-link-arg=-no-pie");
    }
}
