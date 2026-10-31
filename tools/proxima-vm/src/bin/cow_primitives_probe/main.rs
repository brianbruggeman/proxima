//! Copy-on-write primitive probe, µsec campaign slice 3
//! (`tools/proxima-vm/ROADMAP.md`). One lane per host: Hypervisor.framework on
//! Apple silicon, KVM on Linux x86_64; both print the same `key:value` lines.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod hvf;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod kvm;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    hvf::run()
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    kvm::run()
}

#[cfg(not(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(target_os = "linux", target_arch = "x86_64")
)))]
fn main() {
    eprintln!("the copy-on-write probe needs macos/aarch64 Hypervisor.framework or linux/x86_64 KVM");
}
