use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let target_os = env::var("CARGO_CFG_TARGET_OS")?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;

    println!("cargo:rerun-if-changed=src/backend_macos.c");
    println!("cargo:rerun-if-changed=src/probe_cow.h");

    if (target_os.as_str(), target_arch.as_str()) == ("macos", "aarch64") {
        cc::Build::new()
            .file("src/backend_macos.c")
            .warnings(true)
            .compile("proxima_vm_native");
        println!("cargo:rustc-link-lib=framework=Hypervisor");
    }

    Ok(())
}
