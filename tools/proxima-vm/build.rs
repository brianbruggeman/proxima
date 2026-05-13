use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let target_os = env::var("CARGO_CFG_TARGET_OS")?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;

    println!("cargo:rerun-if-changed=src/backend_linux.c");
    println!("cargo:rerun-if-changed=src/backend_macos.c");

    match (target_os.as_str(), target_arch.as_str()) {
        ("linux", "x86_64") => {
            cc::Build::new()
                .file("src/backend_linux.c")
                .warnings(true)
                .compile("proxima_vm_native");
        }
        ("macos", "aarch64") => {
            cc::Build::new()
                .file("src/backend_macos.c")
                .warnings(true)
                .compile("proxima_vm_native");
            println!("cargo:rustc-link-lib=framework=Hypervisor");
        }
        _ => {}
    }

    Ok(())
}
