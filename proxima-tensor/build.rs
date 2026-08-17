// links a statically-built ggml checkout for the `ggml-bench` criterion
// bench only; a no-op for every other build so a normal `cargo build` never
// needs ggml on disk.
use std::env;
use std::path::PathBuf;

fn main() {
    if env::var_os("CARGO_FEATURE_GGML_BENCH").is_none() {
        return;
    }

    let ggml_root = env::var("GGML_BUILD_DIR").unwrap_or_else(|_| {
        "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/\
         a5b35a70-66a5-46bd-882c-643ba7f174f3/scratchpad/ggml"
            .to_string()
    });
    let build_dir = PathBuf::from(&ggml_root).join("build");

    println!("cargo:rustc-link-search=native={}", build_dir.join("src").display());
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rerun-if-env-changed=GGML_BUILD_DIR");
}
