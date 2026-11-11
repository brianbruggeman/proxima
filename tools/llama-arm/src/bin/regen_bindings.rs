//! regenerates src/bindings.rs from the vendored llama.h (llama.cpp ba3c05f71).
//! run: `cargo run --features bindgen --bin regen_bindings`

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

const LIBCLANG_DIR: &str = "/Library/Developer/CommandLineTools/usr/lib";

const FUNCTIONS: &[&str] = &[
    "llama_backend_init",
    "llama_backend_free",
    "llama_model_default_params",
    "llama_context_default_params",
    "llama_model_load_from_file",
    "llama_model_free",
    "llama_init_from_model",
    "llama_free",
    "llama_model_get_vocab",
    "llama_vocab_n_tokens",
    "llama_tokenize",
    "llama_batch_get_one",
    "llama_decode",
    "llama_synchronize",
    "llama_get_logits_ith",
    "llama_sampler_chain_default_params",
    "llama_sampler_chain_init",
    "llama_sampler_chain_add",
    "llama_sampler_init_greedy",
    "llama_sampler_sample",
    "llama_vocab_is_eog",
    "llama_memory_clear",
    "llama_get_memory",
    "llama_print_system_info",
    "llama_perf_context",
    "llama_perf_context_reset",
];

// the structs the functions take or return by value; opaque handles come in as typedef'd pointers
const TYPES: &[&str] = &[
    "llama_model_params",
    "llama_context_params",
    "llama_batch",
    "llama_sampler_chain_params",
    "llama_perf_context_data",
];

fn main() -> ExitCode {
    // safety: first statement of a single-threaded bin, before bindgen spawns anything or reads the env
    unsafe { env::set_var("LIBCLANG_PATH", LIBCLANG_DIR) };
    match generate() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("regen_bindings failed: {message}");
            ExitCode::FAILURE
        }
    }
}

fn generate() -> Result<(), String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let headers = root.join("llama-headers");
    let mut builder = bindgen::Builder::default()
        .header(headers.join("llama.h").to_string_lossy())
        .clang_arg(format!("-I{}", headers.display()))
        .dynamic_library_name("LlamaApi")
        .dynamic_link_require_all(true)
        .layout_tests(true)
        .derive_default(false)
        .derive_debug(true)
        .generate_comments(false);
    for function in FUNCTIONS {
        builder = builder.allowlist_function(function);
    }
    for type_name in TYPES {
        builder = builder.allowlist_type(type_name);
    }
    let bindings = builder.generate().map_err(|error| error.to_string())?;
    bindings
        .write_to_file(root.join("src/bindings.rs"))
        .map_err(|error| error.to_string())
}
