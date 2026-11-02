//! The time one prompt-cache lookup takes with 4, 64, 256 and 1,024 cached
//! 1,024-token conversations, for a prompt that extends one of them (hit) and
//! one that shares nothing past BOS (miss), against the linear scan the block
//! index replaced. Run it alone; a debug build prints `profile=debug` and is
//! not comparable to a release build.
//!
//! `prompt_cache_lookup_cost [shared_header_tokens ...]` (default: `0 512`,
//! no shared opening and a shared 512-token system prompt).
//!
//! `cargo run -p proxima-model-interop --release --features std --example prompt_cache_lookup_cost`

use std::process::ExitCode;

use proxima_model_interop::probes::prompt_cache_lookup_cost;

fn main() -> ExitCode {
    let mut headers: Vec<usize> = Vec::new();
    for argument in std::env::args().skip(1) {
        match argument.parse() {
            Ok(tokens) => headers.push(tokens),
            Err(error) => {
                eprintln!(
                    "prompt_cache_lookup_cost: header {argument:?} is not a token count: {error}"
                );
                return ExitCode::FAILURE;
            }
        }
    }
    if headers.is_empty() {
        headers = vec![0, 512];
    }
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    for header in headers {
        for row in prompt_cache_lookup_cost(header) {
            println!(
                "BLOCK_INDEX header={} entries={} lookup_hit_ns={} lookup_miss_ns={} take_hit_ns={} take_miss_ns={} scan_hit_ns={} scan_miss_ns={} index_bytes={} index_bytes_per_entry={} index_nodes={} bloom_bytes_per_entry={} profile={profile}",
                row.shared_header,
                row.entries,
                row.lookup_hit_ns,
                row.lookup_miss_ns,
                row.take_hit_ns,
                row.take_miss_ns,
                row.scan_hit_ns,
                row.scan_miss_ns,
                row.index_bytes,
                row.index_bytes / row.entries,
                row.index_nodes,
                row.bloom_bytes / row.entries,
            );
        }
    }
    ExitCode::SUCCESS
}
