//! Diagnostic: does a checkpoint's bound program expose per-layer routed
//! boundaries (the pre-gather requirement), and how many expert stacks does
//! it have? Reads only the directory; the expert payload stays demand-paged.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;

use memmap2::Mmap;
use proxima_model_interop::bind_checkpoint;

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let file = File::open(&path).expect("open");
    let map = unsafe { Mmap::map(&file) }.expect("mmap");
    let parsed = proxima_gguf::pipe::parse_complete(&map).expect("parse");
    let bound = bind_checkpoint(&parsed, &map).expect("bind");
    let stacks = parsed
        .tensors
        .iter()
        .filter(|tensor| tensor.name.contains("_exps.weight") && tensor.dims.len() == 3)
        .count();
    println!(
        "path={path} layers={} router_roots={} moe_diagnostics={} expert_stacks={}",
        parsed.tensors.iter().filter(|t| t.name.starts_with("blk.")).count(),
        bound.router_roots.len(),
        bound.moe_layer_diagnostics.len(),
        stacks
    );
}
