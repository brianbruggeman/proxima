//! Can omega's emitter bind and emit the REAL 7B forward graph?
//!
//! Every GPU number in `docs/discipline.md` ROWS 72-77 comes from a
//! synthetic 4096x4096 matvec. Nothing outside this workspace's root
//! depends on omega — `proxima-model-interop`, the crate that actually runs
//! a token, has no reference to it — so the kernel being 1.49x off
//! llama.cpp Metal is worth nothing until the real graph goes through it.
//!
//! This needs no GGUF and no weights: `mistral_cached_forward_program`
//! builds the program from architecture parameters alone, and `bind` +
//! `emit` are pure functions of the program. So this answers the first
//! blocking question cheaply — which ops of a real forward can the emitter
//! not produce a kernel for — instead of assuming an answer.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use proxima_tensor::spec::mistral_cached_forward_program;
use proxima_tensor::{bind, infer};

fn main() {
    // openchat-3.5-1210 / Mistral-7B
    const VOCAB: u32 = 32000;
    const EMBEDDING: u32 = 4096;
    const FEED_FORWARD: u32 = 14336;
    const QUERY_HEADS: u32 = 32;
    const KV_HEADS: u32 = 8;
    const HEAD_DIM: u32 = 128;
    const BLOCKS: u32 = 32;

    let (program, logits_root, cache_roots) = match mistral_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        BLOCKS,
    ) {
        Ok(built) => built,
        Err(error) => {
            println!("program build FAILED: {error}");
            return;
        }
    };

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }
    println!("program nodes={} roots={}", program.len(), roots.len());

    // one decode step: one new position, no cached history
    let symbols = [1u64, 0u64];
    let shapes = match infer(&program, &symbols) {
        Ok(shapes) => shapes,
        Err(error) => {
            println!("infer FAILED: {error}");
            return;
        }
    };
    let bound = match bind(&program, &shapes, &roots) {
        Ok(bound) => bound,
        Err(error) => {
            println!("bind FAILED: {error}");
            return;
        }
    };
    println!("bound ops={}", bound.len());

    let no_packed = BTreeMap::new();
    let mut emitted = 0usize;
    let mut failures: BTreeMap<String, usize> = BTreeMap::new();
    let mut first_failure: Option<String> = None;
    // ROW 93: `source_len`/`entry_len` are the measured evidence behind
    // `kernel_cache_key` (`msl.rs`) keying the Metal pipeline cache on the
    // cheap `entry`-shaped fingerprint instead of the full MSL `source` --
    // the ~236x size gap below is what made the OLD `BTreeMap<String,
    // Pipeline>` lookup (`log2(1196)` string comparisons per call, each up
    // to `source_len_max` bytes) real, measured cost, not a guess.
    let mut source_len_total = 0usize;
    let mut entry_len_total = 0usize;
    let mut source_len_max = 0usize;
    for op in &bound {
        match omega::emit(op, &no_packed) {
            Ok(kernel) => {
                emitted += 1;
                source_len_total += kernel.source.len();
                entry_len_total += kernel.entry.len();
                source_len_max = source_len_max.max(kernel.source.len());
            }
            Err(error) => {
                let reason = format!("{error}");
                *failures.entry(reason.clone()).or_insert(0) += 1;
                if first_failure.is_none() {
                    first_failure = Some(format!("node {:?}: {reason}", op.node));
                }
            }
        }
    }

    let failed: usize = failures.values().sum();
    println!(
        "emit: {emitted} ok, {failed} failed, of {} bound ops",
        bound.len()
    );
    println!(
        "source_len avg={:.1} max={} entry_len avg={:.1} (over {emitted} emitted kernels)",
        source_len_total as f64 / emitted.max(1) as f64,
        source_len_max,
        entry_len_total as f64 / emitted.max(1) as f64,
    );
    for (reason, count) in &failures {
        println!("  {count:>5}x  {reason}");
    }
    if let Some(first) = first_failure {
        println!("first failure: {first}");
    }
    // what would actually have to be uploaded, and in what codec. The
    // emitter is not the blocker; the block set is.
    let mut inputs: Vec<(String, usize)> = Vec::new();
    let mut total_elements = 0usize;
    for (position, op) in program.iter().enumerate() {
        if let proxima_tensor::Op::Input { name, .. } = op {
            let node = proxima_tensor::NodeId(position as u32);
            let count: usize = shapes
                .of(node)
                .iter()
                .map(|extent| *extent as usize)
                .product();
            total_elements += count;
            inputs.push((
                name.clone().unwrap_or_else(|| format!("<{position}>")),
                count,
            ));
        }
    }
    inputs.sort_by_key(|(_, count)| core::cmp::Reverse(*count));
    println!(
        "inputs={} total_elements={} (= {:.2} GB as f32, {:.2} GB as q4_k)",
        inputs.len(),
        total_elements,
        total_elements as f64 * 4.0 / 1e9,
        total_elements as f64 * 0.5625 / 1e9
    );
    for (name, count) in inputs.iter().take(8) {
        println!("  {count:>12}  {name}");
    }

    assert_ne!(bound.len(), 0, "degenerate probe: nothing bound");
    assert_eq!(
        failed,
        0,
        "omega cannot emit {failed} of the real forward's {} ops",
        bound.len()
    );
}
