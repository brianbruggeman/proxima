struct Shape {
    name: String,
    kv_heads: f64,
    groups: f64,
    head_dim: f64,
    live_keys: f64,
    splits: f64,
    chunks: f64,
    element_bytes: f64,
}

struct Timing {
    partial_us: f64,
    merge_us: f64,
}

fn megabytes(bytes: f64) -> f64 {
    bytes / 1.0e6
}

fn gigabytes_per_second(bytes: f64, microseconds: f64) -> f64 {
    bytes / (microseconds * 1.0e-6) / 1.0e9
}

fn report(shape: &Shape, timing: &Timing) {
    let query_heads = shape.kv_heads * shape.groups;
    let key_value_unique = 2.0 * shape.live_keys * shape.kv_heads * shape.head_dim * shape.element_bytes;
    let key_value_requested = key_value_unique * shape.groups;
    let query_bytes = query_heads * shape.head_dim * 4.0;
    let scratch_values = query_heads * shape.splits * shape.head_dim * 4.0;
    let scratch_stats = query_heads * shape.splits * 2.0 * 4.0;
    let scratch = scratch_values + scratch_stats;
    let output = query_heads * shape.head_dim * 4.0;
    let threadgroups = query_heads * shape.splits;
    let keys_per_threadgroup = shape.live_keys / shape.splits;
    let bytes_per_threadgroup = keys_per_threadgroup * 2.0 * shape.head_dim * shape.element_bytes + shape.head_dim * 4.0 + (shape.head_dim + 2.0) * 4.0;
    let partial_moved = key_value_unique + query_bytes + scratch;
    let merge_moved = scratch + output;
    let pair_us = timing.partial_us + timing.merge_us;
    let own_partials = scratch + scratch;
    let pair_moved = partial_moved + merge_moved;
    println!("shape {}: kv_heads={} groups={} head_dim={} live_keys={} splits={} chunks={} element_bytes={}", shape.name, shape.kv_heads, shape.groups, shape.head_dim, shape.live_keys, shape.splits, shape.chunks, shape.element_bytes);
    println!("  threadgroups={threadgroups} threads_per_threadgroup={} keys_per_threadgroup={keys_per_threadgroup:.2} keys_per_simdgroup={:.2} bytes_per_threadgroup_requested={bytes_per_threadgroup:.0}", shape.chunks * 32.0, keys_per_threadgroup / shape.chunks);
    println!("  key+value unique MB={:.4} requested-by-every-query-head MB={:.4} query MB={:.4}", megabytes(key_value_unique), megabytes(key_value_requested), megabytes(query_bytes));
    println!("  partial scratch written MB={:.4} (values {:.4}, stats {:.4}) re-read by merge MB={:.4} merge output MB={:.4}", megabytes(scratch), megabytes(scratch_values), megabytes(scratch_stats), megabytes(scratch), megabytes(output));
    println!("  partial moves MB={:.4} -> {:.1} GB/s unique-bytes basis at {:.2} us; key+value unique alone {:.1} GB/s", megabytes(partial_moved), gigabytes_per_second(partial_moved, timing.partial_us), timing.partial_us, gigabytes_per_second(key_value_unique, timing.partial_us));
    println!("  merge moves MB={:.4} -> {:.1} GB/s at {:.2} us", megabytes(merge_moved), gigabytes_per_second(merge_moved, timing.merge_us), timing.merge_us);
    println!("  pair moves MB={:.4}; the split's own partials (written then re-read) are {:.4} MB = {:.1} % of the pair's bytes; pair {:.2} us", megabytes(pair_moved), megabytes(own_partials), 100.0 * own_partials / pair_moved, pair_us);
}

fn parse(arguments: &[String]) -> (Shape, Timing) {
    let number = |index: usize| -> f64 { arguments[index].parse().unwrap_or_else(|_| panic!("argument {index} is a number")) };
    let shape = Shape {
        name: arguments[1].clone(),
        kv_heads: number(2),
        groups: number(3),
        head_dim: number(4),
        live_keys: number(5),
        splits: number(6),
        chunks: number(7),
        element_bytes: number(8),
    };
    (shape, Timing { partial_us: number(9), merge_us: number(10) })
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    assert!(arguments.len() == 11, "usage: decode_bytes <name> <kv_heads> <groups> <head_dim> <live_keys> <splits> <chunks> <element_bytes> <partial_us> <merge_us>");
    let (shape, timing) = parse(&arguments);
    report(&shape, &timing);
}
