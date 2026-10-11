use std::env;

struct Shape {
    rows: u64,
    tile_rows: u64,
    kv_heads: u64,
    head_dim: u64,
    query_groups: u64,
    simdgroups: u64,
}

fn blocks_of_tile(shape: &Shape, tile: u64, block: u64) -> (u64, u64) {
    let keys_end = ((tile + 1) * shape.tile_rows).min(shape.rows);
    let blocks = keys_end.div_ceil(block);
    let fragment_keys: u64 = (0..blocks).map(|index| (keys_end - index * block).min(block).div_ceil(8) * 8).sum();
    (blocks, fragment_keys)
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    assert!(arguments.len() == 2, "usage: tg_bytes <block keys: 64|32>");
    let block: u64 = arguments[1].parse().expect("block");
    let shape = Shape { rows: 1000, tile_rows: 8, kv_heads: 8, head_dim: 64, query_groups: 2, simdgroups: 2 };
    let half_dim = shape.head_dim / 2;
    let tiles = shape.rows.div_ceil(shape.tile_rows);
    let tile_vectors = (shape.tile_rows / 8) * shape.query_groups * 8;
    let (mut visits, mut keys_total) = (0u64, 0u64);
    for tile in 0..tiles {
        let (blocks, fragment_keys) = blocks_of_tile(&shape, tile, block);
        visits += blocks;
        keys_total += fragment_keys;
    }
    let threadgroups = shape.kv_heads * tiles;
    let key_bytes_per_key = 2 * half_dim * 4;
    let value_bytes_per_key = shape.head_dim * 4;
    let device_key = keys_total * shape.kv_heads * key_bytes_per_key;
    let device_value = keys_total * shape.kv_heads * value_bytes_per_key;
    let query_stage_bytes = shape.query_groups * 2 * 8 * (half_dim + 8) * 4;
    let query_stage_read_per_block = shape.simdgroups * (half_dim / 8 / 2) * 2 * shape.query_groups * 2 * 256 * 2 / 2;
    let score_bytes_per_block = tile_vectors * block * 4;
    println!("block {block} keys, granite step 0 (1000 rows, tile {} rows, {} kv heads, {threadgroups} threadgroups per op):", shape.tile_rows, shape.kv_heads);
    println!("  block visits per op: {} ({:.3} per threadgroup)", visits * shape.kv_heads, (visits * shape.kv_heads) as f64 / threadgroups as f64);
    println!("  keys loaded per op summed over threadgroups (fragment-rounded): {}", keys_total * shape.kv_heads);
    println!("  device bytes read per op by direct K fragment loads: {device_key} ({:.3} per threadgroup)", device_key as f64 / threadgroups as f64);
    println!("  device bytes read per op by direct V fragment loads: {device_value} ({:.3} per threadgroup)", device_value as f64 / threadgroups as f64);
    println!("  a staged copy reads the same device bytes once and adds the same bytes of threadgroup stores and threadgroup fragment loads: K {device_key}, V {device_value} per op");
    println!("  query stage (once per threadgroup): {query_stage_bytes} bytes written, read per block by fragment loads {query_stage_read_per_block}");
    println!("  score tile bytes per block: {score_bytes_per_block} written by the MMA stores, read by the softmax, written as weights, read by the P.V weight loads");
}
