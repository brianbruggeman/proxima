use std::env;

struct Shape {
    total_rows: i64,
    kv_heads: i64,
    query_groups: i64,
    head_dim: i64,
    tile_rows: i64,
    block: i64,
    cached_lower: i64,
    new_upper: i64,
}

struct Traffic {
    tiles: i64,
    block_visits: i64,
    keys_loaded: i64,
    key_bytes: i64,
    value_bytes: i64,
    executed_flops: f64,
}

fn parse(arguments: &[String], index: usize) -> i64 {
    arguments[index].parse().expect("integer argument")
}

fn visits_of_tile(shape: &Shape, tile: i64) -> (i64, i64) {
    let row0 = tile * shape.tile_rows;
    let last_row = row0 + shape.tile_rows.min(shape.total_rows - row0) - 1;
    let total_aligned = shape.total_rows & !7;
    let new_first = 0.max(row0.saturating_add(shape.cached_lower));
    let new_end = shape.total_rows.min(last_row + shape.new_upper + 1);
    let new_start = new_first & !7;
    let mma_end = new_end.min(total_aligned);
    let blocks = if mma_end > new_start {
        (mma_end - new_start + shape.block - 1) / shape.block
    } else {
        0
    };
    let mut visits = 0;
    let mut keys = 0;
    for step in 0..blocks {
        let key0 = new_start + step * shape.block;
        let columns = shape.block.min(mma_end - key0);
        visits += 1;
        keys += (columns + 7) / 8 * 8;
    }
    (visits, keys)
}

fn traffic(shape: &Shape) -> Traffic {
    let tiles = (shape.total_rows + shape.tile_rows - 1) / shape.tile_rows;
    let (mut block_visits, mut keys_loaded) = (0, 0);
    for tile in 0..tiles {
        let (visits, keys) = visits_of_tile(shape, tile);
        block_visits += visits;
        keys_loaded += keys;
    }
    let vectors = shape.tile_rows * shape.query_groups;
    let per_key = shape.head_dim * 4;
    Traffic {
        tiles,
        block_visits: block_visits * shape.kv_heads,
        keys_loaded: keys_loaded * shape.kv_heads,
        key_bytes: keys_loaded * shape.kv_heads * per_key,
        value_bytes: keys_loaded * shape.kv_heads * per_key,
        executed_flops: 4.0 * vectors as f64 * keys_loaded as f64 * shape.kv_heads as f64 * shape.head_dim as f64,
    }
}

fn useful_flops(shape: &Shape) -> f64 {
    let visible: i64 = (0..shape.total_rows)
        .map(|row| {
            let low = 0.max(row.saturating_add(shape.cached_lower));
            let high = row + shape.new_upper;
            (high.min(shape.total_rows - 1) - low + 1).max(0)
        })
        .sum();
    4.0 * shape.head_dim as f64 * visible as f64 * (shape.kv_heads * shape.query_groups) as f64
}

fn square_flops(shape: &Shape) -> f64 {
    4.0 * (shape.total_rows as f64).powi(2) * (shape.kv_heads * shape.query_groups) as f64 * shape.head_dim as f64
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let shape = Shape {
        total_rows: parse(&arguments, 1),
        kv_heads: parse(&arguments, 2),
        query_groups: parse(&arguments, 3),
        head_dim: parse(&arguments, 4),
        tile_rows: parse(&arguments, 5),
        block: parse(&arguments, 6),
        cached_lower: parse(&arguments, 7),
        new_upper: parse(&arguments, 8),
    };
    let result = traffic(&shape);
    let kv_bytes = result.key_bytes + result.value_bytes;
    let useful = useful_flops(&shape);
    print!(
        "tile_rows={} block={} tiles_per_kv_head={} threadgroups={} block_visits={} keys_loaded={} kv_bytes={} kv_mb={:.3} executed_mma_gflop={:.4} useful_gflop={:.4} square_gflop={:.4}",
        shape.tile_rows,
        shape.block,
        result.tiles,
        result.tiles * shape.kv_heads,
        result.block_visits,
        result.keys_loaded,
        kv_bytes,
        kv_bytes as f64 / 1e6,
        result.executed_flops / 1e9,
        useful / 1e9,
        square_flops(&shape) / 1e9
    );
    if let Some(time_us) = arguments.get(9).and_then(|text| text.parse::<f64>().ok()) {
        let seconds = time_us / 1e6;
        print!(
            " time_us={time_us:.1} kv_gbps={:.1} executed_tflops={:.3} useful_tflops={:.3} square_tflops={:.3}",
            kv_bytes as f64 / seconds / 1e9,
            result.executed_flops / seconds / 1e12,
            useful / seconds / 1e12,
            square_flops(&shape) / seconds / 1e12
        );
    }
    println!();
}
