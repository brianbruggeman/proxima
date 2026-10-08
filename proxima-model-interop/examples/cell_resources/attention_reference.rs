use omega::CapturedDispatch;

const ENTRY_PREFIX: &str = "omega_cached_attention_";
const QUERY_EVEN: usize = 0;
const QUERY_ODD: usize = 1;
const KEY_EVEN: usize = 4;
const KEY_ODD: usize = 5;
const VALUE: usize = 7;
const LIVE: usize = 8;

pub struct AttentionShape {
    kv_heads: usize,
    groups: usize,
    head_dim: usize,
    scale: f64,
    cached_lower: i128,
    new_upper: i128,
}

impl AttentionShape {
    pub fn from_entry(entry: &str) -> Option<Self> {
        let fields: Vec<&str> = entry.strip_prefix(ENTRY_PREFIX)?.split('_').collect();
        let field = |prefix: &str| fields.iter().find_map(|text| text.strip_prefix(prefix));
        let scale_bits = u32::from_str_radix(field("s")?, 16).ok()?;
        Some(Self {
            kv_heads: field("h")?.parse().ok()?,
            groups: field("g")?.parse().ok()?,
            head_dim: field("d")?.parse().ok()?,
            scale: f64::from(f32::from_bits(scale_bits)),
            cached_lower: -field("ln")?.parse::<i128>().ok()?,
            new_upper: field("up")?.parse().ok()?,
        })
    }
}

fn floats_at(dispatch: &CapturedDispatch, binding: usize) -> Vec<f32> {
    dispatch
        .bound_buffer_bytes_at(binding)
        .unwrap_or_else(|| panic!("binding {binding} is not a readable buffer"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn dot(left: &[f32], right: &[f32]) -> f64 {
    left.iter().zip(right).map(|(a_value, b_value)| f64::from(*a_value) * f64::from(*b_value)).sum()
}

struct Planes {
    query_even: Vec<f32>,
    query_odd: Vec<f32>,
    key_even: Vec<f32>,
    key_odd: Vec<f32>,
    value: Vec<f32>,
}

impl Planes {
    fn read(dispatch: &CapturedDispatch) -> Self {
        assert_eq!(floats_at(dispatch, LIVE)[0], 0.0, "the reference reads the in-graph keys only");
        Self {
            query_even: floats_at(dispatch, QUERY_EVEN),
            query_odd: floats_at(dispatch, QUERY_ODD),
            key_even: floats_at(dispatch, KEY_EVEN),
            key_odd: floats_at(dispatch, KEY_ODD),
            value: floats_at(dispatch, VALUE),
        }
    }
}

fn attend_vector(planes: &Planes, shape: &AttentionShape, rows: usize, row: usize, kv_head: usize, group: usize, out: &mut [f32]) {
    let half_dim = shape.head_dim / 2;
    let query_index = (row * shape.kv_heads + kv_head) * shape.groups + group;
    let query_even = &planes.query_even[query_index * half_dim..][..half_dim];
    let query_odd = &planes.query_odd[query_index * half_dim..][..half_dim];
    let lowest = (row as i128 + shape.cached_lower).max(0) as usize;
    let highest = (row as i128 + shape.new_upper).min(rows as i128 - 1) as usize;
    let scores: Vec<f64> = (lowest..=highest)
        .map(|key| {
            let offset = key * shape.kv_heads * half_dim + kv_head * half_dim;
            shape.scale
                * (dot(query_even, &planes.key_even[offset..][..half_dim])
                    + dot(query_odd, &planes.key_odd[offset..][..half_dim]))
        })
        .collect();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    for (dimension, slot) in out.iter_mut().enumerate().take(shape.head_dim) {
        let accumulated: f64 = weights
            .iter()
            .zip(lowest..=highest)
            .map(|(weight, key)| {
                weight * f64::from(planes.value[key * shape.kv_heads * shape.head_dim + kv_head * shape.head_dim + dimension])
            })
            .sum();
        *slot = (accumulated / total) as f32;
    }
}

pub fn reference_output(dispatch: &CapturedDispatch, shape: &AttentionShape) -> Vec<f32> {
    let rows = dispatch.extents[0] as usize;
    let planes = Planes::read(dispatch);
    let mut output = vec![0.0_f32; rows * shape.kv_heads * shape.groups * shape.head_dim];
    for row in 0..rows {
        for kv_head in 0..shape.kv_heads {
            for group in 0..shape.groups {
                let query_index = (row * shape.kv_heads + kv_head) * shape.groups + group;
                let target = &mut output[query_index * shape.head_dim..][..shape.head_dim];
                attend_vector(&planes, shape, rows, row, kv_head, group, target);
            }
        }
    }
    output
}

pub fn floats_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect()
}

pub fn agreement(reference: &[f32], other: &[f32]) -> String {
    let mut dot_product = 0.0_f64;
    let mut reference_norm = 0.0_f64;
    let mut other_norm = 0.0_f64;
    let mut difference_norm = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut largest = 0.0_f64;
    for (left, right) in reference.iter().zip(other) {
        let (left, right) = (f64::from(*left), f64::from(*right));
        dot_product += left * right;
        reference_norm += left * left;
        other_norm += right * right;
        difference_norm += (left - right) * (left - right);
        max_abs = max_abs.max((left - right).abs());
        largest = largest.max(left.abs());
    }
    format!(
        "elements={} max_abs={max_abs:e} max_abs_over_largest={:e} cosine={:.9} rel_l2={:e}",
        reference.len(),
        max_abs / largest.max(f64::MIN_POSITIVE),
        dot_product / (reference_norm.sqrt() * other_norm.sqrt()).max(f64::MIN_POSITIVE),
        (difference_norm / reference_norm.max(f64::MIN_POSITIVE)).sqrt()
    )
}
