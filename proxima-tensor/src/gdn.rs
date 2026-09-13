//! Backend-neutral GDN prefill recurrence inputs and reference operation.

use crate::error::TensorError;

/// Concrete dimensions of one caller-buffered GDN prefill scan. `kv_heads`
/// divides `heads` (`heads / kv_heads` is the GQA group size); the
/// non-GQA case is `kv_heads == heads`. `query`/`key` are sized by
/// `kv_heads` (the model's own pre-`repeat_kv_heads` tensors,
/// `BoundOpKind::GatedDeltaNet`'s own doc); `value`/`gate`/`beta`/`state`/
/// `output` are sized by `heads`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnPrefillShape {
    pub positions: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
}

impl GdnPrefillShape {
    #[must_use]
    pub const fn state_len(self) -> usize {
        self.key_dim * self.value_dim * self.heads
    }

    #[must_use]
    pub const fn output_len(self) -> usize {
        self.positions * self.value_dim * self.heads
    }
}

/// Borrowed inputs and caller-owned outputs for a GDN prefill scan.
pub struct GdnPrefillScan<'buffer> {
    pub shape: GdnPrefillShape,
    pub query: &'buffer [f32],
    pub key: &'buffer [f32],
    pub value: &'buffer [f32],
    pub gate: &'buffer [f32],
    pub beta: &'buffer [f32],
    pub inv_sqrt_key_dim: f32,
    pub state: &'buffer mut [f32],
    pub output: &'buffer mut [f32],
}

fn checked_product(left: usize, right: usize) -> Result<usize, TensorError> {
    left.checked_mul(right)
        .ok_or(TensorError::InvalidGdnPrefillShape {
            reason: "dimension product overflowed usize",
        })
}

fn require_buffer(buffer: &'static str, found: usize, expected: usize) -> Result<(), TensorError> {
    if found != expected {
        return Err(TensorError::GdnPrefillBufferSizeMismatch {
            buffer,
            expected,
            found,
        });
    }
    Ok(())
}

/// Executes the reference recurrence for every position in order.
pub fn run_gdn_prefill_scan(scan: GdnPrefillScan<'_>) -> Result<(), TensorError> {
    let GdnPrefillShape {
        positions,
        key_dim,
        value_dim,
        heads,
        kv_heads,
    } = scan.shape;
    if positions == 0 || key_dim == 0 || value_dim == 0 || heads == 0 || kv_heads == 0 {
        return Err(TensorError::InvalidGdnPrefillShape {
            reason: "all dimensions must be nonzero",
        });
    }
    if heads % kv_heads != 0 {
        return Err(TensorError::InvalidGdnPrefillShape {
            reason: "heads must be an exact multiple of kv_heads",
        });
    }
    let group = heads / kv_heads;

    let position_heads = checked_product(positions, heads)?;
    let key_heads = checked_product(key_dim, kv_heads)?;
    let value_heads = checked_product(value_dim, heads)?;
    let query_len = checked_product(positions, key_heads)?;
    let value_len = checked_product(positions, value_heads)?;
    let state_len = checked_product(key_dim, value_heads)?;
    require_buffer("query", scan.query.len(), query_len)?;
    require_buffer("key", scan.key.len(), query_len)?;
    require_buffer("value", scan.value.len(), value_len)?;
    require_buffer("gate", scan.gate.len(), position_heads)?;
    require_buffer("beta", scan.beta.len(), position_heads)?;
    require_buffer("state", scan.state.len(), state_len)?;
    require_buffer("output", scan.output.len(), value_len)?;

    for position in 0..positions {
        for head in 0..heads {
            // llama.cpp's own fused Metal kernel mod-broadcasts a flat query
            // row against `kv_heads` (`gated_delta_net.metal:33-34`); this
            // crate's own axis order makes `kv_heads` the SLOW half of the
            // `(kv_heads, group)` pair rather than the fast one, so the
            // equivalent recovery is `head / group`, not `head % kv_heads`
            // (`BoundOpKind::GatedDeltaNet`'s own doc on this convention).
            let kv_head = head / group;
            let head_offset = position * heads + head;
            let decay = libm::expf(scan.gate[head_offset]);
            for value_index in 0..value_dim {
                let value_offset = position * value_heads + value_index * heads + head;
                let mut predicted = 0.0_f32;
                for key_index in 0..key_dim {
                    let state_offset = key_index * value_heads + value_index * heads + head;
                    let key_offset = position * key_heads + key_index * kv_heads + kv_head;
                    predicted += scan.state[state_offset] * decay * scan.key[key_offset];
                }
                let delta = (scan.value[value_offset] - predicted) * scan.beta[head_offset];
                let mut readout = 0.0_f32;
                for key_index in 0..key_dim {
                    let state_offset = key_index * value_heads + value_index * heads + head;
                    let key_offset = position * key_heads + key_index * kv_heads + kv_head;
                    scan.state[state_offset] =
                        scan.state[state_offset] * decay + scan.key[key_offset] * delta;
                    let query_offset = position * key_heads + key_index * kv_heads + kv_head;
                    readout += scan.state[state_offset]
                        * (scan.query[query_offset] * scan.inv_sqrt_key_dim);
                }
                scan.output[value_offset] = readout;
            }
        }
    }
    Ok(())
}
