//! Backend-neutral GDN prefill recurrence inputs and reference operation.

use crate::error::TensorError;

/// Concrete dimensions of one caller-buffered GDN prefill scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnPrefillShape {
    pub positions: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub heads: usize,
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
    } = scan.shape;
    if positions == 0 || key_dim == 0 || value_dim == 0 || heads == 0 {
        return Err(TensorError::InvalidGdnPrefillShape {
            reason: "all dimensions must be nonzero",
        });
    }

    let position_heads = checked_product(positions, heads)?;
    let key_heads = checked_product(key_dim, heads)?;
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
            let head_offset = position * heads + head;
            let decay = libm::expf(scan.gate[head_offset]);
            for value_index in 0..value_dim {
                let value_offset = position * value_heads + value_index * heads + head;
                let mut predicted = 0.0_f32;
                for key_index in 0..key_dim {
                    let state_offset = key_index * value_heads + value_index * heads + head;
                    let key_offset = position * key_heads + key_index * heads + head;
                    predicted += scan.state[state_offset] * decay * scan.key[key_offset];
                }
                let delta = (scan.value[value_offset] - predicted) * scan.beta[head_offset];
                let mut readout = 0.0_f32;
                for key_index in 0..key_dim {
                    let state_offset = key_index * value_heads + value_index * heads + head;
                    let key_offset = position * key_heads + key_index * heads + head;
                    scan.state[state_offset] =
                        scan.state[state_offset] * decay + scan.key[key_offset] * delta;
                    let query_offset = position * key_heads + key_index * heads + head;
                    readout += scan.state[state_offset]
                        * (scan.query[query_offset] * scan.inv_sqrt_key_dim);
                }
                scan.output[value_offset] = readout;
            }
        }
    }
    Ok(())
}
