//! The two diffusion graph families this workspace runs, built as sans-IO
//! [`Op`] programs over tensors a checkpoint header names: a transformer
//! backbone (prelude, stream blocks, single blocks, final layer) and a
//! convolutional autoencoder decoder. Nothing here names a backend or a
//! concrete element type; every activation type is a parameter and every
//! weight leaf records the checkpoint tensor it binds.

use std::collections::BTreeMap;

use proxima_numeric::DType;
use proxima_tensor::op::{Extent, NodeId, Op};
use proxima_tensor::QuantizedBlock;
use proxima_tensor::spec::input_leaf;

mod dit;
mod load;
pub mod png;
mod settings;
mod vae;

pub use dit::{
    DitConfig, DitInputs, DitOutput, build_dit, build_double_block, build_single_block,
};
pub use load::{Checkpoint, WeightData, Weights, encode, load_weights};
pub use png::{PngImage, read_png, write_png};
pub use settings::DiffusionSettings;
pub use vae::{
    ConvTaps, VaeConfig, VaeOutput, build_conv3x3, build_vae_decode, latent_grouped,
    latent_statistics, latent_unnormalise,
};

/// One weight leaf's source: the checkpoint tensor it reads and, when the leaf
/// is a row slice of that tensor, the half-open row range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightBinding {
    pub leaf: String,
    pub tensor: String,
    pub rows: Option<(u32, u32)>,
    pub columns: Option<(u32, u32)>,
    pub tap: Option<(u32, u32)>,
    /// The whole 3x3 kernel as one `[out_g, out_c, 3, 3, in_g, in_c]` leaf
    /// ([`Self::weight_taps`]) rather than one leaf per kernel position.
    pub taps: bool,
}

/// One host-built activation leaf: a rope table, a gather index, a time
/// sinusoid. `values` is already in the leaf's declared dtype.
#[derive(Debug, Clone, PartialEq)]
pub struct HostLeaf {
    pub name: String,
    pub dtype: DType,
    pub extent: u32,
    pub values: Vec<f32>,
}

/// The graph under construction: the program, the weight leaves it needs from
/// a checkpoint, and the activation leaves the caller fills.
#[derive(Debug)]
pub struct Graph {
    pub program: Vec<Op>,
    pub weights: Vec<WeightBinding>,
    pub hosts: Vec<HostLeaf>,
    host_nodes: BTreeMap<String, NodeId>,
    activation: DType,
}

impl Graph {
    #[must_use]
    pub fn new(activation: DType) -> Self {
        Self {
            program: Vec::new(),
            weights: Vec::new(),
            hosts: Vec::new(),
            host_nodes: BTreeMap::new(),
            activation,
        }
    }

    #[must_use]
    pub const fn activation(&self) -> DType {
        self.activation
    }

    /// One activation leaf the caller binds per call.
    pub fn host(&mut self, name: &str, shape: &[u32], values: Vec<f32>) -> NodeId {
        self.leaf(name, self.activation, shape, values)
    }

    /// One gather index leaf: integer values the backend carries as floats.
    pub fn host_index(&mut self, name: &str, extent: u32, values: Vec<f32>) -> NodeId {
        self.leaf(name, DType::INT32, &[extent], values)
    }

    fn leaf(&mut self, name: &str, dtype: DType, shape: &[u32], values: Vec<f32>) -> NodeId {
        if let Some(node) = self.host_nodes.get(name) {
            return *node;
        }
        let extents = shape.iter().map(|&size| Extent::Static(size)).collect();
        let node = input_leaf(&mut self.program, dtype, extents, name);
        self.host_nodes.insert(name.to_string(), node);
        self.hosts.push(HostLeaf {
            name: name.to_string(),
            dtype,
            extent: shape.iter().product(),
            values,
        });
        node
    }

    /// One gather index leaf, deduplicated by name: an `i32` value list.
    pub fn index_leaf(&mut self, name: &str, values: Vec<i32>) -> NodeId {
        let extent = values.len() as u32;
        let values = values.into_iter().map(|value| value as f32).collect();
        self.host_index(name, extent, values)
    }

    /// One weight leaf: `tensor` is the checkpoint name, `shape` the leaf's
    /// declared extents (the checkpoint tensor's layout, possibly re-viewed as
    /// more axes), and `rows` the tensor's row slice when the leaf binds part
    /// of a fused projection.
    pub fn weight(&mut self, tensor: &str, shape: &[u32], rows: Option<(u32, u32)>) -> NodeId {
        self.weight_slice(tensor, shape, rows, None)
    }

    /// [`Self::weight`] over a contiguous column window of the tensor, which
    /// the loader re-encodes as its own contiguous buffer.
    pub fn weight_slice(
        &mut self,
        tensor: &str,
        shape: &[u32],
        rows: Option<(u32, u32)>,
        columns: Option<(u32, u32)>,
    ) -> NodeId {
        self.weight_window(tensor, shape, rows, columns, None, false)
    }

    /// The whole 3x3 kernel as ONE `[out_g, out_c, 3, 3, in_g, in_c]` leaf, the
    /// weight [`super::vae::conv`]'s three-block form contracts. The bytes are
    /// the checkpoint's `[out, in, 3, 3]` reordered to `[out, 3, 3, in]` by the
    /// loader's own kernel permutation, which is that shape's element order
    /// exactly -- a reshape, not a repack.
    pub fn weight_taps(&mut self, tensor: &str, shape: &[u32]) -> NodeId {
        self.weight_window(tensor, shape, None, None, None, true)
    }

    /// One tap of a 3x3 kernel as its own `[out, in]` leaf: the loader
    /// extracts that kernel position, so the reduce addresses the weight
    /// through single-term axes only.
    pub fn weight_tap(
        &mut self,
        tensor: &str,
        shape: &[u32],
        tap: (u32, u32),
        in_groups: u32,
        in_width: u32,
    ) -> NodeId {
        let _ = (in_groups, in_width);
        self.weight_window(tensor, shape, None, None, Some(tap), false)
    }

    fn weight_window(
        &mut self,
        tensor: &str,
        shape: &[u32],
        rows: Option<(u32, u32)>,
        columns: Option<(u32, u32)>,
        tap: Option<(u32, u32)>,
        taps: bool,
    ) -> NodeId {
        let mut leaf = match rows {
            Some((start, end)) => format!("{tensor}@rows{start}..{end}"),
            None => tensor.to_string(),
        };
        if let Some((start, end)) = columns {
            leaf = format!("{leaf}@cols{start}..{end}");
        }
        if let Some((row, column)) = tap {
            leaf = format!("{leaf}@tap{row}.{column}");
        }
        if taps {
            leaf = format!("{leaf}@taps3");
        }
        if let Some(existing) = self.weights.iter().find(|binding| binding.leaf == leaf) {
            let dtype = self.activation;
            return input_leaf(
                &mut self.program,
                dtype,
                shape.iter().map(|&size| Extent::Static(size)).collect(),
                &existing.leaf,
            );
        }
        let extents = shape.iter().map(|&size| Extent::Static(size)).collect();
        let dtype = self.activation;
        let node = input_leaf(&mut self.program, dtype, extents, &leaf);
        self.weights.push(WeightBinding {
            leaf,
            tensor: tensor.to_string(),
            rows,
            columns,
            tap,
            taps,
        });
        node
    }

    /// Three host leaves joining a `head_rows`-row tensor and a `tail_rows`-row
    /// one along their leading axis: each half's clamp-to-range index and the
    /// mask that picks the head's rows.
    pub fn concat_indices(&mut self, head_rows: u32, tail_rows: u32, name: &str) -> [NodeId; 3] {
        let total = head_rows + tail_rows;
        let mut head = Vec::with_capacity(total as usize);
        let mut tail = Vec::with_capacity(total as usize);
        let mut mask = Vec::with_capacity(total as usize);
        for row in 0..total {
            if row < head_rows {
                head.push(row as f32);
                tail.push(0.0);
                mask.push(1.0);
            } else {
                head.push(0.0);
                tail.push((row - head_rows) as f32);
                mask.push(0.0);
            }
        }
        let head_index = self.host_index(&format!("{name}.head_index"), total, head);
        let tail_index = self.host_index(&format!("{name}.tail_index"), total, tail);
        let head_mask = self.host(&format!("{name}.head_mask"), &[total], mask);
        [head_index, tail_index, head_mask]
    }
}

/// `t * 1000`'s cosine/sine embedding: `channels / 2` frequencies from
/// `10000^(-i / (channels / 2))`, cosine in the first half and sine in the
/// second (ggml's `ggml_timestep_embedding` order).
#[must_use]
pub fn timestep_sinusoid(timestep: f32, channels: u32) -> Vec<f32> {
    let half = channels / 2;
    let mut values = Vec::with_capacity(channels as usize);
    for index in 0..half {
        let exponent = -(index as f32) / half as f32;
        let angle = timestep * 1000.0 * 10_000.0f32.powf(exponent);
        values.push(angle.cos());
    }
    for index in 0..half {
        let exponent = -(index as f32) / half as f32;
        let angle = timestep * 1000.0 * 10_000.0f32.powf(exponent);
        values.push(angle.sin());
    }
    values
}

impl Graph {
    /// The integer buffers the gather leaves bind, positionally matching
    /// [`Self::hosts`].
    #[must_use]
    pub fn int_buffers(&self) -> Vec<Vec<i32>> {
        self.hosts
            .iter()
            .map(|host| {
                if host.dtype == DType::INT32 {
                    host.values.iter().map(|value| *value as i32).collect()
                } else {
                    Vec::new()
                }
            })
            .collect()
    }

    /// Every host leaf's named block, with `overrides` replacing a leaf's
    /// placeholder values (the activations a caller feeds per call).
    #[must_use]
    pub fn host_blocks<'a>(
        &'a self,
        overrides: &[(&str, &'a [f32])],
        integers: &'a [Vec<i32>],
    ) -> Vec<(&'a str, QuantizedBlock<'a>)> {
        self.hosts
            .iter()
            .enumerate()
            .map(|(index, host)| {
                let supplied = overrides
                    .iter()
                    .find(|(name, _)| *name == host.name)
                    .map(|(_, values)| *values);
                let _ = integers;
                let _ = index;
                let block = QuantizedBlock::Float32(supplied.unwrap_or(&host.values));
                (host.name.as_str(), block)
            })
            .collect()
    }
}
