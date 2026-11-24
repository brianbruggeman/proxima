//! The two diffusion graph families this workspace runs, built as sans-IO
//! [`Op`] programs over tensors a checkpoint header names: a transformer
//! backbone (prelude, stream blocks, single blocks, final layer) and a
//! convolutional autoencoder decoder. Nothing here names a backend or a
//! concrete element type; every activation type is a parameter and every
//! weight leaf records the checkpoint tensor it binds.

use std::collections::BTreeMap;
use std::ops::Range;

use proxima_numeric::{DType, NumericError, NumericFormat, PackedRows};
use proxima_tensor::QuantizedBlock;
use proxima_tensor::op::{Extent, NodeId, Op};
use proxima_tensor::spec::input_leaf;

mod dit;
mod host;
mod load;
pub mod png;
mod settings;
mod vae;

pub use dit::{
    DitConfig, DitInputs, DitOutput, attention_scale, build_dit, build_double_block,
    build_single_block, head_norm_count,
};
pub use host::{Decoder, Denoiser};
pub use load::{Checkpoint, WeightData, Weights, load_weights};
pub use png::{PngImage, full_scale, read_png, write_png};
pub use settings::DiffusionSettings;
pub use vae::{ConvTaps, VaeConfig, VaeOutput, build_vae_decode, build_window_same};

/// One weight leaf's source: the checkpoint tensor it reads and, when the leaf
/// is a row slice of that tensor, the half-open row range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightBinding {
    pub leaf: String,
    pub tensor: String,
    pub rows: Option<(u32, u32)>,
    pub columns: Option<(u32, u32)>,
    pub tap: Option<(u32, u32)>,
    /// The whole window x window kernel as one `[out_g, out_c, window, window, in_g, in_c]` leaf
    /// ([`Self::weight_taps`]) rather than one leaf per kernel position.
    pub taps: bool,
}

/// One host-built activation leaf: a rope table, a gather index, a time
/// sinusoid. `values` is packed in the leaf's declared format.
#[derive(Debug, Clone, PartialEq)]
pub struct HostLeaf {
    pub name: String,
    pub dtype: DType,
    pub extent: u32,
    pub values: PackedRows,
    /// `values` re-encoded in the host element when it is in another format, so a bind reads
    /// decoded values; `None` when `values` already is.
    pub(crate) widened: Option<PackedRows>,
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
    activation_format: NumericFormat,
    index: DType,
    settings: DiffusionSettings,
    steps: Vec<(String, Range<usize>)>,
    faults: Vec<String>,
}

impl Graph {
    /// A graph over the environment-resolved settings.
    ///
    /// # Errors
    ///
    /// The settings' load failure rendered as text.
    pub fn new(activation: DType) -> Result<Self, String> {
        let settings = DiffusionSettings::try_resolved().map_err(|why| why.to_string())?;
        Self::with_settings(activation, settings)
    }

    /// A graph over explicit settings.
    ///
    /// # Errors
    ///
    /// A message when `activation` has no format or `gather_index` is not a scalar format.
    pub fn with_settings(activation: DType, settings: DiffusionSettings) -> Result<Self, String> {
        let activation_format = activation
            .format()
            .ok_or_else(|| String::from("the graph's activation dtype has no numeric format"))?;
        Ok(Self {
            program: Vec::new(),
            weights: Vec::new(),
            hosts: Vec::new(),
            host_nodes: BTreeMap::new(),
            activation,
            activation_format,
            index: settings.gather_index_dtype()?,
            settings,
            steps: Vec::new(),
            faults: Vec::new(),
        })
    }

    /// The format an activation leaf's values are packed in.
    #[must_use]
    pub const fn activation_format(&self) -> &NumericFormat {
        &self.activation_format
    }

    /// The format a gather index leaf's values are packed in.
    #[must_use]
    pub const fn index_format(&self) -> &NumericFormat {
        &self.settings.gather_index
    }

    /// An empty activation run, the placeholder for a leaf the caller binds per call.
    pub fn unbound(&self) -> Result<PackedRows, NumericError> {
        PackedRows::new(&self.activation_format)
    }

    /// Packing failures recorded by [`Self::host`] and [`Self::host_index`];
    /// [`Self::host_blocks`] and the stages' constructors report them.
    #[must_use]
    pub fn faults(&self) -> &[String] {
        &self.faults
    }

    fn check_faults(&self) -> Result<(), String> {
        match self.faults.first() {
            Some(first) => Err(first.clone()),
            None => Ok(()),
        }
    }

    /// The settings this graph's builders read their epsilons and formats from.
    #[must_use]
    pub const fn settings(&self) -> &DiffusionSettings {
        &self.settings
    }

    /// Each builder call in order, with the `program` index range it appended
    /// (leaf `Input`s included). A profiler reads this to name the builder step
    /// behind a timed node.
    #[must_use]
    pub fn steps(&self) -> &[(String, Range<usize>)] {
        &self.steps
    }

    fn record_step(&mut self, label: String, start: usize) {
        self.steps.push((label, start..self.program.len()));
    }

    #[must_use]
    pub const fn activation(&self) -> DType {
        self.activation
    }

    /// One activation leaf the caller binds per call.
    ///
    /// `values` is the caller's `PackedRows::from_values(graph.activation_format(), ..)`; a
    /// failure is recorded in [`Self::faults`] rather than returned per call.
    pub fn host(
        &mut self,
        name: &str,
        shape: &[u32],
        values: Result<PackedRows, NumericError>,
    ) -> NodeId {
        self.leaf(name, self.activation, shape, values)
    }

    /// One gather index leaf: integer values the backend carries as floats, packed by the caller
    /// in [`Self::index_format`].
    pub fn host_index(
        &mut self,
        name: &str,
        extent: u32,
        values: Result<PackedRows, NumericError>,
    ) -> NodeId {
        self.leaf(name, self.index, &[extent], values)
    }

    fn leaf(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[u32],
        values: Result<PackedRows, NumericError>,
    ) -> NodeId {
        if let Some(node) = self.host_nodes.get(name) {
            return *node;
        }
        let extents = shape.iter().map(|&size| Extent::Static(size)).collect();
        let node = input_leaf(&mut self.program, dtype, extents, name);
        self.host_nodes.insert(name.to_string(), node);
        let packed = values
            .map_err(|why| format!("host leaf {name}: {why}"))
            .and_then(|values| {
                host::widened(&format!("host leaf {name}"), &values).map(|wide| (values, wide))
            });
        match packed {
            Ok((values, widened)) => self.hosts.push(HostLeaf {
                name: name.to_string(),
                dtype,
                extent: shape.iter().product(),
                values,
                widened,
            }),
            Err(why) => self.faults.push(why),
        }
        node
    }

    /// One gather index leaf, deduplicated by name: an `i32` value list.
    pub fn index_leaf(&mut self, name: &str, values: Vec<i32>) -> NodeId {
        let extent = values.len() as u32;
        let wide: Vec<_> = values.into_iter().map(|value| value as _).collect();
        let packed = PackedRows::from_values(self.index_format(), &wide);
        self.host_index(name, extent, packed)
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

    /// The whole window x window kernel as ONE
    /// `[out_g, out_c, window, window, in_g, in_c]` leaf, the weight
    /// [`super::vae::conv`]'s three-block form contracts. The bytes are the
    /// checkpoint's `[out, in, window, window]` reordered to
    /// `[out, window, window, in]` by the loader's own kernel permutation,
    /// which is that shape's element order exactly -- a reshape, not a repack.
    pub fn weight_taps(&mut self, tensor: &str, shape: &[u32]) -> NodeId {
        self.weight_window(tensor, shape, None, None, None, true)
    }

    /// One tap of a window x window kernel as its own `[out, in]` leaf: the loader
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
            leaf = format!("{leaf}@taps");
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
                head.push(row as _);
                tail.push(0.0);
                mask.push(1.0);
            } else {
                head.push(0.0);
                tail.push((row - head_rows) as _);
                mask.push(0.0);
            }
        }
        let head_packed = PackedRows::from_values(self.index_format(), &head);
        let tail_packed = PackedRows::from_values(self.index_format(), &tail);
        let mask_packed = PackedRows::from_values(&self.activation_format, &mask);
        let head_index = self.host_index(&format!("{name}.head_index"), total, head_packed);
        let tail_index = self.host_index(&format!("{name}.tail_index"), total, tail_packed);
        let head_mask = self.host(&format!("{name}.head_mask"), &[total], mask_packed);
        [head_index, tail_index, head_mask]
    }
}

impl Graph {
    /// Every host leaf's named block, bound as a stage binds it: a leaf in another format is
    /// bound through its host-element form, widened once when the leaf was built. `overrides`
    /// replace a leaf's placeholder values (the activations a caller feeds per call) and bind
    /// as given, as a stage's per-call inputs do.
    ///
    /// # Errors
    ///
    /// A recorded packing or widening fault, or a host-element leaf or override whose bytes
    /// cannot be viewed in place, as text.
    pub fn host_blocks<'a>(
        &'a self,
        overrides: &[(&str, &'a PackedRows)],
    ) -> Result<Vec<(&'a str, QuantizedBlock<'a>)>, String> {
        self.check_faults()?;
        self.hosts
            .iter()
            .map(|host| {
                let supplied = overrides.iter().find(|(name, _)| *name == host.name);
                let block = match supplied {
                    Some((_, values)) => host::host_checked(
                        &host.name,
                        QuantizedBlock::Packed {
                            codec: *values.format(),
                            bytes: values.as_bytes(),
                        },
                    )?,
                    None => host::leaf_block(&host.name, &host.values, host.widened.as_ref())?,
                };
                Ok::<_, String>((host.name.as_str(), block))
            })
            .collect()
    }
}
