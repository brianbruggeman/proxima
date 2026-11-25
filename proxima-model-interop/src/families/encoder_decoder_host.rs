//! Host-side assembly of the generic encoder-decoder builders: read a
//! safetensors checkpoint and its config, build the mel/encoder/cross-KV/
//! decoder graphs, and run a greedy transcript. This is the std half; the
//! graph builders themselves stay alloc-only and model-neutral.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
use proxima_primitives::sync::blocking::Mutex;
use proxima_safetensors::{Manifest, SafetensorsParser};
use proxima_tensor::cpu::{Evaluated, QuantizedBlock};
use proxima_tensor::op::{Extent, NodeId, Op};
use proxima_tensor::spec::{
    CachedLayerRoots, STFT_COS_INPUT, STFT_SIN_INPUT, STFT_WINDOW_INPUT, StftLogMelConfig,
    cross_kv, input_leaf, stft_log_mel, stft_tables,
};
use serde_json::Value;

use crate::bind::retarget_recoded_leaves;
use crate::decode_policy::{DecodeConfig, DecodePolicy};
use crate::families::encoder_decoder::{
    DecoderLayer, DecoderStem, EncoderLayer, EncoderStem, Linear, encoder,
};
use crate::generate::{BackendRuntime, cached_kv_extent, program_hash};
use crate::model_weights::{LeasedWeights, ModelWeights};
use crate::profiles::BindingProfile;
use crate::serving::apply_serving_config;
use crate::serving_settings::ServingSettings;
use crate::source::CheckpointMapping;
use crate::weight_store::{
    device_alignment, emit_weight_store_hit, emit_weight_store_publish, header_prefix, lookup,
    modified_nanos, publish, register_mapping, store_key, unregister_mapping,
};

const CHECKPOINT_FILE: &str = "model.safetensors";

/// The dimensions a checkpoint states, read from its `config.json`.
#[derive(Debug, Clone)]
pub struct Shape {
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_heads: u32,
    pub encoder_head_dim: u32,
    pub decoder_heads: u32,
    pub decoder_head_dim: u32,
    pub width: u32,
    pub mels: u32,
    pub encoder_ffn: u32,
    pub decoder_ffn: u32,
    pub source_positions: u32,
    pub target_positions: u32,
    pub vocab: u32,
    pub eps: f32,
}

impl Shape {
    pub fn from_config(config: &Value) -> Result<Self> {
        let int = |key: &str| -> Result<u32> {
            config
                .get(key)
                .and_then(Value::as_u64)
                .map(|value| value as u32)
                .with_context(|| format!("config.json lacks {key}"))
        };
        let encoder_layers = int("encoder_layers")? as usize;
        let decoder_layers = int("decoder_layers")? as usize;
        let encoder_heads = int("encoder_attention_heads")?;
        let decoder_heads = int("decoder_attention_heads")?;
        let width = int("d_model")?;
        let mels = int("num_mel_bins")?;
        let encoder_ffn = int("encoder_ffn_dim")?;
        let decoder_ffn = int("decoder_ffn_dim")?;
        let source_positions = int("max_source_positions")?;
        let target_positions = int("max_target_positions")?;
        let vocab = int("vocab_size")?;
        if width % encoder_heads != 0 || width % decoder_heads != 0 {
            bail!("d_model {width} is not a multiple of its head counts");
        }
        Ok(Self {
            encoder_layers,
            decoder_layers,
            encoder_heads,
            encoder_head_dim: width / encoder_heads,
            decoder_heads,
            decoder_head_dim: width / decoder_heads,
            width,
            mels,
            encoder_ffn,
            decoder_ffn,
            source_positions,
            target_positions,
            vocab,
            eps: 1e-5,
        })
    }
}

const CACHE_PLANES: [&str; 3] = ["k_even", "k_odd", "v"];

fn cache_plane_name(plane: &str, layer: usize) -> String {
    format!("kv_cache.{layer}.{plane}")
}

fn cache_names(layers: usize) -> impl Iterator<Item = String> {
    (0..layers).flat_map(|layer| CACHE_PLANES.map(|part| cache_plane_name(part, layer)))
}

/// What `Model` keeps of a checkpoint file: its mapping, which the shared
/// binder borrows weights from in place, the tensor directory, and where the
/// tensor data starts.
type Checkpoint = (CheckpointMapping, Manifest, u64);

/// Maps a safetensors file read-only through the same [`CheckpointMapping`] the
/// diffusion loader opens its checkpoints with, then parses its header.
fn read_checkpoint(path: &Path) -> Result<Checkpoint> {
    let mapping = CheckpointMapping::open_file(path).with_context(|| path.display().to_string())?;
    let (manifest, data_start) = parse_checkpoint(mapping.bytes())?;
    Ok((mapping, manifest, data_start))
}

fn parse_checkpoint(bytes: &[u8]) -> Result<(Manifest, u64)> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .and_then(|head| head.try_into().ok())
        .context("safetensors file is shorter than its length prefix")?;
    let header_len = u64::from_le_bytes(prefix) as usize;
    let header = bytes
        .get(..8 + header_len)
        .context("safetensors header runs past the end of the file")?;
    let mut parser = SafetensorsParser::new();
    parser.feed(header);
    parser
        .poll()
        .map_err(|error| anyhow::anyhow!("safetensors header does not parse: {error:?}"))?;
    let SafetensorsParser::TensorData { manifest, .. } = parser else {
        return Err(anyhow::anyhow!("safetensors header did not complete"));
    };
    Ok((manifest, (8 + header_len) as u64))
}

struct Bound<'a> {
    program: Vec<Op>,
    named: BTreeMap<String, Vec<f32>>,
    weights: LeasedWeights<'a, 'a>,
    /// leaves the host feeds each run; every other named leaf is a checkpoint tensor
    runtime_inputs: Vec<String>,
    /// runtime inputs constant across every run of this program, declared
    /// resident beside the weights and released by the owner when the runs end
    call_resident: Vec<String>,
    /// leaves an earlier segment's outputs fill, in the order those outputs arrive
    carried_inputs: Vec<NodeId>,
    shape: Shape,
    /// `Some` asks the next run for per-op GPU time; the run fills the cell
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    op_timings: Option<std::sync::OnceLock<Vec<omega::metal::OpGpuTiming>>>,
}

impl<'a> Bound<'a> {
    fn new(shape: Shape) -> Self {
        Self {
            #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
            op_timings: None,
            program: Vec::new(),
            named: BTreeMap::new(),
            weights: LeasedWeights::detached(),
            runtime_inputs: Vec::new(),
            call_resident: Vec::new(),
            carried_inputs: Vec::new(),
            shape,
        }
    }

    /// Names the leaf `input`, which an earlier segment's output fills. That
    /// value is constant for every run of this program, so the name is
    /// declared resident for the call; the owner releases it when the runs end.
    fn carry(&mut self, input: NodeId) -> Result<String> {
        let Some(Op::Input {
            name: Some(name), ..
        }) = self.program.get(input.0 as usize)
        else {
            bail!("node {input:?} is not a named input leaf");
        };
        self.call_resident.push(name.clone());
        Ok(name.clone())
    }

    fn leaf(&mut self, name: &str, shape: &[u32]) -> NodeId {
        self.runtime_inputs.push(name.into());
        super::encoder_decoder::leaf(&mut self.program, shape, name)
    }

    fn weight(&mut self, name: &str, shape: &[u32]) -> NodeId {
        super::encoder_decoder::leaf(&mut self.program, shape, name)
    }

    /// Binds every checkpoint leaf the program names through the shared
    /// safetensors binder, then retargets the leaves a precision rule recoded.
    fn bind(&mut self, model: &'a Model) -> Result<()> {
        let rules = model.settings.weight_precision_rules()?;
        self.weights = model.weights.bind(
            &model.manifest,
            model.mapping.bytes(),
            model.data_start,
            &self.program,
            &BindingProfile::default(),
            &self.runtime_inputs,
            &rules,
        )?;
        retarget_recoded_leaves(&mut self.program, &self.weights);
        Ok(())
    }

    fn symbolic_leaf(&mut self, name: &str) -> NodeId {
        self.runtime_inputs.push(name.into());
        input_leaf(
            &mut self.program,
            super::encoder_decoder::ACTIVATION,
            vec![Extent::Symbolic(0)],
            name,
        )
    }

    fn cross_leaves(&mut self) -> Vec<(NodeId, NodeId)> {
        let dims = [
            self.shape.source_positions,
            self.shape.decoder_heads,
            self.shape.decoder_head_dim,
        ];
        let leaves: Vec<_> = (0..self.shape.decoder_layers)
            .map(|layer| {
                (
                    self.leaf(&format!("cross_k_{layer}"), &dims),
                    self.leaf(&format!("cross_v_{layer}"), &dims),
                )
            })
            .collect();
        self.carried_inputs
            .extend(leaves.iter().flat_map(|(first, second)| [*first, *second]));
        leaves
    }

    fn cache_leaves(&mut self) -> Vec<CachedLayerRoots> {
        let heads = self.shape.decoder_heads;
        let half = self.shape.decoder_head_dim / 2;
        let full = self.shape.decoder_head_dim;
        let plane = |this: &mut Self, name: String, width: u32| {
            this.named.insert(name.clone(), Vec::new());
            this.runtime_inputs.push(name.clone());
            let extents = vec![
                Extent::Symbolic(1),
                Extent::Static(heads),
                Extent::Static(width),
            ];
            input_leaf(
                &mut this.program,
                super::encoder_decoder::ACTIVATION,
                extents,
                &name,
            )
        };
        (0..self.shape.decoder_layers)
            .map(|layer| {
                let [first, second, value] = CACHE_PLANES.map(|part| cache_plane_name(part, layer));
                (
                    plane(self, first, half),
                    plane(self, second, half),
                    plane(self, value, full),
                )
            })
            .collect()
    }

    fn cached_len_leaf(&mut self) -> NodeId {
        self.runtime_inputs.push("cached_len".into());
        input_leaf(
            &mut self.program,
            super::encoder_decoder::ACTIVATION,
            Vec::new(),
            "cached_len",
        )
    }

    fn cache_row_widths(&self) -> [usize; 3] {
        let heads = self.shape.decoder_heads as usize;
        let half = self.shape.decoder_head_dim as usize / 2;
        let full = self.shape.decoder_head_dim as usize;
        [heads * half, heads * half, heads * full]
    }

    fn fit_cache(&mut self, extent: usize) {
        let widths = self.cache_row_widths();
        for (index, name) in cache_names(self.shape.decoder_layers).enumerate() {
            if let Some(plane) = self.named.get_mut(&name) {
                plane.resize(extent * widths[index % widths.len()], Default::default());
            }
        }
    }

    fn trim_cache(&mut self, kept: usize) {
        let widths = self.cache_row_widths();
        for (index, name) in cache_names(self.shape.decoder_layers).enumerate() {
            if let Some(plane) = self.named.get_mut(&name) {
                plane.truncate(kept * widths[index % widths.len()]);
            }
        }
    }

    fn stage_step(&mut self, ids: &[u32], cached: usize) {
        let tokens = ids.iter().map(|id| *id as _).collect();
        let positions = (cached..cached + ids.len()).map(|at| at as _).collect();
        let live = (cached..=cached).map(|at| at as _).collect();
        self.named.insert("tokens".into(), tokens);
        self.named.insert("positions".into(), positions);
        self.named.insert("cached_len".into(), live);
    }

    fn matrix(&mut self, name: &str, input: u32, output: u32) -> Linear {
        let weight = self.weight(&format!("{name}.weight"), &[output, input]);
        let bias = self.weight(&format!("{name}.bias"), &[output]);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    /// A head-major input projection over the stored `[heads*head_dim, width]`
    /// matrix, read in the stored order.
    fn heads(&mut self, name: &str, bias: bool) -> Linear {
        let (heads, head_dim) = (self.shape.encoder_heads, self.shape.encoder_head_dim);
        let weight = self.weight(
            &format!("{name}.weight"),
            &[heads, head_dim, self.shape.width],
        );
        let bias = bias.then(|| self.weight(&format!("{name}.bias"), &[heads, head_dim]));
        Linear { weight, bias }
    }

    /// The cross-attention K and V projections `cross_kv` reads `[width, heads,
    /// head_dim]`, a contraction-first order the binder reads as `InOut`.
    fn memory_heads(&mut self, name: &str, bias: bool) -> Linear {
        let (heads, head_dim) = (self.shape.encoder_heads, self.shape.encoder_head_dim);
        let weight = self.weight(
            &format!("{name}.weight"),
            &[self.shape.width, heads, head_dim],
        );
        let bias = bias.then(|| self.weight(&format!("{name}.bias"), &[heads, head_dim]));
        Linear { weight, bias }
    }

    fn heads_out(&mut self, name: &str) -> Linear {
        let (heads, head_dim) = (self.shape.encoder_heads, self.shape.encoder_head_dim);
        let weight = self.weight(
            &format!("{name}.weight"),
            &[self.shape.width, heads, head_dim],
        );
        let bias = self.weight(&format!("{name}.bias"), &[self.shape.width]);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn norm(&mut self, name: &str) -> Linear {
        let weight = self.weight(&format!("{name}.weight"), &[self.shape.width]);
        let bias = self.weight(&format!("{name}.bias"), &[self.shape.width]);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn encoder_layer(&mut self, layer: usize) -> EncoderLayer {
        let prefix = format!("model.encoder.layers.{layer}");
        EncoderLayer {
            attn_ln: self.norm(&format!("{prefix}.self_attn_layer_norm")),
            attn: super::encoder_decoder::Attention {
                q: self.heads(&format!("{prefix}.self_attn.q_proj"), true),
                k: self.heads(&format!("{prefix}.self_attn.k_proj"), false),
                v: self.heads(&format!("{prefix}.self_attn.v_proj"), true),
                out: self.heads_out(&format!("{prefix}.self_attn.out_proj")),
            },
            mlp_ln: self.norm(&format!("{prefix}.final_layer_norm")),
            fc1: self.matrix(
                &format!("{prefix}.fc1"),
                self.shape.width,
                self.shape.encoder_ffn,
            ),
            fc2: self.matrix(
                &format!("{prefix}.fc2"),
                self.shape.encoder_ffn,
                self.shape.width,
            ),
        }
    }

    fn decoder_layer(&mut self, layer: usize) -> DecoderLayer {
        let prefix = format!("model.decoder.layers.{layer}");
        DecoderLayer {
            self_ln: self.norm(&format!("{prefix}.self_attn_layer_norm")),
            self_attn: super::encoder_decoder::Attention {
                q: self.heads(&format!("{prefix}.self_attn.q_proj"), true),
                k: self.heads(&format!("{prefix}.self_attn.k_proj"), false),
                v: self.heads(&format!("{prefix}.self_attn.v_proj"), true),
                out: self.heads_out(&format!("{prefix}.self_attn.out_proj")),
            },
            cross_ln: self.norm(&format!("{prefix}.encoder_attn_layer_norm")),
            cross_q: self.heads(&format!("{prefix}.encoder_attn.q_proj"), true),
            cross_out: self.heads_out(&format!("{prefix}.encoder_attn.out_proj")),
            mlp_ln: self.norm(&format!("{prefix}.final_layer_norm")),
            fc1: self.matrix(
                &format!("{prefix}.fc1"),
                self.shape.width,
                self.shape.decoder_ffn,
            ),
            fc2: self.matrix(
                &format!("{prefix}.fc2"),
                self.shape.decoder_ffn,
                self.shape.width,
            ),
        }
    }

    fn encoder_stem(&mut self) -> EncoderStem {
        let width = self.shape.width;
        let mels = self.shape.mels;
        let conv1 = self.weight("model.encoder.conv1.weight", &[width, mels, 3]);
        let conv1_bias = self.weight("model.encoder.conv1.bias", &[width]);
        let conv2 = self.weight("model.encoder.conv2.weight", &[width, width, 3]);
        let conv2_bias = self.weight("model.encoder.conv2.bias", &[width]);
        let position = self.weight(
            "model.encoder.embed_positions.weight",
            &[self.shape.source_positions, width],
        );
        EncoderStem {
            conv1: Linear {
                weight: conv1,
                bias: Some(conv1_bias),
            },
            conv2: Linear {
                weight: conv2,
                bias: Some(conv2_bias),
            },
            position,
            layer_norm: self.norm("model.encoder.layer_norm"),
        }
    }

    /// The tied head reads the embedding leaf itself: one node, one bound buffer.
    fn decoder_stem(&mut self) -> DecoderStem {
        let tokens = self.weight(
            "model.decoder.embed_tokens.weight",
            &[self.shape.vocab, self.shape.width],
        );
        let position = self.weight(
            "model.decoder.embed_positions.weight",
            &[self.shape.target_positions, self.shape.width],
        );
        DecoderStem {
            tokens,
            position,
            layer_norm: self.norm("model.decoder.layer_norm"),
            head: Linear {
                weight: tokens,
                bias: None,
            },
        }
    }

    /// The cached one-step decoder program; returns the logits followed by the
    /// appended cache planes, the outputs every step reads.
    fn decoder_graph(&mut self) -> Result<Vec<NodeId>> {
        let token_leaf = self.symbolic_leaf("tokens");
        let positions = self.symbolic_leaf("positions");
        let stem = self.decoder_stem();
        let layers: Vec<_> = (0..self.shape.decoder_layers)
            .map(|layer| self.decoder_layer(layer))
            .collect();
        let cross_nodes = self.cross_leaves();
        let cache_leaves = self.cache_leaves();
        self.cached_len_leaf();
        let (logits, appended) = super::encoder_decoder::decoder(
            &mut self.program,
            token_leaf,
            positions,
            &stem,
            &layers,
            &cross_nodes,
            Some(&cache_leaves),
            self.shape.decoder_heads,
            self.shape.decoder_head_dim,
            self.shape.eps,
        )?;
        let mut outputs = vec![logits];
        outputs.extend(
            appended
                .iter()
                .flat_map(|(first, second, value)| [*first, *second, *value]),
        );
        Ok(outputs)
    }

    /// Runs the program and returns `outputs`; `stage` (mel, encode, decode)
    /// names the program in the span and in any error.
    #[cfg_attr(
        feature = "instrument",
        proxima_telemetry::instrument(
            level = "debug",
            fields(stage = stage, rows = symbols.first().copied().unwrap_or_default())
        )
    )]
    fn run(
        &self,
        stage: &'static str,
        outputs: &[NodeId],
        symbols: &[u64],
        runtime: &mut BackendRuntime,
    ) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "instrument")]
        let run_started = Instant::now();
        let named: Vec<(&str, QuantizedBlock<'_>)> = self
            .named
            .iter()
            .map(|(name, data)| (name.as_str(), QuantizedBlock::from(data.as_slice())))
            .chain(self.weights.blocks()?)
            .collect();
        let mut resident: BTreeSet<&str> = self.weights.names();
        resident.extend(self.call_resident.iter().map(String::as_str));
        #[cfg(feature = "instrument")]
        let prep = run_started.elapsed();
        #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
        if let Some(cell) = &self.op_timings {
            let (evaluated, timings) = runtime
                .evaluate_segment_op_timed(
                    &self.program,
                    symbols,
                    &named,
                    outputs,
                    &resident,
                    &BTreeMap::new(),
                )
                .map_err(|error| anyhow::anyhow!("{stage} segment does not evaluate: {error:?}"))?;
            let _ = cell.set(timings);
            return collect_outputs(&evaluated, outputs);
        }
        #[cfg(feature = "instrument")]
        let evaluate_started = Instant::now();
        let evaluated = runtime
            .evaluate_segment(&self.program, symbols, &named, outputs, &resident, None)
            .map_err(|error| anyhow::anyhow!("{stage} segment does not evaluate: {error:?}"))?;
        #[cfg(feature = "instrument")]
        let evaluate = evaluate_started.elapsed();
        #[cfg(feature = "instrument")]
        let collect_started = Instant::now();
        let collected = collect_outputs(&evaluated, outputs);
        #[cfg(feature = "instrument")]
        proxima_telemetry::debug!(
            stage = stage,
            prep_us = prep.as_micros() as u64,
            evaluate_us = evaluate.as_micros() as u64,
            collect_us = collect_started.elapsed().as_micros() as u64,
            run_us = run_started.elapsed().as_micros() as u64,
            "encoder-decoder stage run breakdown"
        );
        collected
    }
}

#[cfg_attr(feature = "instrument", proxima_telemetry::instrument(level = "debug"))]
fn collect_outputs(evaluated: &Evaluated, outputs: &[NodeId]) -> Result<Vec<Vec<f32>>> {
    outputs
        .iter()
        .map(|node| {
            evaluated
                .get(*node)
                .map(|(values, _)| values.to_vec())
                .with_context(|| format!("evaluated program has no output for node {node:?}"))
        })
        .collect()
}

/// Makes every closed `#[instrument]` span emit a `debug!` carrying its name and
/// `duration_ns`, so span durations reach the same console and `RUST_LOG` filter
/// as the rest of the telemetry. Uses the recorder's own duration observer
/// ([`proxima_telemetry::recorder::Recorder::set_duration_observer`]); no timer
/// of its own.
#[cfg(feature = "instrument-metrics")]
pub fn log_span_durations(recorder: &proxima_telemetry::recorder::Recorder) {
    recorder.set_duration_observer(|name, duration_ns| {
        proxima_telemetry::debug!(name = name, duration_ns = duration_ns, "span closed");
    });
}

/// One transcription and the values the caller can check.
pub struct Transcript {
    pub tokens: Vec<u32>,
    pub text: String,
    pub ended_on_eot: bool,
    pub mel: std::time::Duration,
    pub encode: std::time::Duration,
    pub decode: std::time::Duration,
    pub prompt_tokens: usize,
    /// transcription start to the end of the decode step that picked the first generated token
    pub first_token: Duration,
    /// transcription start to the end of the last decode step
    pub last_token: Duration,
    /// one entry per decode step, each from the step's start to its token pick
    pub step_times: Vec<Duration>,
    /// bytes the device had allocated when the transcription ended; `None` on the host
    pub gpu_allocated_bytes: Option<u64>,
    /// largest allocation sampled after each stage and each decode step; `None` on the host
    pub gpu_allocated_peak_bytes: Option<u64>,
}

fn sample_peak(peak: &mut Option<u64>, runtime: &BackendRuntime) {
    *peak = (*peak).max(runtime.allocated_bytes());
}

/// A loaded checkpoint ready to transcribe.
pub struct Model {
    /// the mapped checkpoint file; every bound weight borrows from it in place
    mapping: CheckpointMapping,
    manifest: Manifest,
    data_start: u64,
    shape: Shape,
    generation: Value,
    config: Value,
    dir: PathBuf,
    settings: ServingSettings,
    /// owned weight buffers, materialized once and offered identically on every call
    weights: ModelWeights,
    /// the source checkpoint's header bytes, length and modification time: what a store key is made of
    source_identity: (Vec<u8>, u64, u128),
    /// where the first call publishes its weights, and under which key; `None` when the store is off or this load hit it
    store_miss: Option<(PathBuf, u128)>,
    /// set by the first publish attempt that succeeds, so later calls skip it
    store_published: AtomicBool,
    /// set by [`Model::with_encode_op_timings`]: the encode program then runs one command buffer per op
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    profile_encode: bool,
    /// the per-op GPU time of the encode program, one entry per profiled transcription (one pass each)
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    encode_timings: Mutex<Vec<Vec<omega::metal::OpGpuTiming>>>,
}

impl Drop for Model {
    fn drop(&mut self) {
        unregister_mapping(self.mapping.bytes());
    }
}

impl Model {
    pub fn load(dir: &Path) -> Result<Self> {
        let config: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).context("config.json")?,
        )?;
        let generation: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("generation_config.json"))
                .context("generation_config.json")?,
        )?;
        let shape = Shape::from_config(&config)?;
        let checkpoint = read_checkpoint(&dir.join(CHECKPOINT_FILE))?;
        Ok(Self::from_parts(checkpoint, shape, generation, config, dir))
    }

    fn from_parts(
        (mapping, manifest, data_start): Checkpoint,
        shape: Shape,
        generation: Value,
        config: Value,
        dir: &Path,
    ) -> Self {
        let source_identity = (
            header_prefix(mapping.bytes()).unwrap_or_default().to_vec(),
            mapping.bytes().len() as u64,
            modified_nanos(&dir.join(CHECKPOINT_FILE)),
        );
        Self {
            mapping,
            manifest,
            data_start,
            shape,
            generation,
            config,
            dir: dir.to_path_buf(),
            settings: ServingSettings::default(),
            weights: ModelWeights::new(),
            source_identity,
            store_miss: None,
            store_published: AtomicBool::new(false),
            #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
            profile_encode: false,
            #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
            encode_timings: Mutex::new(Vec::new()),
        }
    }

    /// Replaces the serving settings every stage lowers into its runtime. With
    /// `weight_store` set this is also where the store is consulted, because
    /// the key needs the `weight_precision` rules and `load` has none: a hit
    /// swaps the mapped checkpoint for the stored one and emits
    /// `weight_store_hit`; a miss changes nothing and arms the publish after
    /// the first transcription.
    #[must_use]
    pub fn with_settings(mut self, settings: ServingSettings) -> Self {
        self.settings = settings;
        self.resolve_store();
        self
    }

    /// Runs the encode program of the next transcription one command buffer per op, so
    /// [`Self::encode_op_timings`] can report each op's GPU time. A diagnostic: the serving loop never sets it.
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    #[must_use]
    pub fn with_encode_op_timings(mut self) -> Self {
        self.profile_encode = true;
        self
    }

    /// The per-op GPU time of the encode program, one `Vec` per profiled transcription in run order
    /// (empty before the first). A single pass is not a usable measurement: reduce the passes with
    /// [`reduce_op_passes`]. Bound through the same weights and `weight_precision` rules as the serving encoder.
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    pub fn encode_op_timing_passes(&self) -> Vec<Vec<omega::metal::OpGpuTiming>> {
        self.encode_timings.lock().clone()
    }

    fn resolve_store(&mut self) {
        self.store_miss = None;
        let Some(store_dir) = self.settings.weight_store.as_deref().map(PathBuf::from) else {
            return;
        };
        let Ok(rules) = self.settings.weight_precision_rules() else {
            return;
        };
        let (header, len, modified) = &self.source_identity;
        let key = store_key(header, *len, *modified, &rules, device_alignment());
        match lookup(&store_dir, key).and_then(|path| read_checkpoint(&path).ok()) {
            Some((mapping, manifest, data_start)) => {
                emit_weight_store_hit(key, mapping.bytes().len());
                self.mapping = mapping;
                self.manifest = manifest;
                self.data_start = data_start;
                self.weights = ModelWeights::new();
            }
            None => {
                self.store_published.store(false, Ordering::Release);
                self.store_miss = Some((store_dir, key));
            }
        }
    }

    /// Publishes the weights the first call materialized, once. A failed write
    /// leaves the flag clear, so the next call retries, and is reported as a
    /// `weight_store_publish` event with `ok = false`.
    fn publish_store(&self) {
        let Some((store_dir, key)) = &self.store_miss else {
            return;
        };
        if self.store_published.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut stored = Vec::new();
        let written = self
            .weights
            .persist_aligned(
                &self.manifest,
                self.mapping.bytes(),
                self.data_start,
                device_alignment(),
                &mut stored,
            )
            .and_then(|_| publish(store_dir, *key, &stored));
        if written.is_err() {
            self.store_published.store(false, Ordering::Release);
        }
        emit_weight_store_publish(*key, stored.len(), written.is_ok());
    }

    /// The cached-range rows a step binds, decided exactly as the llm decides
    /// them: the runtime probes the program, and pads to the capacity only where
    /// the fused cached attention's runtime bound excludes the pad rows.
    fn cache_extent(
        &self,
        runtime: &mut BackendRuntime,
        decoder: &Bound<'_>,
        outputs: &[NodeId],
        (program_hash, cached, new_count): (u64, usize, usize),
    ) -> usize {
        let capacity = self.shape.target_positions as usize;
        let probe_extent = capacity.max(1) as u64;
        runtime.settle_kv_padding(
            &decoder.program,
            program_hash,
            &[new_count as u64, probe_extent],
            || outputs.to_vec(),
        );
        cached_kv_extent(
            runtime.pads_cached_kv(program_hash, new_count),
            cached,
            new_count,
            self.settings.kv_bucket_tokens,
            capacity,
        )
        .max(cached)
    }

    #[cfg_attr(feature = "instrument", proxima_telemetry::instrument(level = "debug"))]
    fn runtime(&self) -> Result<BackendRuntime> {
        let config = self.settings.as_serving_config(&[]);
        apply_serving_config(&config, 0)
            .map_err(|error| anyhow::anyhow!("serving settings are not admitted: {error:?}"))?;
        // the registration is thread-local, so it is made on the thread that builds this call's runtime
        register_mapping(self.mapping.bytes());
        Ok(BackendRuntime::new(&config))
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Transcribes on the device the serving settings select: `gpu_layers` of `GPU_LAYERS_ALL` runs the stages on the device.
    /// On the first call of a weight-store miss it then publishes the weights the call materialized: the bind leases are
    /// back in the model by then, so `persist_aligned` sees every recoded leaf.
    pub fn transcribe(&self, wav: &[f32], language: &str) -> Result<Transcript> {
        let outcome = self.transcribe_with(wav, language);
        unregister_mapping(self.mapping.bytes());
        let transcript = outcome?;
        self.publish_store();
        Ok(transcript)
    }

    /// One transcription with no weight-store publish; [`Self::transcribe`] is the call that publishes.
    pub fn transcribe_with(&self, wav: &[f32], language: &str) -> Result<Transcript> {
        let start = Instant::now();
        let policy = DecodePolicy::from_checkpoint(
            &self.generation,
            &self.config,
            &DecodeConfig {
                language: language.into(),
                ..DecodeConfig::default()
            },
        )?;

        #[cfg(feature = "instrument")]
        let runtime_started = Instant::now();
        let mut runtime = self.runtime()?;
        #[cfg(feature = "instrument")]
        proxima_telemetry::debug!(
            runtime_us = runtime_started.elapsed().as_micros() as u64,
            "transcribe fresh backend runtime built"
        );
        let mut gpu_peak = None;
        let padded = pad(wav)?;
        let front = front_end(&self.dir)?;
        if front.mels != self.shape.mels {
            bail!(
                "preprocessor_config.json says {} mels, config.json says {}",
                front.mels,
                self.shape.mels
            );
        }
        let frames = ((padded.len() as u32 - front.n_fft) / front.hop) + 1;
        let mut mel_bound = Bound::new(self.shape.clone());
        let signal = mel_bound.leaf("signal", &[padded.len() as u32]);
        let filters = front.filters;
        let bank = mel_bound.leaf("bank", &[front.mels, front.n_fft / 2 + 1]);
        let config = StftLogMelConfig {
            n_fft: front.n_fft,
            hop: front.hop,
            ..StftLogMelConfig::default()
        };
        let mel = stft_log_mel(
            &mut mel_bound.program,
            &config,
            signal,
            padded.len() as u32,
            bank,
        )?;
        let tables = stft_tables(&config, crate::families::encoder_decoder::ACTIVATION)?;
        let decode = |bytes: &[u8]| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect()
        };
        mel_bound.named.insert("signal".into(), padded);
        mel_bound.named.insert("bank".into(), filters);
        mel_bound
            .named
            .insert(STFT_WINDOW_INPUT.into(), decode(&tables.window));
        mel_bound
            .named
            .insert(STFT_COS_INPUT.into(), decode(&tables.cos));
        mel_bound
            .named
            .insert(STFT_SIN_INPUT.into(), decode(&tables.sin));
        let mel_start = std::time::Instant::now();
        let mel_values = mel_bound.run("mel", &[mel], &[], &mut runtime)?.remove(0);
        let mel = mel_start.elapsed();
        sample_peak(&mut gpu_peak, &runtime);
        // The encoder consumes the first 2*source_positions frames.
        let needed = (self.shape.source_positions * 2) as usize;
        let mut mel_input = vec![0.0; (self.shape.mels as usize) * needed];
        for row in 0..self.shape.mels as usize {
            for frame in 0..needed {
                mel_input[row * needed + frame] = mel_values[row * frames as usize + frame];
            }
        }

        #[cfg(feature = "instrument")]
        let graph_started = Instant::now();
        let mut bound = Bound::new(self.shape.clone());
        let mel_leaf = bound.leaf("mel", &[1, self.shape.mels, needed as u32]);
        bound.named.insert("mel".into(), mel_input);
        let stem = bound.encoder_stem();
        let layers: Vec<_> = (0..self.shape.encoder_layers)
            .map(|layer| bound.encoder_layer(layer))
            .collect();
        let memory = encoder(
            &mut bound.program,
            mel_leaf,
            &stem,
            &layers,
            self.shape.encoder_heads,
            self.shape.encoder_head_dim,
            self.shape.eps,
            self.shape.mels,
            needed as u32,
        )?;
        let mut cross = Vec::new();
        for layer in 0..self.shape.decoder_layers {
            let prefix = format!("model.decoder.layers.{layer}.encoder_attn");
            let k = bound.memory_heads(&format!("{prefix}.k_proj"), false);
            let v = bound.memory_heads(&format!("{prefix}.v_proj"), true);
            let (k, v) = cross_kv(
                &mut bound.program,
                memory,
                k.weight,
                k.bias,
                v.weight,
                v.bias,
            )?;
            cross.push((k, v));
        }
        let mut outputs = vec![memory];
        for (k, v) in &cross {
            outputs.push(*k);
            outputs.push(*v);
        }
        #[cfg(feature = "instrument")]
        let bind_started = Instant::now();
        bound.bind(self)?;
        #[cfg(feature = "instrument")]
        proxima_telemetry::debug!(
            graph_us = bind_started.duration_since(graph_started).as_micros() as u64,
            bind_us = bind_started.elapsed().as_micros() as u64,
            "encoder graph built and weights bound, outside encode_us"
        );
        #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
        {
            bound.op_timings = self.profile_encode.then(std::sync::OnceLock::new);
        }
        let encode_start = std::time::Instant::now();
        let mut values = bound.run("encode", &outputs, &[], &mut runtime)?;
        let encode = encode_start.elapsed();
        #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
        if let Some(timings) = bound
            .op_timings
            .take()
            .and_then(std::sync::OnceLock::into_inner)
        {
            self.encode_timings.lock().push(timings);
        }
        sample_peak(&mut gpu_peak, &runtime);

        let mut decoder = Bound::new(self.shape.clone());
        let step_outputs = decoder.decoder_graph()?;
        let carried = decoder.carried_inputs.clone();
        for (input, value) in carried.into_iter().zip(values.drain(1..)) {
            let name = decoder.carry(input)?;
            decoder.named.insert(name, value);
        }
        decoder.bind(self)?;
        let program_hash = program_hash(&decoder.program);

        let mut tokens: Vec<u32> = policy.prompt().to_vec();
        let mut cached = 0;
        let mut ended = false;
        let mut step_times = Vec::new();
        let mut first_token = Duration::ZERO;
        let mut last_token = Duration::ZERO;
        let decode_start = Instant::now();
        let mut decode_steps = || -> Result<()> {
            for step in 0..policy.step_cap() as usize {
                let step_start = Instant::now();
                let fresh = &tokens[cached..];
                decoder.stage_step(fresh, cached);
                let extent = self.cache_extent(
                    &mut runtime,
                    &decoder,
                    &step_outputs,
                    (program_hash, cached, fresh.len()),
                );
                decoder.fit_cache(extent);
                let symbols = [fresh.len() as u64, extent as u64];
                let mut rows = decoder.run("decode", &step_outputs, &symbols, &mut runtime)?;
                let flat = rows.remove(0);
                decoder.trim_cache(cached);
                for (name, added) in cache_names(self.shape.decoder_layers).zip(rows) {
                    if let Some(plane) = decoder.named.get_mut(&name) {
                        plane.extend(added);
                    }
                }
                cached = tokens.len();
                let vocab = self.shape.vocab as usize;
                let mut row = flat[flat.len() - vocab..].to_vec();
                let chosen = policy.pick(&mut row, step)?;
                tokens.push(chosen);
                sample_peak(&mut gpu_peak, &runtime);
                step_times.push(step_start.elapsed());
                last_token = start.elapsed();
                if step == 0 {
                    first_token = last_token;
                }
                if chosen == policy.end_token() {
                    ended = true;
                    break;
                }
            }
            Ok(())
        };
        let outcome = decode_steps();
        let held: Vec<&str> = decoder.call_resident.iter().map(String::as_str).collect();
        runtime.release_call_residents(&held);
        outcome?;
        let decode = decode_start.elapsed();
        let generated = &tokens[policy.prompt().len()..];
        let text = tokenizer_text(&self.dir, generated)?;
        Ok(Transcript {
            tokens: generated.to_vec(),
            text,
            ended_on_eot: ended,
            mel,
            encode,
            decode,
            prompt_tokens: policy.prompt().len(),
            first_token,
            last_token,
            step_times,
            gpu_allocated_bytes: runtime.allocated_bytes(),
            gpu_allocated_peak_bytes: gpu_peak,
        })
    }
}

struct FrontEnd {
    filters: Vec<f32>,
    n_fft: u32,
    hop: u32,
    mels: u32,
}

fn front_end(dir: &Path) -> Result<FrontEnd> {
    let text = std::fs::read_to_string(dir.join("preprocessor_config.json"))?;
    let value: Value = serde_json::from_str(&text)?;
    let integer = |key: &str| -> Result<u32> {
        value
            .get(key)
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .with_context(|| format!("preprocessor_config.json lacks {key}"))
    };
    let mut filters = Vec::new();
    if let Some(rows) = value["mel_filters"].as_array() {
        for row in rows {
            for cell in row.as_array().context("filter row")? {
                filters.push(cell.as_f64().context("filter cell")? as f32);
            }
        }
    } else {
        // Some published checkpoints omit the filterbank from
        // preprocessor_config.json; the checkpoint's own ggml file carries it.
        let bytes = std::fs::read(dir.join("mel_filters.f32"))
            .context("preprocessor_config.json has no mel_filters and mel_filters.f32 is absent")?;
        filters = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
    }
    Ok(FrontEnd {
        filters,
        n_fft: integer("n_fft")?,
        hop: integer("hop_length")?,
        mels: integer("feature_size")?,
    })
}

fn tokenizer_text(dir: &Path, ids: &[u32]) -> Result<String> {
    let bytes = std::fs::read(dir.join("tokenizer.json")).context("tokenizer.json")?;
    let config: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let id = |key: &str| config.get(key).and_then(Value::as_u64).map(|v| v as u32);
    let vocab = proxima_tokenizer::tokenizer_json::load_tokenizer_json(
        &bytes,
        id("bos_token_id"),
        id("eos_token_id"),
        None,
    )?;
    proxima_tokenizer::pipe::decode(ids, &vocab).map_err(Into::into)
}

fn pad(wav: &[f32]) -> Result<Vec<f32>> {
    if wav.len() < 200 || wav.len() > 480_040 {
        bail!("one 30 s window is 200..=480040 samples; got {}", wav.len());
    }
    let mut out = vec![0.0; 480_240];
    for index in 0..200 {
        out[199 - index] = wav[index + 1];
    }
    out[200..200 + wav.len()].copy_from_slice(wav);
    Ok(out)
}

/// The identity and GPU time of one timed op, the part of `omega::metal::OpGpuTiming` the pass
/// reduction reads.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpSample {
    pub node: NodeId,
    pub kind: &'static str,
    pub extents: Vec<u64>,
    pub gpu_ns: u64,
}

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
impl From<&omega::metal::OpGpuTiming> for OpSample {
    fn from(timing: &omega::metal::OpGpuTiming) -> Self {
        Self {
            node: timing.node,
            kind: timing.kind,
            extents: timing.extents.clone(),
            gpu_ns: timing.gpu_ns,
        }
    }
}

/// One op's GPU time over every timed pass: `min_ns` and `median_ns` (the mean of the two middle
/// values for an even pass count, integer division), with `passes` naming how many went in.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpPassStats {
    pub node: NodeId,
    pub kind: &'static str,
    pub extents: Vec<u64>,
    pub passes: usize,
    pub min_ns: u64,
    pub median_ns: u64,
}

/// Reduces `passes` (each the encode program's ops in program order) to per-op min and median
/// `gpu_ns`. Every pass is included, the first too. Errors when there are no passes or the passes
/// disagree on the op count or on an op's node or kind.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub fn reduce_op_passes(passes: &[Vec<OpSample>]) -> Result<Vec<OpPassStats>> {
    let first = passes.first().context("no timed passes to reduce")?;
    for (index, pass) in passes.iter().enumerate() {
        if pass.len() != first.len() {
            bail!(
                "pass {index} timed {} ops, pass 0 timed {}",
                pass.len(),
                first.len()
            );
        }
    }
    first
        .iter()
        .enumerate()
        .map(|(position, anchor)| {
            let mut times = Vec::with_capacity(passes.len());
            for (index, pass) in passes.iter().enumerate() {
                let sample = &pass[position];
                if sample.node != anchor.node || sample.kind != anchor.kind {
                    bail!(
                        "pass {index} op {position} is node {} kind {}, pass 0 has node {} kind {}",
                        sample.node.0,
                        sample.kind,
                        anchor.node.0,
                        anchor.kind
                    );
                }
                times.push(sample.gpu_ns);
            }
            times.sort_unstable();
            let middle = times.len() / 2;
            let median_ns = if times.len() % 2 == 0 {
                (times[middle - 1] + times[middle]) / 2
            } else {
                times[middle]
            };
            Ok(OpPassStats {
                node: anchor.node,
                kind: anchor.kind,
                extents: anchor.extents.clone(),
                passes: times.len(),
                min_ns: times[0],
                median_ns,
            })
        })
        .collect()
}

/// Per-kind `(ops, sum of per-op mins, sum of per-op medians)` over the reduced ops.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub fn kind_totals(stats: &[OpPassStats]) -> BTreeMap<&'static str, (usize, u64, u64)> {
    let mut totals: BTreeMap<&'static str, (usize, u64, u64)> = BTreeMap::new();
    for stat in stats {
        let entry = totals.entry(stat.kind).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += stat.min_ns;
        entry.2 += stat.median_ns;
    }
    totals
}

#[cfg(test)]
mod tests {
    use conflaguration::Settings;
    use proxima_numeric::{NumericPolicy, decode_blocks, encode_blocks};
    use proxima_safetensors::{SafetensorsModel, TensorPayload, is_filler_name, write_complete};

    use super::*;

    const BIT_EXACT_JSON: &str = r#"{"contraction":false,"reassociation":false,"nan_assumptions":false,"signed_zero":false,"approx_functions":false,"epilogue_sources":false}"#;

    const EMBEDDING: &str = "model.decoder.embed_tokens.weight";

    fn decoder_tensor_shapes() -> Vec<(String, Vec<u64>)> {
        let layer = "model.decoder.layers.0";
        let matrix = vec![4, 4];
        let vector = vec![4];
        let attention = |name: &str| {
            [
                (format!("{layer}.{name}.q_proj.weight"), matrix.clone()),
                (format!("{layer}.{name}.q_proj.bias"), vector.clone()),
                (format!("{layer}.{name}.out_proj.weight"), matrix.clone()),
                (format!("{layer}.{name}.out_proj.bias"), vector.clone()),
            ]
        };
        let norm = |name: &str| {
            [
                (format!("{name}.weight"), vector.clone()),
                (format!("{name}.bias"), vector.clone()),
            ]
        };
        let mut shapes = vec![
            (EMBEDDING.to_string(), vec![6, 4]),
            (
                "model.decoder.embed_positions.weight".to_string(),
                vec![3, 4],
            ),
            (format!("{layer}.self_attn.k_proj.weight"), matrix.clone()),
            (format!("{layer}.self_attn.v_proj.weight"), matrix.clone()),
            (format!("{layer}.self_attn.v_proj.bias"), vector.clone()),
            (format!("{layer}.fc1.weight"), vec![8, 4]),
            (format!("{layer}.fc1.bias"), vec![8]),
            (format!("{layer}.fc2.weight"), vec![4, 8]),
            (format!("{layer}.fc2.bias"), vec![4]),
        ];
        shapes.extend(attention("self_attn"));
        shapes.extend(attention("encoder_attn"));
        shapes.extend(norm(&format!("{layer}.self_attn_layer_norm")));
        shapes.extend(norm(&format!("{layer}.encoder_attn_layer_norm")));
        shapes.extend(norm(&format!("{layer}.final_layer_norm")));
        shapes.extend(norm("model.decoder.layer_norm"));
        shapes
    }

    /// a real safetensors file holding every tensor the decoder program reads,
    /// each element distinct from every other tensor's
    fn decoder_checkpoint() -> Checkpoint {
        let dtype = super::super::encoder_decoder::ACTIVATION;
        let format = dtype
            .format()
            .expect("the activation dtype has a numeric format");
        let payloads: Vec<(String, Vec<u64>, Vec<u8>)> = decoder_tensor_shapes()
            .into_iter()
            .enumerate()
            .map(|(index, (name, shape))| {
                let count = shape.iter().product::<u64>() as usize;
                let values: Vec<_> = (0..count)
                    .map(|element| (index * 1000 + element) as _)
                    .collect();
                let mut data = vec![0; count * format.size_bytes()];
                encode_blocks(&format, values.as_slice(), &mut data).expect("encodes the tensor");
                (name, shape, data)
            })
            .collect();
        let file = write_complete(&SafetensorsModel {
            tensors: payloads
                .iter()
                .map(|(name, shape, data)| TensorPayload {
                    name: name.clone(),
                    dtype,
                    shape: shape.clone(),
                    data,
                })
                .collect(),
            metadata: BTreeMap::new(),
        })
        .expect("writes a real safetensors buffer");
        let (manifest, data_start) = parse_checkpoint(&file).expect("the written header parses");
        (mapped(&file), manifest, data_start)
    }

    /// maps `bytes` through a real file, as `Model::load` maps a checkpoint; the
    /// mapping outlives the unlinked file on unix
    fn mapped(bytes: &[u8]) -> CheckpointMapping {
        let directory = tempfile::tempdir().expect("creates a checkpoint directory");
        let path = directory.path().join("model.safetensors");
        std::fs::write(&path, bytes).expect("writes the checkpoint file");
        CheckpointMapping::open_file(&path).expect("maps the checkpoint file")
    }

    fn decoder_model() -> Model {
        let config = serde_json::json!({
            "encoder_layers": 1, "decoder_layers": 1,
            "encoder_attention_heads": 2, "decoder_attention_heads": 2,
            "d_model": 4, "num_mel_bins": 2, "encoder_ffn_dim": 8, "decoder_ffn_dim": 8,
            "max_source_positions": 2, "max_target_positions": 3, "vocab_size": 6,
        });
        let shape = Shape::from_config(&config).expect("the decoder config is complete");
        Model::from_parts(
            decoder_checkpoint(),
            shape,
            Value::Null,
            config,
            Path::new(""),
        )
    }

    fn bound_decoder(model: &Model) -> Bound<'_> {
        let mut decoder = Bound::new(model.shape.clone());
        decoder.decoder_graph().expect("the decoder graph builds");
        decoder
            .bind(model)
            .expect("the shared binder binds the decoder");
        decoder
    }

    #[test]
    fn decoder_binds_with_no_flipped_names_and_no_decoded_copies() {
        let model = decoder_model();

        let decoder = bound_decoder(&model);

        let flipped = decoder
            .weights
            .names()
            .into_iter()
            .filter(|name| name.contains("#flipped"))
            .count();
        assert_eq!(flipped, 0, "a stored-order leaf binds under its own name");
        assert_eq!(
            decoder.weights.owned().len(),
            0,
            "no weight is decoded into a copy"
        );
    }

    #[test]
    fn tied_head_binds_the_embedding_as_one_buffer() {
        let model = decoder_model();
        let entry = model
            .manifest
            .tensor(EMBEDDING)
            .expect("the embedding is stored");
        let (start, end) = entry.data_offsets;
        let file = model.mapping.bytes();
        let raw = &file[(model.data_start + start) as usize..(model.data_start + end) as usize];
        let format = entry
            .dtype
            .format()
            .expect("the embedding has a numeric format");
        let mut values = vec![0.0; entry.shape.iter().product::<u64>() as usize];
        decode_blocks(&format, raw, &mut values).expect("decodes the embedding");
        let embedding = format!("{:?}", QuantizedBlock::from(values.as_slice()));

        let decoder = bound_decoder(&model);

        let blocks = decoder.weights.blocks().expect("the bound blocks resolve");
        let carrying = blocks
            .iter()
            .filter(|(_, block)| format!("{block:?}") == embedding)
            .count();
        assert_eq!(carrying, 1, "the head reads the embedding leaf, not a copy");
        assert!(!decoder.weights.names().contains("lm_head"));
    }

    fn empty_model() -> Model {
        let tensors = (mapped(b"-"), Manifest::default(), 0);
        let config = serde_json::json!({
            "encoder_layers": 1, "decoder_layers": 1,
            "encoder_attention_heads": 1, "decoder_attention_heads": 1,
            "d_model": 4, "num_mel_bins": 1, "encoder_ffn_dim": 4, "decoder_ffn_dim": 4,
            "max_source_positions": 1, "max_target_positions": 1, "vocab_size": 1,
        });
        let shape = Shape::from_config(&config).expect("the tiny config is complete");
        Model::from_parts(tensors, shape, Value::Null, config, Path::new(""))
    }

    fn model_with(dir: &Path, settings: ServingSettings) -> Model {
        let config = serde_json::json!({
            "encoder_layers": 1, "decoder_layers": 1,
            "encoder_attention_heads": 2, "decoder_attention_heads": 2,
            "d_model": 4, "num_mel_bins": 2, "encoder_ffn_dim": 8, "decoder_ffn_dim": 8,
            "max_source_positions": 2, "max_target_positions": 3, "vocab_size": 6,
        });
        let shape = Shape::from_config(&config).expect("the decoder config is complete");
        let checkpoint = read_checkpoint(&dir.join(CHECKPOINT_FILE)).expect("reads the source");
        Model::from_parts(checkpoint, shape, Value::Null, config, dir).with_settings(settings)
    }

    fn stored_settings(store: &Path) -> ServingSettings {
        ServingSettings::builder()
            .weight_store(store.display().to_string())
            .build()
    }

    fn source_directory() -> tempfile::TempDir {
        let scratch = tempfile::tempdir().expect("creates a scratch directory");
        let (source, ..) = decoder_checkpoint();
        std::fs::write(scratch.path().join(CHECKPOINT_FILE), source.bytes())
            .expect("writes the source checkpoint");
        scratch
    }

    #[test]
    fn a_store_miss_publishes_once_and_the_next_load_maps_the_stored_file() {
        let scratch = source_directory();
        let store = scratch.path().join("store");

        let first = model_with(scratch.path(), stored_settings(&store));
        assert!(first.store_miss.is_some(), "an empty store is a miss");
        first.publish_store();
        let second = model_with(scratch.path(), stored_settings(&store));

        assert!(second.store_miss.is_none(), "the published entry is a hit");
        let stored = std::fs::read_dir(&store)
            .expect("lists the store")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        assert_eq!(stored.len(), 1, "one entry, no torn temp file: {stored:?}");
        assert_eq!(
            second.mapping.bytes(),
            std::fs::read(&stored[0])
                .expect("reads the entry")
                .as_slice()
        );
        let originals = second
            .manifest
            .tensors
            .iter()
            .filter(|entry| !is_filler_name(&entry.name))
            .count();
        assert_eq!(
            originals,
            first.manifest.tensors.len(),
            "every source tensor is stored"
        );
    }

    #[test]
    fn a_changed_rule_list_is_a_miss_against_an_existing_store() {
        let scratch = source_directory();
        let store = scratch.path().join("store");
        model_with(scratch.path(), stored_settings(&store)).publish_store();
        let mut reruled: ServingSettings = conflaguration::from_toml_str(
            "[[weight_precision]]\npattern_kind = \"suffix\"\npattern = \"fc1.weight\"\ntarget = \"q8_0\"",
        )
        .expect("the rule toml parses");
        reruled.weight_store = Some(store.display().to_string());

        let model = model_with(scratch.path(), reruled);

        assert!(model.store_miss.is_some());
    }

    #[test]
    fn model_defaults_to_the_serving_settings_policy() {
        let model = empty_model();

        assert_eq!(
            model.settings.numeric_policy,
            ServingSettings::default().numeric_policy
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn runtime_uses_the_gpu_exactly_when_settings_offload_every_layer() {
        let cpu = empty_model().with_settings(ServingSettings {
            gpu_layers: 0,
            ..ServingSettings::default()
        });
        let offloaded = empty_model().with_settings(ServingSettings {
            gpu_layers: crate::serving::GPU_LAYERS_ALL,
            ..ServingSettings::default()
        });

        assert!(!cpu.runtime().expect("cpu settings are admitted").uses_gpu());
        assert!(
            offloaded
                .runtime()
                .expect("offload settings are admitted")
                .uses_gpu()
        );
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn every_layer_offload_is_refused_without_a_gpu_build() {
        let model = empty_model().with_settings(ServingSettings {
            gpu_layers: crate::serving::GPU_LAYERS_ALL,
            ..ServingSettings::default()
        });

        let refusal = model
            .runtime()
            .err()
            .expect("a build without a gpu backend refuses -ngl all");

        assert!(format!("{refusal}").contains("no GPU backend"), "{refusal}");
    }

    #[test]
    fn serving_env_spells_the_bit_exact_policy() {
        let loaded = temp_env::with_vars(
            [("PROXIMA_SERVING_NUMERIC_POLICY", Some(BIT_EXACT_JSON))],
            || ServingSettings::from_env().expect("the bit-exact env parses"),
        );

        assert_eq!(loaded.numeric_policy, NumericPolicy::bit_exact());
        assert_ne!(
            loaded.numeric_policy,
            ServingSettings::default().numeric_policy
        );
    }

    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    fn sample(node: u32, kind: &'static str, gpu_ns: u64) -> OpSample {
        OpSample {
            node: NodeId(node),
            kind,
            extents: vec![1500, 64],
            gpu_ns,
        }
    }

    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    #[test]
    fn pass_reduction_takes_min_and_median_per_op_with_a_slow_first_pass() {
        // the attention op is slow on the first pass, as the measured node 170 was
        let passes = vec![
            vec![sample(170, "cached_attention", 4_800_000), sample(171, "ffn", 900)],
            vec![sample(170, "cached_attention", 1_650_000), sample(171, "ffn", 700)],
            vec![sample(170, "cached_attention", 1_630_000), sample(171, "ffn", 800)],
        ];

        let stats = reduce_op_passes(&passes).expect("three aligned passes reduce");

        assert_eq!(stats[0].min_ns, 1_630_000);
        assert_eq!(stats[0].median_ns, 1_650_000);
        assert_eq!(stats[0].passes, 3);
        assert_eq!((stats[1].min_ns, stats[1].median_ns), (700, 800));
    }

    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    #[test]
    fn pass_reduction_median_of_an_even_count_is_the_mean_of_the_middle_two() {
        let passes = vec![
            vec![sample(1, "ffn", 10)],
            vec![sample(1, "ffn", 40)],
            vec![sample(1, "ffn", 20)],
            vec![sample(1, "ffn", 100)],
        ];

        let stats = reduce_op_passes(&passes).expect("four aligned passes reduce");

        assert_eq!((stats[0].min_ns, stats[0].median_ns), (10, 30));
    }

    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    #[test]
    fn pass_reduction_rejects_misaligned_or_empty_passes() {
        assert!(reduce_op_passes(&[]).is_err());
        let short = vec![vec![sample(1, "ffn", 5)], vec![]];
        assert!(reduce_op_passes(&short).is_err());
        let moved = vec![vec![sample(1, "ffn", 5)], vec![sample(2, "ffn", 5)]];
        assert!(reduce_op_passes(&moved).is_err());
    }

    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    #[test]
    fn kind_totals_sum_per_op_mins_and_medians_by_kind() {
        let passes = vec![
            vec![sample(1, "ffn", 10), sample(2, "ffn", 50), sample(3, "norm", 7)],
            vec![sample(1, "ffn", 30), sample(2, "ffn", 40), sample(3, "norm", 5)],
        ];
        let stats = reduce_op_passes(&passes).expect("two aligned passes reduce");

        let totals = kind_totals(&stats);

        assert_eq!(totals["ffn"], (2, 50, 20 + 45));
        assert_eq!(totals["norm"], (1, 5, 6));
    }
}
