//! Host-side assembly of the generic encoder-decoder builders: read a
//! safetensors checkpoint and its config, build the mel/encoder/cross-KV/
//! decoder graphs, and run a greedy transcript. This is the std half; the
//! graph builders themselves stay alloc-only and model-neutral.

use std::collections::BTreeMap;
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use proxima_numeric::decode_blocks;
use proxima_safetensors::{Manifest, SafetensorsParser, TensorEntry};
use proxima_tensor::cpu::{Evaluated, QuantizedBlock, evaluate_quantized_named};
use proxima_tensor::op::{Extent, NodeId, Op};
use proxima_tensor::spec::{
    CachedLayerRoots, STFT_COS_INPUT, STFT_SIN_INPUT, STFT_WINDOW_INPUT, StftLogMelConfig,
    cross_kv, input_leaf, stft_log_mel, stft_tables,
};
use serde_json::Value;

use crate::bind::transpose_out_in_to_in_out;
use crate::decode_policy::{DecodeConfig, DecodePolicy};
use crate::families::encoder_decoder::{
    DecoderLayer, DecoderStem, EncoderLayer, EncoderStem, Linear, encoder,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::generate::{BackendRuntime, kv_extent};
use crate::hf_bind::safetensors_tensor_as_packed_block;
use crate::serving::{ServingConfig, apply_serving_config};
use crate::serving_settings::ServingSettings;

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

/// suffix of the cache key holding a stored `[out, in]` matrix as `[in, out]`
const FLIPPED: &str = "#flipped";

const CACHE_PLANES: [&str; 3] = ["first", "second", "value"];

fn cache_plane_name(plane: &str, layer: usize) -> String {
    format!("cache_{plane}_{layer}")
}

fn cache_names(layers: usize) -> impl Iterator<Item = String> {
    (0..layers).flat_map(|layer| CACHE_PLANES.map(|part| cache_plane_name(part, layer)))
}

/// The checkpoint's tensors, in the layout the builders read.
pub struct Tensors {
    bytes: Vec<u8>,
    manifest: Manifest,
    data_start: u64,
    /// stored half-width leaves a builder reads decoded, filled on first use and kept across decode steps
    decoded: BTreeMap<String, OnceLock<Vec<f32>>>,
}

impl Tensors {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
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
        let decoded = manifest
            .tensors
            .iter()
            .flat_map(|entry| [entry.name.clone(), format!("{}{FLIPPED}", entry.name)])
            .map(|key| (key, OnceLock::new()))
            .collect();
        Ok(Self {
            bytes,
            manifest,
            data_start: (8 + header_len) as u64,
            decoded,
        })
    }

    fn block(&self, name: &str) -> QuantizedBlock<'_> {
        safetensors_tensor_as_packed_block(&self.manifest, &self.bytes, self.data_start, name)
            .unwrap_or_else(|error| panic!("{name} binds in its stored format: {error}"))
    }

    fn entry(&self, name: &str) -> &TensorEntry {
        self.manifest
            .tensors
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
    }

    fn decoded(&self, name: &str) -> &[f32] {
        self.decoded
            .get(name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
            .get_or_init(|| {
                let entry = self.entry(name);
                let (start, end) = entry.data_offsets;
                let raw = &self.bytes
                    [(self.data_start + start) as usize..(self.data_start + end) as usize];
                let format = entry
                    .dtype
                    .format()
                    .unwrap_or_else(|| panic!("{name} has no numeric format ({:?})", entry.dtype));
                let mut tensor = vec![0.0; entry.shape.iter().product::<u64>() as usize];
                decode_blocks(&format, raw, &mut tensor)
                    .unwrap_or_else(|error| panic!("decoding {name} as {format:?}: {error}"));
                tensor
            })
    }

    fn dims(&self, name: &str) -> Vec<usize> {
        self.entry(name)
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .collect()
    }
}

/// the serving runtime every Metal stage plans and executes through
#[cfg(all(feature = "metal", target_os = "macos"))]
type Runtime = BackendRuntime;
#[cfg(not(all(feature = "metal", target_os = "macos")))]
struct Runtime;

#[cfg(all(feature = "metal", target_os = "macos"))]
fn open_runtime(config: &ServingConfig<'_>) -> Runtime {
    BackendRuntime::new(config)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn open_runtime(_config: &ServingConfig<'_>) -> Runtime {
    Runtime
}

struct Bound<'a> {
    program: Vec<Op>,
    named: BTreeMap<String, Vec<f32>>,
    blocks: Vec<(String, QuantizedBlock<'a>)>,
    shape: Shape,
}

impl<'a> Bound<'a> {
    fn new(shape: Shape) -> Self {
        Self {
            program: Vec::new(),
            named: BTreeMap::new(),
            blocks: Vec::new(),
            shape,
        }
    }

    fn leaf(&mut self, name: &str, shape: &[u32]) -> NodeId {
        super::encoder_decoder::leaf(&mut self.program, shape, name)
    }

    fn symbolic_leaf(&mut self, name: &str) -> NodeId {
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
        (0..self.shape.decoder_layers)
            .map(|layer| {
                (
                    self.leaf(&format!("cross_k_{layer}"), &dims),
                    self.leaf(&format!("cross_v_{layer}"), &dims),
                )
            })
            .collect()
    }

    fn cache_leaves(&mut self) -> Vec<CachedLayerRoots> {
        let heads = self.shape.decoder_heads;
        let half = self.shape.decoder_head_dim / 2;
        let full = self.shape.decoder_head_dim;
        let plane = |this: &mut Self, name: String, width: u32| {
            this.named.insert(name.clone(), Vec::new());
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

    fn weight(&mut self, name: &str, shape: &[u32], stored: &str, tensors: &'a Tensors) -> NodeId {
        let node = self.leaf(name, shape);
        let block = tensors.block(stored);
        if block.packed_bytes().is_some() {
            self.blocks.push((name.into(), block));
        } else {
            let flipped = tensors
                .decoded
                .get(&format!("{stored}{FLIPPED}"))
                .unwrap_or_else(|| panic!("no tensor {stored}"))
                .get_or_init(|| {
                    let dims = tensors.dims(stored);
                    transpose_out_in_to_in_out(tensors.decoded(stored), stored, dims[0], dims[1])
                        .unwrap_or_else(|error| panic!("{stored} transposes: {error}"))
                });
            self.blocks
                .push((name.into(), QuantizedBlock::from(flipped.as_slice())));
        }
        node
    }

    fn plain(&mut self, name: &str, shape: &[u32], tensors: &'a Tensors) -> NodeId {
        let node = self.leaf(name, shape);
        let block = tensors.block(name);
        let block = if block.packed_bytes().is_some() {
            QuantizedBlock::from(tensors.decoded(name))
        } else {
            block
        };
        self.blocks.push((name.into(), block));
        node
    }

    fn matrix(&mut self, name: &str, input: u32, output: u32, tensors: &'a Tensors) -> Linear {
        let weight_name = format!("{name}.weight");
        let weight = self.weight(&weight_name, &[input, output], &weight_name, tensors);
        let bias = self.plain(&format!("{name}.bias"), &[output], tensors);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn heads(&mut self, name: &str, tensors: &'a Tensors, bias: bool) -> Linear {
        let weight_name = format!("{name}.weight");
        let weight = self.weight(
            &weight_name,
            &[
                self.shape.width,
                self.shape.encoder_heads,
                self.shape.encoder_head_dim,
            ],
            &weight_name,
            tensors,
        );
        let bias = bias.then(|| {
            self.plain(
                &format!("{name}.bias"),
                &[self.shape.encoder_heads, self.shape.encoder_head_dim],
                tensors,
            )
        });
        Linear { weight, bias }
    }

    fn heads_out(&mut self, name: &str, tensors: &'a Tensors) -> Linear {
        let weight_name = format!("{name}.weight");
        let weight = self.weight(
            &weight_name,
            &[
                self.shape.encoder_heads,
                self.shape.encoder_head_dim,
                self.shape.width,
            ],
            &weight_name,
            tensors,
        );
        let bias = self.plain(&format!("{name}.bias"), &[self.shape.width], tensors);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn norm(&mut self, name: &str, tensors: &'a Tensors) -> Linear {
        let weight = self.plain(&format!("{name}.weight"), &[self.shape.width], tensors);
        let bias = self.plain(&format!("{name}.bias"), &[self.shape.width], tensors);
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn encoder_layer(&mut self, layer: usize, tensors: &'a Tensors) -> EncoderLayer {
        let prefix = format!("model.encoder.layers.{layer}");
        EncoderLayer {
            attn_ln: self.norm(&format!("{prefix}.self_attn_layer_norm"), tensors),
            attn: super::encoder_decoder::Attention {
                q: self.heads(&format!("{prefix}.self_attn.q_proj"), tensors, true),
                k: self.heads(&format!("{prefix}.self_attn.k_proj"), tensors, false),
                v: self.heads(&format!("{prefix}.self_attn.v_proj"), tensors, true),
                out: self.heads_out(&format!("{prefix}.self_attn.out_proj"), tensors),
            },
            mlp_ln: self.norm(&format!("{prefix}.final_layer_norm"), tensors),
            fc1: self.matrix(
                &format!("{prefix}.fc1"),
                self.shape.width,
                self.shape.encoder_ffn,
                tensors,
            ),
            fc2: self.matrix(
                &format!("{prefix}.fc2"),
                self.shape.encoder_ffn,
                self.shape.width,
                tensors,
            ),
        }
    }

    fn decoder_layer(&mut self, layer: usize, tensors: &'a Tensors) -> DecoderLayer {
        let prefix = format!("model.decoder.layers.{layer}");
        DecoderLayer {
            self_ln: self.norm(&format!("{prefix}.self_attn_layer_norm"), tensors),
            self_attn: super::encoder_decoder::Attention {
                q: self.heads(&format!("{prefix}.self_attn.q_proj"), tensors, true),
                k: self.heads(&format!("{prefix}.self_attn.k_proj"), tensors, false),
                v: self.heads(&format!("{prefix}.self_attn.v_proj"), tensors, true),
                out: self.heads_out(&format!("{prefix}.self_attn.out_proj"), tensors),
            },
            cross_ln: self.norm(&format!("{prefix}.encoder_attn_layer_norm"), tensors),
            cross_q: self.heads(&format!("{prefix}.encoder_attn.q_proj"), tensors, true),
            cross_out: self.heads_out(&format!("{prefix}.encoder_attn.out_proj"), tensors),
            mlp_ln: self.norm(&format!("{prefix}.final_layer_norm"), tensors),
            fc1: self.matrix(
                &format!("{prefix}.fc1"),
                self.shape.width,
                self.shape.decoder_ffn,
                tensors,
            ),
            fc2: self.matrix(
                &format!("{prefix}.fc2"),
                self.shape.decoder_ffn,
                self.shape.width,
                tensors,
            ),
        }
    }

    fn encoder_stem(&mut self, tensors: &'a Tensors) -> EncoderStem {
        let width = self.shape.width;
        let mels = self.shape.mels;
        let conv1 = self.plain("model.encoder.conv1.weight", &[width, mels, 3], tensors);
        let conv1_bias = self.plain("model.encoder.conv1.bias", &[width], tensors);
        let conv2 = self.plain("model.encoder.conv2.weight", &[width, width, 3], tensors);
        let conv2_bias = self.plain("model.encoder.conv2.bias", &[width], tensors);
        let position = self.plain(
            "model.encoder.embed_positions.weight",
            &[self.shape.source_positions, width],
            tensors,
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
            layer_norm: self.norm("model.encoder.layer_norm", tensors),
        }
    }

    fn decoder_stem(&mut self, tensors: &'a Tensors) -> DecoderStem {
        let tokens_name = "model.decoder.embed_tokens.weight";
        let tokens = self.plain(tokens_name, &[self.shape.vocab, self.shape.width], tensors);
        let position = self.plain(
            "model.decoder.embed_positions.weight",
            &[self.shape.target_positions, self.shape.width],
            tensors,
        );
        let head = self.weight(
            "lm_head",
            &[self.shape.width, self.shape.vocab],
            tokens_name,
            tensors,
        );
        DecoderStem {
            tokens,
            position,
            layer_norm: self.norm("model.decoder.layer_norm", tensors),
            head: Linear {
                weight: head,
                bias: None,
            },
        }
    }

    fn run(
        &self,
        outputs: &[NodeId],
        symbols: &[u64],
        runtime: &mut Runtime,
    ) -> Result<Vec<Vec<f32>>> {
        let named: Vec<(&str, QuantizedBlock<'_>)> = self
            .named
            .iter()
            .map(|(name, data)| (name.as_str(), QuantizedBlock::from(data.as_slice())))
            .chain(
                self.blocks
                    .iter()
                    .map(|(name, block)| (name.as_str(), *block)),
            )
            .collect();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if runtime.is_metal() {
            let resident: BTreeSet<&str> =
                self.blocks.iter().map(|(name, _)| name.as_str()).collect();
            let evaluated = runtime
                .evaluate_segment(&self.program, symbols, &named, outputs, &resident, None)
                .map_err(|error| anyhow::anyhow!("metal segment: {error:?}"))?;
            return collect_outputs(&evaluated, outputs);
        }
        let _ = runtime;
        let evaluated = evaluate_quantized_named(&self.program, symbols, &named, outputs)
            .map_err(|error| anyhow::anyhow!("program does not evaluate: {error:?}"))?;
        collect_outputs(&evaluated, outputs)
    }
}

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
    /// bytes Metal had allocated when the transcription ended; `None` off Metal
    pub gpu_allocated_bytes: Option<u64>,
    /// largest allocation sampled after each stage and each decode step; `None` off Metal
    pub gpu_allocated_peak_bytes: Option<u64>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn gpu_allocated(runtime: &Runtime) -> Option<u64> {
    runtime
        .is_metal()
        .then(omega::metal::current_allocated_size)
        .flatten()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn gpu_allocated(_runtime: &Runtime) -> Option<u64> {
    None
}

fn sample_peak(peak: &mut Option<u64>, runtime: &Runtime) {
    *peak = (*peak).max(gpu_allocated(runtime));
}

/// A loaded checkpoint ready to transcribe.
pub struct Model {
    tensors: Tensors,
    shape: Shape,
    generation: Value,
    config: Value,
    dir: PathBuf,
    settings: ServingSettings,
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
        let tensors = Tensors::load(&dir.join("model.safetensors"))?;
        Ok(Self::from_parts(tensors, shape, generation, config, dir))
    }

    fn from_parts(
        tensors: Tensors,
        shape: Shape,
        generation: Value,
        config: Value,
        dir: &Path,
    ) -> Self {
        Self {
            tensors,
            shape,
            generation,
            config,
            dir: dir.to_path_buf(),
            settings: ServingSettings::default(),
        }
    }

    /// Replaces the serving settings every Metal stage lowers into its runtime.
    #[must_use]
    pub fn with_settings(mut self, settings: ServingSettings) -> Self {
        self.settings = settings;
        self
    }

    /// The cached-range rows a step binds: the llm's bucketed extent when the
    /// fused cached attention's runtime bound can exclude the pad rows (the
    /// metal plan), otherwise exactly `cached`.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn cache_extent(&self, runtime: &Runtime, cached: usize) -> usize {
        if !runtime.is_metal() {
            return cached;
        }
        let capacity = self.shape.target_positions as usize;
        let bucket = self.settings.kv_bucket_tokens;
        kv_extent(cached, capacity, bucket).max(cached)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn cache_extent(&self, _runtime: &Runtime, cached: usize) -> usize {
        cached
    }

    fn runtime(&self) -> Result<Runtime> {
        let config = self.settings.as_serving_config(&[]);
        apply_serving_config(&config, 0)
            .map_err(|error| anyhow::anyhow!("serving settings are not admitted: {error:?}"))?;
        Ok(open_runtime(&config))
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Transcribes on the device the serving settings select: `gpu_layers` of `GPU_LAYERS_ALL` runs the stages on Metal.
    pub fn transcribe(&self, wav: &[f32], language: &str) -> Result<Transcript> {
        self.transcribe_with(wav, language)
    }

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

        let mut runtime = self.runtime()?;
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
        let mel_values = mel_bound.run(&[mel], &[], &mut runtime)?.remove(0);
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

        let mut bound = Bound::new(self.shape.clone());
        let mel_leaf = bound.leaf("mel", &[1, self.shape.mels, needed as u32]);
        bound.named.insert("mel".into(), mel_input);
        let stem = bound.encoder_stem(&self.tensors);
        let layers: Vec<_> = (0..self.shape.encoder_layers)
            .map(|layer| bound.encoder_layer(layer, &self.tensors))
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
            let k = bound.heads(&format!("{prefix}.k_proj"), &self.tensors, false);
            let v = bound.heads(&format!("{prefix}.v_proj"), &self.tensors, true);
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
        let encode_start = std::time::Instant::now();
        let mut values = bound.run(&outputs, &[], &mut runtime)?;
        let encode = encode_start.elapsed();
        sample_peak(&mut gpu_peak, &runtime);

        let mut decoder = Bound::new(self.shape.clone());
        let token_leaf = decoder.symbolic_leaf("tokens");
        let positions = decoder.symbolic_leaf("positions");
        let stem = decoder.decoder_stem(&self.tensors);
        let layers: Vec<_> = (0..self.shape.decoder_layers)
            .map(|layer| decoder.decoder_layer(layer, &self.tensors))
            .collect();
        let cross_nodes = decoder.cross_leaves();
        for layer in 0..self.shape.decoder_layers {
            let k = std::mem::take(&mut values[1 + layer * 2]);
            let v = std::mem::take(&mut values[2 + layer * 2]);
            decoder.named.insert(format!("cross_k_{layer}"), k);
            decoder.named.insert(format!("cross_v_{layer}"), v);
        }
        let cache_leaves = decoder.cache_leaves();
        decoder.cached_len_leaf();
        let (logits, appended) = super::encoder_decoder::decoder(
            &mut decoder.program,
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
        let mut step_outputs = vec![logits];
        step_outputs.extend(
            appended
                .iter()
                .flat_map(|(first, second, value)| [*first, *second, *value]),
        );

        let mut tokens: Vec<u32> = policy.prompt().to_vec();
        let mut cached = 0;
        let mut ended = false;
        let mut step_times = Vec::new();
        let mut first_token = Duration::ZERO;
        let mut last_token = Duration::ZERO;
        let decode_start = Instant::now();
        for step in 0..policy.step_cap() as usize {
            let step_start = Instant::now();
            let fresh = &tokens[cached..];
            decoder.stage_step(fresh, cached);
            let extent = self.cache_extent(&runtime, cached);
            decoder.fit_cache(extent);
            let symbols = [fresh.len() as u64, extent as u64];
            let mut rows = decoder.run(&step_outputs, &symbols, &mut runtime)?;
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
            gpu_allocated_bytes: gpu_allocated(&runtime),
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

#[cfg(test)]
mod tests {
    use conflaguration::Settings;
    use proxima_numeric::NumericPolicy;

    use super::*;

    const BIT_EXACT_JSON: &str = r#"{"contraction":false,"reassociation":false,"nan_assumptions":false,"signed_zero":false,"approx_functions":false,"epilogue_sources":false}"#;

    fn empty_model() -> Model {
        let tensors = Tensors {
            bytes: Vec::new(),
            manifest: Manifest::default(),
            data_start: 0,
            decoded: BTreeMap::new(),
        };
        let config = serde_json::json!({
            "encoder_layers": 1, "decoder_layers": 1,
            "encoder_attention_heads": 1, "decoder_attention_heads": 1,
            "d_model": 4, "num_mel_bins": 1, "encoder_ffn_dim": 4, "decoder_ffn_dim": 4,
            "max_source_positions": 1, "max_target_positions": 1, "vocab_size": 1,
        });
        let shape = Shape::from_config(&config).expect("the tiny config is complete");
        Model::from_parts(tensors, shape, Value::Null, config, Path::new(""))
    }

    #[test]
    fn model_defaults_to_the_serving_settings_policy() {
        let model = empty_model();

        assert_eq!(
            model.settings.numeric_policy,
            ServingSettings::default().numeric_policy
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn runtime_is_metal_exactly_when_settings_offload_every_layer() {
        let cpu = empty_model().with_settings(ServingSettings {
            gpu_layers: 0,
            ..ServingSettings::default()
        });
        let metal = empty_model().with_settings(ServingSettings {
            gpu_layers: crate::serving::GPU_LAYERS_ALL,
            ..ServingSettings::default()
        });

        assert!(!cpu.runtime().expect("cpu settings are admitted").is_metal());
        assert!(metal.runtime().expect("metal settings are admitted").is_metal());
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn every_layer_offload_is_refused_without_a_gpu_build() {
        let model = empty_model().with_settings(ServingSettings {
            gpu_layers: crate::serving::GPU_LAYERS_ALL,
            ..ServingSettings::default()
        });

        let refusal = model.runtime().err().expect("a build without a gpu backend refuses -ngl all");

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
}
