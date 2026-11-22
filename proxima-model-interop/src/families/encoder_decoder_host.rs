//! Host-side assembly of the generic encoder-decoder builders: read a
//! safetensors checkpoint and its config, build the mel/encoder/cross-KV/
//! decoder graphs, and run a greedy transcript. This is the std half; the
//! graph builders themselves stay alloc-only and model-neutral.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use proxima_numeric::DType;
use proxima_safetensors::SafetensorsParser;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::{
    STFT_COS_INPUT, STFT_SIN_INPUT, STFT_WINDOW_INPUT, StftLogMelConfig, stft_log_mel, stft_tables,
};
use serde_json::Value;

use crate::decode_policy::{DecodeConfig, DecodePolicy};
use crate::families::encoder_decoder::{
    DecoderLayer, DecoderStem, EncoderLayer, EncoderStem, Linear, encoder, encoder_conv,
};

/// The dimensions a checkpoint states, read from its `config.json`.
#[derive(Debug, Clone)]
pub struct Shape {
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub heads: u32,
    pub head_dim: u32,
    pub width: u32,
    pub mels: u32,
    pub ffn: u32,
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
        let heads = int("encoder_attention_heads")?;
        let width = int("d_model")?;
        let mels = int("num_mel_bins")?;
        let ffn = int("encoder_ffn_dim")?;
        let source_positions = int("max_source_positions")?;
        let target_positions = int("max_target_positions")?;
        let vocab = int("vocab_size")?;
        if width % heads != 0 {
            bail!("d_model {width} is not a multiple of {heads} heads");
        }
        Ok(Self {
            encoder_layers,
            decoder_layers,
            heads,
            head_dim: width / heads,
            width,
            mels,
            ffn,
            source_positions,
            target_positions,
            vocab,
            eps: 1e-5,
        })
    }
}

/// The checkpoint's tensors, in the layout the builders read.
pub struct Tensors {
    values: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Tensors {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
        let header_len =
            u64::from_le_bytes(bytes[..8].try_into().expect("eight byte header length")) as usize;
        let mut parser = SafetensorsParser::new();
        parser.feed(&bytes[..8 + header_len]);
        parser.poll().expect("safetensors header parses");
        let SafetensorsParser::TensorData { manifest, .. } = parser else {
            return Err(anyhow::anyhow!("safetensors header did not complete"));
        };
        let data = &bytes[8 + header_len..];
        let mut values = BTreeMap::new();
        for entry in &manifest.tensors {
            let (start, end) = entry.data_offsets;
            let raw = &data[start as usize..end as usize];
            let mut tensor: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect();
            let shape: Vec<usize> = entry.shape.iter().map(|&dim| dim as usize).collect();
            if shape.len() >= 2
                && !entry.name.contains("embed_positions")
                && !entry.name.contains("layer_norm")
            {
                tensor
                    .iter_mut()
                    .for_each(|value| *value = f16_round(*value));
            }
            values.insert(entry.name.clone(), (tensor, shape));
        }
        Ok(Self { values })
    }

    fn data(&self, name: &str) -> &[f32] {
        &self
            .values
            .get(name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
            .0
    }

    fn shape(&self, name: &str) -> &[usize] {
        &self
            .values
            .get(name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
            .1
    }

    fn transposed(&self, name: &str) -> Vec<f32> {
        let shape = self.shape(name);
        assert_eq!(shape.len(), 2, "{name} is a matrix");
        let (rows, columns) = (shape[0], shape[1]);
        let source = self.data(name);
        let mut out = vec![0.0_f32; source.len()];
        for row in 0..rows {
            for column in 0..columns {
                out[column * rows + row] = source[row * columns + column];
            }
        }
        out
    }
}

fn f16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let sign = bits & 0x8000_0000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }
    if exponent >= 0x1f {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    let mantissa = (bits >> 13) & 0x3ff;
    let round = (bits >> 12) & 1;
    let mut half = (exponent as u32) << 10 | mantissa;
    if round == 1 {
        half += 1;
    }
    let out_exp = ((half >> 10) & 0x1f) as i32 - 15 + 127;
    let out_mant = (half & 0x3ff) << 13;
    f32::from_bits(sign | ((out_exp as u32) << 23) | out_mant)
}

struct Bound {
    program: Vec<Op>,
    named: BTreeMap<String, Vec<f32>>,
    shape: Shape,
}

impl Bound {
    fn new(shape: Shape) -> Self {
        Self {
            program: Vec::new(),
            named: BTreeMap::new(),
            shape,
        }
    }

    fn leaf(&mut self, name: &str, shape: &[u32]) -> NodeId {
        super::encoder_decoder::leaf(&mut self.program, shape, name)
    }

    fn matrix(&mut self, name: &str, input: u32, output: u32, tensors: &Tensors) -> Linear {
        let weight_name = format!("{name}.weight");
        let weight = self.leaf(&weight_name, &[input, output]);
        self.named
            .insert(weight_name, tensors.transposed(&format!("{name}.weight")));
        let bias_name = format!("{name}.bias");
        let bias = self.leaf(&bias_name, &[output]);
        self.named.insert(
            bias_name.clone(),
            tensors.data(&format!("{name}.bias")).to_vec(),
        );
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn heads(&mut self, name: &str, tensors: &Tensors, bias: bool) -> Linear {
        let weight_name = format!("{name}.weight");
        let weight = self.leaf(
            &weight_name,
            &[self.shape.width, self.shape.heads, self.shape.head_dim],
        );
        self.named
            .insert(weight_name, tensors.transposed(&format!("{name}.weight")));
        let bias = bias.then(|| {
            let bias_name = format!("{name}.bias");
            let node = self.leaf(&bias_name, &[self.shape.heads, self.shape.head_dim]);
            self.named.insert(
                bias_name.clone(),
                tensors.data(&format!("{name}.bias")).to_vec(),
            );
            node
        });
        Linear { weight, bias }
    }

    fn heads_out(&mut self, name: &str, tensors: &Tensors) -> Linear {
        let weight_name = format!("{name}.weight");
        let weight = self.leaf(
            &weight_name,
            &[self.shape.heads, self.shape.head_dim, self.shape.width],
        );
        self.named
            .insert(weight_name, tensors.transposed(&format!("{name}.weight")));
        let bias_name = format!("{name}.bias");
        let bias = self.leaf(&bias_name, &[self.shape.width]);
        self.named.insert(
            bias_name.clone(),
            tensors.data(&format!("{name}.bias")).to_vec(),
        );
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn norm(&mut self, name: &str, tensors: &Tensors) -> Linear {
        let weight_name = format!("{name}.weight");
        let weight = self.leaf(&weight_name, &[self.shape.width]);
        self.named.insert(
            weight_name,
            tensors.data(&format!("{name}.weight")).to_vec(),
        );
        let bias_name = format!("{name}.bias");
        let bias = self.leaf(&bias_name, &[self.shape.width]);
        self.named.insert(
            bias_name.clone(),
            tensors.data(&format!("{name}.bias")).to_vec(),
        );
        Linear {
            weight,
            bias: Some(bias),
        }
    }

    fn encoder_layer(&mut self, layer: usize, tensors: &Tensors) -> EncoderLayer {
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
                self.shape.ffn,
                tensors,
            ),
            fc2: self.matrix(
                &format!("{prefix}.fc2"),
                self.shape.ffn,
                self.shape.width,
                tensors,
            ),
        }
    }

    fn decoder_layer(&mut self, layer: usize, tensors: &Tensors) -> DecoderLayer {
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
                self.shape.ffn,
                tensors,
            ),
            fc2: self.matrix(
                &format!("{prefix}.fc2"),
                self.shape.ffn,
                self.shape.width,
                tensors,
            ),
        }
    }

    fn encoder_stem(&mut self, tensors: &Tensors) -> EncoderStem {
        let width = self.shape.width;
        let mels = self.shape.mels;
        let conv1_name = "model.encoder.conv1.weight";
        let conv1 = self.leaf(conv1_name, &[width, mels, 3]);
        self.named
            .insert(conv1_name.into(), tensors.data(conv1_name).to_vec());
        let conv1_bias_name = "model.encoder.conv1.bias";
        let conv1_bias = self.leaf(conv1_bias_name, &[width]);
        self.named.insert(
            conv1_bias_name.into(),
            tensors.data(conv1_bias_name).to_vec(),
        );
        let conv2_name = "model.encoder.conv2.weight";
        let conv2 = self.leaf(conv2_name, &[width, width, 3]);
        self.named
            .insert(conv2_name.into(), tensors.data(conv2_name).to_vec());
        let conv2_bias_name = "model.encoder.conv2.bias";
        let conv2_bias = self.leaf(conv2_bias_name, &[width]);
        self.named.insert(
            conv2_bias_name.into(),
            tensors.data(conv2_bias_name).to_vec(),
        );
        let position_name = "model.encoder.embed_positions.weight";
        let position = self.leaf(position_name, &[self.shape.source_positions, width]);
        self.named
            .insert(position_name.into(), tensors.data(position_name).to_vec());
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

    fn decoder_stem(&mut self, tensors: &Tensors) -> DecoderStem {
        let tokens_name = "model.decoder.embed_tokens.weight";
        let tokens = self.leaf(tokens_name, &[self.shape.vocab, self.shape.width]);
        self.named
            .insert(tokens_name.into(), tensors.data(tokens_name).to_vec());
        let position_name = "model.decoder.embed_positions.weight";
        let position = self.leaf(
            position_name,
            &[self.shape.target_positions, self.shape.width],
        );
        self.named
            .insert(position_name.into(), tensors.data(position_name).to_vec());
        let head_name = "lm_head";
        let head = self.leaf(head_name, &[self.shape.width, self.shape.vocab]);
        self.named.insert(
            head_name.into(),
            tensors.transposed("model.decoder.embed_tokens.weight"),
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

    fn run(&self, outputs: &[NodeId]) -> Vec<Vec<f32>> {
        let named: Vec<(&str, &[f32])> = self
            .named
            .iter()
            .map(|(name, data)| (name.as_str(), data.as_slice()))
            .collect();
        let evaluated =
            evaluate_named(&self.program, &[], &named, outputs).expect("program evaluates");
        outputs
            .iter()
            .map(|node| evaluated.get(*node).expect("output").0.to_vec())
            .collect()
    }
}

/// One transcription and the values the caller can check.
pub struct Transcript {
    pub tokens: Vec<u32>,
    pub text: String,
    pub ended_on_eot: bool,
}

/// A loaded checkpoint ready to transcribe.
pub struct Model {
    tensors: Tensors,
    shape: Shape,
    generation: Value,
    config: Value,
    dir: PathBuf,
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
        Ok(Self {
            tensors,
            shape,
            generation,
            config,
            dir: dir.to_path_buf(),
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn transcribe(&self, wav: &[f32], language: &str) -> Result<Transcript> {
        let policy = DecodePolicy::from_checkpoint(
            &self.generation,
            &self.config,
            &DecodeConfig {
                language: language.into(),
                ..DecodeConfig::default()
            },
        )?;

        let padded = pad(wav);
        let frames = ((padded.len() as u32 - 400) / 160) + 1;
        let mut mel_bound = Bound::new(self.shape.clone());
        let signal = mel_bound.leaf("signal", &[padded.len() as u32]);
        let filters = mel_filters(&self.dir)?;
        let bank = mel_bound.leaf("bank", &[self.shape.mels, 201]);
        let mel = stft_log_mel(
            &mut mel_bound.program,
            &StftLogMelConfig::default(),
            signal,
            padded.len() as u32,
            bank,
        )?;
        let tables = stft_tables(&StftLogMelConfig::default(), DType::FLOAT32)?;
        let decode = |bytes: &[u8]| -> Vec<f32> {
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
        let mel_values = mel_bound.run(&[mel]).remove(0);
        // The encoder consumes the first 2*source_positions frames.
        let needed = (self.shape.source_positions * 2) as usize;
        let mut mel_input = vec![0.0_f32; (self.shape.mels as usize) * needed];
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
        let _conv = encoder_conv(
            &mut bound.program,
            mel_leaf,
            &stem,
            self.shape.mels,
            self.shape.width,
            needed as u32,
        )?;
        let memory = encoder(
            &mut bound.program,
            mel_leaf,
            &stem,
            &layers,
            self.shape.heads,
            self.shape.head_dim,
            self.shape.eps,
            self.shape.mels,
            needed as u32,
        )?;
        let mut cross = Vec::new();
        for layer in 0..self.shape.decoder_layers {
            let prefix = format!("model.decoder.layers.{layer}.encoder_attn");
            let k = bound.heads(&format!("{prefix}.k_proj"), &self.tensors, false);
            let v = bound.heads(&format!("{prefix}.v_proj"), &self.tensors, true);
            let (k, v) = super::encoder_decoder::cross_kv(
                &mut bound.program,
                memory,
                k,
                v,
                self.shape.head_dim,
            )?;
            cross.push((k, v));
        }
        let mut outputs = vec![memory];
        for (k, v) in &cross {
            outputs.push(*k);
            outputs.push(*v);
        }
        let values = bound.run(&outputs);
        let cross_values: Vec<(Vec<f32>, Vec<f32>)> = (0..self.shape.decoder_layers)
            .map(|layer| (values[1 + layer * 2].clone(), values[2 + layer * 2].clone()))
            .collect();

        let mut tokens: Vec<u32> = policy.prompt().to_vec();
        let mut ended = false;
        for step in 0..policy.step_cap() as usize {
            let mut decoder = Bound::new(self.shape.clone());
            let n = tokens.len() as u32;
            let token_leaf = decoder.leaf("tokens", &[n]);
            let positions = decoder.leaf("positions", &[n]);
            let _ = &[0_u8; 0];
            let stem = decoder.decoder_stem(&self.tensors);
            let layers: Vec<_> = (0..self.shape.decoder_layers)
                .map(|layer| decoder.decoder_layer(layer, &self.tensors))
                .collect();
            let mut cross_nodes = Vec::new();
            for (layer, (k, v)) in cross_values.iter().enumerate() {
                let k_name = format!("cross_k_{layer}");
                let v_name = format!("cross_v_{layer}");
                let k_node = decoder.leaf(
                    &k_name,
                    &[
                        self.shape.source_positions,
                        self.shape.heads,
                        self.shape.head_dim,
                    ],
                );
                let v_node = decoder.leaf(
                    &v_name,
                    &[
                        self.shape.source_positions,
                        self.shape.heads,
                        self.shape.head_dim,
                    ],
                );
                decoder.named.insert(k_name, k.clone());
                decoder.named.insert(v_name, v.clone());
                cross_nodes.push((k_node, v_node));
            }
            let logits = super::encoder_decoder::decoder(
                &mut decoder.program,
                token_leaf,
                positions,
                &stem,
                &layers,
                &cross_nodes,
                self.shape.heads,
                self.shape.head_dim,
                self.shape.eps,
            )?;
            decoder.named.insert(
                "tokens".into(),
                tokens.iter().map(|id| *id as f32).collect(),
            );
            decoder
                .named
                .insert("positions".into(), (0..n).map(|i| i as f32).collect());
            let flat = decoder.run(&[logits]).remove(0);
            let vocab = self.shape.vocab as usize;
            let last = &flat[(n as usize - 1) * vocab..n as usize * vocab];
            let mut row = last.to_vec();
            let chosen = policy.pick(&mut row, step)?;
            tokens.push(chosen);
            if chosen == policy.end_token() {
                ended = true;
                break;
            }
        }
        let generated = &tokens[policy.prompt().len()..];
        let text = tokenizer_text(&self.dir, generated)?;
        Ok(Transcript {
            tokens: generated.to_vec(),
            text,
            ended_on_eot: ended,
        })
    }
}

fn mel_filters(dir: &Path) -> Result<Vec<f32>> {
    let text = std::fs::read_to_string(dir.join("preprocessor_config.json"))?;
    let value: Value = serde_json::from_str(&text)?;
    let rows = value["mel_filters"]
        .as_array()
        .context("preprocessor_config.json has mel_filters")?;
    let mut filters = Vec::new();
    for row in rows {
        for cell in row.as_array().context("filter row")? {
            filters.push(cell.as_f64().context("filter cell")? as f32);
        }
    }
    Ok(filters)
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

fn pad(wav: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; 480_240];
    let reflect = wav.len().min(200);
    for index in 0..reflect {
        out[reflect - 1 - index] = wav[index + 1];
    }
    out[200..200 + wav.len()].copy_from_slice(wav);
    out
}
