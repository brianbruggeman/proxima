//! Greedy-decode comparison of the CPU oracle against the CUDA backend.
//!
//! Usage: `decode_compare <model.gguf> [prompt] [tokens] [backends]` (default prompt
//! `The capital of France is`, default 16 tokens, `backends` is `both` or `cuda`,
//! default `both`; `cuda` skips the CPU oracle and prints `matched=n/a`).
//!
//! Loads the checkpoint once, runs `LoadedModel::generate_with_serving_config`
//! with `gpu_layers: 0` and with `GPU_LAYERS_ALL`, prints one line per position,
//! then `decode_compare model=<name> tokens=N matched=M first_divergence=<pos|none>`,
//! where N is the count produced (generation stops at the model's end-of-sequence id).
//! On divergence it teacher-forces the CPU prefix through both backends
//! (`LoadedModel::teacher_forced_logits_from_ids`) and prints the top-2 logit
//! margin at that position on each side, so a near-tie reads differently from a
//! real routing or kernel bug.
//!
//! Each backend's run is timed with `Instant` at every streamed token event:
//! `decode_step backend=<cuda|cpu> step=<i> ms=<f>` per decoded token and
//! `decode_timing backend=.. prefill_ms=.. decode_ms_mean=.. decode_ms_cov=.. steps=N`.
//! `prefill_ms` runs from the call start through the step-0 token event (the token
//! sampled from the prefill pass), so it includes tokenization and runtime
//! construction, which the streaming API does not expose separately. Decode steps
//! start at the second token event; step `i` is the gap since the previous event.

use std::env;
use std::fs::File;
use std::ops::ControlFlow;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{
    GPU_LAYERS_ALL, InteropError, LoadedModel, Phase, ServingConfig, TokenEvent,
};
use proxima_telemetry::emit::{EnvFilter, global};
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::recorder::Recorder;
use proxima_tokenizer::gguf::vocab_from_metadata;
use proxima_tokenizer::{TokenizerError, Vocab, encode_with_bos_eos};

const DEFAULT_PROMPT: &str = "The capital of France is";
const DEFAULT_TOKENS: usize = 16;

#[derive(Debug, thiserror::Error)]
enum ExampleError {
    #[error("usage: decode_compare <model.gguf> [prompt] [tokens] [both|cuda]")]
    Usage,
    #[error("{path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("{path}: gguf parse: {message}")]
    Gguf { path: String, message: String },
    #[error("interop: {0}")]
    Interop(#[from] InteropError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("backends {0:?} is not one of both, cuda")]
    BadBackends(String),
    #[error("telemetry: {0}")]
    Telemetry(String),
    #[error("token count {0:?} is not a number")]
    BadCount(String),
    #[error("{backend} produced no logits row at position {position}")]
    MissingRow {
        backend: &'static str,
        position: usize,
    },
}

#[derive(Clone, Copy)]
enum Backends {
    Both,
    Cuda,
}

struct Top2 {
    first: (usize, f32),
    second: (usize, f32),
}

impl Top2 {
    fn of(row: &[f32]) -> Self {
        let mut ranked: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        Self {
            first: ranked[0],
            second: ranked[1],
        }
    }

    fn margin(&self) -> f32 {
        self.first.1 - self.second.1
    }

    fn render(&self) -> String {
        format!(
            "top1={}:{:.6} top2={}:{:.6} margin={:.6}",
            self.first.0,
            self.first.1,
            self.second.0,
            self.second.1,
            self.margin()
        )
    }
}

fn config(gpu_layers: i32) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers,
        ..ServingConfig::default()
    }
}

fn prompt_ids(model_vocab: &Vocab, prompt: &str) -> Result<Vec<u32>, TokenizerError> {
    let wants_bos = model_vocab
        .add_bos_token()
        .unwrap_or_else(|| model_vocab.bos_token_id().is_some());
    let wants_eos = model_vocab.add_eos_token().unwrap_or(false);
    encode_with_bos_eos(prompt, model_vocab, wants_bos, wants_eos)
}

struct Timing {
    prefill_ms: f64,
    step_ms: Vec<f64>,
}

impl Timing {
    fn mean(&self) -> f64 {
        self.step_ms.iter().sum::<f64>() / self.step_ms.len().max(1) as f64
    }

    fn cov(&self) -> f64 {
        let mean = self.mean();
        let variance = self
            .step_ms
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / self.step_ms.len().max(1) as f64;
        if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        }
    }

    fn report(&self, backend: &str) {
        for (step, millis) in self.step_ms.iter().enumerate() {
            println!("decode_step backend={backend} step={step} ms={millis:.3}");
        }
        println!(
            "decode_timing backend={backend} prefill_ms={:.3} decode_ms_mean={:.3} decode_ms_cov={:.4} steps={}",
            self.prefill_ms,
            self.mean(),
            self.cov(),
            self.step_ms.len()
        );
    }
}

fn decode(
    model: &LoadedModel,
    prompt: &str,
    tokens: usize,
    gpu_layers: i32,
) -> Result<(Vec<u32>, String, Timing), InteropError> {
    let mut timing = Timing {
        prefill_ms: 0.0,
        step_ms: Vec::new(),
    };
    let mut previous = Instant::now();
    let mut on_token = |event: TokenEvent<'_>| {
        let now = Instant::now();
        let millis = now.duration_since(previous).as_secs_f64() * 1000.0;
        previous = now;
        match event.phase {
            Phase::Prefill { .. } => timing.prefill_ms = millis,
            Phase::Token if event.step == 0 => timing.prefill_ms += millis,
            Phase::Token => timing.step_ms.push(millis),
        }
        ControlFlow::Continue(())
    };
    let (ids, text, _stopped) =
        model.generate_streaming(prompt, tokens, config(gpu_layers), &mut on_token)?;
    Ok((ids, text, timing))
}

fn first_divergence(cpu: &[u32], cuda: &[u32]) -> Option<usize> {
    let common = cpu.len().max(cuda.len());
    (0..common).find(|&position| cpu.get(position) != cuda.get(position))
}

fn margin_at(
    model: &LoadedModel,
    ids: &[u32],
    forced: &[u32],
    position: usize,
    gpu_layers: i32,
    backend: &'static str,
) -> Result<Top2, ExampleError> {
    let rows = model.teacher_forced_logits_from_ids(ids, forced, &config(gpu_layers))?;
    rows.get(position)
        .map(|row| Top2::of(row))
        .ok_or(ExampleError::MissingRow { backend, position })
}

fn report_margins(
    model: &LoadedModel,
    vocab: &Vocab,
    prompt: &str,
    cpu: &[u32],
    position: usize,
) -> Result<(), ExampleError> {
    let ids = prompt_ids(vocab, prompt)?;
    let forced = &cpu[..=position.min(cpu.len().saturating_sub(1))];
    let cpu_margin = margin_at(model, &ids, forced, position, 0, "cpu")?;
    let cuda_margin = margin_at(model, &ids, forced, position, GPU_LAYERS_ALL, "cuda")?;
    println!("margin position={position} cpu {}", cpu_margin.render());
    println!("margin position={position} cuda {}", cuda_margin.render());
    Ok(())
}

fn run(path: &str, prompt: &str, tokens: usize, backends: Backends) -> Result<(), ExampleError> {
    let io_error = |source| ExampleError::Io {
        path: path.to_owned(),
        source,
    };
    let file = File::open(path).map_err(io_error)?;
    // the checkpoint is not written while this process reads it
    let mapping = unsafe { Mmap::map(&file) }.map_err(io_error)?;
    let parsed = parse_complete(&mapping).map_err(|error| ExampleError::Gguf {
        path: path.to_owned(),
        message: error.to_string(),
    })?;
    let vocab = vocab_from_metadata(&parsed)?;
    let model = LoadedModel::load(&parsed, &mapping)?;
    println!("prompt_ids {:?}", prompt_ids(&vocab, prompt)?);
    let cpu = match backends {
        Backends::Both => Some(decode(&model, prompt, tokens, 0)?),
        Backends::Cuda => None,
    };
    let (cuda, cuda_text, cuda_timing) = decode(&model, prompt, tokens, GPU_LAYERS_ALL)?;
    if let Some((_, _, cpu_timing)) = &cpu {
        cpu_timing.report("cpu");
    }
    cuda_timing.report("cuda");
    println!("decode_text backend=cuda ids={cuda:?} text={cuda_text:?}");
    match cpu {
        Some((cpu_ids, cpu_text, _)) => {
            println!("decode_text backend=cpu ids={cpu_ids:?} text={cpu_text:?}");
            summarize(&model, &vocab, path, prompt, &cpu_ids, &cuda)
        }
        None => {
            println!(
                "decode_compare model={} tokens={} matched=n/a first_divergence=n/a",
                model_label(&model, path),
                cuda.len()
            );
            Ok(())
        }
    }
}

fn model_label(model: &LoadedModel, path: &str) -> String {
    model.model_name().map(str::to_owned).unwrap_or_else(|| {
        Path::new(path)
            .file_stem()
            .map_or_else(String::new, |stem| stem.to_string_lossy().into_owned())
    })
}

fn summarize(
    model: &LoadedModel,
    vocab: &Vocab,
    path: &str,
    prompt: &str,
    cpu: &[u32],
    cuda: &[u32],
) -> Result<(), ExampleError> {
    let show =
        |ids: &[u32], position: usize| ids.get(position).map_or("-".to_owned(), u32::to_string);
    let produced = cpu.len().max(cuda.len());
    let mut matched = 0;
    for position in 0..produced {
        let same = cpu.get(position).is_some() && cpu.get(position) == cuda.get(position);
        matched += usize::from(same);
        let verdict = if same { "yes" } else { "no" };
        println!(
            "pos={position} cpu={} cuda={} match={verdict}",
            show(cpu, position),
            show(cuda, position)
        );
    }
    let divergence = first_divergence(cpu, cuda);
    let name = model_label(model, path);
    let rendered = divergence.map_or("none".to_owned(), |position| position.to_string());
    println!(
        "decode_compare model={name} tokens={produced} matched={matched} first_divergence={rendered}"
    );
    divergence
        .filter(|&position| position < cpu.len())
        .map_or(Ok(()), |position| {
            report_margins(model, vocab, prompt, cpu, position)
        })
}

fn install_telemetry() -> Result<Option<Arc<Recorder>>, ExampleError> {
    let Ok(log_path) = env::var("DECODE_COMPARE_TELEMETRY") else {
        return Ok(None);
    };
    global::install(EnvFilter::parse("proxima_model_interop=debug,omega=debug"));
    let recorder = Recorder::builder()
        .export(Exporter::file(&log_path).format(Formatter::Text))
        .map_err(|error| ExampleError::Telemetry(error.to_string()))?
        .install()
        .map_err(|error| ExampleError::Telemetry(error.to_string()))?;
    Ok(Some(recorder))
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let outcome = args.next().ok_or(ExampleError::Usage).and_then(|path| {
        let prompt = args.next().unwrap_or_else(|| DEFAULT_PROMPT.to_owned());
        let tokens = args.next().map_or(Ok(DEFAULT_TOKENS), |text| {
            text.parse().map_err(|_| ExampleError::BadCount(text))
        })?;
        let backends = args
            .next()
            .map_or(Ok(Backends::Both), |text| match text.as_str() {
                "both" => Ok(Backends::Both),
                "cuda" => Ok(Backends::Cuda),
                _ => Err(ExampleError::BadBackends(text)),
            })?;
        let recorder = install_telemetry()?;
        let result = run(&path, &prompt, tokens, backends);
        if let Some(recorder) = recorder {
            let _drained = recorder.drain();
        }
        result
    });
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("decode_compare: {error}");
            ExitCode::FAILURE
        }
    }
}
