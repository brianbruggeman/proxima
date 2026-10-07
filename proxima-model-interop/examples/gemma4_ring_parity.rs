//! Determinism and external-oracle harness for gemma4-E2B long-context work
//! (`proxima-tensor/specs/long-context/SPEC.md`).
//!
//! Usage:
//! `cargo run -p proxima-model-interop --example gemma4_ring_parity --features std,metal -- \
//!   <gguf> (--record FILE [--max-tokens N] | --expect FILE [--ring-offset ROWS] | --ollama-facts)`
//!
//! `--record` decodes the same 2,048-token chat-templated prompt for up to
//! `--max-tokens N` (default 256) greedy tokens through the full KV cache
//! (`KvLayout::Full`), writes however many the run produced, and prints
//! `recorded K`. `--expect` decodes it twice, once through the full cache and
//! once through the sliding ring (`KvLayout::SlidingRing`, what
//! `LoadedModel::load` binds), and prints `full vs head: M/K` and
//! `ring vs head: M/K` plus the first divergence index. A length difference is a
//! mismatch: positions past the shorter side count as mismatches.
//! The prompt is public-domain text (Gettysburg Address, Declaration of
//! Independence, Moby-Dick, Pride and Prejudice, Origin of Species) cycled and
//! trimmed to exactly 2,048 tokens. `--ollama-facts` sends the four
//! correctness-gate prompts to Ollama raw.
//!
//! `--ring-offset ROWS` is the control: it writes every sliding-ring row that
//! many slots away from where the read looks for it, so `ring vs head` must
//! fall below K while `full vs head` stays K. A ring check that still
//! passes with an offset is not reading the ring.
//!
//! Speculation runs at the production default, so the ring is built with its
//! draft slack and verify evaluations rewind it; greedy ids are unchanged by
//! speculation, which is what the comparison to the head recording relies on.

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use memmap2::Mmap;
use proxima_gguf::GgmlType;
use proxima_gguf::parse_complete;
use proxima_model_interop::{GPU_LAYERS_ALL, InteropError, KvLayout, LoadedModel, ServingConfig};
use proxima_tokenizer::gguf::vocab_from_metadata;
use proxima_tokenizer::{TokenizerError, Vocab, encode_with_bos_eos};
use serde_json::{Value, json};

const PROMPT_TOKENS: usize = 2048;
const DEFAULT_MAX_TOKENS: usize = 256;
const FACT_MAX_TOKENS: usize = 48;
const OLLAMA_ADDRESS: &str = "localhost:11434";
const OLLAMA_MODEL: &str = "gemma4:e2b-it-qat";
const FILLER_WORD: &str = ".";
const CLOSING_QUESTION: &str = "In two sentences, what are these passages about?";

const CORPUS: [&str; 5] = [
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live.",
    "When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.",
    "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth, whenever it is a damp, drizzly November in my soul, then I account it high time to get to sea as soon as I can.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.",
    "When we look to the individuals of the same variety or sub-variety of our older cultivated plants and animals, one of the first points which strikes us, is, that they generally differ much more from each other, than do the individuals of any one species or variety in a state of nature. When we reflect on the vast diversity of the plants and animals which have been cultivated, and which have varied during all ages under the most different climates and treatment, we are driven to conclude that this great variability is due to our domestic productions having been raised under conditions of life not so uniform as, and somewhat different from, those to which the parent-species had been exposed under nature.",
];

#[derive(Debug, thiserror::Error)]
enum ExampleError {
    #[error(
        "usage: gemma4_ring_parity <model.gguf> (--record FILE [--max-tokens N] | --expect FILE [--ring-offset ROWS] | --ollama-facts)"
    )]
    Usage,
    #[error("{context}: {source}")]
    Io {
        context: String,
        source: std::io::Error,
    },
    #[error("interop: {0}")]
    Interop(#[from] InteropError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Failed(String),
}

enum Mode {
    Record { output: PathBuf, max_tokens: usize },
    Expect { head: PathBuf, ring_offset: usize },
    OllamaFacts,
}

struct Divergence {
    matched: usize,
    compared: usize,
    first: Option<usize>,
}

enum CacheArm {
    Full,
    Ring { write_offset: usize },
}

impl CacheArm {
    fn label(&self) -> &'static str {
        match self {
            CacheArm::Full => "full",
            CacheArm::Ring { .. } => "ring",
        }
    }

    fn layout(&self) -> KvLayout {
        match self {
            CacheArm::Full => KvLayout::Full,
            CacheArm::Ring { .. } => KvLayout::SlidingRing,
        }
    }

    fn write_offset(&self) -> usize {
        match self {
            CacheArm::Full => 0,
            CacheArm::Ring { write_offset } => *write_offset,
        }
    }
}

struct FactCheck {
    name: &'static str,
    prompt: String,
    expected_substring: &'static str,
}

fn chat_prompt(user_turn: &str) -> String {
    format!("<|turn>user\n{user_turn}<turn|>\n<|turn>model\n")
}

fn fact_checks() -> Vec<FactCheck> {
    vec![
        FactCheck {
            name: "paris",
            prompt: "The capital of France is".to_string(),
            expected_substring: "paris",
        },
        FactCheck {
            name: "soliloquy",
            prompt: "In drama, a speech in which a character, alone on stage, speaks their inner thoughts aloud is called a".to_string(),
            expected_substring: "soliloquy",
        },
        FactCheck {
            name: "ant_vs_briefcase",
            prompt: chat_prompt("Which is bigger, an ant or a briefcase?"),
            expected_substring: "briefcase",
        },
        FactCheck {
            name: "hippo_vs_building",
            prompt: chat_prompt(
                "Which of these is smaller in size: a hippopotamus or a large office building?",
            ),
            expected_substring: "hippopotamus",
        },
    ]
}

fn io_error(context: &str, source: std::io::Error) -> ExampleError {
    ExampleError::Io {
        context: context.to_string(),
        source,
    }
}

fn parse_mode(mut args: impl Iterator<Item = String>) -> Result<Mode, ExampleError> {
    let mode = match args.next().as_deref() {
        Some("--record") => {
            let output = args.next().ok_or(ExampleError::Usage)?.into();
            let max_tokens = match args.next().as_deref() {
                None => DEFAULT_MAX_TOKENS,
                Some("--max-tokens") => args
                    .next()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|value| *value > 0)
                    .ok_or(ExampleError::Usage)?,
                Some(_) => return Err(ExampleError::Usage),
            };
            Mode::Record { output, max_tokens }
        }
        Some("--expect") => {
            let head = args.next().ok_or(ExampleError::Usage)?.into();
            let ring_offset = match args.next().as_deref() {
                None => 0,
                Some("--ring-offset") => args
                    .next()
                    .and_then(|value| value.parse::<usize>().ok())
                    .ok_or(ExampleError::Usage)?,
                Some(_) => return Err(ExampleError::Usage),
            };
            Mode::Expect { head, ring_offset }
        }
        Some("--ollama-facts") => Mode::OllamaFacts,
        _ => return Err(ExampleError::Usage),
    };
    match args.next() {
        None => Ok(mode),
        Some(_) => Err(ExampleError::Usage),
    }
}

fn corpus_words() -> Vec<&'static str> {
    CORPUS
        .iter()
        .cycle()
        .take(CORPUS.len() * 8)
        .flat_map(|paragraph| paragraph.split_whitespace())
        .collect()
}

fn templated_prompt(passage: &str) -> String {
    chat_prompt(&format!("{passage}\n\n{CLOSING_QUESTION}"))
}

fn token_count(vocab: &Vocab, passage: &str) -> Result<usize, ExampleError> {
    let ids = encode_with_bos_eos(&templated_prompt(passage), vocab, true, false)?;
    Ok(ids.len())
}

fn largest_word_prefix(vocab: &Vocab, words: &[&str]) -> Result<usize, ExampleError> {
    let (mut low, mut high) = (0, words.len());
    while low < high {
        let middle = (low + high).div_ceil(2);
        if token_count(vocab, &words[..middle].join(" "))? <= PROMPT_TOKENS {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low)
}

fn build_long_prompt(vocab: &Vocab) -> Result<String, ExampleError> {
    let words = corpus_words();
    let prefix_words = largest_word_prefix(vocab, &words)?;
    let mut passage = words[..prefix_words].join(" ");
    while token_count(vocab, &passage)? < PROMPT_TOKENS {
        passage.push(' ');
        passage.push_str(FILLER_WORD);
    }
    let count = token_count(vocab, &passage)?;
    if count != PROMPT_TOKENS {
        return Err(ExampleError::Failed(format!(
            "prompt is {count} tokens, expected {PROMPT_TOKENS}"
        )));
    }
    Ok(templated_prompt(&passage))
}

fn serving_config() -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

fn run_arm(model: &LoadedModel, prompt: &str, max_tokens: usize) -> Result<Vec<u32>, ExampleError> {
    let (ids, _text, _stopped_by_eos) =
        model.generate_with_serving_config(prompt, max_tokens, serving_config())?;
    Ok(ids)
}

fn read_head_ids(path: &Path) -> Result<Vec<u32>, ExampleError> {
    let text =
        fs::read_to_string(path).map_err(|source| io_error(&path.display().to_string(), source))?;
    let ids = text
        .split_whitespace()
        .map(|token| {
            token.parse::<u32>().map_err(|error| {
                ExampleError::Failed(format!(
                    "{}: bad token id {token:?}: {error}",
                    path.display()
                ))
            })
        })
        .collect::<Result<Vec<u32>, ExampleError>>()?;
    Ok(ids)
}

fn compare_to_head(ids: &[u32], head: &[u32]) -> Divergence {
    let compared = ids.len().max(head.len());
    let first = (0..compared).find(|&index| ids.get(index) != head.get(index));
    let matched = ids
        .iter()
        .zip(head)
        .filter(|(left, right)| left == right)
        .count();
    Divergence {
        matched,
        compared: head.len(),
        first,
    }
}

fn report_against_head(arm: &CacheArm, ids: &[u32], head: &[u32]) {
    let divergence = compare_to_head(ids, head);
    let first = divergence
        .first
        .map_or_else(|| "none".to_string(), |index| index.to_string());
    println!(
        "{} vs head: {}/{} first_divergence={first} arm_len={} head_len={}",
        arm.label(),
        divergence.matched,
        divergence.compared,
        ids.len(),
        head.len()
    );
}

fn with_model<T>(
    gguf_path: &str,
    arm: &CacheArm,
    action: impl FnOnce(&LoadedModel, &Vocab) -> Result<T, ExampleError>,
) -> Result<T, ExampleError> {
    let file = fs::File::open(gguf_path).map_err(|source| io_error(gguf_path, source))?;
    // SAFETY: read-only mapping of a checkpoint no other process writes during this run.
    let mapping = unsafe { Mmap::map(&file) }.map_err(|source| io_error(gguf_path, source))?;
    let parsed = parse_complete(&mapping)
        .map_err(|error| ExampleError::Failed(format!("{gguf_path}: gguf parse: {error}")))?;
    let vocab = vocab_from_metadata(&parsed)?;
    let model = LoadedModel::load_with_kv_layout(&parsed, &mapping, arm.layout())?
        .with_ring_write_offset_for_parity_control(arm.write_offset());
    action(&model, &vocab)
}

fn record(gguf_path: &str, output: &Path, max_tokens: usize) -> Result<(), ExampleError> {
    with_model(gguf_path, &CacheArm::Full, |model, vocab| {
        let prompt = build_long_prompt(vocab)?;
        let ids = run_arm(model, &prompt, max_tokens)?;
        let line = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
        fs::write(output, line)
            .map_err(|source| io_error(&output.display().to_string(), source))?;
        println!("recorded {}", ids.len());
        Ok(())
    })
}

fn expect_arm(gguf_path: &str, arm: &CacheArm, head: &[u32]) -> Result<(), ExampleError> {
    with_model(gguf_path, arm, |model, vocab| {
        let prompt = build_long_prompt(vocab)?;
        let ids = run_arm(model, &prompt, head.len() + 1)?;
        report_against_head(arm, &ids, head);
        Ok(())
    })
}

fn expect(gguf_path: &str, head_path: &Path, ring_offset: usize) -> Result<(), ExampleError> {
    let head = read_head_ids(head_path)?;
    expect_arm(gguf_path, &CacheArm::Full, &head)?;
    expect_arm(
        gguf_path,
        &CacheArm::Ring {
            write_offset: ring_offset,
        },
        &head,
    )
}

fn decode_chunked(body: &[u8]) -> Result<Vec<u8>, ExampleError> {
    let mut decoded = Vec::new();
    let mut rest = body;
    loop {
        let line_end = rest
            .windows(2)
            .position(|pair| pair == b"\r\n")
            .ok_or_else(|| ExampleError::Failed("chunked body: missing size line".to_string()))?;
        let size_text = String::from_utf8_lossy(&rest[..line_end]);
        let size = usize::from_str_radix(size_text.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|error| ExampleError::Failed(format!("chunked body: bad size: {error}")))?;
        let start = line_end + 2;
        if size == 0 {
            return Ok(decoded);
        }
        let chunk = rest
            .get(start..start + size)
            .ok_or_else(|| ExampleError::Failed("chunked body: truncated chunk".to_string()))?;
        decoded.extend_from_slice(chunk);
        rest = rest
            .get(start + size + 2..)
            .ok_or_else(|| ExampleError::Failed("chunked body: missing chunk end".to_string()))?;
    }
}

fn http_post_json(path: &str, body: &str) -> Result<Vec<u8>, ExampleError> {
    let mut stream =
        TcpStream::connect(OLLAMA_ADDRESS).map_err(|source| io_error(OLLAMA_ADDRESS, source))?;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {OLLAMA_ADDRESS}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|source| io_error("ollama write", source))?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|source| io_error("ollama read", source))?;
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| ExampleError::Failed("ollama: response has no header end".to_string()))?;
    let header = String::from_utf8_lossy(&raw[..header_end]).to_lowercase();
    let payload = &raw[header_end + 4..];
    if !header.starts_with("http/1.1 200") {
        return Err(ExampleError::Failed(format!(
            "ollama: {} body={}",
            header.lines().next().unwrap_or(""),
            String::from_utf8_lossy(payload)
        )));
    }
    if header.contains("transfer-encoding: chunked") {
        decode_chunked(payload)
    } else {
        Ok(payload.to_vec())
    }
}

fn ollama_answer(check: &FactCheck) -> Result<String, ExampleError> {
    let body = json!({
        "model": OLLAMA_MODEL,
        "prompt": check.prompt,
        "raw": true,
        "stream": false,
        "options": {"temperature": 0, "num_predict": FACT_MAX_TOKENS},
    });
    let payload = http_post_json("/api/generate", &body.to_string())?;
    let reply: Value = serde_json::from_slice(&payload)?;
    reply
        .get("response")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| ExampleError::Failed(format!("{}: reply has no response field", check.name)))
}

fn ollama_facts() -> Result<(), ExampleError> {
    for check in fact_checks() {
        let text = ollama_answer(&check)?;
        let answered_correctly = text.to_lowercase().contains(check.expected_substring);
        println!(
            "ollama answered_correctly={answered_correctly} check={} text={text:?}",
            check.name
        );
    }
    Ok(())
}

fn run() -> Result<(), ExampleError> {
    let mut args = env::args().skip(1);
    let gguf_path = args.next().ok_or(ExampleError::Usage)?;
    match parse_mode(args)? {
        Mode::Record { output, max_tokens } => record(&gguf_path, &output, max_tokens),
        Mode::Expect { head, ring_offset } => expect(&gguf_path, &head, ring_offset),
        Mode::OllamaFacts => ollama_facts(),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gemma4_ring_parity: {error}");
            ExitCode::FAILURE
        }
    }
}
