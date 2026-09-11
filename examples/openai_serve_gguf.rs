//! An OpenAI-compatible `POST /v1/chat/completions` server backed by a real
//! GGUF checkpoint, served over proxima's own tokio-free h1 listener.
//! Mirrors `examples/h1_native_prime_round_trip.rs` for the serve side
//! (`PrimeRuntime::builder()` + `PrimeServeExt::serve_http` +
//! `proxima::pipe::into_handle`) and `examples/qwen_gguf_smoke.rs` for the
//! model side (`proxima_gguf::pipe::parse_complete` ->
//! `proxima_model_interop::LoadedModel::load` ->
//! `generate_with_serving_config`).
//!
//! Request/response shapes are fixed to the minimal OpenAI chat-completions
//! wire contract this server is meant to satisfy -- just enough of it to
//! round-trip a chat request against a real checkpoint:
//!
//! ```text
//! POST /v1/chat/completions
//! {"model": "...", "messages": [{"role": "user", "content": "..."}]}
//!
//! -> {"choices": [{"message": {"content": "..."}}],
//!     "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}}
//! ```
//!
//! Chat template: this crate reads `tokenizer.chat_template`
//! (`proxima-model-interop/src/bind.rs:3340` reads the same key) only to
//! report whether the checkpoint carries one -- there is no jinja2 renderer
//! in this workspace's dependency graph, so rendering it correctly is out
//! of scope for this smoke server; every request is rendered by plain
//! `role: content` concatenation regardless, and which path ran is logged
//! per request.
//!
//! `LoadedModel<'file>` borrows the checkpoint bytes for as long as it
//! lives (`generate.rs`'s own doc: "this crate never opens a file itself").
//! A long-lived server needs those bytes to outlive every request, so this
//! binary reads the whole file once at startup and leaks it
//! (`Box::leak`) into a `&'static [u8]` -- a deliberate, one-time,
//! whole-process-lifetime leak, not a per-request one; the alternative
//! (an `Arc<Vec<u8>>` the model borrows through a self-referential
//! struct) is what `LoadedModel`'s own borrowed-not-owned design already
//! avoids needing.
//!
//! Run: `cargo run --example openai_serve_gguf --features "serve-prime,
//! runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,
//! runtime-prime-bgpool,http1-native,macros" -- <gguf-path> [bind-addr]
//! [max-tokens]`
//!
//! Set `PROXIMA_QWEN35MOE_SIDECAR` to mmap and validate an expert sidecar at
//! startup. The model owns the mapping and descriptor index after
//! `LoadedModel::attach_expert_sidecar`; every routed-expert boundary can
//! switch its three projection sources without reopening or copying it.

use std::error::Error;
use std::fmt::Write as _;
use std::fs::File;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use memmap2::MmapOptions;
use proxima::SendPipe;
use proxima::error::ProximaError;
use proxima::pipe::into_handle;
use proxima::prime::PrimeRuntime;
use proxima::request::{Request, Response};
use proxima::runtime::PrimeServeExt;
use proxima_gguf::pipe::parse_complete;
use proxima_gguf::value::MetadataValue;
use proxima_model_interop::{LoadedModel, MappedExpertSidecar, ServingConfig};
use serde::Deserialize;
use serde_json::json;

const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
const DEFAULT_BIND_ADDR: &str = "127.0.0.1:8081";
const DEFAULT_MAX_TOKENS: usize = 64;
const SIDECAR_ENV: &str = "PROXIMA_QWEN35MOE_SIDECAR";

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[allow(dead_code)]
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// The model plus everything a request needs to build a prompt and render
/// a response: whether the checkpoint carries `tokenizer.chat_template`
/// (see module doc for why that key is only observed, not rendered), and
/// the per-request `max_tokens` budget this binary was started with.
struct ServedModel {
    model: LoadedModel<'static>,
    vocab: proxima_tokenizer::Vocab,
    has_chat_template: bool,
    max_tokens: usize,
}

fn load_expert_sidecar() -> Result<Option<Arc<memmap2::Mmap>>, Box<dyn Error>> {
    let Some(path) = std::env::var_os(SIDECAR_ENV) else {
        return Ok(None);
    };
    let file = File::open(&path)?;
    // the server retains the read-only mapping for its whole process lifetime.
    let mapping = Arc::new(unsafe { MmapOptions::new().map(&file)? });
    MappedExpertSidecar::new(Arc::clone(&mapping))
        .map_err(|error| format!("{} {:?}: {error}", SIDECAR_ENV, path))?;
    Ok(Some(mapping))
}

fn render_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let _ = writeln!(prompt, "{}: {}", message.role, message.content);
    }
    prompt.push_str("assistant: ");
    prompt
}

fn supported_serving_config(model_path: &str, qwen35moe_pre_gather: bool) -> ServingConfig<'_> {
    ServingConfig {
        model_path,
        kv_cache_key_quant: proxima_gguf::types::GgmlType::F32,
        kv_cache_value_quant: proxima_gguf::types::GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers: 0,
        reasoning_budget: 0,
        qwen35moe_pre_gather,
        ..ServingConfig::default()
    }
}

struct ChatCompletions {
    served: Arc<ServedModel>,
    gguf_path: String,
}

impl SendPipe for ChatCompletions {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            if request.method != "POST" || request.path.as_ref() != CHAT_COMPLETIONS_PATH.as_bytes()
            {
                return Ok(Response::not_found());
            }
            let (_, body) = request.body_bytes().await?;
            let chat_request: ChatCompletionRequest = serde_json::from_slice(&body)
                .map_err(|error| ProximaError::Decode(format!("chat request json: {error}")))?;

            let prompt_template_path = if self.served.has_chat_template {
                "plain-concat (chat_template present but not rendered, see module doc)"
            } else {
                "plain-concat (no tokenizer.chat_template key)"
            };
            let prompt = render_prompt(&chat_request.messages);

            let sidecar_descriptors = self.served.model.expert_sidecar_descriptor_count();
            let serving_config =
                supported_serving_config(&self.gguf_path, sidecar_descriptors != 0);
            let generate_started = Instant::now();
            let (generated_ids, text, _stopped_by_eos) = self
                .served
                .model
                .generate_with_serving_config(&prompt, self.served.max_tokens, serving_config)
                .map_err(|error| ProximaError::Decode(format!("generation failed: {error}")))?;
            let generate_ms = generate_started.elapsed().as_secs_f64() * 1000.0;

            let prompt_tokens = proxima_tokenizer::encode_with_bos_eos(
                &prompt,
                &self.served.vocab,
                self.served.vocab.add_bos_token().unwrap_or(true),
                false,
            )
            .map(|ids| ids.len())
            .unwrap_or(0);
            let completion_tokens = generated_ids.len();
            let tokens_per_sec = if generate_ms > 0.0 {
                (completion_tokens as f64) / (generate_ms / 1000.0)
            } else {
                0.0
            };

            println!(
                "prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} \
                 generate_ms={generate_ms:.3} tokens_per_sec={tokens_per_sec:.3} \
                 prompt_render={prompt_template_path} sidecar_descriptors={sidecar_descriptors}"
            );

            let body = json!({
                "choices": [{ "message": { "content": text } }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            });
            Ok(
                Response::ok(Bytes::from(serde_json::to_vec(&body).map_err(|error| {
                    ProximaError::Encode(format!("chat response json: {error}"))
                })?))
                .with_header("content-type", "application/json"),
            )
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let gguf_path = args.next().ok_or("argv[1]: path to a .gguf checkpoint")?;
    let bind_addr = args.next().unwrap_or_else(|| DEFAULT_BIND_ADDR.to_string());
    let max_tokens: usize = args
        .next()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(DEFAULT_MAX_TOKENS);

    println!("gguf_path = {gguf_path}");
    println!("bind_addr = {bind_addr}");
    println!("max_tokens = {max_tokens}");

    let expert_sidecar = load_expert_sidecar()?;
    if let Some(mapping) = expert_sidecar.as_ref() {
        let sidecar = MappedExpertSidecar::new(Arc::clone(mapping))?;
        println!(
            "expert_sidecar_bytes = {} expert_sidecar_descriptors = {}",
            mapping.len(),
            sidecar.descriptor_count()
        );
    }

    let load_started = Instant::now();
    let file_bytes: &'static [u8] = Box::leak(std::fs::read(&gguf_path)?.into_boxed_slice());
    let parsed =
        parse_complete(file_bytes).map_err(|error| format!("gguf parse failed: {error}"))?;
    let has_chat_template = matches!(
        parsed.metadata_value("tokenizer.chat_template"),
        Some(MetadataValue::String(_))
    );
    println!("has_chat_template = {has_chat_template}");

    let mut model = LoadedModel::load(&parsed, file_bytes)
        .map_err(|error| format!("weight load failed: {error}"))?;
    if let Some(mapping) = expert_sidecar {
        model
            .attach_expert_sidecar(mapping)
            .map_err(|error| format!("expert sidecar attach failed: {error}"))?;
    }
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    println!("weight_load_ms = {load_ms:.3}");

    // `LoadedModel` borrows the checkpoint bytes and keeps its own `vocab`
    // private (`generate.rs`'s own doc: the crate owns tokenization
    // internally for `generate_with_serving_config`), so this server builds
    // its own copy from the same parsed metadata to compute `prompt_tokens`
    // for the response's `usage` block.
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .map_err(|error| format!("vocab build failed: {error}"))?;

    let served = Arc::new(ServedModel {
        model,
        vocab,
        has_chat_template,
        max_tokens,
    });
    let pipe = ChatCompletions {
        served,
        gguf_path: gguf_path.clone(),
    };
    let handle_pipe = into_handle(pipe);

    let runtime = Arc::new(
        PrimeRuntime::builder()
            .cores(1)
            .background_inline()
            .build()?,
    );
    let addr = bind_addr.parse()?;
    let handle = runtime.serve_http(addr, handle_pipe)?;
    let bound = handle
        .bind_addr()
        .ok_or("listener did not report a bound address")?;

    println!("LISTENING {bound}");
    use std::io::Write as _;
    std::io::stdout().flush()?;

    loop {
        std::thread::park();
    }
}
