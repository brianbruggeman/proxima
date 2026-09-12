//! Score one query/document pair with a Qwen3-style yes/no reranker.
//!
//! Usage: `rerank_local <model.gguf> <cpu|cuda|vulkan> <query> <document>`

#![allow(clippy::expect_used)]

use std::env;
use std::fs::File;
use std::time::Instant;

use memmap2::MmapOptions;
use proxima_gguf::parse_complete;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ModelTask, classify_task};

fn usage() -> ! {
    eprintln!("usage: rerank_local <model.gguf> <cpu|cuda|vulkan> <query> <document>");
    std::process::exit(2)
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().unwrap_or_else(|| usage());
    let backend = args.next().unwrap_or_else(|| usage());
    let query = args.next().unwrap_or_else(|| usage());
    let document = args.next().unwrap_or_else(|| usage());
    if args.next().is_some() || !matches!(backend.as_str(), "cpu" | "cuda" | "vulkan") {
        usage();
    }
    if backend == "cuda" && !cfg!(feature = "cuda") {
        eprintln!("rerank_local: cuda mode requires --features cuda");
        std::process::exit(2);
    }
    if backend == "vulkan" && !cfg!(feature = "vulkan") {
        eprintln!("rerank_local: vulkan mode requires --features vulkan");
        std::process::exit(2);
    }
    if backend != "cpu" && cfg!(all(feature = "cuda", feature = "vulkan")) {
        eprintln!("rerank_local: build exactly one of cuda or vulkan");
        std::process::exit(2);
    }
    if backend == "cuda" {
        #[cfg(feature = "cuda")]
        {
            let driver = omega::CudaDriver::new(0)
                .unwrap_or_else(|error| panic!("cuda device unavailable: {error}"));
            driver.memory_info().expect("cuda memory query");
        }
    }
    if backend == "vulkan" {
        #[cfg(feature = "vulkan")]
        omega::probe_wgpu().expect("vulkan adapter");
    }

    let file = File::open(&path).expect("open GGUF checkpoint");
    // SAFETY: `file` remains alive for the mapping and `model` borrows it only
    // until this process exits.
    let mapping = unsafe { MmapOptions::new().map(&file) }.expect("map GGUF checkpoint");
    let parsed = parse_complete(&mapping).expect("parse GGUF checkpoint");
    let profile = classify_task(&parsed);
    if profile.task != ModelTask::Reranker {
        eprintln!(
            "rerank_local: expected reranker task, detected {} ({:?})",
            profile.task.name(),
            profile.evidence
        );
        std::process::exit(3);
    }

    let model = LoadedModel::load(&parsed, &mapping).expect("bind reranker checkpoint");
    let yes = model
        .token_id_for_piece("yes")
        .expect("reranker vocabulary contains the yes token");
    let no = model
        .token_id_for_piece("no")
        .expect("reranker vocabulary contains the no token");
    let instruction = "Given a web search query, retrieve relevant passages that answer the query";
    let prompt = format!(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let started = Instant::now();
    let logits = model
        .forward_logits_on_backend(&prompt, if backend == "cpu" { 0 } else { GPU_LAYERS_ALL })
        .expect("run reranker forward");
    let forward_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let yes_logit = logits[yes as usize];
    let no_logit = logits[no as usize];
    let max_logit = yes_logit.max(no_logit);
    let yes_probability = (f64::from(yes_logit - max_logit)).exp()
        / ((f64::from(yes_logit - max_logit)).exp() + (f64::from(no_logit - max_logit)).exp());
    println!(
        "{{\"task\":\"reranker\",\"yes_token\":{yes},\"no_token\":{no},\"yes_logit\":{yes_logit:.7},\"no_logit\":{no_logit:.7},\"yes_probability\":{yes_probability:.7},\"forward_ms\":{forward_ms:.3}}}"
    );
}
