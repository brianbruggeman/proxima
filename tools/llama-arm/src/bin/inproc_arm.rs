//! one llama-inproc arm of `kv_scoreboard`: a single process, the pinned libllama behind its
//! sha256 + build-number gate, greedy decode over prompt ids the scoreboard supplies.
//!
//! every line goes to stderr in the shapes `kv_scoreboard` already parses from the proxima arm:
//! `step_time run_index= step= step_ms= step_ns= phase=` per token (step 0 is the prefill, the
//! rest are decode), `token_ids run_index= ids=[..]` and `run=done run_index= tokens_generated=
//! prompt_token_count=`, plus one `gate` line carrying the sha256, build number and the context
//! settings that were requested. the scoreboard owns the span arithmetic; this bin only reports
//! records. timings are whatever the box is doing; the scoreboard labels them.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use llama_arm::{DecodeRequest, GreedyRun, KvType, greedy_decode_runs, lib_dir_from_env, load};

const LINE_PREFIX: &str = "llama_arm";

struct Arguments {
    model: PathBuf,
    ids: Vec<i32>,
    n_ctx: u32,
    n_predict: usize,
    runs: usize,
    type_k: KvType,
    type_v: KvType,
}

fn parse_arguments(arguments: impl IntoIterator<Item = String>) -> Result<Arguments, String> {
    let mut values: BTreeMap<String, String> = BTreeMap::new();
    let mut tokens = arguments.into_iter();
    while let Some(flag) = tokens.next() {
        let value = tokens.next().ok_or_else(|| format!("flag {flag} needs a value"))?;
        values.insert(flag, value);
    }
    let required = |flag: &str| values.get(flag).ok_or_else(|| format!("missing required flag {flag}"));
    let number = |flag: &str| {
        let text = required(flag)?;
        text.parse::<u32>().map_err(|_| format!("{flag} expects an integer, got {text}"))
    };
    let kv_type = |flag: &str| {
        let text = required(flag)?;
        KvType::parse(text).ok_or_else(|| format!("{flag} expects f32|f16|bf16, got {text}"))
    };
    let ids = required("--ids")?
        .split(',')
        .map(|text| text.trim().parse::<i32>().map_err(|_| format!("--ids has a non-integer entry {text:?}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arguments {
        model: PathBuf::from(required("--model")?),
        ids,
        n_ctx: number("--n-ctx")?,
        n_predict: usize::try_from(number("--n-predict")?).map_err(|error| error.to_string())?,
        runs: usize::try_from(number("--runs")?).map_err(|error| error.to_string())?,
        type_k: kv_type("--type-k")?,
        type_v: kv_type("--type-v")?,
    })
}

fn emit_run(run_index: usize, run: &GreedyRun, prompt_len: usize) {
    let (ids, step_ns, _) = run;
    for (step, &nanos) in step_ns.iter().enumerate() {
        let phase = if step == 0 { "prefill" } else { "decode" };
        eprintln!("{LINE_PREFIX} step_time run_index={run_index} step={step} step_ms={} step_ns={nanos} phase={phase}", nanos / 1_000_000);
    }
    eprintln!("{LINE_PREFIX} token_ids run_index={run_index} ids={ids:?}");
    eprintln!(
        "{LINE_PREFIX} run=done run_index={run_index} tokens_generated={} prompt_token_count={prompt_len}",
        ids.len()
    );
}

fn run(arguments: &Arguments) -> Result<(), String> {
    let dir = lib_dir_from_env();
    let lib = load(&dir).map_err(|error| error.to_string())?;
    eprintln!(
        "{LINE_PREFIX} gate sha256={} build={} type_k={} type_v={} flash_attn=enabled n_ctx={} n_predict={} runs={}",
        lib.report.sha256,
        lib.report.build_number,
        arguments.type_k.name(),
        arguments.type_v.name(),
        arguments.n_ctx,
        arguments.n_predict,
        arguments.runs,
    );
    let request = DecodeRequest {
        model_path: &arguments.model,
        prompt: &arguments.ids,
        n_ctx: arguments.n_ctx,
        n_predict: arguments.n_predict,
        type_k: arguments.type_k,
        type_v: arguments.type_v,
        runs: arguments.runs,
    };
    let runs = greedy_decode_runs(&lib.api, &request).map_err(|error| error.to_string())?;
    for (run_index, run) in runs.iter().enumerate() {
        emit_run(run_index, run, arguments.ids.len());
    }
    Ok(())
}

fn main() -> ExitCode {
    let outcome = parse_arguments(std::env::args().skip(1)).and_then(|arguments| run(&arguments));
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("inproc_arm failed: {message}");
            ExitCode::FAILURE
        }
    }
}
