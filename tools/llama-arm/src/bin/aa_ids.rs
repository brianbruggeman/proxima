//! A/A check: the in-process greedy decode against the pinned llama-server on the same prompt ids.
//!
//! prompts, model, `-c` rule and server flags mirror `proxima-model-interop/examples/kv_scoreboard.rs`
//! (`SHORT_PROMPT`, `kv_capacity`, `llama_padded_capacity`, `TimedServer::spawn`, `completion_body`).
//! ids are tokenized once by the pinned libllama and sent to both arms. timings are whatever the box
//! is doing right now; they are labelled not quiet.

use std::env;
use std::fs::{self, File};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::thread::sleep;
use std::time::Duration;

use llama_arm::{LlamaApi, greedy_decode, lib_dir_from_env, load, tokenize};
use serde_json::{Value, json};

const MODEL: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
const LONG_PROMPT_FILE: &str = "/Users/brianbruggeman/.cache/kvgate/k12/long_prompt.txt";
const OUT_DIR: &str = "/Users/brianbruggeman/.cache/kvgate/proxnum52";
const SHORT_PROMPT: &str = "<|turn>user\nWhich of these is smaller in size: a hippopotamus or a large office building?<turn|>\n<|turn>model\n";
const N_PREDICT: usize = 128;
const KV_BUCKET: usize = 32;
const LLAMA_KV_PAD: usize = 256;
const CONTEXT_LENGTH: usize = 131_072;
const NOT_QUIET: &str = "not quiet";

fn setting(name: &str, default: &str) -> PathBuf {
    PathBuf::from(env::var(name).unwrap_or_else(|_| default.to_owned()))
}

fn capacity(prompt_ids: usize) -> usize {
    ((prompt_ids + N_PREDICT).div_ceil(KV_BUCKET) * KV_BUCKET).min(CONTEXT_LENGTH)
}

fn free_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|error| error.to_string())?;
    listener.local_addr().map(|address| address.port()).map_err(|error| error.to_string())
}

fn curl(args: &[&str]) -> Result<String, String> {
    let output = Command::new("curl").arg("-s").args(args).output().map_err(|error| error.to_string())?;
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

struct Server {
    child: Child,
    port: u16,
    command_line: String,
}

fn spawn_server(dir: &Path, model: &Path, capacity: usize, log: &Path) -> Result<Server, String> {
    let port = free_port()?;
    let binary = dir.join("llama-server");
    let arguments = [
        "-m", &model.display().to_string(), "-ngl", "99", "-fa", "on", "-np", "1", "--no-warmup", "--seed", "1",
        "-c", &capacity.to_string(), "-ctk", "f16", "-ctv", "f16", "--host", "127.0.0.1", "--port", &port.to_string(), "-lv", "4",
    ];
    let command_line = format!("{} {}", binary.display(), arguments.join(" "));
    let stderr = File::create(log).map_err(|error| error.to_string())?;
    let child = Command::new(&binary)
        .args(arguments)
        .stdout(Stdio::null())
        .stderr(stderr)
        .spawn()
        .map_err(|error| error.to_string())?;
    Ok(Server { child, port, command_line })
}

fn wait_ready(server: &Server) -> bool {
    let url = format!("http://127.0.0.1:{}/health", server.port);
    (0..240).any(|_| {
        sleep(Duration::from_secs(1));
        curl(&["-o", "/dev/null", "-w", "%{http_code}", &url]).is_ok_and(|code| code == "200")
    })
}

fn completion(server: &Server, ids: &[i32]) -> Result<(Vec<i32>, f64), String> {
    let body = json!({
        "prompt": ids, "n_predict": N_PREDICT, "temperature": 0, "top_k": 1, "seed": 1,
        "cache_prompt": false, "ignore_eos": true, "return_tokens": true,
    })
    .to_string();
    let url = format!("http://127.0.0.1:{}/completion", server.port);
    let text = curl(&["--max-time", "300", "-X", "POST", "-H", "Content-Type: application/json", "-d", &body, &url])?;
    let value: Value = serde_json::from_str(&text).map_err(|error| format!("{error}: {text}"))?;
    let tokens = value.get("tokens").and_then(Value::as_array).ok_or("response has no tokens")?;
    let ids = tokens.iter().filter_map(|id| id.as_i64().and_then(|id| i32::try_from(id).ok())).collect();
    let per_token = value.pointer("/timings/predicted_per_token_ms").and_then(Value::as_f64).ok_or("no predicted_per_token_ms")?;
    Ok((ids, per_token))
}

fn first_divergence(left: &[i32], right: &[i32]) -> Option<usize> {
    let shared = left.len().min(right.len());
    (0..shared).find(|&index| left[index] != right[index]).or((left.len() != right.len()).then_some(shared))
}

fn mean_ms(step_ns: &[u64]) -> f64 {
    let decode = &step_ns[1.min(step_ns.len())..];
    let count = u64::try_from(decode.len().max(1)).unwrap_or(1);
    let mean_ns = decode.iter().sum::<u64>() / count;
    f64::from(u32::try_from(mean_ns).unwrap_or(u32::MAX)) / 1e6
}

fn compare(name: &str, api: &LlamaApi, dir: &Path, model: &Path, text: &str, out: &Path) -> Result<(), String> {
    let ids = tokenize(api, model, text).map_err(|error| error.to_string())?;
    let capacity = capacity(ids.len());
    let n_ctx = capacity.div_ceil(LLAMA_KV_PAD) * LLAMA_KV_PAD;
    let (own, step_ns, top2) = greedy_decode(api, model, &ids, u32::try_from(n_ctx).map_err(|e| e.to_string())?, N_PREDICT)
        .map_err(|error| error.to_string())?;
    let mut server = spawn_server(dir, model, capacity, &out.join(format!("server-{name}.err")))?;
    println!("server command: {}", server.command_line);
    let asked = wait_ready(&server).then(|| completion(&server, &ids).and_then(|cold| completion(&server, &ids).map(|warm| (cold, warm))));
    server.child.kill().map_err(|error| error.to_string())?;
    server.child.wait().map_err(|error| error.to_string())?;
    let ((theirs, _), (warm_ids, per_token)) = asked.ok_or("llama-server never became healthy")??;
    fs::write(out.join(format!("ids-{name}.txt")), format!("prompt {ids:?}\nin_process {own:?}\nserver_cold {theirs:?}\nserver_warm {warm_ids:?}\n"))
        .map_err(|error| error.to_string())?;
    let divergence = first_divergence(&own, &theirs);
    println!(
        "{name}: prompt_ids={} capacity={capacity} n_ctx={n_ctx} in_process_ids={} server_ids={} first_divergence={} server_cold_eq_warm={}",
        ids.len(), own.len(), theirs.len(), divergence.map_or("none".to_owned(), |index| index.to_string()), theirs == warm_ids,
    );
    println!(
        "{name}: in_process mean decode {:.3} ms/token ({NOT_QUIET}) | llama-server predicted_per_token_ms {per_token:.3} ({NOT_QUIET})",
        mean_ms(&step_ns),
    );
    if let Some(index) = divergence {
        println!("{name}: DIVERGED at {index}: in_process={:?} server={:?} in_process_top2={:?}", own.get(index), theirs.get(index), top2.get(index));
    }
    Ok(())
}

fn main() -> ExitCode {
    let model = setting("AA_MODEL", MODEL);
    let long_text = match fs::read_to_string(setting("AA_LONG_PROMPT", LONG_PROMPT_FILE)) {
        Ok(text) => text.trim_end_matches('\n').to_owned(),
        Err(error) => return fail(&error.to_string()),
    };
    let out = setting("AA_OUT", OUT_DIR);
    let dir = lib_dir_from_env();
    let lib = match load(&dir) {
        Ok(lib) => lib,
        Err(error) => return fail(&error.to_string()),
    };
    println!("gate: sha256={} build={}", lib.report.sha256, lib.report.build_number);
    for (name, text) in [("short", SHORT_PROMPT.to_owned()), ("long", long_text)] {
        if let Err(message) = compare(name, &lib.api, &dir, &model, &text, &out) {
            return fail(&format!("{name}: {message}"));
        }
    }
    ExitCode::SUCCESS
}

fn fail(message: &str) -> ExitCode {
    eprintln!("aa_ids failed: {message}");
    ExitCode::FAILURE
}
