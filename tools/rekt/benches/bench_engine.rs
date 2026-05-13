// custom-harness bench (no criterion). attributes the per-send cost across the
// proxima dispatch layers, in-process synth (no network) on one prime core.
//
// FINDING (corrected): the dyn-future box is NOT the cost. the cost is a per-send
// `serde_json::Value` clone inside `Client::handle()`. the arms peel it apart:
//
//   N null backend `SendPipe::call(&Arc<NullPipe>, req)` — returns a free
//     `Response::new(200)`. rekt's TRUE floor: request build + dispatch only.
//     A-N is the synth's per-call cost (it allocs a `content-length` header per
//     call), NOT rekt's — the synth is not the free backend it looks like.
//   A monomorphic  `SendPipe::call(&Arc<SynthUpstream>, req)` — concrete pipe,
//     monomorphized `impl Future`, zero box.
//   B boxed handle `SendPipe::call(&Arc<dyn DynPipe>, req)` via `into_handle` —
//     adds ONLY `Box::pin` + vtable. measures ~= A: the box is free.
//   D Client-as-Pipe `SendPipe::call(&Client, req)` (prebuilt req, no sugar) —
//     adds the full `Client::dispatch`. ~2x A.
//   E Client via `from_handle` — same dispatch as D but the `injected` handle
//     short-circuits `Client::handle()`, so the per-send `spec.clone()` never
//     runs. lands back near B: the D-E gap IS that spec clone.
//   C `Client.call(method,path).send()` — D plus the builder-sugar String allocs.
//
// proxima fixes this bench argues for (both in src/client/handle.rs):
//   1. peek the cached handle before cloning the spec — `if let Some(h) =
//      self.inner.handle.get() { return Ok(h.clone()) }` ahead of the per-send
//      `spec.clone()`. makes the `from_value`/`http` path behave like arm E
//      (~1.1x A) instead of arm D (~2x A). this is the 204 ns line item.
//   2. the cached handle is a `tokio::sync::OnceCell` in the DEFAULT build (the
//      `sync-wrappers`-gated `crate::sync::OnceCell` is off by default) — a tokio
//      primitive in the prime hot path. proxima-sync already ships the tokio-free
//      `async_lock::OnceCell` drop-in built for exactly this site; the field
//      should use it unconditionally. `.get()` is present on both, so (1) lands
//      either way.
// the runtime is booted once per arm with a warm-up so registry/runtime init are
// not charged to the measured sends.
//
// a second table drives the same sends on BOTH proxima runtimes — prime
// (`run`) and tokio (`run_tokio`, current-thread) — to confirm the
// per-send cost is the request build + synth backend, not the executor. at
// concurrency-1 in-process the two runtimes are within noise; the Client arm
// uses `"wire":"tokio"` on tokio so dispatch stays inline instead of hopping
// back to prime via `call_on_worker`.

#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::time::Instant;

use std::future::Future;

use proxima::ProximaError;
use proxima::SendPipe;
use proxima::client::Client;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::{run, run_tokio};
use proxima::upstreams::SynthUpstream;
use serde_json::json;

// a truly-free backend: returns `Response::new(200)` (empty HeaderList, empty
// Bytes — no per-call alloc). isolates rekt's own floor (request build +
// dispatch) from the synth, which allocates per call (`content-length`
// `to_string()` + header copy + body/header clones).
struct NullPipe;

impl SendPipe for NullPipe {
    type In = Request<bytes::Bytes>;
    type Out = Response<bytes::Bytes>;
    type Err = ProximaError;

    fn call(&self, _input: Request<bytes::Bytes>) -> impl Future<Output = Result<Response<bytes::Bytes>, ProximaError>> + Send {
        let response = Response::new(200);
        async move { Ok(response) }
    }
}

const RUNS: usize = 5;
const COUNT: u64 = 50_000;
const WARMUP: u64 = 2_000;

fn mean_cov(samples: &[f64]) -> (f64, f64) {
    let count = samples.len();
    if count <= 1 {
        return (samples.first().copied().unwrap_or(0.0), 0.0);
    }
    let mean = samples.iter().sum::<f64>() / count as f64;
    let var = samples
        .iter()
        .map(|s| (s - mean).powi(2))
        .sum::<f64>()
        / (count as f64 - 1.0);
    let cov = if mean > 0.0 { var.sqrt() / mean * 100.0 } else { 0.0 };
    (mean, cov)
}

// `Bytes::from_static` over `&str` so building a request does NOT heap-copy the
// method/path (the `&str` IntoHeaderBytes impl does `Bytes::copy_from_slice`).
// `IntoHeaderBytes for Bytes` is identity, so this is zero-alloc.
fn request() -> Request<bytes::Bytes> {
    Request::builder()
        .method(bytes::Bytes::from_static(b"GET"))
        .path(bytes::Bytes::from_static(b"/"))
        .build()
        .expect("build request")
}

// concrete null backend: request build + dispatch + a free response. rekt's
// true per-send floor, with the synth's per-call allocs removed.
fn bench_null() -> (f64, f64) {
    let pipe = Arc::new(NullPipe);
    run(async move {
        for _ in 0..WARMUP {
            let _ = SendPipe::call(&pipe, request()).await;
        }
        let mut samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            for _ in 0..COUNT {
                let _ = SendPipe::call(&pipe, request()).await;
            }
            samples.push(start.elapsed().as_secs_f64() * 1e9 / COUNT as f64);
        }
        mean_cov(&samples)
    })
    .expect("prime run")
}

// null backend, but build ONE request and clone it per send (the new buffered
// `Request: Clone`). vs `bench_null` (fresh build per send), the delta is
// request build cost minus clone cost — the build-once/clone-per-send lever.
fn bench_null_reuse() -> (f64, f64) {
    let pipe = Arc::new(NullPipe);
    let template = request();
    run(async move {
        for _ in 0..WARMUP {
            let _ = SendPipe::call(&pipe, template.clone()).await;
        }
        let mut samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            for _ in 0..COUNT {
                let _ = SendPipe::call(&pipe, template.clone()).await;
            }
            samples.push(start.elapsed().as_secs_f64() * 1e9 / COUNT as f64);
        }
        mean_cov(&samples)
    })
    .expect("prime run")
}

// monomorphic: concrete `SynthUpstream`, unboxed `SendPipe::call`.
fn bench_monomorphic() -> (f64, f64) {
    let pipe = Arc::new(SynthUpstream::new("synth", 200, "ok".to_string()));
    run(async move {
        for _ in 0..WARMUP {
            let _ = SendPipe::call(&pipe, request()).await;
        }
        let mut samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            for _ in 0..COUNT {
                let _ = SendPipe::call(&pipe, request()).await;
            }
            samples.push(start.elapsed().as_secs_f64() * 1e9 / COUNT as f64);
        }
        mean_cov(&samples)
    })
    .expect("prime run")
}

// boxed handle, NO Client: `Arc<dyn DynPipe>` directly. isolates JUST the dyn
// future box + vtable hop — the same `into_handle(synth)` the `Client` wraps,
// but called without RequestBuilder/clone/OnceCell/dispatch on top.
fn bench_boxed_handle() -> (f64, f64) {
    let handle: PipeHandle = into_handle(SynthUpstream::new("synth", 200, "ok".to_string()));
    run(async move {
        for _ in 0..WARMUP {
            let _ = SendPipe::call(&handle, request()).await;
        }
        let mut samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            for _ in 0..COUNT {
                let _ = SendPipe::call(&handle, request()).await;
            }
            samples.push(start.elapsed().as_secs_f64() * 1e9 / COUNT as f64);
        }
        mean_cov(&samples)
    })
    .expect("prime run")
}

// Client AS A PIPE: hold `Arc<Client>` and call `SendPipe::call(&client, req)`
// with a PREBUILT request — no `call(method,path).send()` builder sugar, so no
// per-send String allocs / RequestBuilder. isolates dispatch cost from sugar.
fn bench_client_as_pipe() -> (f64, f64) {
    let client = Arc::new(Client::from_value(json!({ "synth": { "status": 200, "body": "ok" } })).expect("synth client"));
    run(async move {
        for _ in 0..WARMUP {
            let _ = SendPipe::call(&client, request()).await;
        }
        let mut samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            for _ in 0..COUNT {
                let _ = SendPipe::call(&client, request()).await;
            }
            samples.push(start.elapsed().as_secs_f64() * 1e9 / COUNT as f64);
        }
        mean_cov(&samples)
    })
    .expect("prime run")
}

// Client built via `from_handle` (injected handle) called as a Pipe. `handle()`
// returns at the `injected` short-circuit, so the per-send `spec.clone()` +
// `factories.clone()` NEVER run. if this lands back near arm B, the whole D-B
// gap is that per-call spec clone inside `Client::handle()`.
fn bench_client_injected() -> (f64, f64) {
    let client = Arc::new(Client::from_handle(into_handle(SynthUpstream::new("synth", 200, "ok".to_string()))));
    run(async move {
        for _ in 0..WARMUP {
            let _ = SendPipe::call(&client, request()).await;
        }
        let mut samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            for _ in 0..COUNT {
                let _ = SendPipe::call(&client, request()).await;
            }
            samples.push(start.elapsed().as_secs_f64() * 1e9 / COUNT as f64);
        }
        mean_cov(&samples)
    })
    .expect("prime run")
}

// type-erased: `Client` over an `Arc<dyn DynPipe>`, one boxed future per send.
fn bench_type_erased() -> (f64, f64) {
    let client = Client::from_value(json!({ "synth": { "status": 200, "body": "ok" } })).expect("synth client");
    run(async move {
        for _ in 0..WARMUP {
            let _ = client.call("GET", "/").send().await;
        }
        let mut samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            for _ in 0..COUNT {
                let _ = client.call("GET", "/").send().await;
            }
            samples.push(start.elapsed().as_secs_f64() * 1e9 / COUNT as f64);
        }
        mean_cov(&samples)
    })
    .expect("prime run")
}

// ---- tokio-runtime arms: same sends, driven on proxima's tokio wrapper
// (`run_tokio`, current-thread) instead of the prime runtime. the concrete
// + boxed arms route identically on either runtime; the Client arm needs
// `"wire":"tokio"` so dispatch takes the tokio inline branch instead of hopping
// back to prime via `call_on_worker`.

async fn timed_loop<Fut, MakeFut>(mut make: MakeFut) -> (f64, f64)
where
    MakeFut: FnMut() -> Fut,
    Fut: std::future::Future,
{
    for _ in 0..WARMUP {
        let _ = make().await;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        for _ in 0..COUNT {
            let _ = make().await;
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / COUNT as f64);
    }
    mean_cov(&samples)
}

#[allow(clippy::disallowed_methods)] // bench deliberately names the tokio backend to compare it against prime
fn tokio_monomorphic() -> (f64, f64) {
    let pipe = Arc::new(SynthUpstream::new("synth", 200, "ok".to_string()));
    run_tokio(false, None, timed_loop(|| SendPipe::call(&pipe, request()))).expect("tokio run")
}

#[allow(clippy::disallowed_methods)] // bench deliberately names the tokio backend to compare it against prime
fn tokio_boxed_handle() -> (f64, f64) {
    let handle: PipeHandle = into_handle(SynthUpstream::new("synth", 200, "ok".to_string()));
    run_tokio(false, None, timed_loop(|| SendPipe::call(&handle, request()))).expect("tokio run")
}

#[allow(clippy::disallowed_methods)] // bench deliberately names the tokio backend to compare it against prime
fn tokio_client_as_pipe() -> (f64, f64) {
    let client = Arc::new(Client::from_value(json!({ "synth": { "status": 200, "body": "ok" }, "wire": "tokio" })).expect("synth client"));
    run_tokio(false, None, timed_loop(|| SendPipe::call(&client, request()))).expect("tokio run")
}

fn main() {
    println!("per-send cost vs in-process synth, sequential on one prime core ({RUNS} runs x {COUNT})\n");

    let (null_ns, null_cov) = bench_null();
    let (reuse_ns, reuse_cov) = bench_null_reuse();
    let (mono_ns, mono_cov) = bench_monomorphic();
    let (boxed_ns, boxed_cov) = bench_boxed_handle();
    let (pipe_ns, pipe_cov) = bench_client_as_pipe();
    let (inj_ns, inj_cov) = bench_client_injected();
    let (erased_ns, erased_cov) = bench_type_erased();

    println!("  N null backend  SendPipe::call (Arc<NullPipe>, free resp)    {null_ns:8.1} ns/send  (cov {null_cov:.1}%)");
    println!("  R null + reuse  (build once, Request::clone per send)        {reuse_ns:8.1} ns/send  (cov {reuse_cov:.1}%)");
    println!("  A monomorphic SendPipe::call (Arc<SynthUpstream>, zero box)  {mono_ns:8.1} ns/send  (cov {mono_cov:.1}%)");
    println!("  B boxed handle  SendPipe::call (Arc<dyn DynPipe>, one box)    {boxed_ns:8.1} ns/send  (cov {boxed_cov:.1}%)");
    println!("  D Client-as-Pipe SendPipe::call(&Client, prebuilt req)       {pipe_ns:8.1} ns/send  (cov {pipe_cov:.1}%)");
    println!("  E Client from_handle (injected, skips per-send spec clone)   {inj_ns:8.1} ns/send  (cov {inj_cov:.1}%)");
    println!("  C Client.call(m,p).send()  (full builder sugar)              {erased_ns:8.1} ns/send  (cov {erased_cov:.1}%)");
    if mono_ns > 0.0 {
        println!("\n  N-R request build saved by clone-reuse  : {:8.0} ns/send", null_ns - reuse_ns);
        println!("  A-N synth backend per-call cost (alloc) : {:8.0} ns/send", mono_ns - null_ns);
        println!("  B-A dyn-future box + vtable             : {:8.0} ns/send", boxed_ns - mono_ns);
        println!("  E-B Client dispatch sans spec clone     : {:8.0} ns/send", inj_ns - boxed_ns);
        println!("  D-E per-send spec clone (now ~0; was 204) : {:8.0} ns/send  <- regression guard", pipe_ns - inj_ns);
        println!("  C-D builder sugar (String allocs)       : {:8.0} ns/send", erased_ns - pipe_ns);
        println!("  C/A total vs concrete : {:.2}x   E/A injected-Client vs concrete : {:.2}x", erased_ns / mono_ns, inj_ns / mono_ns);
    }

    println!("\nprime vs tokio runtime (proxima::runtime::run_tokio, current-thread; same sends)");
    let (tmono_ns, tmono_cov) = tokio_monomorphic();
    let (tboxed_ns, tboxed_cov) = tokio_boxed_handle();
    let (tpipe_ns, tpipe_cov) = tokio_client_as_pipe();
    println!("                                                  prime          tokio");
    println!("  A concrete SendPipe::call          {mono_ns:8.1}      {tmono_ns:8.1} ns/send  (cov {tmono_cov:.1}%)");
    println!("  B boxed handle  Arc<dyn DynPipe>    {boxed_ns:8.1}      {tboxed_ns:8.1} ns/send  (cov {tboxed_cov:.1}%)");
    println!("  D Client-as-Pipe (wire-native)     {pipe_ns:8.1}      {tpipe_ns:8.1} ns/send  (cov {tpipe_cov:.1}%)");
}
