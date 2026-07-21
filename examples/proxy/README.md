# proxy

Forward to an upstream.

## Builds on

[transform](../transform/README.md) — a pipe's `call` maps `In -> Out`. Here the map IS the
forward: `In` is the inbound request, `Out` is whatever the upstream returned.

## What it demonstrates

A reverse proxy is not special machinery bolted onto a listener — it's a `Pipe` whose transform
happens to be "hand this request to an upstream and return what comes back". `proxima::Client`
is itself a `SendPipe<In = Request<Bytes>, Out = Response<Bytes>>` (the same shape every pipe in
this curriculum has been), so `ProxyPipe`'s entire `call` body is one line:

```rust
#[piped(send)]
impl ProxyPipe {
    async fn call(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let client = self.client.clone();
        SendPipe::call(&client, request).await
    }
}
```

`ProxyPipe` holds state (`client: Client`), so it can't be the fieldless struct
`#[proxima::piped]`'s free-function form generates — this is that macro's other
form instead: a plain `impl ProxyPipe { .. }` block (no trait named) with one
method called `call`. `ProxyPipe` itself is unchanged, hand-written exactly as
before; what the macro removes is the trait header (`impl SendPipe for ProxyPipe
{ type In = ..; .. }`) and the `async move { .. }` wrapper a hand-written
`SendPipe::call` needs to turn a plain `Result` into a `Future` — the macro
reads `In`/`Out`/`Err` off `call`'s own signature and, because `call` here is
`async fn`, passes its relocated body straight through as the future.
`00-foundations.md` section 7 has the full before/after and the rule for when
this form applies vs. the free-function one.

No field-by-field copying, no header rewriting, no separate "proxy mode". The inbound
`Request<Bytes>` — method, path, headers, body, all of it — is handed straight to `Client`,
which resolves against the upstream base URL and issues the real HTTP call; the `Response<Bytes>`
it returns is handed straight back. Composition, not a special case.

| role | pipe | shape |
|---|---|---|
| origin | `origin_pipe` | `Request<Bytes> -> Response<Bytes>`, returns a fixed 201 + header + body |
| proxy | `ProxyPipe` | `Request<Bytes> -> Response<Bytes>`, forwards to `Client` and returns its answer |
| client | `Client` (`proxima::Client`) | itself a `SendPipe<Request<Bytes>, Response<Bytes>>` bound to the origin's URL |

The example stands up both as real listeners (`origin_pipe` on `127.0.0.1:8081`, `ProxyPipe` on
`127.0.0.1:8080`), sends one plain blocking `GET` to the proxy, and asserts the response that
comes back is the origin's status, header, and body — byte for byte, not approximated.

## Run

```
cargo run --example proxy
```

No extra features needed beyond the default set (`http-prime-deps` + the `runtime-prime-*`
quartet, same as `hello` and `distributed_trace`) — `Client::http` resolves the prime HTTP
backend those features register.

## What you'll see

```
origin listening on 127.0.0.1:8081
proxy  listening on 127.0.0.1:8080, forwards to 127.0.0.1:8081

client -> proxy raw response:
HTTP/1.1 201 Created
x-origin: proxima-origin
traceparent: 00-d87ad36b619fceba99584a1c087dd6c1-c7f80548846d4ced-01
content-length: 21

origin response body

PASS: forward-to-upstream is composition — the proxy pipe added no bytes, dropped none.
proxy  drained: cores_acked=1 hooks_drained=0
origin drained: cores_acked=1 hooks_drained=0
```

`origin_pipe` answers 201, not 200 — deliberately distinct from the default success status so the
proxy can't pass the assertions by accident with a hard-coded `Response::ok(...)`. The client
sees `201 Created`, the `x-origin` header, and the exact body `origin_pipe` wrote: proof the
pipe-to-pipe forward through `Client` carries status, headers, and body unchanged. The
`traceparent` header is added by the h1 listener's own request-context stamping (the same
mechanism `distributed_trace` exercises directly) — not something this example's code emits.
