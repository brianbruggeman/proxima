# hello

A service is a `Pipe` behind a listener, served until a signal. That's the
production shape at the smallest scale it can run — write the pipe, mount it,
serve it.

## Builds on

*(nothing — this is the entry point)*

## What it demonstrates

Every proxima service has the same three parts, no matter how big it gets:

1. a `SendPipe` — `Request -> Result<Response, Err>`, the thing that answers.
2. a `Runtime` — dispatches the pipe's calls (`App::new()` adopts whichever
   runtime `#[proxima::main]` already booted).
3. an `AcceptorFactory` — turns raw sockets into requests the pipe can read
   (paired with the adopted runtime; nothing here overrides either).

`#[proxima::main]` boots the runtime that drives `main`'s body across the
machine's cores; `App::new()` adopts that SAME runtime instead of building an
independent one. `.mount("/", hello)` hands the pipe
to the router. `app.serve(RunConfig::http(bind))` spawns the listener and
returns only once it is actually accepting — no polling, no sleeping, no
discovering `ECONNREFUSED` the hard way. `server.run_until_signal().await` then
blocks until SIGINT/SIGTERM, stops accepting, and lets in-flight requests
finish. That single line is the entire shutdown story.

`hello` is deliberately empty of anything but the contract: it reads nothing
from the request and always answers `hello, proxima\n`. That minimalism is the
point — every later example (`transform` onward) adds behavior to the pipe or
combinators around it, but the shape wired to the listener never changes.
`runtime-select` reuses this identical pipe and swaps only the `Runtime` +
`AcceptorFactory` pair underneath it.

## Run

```sh
cargo run --example hello
# in another shell:
curl http://127.0.0.1:8080/     # -> hello, proxima
# ctrl-c the server to drain and exit.
```

## What you'll see

The server prints its bind address and then serves until you stop it:

```
listening on http://127.0.0.1:8080
```

A `curl http://127.0.0.1:8080/` in another shell returns `hello, proxima` —
proof that what's listening is a real HTTP/1 server reached over the wire, not
an in-process call. Pressing ctrl-c (SIGINT) makes `run_until_signal` return:
the listener stops accepting, in-flight requests drain, and the process exits
`0`.
