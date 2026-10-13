# proxima-cli

Command-line interface for proxima. The `proxima` binary loads a pipe config and serves it over HTTP, emits registered schemas (json-schema, openapi, toml), talks to a running daemon over its UDS control plane, hot-swaps a running pipe to a new spec, walks a recorded causal graph backward from an output byte, and submits and inspects pipelines on `proximad`. A `proximad` binary is built alongside it.

## What it composes

`proxima-lib` (config loading, scenarios, the app), `proxima-telemetry`, `proxima-tensor`, `proxima-safetensors`, `proxima-model-interop`.

Run `proxima --help` for the full subcommand list.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
