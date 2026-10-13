# proxima-plugin-skeleton

Canonical plugin skeleton for proxima: copy the directory, rename the crate, and replace the example `StampHeader` middleware with your own pipe. The convention is that a plugin crate exposes `register(builder) -> Result<AppBuilder>` so users get one-line composition onto an `App` builder.

## What it composes

`proxima-primitives` (the `Pipe` the middleware implements), `bytes`, `serde_json`.

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
