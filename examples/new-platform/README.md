# proxima-example-new-platform

Capstone example: bringing up a new target is not a runtime switch. `build.rs` reads a TOML profile named by `PROXIMA_PROFILE` and bakes the answer into `pub const`s through `proxima-build`; there is no runtime branch on a config value, so whichever profile built the binary is the only one it knows.

`PROXIMA_PROFILE` names a profile TOML under the `profiles/` directory of the repository (`linux-daemon`, `bare-metal`, `wasm-edge`, ...); unset, it falls back to the default profile. Run it once per profile from a repository checkout and diff the output:

```text
PROXIMA_PROFILE=linux-daemon cargo run -p proxima-example-new-platform
PROXIMA_PROFILE=bare-metal   cargo run -p proxima-example-new-platform
```

## What it composes

`proxima-build` (build dependency).

## License

Licensed under MIT OR Apache-2.0, at your option.

Repository: <https://github.com/brianbruggeman/proxima>
