# 01 — hello world REST

Returns `hello world\n` on any path/method. Pure config. The
`synth` upstream returns a fixed response; the HTTP listener does
the bind.

## run

```bash
proxima serve \
  --config examples/config/01-hello-rest/proxima.toml \
  --mount '/{*path}' \
  --addr 127.0.0.1:8080
```

## test

```bash
curl http://127.0.0.1:8080/anything
# hello world
```

## what's composed

- `synth` upstream (built-in)
- HTTP listener (CLI default protocol when `--addr` is given)
- wildcard mount `/{*path}` (catches every request)

Zero new code, zero application logic.
