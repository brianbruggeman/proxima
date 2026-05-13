# nginx-style — static + cache + fallback in one config

What it composes:

| Layer | Primitive | Purpose |
|---|---|---|
| 1 | `kv:cache` upstream | cache; first miss, subsequent hits |
| 2 | `fs` upstream | static file root |
| 3 | `synth` upstream | placeholder fallback (swap for `http = "..."` for a real backend) |
| select | `fallthrough` with `miss_on = ["no_data", "404"]` | kv miss or file-not-found advances |
| write_back | `[["fallback", "cache"]]` | populate cache from fallback responses |

Zero glue code. Replace the synth with `http = "http://backend:9001"` and a multi-upstream LB pool to get a full reverse proxy + LB + static + cache.

## run

```bash
mkdir -p /var/www/static
echo "<h1>hello</h1>" > /var/www/static/index.html
proxima serve --config proxima.toml --mount '/{*path}' --addr 127.0.0.1:8080
```

## verify

```bash
curl http://127.0.0.1:8080/index.html      # static file
curl http://127.0.0.1:8080/api/anything    # falls through to fallback
curl http://127.0.0.1:8080/api/anything    # cache hit (same response)
```

## test

```bash
cargo test --test example_nginx_style_smoke
```

The test substitutes a tempdir for `/var/www/static`, drives `Pipe::call` directly via the library API, asserts each layer fires correctly.

## not yet

What's *not* in this example (and what's blocking it):

| Want | Blocker |
|---|---|
| Multiple `[[listen]]` blocks (HTTP + HTTPS + admin port) | spec format extension — `proxima serve` reads only one bind via `--addr` |
| Host-based routing (`server_name api.example.com`) | mount router is path-only |
| Multiple named pipes in one TOML | spec extension — `[[pipe]]` block parsing |
| Per-location middleware overrides | mount router doesn't carry middleware overrides |
| Stream listeners (TCP/Unix) via `proxima serve` | `StreamListenerProtocol` not default-registered + CLI is HTTP-only |

These are spec / CLI ergonomics gaps, not new primitives. Tracked in `parking-lot.md`.
