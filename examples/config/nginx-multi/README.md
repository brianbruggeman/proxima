# nginx-multi — full nginx-style multi-listener config

Multiple listeners, named pipes, host-based routing, location-based
mounts, method filters, LB pools. All in one TOML file. Zero glue code.

## structure

```
[[pipe]] api                 # round-robin LB across be1/be2/be3
[[pipe]] static              # fs root
[[pipe]] admin               # synth (placeholder)

[[listen]] :8080 (HTTP)
  [[listen.mount]] /api/{*path}     → api      host=[api.example.com, localhost]
  [[listen.mount]] /static/{*path}  → static
  [[listen.mount]] /{*path}         → static   (default)

[[listen]] :8081 (HTTP)             # admin port
  [[listen.mount]] /{*path}         → admin    methods=[GET]
```

Each `[[listen]]` gets its own router — mounts on listener A do **not**
leak to listener B. Same nginx semantics: an `admin` endpoint exposed
only on the admin port.

## run

```bash
mkdir -p /var/www/static
echo "<h1>STATIC HOME</h1>" > /var/www/static/index.html
proxima serve --config proxima.toml
```

`proxima serve` detects the `[[listen]]` blocks and binds all of them.
No `--addr`, no `--mount`. The CLI prints `READY <addr>` per listener.

## test it

```bash
# default static
curl http://127.0.0.1:8080/index.html
# -> <h1>STATIC HOME</h1>

# api LB pool — Host header required
curl -H "Host: localhost" http://127.0.0.1:8080/api/v1
# -> from-be1
curl -H "Host: localhost" http://127.0.0.1:8080/api/v1
# -> from-be2
curl -H "Host: localhost" http://127.0.0.1:8080/api/v1
# -> from-be3 (round-robin)

# api without right Host → host filter rejects, falls to static, 404
curl -i http://127.0.0.1:8080/api/v1
# -> 404

# admin port, separate router
curl http://127.0.0.1:8081/admin
# -> admin endpoint

# POST rejected by method filter
curl -i -X POST http://127.0.0.1:8081/admin
# -> 404

# admin endpoint NOT visible on api port
curl -i http://127.0.0.1:8080/admin
# -> 404 (per-listener router isolation)
```

## what's in the substrate that makes this work

| nginx concept | proxima primitive |
|---|---|
| `listen 80;` | `[[listen]] type = "http" bind = "..."` |
| `server { server_name api.example.com; }` | `[[listen.mount]] host = ["api.example.com"]` |
| `location /api/ { proxy_pass http://backend; }` | `[[listen.mount]] path = "/api/{*path}" pipe = "api"` |
| `upstream backend { server be1; server be2; server be3; }` | `[[pipe.upstreams]]` array + `select.algorithm = "round_robin"` |
| `root /var/www/static;` | `fs = { root = "..." }` upstream |
| `limit_except GET { deny all; }` | `methods = ["GET"]` per mount |

zero new traits. zero new primitives. just spec parsing + per-listener
router (`App::load_full`) + the `fs` upstream.

## test

```bash
cargo test --test example_nginx_multi_smoke
```
