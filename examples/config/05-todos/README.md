# 05-todos

CRUD + LIST todo app, zero Rust code. The whole API is a config file.

## Composition

| Layer | Primitive | Role |
| --- | --- | --- |
| `[[schema]] name = "Todo"` | top-level shape | one source of truth, referenced + emittable via `describe` |
| `validate` middleware | `schema = "Todo"` | gates POST/PUT bodies; short-circuits on empty bodies for GET/DELETE |
| `kv = "file"` + `list_mode = true` | durable backend + collection-list dispatch | CRUD via `/todos/{id}`, LIST via `/todos` |
| Two `[[listen.mount]]` entries | path-pattern routing | `/todos/{id}` binds `id`; `/todos` binds nothing → list-mode |

## Run

```bash
proxima serve --config examples/config/05-todos/proxima.toml
```

## Exercise the API

```bash
# create
curl -X POST -d '{"title":"buy milk"}' http://127.0.0.1:8080/todos/abc
# → {"title":"buy milk"}

# read
curl http://127.0.0.1:8080/todos/abc
# → {"title":"buy milk"}

# update
curl -X PUT -d '{"title":"buy oat milk","done":false}' http://127.0.0.1:8080/todos/abc
# → {"title":"buy oat milk","done":false}

# list
curl http://127.0.0.1:8080/todos
# → [{"title":"buy oat milk","done":false}, ...]

# delete
curl -X DELETE http://127.0.0.1:8080/todos/abc -i
# → 204 No Content

# reject missing required field
curl -X POST -d '{}' http://127.0.0.1:8080/todos/foo -i
# → 400 + {"error":"validation_failed","message":"missing required field `title`","path":"$"}
```

## Emit the OpenAPI spec for external consumers

```bash
proxima describe --config examples/config/05-todos/proxima.toml --format openapi
```
