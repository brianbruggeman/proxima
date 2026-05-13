# 04-hello-name

POST JSON, get a templated response. The request body is validated against a
named `Schema` before the synth upstream templates `{{body.name}}` into the
response.

## Composition

| Layer | Primitive | Role |
| --- | --- | --- |
| `[[schema]] name = "User"` | top-level shape declaration | one place; referenced by middleware and exported by `describe` |
| `validate` middleware | `schema = "User"` | rejects malformed bodies with 400 + structured error |
| `synth` upstream | `body_template = "hello, {{body.name}}\n"` | renders the response using the request body |

## Run

```bash
proxima serve --config examples/config/04-hello-name/proxima.toml
```

Then in another shell:

```bash
curl -X POST -d '{"name":"brian"}' http://127.0.0.1:8080/
# hello, brian

curl -X POST -d '{}' http://127.0.0.1:8080/ -i
# HTTP/1.1 400 Bad Request
# {"error":"validation_failed","message":"missing required field `name`","path":"$"}

curl -X POST -d '{"name":""}' http://127.0.0.1:8080/ -i
# HTTP/1.1 400 Bad Request
# {"error":"validation_failed","message":"string length 0 < min_len 1","path":"$.name"}
```

## Emit contracts for external consumers

```bash
proxima describe --config examples/config/04-hello-name/proxima.toml --format json-schema
proxima describe --config examples/config/04-hello-name/proxima.toml --format openapi --title hello-name --version 0.1.0
proxima describe --config examples/config/04-hello-name/proxima.toml --format toml
```

Same `Schema` IR, three published forms. JSON Schema 2020-12, OpenAPI 3.1, or
proxima's own self-describing TOML.
