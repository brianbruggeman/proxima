# AlertEvent documented JSON shape

**Relocation + API note (2026-07, grep-verified):** `AlertEvent` now lives at
`proxima-patterns/src/alert/event.rs` (crate `proxima-patterns`, features
`alert,proto`; see `docs/proxima-notify/discipline.md`'s top note for the
full path mapping). There is **no** `AlertEvent::as_json_shape()` method —
the mapping this document describes is implemented as free functions in the
`json_shape` module (feature `json-shape`): `alert_event_to_json(&AlertEvent)
-> serde_json::Value`, plus `guidance_question_to_json` /
`guidance_answer_to_json`. `AlertId` is `AlertId(pub Ulid)` (a `ulid::Ulid`
newtype), not `AlertId(u128)` as stated below. There is no
`AlertEvent::builder()` / `AlertEventBuilder` / `BuildError::CapExceeded` —
`AlertEvent` is constructed via a plain struct literal (see
`proxima-patterns/examples/alert_walkthrough.rs`); a caps-exceeded field
returns an error from the underlying `heapless` container's `try_from`/
`insert`, not a dedicated builder error type. The fixture paths named below
(`tests/fixtures/alert_events/heartbeat_at_caps.json`/`.postcard`) do not
exist — `tests/fixtures/alert_events/` contains only a `.gitkeep`; the parity
assertions instead live inline in `proxima-patterns/tests/event_roundtrip.rs`.

This is the schema that the documented `AlertEvent::as_json_shape()` helper maps to. It exists so:

- C1's parity test asserts `AlertEvent → AlertEventJsonShape → serde_json::Value` matches a known `serde_json::json!{...}` fixture, even though postcard bytes are not serde_json bytes.
- Consumers reading alerts off `ProtocolEvent::Custom { kind: "alert", payload }` know the field layout.
- The tier-3 heapless fields have an externally-documented contract.

This document is the contract; the helper is the implementation.

## Top-level shape

```json
{
  "id": "01HF3JM5N6P7Q8R9STVWXYZA0B",
  "severity": "warn",
  "kind": "heartbeat",
  "labels": {
    "source": "scheduled_trigger",
    "host": "proxima-node-1"
  },
  "payload_bytes_base64": "BAACAAEAAAAAAAAAAA",
  "fired_at_micros": 1748284800000000
}
```

## Field semantics

### `id` — string, Crockford base32 (26 chars)

The `AlertId` ULID rendered as Crockford base32. Lexicographically sortable; encodes a millisecond-precision timestamp + 80 bits of entropy. Source: the tier-3 newtype `AlertId(u128)`.

### `severity` — string, one of `trace | debug | info | warn | error | fatal`

Lowercase. Maps 1:1 to the tier-3 `enum Severity` repr(u8) variant. Numeric values per `proxima-telemetry::Level`:
- `trace` = 1
- `debug` = 5
- `info`  = 9
- `warn`  = 13
- `error` = 17
- `fatal` = 21

### `kind` — string, ≤ LABEL_KEY_MAX bytes

Bounded UTF-8. The semantic type of the alert (`heartbeat`, `threshold_breach`, `door_open`, etc.). Sourced from the tier-3 `heapless::String<LABEL_KEY_MAX>` field. The cap (`LABEL_KEY_MAX`) is set in `proxima-notify-proto.toml`.

### `labels` — object<string, string>, ≤ LABELS_MAX entries

Bounded label map. Keys ≤ `LABEL_KEY_MAX` bytes, values ≤ `LABEL_VAL_MAX` bytes. JSON shape is a plain object; sorted by key on serialize for deterministic output. Sourced from the tier-3 `heapless::IndexMap<heapless::String<LABEL_KEY_MAX>, heapless::String<LABEL_VAL_MAX>, LABELS_MAX>`.

### `payload_bytes_base64` — string, base64 (URL-safe, no padding) of the opaque payload

The tier-3 `heapless::Vec<u8, PAYLOAD_MAX>` payload is base64-encoded for JSON transport. JSON consumers decode via standard base64. Cap: `PAYLOAD_MAX` from `proxima-notify-proto.toml`.

The payload bytes are opaque to the proto layer. By convention, they are postcard-encoded for in-band carriage of additional typed fields (caller's choice).

### `fired_at_micros` — integer, microseconds since UNIX epoch

Caller-provided per `WithoutTime` marker — the proto layer does not read the clock. Producers (e.g. `ScheduledTriggerPipe`) populate this at the moment of firing.

## Worked example (used by C1's parity test fixture)

Input AlertEvent (constructed via `AlertEvent::builder()`):
- `id` = `AlertId(0x01923B5C7E8F4A6D9E0F0123456789AB)` → Crockford `01J8XNRZMF99PSW3R14D2PF2DB`
- `severity` = `Severity::Warn`
- `kind` = `"heartbeat"`
- `labels` = `{ "source": "scheduled_trigger", "host": "proxima-node-1" }`
- `payload_bytes` = `[0x04, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]`
- `fired_at_micros` = `1748284800000000`

Expected `serde_json::Value` after `as_json_shape()`:

```json
{
  "id": "01J8XNRZMF99PSW3R14D2PF2DB",
  "severity": "warn",
  "kind": "heartbeat",
  "labels": {
    "host": "proxima-node-1",
    "source": "scheduled_trigger"
  },
  "payload_bytes_base64": "BAACAAEAAAAAAAAAAA",
  "fired_at_micros": 1748284800000000
}
```

(Labels sorted lexically on serialize: `host` before `source`.)

C1's parity test asserts:
1. Round-trip: `AlertEvent → postcard::encode → postcard::decode → AlertEvent` is byte-identical.
2. JSON shape: `AlertEvent → AlertEvent::as_json_shape()` matches the above `serde_json::json!{...}` fixture structurally.

The fixture lives at `tests/fixtures/alert_events/heartbeat_at_caps.json` and the postcard form at `tests/fixtures/alert_events/heartbeat_at_caps.postcard`.

## Failure modes documented at C1 seal time

- Caps exceeded (`kind` longer than `LABEL_KEY_MAX`, etc.) → `AlertEventBuilder::build()` returns `Err(BuildError::CapExceeded { field, max, actual })`.
- Postcard decode of corrupted bytes → `Err(DecodeError::PostcardCorrupt(_))`.
- JSON shape with unknown fields on deserialize → ignored (forward-compat).
- JSON shape with missing required fields → `Err(JsonShapeError::MissingField(name))`.

C1's discipline-log row Notes will reference this document by path.
