# replay — serve captured traffic back, byte-identical

## Builds on

[record](../record/README.md) — replay serves back what record captured.

## What it demonstrates

`record` tees live traffic into a cassette: a JSONL log of `RecordingEvent`s
(`Started` → request chunk(s) → `RequestEnded` → `ResponseStarted` → response
chunk(s) → `Ended`). `ReplayUpstream` reads that cassette and becomes a
`SendPipe` in its own right — no upstream call happens; a matching request
gets back the exact recorded status, headers, and response chunks, in the
exact framing they were captured with.

Matching is by method + path + sorted query (`MatchSpec::include_body` adds a
body digest when two same-path requests need to resolve to different
recordings, e.g. an LLM POST). A request that doesn't match anything in the
cassette is a typed `ProximaError::ReplayMiss`, not a wrong-body 200 — replay
never guesses.

This is the basis for deterministic tests and fakes: front a flaky or costly
upstream with a cassette recorded once, and every later run gets the same
bytes back, with no network call and no timing variance.

## Run

```
cargo run --example replay
```

## What you'll see

```
recorded 1 interaction (1157 bytes) to /tmp/.../chat.jsonl
loaded cassette, known match keys: ["POST /v1/chat/completions?model=gpt-mini"]
replayed status 200 and 2 headers, byte-identical to what was recorded
  chunk 0: "{\"delta\":\"Hel\"}"
  chunk 1: "{\"delta\":\"lo, \"}"
  chunk 2: "{\"delta\":\"world\"}"
unrecorded request correctly missed: GET /v1/never-recorded?
```

The example builds one recorded interaction — a streamed 3-chunk chat
response — the same `RecordingEvent` shape `record`'s sink writes, and writes
it to a tmp cassette without running `record` at all: `JsonFormat::new().encode_block(events)`
turns the hand-built events straight into cassette bytes, the same codec
`record`'s own JSONL sink calls internally. `ReplayUpstream::from_jsonl` loads it, and a matching
`POST /v1/chat/completions?model=gpt-mini` request gets back status 200, both
headers, and all three response chunks — compared not just as concatenated
bytes but chunk-for-chunk against what went into the cassette, so the
assertion proves framing survives the round trip, not only the total
payload. A second, unrecorded request (`GET /v1/never-recorded`) proves the
flip side: no match means a typed `ReplayMiss`, never a fabricated response.
