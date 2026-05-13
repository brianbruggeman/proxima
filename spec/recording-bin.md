# `proxima.recording.bin` — binary recording layout

Same event vocabulary as `recording-jsonl.schema.json`. Use this when
storage matters or random-access seek is needed.

## file pair

| file | purpose |
| --- | --- |
| `<name>.bin` | event frames in append order, crash-safe per frame |
| `<name>.bin.idx` | fixed-size index records, one per frame; O(log n) seek by `ts_ms`, O(n) demux by interaction id |

The writer flushes both files. Readers tolerate a short `.idx` (writer
crashed mid-append): index entries past `.bin` are ignored; data past
the index is scannable but unindexed.

## frame format (`.bin`)

```
[u32 frame_len_le]
[zstd-compressed bytes ... length = frame_len_le]
```

Decompress → a postcard-encoded `BinEnvelope`:

```
BinEnvelope = struct {
    version: u32,         // 1; readers must reject unknown values
    event:   BinEvent,    // see below
}
```

`BinEvent` is a postcard enum with the same six variants as jsonl:
`Start`, `RequestChunk`, `RequestEnd`, `ResponseStart`, `ResponseChunk`,
`InteractionEnd`. Two encoding wrinkles:

- `Start.ts` is split into `(ts_unix_nanos_lo: u64, ts_unix_nanos_hi: u64, ts_negative: bool)` — postcard has no native `i128`.
- `meta.extra` is `extra_json: Option<String>` — JSON-encoded only when
  the recorder produced a non-empty map.

Per-frame zstd (one zstd frame = one event) is deliberate: a corrupt
frame fails locally without poisoning the file. Default level 3,
override via `BinSink::create_with_level`.

## index format (`.bin.idx`)

Fixed 36-byte records, little-endian, packed back-to-back:

```
[u64 entry_offset]    byte offset of the frame in .bin
[u64 ts_ms]           event ts_ms (0 for InteractionStart)
[u32 frame_len]       zstd payload length (excludes the u32 prefix in .bin)
[u8;16 ulid_bytes]    InteractionId
```

`ts_ms` is monotonically non-decreasing **per interaction** but the
global order may interleave at millisecond resolution.

## reader contract

- Read `[u32 frame_len]` then `frame_len` bytes, feed through
  `zstd::stream::decode_all` and `postcard::from_bytes::<BinEnvelope>`.
- Reject `version != 1` with `ProximaError::Record`. Never produce a
  partial event stream.
- Seek-by-timestamp: binary-search `.idx` for the first
  `ts_ms >= target`, dereference its `entry_offset`, resume in `.bin`.

## versioning

Breaking changes bump `BinEnvelope.version` and ship a converter
(`proxima recording convert --from-version=N`). Readers refuse unknown
versions; never silently downgrade.
