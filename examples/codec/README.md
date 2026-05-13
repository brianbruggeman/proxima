# codec

**Builds on:** transform — a codec's decoded frame value is an ordinary
`Pipe` input; nothing extra is needed to compose it.

## The one concept

A codec is two functions implementing `proxima_codec::FrameCodec`:

```rust
pub trait FrameCodec: Send + Sync + 'static {
    type Frame<'a>;
    type Error: core::error::Error + Send + Sync + 'static;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(Self::Frame<'a>, usize), Self::Error>;
    fn encode_frame(&self, frame: &Self::Frame<'_>, dest: &mut Vec<u8>) -> Result<(), Self::Error>;
}
```

`parse_frame` borrows a frame out of a buffer and reports how many bytes it
consumed; `encode_frame` appends a frame's wire bytes to a caller-owned
`Vec<u8>`. No IO, no allocation on the decode path — the same shape
`LengthDelimitedCodec` (the H1/H2/H3 framer) already implements. Once a
type implements this trait, it composes into the pipe algebra like any
other codec: a decoded frame's payload is just an `In` for the next `Pipe`
— teaching proxima a new wire protocol is nothing more than this trait.

This example defines a new toy protocol from scratch, `kv-tlv`, to prove
the trait is the whole story — no hidden proxima-specific machinery beyond
`parse_frame` / `encode_frame`.

### Frame layout

| offset | bytes | field | meaning |
|---|---|---|---|
| 0 | 1 | `kind` | `0x01` = Set, `0x02` = Ping, `0x03` = Ack |
| 1 | 2 | `value_len` | big-endian `u16`, length of `value` |
| 3 | `value_len` | `value` | opaque payload bytes |

`KvCodec` implements `FrameCodec` for this layout and the example proves
four invariants:

- **round-trip** — three records encoded back to back into one buffer,
  then decoded off the front one at a time using `parse_frame`'s
  `consumed` count; every decoded frame equals the original.
- **partial buffer** — a frame truncated by two bytes returns
  `Err(FrameError::Incomplete)`, the normal "read more and retry" signal
  a streaming read loop sees mid-fill — never a panic, never a silent
  truncation.
- **malformed frame** — a `kind` byte outside `{0x01, 0x02, 0x03}` returns
  `Err(FrameError::UnknownKind)` — proxima's answer to attacker-controlled
  or corrupted bytes is a typed `Err`, not undefined behavior.
- **compose** — a decoded frame's `value` (`&[u8]`) is copied into an
  ordinary `Pipe<In = Vec<u8>, Out = Vec<u8>>` (`Uppercase`) and run
  through it — the codec's job ends at "here is a frame"; everything
  downstream is the same pipe algebra every other example uses.

## Run

```sh
cargo run --example codec
```

## What you'll see

Deterministic — same input, same output, every run:

```
codec: protocol = kv-tlv ([u8 kind][u16 BE value_len][value]), codec = KvCodec
codec: round-trip — 3 frames encoded then decoded, all equal
codec: partial    — truncated buffer signaled Incomplete, not a panic
codec: malformed  — unknown kind byte rejected as UnknownKind(255)
codec: compose    — decoded Set value fed through Uppercase pipe -> ANSWER=42
```

Each line is backed by `assert!`/`assert_eq!` in `main.rs`: round-trip
asserts every decoded frame equals the frame that was encoded and that
every byte of the buffer was consumed; partial and malformed each assert
the exact `FrameError` variant returned; compose asserts the uppercased
bytes equal the expected value. The same four functions are also run as
`#[test]`s.

## Required features

None — `proxima-codec` and `proxima-primitives` are both default (non-optional)
dependencies, so this example builds with default features; no extra
`--features` flag is needed.
