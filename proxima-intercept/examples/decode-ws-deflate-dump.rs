use std::path::PathBuf;

use flate2::{Decompress, FlushDecompress};
use futures::StreamExt;
use proxima_recording::BinSource;
use proxima_recording::event::{HttpEvent, ProtocolEvent};
use proxima_recording::source::RecordingSource;

// one-shot capture forensics: pull the websocket turn out of intercept.bin
// and inflate the permessage-deflate frames so the inner json is legible.
// proves the wire shape before the integration crate parses it for real.

fn main() -> Result<(), proxima_core::ProximaError> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
                .join(".proxima")
                .join("intercept.bin")
        });

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| proxima_core::ProximaError::Config(format!("runtime: {err}")))?;

    let offload_runtime = proxima::offline_runtime()?;
    runtime.block_on(async move {
        let source = BinSource::new(&path, offload_runtime);
        let mut stream = source.events();
        let mut request_bytes: Vec<u8> = Vec::new();
        let mut response_bytes: Vec<u8> = Vec::new();

        // events from concurrent connections interleave in the flat stream, so a
        // path flag is unreliable. the ws frames are the only non-utf8 chunks
        // (telemetry posts are utf8 json), which isolates the captured turn cleanly.
        while let Some(item) = stream.next().await {
            let event = match item {
                Ok(event) => event,
                Err(err) => {
                    eprintln!("read error: {err}");
                    break;
                }
            };
            match &event.event {
                ProtocolEvent::Http(HttpEvent::RequestChunk { data, .. })
                    if std::str::from_utf8(data).is_err() =>
                {
                    request_bytes.extend_from_slice(data);
                }
                ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. })
                    if std::str::from_utf8(data).is_err() =>
                {
                    response_bytes.extend_from_slice(data);
                }
                _ => {}
            }
        }

        println!("ws request ({} raw bytes)", request_bytes.len());
        dump_ws_messages(&request_bytes);
        println!("\nws response ({} raw bytes)", response_bytes.len());
        dump_ws_messages(&response_bytes);
        Ok::<(), proxima_core::ProximaError>(())
    })
}

struct Frame {
    opcode: u8,
    rsv1: bool,
    payload: Vec<u8>,
}

// parse rfc6455 frames out of a flat byte stream; tolerates a trailing
// partial frame (capture boundary) by stopping cleanly.
fn parse_frames(mut buffer: &[u8]) -> Vec<Frame> {
    let mut frames = Vec::new();
    while buffer.len() >= 2 {
        let first = buffer[0];
        let opcode = first & 0x0f;
        let rsv1 = first & 0x40 != 0;
        let masked = buffer[1] & 0x80 != 0;
        let len_code = (buffer[1] & 0x7f) as usize;
        let mut cursor = 2;
        let payload_len = match len_code {
            126 => {
                if buffer.len() < cursor + 2 {
                    break;
                }
                let len = u16::from_be_bytes([buffer[cursor], buffer[cursor + 1]]) as usize;
                cursor += 2;
                len
            }
            127 => {
                if buffer.len() < cursor + 8 {
                    break;
                }
                let mut len_bytes = [0u8; 8];
                len_bytes.copy_from_slice(&buffer[cursor..cursor + 8]);
                cursor += 8;
                u64::from_be_bytes(len_bytes) as usize
            }
            other => other,
        };
        let mask_key = if masked {
            if buffer.len() < cursor + 4 {
                break;
            }
            let key = [
                buffer[cursor],
                buffer[cursor + 1],
                buffer[cursor + 2],
                buffer[cursor + 3],
            ];
            cursor += 4;
            Some(key)
        } else {
            None
        };
        if buffer.len() < cursor + payload_len {
            break;
        }
        let mut payload = buffer[cursor..cursor + payload_len].to_vec();
        if let Some(key) = mask_key {
            for (index, byte) in payload.iter_mut().enumerate() {
                *byte ^= key[index % 4];
            }
        }
        frames.push(Frame {
            opcode,
            rsv1,
            payload,
        });
        buffer = &buffer[cursor + payload_len..];
    }
    frames
}

// permessage-deflate with context takeover: one shared inflate stream across
// all compressed messages, each message terminated by the 00 00 ff ff tail.
fn dump_ws_messages(buffer: &[u8]) {
    let frames = parse_frames(buffer);
    println!("  parsed {} frames", frames.len());
    let mut inflate = Decompress::new(false);
    let mut message: Vec<u8> = Vec::new();
    let mut compressed = false;
    let mut index = 0;
    for frame in &frames {
        match frame.opcode {
            0x1 | 0x2 => {
                message.clear();
                message.extend_from_slice(&frame.payload);
                compressed = frame.rsv1;
            }
            0x0 => message.extend_from_slice(&frame.payload),
            0x8 => {
                println!("  [close frame]");
                continue;
            }
            0x9 | 0xa => continue,
            other => {
                println!("  [opcode {other:#x}, {} bytes]", frame.payload.len());
                continue;
            }
        }
        if frame.opcode == 0x9 || frame.opcode == 0xa {
            continue;
        }
        if compressed {
            message.extend_from_slice(&[0x00, 0x00, 0xff, 0xff]);
        }
        let mut out = vec![0u8; 256 * 1024];
        let before = inflate.total_out();
        let status = inflate.decompress(&message, &mut out, FlushDecompress::Sync);
        match status {
            Ok(_) => {
                let produced = (inflate.total_out() - before) as usize;
                let text = String::from_utf8_lossy(&out[..produced]);
                index += 1;
                println!("  [msg {index}] {} bytes -> {}", produced, text.trim());
            }
            Err(err) => println!("  inflate error: {err}"),
        }
    }
}
