//! Decode an h2 client→server dump captured by the intercept proxy's h2 relay
//! (`relay_h2_capture` writes `proxima-h2-{host}-c2s.bin`). Feeds the bytes to
//! proxima-h2-codec's server `Connection` and prints the request HEADERS and DATA
//! frames — so the Connect/protobuf vocab of an h2 client can be
//! characterized from REAL captured bytes (§14), not reverse-engineered guesses.
//!
//! Run: `cargo run -p proxima-intercept --example decode-h2-dump -- /tmp/proxima-h2-api2.example.com-c2s.bin`

use proxima_protocols::http2_codec::connection::{Connection, ConnectionEvent};
use proxima_protocols::http2_codec::frame::StandardSettings;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: decode-h2-dump <c2s-dump.bin>");
        std::process::exit(2);
    });
    let bytes = std::fs::read(&path).unwrap_or_else(|err| {
        eprintln!("read {path}: {err}");
        std::process::exit(1);
    });
    eprintln!("decoding {} bytes from {path}\n", bytes.len());

    let mut connection = Connection::new(StandardSettings::default());
    if let Err(err) = connection.feed(&bytes) {
        eprintln!("[h2 feed error after partial decode] {err:?}");
    }

    let mut requests = 0usize;
    let mut data_frames = 0usize;
    while let Some(event) = connection.next_event() {
        match event {
            ConnectionEvent::RequestHead {
                stream_id,
                headers,
                end_stream,
            } => {
                requests += 1;
                println!("── stream {stream_id} HEADERS (end_stream={end_stream})");
                for (name, value) in &headers {
                    println!("   {}: {}", show(name), show(value));
                }
            }
            ConnectionEvent::BodyData {
                stream_id,
                data,
                end_stream,
            } => {
                data_frames += 1;
                println!(
                    "── stream {stream_id} DATA {} bytes (end_stream={end_stream})",
                    data.len()
                );
                println!("   utf8 : {}", show(&data[..data.len().min(160)]));
                println!("   hex  : {}", hex_preview(&data[..data.len().min(48)]));
            }
            ConnectionEvent::SettingsApplied => println!("── peer SETTINGS applied"),
            other => println!("── {other:?}"),
        }
    }
    println!("\nsummary: {requests} request head(s), {data_frames} data frame(s)");
    println!(
        "(content-type on a HEADERS frame names the vocab: application/grpc, application/connect+proto, application/json, ...)"
    );
}

fn show(raw: &[u8]) -> String {
    String::from_utf8_lossy(raw)
        .chars()
        .map(|character| {
            if character.is_control() {
                '.'
            } else {
                character
            }
        })
        .collect()
}

fn hex_preview(raw: &[u8]) -> String {
    raw.iter().map(|byte| format!("{byte:02x} ")).collect()
}
