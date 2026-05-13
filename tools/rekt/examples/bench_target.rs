// a deliberately trivial, fast HTTP/1.1 keep-alive server: the FIXED target both
// rekt and wrk hit, so the comparison measures the load GENERATORS, not the
// server. thread-per-connection, no routing, no parse beyond finding the header
// terminator, one fixed response per request. std-only (no proxima), so it builds
// without the scheduler feature.
//
//   cargo run --release --example bench_target -- 127.0.0.1:8080

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

const RESPONSE_OK: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok";
const RESPONSE_EMPTY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";

fn main() {
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:8080".to_string());
    // arg 2: "empty" → Content-Length: 0; anything else → 2-byte "ok" body.
    let response: &'static [u8] = match std::env::args().nth(2).as_deref() {
        Some("empty") => RESPONSE_EMPTY,
        _ => RESPONSE_OK,
    };
    let listener = match TcpListener::bind(&addr) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("bench_target: bind {addr}: {error}");
            std::process::exit(1);
        }
    };
    println!("bench_target: listening on {addr}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                thread::spawn(move || serve(stream, response));
            }
            Err(error) => eprintln!("bench_target: accept: {error}"),
        }
    }
}

// one connection: read requests, answer each with the fixed response, until the
// peer closes. allocation-free hot loop — a small no-body GET always lands in one
// read, so we count header terminators in the read buffer and answer each. (a
// request split across reads would miscount; fine for this fixed benchmark GET.)
fn serve(mut stream: TcpStream, response: &'static [u8]) {
    let _ = stream.set_nodelay(true);
    let mut buf = [0u8; 16 * 1024];
    loop {
        let read = match stream.read(&mut buf) {
            Ok(0) => return,
            Ok(read) => read,
            Err(_) => return,
        };
        for _ in 0..count_terminators(&buf[..read]) {
            if stream.write_all(response).is_err() {
                return;
            }
        }
    }
}

// number of `\r\n\r\n` head terminators in the buffer — one per pipelined request.
fn count_terminators(bytes: &[u8]) -> usize {
    bytes
        .windows(4)
        .filter(|window| *window == b"\r\n\r\n")
        .count()
}
