//! Capture real Redis/Valkey server wire bytes into vendored fixtures — using
//! proxima's OWN client ([`proxima_redis::RedisClient`]), no `redis` crate, no
//! proxy, no async runtime. Pure `std::net` + our codec.
//!
//! Principle 9 (real-world data) + principle 14 (parity vs the canonical
//! incumbent) + principle 16 (vectors live in the repo): the codec's parse
//! oracle must be the server's *actual* output. Driving real Redis with our own
//! client (which does the `HELLO`/`AUTH` handshake and the commands via the same
//! codec under test) both dogfoods the stack and tees the verbatim server byte
//! stream into the fixtures via `RedisClient::captured`.
//!
//! Run against a docker server:
//!   docker run --rm -p 6379:6379 redis:7
//!   REDIS_REAL_HOST=127.0.0.1 REDIS_REAL_PORT=6379 \
//!     cargo run -p proxima-redis --features client --example capture_realredis
//!
//! The corpus test (`proxima-redis/tests/realredis_corpus.rs`) then parses the
//! vendored bytes on every build with no server required.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::env;
use std::net::TcpStream;
use std::path::{Path, PathBuf};

use proxima_redis::{RedisClient, RedisClientConfig};

const FIXTURE_DIR: &str = "proxima-redis/tests/fixtures/realredis";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = env::var("REDIS_REAL_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("REDIS_REAL_PORT").unwrap_or_else(|_| "6379".to_string());
    let out_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FIXTURE_DIR));
    std::fs::create_dir_all(&out_dir)?;
    let addr = format!("{host}:{port}");

    let resp3 = RedisClientConfig::builder()
        .host(host.clone())
        .port(port.parse()?)
        .build();
    let resp2 = RedisClientConfig::builder()
        .host(host)
        .port(port.parse()?)
        .resp3(false)
        .build();

    // hello3: the RESP3 HELLO 3 handshake reply — a map of server properties.
    capture(&addr, &out_dir, "hello3", &resp3, |_client| {})?;

    // string_ops: SET / GET hit, GET miss (null).
    capture(&addr, &out_dir, "string_ops", &resp2, |client| {
        assert_eq!(reply_text(client, &[b"SET", b"proxima:s", b"hello"]), "OK");
        assert_eq!(reply_text(client, &[b"GET", b"proxima:s"]), "hello");
        let miss = client
            .command(&[b"GET", b"proxima:absent"])
            .expect("get miss");
        assert!(miss.is_null(), "missing key -> null, got {miss:?}");
    })?;

    // numeric: integer replies.
    capture(&addr, &out_dir, "numeric", &resp2, |client| {
        let _ = client.command(&[b"DEL", b"proxima:n"]).expect("del");
        assert_eq!(
            client.command(&[b"INCR", b"proxima:n"]).unwrap().as_i64(),
            Some(1)
        );
        assert_eq!(
            client
                .command(&[b"INCRBY", b"proxima:n", b"41"])
                .unwrap()
                .as_i64(),
            Some(42)
        );
    })?;

    // list: an array reply.
    capture(&addr, &out_dir, "list", &resp2, |client| {
        let _ = client.command(&[b"DEL", b"proxima:l"]).expect("del");
        assert_eq!(
            client
                .command(&[b"RPUSH", b"proxima:l", b"a", b"b", b"c"])
                .unwrap()
                .as_i64(),
            Some(3)
        );
        let range = client
            .command(&[b"LRANGE", b"proxima:l", b"0", b"-1"])
            .expect("lrange");
        assert_eq!(range.as_array().map(<[_]>::len), Some(3));
    })?;

    // error: a -ERR simple error reply (unknown command).
    capture(&addr, &out_dir, "error", &resp2, |client| {
        let outcome = client
            .command(&[b"FLIBBERTIGIBBET"])
            .expect("unknown command is a reply");
        assert!(
            outcome.as_error().is_some(),
            "unknown command -> error reply, got {outcome:?}"
        );
    })?;

    eprintln!("captured real-redis fixtures into {}", out_dir.display());
    Ok(())
}

fn reply_text(client: &mut RedisClient<TcpStream>, argv: &[&[u8]]) -> String {
    let value = client.command(argv).expect("command");
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| panic!("expected text reply, got {value:?}"))
}

/// Connect our client (capturing), run `scenario`, persist the captured server
/// byte stream to `<name>.bin`.
fn capture(
    addr: &str,
    out_dir: &Path,
    name: &str,
    config: &RedisClientConfig,
    scenario: impl FnOnce(&mut RedisClient<TcpStream>),
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = TcpStream::connect(addr)?;
    stream.set_nodelay(true)?;
    let mut client = RedisClient::connect_capturing(stream, config)?;
    scenario(&mut client);
    let bytes = std::mem::take(&mut client.captured);
    let path = out_dir.join(format!("{name}.bin"));
    std::fs::write(&path, &bytes)?;
    eprintln!(
        "  {name}: {} server bytes -> {}",
        bytes.len(),
        path.display()
    );
    let _ = client.close();
    Ok(())
}
