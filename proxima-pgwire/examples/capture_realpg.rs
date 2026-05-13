//! Capture real PostgreSQL backend wire bytes into vendored fixtures —
//! using proxima's OWN PostgreSQL client ([`proxima_pgwire::PgClient`]), no
//! `tokio-postgres`, no proxy, no async runtime. Pure `std::net` + our codec.
//!
//! Principle 9 (real-world data) + principle 14 (parity vs the canonical
//! incumbent) + principle 16 (vectors live in the repo): the codec's parse
//! oracle must be PostgreSQL's *actual* output. Driving real PG with our own
//! client (which does the startup, SCRAM-SHA-256 handshake, and simple/extended
//! queries via the same codec under test) both dogfoods the stack and tees the
//! verbatim backend byte stream into the fixtures via `PgClient::captured`.
//!
//! Run against a docker server:
//!   docker run --rm -e POSTGRES_PASSWORD=proxima \
//!     -e POSTGRES_HOST_AUTH_METHOD=scram-sha-256 -p 5432:5432 postgres:17
//!   PGWIRE_REALPG_HOST=127.0.0.1 PGWIRE_REALPG_PORT=5432 \
//!     PGWIRE_REALPG_PASSWORD=proxima \
//!     cargo run -p proxima-pgwire --example capture_realpg
//!
//! The corpus test (`proxima-pgwire-codec/tests/realpg_corpus.rs`) then parses
//! the vendored bytes on every build with no server required.

#![allow(clippy::expect_used)]

use std::env;
use std::net::TcpStream;
use std::path::PathBuf;

use proxima_pgwire::PgClient;

const FIXTURE_DIR: &str = "proxima-pgwire-codec/tests/fixtures/realpg";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = env::var("PGWIRE_REALPG_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("PGWIRE_REALPG_PORT").unwrap_or_else(|_| "5432".to_string());
    let user = env::var("PGWIRE_REALPG_USER").unwrap_or_else(|_| "postgres".to_string());
    let password = env::var("PGWIRE_REALPG_PASSWORD").unwrap_or_else(|_| "proxima".to_string());
    let database = env::var("PGWIRE_REALPG_DB").unwrap_or_else(|_| "postgres".to_string());
    let out_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FIXTURE_DIR));
    std::fs::create_dir_all(&out_dir)?;
    let addr = format!("{host}:{port}");

    // scram_startup: the connect alone captures startup + SCRAM handshake +
    // ParameterStatus(es) + BackendKeyData + ReadyForQuery.
    capture(
        &addr,
        &out_dir,
        "scram_startup",
        &user,
        &password,
        &database,
        |_client| {},
    )?;

    capture(
        &addr,
        &out_dir,
        "simple_query",
        &user,
        &password,
        &database,
        |client| {
            let result = client
                .simple_query("select 42::int4 as n, 'alice'::text as who, true as flag")
                .expect("simple query");
            assert_eq!(result.columns.len(), 3);
        },
    )?;

    capture(
        &addr,
        &out_dir,
        "extended_query",
        &user,
        &password,
        &database,
        |client| {
            let result = client
                .extended_query("select $1::int4 as n, $2::text as who", &["7", "bob"])
                .expect("extended query");
            assert_eq!(result.text(0, 0), Some("7"));
        },
    )?;

    capture(
        &addr,
        &out_dir,
        "error_response",
        &user,
        &password,
        &database,
        |client| {
            // an expected server error; the session consumes the recovering RFQ
            // before surfacing the error, so the fixture carries ErrorResponse AND
            // ReadyForQuery.
            let outcome = client.simple_query("select * from a_table_that_does_not_exist");
            assert!(outcome.is_err(), "missing table must error");
        },
    )?;

    eprintln!("captured real-pg fixtures into {}", out_dir.display());
    Ok(())
}

/// Connect our client (capturing), run `scenario`, persist the captured
/// backend byte stream to `<name>.bin`.
fn capture(
    addr: &str,
    out_dir: &std::path::Path,
    name: &str,
    user: &str,
    password: &str,
    database: &str,
    scenario: impl FnOnce(&mut PgClient<TcpStream>),
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = TcpStream::connect(addr)?;
    stream.set_nodelay(true)?;
    let mut client = PgClient::connect_capturing(stream, user, password, database)?;
    scenario(&mut client);
    let bytes = std::mem::take(&mut client.captured);
    let path = out_dir.join(format!("{name}.bin"));
    std::fs::write(&path, &bytes)?;
    eprintln!(
        "  {name}: {} backend bytes -> {}",
        bytes.len(),
        path.display()
    );
    let _ = client.close();
    Ok(())
}
