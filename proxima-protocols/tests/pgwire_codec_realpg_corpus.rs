//! Real-PostgreSQL parse-parity corpus (gate G13).
//!
//! Principle 14 (the incumbent is the *canonical* one — real PostgreSQL, not
//! the `pgwire 0.28` crate, which is itself a reimplementation we could share a
//! bug with) + principle 9 (real-world bytes) + principle 16 (vectors vendored
//! in-repo, re-proven on every build with no server required).
//!
//! Each fixture is the verbatim server->client byte stream of one real
//! `tokio-postgres` session against `postgres:17`, captured by
//! `cargo run -p proxima-pgwire --example capture_realpg`. Because every
//! scenario opens a fresh connection, each fixture also exercises the full
//! startup + SCRAM-SHA-256 handshake prefix — so the SASL parse path is proven
//! against PostgreSQL's actual challenge bytes, not our own encoder's.

#![allow(clippy::expect_used)]

use std::path::PathBuf;

use proxima_protocols::pgwire_codec::BackendMessage;
use proxima_protocols::pgwire_codec::backend::{AuthRequest, parse_backend};
use proxima_protocols::pgwire_codec::types::Oid;

const OID_INT4: Oid = Oid(23);
const OID_TEXT: Oid = Oid(25);
const OID_BOOL: Oid = Oid(16);

fn fixture(name: &str) -> Vec<u8> {
    // CI re-captures from the live `postgres` service into a temp dir and points
    // here, so the same semantic assertions re-prove against the current server
    // version (principle 16 canary); locally/default the vendored bytes are read.
    let dir = std::env::var("PGWIRE_REALPG_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/realpg")
        });
    let path = dir.join(format!("{name}.bin"));
    std::fs::read(&path).unwrap_or_else(|err| {
        panic!(
            "missing vendored real-pg fixture {} ({err}); regenerate with \
             `cargo run -p proxima-pgwire --example capture_realpg` against a docker postgres",
            path.display()
        )
    })
}

/// Parse the entire backend stream message by message. A clean parse with zero
/// trailing bytes IS the parity claim: our codec consumes exactly what real
/// PostgreSQL emitted.
fn parse_all(bytes: &[u8]) -> Vec<BackendMessage<'_>> {
    let mut offset = 0;
    let mut messages = Vec::new();
    while offset < bytes.len() {
        match parse_backend(&bytes[offset..]) {
            Ok(Some((message, consumed))) => {
                assert!(
                    consumed > 0,
                    "zero-length consume would loop forever at offset {offset}"
                );
                messages.push(message);
                offset += consumed;
            }
            Ok(None) => panic!(
                "incomplete trailing backend bytes: parsed to offset {offset} of {} total \
                 (real PostgreSQL never emits a partial message)",
                bytes.len()
            ),
            Err(err) => panic!(
                "real-postgres backend bytes failed to parse at offset {offset}/{}: {err:?}",
                bytes.len()
            ),
        }
    }
    assert!(!messages.is_empty(), "fixture parsed to zero messages");
    messages
}

fn count(
    messages: &[BackendMessage<'_>],
    predicate: impl Fn(&BackendMessage<'_>) -> bool,
) -> usize {
    messages.iter().filter(|message| predicate(message)).count()
}

#[test]
fn every_fixture_parses_with_zero_trailing_bytes() {
    for name in [
        "scram_startup",
        "simple_query",
        "extended_query",
        "error_response",
    ] {
        let _messages = parse_all(&fixture(name));
    }
}

#[test]
fn scram_startup_carries_the_real_sasl_handshake() {
    let bytes = fixture("scram_startup");
    let messages = parse_all(&bytes);

    let sasl_offered = count(&messages, |message| {
        matches!(
            message,
            BackendMessage::Authentication(AuthRequest::Sasl { .. })
        )
    });
    let sasl_continue = count(&messages, |message| {
        matches!(
            message,
            BackendMessage::Authentication(AuthRequest::SaslContinue { .. })
        )
    });
    let sasl_final = count(&messages, |message| {
        matches!(
            message,
            BackendMessage::Authentication(AuthRequest::SaslFinal { .. })
        )
    });
    let auth_ok = count(&messages, |message| {
        matches!(message, BackendMessage::Authentication(AuthRequest::Ok))
    });

    assert_eq!(sasl_offered, 1, "real PG offers SASL exactly once");
    assert_eq!(sasl_continue, 1, "one SCRAM server-first challenge");
    assert_eq!(sasl_final, 1, "one SCRAM server-final message");
    assert_eq!(
        auth_ok, 1,
        "AuthenticationOk after a successful SCRAM exchange"
    );

    assert!(
        count(&messages, |message| matches!(
            message,
            BackendMessage::BackendKeyData { .. }
        )) == 1,
        "BackendKeyData closes the startup phase"
    );
    assert!(
        count(&messages, |message| matches!(
            message,
            BackendMessage::ParameterStatus { .. }
        )) >= 1,
        "real PG reports server_version/client_encoding/etc as ParameterStatus"
    );
    assert!(
        count(&messages, |message| matches!(
            message,
            BackendMessage::ReadyForQuery { .. }
        )) >= 1,
        "ReadyForQuery signals the server is idle"
    );
}

#[test]
fn simple_query_row_description_matches_real_pg_type_oids() {
    let bytes = fixture("simple_query");
    let messages = parse_all(&bytes);

    let row_description = messages
        .iter()
        .find_map(|message| match message {
            BackendMessage::RowDescription { fields } => Some(*fields),
            _ => None,
        })
        .expect("simple query yields a RowDescription");

    let columns: Vec<_> = row_description.iter().collect();
    assert_eq!(columns.len(), 3, "select n, who, flag -> three columns");
    assert_eq!(columns[0].type_oid, OID_INT4, "42::int4 -> oid 23");
    assert_eq!(columns[1].type_oid, OID_TEXT, "'alice'::text -> oid 25");
    assert_eq!(columns[2].type_oid, OID_BOOL, "true -> oid 16");
    assert!(columns[0].name == "n", "column alias preserved on the wire");

    assert!(
        count(&messages, |message| matches!(
            message,
            BackendMessage::DataRow { .. }
        )) >= 1,
        "at least one DataRow"
    );
    let command_complete = messages
        .iter()
        .find_map(|message| match message {
            BackendMessage::CommandComplete { tag } => Some(*tag),
            _ => None,
        })
        .expect("CommandComplete present");
    assert!(
        command_complete == "SELECT 1",
        "real PG tags a single-row select as `SELECT 1`"
    );
}

#[test]
fn extended_query_carries_parameter_description_and_replies() {
    let bytes = fixture("extended_query");
    let messages = parse_all(&bytes);

    let parameter_description = messages
        .iter()
        .find_map(|message| match message {
            BackendMessage::ParameterDescription { parameter_types } => Some(*parameter_types),
            _ => None,
        })
        .expect("prepared statement Describe yields a ParameterDescription");
    let parameter_oids: Vec<Oid> = parameter_description.iter().collect();
    assert_eq!(
        parameter_oids,
        vec![OID_INT4, OID_TEXT],
        "$1::int4, $2::text"
    );

    assert_eq!(
        count(&messages, |message| matches!(
            message,
            BackendMessage::ParseComplete
        )),
        1,
        "exactly one ParseComplete"
    );
    assert_eq!(
        count(&messages, |message| matches!(
            message,
            BackendMessage::BindComplete
        )),
        1,
        "exactly one BindComplete"
    );
    assert!(
        count(&messages, |message| matches!(
            message,
            BackendMessage::RowDescription { .. }
        )) >= 1,
        "RowDescription for the result shape"
    );
    assert!(
        count(&messages, |message| matches!(
            message,
            BackendMessage::DataRow { .. }
        )) >= 1,
        "the bound row comes back"
    );
}

#[test]
fn error_response_carries_real_postgres_sqlstate() {
    let bytes = fixture("error_response");
    let messages = parse_all(&bytes);

    let error_fields = messages
        .iter()
        .find_map(|message| match message {
            BackendMessage::ErrorResponse { fields } => Some(*fields),
            _ => None,
        })
        .expect("querying a missing table yields an ErrorResponse");

    let mut sqlstate = None;
    let mut severity = None;
    for (code, value) in error_fields.iter() {
        match code {
            b'C' => sqlstate = Some(value),
            b'S' => severity = Some(value),
            _ => {}
        }
    }
    assert!(
        sqlstate.expect("ErrorResponse carries a C (SQLSTATE) field") == "42P01",
        "undefined_table is SQLSTATE 42P01 in real PostgreSQL"
    );
    assert!(severity.expect("ErrorResponse carries an S (severity) field") == "ERROR");

    assert!(
        count(&messages, |message| matches!(
            message,
            BackendMessage::ReadyForQuery { .. }
        )) >= 1,
        "the server recovers to ReadyForQuery after the error"
    );
}
