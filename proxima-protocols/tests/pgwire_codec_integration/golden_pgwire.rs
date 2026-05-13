#![allow(clippy::unwrap_used, clippy::expect_used)]

use bytes::BytesMut;
use pgwire::messages::Message;
use pgwire::messages::copy::{
    CopyBothResponse, CopyDone, CopyInResponse, CopyOutResponse, MESSAGE_TYPE_BYTE_COPY_DONE,
    MESSAGE_TYPE_BYTE_COPY_FAIL,
};
use pgwire::messages::data::{
    DataRow, FORMAT_CODE_BINARY, FORMAT_CODE_TEXT, FieldDescription, NoData, ParameterDescription,
    RowDescription,
};
use pgwire::messages::extendedquery::{
    Bind, BindComplete, Close, CloseComplete, Describe, Execute, Flush, Parse, ParseComplete,
    PortalSuspended, Sync, TARGET_TYPE_BYTE_STATEMENT,
};
use pgwire::messages::response::{
    CommandComplete, EmptyQueryResponse, ErrorResponse, NoticeResponse, NotificationResponse,
    ReadyForQuery, TransactionStatus,
};
use pgwire::messages::simplequery::Query;
use pgwire::messages::startup::{
    Authentication, BackendKeyData, ParameterStatus, Password, SASLInitialResponse, SASLResponse,
    SslRequest, Startup,
};
use pgwire::messages::terminate::Terminate;
use proxima_protocols::pgwire_codec::backend::{
    SslResponse, encode_copy_both_response, encode_copy_in_response, encode_copy_out_response,
    parse_backend, parse_ssl_response,
};
use proxima_protocols::pgwire_codec::frontend::{BindWriter, parse_frontend, parse_initial};
use proxima_protocols::pgwire_codec::types::{
    CopyFormat, FormatCode, Oid, ProtocolVersion, StatementTarget,
    TransactionStatus as OurTransactionStatus, error_field,
};
use proxima_protocols::pgwire_codec::{BackendMessage, FrontendMessage, InitialMessage};

fn upstream_encode<M: Message>(msg: &M) -> BytesMut {
    let mut buf = BytesMut::new();
    msg.encode(&mut buf).expect("upstream encode must succeed");
    buf
}

#[test]
fn startup_with_psql_params_golden_parity() {
    let mut upstream = Startup::default();
    upstream
        .parameters
        .insert("user".to_owned(), "alice".to_owned());
    upstream
        .parameters
        .insert("database".to_owned(), "appdb".to_owned());
    upstream
        .parameters
        .insert("application_name".to_owned(), "psql".to_owned());
    upstream
        .parameters
        .insert("client_encoding".to_owned(), "UTF8".to_owned());

    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_initial(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len(), "consumed all bytes");

    let InitialMessage::Startup(startup) = msg else {
        panic!("expected Startup");
    };
    assert_eq!(startup.version, ProtocolVersion::V3_0);
    assert_eq!(
        startup.user(),
        Some(proxima_protocols::pgwire_codec::PgStr::new(b"alice"))
    );
    assert_eq!(
        startup.database(),
        Some(proxima_protocols::pgwire_codec::PgStr::new(b"appdb"))
    );

    let mut our_buf = vec![0u8; 512];
    let written = msg.encode(&mut our_buf).expect("our encode must succeed");
    assert_eq!(
        &our_buf[..written],
        &upstream_bytes[..],
        "byte-for-byte parity"
    );
}

#[test]
fn ssl_request_golden_parity() {
    let upstream = SslRequest::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_initial(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, InitialMessage::SslRequest));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn password_golden_parity() {
    let upstream = Password::new("s3cr3t".to_owned());
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let FrontendMessage::AuthData(auth) = msg else {
        panic!("expected AuthData");
    };
    let password = auth.as_password().expect("must be valid password");
    assert_eq!(password, "s3cr3t");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn sasl_initial_response_golden_parity() {
    use bytes::Bytes;
    let upstream = SASLInitialResponse::new(
        "SCRAM-SHA-256".to_owned(),
        Some(Bytes::from_static(b"client-first-message")),
    );
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let FrontendMessage::AuthData(auth) = msg else {
        panic!("expected AuthData");
    };
    let sasl = auth.as_sasl_initial().expect("must be valid sasl initial");
    assert_eq!(sasl.mechanism, "SCRAM-SHA-256");
    assert_eq!(sasl.data, Some(b"client-first-message".as_slice()));

    let mut our_buf = vec![0u8; 256];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn sasl_response_golden_parity() {
    use bytes::Bytes;
    let upstream = SASLResponse::new(Bytes::from_static(b"client-final-message"));
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, _consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");

    let FrontendMessage::AuthData(auth) = msg else {
        panic!("expected AuthData");
    };
    assert_eq!(auth.as_sasl_response(), b"client-final-message");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn query_golden_parity() {
    let upstream = Query::new("select id, email from users where id = $1".to_owned());
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let FrontendMessage::Query { sql } = msg else {
        panic!("expected Query");
    };
    assert_eq!(sql, "select id, email from users where id = $1");

    let mut our_buf = vec![0u8; 256];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn parse_message_golden_parity() {
    let upstream = Parse::new(
        Some("get-user".to_owned()),
        "select id, email from users where id = $1".to_owned(),
        vec![23u32],
    );
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let FrontendMessage::Parse(parse) = msg else {
        panic!("expected Parse");
    };
    assert_eq!(parse.statement, "get-user");
    assert_eq!(parse.sql, "select id, email from users where id = $1");
    let oids: Vec<Oid> = parse.parameter_types.iter().collect();
    assert_eq!(oids, vec![Oid(23)]);

    let mut our_buf = vec![0u8; 256];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn bind_mixed_formats_null_param_golden_parity() {
    use bytes::Bytes;
    let upstream = Bind::new(
        Some("my-portal".to_owned()),
        Some("get-user".to_owned()),
        vec![0i16, 1i16],
        vec![
            Some(Bytes::from_static(b"42")),
            None,
            Some(Bytes::from_static(b"hello")),
        ],
        vec![0i16],
    );
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let FrontendMessage::Bind(bind) = msg else {
        panic!("expected Bind");
    };
    assert_eq!(bind.portal, "my-portal");
    assert_eq!(bind.statement, "get-user");
    assert_eq!(bind.parameter_formats.len(), 2);
    assert_eq!(bind.parameters.len(), 3);

    let params: Vec<Option<&[u8]>> = bind.parameters.iter().collect();
    assert_eq!(params[0], Some(b"42".as_slice()));
    assert_eq!(params[1], None);
    assert_eq!(params[2], Some(b"hello".as_slice()));

    let mut our_buf = vec![0u8; 256];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn describe_statement_golden_parity() {
    let upstream = Describe::new(TARGET_TYPE_BYTE_STATEMENT, Some("get-user".to_owned()));
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let FrontendMessage::Describe { target, name } = msg else {
        panic!("expected Describe");
    };
    assert_eq!(target, StatementTarget::Statement);
    assert_eq!(name, "get-user");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn execute_golden_parity() {
    let upstream = Execute::new(Some("my-portal".to_owned()), 100);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let FrontendMessage::Execute { portal, max_rows } = msg else {
        panic!("expected Execute");
    };
    assert_eq!(portal, "my-portal");
    assert_eq!(max_rows, 100);

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn close_statement_golden_parity() {
    let upstream = Close::new(TARGET_TYPE_BYTE_STATEMENT, Some("get-user".to_owned()));
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let FrontendMessage::Close { target, name } = msg else {
        panic!("expected Close");
    };
    assert_eq!(target, StatementTarget::Statement);
    assert_eq!(name, "get-user");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn flush_golden_parity() {
    let upstream = Flush::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, FrontendMessage::Flush));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn sync_golden_parity() {
    let upstream = Sync::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, FrontendMessage::Sync));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn terminate_golden_parity() {
    let upstream = Terminate::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, FrontendMessage::Terminate));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn copy_data_frontend_golden_parity() {
    use bytes::Bytes;
    let upstream = pgwire::messages::copy::CopyData::new(Bytes::from_static(b"row1\trow2\n"));
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let FrontendMessage::CopyData { data } = msg else {
        panic!("expected CopyData");
    };
    assert_eq!(data, b"row1\trow2\n");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn copy_done_frontend_golden_parity() {
    let upstream = CopyDone::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_frontend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, FrontendMessage::CopyDone));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn copy_fail_spec_vector_correct_tag() {
    let message = "copy operation failed by client";
    let msg_bytes = message.as_bytes();
    let body_len = 4 + msg_bytes.len() + 1;
    let mut expected = vec![b'f'];
    expected.extend_from_slice(&(body_len as i32).to_be_bytes());
    expected.extend_from_slice(msg_bytes);
    expected.push(0);

    let mut our_buf = vec![0u8; 64];
    let fail_msg = FrontendMessage::CopyFail {
        message: proxima_protocols::pgwire_codec::PgStr::new(message.as_bytes()),
    };
    let written = fail_msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(
        &our_buf[..written],
        &expected[..],
        "our tag must be 'f' per spec"
    );
}

#[test]
fn upstream_pgwire_copyfail_tag_is_wrong_per_spec() {
    use pgwire::messages::copy::CopyFail;
    let upstream = CopyFail::new("copy operation failed by client".to_owned());
    let mut buf = BytesMut::new();
    upstream.encode(&mut buf).expect("upstream encode succeeds");

    assert_eq!(
        buf[0], MESSAGE_TYPE_BYTE_COPY_DONE,
        "upstream pgwire 0.28 CopyFail::message_type() returns the CopyDone constant b'c', \
         not b'f' — this is a bug in upstream (copy.rs references MESSAGE_TYPE_BYTE_COPY_DONE)"
    );
    assert_ne!(
        buf[0], MESSAGE_TYPE_BYTE_COPY_FAIL,
        "upstream does NOT emit the spec-correct 'f'"
    );
}

#[test]
fn auth_ok_golden_parity() {
    let upstream = Authentication::Ok;
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::Authentication(proxima_protocols::pgwire_codec::AuthRequest::Ok) = msg else {
        panic!("expected AuthOk");
    };

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn auth_cleartext_password_golden_parity() {
    let upstream = Authentication::CleartextPassword;
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, _) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");

    let BackendMessage::Authentication(proxima_protocols::pgwire_codec::AuthRequest::CleartextPassword) = msg
    else {
        panic!("expected CleartextPassword");
    };

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn auth_md5_password_golden_parity() {
    let upstream = Authentication::MD5Password(vec![1u8, 2, 3, 4]);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::Authentication(proxima_protocols::pgwire_codec::AuthRequest::Md5Password { salt }) =
        msg
    else {
        panic!("expected Md5Password");
    };
    assert_eq!(salt, [1, 2, 3, 4]);

    let mut our_buf = vec![0u8; 32];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn auth_sasl_golden_parity() {
    let upstream = Authentication::SASL(vec![
        "SCRAM-SHA-256".to_owned(),
        "SCRAM-SHA-256-PLUS".to_owned(),
    ]);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::Authentication(proxima_protocols::pgwire_codec::AuthRequest::Sasl { mechanisms }) =
        msg
    else {
        panic!("expected Sasl");
    };
    let mechs: Vec<proxima_protocols::pgwire_codec::PgStr<'_>> = mechanisms.iter().collect();
    assert_eq!(mechs[0], "SCRAM-SHA-256");
    assert_eq!(mechs[1], "SCRAM-SHA-256-PLUS");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn auth_sasl_continue_golden_parity() {
    use bytes::Bytes;
    let upstream = Authentication::SASLContinue(Bytes::from_static(b"server-first-message-data"));
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::Authentication(proxima_protocols::pgwire_codec::AuthRequest::SaslContinue { data }) =
        msg
    else {
        panic!("expected SaslContinue");
    };
    assert_eq!(data, b"server-first-message-data");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn auth_sasl_final_golden_parity() {
    use bytes::Bytes;
    let upstream = Authentication::SASLFinal(Bytes::from_static(b"server-final-message-data"));
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::Authentication(proxima_protocols::pgwire_codec::AuthRequest::SaslFinal { data }) = msg
    else {
        panic!("expected SaslFinal");
    };
    assert_eq!(data, b"server-final-message-data");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn parameter_status_golden_parity() {
    let upstream = ParameterStatus::new("server_version".to_owned(), "15.2".to_owned());
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::ParameterStatus { name, value } = msg else {
        panic!("expected ParameterStatus");
    };
    assert_eq!(name, "server_version");
    assert_eq!(value, "15.2");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn backend_key_data_golden_parity() {
    let upstream = BackendKeyData::new(12345, 67890);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::BackendKeyData {
        process_id,
        secret_key,
    } = msg
    else {
        panic!("expected BackendKeyData");
    };
    assert_eq!(process_id, 12345);
    assert_eq!(secret_key, &67890i32.to_be_bytes());

    let mut our_buf = vec![0u8; 32];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn parse_complete_golden_parity() {
    let upstream = ParseComplete::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, BackendMessage::ParseComplete));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn bind_complete_golden_parity() {
    let upstream = BindComplete::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, BackendMessage::BindComplete));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn close_complete_golden_parity() {
    let upstream = CloseComplete::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, BackendMessage::CloseComplete));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn portal_suspended_golden_parity() {
    let upstream = PortalSuspended::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, BackendMessage::PortalSuspended));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn command_complete_golden_parity() {
    let upstream = CommandComplete::new("SELECT 42".to_owned());
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::CommandComplete { tag } = msg else {
        panic!("expected CommandComplete");
    };
    assert_eq!(tag, "SELECT 42");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn empty_query_response_golden_parity() {
    let upstream = EmptyQueryResponse::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, BackendMessage::EmptyQueryResponse));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn ready_for_query_idle_golden_parity() {
    let upstream = ReadyForQuery::new(TransactionStatus::Idle);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::ReadyForQuery { status } = msg else {
        panic!("expected ReadyForQuery");
    };
    assert_eq!(status, OurTransactionStatus::Idle);

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn ready_for_query_in_transaction_golden_parity() {
    let upstream = ReadyForQuery::new(TransactionStatus::Transaction);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, _) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");

    let BackendMessage::ReadyForQuery { status } = msg else {
        panic!("expected ReadyForQuery");
    };
    assert_eq!(status, OurTransactionStatus::InTransaction);

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn ready_for_query_failed_golden_parity() {
    let upstream = ReadyForQuery::new(TransactionStatus::Error);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, _) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");

    let BackendMessage::ReadyForQuery { status } = msg else {
        panic!("expected ReadyForQuery");
    };
    assert_eq!(status, OurTransactionStatus::Failed);

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn error_response_multi_fields_golden_parity() {
    let mut upstream = ErrorResponse::default();
    upstream
        .fields
        .push((error_field::SEVERITY, "ERROR".to_owned()));
    upstream
        .fields
        .push((error_field::SEVERITY_NON_LOCALIZED, "ERROR".to_owned()));
    upstream
        .fields
        .push((error_field::CODE, "42P01".to_owned()));
    upstream.fields.push((
        error_field::MESSAGE,
        "relation \"users\" does not exist".to_owned(),
    ));
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::ErrorResponse { fields } = msg else {
        panic!("expected ErrorResponse");
    };
    assert_eq!(
        fields.get(error_field::SEVERITY),
        Some(proxima_protocols::pgwire_codec::PgStr::new(b"ERROR"))
    );
    assert_eq!(
        fields.get(error_field::CODE),
        Some(proxima_protocols::pgwire_codec::PgStr::new(b"42P01"))
    );

    let mut our_buf = vec![0u8; 256];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn notice_response_golden_parity() {
    let mut upstream = NoticeResponse::default();
    upstream
        .fields
        .push((error_field::SEVERITY, "NOTICE".to_owned()));
    upstream
        .fields
        .push((error_field::MESSAGE, "implicit transaction".to_owned()));
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::NoticeResponse { fields } = msg else {
        panic!("expected NoticeResponse");
    };
    assert_eq!(
        fields.get(error_field::SEVERITY),
        Some(proxima_protocols::pgwire_codec::PgStr::new(b"NOTICE"))
    );

    let mut our_buf = vec![0u8; 128];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn notification_response_golden_parity() {
    let upstream =
        NotificationResponse::new(55555, "my_channel".to_owned(), "my_payload".to_owned());
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::NotificationResponse {
        process_id,
        channel,
        payload,
    } = msg
    else {
        panic!("expected NotificationResponse");
    };
    assert_eq!(process_id, 55555);
    assert_eq!(channel, "my_channel");
    assert_eq!(payload, "my_payload");

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn row_description_multi_fields_golden_parity() {
    let mut upstream = RowDescription::default();

    let mut field_id = FieldDescription::default();
    field_id.name = "id".to_owned();
    field_id.table_id = 16384;
    field_id.column_id = 1;
    field_id.type_id = 23;
    field_id.type_size = 4;
    field_id.type_modifier = -1;
    field_id.format_code = FORMAT_CODE_TEXT;
    upstream.fields.push(field_id);

    let mut field_email = FieldDescription::default();
    field_email.name = "email".to_owned();
    field_email.table_id = 16384;
    field_email.column_id = 2;
    field_email.type_id = 25;
    field_email.type_size = -1;
    field_email.type_modifier = -1;
    field_email.format_code = FORMAT_CODE_TEXT;
    upstream.fields.push(field_email);

    let mut field_active = FieldDescription::default();
    field_active.name = "active".to_owned();
    field_active.table_id = 16384;
    field_active.column_id = 3;
    field_active.type_id = 16;
    field_active.type_size = 1;
    field_active.type_modifier = -1;
    field_active.format_code = FORMAT_CODE_BINARY;
    upstream.fields.push(field_active);

    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::RowDescription { fields } = msg else {
        panic!("expected RowDescription");
    };
    assert_eq!(fields.len(), 3);

    let field_list: Vec<_> = fields.iter().collect();
    assert_eq!(field_list[0].name, "id");
    assert_eq!(field_list[0].type_oid, Oid(23));
    assert_eq!(field_list[1].name, "email");
    assert_eq!(field_list[1].type_oid, Oid(25));
    assert_eq!(field_list[2].name, "active");
    assert_eq!(field_list[2].format, FormatCode::Binary);

    let mut our_buf = vec![0u8; 256];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn parameter_description_golden_parity() {
    let upstream = ParameterDescription::new(vec![23u32, 25, 16]);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::ParameterDescription { parameter_types } = msg else {
        panic!("expected ParameterDescription");
    };
    let oids: Vec<Oid> = parameter_types.iter().collect();
    assert_eq!(oids, vec![Oid(23), Oid(25), Oid(16)]);

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn data_row_null_and_values_golden_parity() {
    let mut upstream = DataRow::default();
    upstream.data.extend_from_slice(&2i32.to_be_bytes());
    upstream.data.extend_from_slice(b"42");
    upstream.data.extend_from_slice(&(-1i32).to_be_bytes());
    upstream.data.extend_from_slice(&5i32.to_be_bytes());
    upstream.data.extend_from_slice(b"hello");
    upstream.field_count = 3;

    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::DataRow { columns } = msg else {
        panic!("expected DataRow");
    };
    assert_eq!(columns.len(), 3);

    let values: Vec<Option<&[u8]>> = columns.iter().collect();
    assert_eq!(values[0], Some(b"42".as_slice()));
    assert_eq!(values[1], None);
    assert_eq!(values[2], Some(b"hello".as_slice()));

    let mut our_buf = vec![0u8; 128];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn no_data_golden_parity() {
    let upstream = NoData::new();
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, BackendMessage::NoData));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn copy_in_response_golden_parity() {
    let upstream = CopyInResponse::new(0, 3, vec![0i16, 0, 1]);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::CopyInResponse {
        format,
        column_formats,
    } = msg
    else {
        panic!("expected CopyInResponse");
    };
    assert_eq!(format, CopyFormat::Text);
    assert_eq!(column_formats.len(), 3);

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn copy_out_response_golden_parity() {
    let upstream = CopyOutResponse::new(1, 2, vec![1i16, 1]);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());

    let BackendMessage::CopyOutResponse { format, .. } = msg else {
        panic!("expected CopyOutResponse");
    };
    assert_eq!(format, CopyFormat::Binary);

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn copy_both_response_golden_parity() {
    let upstream = CopyBothResponse::new(0, 1, vec![0i16]);
    let upstream_bytes = upstream_encode(&upstream);

    let (msg, consumed) = parse_backend(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, upstream_bytes.len());
    assert!(matches!(msg, BackendMessage::CopyBothResponse { .. }));

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn cancel_request_spec_vector() {
    let process_id: i32 = 12345;
    let secret_key: i32 = 67890;

    let expected: Vec<u8> = {
        let mut buf = vec![];
        buf.extend_from_slice(&16i32.to_be_bytes());
        buf.extend_from_slice(&80877102i32.to_be_bytes());
        buf.extend_from_slice(&process_id.to_be_bytes());
        buf.extend_from_slice(&secret_key.to_be_bytes());
        buf
    };

    let (msg, consumed) = parse_initial(&expected)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, expected.len());

    let InitialMessage::Cancel(cancel) = msg else {
        panic!("expected Cancel");
    };
    assert_eq!(cancel.process_id, process_id);
    assert_eq!(cancel.secret_key, secret_key.to_be_bytes());

    let mut our_buf = vec![0u8; 32];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &expected[..]);
}

#[test]
fn gssenc_request_spec_vector() {
    let expected: Vec<u8> = {
        let mut buf = vec![];
        buf.extend_from_slice(&8i32.to_be_bytes());
        buf.extend_from_slice(&80877104i32.to_be_bytes());
        buf
    };

    let (msg, consumed) = parse_initial(&expected)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, expected.len());
    assert!(matches!(msg, InitialMessage::GssEncRequest));

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &expected[..]);
}

#[test]
fn function_call_spec_vector_round_trip() {
    let object_oid: u32 = 1753;
    let arg_data = b"hello";

    let expected: Vec<u8> = {
        let body_len: i32 = 4 + 2 + 2 * 1i16 as i32 + 2 + (4 + arg_data.len() as i32) + 2;
        let total_len = body_len + 4;
        let mut buf = vec![b'F'];
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(&object_oid.to_be_bytes());
        buf.extend_from_slice(&1i16.to_be_bytes());
        buf.extend_from_slice(&0i16.to_be_bytes());
        buf.extend_from_slice(&1i16.to_be_bytes());
        buf.extend_from_slice(&(arg_data.len() as i32).to_be_bytes());
        buf.extend_from_slice(arg_data);
        buf.extend_from_slice(&0i16.to_be_bytes());
        buf
    };

    let (msg, consumed) = parse_frontend(&expected)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, expected.len());

    let FrontendMessage::FunctionCall(call) = msg else {
        panic!("expected FunctionCall");
    };
    assert_eq!(call.object, Oid(1753));
    assert_eq!(call.result_format, FormatCode::Text);

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &expected[..]);
}

#[test]
fn function_call_response_spec_vector_round_trip() {
    let result_data = b"result";

    let expected: Vec<u8> = {
        let mut buf = vec![b'V'];
        let body_len: i32 = 4 + 4 + result_data.len() as i32;
        buf.extend_from_slice(&body_len.to_be_bytes());
        buf.extend_from_slice(&(result_data.len() as i32).to_be_bytes());
        buf.extend_from_slice(result_data);
        buf
    };

    let (msg, consumed) = parse_backend(&expected)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, expected.len());

    let BackendMessage::FunctionCallResponse { value: Some(value) } = msg else {
        panic!("expected FunctionCallResponse with value");
    };
    assert_eq!(value, result_data);

    let mut our_buf = vec![0u8; 64];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &expected[..]);
}

#[test]
fn function_call_response_null_spec_vector_round_trip() {
    let expected: Vec<u8> = {
        let mut buf = vec![b'V'];
        let body_len: i32 = 4 + 4;
        buf.extend_from_slice(&body_len.to_be_bytes());
        buf.extend_from_slice(&(-1i32).to_be_bytes());
        buf
    };

    let (msg, consumed) = parse_backend(&expected)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, expected.len());

    let BackendMessage::FunctionCallResponse { value: None } = msg else {
        panic!("expected FunctionCallResponse with null");
    };

    let mut our_buf = vec![0u8; 16];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &expected[..]);
}

#[test]
fn negotiate_protocol_version_spec_vector_round_trip() {
    let newest_minor: i32 = 1;
    let options = &[b"_pq_.trace".as_slice(), b"_pq_.unknown".as_slice()];

    let expected: Vec<u8> = {
        let body_len: i32 = 4 + 4 + options[0].len() as i32 + 1 + options[1].len() as i32 + 1;
        let length_field = body_len + 4;
        let mut buf = vec![b'v'];
        buf.extend_from_slice(&length_field.to_be_bytes());
        buf.extend_from_slice(&newest_minor.to_be_bytes());
        buf.extend_from_slice(&(options.len() as i32).to_be_bytes());
        for option in options {
            buf.extend_from_slice(option);
            buf.push(0);
        }
        buf
    };

    let (msg, consumed) = parse_backend(&expected)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, expected.len());

    let BackendMessage::NegotiateProtocolVersion {
        newest_minor: got_minor,
        unsupported_options,
    } = msg
    else {
        panic!("expected NegotiateProtocolVersion");
    };
    assert_eq!(got_minor, newest_minor);
    assert_eq!(unsupported_options.len(), 2);
    let opts: Vec<proxima_protocols::pgwire_codec::PgStr<'_>> = unsupported_options.iter().collect();
    assert_eq!(opts[0], "_pq_.trace");
    assert_eq!(opts[1], "_pq_.unknown");

    let mut our_buf = vec![0u8; 128];
    let written = msg.encode(&mut our_buf).expect("encode must succeed");
    assert_eq!(&our_buf[..written], &expected[..]);
}

#[test]
fn ssl_response_accept_round_trip() {
    let mut our_buf = vec![0u8; 4];
    let written = SslResponse::Accept
        .encode(&mut our_buf)
        .expect("encode must succeed");
    assert_eq!(written, 1);
    assert_eq!(our_buf[0], b'S');

    let (response, consumed) = parse_ssl_response(&our_buf[..written])
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(consumed, 1);
    assert_eq!(response, SslResponse::Accept);
}

#[test]
fn ssl_response_refuse_round_trip() {
    let mut our_buf = vec![0u8; 4];
    SslResponse::Refuse
        .encode(&mut our_buf)
        .expect("encode must succeed");
    assert_eq!(our_buf[0], b'N');

    let (response, _) = parse_ssl_response(&our_buf[..1])
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(response, SslResponse::Refuse);
}

#[test]
fn bind_writer_parity_with_parse() {
    use bytes::Bytes;
    let upstream = Bind::new(
        Some("portal-1".to_owned()),
        Some("stmt-1".to_owned()),
        vec![0i16],
        vec![Some(Bytes::from_static(b"42")), None],
        vec![0i16],
    );
    let upstream_bytes = upstream_encode(&upstream);

    let mut our_buf = vec![0u8; 256];
    let written = {
        let mut writer =
            BindWriter::begin(&mut our_buf, b"portal-1", b"stmt-1", &[FormatCode::Text])
                .expect("begin must succeed");
        writer.parameter(Some(b"42")).expect("param 1 must succeed");
        writer.parameter(None).expect("null param must succeed");
        writer
            .finish(&[FormatCode::Text])
            .expect("finish must succeed")
    };

    assert_eq!(
        &our_buf[..written],
        &upstream_bytes[..],
        "BindWriter byte parity with upstream"
    );
}

#[test]
fn ssl_response_encode_decode_upstream_parity() {
    use pgwire::messages::response::SslResponse as UpstreamSslResponse;
    let upstream = UpstreamSslResponse::Accept;
    let upstream_bytes = upstream_encode(&upstream);

    assert_eq!(upstream_bytes.len(), 1);
    assert_eq!(upstream_bytes[0], b'S');

    let (our, _) = parse_ssl_response(&upstream_bytes)
        .expect("parse must succeed")
        .expect("must be complete");
    assert_eq!(our, SslResponse::Accept);
}

#[test]
fn encode_copy_in_response_helper_parity() {
    let upstream = CopyInResponse::new(0, 2, vec![0i16, 1]);
    let upstream_bytes = upstream_encode(&upstream);

    let mut our_buf = vec![0u8; 64];
    let written = encode_copy_in_response(
        &mut our_buf,
        CopyFormat::Text,
        &[FormatCode::Text, FormatCode::Binary],
    )
    .expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn encode_copy_out_response_helper_parity() {
    let upstream = CopyOutResponse::new(0, 1, vec![0i16]);
    let upstream_bytes = upstream_encode(&upstream);

    let mut our_buf = vec![0u8; 64];
    let written = encode_copy_out_response(&mut our_buf, CopyFormat::Text, &[FormatCode::Text])
        .expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}

#[test]
fn encode_copy_both_response_helper_parity() {
    let upstream = CopyBothResponse::new(0, 1, vec![0i16]);
    let upstream_bytes = upstream_encode(&upstream);

    let mut our_buf = vec![0u8; 64];
    let written = encode_copy_both_response(&mut our_buf, CopyFormat::Text, &[FormatCode::Text])
        .expect("encode must succeed");
    assert_eq!(&our_buf[..written], &upstream_bytes[..]);
}
