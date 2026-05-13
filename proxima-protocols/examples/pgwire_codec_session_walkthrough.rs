#![allow(clippy::expect_used)]

use proxima_protocols::pgwire_codec::frontend::{
    FrontendMessage, InitialMessage, parse_frontend, parse_initial,
};
use proxima_protocols::pgwire_codec::session::{AuthFlow, Session, StateName};
use proxima_protocols::pgwire_codec::types::{ProtocolVersion, StatementTarget};

fn step(label: &str, before: StateName, after: StateName, session: &Session) {
    println!(
        "{before:?} -> {label} -> {after:?} (wire_phase={:?})",
        session.wire_phase()
    );
}

fn feed_initial(session: &mut Session, bytes: &[u8]) {
    let before = session.state_name();
    let (msg, _) = parse_initial(bytes)
        .expect("parse_initial succeeds")
        .expect("complete frame");
    let disposition = session.on_initial(&msg).expect("on_initial accepted");
    step(
        &format!("{:?} -> {disposition:?}", msg_initial_name(&msg)),
        before,
        session.state_name(),
        session,
    );
}

fn feed_frontend(session: &mut Session, bytes: &[u8]) {
    let before = session.state_name();
    let (msg, _) = parse_frontend(bytes)
        .expect("parse_frontend succeeds")
        .expect("complete frame");
    let disposition = session.on_frontend(&msg).expect("on_frontend accepted");
    step(
        &format!("{} -> {disposition:?}", msg_tag_name(msg.tag())),
        before,
        session.state_name(),
        session,
    );
}

fn msg_initial_name(msg: &InitialMessage<'_>) -> &'static str {
    match msg {
        InitialMessage::SslRequest => "SslRequest",
        InitialMessage::GssEncRequest => "GssEncRequest",
        InitialMessage::Startup(_) => "Startup",
        InitialMessage::Cancel(_) => "Cancel",
    }
}

fn msg_tag_name(tag: u8) -> &'static str {
    match tag {
        b'p' => "AuthData",
        b'Q' => "Query",
        b'P' => "Parse",
        b'B' => "Bind",
        b'D' => "Describe",
        b'E' => "Execute",
        b'C' => "Close",
        b'H' => "Flush",
        b'S' => "Sync",
        b'd' => "CopyData",
        b'c' => "CopyDone",
        b'f' => "CopyFail",
        b'F' => "FunctionCall",
        b'X' => "Terminate",
        _ => "Unknown",
    }
}

fn ssl_request_bytes() -> [u8; 8] {
    let mut buf = [0u8; 8];
    InitialMessage::SslRequest
        .encode(&mut buf)
        .expect("ssl request encodes");
    buf
}

fn startup_bytes() -> Vec<u8> {
    let params: &[u8] = b"user\0walker\0database\0demo\0\0";
    let version = ProtocolVersion::V3_0.as_code().to_be_bytes();
    let length = (4 + 4 + params.len()) as i32;
    let mut buf = Vec::with_capacity(8 + params.len());
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(&version);
    buf.extend_from_slice(params);
    buf
}

fn cancel_bytes() -> Vec<u8> {
    let mut buf = vec![0u8; 16];
    let msg = InitialMessage::Cancel(proxima_protocols::pgwire_codec::frontend::CancelRequest {
        process_id: 7,
        secret_key: &[0xde, 0xad, 0xbe, 0xef],
    });
    let size = msg.encode(&mut buf).expect("cancel encodes");
    buf.truncate(size);
    buf
}

fn auth_data_bytes(data: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; 5 + data.len()];
    let msg = FrontendMessage::AuthData(proxima_protocols::pgwire_codec::frontend::AuthData { data });
    let size = msg.encode(&mut buf).expect("auth data encodes");
    buf.truncate(size);
    buf
}

fn query_bytes(sql: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; 5 + sql.len() + 1];
    let msg = FrontendMessage::Query {
        sql: proxima_protocols::pgwire_codec::types::PgStr::new(sql),
    };
    let size = msg.encode(&mut buf).expect("query encodes");
    buf.truncate(size);
    buf
}

fn parse_msg_bytes() -> Vec<u8> {
    let statement: &[u8] = b"\x00";
    let sql: &[u8] = b"SELECT $1\x00";
    let param_count: &[u8] = &[0x00, 0x00];
    let body_len = statement.len() + sql.len() + param_count.len();
    let length = (4 + body_len) as i32;
    let mut buf = Vec::with_capacity(5 + body_len);
    buf.push(b'P');
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(statement);
    buf.extend_from_slice(sql);
    buf.extend_from_slice(param_count);
    buf
}

fn bind_bytes() -> Vec<u8> {
    let mut buf = vec![0u8; 128];
    let size = proxima_protocols::pgwire_codec::frontend::BindWriter::begin(&mut buf, b"", b"", &[])
        .expect("bind writer begins")
        .finish(&[])
        .expect("bind writer finishes");
    buf.truncate(size);
    buf
}

fn describe_bytes() -> Vec<u8> {
    let mut buf = vec![0u8; 16];
    let msg = FrontendMessage::Describe {
        target: StatementTarget::Statement,
        name: proxima_protocols::pgwire_codec::types::PgStr::new(b""),
    };
    let size = msg.encode(&mut buf).expect("describe encodes");
    buf.truncate(size);
    buf
}

fn execute_bytes() -> Vec<u8> {
    let mut buf = vec![0u8; 16];
    let msg = FrontendMessage::Execute {
        portal: proxima_protocols::pgwire_codec::types::PgStr::new(b""),
        max_rows: 0,
    };
    let size = msg.encode(&mut buf).expect("execute encodes");
    buf.truncate(size);
    buf
}

fn flush_bytes() -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    let size = FrontendMessage::Flush
        .encode(&mut buf)
        .expect("flush encodes");
    buf.truncate(size);
    buf
}

fn sync_bytes() -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    let size = FrontendMessage::Sync
        .encode(&mut buf)
        .expect("sync encodes");
    buf.truncate(size);
    buf
}

fn copy_data_bytes(data: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; 5 + data.len()];
    let msg = FrontendMessage::CopyData { data };
    let size = msg.encode(&mut buf).expect("copy data encodes");
    buf.truncate(size);
    buf
}

fn copy_done_bytes() -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    let size = FrontendMessage::CopyDone
        .encode(&mut buf)
        .expect("copy done encodes");
    buf.truncate(size);
    buf
}

fn copy_fail_bytes() -> Vec<u8> {
    let message = b"client abort";
    let mut buf = vec![0u8; 5 + message.len() + 1];
    let msg = FrontendMessage::CopyFail {
        message: proxima_protocols::pgwire_codec::types::PgStr::new(message),
    };
    let size = msg.encode(&mut buf).expect("copy fail encodes");
    buf.truncate(size);
    buf
}

fn function_call_bytes() -> Vec<u8> {
    let oid: u32 = 1247;
    let body_len: i32 = 4 + 2 + 2 + 2;
    let length: i32 = 4 + body_len;
    let mut buf = Vec::with_capacity(5 + body_len as usize);
    buf.push(b'F');
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(&oid.to_be_bytes());
    buf.extend_from_slice(&0i16.to_be_bytes());
    buf.extend_from_slice(&0i16.to_be_bytes());
    buf.extend_from_slice(&0i16.to_be_bytes());
    buf
}

fn terminate_bytes() -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    let size = FrontendMessage::Terminate
        .encode(&mut buf)
        .expect("terminate encodes");
    buf.truncate(size);
    buf
}

fn server_transition(session: &mut Session, action: &str) {
    let before = session.state_name();
    match action {
        "ssl_refused" => session.ssl_refused().expect("ssl_refused accepted"),
        "ssl_accepted" => session.ssl_accepted().expect("ssl_accepted accepted"),
        "tls_established" => session.tls_established().expect("tls_established accepted"),
        "auth_cleartext" => session
            .auth_requested(AuthFlow::Cleartext)
            .expect("auth_requested accepted"),
        "auth_ok" => session.auth_ok().expect("auth_ok accepted"),
        "ready_for_query" => {
            let status = session.ready_for_query().expect("ready_for_query accepted");
            let after = session.state_name();
            println!(
                "{before:?} -> server:ready_for_query(status={status:?}) -> {after:?} (wire_phase={:?})",
                session.wire_phase()
            );
            return;
        }
        "extended_error" => session.extended_error().expect("extended_error accepted"),
        "copy_in_begun" => session.copy_in_begun().expect("copy_in_begun accepted"),
        "copy_out_begun" => session.copy_out_begun().expect("copy_out_begun accepted"),
        "copy_both_begun" => session.copy_both_begun().expect("copy_both_begun accepted"),
        "copy_finished" => session.copy_finished().expect("copy_finished accepted"),
        _ => panic!("unknown action: {action}"),
    }
    let after = session.state_name();
    println!(
        "{before:?} -> server:{action} -> {after:?} (wire_phase={:?})",
        session.wire_phase()
    );
}

fn scenario_header(label: &str) {
    println!("\n{label}");
}

fn scenario_a() {
    scenario_header("(a) ssl-refused cleartext auth + simple query + terminate");
    let mut session = Session::new();

    feed_initial(&mut session, &ssl_request_bytes());
    server_transition(&mut session, "ssl_refused");
    feed_initial(&mut session, &startup_bytes());
    server_transition(&mut session, "auth_cleartext");
    feed_frontend(&mut session, &auth_data_bytes(b"hunter2\0"));
    server_transition(&mut session, "auth_ok");
    server_transition(&mut session, "ready_for_query");

    feed_frontend(&mut session, &query_bytes(b"SELECT version()"));
    server_transition(&mut session, "ready_for_query");

    feed_frontend(&mut session, &terminate_bytes());
    println!("connection closed: is_closed={}", session.is_closed());
}

fn scenario_b() {
    scenario_header("(b) tls-accepted path");
    let mut session = Session::new();

    feed_initial(&mut session, &ssl_request_bytes());
    server_transition(&mut session, "ssl_accepted");
    server_transition(&mut session, "tls_established");
    feed_initial(&mut session, &startup_bytes());
    server_transition(&mut session, "auth_ok");
    server_transition(&mut session, "ready_for_query");

    println!("tls session idle: state={:?}", session.state_name());

    let second_ssl = ssl_request_bytes();
    let (msg, _) = parse_initial(&second_ssl)
        .expect("parse ok")
        .expect("complete");
    let result = session.on_initial(&msg);
    println!("second SSLRequest inside tunnel: {:?}", result);
}

fn scenario_c() {
    scenario_header("(c) extended pipeline with error recovery");
    let mut session = Session::new();
    feed_initial(&mut session, &startup_bytes());
    server_transition(&mut session, "auth_ok");
    server_transition(&mut session, "ready_for_query");

    feed_frontend(&mut session, &parse_msg_bytes());
    feed_frontend(&mut session, &bind_bytes());
    feed_frontend(&mut session, &describe_bytes());
    server_transition(&mut session, "extended_error");

    feed_frontend(&mut session, &execute_bytes());
    feed_frontend(&mut session, &flush_bytes());
    feed_frontend(&mut session, &sync_bytes());
    server_transition(&mut session, "ready_for_query");
    println!("recovered to idle: state={:?}", session.state_name());
}

fn scenario_d() {
    scenario_header("(d) copy-in and copy-out");
    let mut session = Session::new();
    feed_initial(&mut session, &startup_bytes());
    server_transition(&mut session, "auth_ok");
    server_transition(&mut session, "ready_for_query");

    feed_frontend(&mut session, &query_bytes(b"COPY t FROM STDIN"));
    server_transition(&mut session, "copy_in_begun");
    feed_frontend(&mut session, &copy_data_bytes(b"row1\n"));
    feed_frontend(&mut session, &copy_data_bytes(b"row2\n"));
    feed_frontend(&mut session, &copy_done_bytes());
    server_transition(&mut session, "ready_for_query");
    println!("after copy-in: state={:?}", session.state_name());

    feed_frontend(&mut session, &query_bytes(b"COPY t TO STDOUT"));
    server_transition(&mut session, "copy_out_begun");
    feed_frontend(&mut session, &copy_fail_bytes());
    println!("copy_fail during copy-out is discarded (protocol note)");
    server_transition(&mut session, "copy_finished");
    server_transition(&mut session, "ready_for_query");
    println!("after copy-out: state={:?}", session.state_name());
}

fn scenario_e() {
    scenario_header("(e) function call");
    let mut session = Session::new();
    feed_initial(&mut session, &startup_bytes());
    server_transition(&mut session, "auth_ok");
    server_transition(&mut session, "ready_for_query");

    feed_frontend(&mut session, &function_call_bytes());
    server_transition(&mut session, "ready_for_query");
    println!("after function call: state={:?}", session.state_name());
}

fn scenario_f() {
    scenario_header("(f) cancel connection");
    let mut session = Session::new();
    feed_initial(&mut session, &cancel_bytes());
    println!(
        "cancel: state={:?} wire_phase={:?} is_closed={}",
        session.state_name(),
        session.wire_phase(),
        session.is_closed()
    );
}

fn main() {
    println!("proxima-pgwire-codec session FSM walkthrough");
    println!("format: before_state -> message/action -> after_state (wire_phase)");

    scenario_a();
    scenario_b();
    scenario_c();
    scenario_d();
    scenario_e();
    scenario_f();

    println!("\nwalkthrough complete");
}
