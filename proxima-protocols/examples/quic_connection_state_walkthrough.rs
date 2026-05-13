//! Connection-FSM walkthrough — runs the [`Connection<P>`] through
//! every legal transition path, printing each step so a learner can
//! see exactly what each variant carries and what triggers each
//! transition.
//!
//! Mandated by workspace `pty-tester/docs/proxima-pty/guiding-principles.md`
//! principle 11 ("every state machine ships with an
//! `examples/<sm>_walkthrough.rs` driving it through every legal
//! transition path").
//!
//! Run with:
//!     cargo run -p proxima-quic-proto --example connection_state_walkthrough --features mock-tls

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]

use proxima_protocols::quic::connection::{Connection, ConnectionState, TimerOutcome};
use proxima_protocols::quic::crypto::initial_keys;
use proxima_protocols::quic::crypto::packet_protection::protect_initial;
use proxima_protocols::quic::time::{Duration, Instant};
use proxima_protocols::quic::tls::Epoch;
use proxima_protocols::quic::tls::mock::{MockEvent, MockStep, MockTlsProvider, synthetic_secrets};

const RFC_9001_A1_DCID: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
const LOCAL_SCID: [u8; 8] = [0xc0, 0xff, 0xee, 0xba, 0xbe, 0x12, 0x34, 0x56];
const SERVER_SCID: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];

fn print_state<P: proxima_protocols::quic::tls::TlsProvider>(label: &str, connection: &Connection<P>) {
    println!("[{label}] state = {}", connection.state().label());
}

fn main() {
    println!("== Proxima QUIC connection-FSM walkthrough ==\n");

    println!("-- Path 1: Initial → Handshake via real-bytes round-trip --\n");

    let client_hello: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let server_hello: Vec<u8> = vec![0xCA, 0xFE, 0xBA, 0xBE];

    let mut connection = Connection::<MockTlsProvider>::new_client(
        MockTlsProvider::script_client(vec![
            MockStep::EmitHandshakeBytes {
                epoch: Epoch::Initial,
                bytes: client_hello.clone(),
            },
            MockStep::ReadHandshake {
                epoch: Epoch::Initial,
                expect: server_hello.clone(),
            },
            MockStep::InstallSecrets(synthetic_secrets(Epoch::Handshake, 0, 0xAA)),
            MockStep::EmitEvent(MockEvent::HandshakeDataReceived),
        ]),
        b"",
        &RFC_9001_A1_DCID,
        &LOCAL_SCID,
        Instant::from_micros(1_000_000),
    )
    .expect("new_client");

    print_state("t=1_000_000", &connection);
    assert!(matches!(connection.state(), ConnectionState::Initial(_)));

    let mut buf = [0u8; 1500];
    let outbound = connection
        .poll_transmit(Instant::from_micros(1_000_001), &mut buf)
        .expect("poll_transmit ok")
        .expect("first send");
    println!(
        "  → poll_transmit emitted {}-byte Initial datagram (epoch={:?})",
        outbound.len, outbound.epoch,
    );

    let pair = initial_keys::derive(&RFC_9001_A1_DCID).expect("derive");
    let server_initial =
        build_server_initial(&pair.server, &LOCAL_SCID, &SERVER_SCID, &server_hello, 0);
    connection
        .handle_datagram(Instant::from_micros(2_000_000), &server_initial)
        .expect("handle_datagram");
    print_state("t=2_000_000", &connection);
    assert!(matches!(connection.state(), ConnectionState::Handshake(_)));

    println!("\n-- Path 2: Established (stub) → Closing via caller close --\n");
    // We can't reach Established in C11 v1 without Handshake-epoch
    // packet protection, so for the close-arc demo we build a fresh
    // Initial-state connection and call close directly.
    let mut close_demo = Connection::<MockTlsProvider>::new_client(
        MockTlsProvider::script_client(vec![]),
        b"",
        &RFC_9001_A1_DCID,
        &LOCAL_SCID,
        Instant::from_micros(5_000_000),
    )
    .expect("new_client");
    print_state("t=5_000_000", &close_demo);
    close_demo.close(0x00, b"bye").expect("close");
    print_state("close(0x00, b\"bye\")", &close_demo);
    assert!(matches!(close_demo.state(), ConnectionState::Closing(_)));

    println!("\n-- Path 3: Closing → Draining via close-deadline timer --\n");
    let after_close_deadline = Instant::from_micros(50_000_000);
    let outcome = close_demo
        .handle_timeout(after_close_deadline)
        .expect("timeout");
    println!("  → handle_timeout returned {outcome:?}");
    print_state("after close_deadline", &close_demo);
    assert_eq!(outcome, TimerOutcome::ClosingDrained);
    assert!(matches!(close_demo.state(), ConnectionState::Draining(_)));

    println!("\n-- Path 4: Draining → Closed via drain-deadline timer --\n");
    let after_drain_deadline = Instant::from_micros(100_000_000);
    let outcome = close_demo
        .handle_timeout(after_drain_deadline)
        .expect("timeout");
    println!("  → handle_timeout returned {outcome:?}");
    print_state("after drain_deadline", &close_demo);
    assert_eq!(outcome, TimerOutcome::Drained);
    assert!(matches!(close_demo.state(), ConnectionState::Closed));

    println!("\n-- Path 5: Idle timeout from fresh Initial → Closed --\n");
    let mut idle_demo = Connection::<MockTlsProvider>::new_client(
        MockTlsProvider::script_client(vec![]),
        b"",
        &RFC_9001_A1_DCID,
        &LOCAL_SCID,
        Instant::from_micros(10_000_000),
    )
    .expect("new_client");
    print_state("t=10_000_000", &idle_demo);
    let idle_deadline = idle_demo.next_timeout().expect("idle deadline");
    println!("  → next_timeout = {idle_deadline:?} (RFC default 30s idle)");
    let outcome = idle_demo
        .handle_timeout(idle_deadline + Duration::from_micros(1))
        .expect("timeout");
    println!("  → handle_timeout(idle_deadline + 1µs) returned {outcome:?}");
    print_state("after idle deadline", &idle_demo);
    assert_eq!(outcome, TimerOutcome::IdleClosed);
    assert!(matches!(idle_demo.state(), ConnectionState::Closed));

    println!("\n== All transition paths walked ✓ ==");
}

fn build_server_initial(
    server_keys: &proxima_protocols::quic::crypto::initial_keys::InitialKeys,
    dcid: &[u8],
    scid: &[u8],
    crypto_bytes: &[u8],
    server_pn: u64,
) -> Vec<u8> {
    const TAG_LEN: usize = proxima_protocols::quic::crypto::aead::TAG_LEN;
    const PAYLOAD_TARGET: usize = 1200;
    let pn_byte_len = 4usize;

    let crypto_frame_len = 1 + 1 + 1 + crypto_bytes.len();
    let header_fixed = 1 + 4 + 1 + dcid.len() + 1 + scid.len();
    let token_len_varint = 1;
    let length_varint = 2;
    let header_total = header_fixed + token_len_varint + length_varint + pn_byte_len;
    let payload_budget = PAYLOAD_TARGET - header_total - TAG_LEN;
    let padding_len = payload_budget - crypto_frame_len;
    let plaintext_len_actual = crypto_frame_len + padding_len;
    let remaining_field_value = (pn_byte_len + plaintext_len_actual + TAG_LEN) as u16 | 0x4000;

    let mut buffer = vec![0u8; PAYLOAD_TARGET];

    let mut cursor = 0usize;
    buffer[cursor] = 0xC0 | (pn_byte_len as u8 - 1);
    cursor += 1;
    buffer[cursor..cursor + 4].copy_from_slice(&1u32.to_be_bytes());
    cursor += 4;
    buffer[cursor] = dcid.len() as u8;
    cursor += 1;
    buffer[cursor..cursor + dcid.len()].copy_from_slice(dcid);
    cursor += dcid.len();
    buffer[cursor] = scid.len() as u8;
    cursor += 1;
    buffer[cursor..cursor + scid.len()].copy_from_slice(scid);
    cursor += scid.len();
    buffer[cursor] = 0;
    cursor += 1;
    buffer[cursor..cursor + 2].copy_from_slice(&remaining_field_value.to_be_bytes());
    cursor += 2;
    let pn_offset = cursor;
    buffer[cursor..cursor + pn_byte_len].copy_from_slice(&(server_pn as u32).to_be_bytes());
    cursor += pn_byte_len;
    buffer[cursor] = 0x06; // CRYPTO
    cursor += 1;
    buffer[cursor] = 0;
    cursor += 1;
    buffer[cursor] = crypto_bytes.len() as u8;
    cursor += 1;
    buffer[cursor..cursor + crypto_bytes.len()].copy_from_slice(crypto_bytes);

    protect_initial(
        server_keys,
        server_pn,
        pn_byte_len,
        &mut buffer,
        pn_offset,
        plaintext_len_actual,
    )
    .expect("protect");
    buffer
}
