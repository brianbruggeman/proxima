#![cfg(all(feature = "serve-prime", feature = "http-prime-deps"))]

#[path = "../examples/support/windows_port.rs"]
mod support;

#[proxima::test]
async fn windows_port_default_runtime_tcp_payload() {
    support::incumbent_exchange(false).expect("standard library payload oracle");
    assert!(
        support::incumbent_exchange(true).is_err(),
        "dropped byte must fail payload oracle"
    );
    support::tcp_exchange().expect("default runtime exchanges exact TCP payload");
    #[cfg(feature = "http1-native")]
    temp_env::async_with_vars([("PROXIMA_HTTP_HANDLER_SPREAD", Some("0"))], async {
        support::multiworker_listener_exchange()
            .await
            .expect("configured HTTP listener uses two-worker runtime");
    })
    .await;
}

#[test]
fn windows_port_default_runtime_udp_payload() {
    support::udp_exchange().expect("default runtime exchanges exact UDP payload");
}
