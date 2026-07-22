// one binary instead of thirty avoids linking the whole workspace thirty times.
#[path = "e2e/agnostic_http_spread.rs"]
mod agnostic_http_spread;
#[path = "e2e/end_to_end.rs"]
mod end_to_end;
#[path = "e2e/example_nginx_multi_smoke.rs"]
mod example_nginx_multi_smoke;
#[path = "e2e/example_nginx_style_smoke.rs"]
mod example_nginx_style_smoke;
#[path = "e2e/io_uring_prime_smoke.rs"]
mod io_uring_prime_smoke;
#[path = "e2e/listener_client_interop.rs"]
mod listener_client_interop;
#[path = "e2e/listener_expect_continue.rs"]
mod listener_expect_continue;
#[path = "e2e/listener_h2.rs"]
mod listener_h2;
#[path = "e2e/listener_h2_native.rs"]
mod listener_h2_native;
#[path = "e2e/listener_h3.rs"]
mod listener_h3;
#[path = "e2e/listener_h3_native.rs"]
mod listener_h3_native;
#[path = "e2e/listener_pgwire_native.rs"]
mod listener_pgwire_native;
#[path = "e2e/listener_preface_dispatch.rs"]
mod listener_preface_dispatch;
#[path = "e2e/listener_streaming.rs"]
mod listener_streaming;
#[path = "e2e/listener_streaming_iouring.rs"]
mod listener_streaming_iouring;
#[path = "e2e/listener_tls.rs"]
mod listener_tls;
#[path = "e2e/listener_tls_iouring.rs"]
mod listener_tls_iouring;
#[path = "e2e/listener_trailers.rs"]
mod listener_trailers;
#[path = "e2e/listener_upgrade.rs"]
mod listener_upgrade;
#[path = "e2e/listener_upgrade_iouring.rs"]
mod listener_upgrade_iouring;
#[path = "e2e/locald_replacement.rs"]
mod locald_replacement;
#[path = "e2e/macos_spread_demo.rs"]
mod macos_spread_demo;
#[path = "e2e/native_upstream_socket.rs"]
mod native_upstream_socket;
#[path = "e2e/open_loop_scenario.rs"]
mod open_loop_scenario;
#[path = "e2e/prime_h2_spawn_blocking_repro.rs"]
mod prime_h2_spawn_blocking_repro;
#[path = "e2e/prime_serve.rs"]
mod prime_serve;
#[path = "e2e/proxima_test_smoke.rs"]
mod proxima_test_smoke;
#[path = "e2e/ptu_https_wire.rs"]
mod ptu_https_wire;
#[path = "e2e/recording_streaming.rs"]
mod recording_streaming;
#[path = "e2e/run_until_signal_ready.rs"]
mod run_until_signal_ready;
#[path = "e2e/runtime_conformance.rs"]
mod runtime_conformance;
#[path = "e2e/server_fluent.rs"]
mod server_fluent;
#[path = "e2e/shutdown_barrier.rs"]
mod shutdown_barrier;
#[path = "e2e/wire_back_compat.rs"]
mod wire_back_compat;
