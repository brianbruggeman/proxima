use std::env;
use std::fs;
use std::path::PathBuf;

use toml::Value;

fn require_pow2(name: &str, value: i64) -> usize {
    let value = usize::try_from(value)
        .unwrap_or_else(|_| panic!("{name} must be a non-negative integer; got {value}"));
    assert!(
        value > 0 && value.is_power_of_two(),
        "{name} must be a non-zero power of two; got {value}"
    );
    value
}

fn require_nonzero(name: &str, value: i64) -> usize {
    let value = usize::try_from(value)
        .unwrap_or_else(|_| panic!("{name} must be a non-negative integer; got {value}"));
    assert!(value > 0, "{name} must be non-zero");
    value
}

fn require_usize(name: &str, value: i64) -> usize {
    usize::try_from(value)
        .unwrap_or_else(|_| panic!("{name} must be a non-negative integer; got {value}"))
}

fn get_int(table: &Value, section: &str, key: &str) -> i64 {
    table
        .get(section)
        .and_then(|section_value| section_value.get(key))
        .and_then(Value::as_integer)
        .unwrap_or_else(|| panic!("prime-runtime.toml: missing or non-integer [{section}].{key}"))
}

/// like `get_int`, but a `PRIME_<SECTION>_<KEY>` env var overrides the TOML
/// value when present — the per-key build-time override pattern used by
/// proxima-net-xdp's build.rs.
fn resolve_int(table: &Value, section: &str, key: &str) -> i64 {
    let env_name = format!(
        "PRIME_{section}_{key}",
        section = section.to_uppercase(),
        key = key.to_uppercase()
    );
    println!("cargo:rerun-if-env-changed={env_name}");
    match env::var(&env_name) {
        Ok(raw) => raw
            .parse::<i64>()
            .unwrap_or_else(|err| panic!("{env_name}={raw} must parse as i64: {err}")),
        Err(_) => get_int(table, section, key),
    }
}

#[allow(clippy::expect_used)]
fn emit_sizing_consts(out_dir: &std::path::Path) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
    let toml_path = PathBuf::from(&manifest_dir).join("prime-runtime.toml");
    println!("cargo:rerun-if-changed=prime-runtime.toml");

    let text = fs::read_to_string(&toml_path)
        .unwrap_or_else(|err| panic!("read {}: {err}", toml_path.display()));
    let root: Value = text
        .parse()
        .unwrap_or_else(|err| panic!("parse {}: {err}", toml_path.display()));

    let inbox_capacity = require_pow2("inbox.capacity", get_int(&root, "inbox", "capacity"));
    let inbox_lanes_per_core = require_nonzero(
        "inbox.lanes_per_core",
        get_int(&root, "inbox", "lanes_per_core"),
    );
    let inbox_lanes_headroom = require_usize(
        "inbox.lanes_headroom",
        get_int(&root, "inbox", "lanes_headroom"),
    );
    let timer_bottom_slots = require_pow2(
        "timer.bottom_slots",
        get_int(&root, "timer", "bottom_slots"),
    );
    let timer_upper_slots =
        require_pow2("timer.upper_slots", get_int(&root, "timer", "upper_slots"));
    let timer_slot_inline =
        require_nonzero("timer.slot_inline", get_int(&root, "timer", "slot_inline"));
    let task_slab_initial_cap = require_nonzero(
        "executor.task_slab_initial_cap",
        get_int(&root, "executor", "task_slab_initial_cap"),
    );
    let run_queue_capacity = require_pow2(
        "executor.run_queue_capacity",
        get_int(&root, "executor", "run_queue_capacity"),
    );
    let reactor_slab_initial_cap = require_nonzero(
        "reactor.slab_initial_cap",
        get_int(&root, "reactor", "slab_initial_cap"),
    );
    // spin-then-park drive policy (worker_main's idle loop, core_shard.rs).
    // env-overridable per target: bare metal wants BUSY=0 (never spin, park
    // immediately); HFT-style low-latency wants a larger BUSY to hide the
    // syscall on bursty producer traffic. both BUSY and IDLE may legitimately
    // be 0 (see prime-runtime.toml doc), so require_usize (not
    // require_nonzero) — the only hard constraint is "parses as a
    // non-negative integer."
    let reactor_spin_before_park_busy = require_usize(
        "reactor.spin_before_park_busy",
        resolve_int(&root, "reactor", "spin_before_park_busy"),
    );
    let reactor_spin_before_park_idle = require_usize(
        "reactor.spin_before_park_idle",
        resolve_int(&root, "reactor", "spin_before_park_idle"),
    );
    let reactor_idle_park_threshold = require_usize(
        "reactor.idle_park_threshold",
        resolve_int(&root, "reactor", "idle_park_threshold"),
    );
    let io_uring_buf_size = require_nonzero(
        "io_uring.buf_size",
        resolve_int(&root, "io_uring", "buf_size"),
    );
    let bg_pool_default_threads = require_usize(
        "background_pool.default_threads",
        get_int(&root, "background_pool", "default_threads"),
    );

    let quic_max_bidi = require_nonzero(
        "quic.max_concurrent_bidi_streams",
        get_int(&root, "quic", "max_concurrent_bidi_streams"),
    );
    let quic_max_uni = require_nonzero(
        "quic.max_concurrent_uni_streams",
        get_int(&root, "quic", "max_concurrent_uni_streams"),
    );
    let quic_recv_window = require_nonzero(
        "quic.recv_window_bytes",
        get_int(&root, "quic", "recv_window_bytes"),
    );
    let quic_send_window = require_nonzero(
        "quic.send_window_bytes",
        get_int(&root, "quic", "send_window_bytes"),
    );
    let quic_max_idle_timeout_ms = require_nonzero(
        "quic.max_idle_timeout_ms",
        get_int(&root, "quic", "max_idle_timeout_ms"),
    );
    let quic_initial_mtu_bytes = require_nonzero(
        "quic.initial_mtu_bytes",
        get_int(&root, "quic", "initial_mtu_bytes"),
    );
    let quic_max_udp_payload_size = require_nonzero(
        "quic.max_udp_payload_size",
        get_int(&root, "quic", "max_udp_payload_size"),
    );
    let quic_initial_rtt_ms = require_nonzero(
        "quic.initial_rtt_ms",
        get_int(&root, "quic", "initial_rtt_ms"),
    );
    let quic_ack_delay_exponent = require_usize(
        "quic.ack_delay_exponent",
        get_int(&root, "quic", "ack_delay_exponent"),
    );
    let quic_max_ack_delay_ms = require_nonzero(
        "quic.max_ack_delay_ms",
        get_int(&root, "quic", "max_ack_delay_ms"),
    );
    let quic_active_cid_limit = require_nonzero(
        "quic.active_connection_id_limit",
        get_int(&root, "quic", "active_connection_id_limit"),
    );
    let quic_max_ack_ranges = require_nonzero(
        "quic.max_ack_ranges",
        get_int(&root, "quic", "max_ack_ranges"),
    );
    let quic_sent_packets_cap = require_nonzero(
        "quic.sent_packets_cap",
        get_int(&root, "quic", "sent_packets_cap"),
    );
    let quic_dcid_table_cap = require_nonzero(
        "quic.dcid_table_cap",
        get_int(&root, "quic", "dcid_table_cap"),
    );
    let quic_datagram_send_queue_cap = require_nonzero(
        "quic.datagram_send_queue_cap",
        get_int(&root, "quic", "datagram_send_queue_cap"),
    );
    let quic_datagram_recv_queue_cap = require_nonzero(
        "quic.datagram_recv_queue_cap",
        get_int(&root, "quic", "datagram_recv_queue_cap"),
    );
    let quic_max_paths_per_connection = require_nonzero(
        "quic.max_paths_per_connection",
        get_int(&root, "quic", "max_paths_per_connection"),
    );
    let h3_max_concurrent_requests = require_nonzero(
        "h3.max_concurrent_requests",
        get_int(&root, "h3", "max_concurrent_requests"),
    );
    let h3_qpack_max_table_capacity = require_nonzero(
        "h3.qpack_max_table_capacity",
        get_int(&root, "h3", "qpack_max_table_capacity"),
    );
    let h3_qpack_blocked_streams = require_nonzero(
        "h3.qpack_blocked_streams",
        get_int(&root, "h3", "qpack_blocked_streams"),
    );
    let h3_max_field_section_size = require_nonzero(
        "h3.max_field_section_size",
        get_int(&root, "h3", "max_field_section_size"),
    );
    // inverted-compat worker tuning ("the split" knobs). compile-time defaults
    // for tiers that do not load the conflaguration runtime config; std/alloc
    // can override the MODE via PRIME_COMPAT, but these counts stay baked.
    let compat_sister_drive_yields = require_nonzero(
        "compat.sister_drive_yields",
        get_int(&root, "compat", "sister_drive_yields"),
    );
    let compat_park_timeout_ms = require_nonzero(
        "compat.park_timeout_ms",
        get_int(&root, "compat", "park_timeout_ms"),
    );

    let out = format!(
        "// AUTO-GENERATED by build.rs from prime-runtime.toml. DO NOT EDIT.\n\
         pub const INBOX_CAPACITY: usize = {inbox_capacity};\n\
         pub const INBOX_LANES_PER_CORE: usize = {inbox_lanes_per_core};\n\
         pub const INBOX_LANES_HEADROOM: usize = {inbox_lanes_headroom};\n\
         pub const TIMER_BOTTOM_SLOTS: usize = {timer_bottom_slots};\n\
         pub const TIMER_UPPER_SLOTS: usize = {timer_upper_slots};\n\
         pub const TIMER_SLOT_INLINE: usize = {timer_slot_inline};\n\
         pub const TASK_SLAB_INITIAL_CAP: usize = {task_slab_initial_cap};\n\
         pub const RUN_QUEUE_CAPACITY: usize = {run_queue_capacity};\n\
         pub const REACTOR_SLAB_INITIAL_CAP: usize = {reactor_slab_initial_cap};\n\
         pub const REACTOR_SPIN_BEFORE_PARK_BUSY: u32 = {reactor_spin_before_park_busy};\n\
         pub const REACTOR_SPIN_BEFORE_PARK_IDLE: u32 = {reactor_spin_before_park_idle};\n\
         pub const REACTOR_IDLE_PARK_THRESHOLD: u32 = {reactor_idle_park_threshold};\n\
         pub const IO_URING_BUF_SIZE: usize = {io_uring_buf_size};\n\
         pub const BG_POOL_DEFAULT_THREADS: usize = {bg_pool_default_threads};\n\
         pub const PROXIMA_QUIC_MAX_CONCURRENT_BIDI_STREAMS: usize = {quic_max_bidi};\n\
         pub const PROXIMA_QUIC_MAX_CONCURRENT_UNI_STREAMS: usize = {quic_max_uni};\n\
         pub const PROXIMA_QUIC_RECV_WINDOW_BYTES: usize = {quic_recv_window};\n\
         pub const PROXIMA_QUIC_SEND_WINDOW_BYTES: usize = {quic_send_window};\n\
         pub const PROXIMA_QUIC_MAX_IDLE_TIMEOUT_MS: usize = {quic_max_idle_timeout_ms};\n\
         pub const PROXIMA_QUIC_INITIAL_MTU_BYTES: usize = {quic_initial_mtu_bytes};\n\
         pub const PROXIMA_QUIC_MAX_UDP_PAYLOAD_SIZE: usize = {quic_max_udp_payload_size};\n\
         pub const PROXIMA_QUIC_INITIAL_RTT_MS: usize = {quic_initial_rtt_ms};\n\
         pub const PROXIMA_QUIC_ACK_DELAY_EXPONENT: usize = {quic_ack_delay_exponent};\n\
         pub const PROXIMA_QUIC_MAX_ACK_DELAY_MS: usize = {quic_max_ack_delay_ms};\n\
         pub const PROXIMA_QUIC_ACTIVE_CONNECTION_ID_LIMIT: usize = {quic_active_cid_limit};\n\
         pub const PROXIMA_QUIC_MAX_ACK_RANGES: usize = {quic_max_ack_ranges};\n\
         pub const PROXIMA_QUIC_SENT_PACKETS_CAP: usize = {quic_sent_packets_cap};\n\
         pub const PROXIMA_QUIC_DCID_TABLE_CAP: usize = {quic_dcid_table_cap};\n\
         pub const PROXIMA_QUIC_DATAGRAM_SEND_QUEUE_CAP: usize = {quic_datagram_send_queue_cap};\n\
         pub const PROXIMA_QUIC_DATAGRAM_RECV_QUEUE_CAP: usize = {quic_datagram_recv_queue_cap};\n\
         pub const PROXIMA_QUIC_MAX_PATHS_PER_CONNECTION: usize = {quic_max_paths_per_connection};\n\
         pub const PROXIMA_H3_MAX_CONCURRENT_REQUESTS: usize = {h3_max_concurrent_requests};\n\
         pub const PROXIMA_H3_QPACK_MAX_TABLE_CAPACITY: usize = {h3_qpack_max_table_capacity};\n\
         pub const PROXIMA_H3_QPACK_BLOCKED_STREAMS: usize = {h3_qpack_blocked_streams};\n\
         pub const PROXIMA_H3_MAX_FIELD_SECTION_SIZE: usize = {h3_max_field_section_size};\n\
         pub const COMPAT_SISTER_DRIVE_YIELDS: u32 = {compat_sister_drive_yields};\n\
         pub const COMPAT_PARK_TIMEOUT_MS: u64 = {compat_park_timeout_ms};\n",
    );

    let out_path = out_dir.join("proxima_runtime_sized.rs");
    fs::write(&out_path, out).unwrap_or_else(|err| panic!("write {}: {err}", out_path.display()));
}

#[allow(clippy::expect_used)]
fn main() {
    let out_dir_str = env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let out_dir = PathBuf::from(&out_dir_str);

    // sizing consts from prime-runtime.toml — always emitted regardless of profile
    emit_sizing_consts(&out_dir);

    // profile-driven consts + cfg directives — only when PROXIMA_PROFILE is set.
    // skipped silently when building without a profile (e.g. plain cargo build
    // without the profile wrapper). DC6 (xtask) makes PROXIMA_PROFILE mandatory.
    if env::var("PROXIMA_PROFILE").is_ok() {
        let resolved = proxima_build::resolve_profile().expect("resolve proxima profile");
        proxima_build::emit_generated_module(&resolved).expect("emit proxima_profile.rs");
        proxima_build::emit_cfg_directives(&resolved);
        proxima_build::emit_rerun_directives(&resolved);
    }
}
