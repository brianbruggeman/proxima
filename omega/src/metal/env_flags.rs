//! Debug and diagnostic switches the metal driver reads from the process
//! environment, each read once per process.
//!
//! `std::env::var_os` takes the environ lock and scans it linearly
//! (`__findenv_locked`); the driver asks per bound buffer, per dispatch and
//! per command buffer, so the lookup is cached in a [`OnceLock`] per switch.
//! A switch therefore holds the value it had at its first read; to toggle one
//! inside a single process (the capture and kind-filter switches), read the
//! environment at the call site instead -- those are deliberately not here.
//! The sizing and config path (`crate::sized`, `omega-runtime.toml`) carries
//! build-time constants only, so a runtime debug switch does not belong in it.

#[cfg(feature = "instrument")]
use std::ffi::OsString;
use std::sync::OnceLock;

use proxima_tensor::NodeId;

fn is_present(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

fn is_one(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|value| value == "1")
}

#[cfg(feature = "instrument")]
fn node_ids(raw: &str) -> Vec<NodeId> {
    raw.split(',')
        .filter_map(|entry| entry.trim().parse::<u32>().ok())
        .map(NodeId)
        .collect()
}

pub(super) fn segment_host() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| is_present("PROXIMA_DEBUG_SEGMENT_HOST"))
}

pub(super) fn encoder_error_status() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| is_one("PROXIMA_METAL_ENCODER_ERROR_STATUS"))
}

pub(super) fn debug_metal_source() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| is_present("PROXIMA_DEBUG_METAL_SOURCE"))
}

pub(super) fn debug_placement_keys() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| is_present("PROXIMA_DEBUG_PLACEMENT_KEYS"))
}

pub(super) fn boundaries_prefill() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| is_present("PROXIMA_BOUNDARIES_PREFILL"))
}

#[cfg(feature = "instrument")]
pub(super) fn head_sentinel_fill() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| is_present("PROXIMA_HEAD_SENTINEL_FILL"))
}

#[cfg(feature = "instrument")]
pub(super) fn nan_check() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| is_present("PROXIMA_METAL_NAN_CHECK"))
}

pub(super) fn debug_expert_emit() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| is_present("PROXIMA_DEBUG_EXPERT_EMIT"))
}

#[cfg(feature = "instrument")]
pub(super) fn debug_metal_stages() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| is_present("PROXIMA_DEBUG_METAL_STAGES"))
}

#[cfg(feature = "instrument")]
pub(super) fn repeat_verify() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| is_present("PROXIMA_REPEAT_VERIFY"))
}

#[cfg(feature = "instrument")]
pub(super) fn substitute_blit_after_kernel() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("PROXIMA_SUBSTITUTE_MODE").as_deref() == Ok("blit-after-kernel")
    })
}

#[cfg(feature = "instrument")]
pub(super) fn substitute_dump_dir() -> Option<&'static OsString> {
    static CELL: OnceLock<Option<OsString>> = OnceLock::new();
    CELL.get_or_init(|| std::env::var_os("PROXIMA_SUBSTITUTE_DUMP_DIR")).as_ref()
}

pub(super) fn compare_bound_node() -> Option<NodeId> {
    static CELL: OnceLock<Option<NodeId>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .map(NodeId)
    })
}

#[cfg(feature = "instrument")]
pub(super) fn repeat_nodes() -> &'static [NodeId] {
    static CELL: OnceLock<Vec<NodeId>> = OnceLock::new();
    CELL.get_or_init(|| {
        std::env::var("PROXIMA_REPEAT_NODES")
            .map(|value| node_ids(&value))
            .unwrap_or_default()
    })
}

#[cfg(feature = "instrument")]
pub(super) fn repeat_count() -> u32 {
    static CELL: OnceLock<u32> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("PROXIMA_REPEAT_COUNT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presence_switch_is_on_for_any_value_including_empty() {
        let name = "PROXIMA_DEBUG_SEGMENT_HOST";
        temp_env::with_var(name, Some("1"), || assert!(is_present(name)));
        temp_env::with_var(name, Some(""), || assert!(is_present(name)));
        temp_env::with_var(name, None::<&str>, || assert!(!is_present(name)));
    }

    #[test]
    fn value_switch_is_on_only_for_one() {
        let name = "PROXIMA_METAL_ENCODER_ERROR_STATUS";
        temp_env::with_var(name, Some("1"), || assert!(is_one(name)));
        temp_env::with_var(name, Some("0"), || assert!(!is_one(name)));
        temp_env::with_var(name, None::<&str>, || assert!(!is_one(name)));
    }

    #[cfg(feature = "instrument")]
    #[test]
    fn node_list_keeps_the_parsable_entries_in_order() {
        let parsed = node_ids("2994, 3009,not-a-node,,7");
        assert_eq!(parsed, vec![NodeId(2994), NodeId(3009), NodeId(7)]);
    }

    #[test]
    fn segment_host_reads_the_environment_once_per_process() {
        let first = temp_env::with_var("PROXIMA_DEBUG_SEGMENT_HOST", Some("1"), segment_host);
        let second = temp_env::with_var("PROXIMA_DEBUG_SEGMENT_HOST", None::<&str>, segment_host);
        assert_eq!(first, second, "the second read must serve the first read's value");
    }
}
