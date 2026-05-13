//! resolved sizing for runtime-tunable primitives. compile-time consts from
//! `core::sized` are the default; when `runtime-prime-config` is enabled,
//! a TOML at `$PROXIMA_RUNTIME_CONFIG` (or `./proxima-runtime.toml`) can
//! override any subset of keys. missing keys fall back to compiled — the
//! schema is *identical* to the build-time TOML.
//!
//! const-generic types (no_alloc Inbox) cannot be overridden at runtime and
//! always use `core::sized::*` directly.

use super::super::core::sized;

/// resolved sizing knobs. `COMPILED` is the value emitted by build.rs;
/// `resolved()` returns those defaults unless the runtime-config feature
/// is enabled and an override TOML is found, in which case any present
/// keys win per-field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizing {
    pub inbox_capacity: usize,
    pub timer_bottom_slots: usize,
    pub timer_upper_slots: usize,
    pub timer_slot_inline: usize,
    pub task_slab_initial_cap: usize,
    pub run_queue_capacity: usize,
    pub reactor_slab_initial_cap: usize,
    pub bg_pool_default_threads: usize,
}

impl Sizing {
    pub const COMPILED: Self = Self {
        inbox_capacity: sized::INBOX_CAPACITY,
        timer_bottom_slots: sized::TIMER_BOTTOM_SLOTS,
        timer_upper_slots: sized::TIMER_UPPER_SLOTS,
        timer_slot_inline: sized::TIMER_SLOT_INLINE,
        task_slab_initial_cap: sized::TASK_SLAB_INITIAL_CAP,
        run_queue_capacity: sized::RUN_QUEUE_CAPACITY,
        reactor_slab_initial_cap: sized::REACTOR_SLAB_INITIAL_CAP,
        bg_pool_default_threads: sized::BG_POOL_DEFAULT_THREADS,
    };

    /// returns compiled values unless `runtime-prime-config` is enabled and
    /// an override TOML is discoverable, in which case present keys override.
    #[must_use]
    pub fn resolved() -> Self {
        #[cfg(feature = "runtime-prime-config")]
        {
            runtime_config::resolve(Self::COMPILED)
        }
        #[cfg(not(feature = "runtime-prime-config"))]
        {
            Self::COMPILED
        }
    }
}

#[cfg(feature = "runtime-prime-config")]
mod runtime_config {
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use toml::Value;

    use super::Sizing;

    const ENV_KEY: &str = "PROXIMA_RUNTIME_CONFIG";
    const DEFAULT_PATH: &str = "proxima-runtime.toml";

    pub fn resolve(compiled: Sizing) -> Sizing {
        let path = match config_path() {
            Some(path) => path,
            None => return compiled,
        };
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "proxima runtime config unreadable; falling back to compiled sizing",
                );
                return compiled;
            }
        };
        let parsed: Value = match text.parse() {
            Ok(value) => value,
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "proxima runtime config parse error; falling back to compiled sizing",
                );
                return compiled;
            }
        };
        merge(compiled, &parsed)
    }

    fn config_path() -> Option<PathBuf> {
        if let Ok(explicit) = env::var(ENV_KEY) {
            return Some(PathBuf::from(explicit));
        }
        let fallback = PathBuf::from(DEFAULT_PATH);
        if fallback.is_file() {
            Some(fallback)
        } else {
            None
        }
    }

    fn merge(compiled: Sizing, parsed: &Value) -> Sizing {
        Sizing {
            inbox_capacity: pick(parsed, "inbox", "capacity", compiled.inbox_capacity),
            timer_bottom_slots: pick(parsed, "timer", "bottom_slots", compiled.timer_bottom_slots),
            timer_upper_slots: pick(parsed, "timer", "upper_slots", compiled.timer_upper_slots),
            timer_slot_inline: pick(parsed, "timer", "slot_inline", compiled.timer_slot_inline),
            task_slab_initial_cap: pick(
                parsed,
                "executor",
                "task_slab_initial_cap",
                compiled.task_slab_initial_cap,
            ),
            run_queue_capacity: pick(
                parsed,
                "executor",
                "run_queue_capacity",
                compiled.run_queue_capacity,
            ),
            reactor_slab_initial_cap: pick(
                parsed,
                "reactor",
                "slab_initial_cap",
                compiled.reactor_slab_initial_cap,
            ),
            bg_pool_default_threads: pick(
                parsed,
                "background_pool",
                "default_threads",
                compiled.bg_pool_default_threads,
            ),
        }
    }

    fn pick(parsed: &Value, section: &str, key: &str, fallback: usize) -> usize {
        let Some(integer) = parsed
            .get(section)
            .and_then(|section_value| section_value.get(key))
            .and_then(Value::as_integer)
        else {
            return fallback;
        };
        match usize::try_from(integer) {
            Ok(value) => value,
            Err(_) => {
                tracing::warn!(
                    section,
                    key,
                    value = integer,
                    "proxima runtime config value not a non-negative integer; using compiled",
                );
                fallback
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn compiled_values_match_build_consts() {
        let sizing = Sizing::COMPILED;
        assert_eq!(sizing.inbox_capacity, sized::INBOX_CAPACITY);
        assert_eq!(sizing.timer_bottom_slots, sized::TIMER_BOTTOM_SLOTS);
        assert_eq!(sizing.timer_upper_slots, sized::TIMER_UPPER_SLOTS);
        assert_eq!(sizing.timer_slot_inline, sized::TIMER_SLOT_INLINE);
        assert_eq!(sizing.task_slab_initial_cap, sized::TASK_SLAB_INITIAL_CAP);
        assert_eq!(sizing.run_queue_capacity, sized::RUN_QUEUE_CAPACITY);
        assert_eq!(
            sizing.reactor_slab_initial_cap,
            sized::REACTOR_SLAB_INITIAL_CAP
        );
        assert_eq!(
            sizing.bg_pool_default_threads,
            sized::BG_POOL_DEFAULT_THREADS
        );
    }

    #[cfg(feature = "runtime-prime-config")]
    #[test]
    fn resolved_without_env_or_file_returns_compiled() {
        temp_env::with_var("PROXIMA_RUNTIME_CONFIG", None::<&str>, || {
            let resolved = Sizing::resolved();
            assert_eq!(resolved, Sizing::COMPILED);
        });
    }

    #[cfg(feature = "runtime-prime-config")]
    #[test]
    fn resolved_with_partial_override_keeps_compiled_for_missing_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("override.toml");
        std::fs::write(
            &path,
            "[inbox]\ncapacity = 4096\n[timer]\nslot_inline = 8\n",
        )
        .expect("write override toml");
        temp_env::with_var("PROXIMA_RUNTIME_CONFIG", Some(path.as_os_str()), || {
            let resolved = Sizing::resolved();
            assert_eq!(resolved.inbox_capacity, 4096);
            assert_eq!(resolved.timer_slot_inline, 8);
            assert_eq!(
                resolved.timer_bottom_slots,
                Sizing::COMPILED.timer_bottom_slots
            );
            assert_eq!(
                resolved.run_queue_capacity,
                Sizing::COMPILED.run_queue_capacity
            );
        });
    }

    #[cfg(feature = "runtime-prime-config")]
    #[test]
    fn resolved_with_missing_config_path_falls_back_to_compiled() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("nonexistent.toml");
        temp_env::with_var("PROXIMA_RUNTIME_CONFIG", Some(missing.as_os_str()), || {
            let resolved = Sizing::resolved();
            assert_eq!(resolved, Sizing::COMPILED);
        });
    }
}
