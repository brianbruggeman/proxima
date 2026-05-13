//! Shared per-field merge core for every hand-rolled `XxxLayerBuilder` in this
//! crate (`TelemetryConfig`, `EmitConfig`, `InstrumentConfig`). `.from_path`/
//! `.from_env`/`.underlay_path`/`.underlay_env` each contribute only the
//! fields their source actually specifies, merged onto the accumulated
//! config — never a wholesale re-resolve that drops a prior layer's values.
//!
//! Two merge flavors, composable in any call order:
//! - [`MergeMode::Override`]: the incoming field wins over whatever is
//!   already accumulated — last writer wins, per field.
//! - [`MergeMode::Underlay`]: the incoming field applies ONLY if nothing has
//!   set it yet; an already-set value is never clobbered.
//!
//! Every field in these house-pattern configs is a scalar or a Vec/Map data
//! collection (never a `#[setting(nested)]` sub-config), so a one-level
//! object merge covers every real field — a collection is replaced wholesale
//! when a source provides it, never element-merged. Genuine nested-struct
//! recursion (a source setting one subfield without wiping its siblings) is
//! exercised directly by this module's own tests.

use std::collections::BTreeSet;

use conflaguration::ValidationMessage;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

/// Whether an incoming layer's fields win over an already-touched field
/// (`Override`) or only fill a field nothing has set yet (`Underlay`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MergeMode {
    Override,
    Underlay,
}

/// Merge `incoming`'s present fields onto `inner`, tracking which fields have
/// been touched (by dotted path) so `Underlay` layers never clobber an
/// already-set value. `nested_keys` names fields that are themselves a
/// single embedded config — as opposed to a Vec/Map data collection, which
/// always replaces wholesale when present — and therefore recurse one level
/// instead of being replaced outright. Pass `&[]` when the type has no
/// nested sub-config fields (every production config in this crate today).
pub(crate) fn apply_layer<T>(
    inner: &mut T,
    touched: &mut BTreeSet<String>,
    incoming: Value,
    mode: MergeMode,
    nested_keys: &[&str],
) -> Result<(), conflaguration::Error>
where
    T: Serialize + DeserializeOwned,
{
    let Value::Object(incoming_map) = incoming else {
        return Ok(());
    };
    let mut base = to_value(inner)?;
    let Value::Object(base_map) = &mut base else {
        return Ok(());
    };
    merge_object(base_map, incoming_map, mode, nested_keys, touched, "");
    *inner = from_value(base)?;
    Ok(())
}

fn merge_object(
    base: &mut Map<String, Value>,
    incoming: Map<String, Value>,
    mode: MergeMode,
    nested_keys: &[&str],
    touched: &mut BTreeSet<String>,
    prefix: &str,
) {
    for (key, incoming_value) in incoming {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        if nested_keys.contains(&key.as_str())
            && let Value::Object(incoming_child) = incoming_value
        {
            let base_child = base
                .entry(key.clone())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Value::Object(base_child_map) = base_child {
                merge_object(base_child_map, incoming_child, mode, &[], touched, &path);
            }
            continue;
        }
        apply_leaf(base, &key, incoming_value, mode, &path, touched);
    }
}

fn apply_leaf(
    map: &mut Map<String, Value>,
    key: &str,
    value: Value,
    mode: MergeMode,
    touched_path: &str,
    touched: &mut BTreeSet<String>,
) {
    let should_apply = match mode {
        MergeMode::Override => true,
        MergeMode::Underlay => !touched.contains(touched_path),
    };
    if should_apply {
        map.insert(key.to_string(), value);
        touched.insert(touched_path.to_string());
    }
}

/// Add `field` to `partial` only if any of `env_names` is actually set in the
/// process environment — never because `value` happens to equal a default.
pub(crate) fn insert_if_env_set<T: Serialize>(
    partial: &mut Map<String, Value>,
    field: &str,
    env_names: &[&str],
    value: &T,
) -> Result<(), conflaguration::Error> {
    if env_names.iter().any(|name| std::env::var(name).is_ok()) {
        partial.insert(field.to_string(), to_value(value)?);
    }
    Ok(())
}

pub(crate) fn to_value<T: Serialize>(value: &T) -> Result<Value, conflaguration::Error> {
    serde_json::to_value(value).map_err(|error| conflaguration::Error::Validation {
        errors: vec![ValidationMessage::new(
            "layered",
            format!("serialize failed: {error}"),
        )],
    })
}

fn from_value<T: DeserializeOwned>(value: Value) -> Result<T, conflaguration::Error> {
    serde_json::from_value(value).map_err(|error| conflaguration::Error::Validation {
        errors: vec![ValidationMessage::new(
            "layered",
            format!("deserialize failed: {error}"),
        )],
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
    struct Inner {
        #[serde(default)]
        a: u32,
        #[serde(default)]
        b: u32,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
    struct Outer {
        #[serde(default)]
        scalar: u32,
        #[serde(default)]
        list: Vec<u32>,
        #[serde(default)]
        nested: Inner,
    }

    // a nested layer setting only one subfield must not wipe the sibling
    // subfield a prior layer set (RECURSIVE merge for a genuine sub-config).
    #[test]
    fn nested_object_merges_per_subfield_not_wholesale() {
        let mut inner = Outer::default();
        let mut touched = BTreeSet::new();

        let first = serde_json::json!({ "nested": { "a": 7 } });
        apply_layer(
            &mut inner,
            &mut touched,
            first,
            MergeMode::Override,
            &["nested"],
        )
        .unwrap();
        assert_eq!(inner.nested, Inner { a: 7, b: 0 });

        let second = serde_json::json!({ "nested": { "b": 9 } });
        apply_layer(
            &mut inner,
            &mut touched,
            second,
            MergeMode::Override,
            &["nested"],
        )
        .unwrap();
        assert_eq!(
            inner.nested,
            Inner { a: 7, b: 9 },
            "setting b must not wipe the a the first layer set"
        );
    }

    // underlay into a nested field only fills the subfield that's still
    // unset; an already-touched subfield is preserved.
    #[test]
    fn nested_object_underlay_fills_only_unset_subfields() {
        let mut inner = Outer::default();
        let mut touched = BTreeSet::new();

        let explicit = serde_json::json!({ "nested": { "a": 1 } });
        apply_layer(
            &mut inner,
            &mut touched,
            explicit,
            MergeMode::Override,
            &["nested"],
        )
        .unwrap();

        let fallback = serde_json::json!({ "nested": { "a": 99, "b": 5 } });
        apply_layer(
            &mut inner,
            &mut touched,
            fallback,
            MergeMode::Underlay,
            &["nested"],
        )
        .unwrap();

        assert_eq!(
            inner.nested,
            Inner { a: 1, b: 5 },
            "a stays at the explicit value; b fills from the underlay since it was unset"
        );
    }

    // a Vec field is replaced wholesale when a layer provides it, never
    // element-merged/appended.
    #[test]
    fn collection_field_replaces_wholesale_on_override() {
        let mut inner = Outer::default();
        let mut touched = BTreeSet::new();

        let first = serde_json::json!({ "list": [1, 2, 3] });
        apply_layer(&mut inner, &mut touched, first, MergeMode::Override, &[]).unwrap();
        assert_eq!(inner.list, vec![1, 2, 3]);

        let second = serde_json::json!({ "list": [9] });
        apply_layer(&mut inner, &mut touched, second, MergeMode::Override, &[]).unwrap();
        assert_eq!(inner.list, vec![9], "replace wholesale, not append/union");
    }

    // underlay never touches a collection that's already set, even partially.
    #[test]
    fn collection_field_underlay_never_touches_already_set_collection() {
        let mut inner = Outer::default();
        let mut touched = BTreeSet::new();

        let explicit = serde_json::json!({ "list": [1] });
        apply_layer(&mut inner, &mut touched, explicit, MergeMode::Override, &[]).unwrap();

        let fallback = serde_json::json!({ "list": [9, 9, 9] });
        apply_layer(&mut inner, &mut touched, fallback, MergeMode::Underlay, &[]).unwrap();

        assert_eq!(
            inner.list,
            vec![1],
            "already-set collection is never clobbered or merged"
        );
    }

    // underlay DOES fill a collection that's still fully unset.
    #[test]
    fn collection_field_underlay_fills_when_unset() {
        let mut inner = Outer::default();
        let mut touched = BTreeSet::new();

        let fallback = serde_json::json!({ "list": [4, 5] });
        apply_layer(&mut inner, &mut touched, fallback, MergeMode::Underlay, &[]).unwrap();

        assert_eq!(inner.list, vec![4, 5]);
    }
}
