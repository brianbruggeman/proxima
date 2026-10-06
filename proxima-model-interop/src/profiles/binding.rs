//! Binding profiles: the names a checkpoint's tensor directory uses that a
//! lowered program's `Op::Input` leaves do not, one embedded TOML file per
//! family plus a `default.toml` every family inherits.
//!
//! [`crate::bind_leaves::bind_program_leaves`] walks the leaves of a lowered
//! program and looks each one up in the GGUF tensor directory. A leaf the
//! directory names exactly binds as is. A leaf it does not name falls through
//! to the first [`TensorAlias`] that matches, so a tied output projection, a
//! fused tensor the program splits, or a tensor stored under a different name
//! is data here and never a branch in the binder.
//!
//! This directory is the only place a family name appears, the same rule
//! [`super::family_profile`] states for the descriptor profiles. A family with
//! no file in `binding/` binds with the defaults alone.

use serde::Deserialize;

use crate::error::InteropError;

const DEFAULT_BINDING: &str = include_str!("binding/default.toml");

const FAMILY_BINDINGS: &[(&str, &str)] = &[
    ("gemma4", include_str!("binding/gemma4.toml")),
    ("qwen35", include_str!("binding/qwen35.toml")),
    ("qwen35moe", include_str!("binding/qwen35moe.toml")),
    ("lfm2", include_str!("binding/lfm2.toml")),
    ("lfm2moe", include_str!("binding/lfm2.toml")),
];

/// One way a program leaf is satisfied by tensors the directory names
/// differently. `leaf` and every `from` are name suffixes that must start at a
/// `.` boundary (or at the start of the name), and the text before the `leaf`
/// suffix is carried onto each `from`, so `blk.7.` is preserved across the
/// rewrite.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TensorAlias {
    /// The leaf reads another tensor's bytes under its own name (a tied output
    /// projection reading the embedding table).
    Rename { leaf: String, from: String },
    /// The leaf is part `part` of `of` equal slices of one fused tensor's row
    /// axis (`ne1`), taken within every outer index of any axis beyond it.
    Part {
        leaf: String,
        from: String,
        part: u32,
        of: u32,
    },
    /// The leaf is the in-order concatenation of several tensors, bound as one
    /// packed operand.
    Join { leaf: String, from: Vec<String> },
}

impl TensorAlias {
    fn leaf(&self) -> &str {
        match self {
            Self::Rename { leaf, .. } | Self::Part { leaf, .. } | Self::Join { leaf, .. } => leaf,
        }
    }
}

/// How one family's tensor directory differs from its lowered program's
/// leaves. Every field defaults to empty, so a family that needs nothing
/// ships no file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BindingProfile {
    /// Tried in order after an exact directory match fails; first match wins.
    #[serde(rename = "alias")]
    pub aliases: Vec<TensorAlias>,
    /// Leaf name suffixes bound as owned `f32`, decoded from whatever codec
    /// the file stores, instead of staying packed.
    pub decode_f32: Vec<String>,
    /// Tensors bound by name although no leaf reads them: a lookup table a
    /// step input derives its values from.
    pub extra: Vec<String>,
}

/// The text before `suffix` when `name` ends with `suffix` at a `.` boundary:
/// `""` for a whole-name match, `"blk.7."` for `blk.7.ssm_in.weight`.
pub(crate) fn name_prefix_before<'name>(name: &'name str, suffix: &str) -> Option<&'name str> {
    let prefix = name.strip_suffix(suffix)?;
    (prefix.is_empty() || prefix.ends_with('.')).then_some(prefix)
}

impl BindingProfile {
    /// Every alias whose `leaf` suffix ends `name`, in precedence order, each
    /// with the prefix to put back in front of its sources. A caller takes the
    /// first one whose sources the tensor directory actually has.
    pub fn aliases_for<'profile, 'name>(
        &'profile self,
        name: &'name str,
    ) -> impl Iterator<Item = (&'profile TensorAlias, &'name str)> {
        self.aliases
            .iter()
            .filter_map(move |alias| Some((alias, name_prefix_before(name, alias.leaf())?)))
    }

    /// `true` when `name` ends with one of [`Self::decode_f32`] at a `.`
    /// boundary.
    #[must_use]
    pub fn decodes_to_f32(&self, name: &str) -> bool {
        self.decode_f32
            .iter()
            .any(|suffix| name_prefix_before(name, suffix).is_some())
    }

    fn inherits(self, defaults: Self) -> Self {
        Self {
            aliases: self.aliases.into_iter().chain(defaults.aliases).collect(),
            decode_f32: self.decode_f32,
            extra: self.extra,
        }
    }
}

fn parse(family: &str, text: &str) -> Result<BindingProfile, InteropError> {
    toml::from_str(text).map_err(|error| InteropError::InvalidBindingProfile {
        family: family.into(),
        message: error.to_string(),
    })
}

/// The [`BindingProfile`] for `family` (`general.architecture`): the family's
/// own file layered over the defaults, or the defaults alone when the family
/// ships none.
///
/// # Errors
///
/// [`InteropError::InvalidBindingProfile`] when an embedded file does not
/// parse.
pub fn binding_profile(family: &str) -> Result<BindingProfile, InteropError> {
    let defaults = parse("default", DEFAULT_BINDING)?;
    match FAMILY_BINDINGS.iter().find(|(name, _)| *name == family) {
        Some((_, text)) => Ok(parse(family, text)?.inherits(defaults)),
        None => Ok(defaults),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_binding_file_parses() {
        assert_eq!(FAMILY_BINDINGS.len(), 5, "one row per embedded family string");
        for (family, _) in FAMILY_BINDINGS {
            binding_profile(family).unwrap_or_else(|error| panic!("{family}: {error}"));
        }
    }

    #[test]
    fn a_family_without_a_file_binds_with_the_defaults_alone() {
        let profile = binding_profile("llama").expect("defaults parse");

        assert_eq!(profile, parse("default", DEFAULT_BINDING).unwrap());
        assert!(profile.decode_f32.is_empty() && profile.extra.is_empty());
    }

    #[test]
    fn an_alias_matches_only_at_a_dot_boundary() {
        let profile = binding_profile("llama").expect("defaults parse");

        assert!(profile.aliases_for("output.weight").next().is_some());
        assert!(profile.aliases_for("blk.0.attn_output.weight").next().is_none());
    }

    #[test]
    fn the_layer_prefix_is_carried_onto_the_alias_source() {
        let profile = binding_profile("qwen35").expect("qwen35 parses");

        let (alias, prefix) = profile.aliases_for("blk.12.ssm_in.weight").next().expect("ssm_in aliases");

        assert_eq!(prefix, "blk.12.");
        assert!(matches!(alias, TensorAlias::Rename { from, .. } if from == "attn_qkv.weight"));
    }

    #[test]
    fn a_family_alias_outranks_the_default_that_names_the_same_leaf() {
        let family = parse(
            "fixture",
            "[[alias]]\nkind = \"rename\"\nleaf = \"output.weight\"\nfrom = \"lm_head.weight\"\n",
        )
        .unwrap()
        .inherits(parse("default", DEFAULT_BINDING).unwrap());

        let (alias, _) = family.aliases_for("output.weight").next().expect("output aliases");

        assert!(matches!(alias, TensorAlias::Rename { from, .. } if from == "lm_head.weight"));
    }

    #[test]
    fn an_unknown_key_is_refused() {
        let refused = parse("fixture", "decode = [\"x\"]\n");

        assert!(matches!(refused, Err(InteropError::InvalidBindingProfile { .. })));
    }

    #[test]
    fn the_gemma4_file_binds_the_rope_factor_table_the_sliding_step_reads() {
        let profile = binding_profile("gemma4").expect("gemma4 parses");

        assert_eq!(profile.extra, ["rope_freqs.weight"]);
    }

    #[test]
    fn the_moe_hybrid_decodes_its_two_small_projections_to_f32() {
        let profile = binding_profile("qwen35moe").expect("qwen35moe parses");

        assert!(profile.decodes_to_f32("blk.3.ssm_alpha.weight"));
        assert!(!profile.decodes_to_f32("blk.3.ssm_out.weight"));
    }
}
