//! How an incumbent loader names and sizes the quantized form of a family,
//! when that differs from the Hugging Face checkpoint the family was
//! converted from.
//!
//! mlx-lm splits a fused expert stack into one module per projection and
//! quantizes some modules at a wider bit width than the rest, so its
//! quantized files name `switch_mlp.gate_proj` where the IBM checkpoint names
//! one `input_linear`. [`QuantizedConventions`] is that difference as data:
//! [`super::export_affine_named`] applies it to the [`MlxNames`] that
//! [`super::dense_names`] produced, so the fused leaves become ordinary
//! mapped tensors and the packing path needs no second case.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::mlx_affine::AffineParams;
use super::mlx_names::MlxNames;
use crate::error::InteropError;

/// The quantized-export conventions of one family; [`Self::NONE`] keeps the
/// checkpoint names and one bit width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedConventions {
    /// `(stem, projections)`: a tensor name containing `stem` is renamed with
    /// `stem` replaced by `projections[0]`, or, for part `n` of a fused
    /// tensor of `projections.len()` parts, by `projections[n]`, e.g.
    /// `("block_sparse_moe.input_linear", &["block_sparse_moe.switch_mlp.gate_proj",
    /// "block_sparse_moe.switch_mlp.up_proj"])`.
    pub renames: &'static [(&'static str, &'static [&'static str])],
    /// `(module suffix, bits)`: a module whose path ends in the suffix is
    /// quantized at `bits` instead of the export's, e.g.
    /// `("block_sparse_moe.router.layer", 8)`.
    pub bits: &'static [(&'static str, u8)],
}

impl QuantizedConventions {
    pub const NONE: Self = Self {
        renames: &[],
        bits: &[],
    };

    /// `names` with every fused leaf a rule covers moved into `mapped` under
    /// its projection name, and every mapped name a rule covers renamed.
    #[must_use]
    pub fn apply(&self, names: MlxNames) -> MlxNames {
        let MlxNames {
            mapped,
            fused,
            unmapped,
        } = names;
        let (split, kept): (Vec<_>, Vec<_>) = fused
            .into_iter()
            .partition(|item| self.projection(&item.hf, item.part, item.of).is_some());
        let split_pairs = split.into_iter().filter_map(|item| {
            let name = self.projection(&item.hf, item.part, item.of)?;
            Some((item.leaf, name))
        });
        let renamed = mapped.into_iter().map(|(leaf, hf)| {
            let name = self.projection(&hf, 0, 1).unwrap_or(hf);
            (leaf, name)
        });
        MlxNames {
            mapped: renamed.chain(split_pairs).collect(),
            fused: kept,
            unmapped,
        }
    }

    fn projection(&self, hf: &str, part: u32, of: u32) -> Option<String> {
        let part = usize::try_from(part).ok()?;
        let of = usize::try_from(of).ok()?;
        self.renames
            .iter()
            .filter(|(stem, projections)| hf.contains(stem) && projections.len() == of)
            .find_map(|(stem, projections)| Some(hf.replace(stem, projections.get(part)?)))
    }

    /// The parameters for the module at `hf` (a tensor or module name): the
    /// rule's bit width at `base`'s group size, else `base` itself.
    ///
    /// # Errors
    ///
    /// [`InteropError::AffineBitsUnsupported`] for a rule outside mlx-lm's
    /// bit grid.
    pub fn params_for(&self, hf: &str, base: AffineParams) -> Result<AffineParams, InteropError> {
        let module = hf.strip_suffix(".weight").unwrap_or(hf);
        match self
            .bits
            .iter()
            .find(|(suffix, _)| module.ends_with(suffix))
        {
            Some((_, bits)) => AffineParams::new(*bits, base.group_size()),
            None => Ok(base),
        }
    }

    /// `(module path, bits)` for every quantized module of `tensor_names`
    /// whose rule differs from `base`: what `config.json` lists per module.
    #[must_use]
    pub fn overrides<'name>(
        &self,
        tensor_names: impl IntoIterator<Item = &'name str>,
        base: AffineParams,
    ) -> Vec<(String, u8)> {
        tensor_names
            .into_iter()
            .filter_map(|name| name.strip_suffix(".scales"))
            .filter_map(|module| {
                let bits = self.params_for(module, base).ok()?.bits();
                (bits != base.bits()).then(|| (module.to_string(), bits))
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::transform::mlx_config::GRANITE_QUANTIZED as GRANITE;
    use crate::transform::mlx_names::FusedName;

    fn fused(leaf: &str, part: u32) -> FusedName {
        FusedName {
            leaf: leaf.to_string(),
            hf: "model.layers.3.block_sparse_moe.input_linear.weight".to_string(),
            part,
            of: 2,
        }
    }

    fn names() -> MlxNames {
        MlxNames {
            mapped: vec![
                (
                    "blk.3.ffn_down_exps.weight".to_string(),
                    "model.layers.3.block_sparse_moe.output_linear.weight".to_string(),
                ),
                (
                    "blk.3.attn_q.weight".to_string(),
                    "model.layers.3.self_attn.q_proj.weight".to_string(),
                ),
            ],
            fused: vec![
                fused("blk.3.ffn_gate_exps.weight", 0),
                fused("blk.3.ffn_up_exps.weight", 1),
            ],
            unmapped: Vec::new(),
        }
    }

    #[test]
    fn mlx_conventions_fused_halves_become_one_mapped_projection_each() {
        let applied = GRANITE.apply(names());

        assert!(applied.fused.is_empty());
        assert!(applied.mapped.contains(&(
            "blk.3.ffn_gate_exps.weight".to_string(),
            "model.layers.3.block_sparse_moe.switch_mlp.gate_proj.weight".to_string()
        )));
        assert!(applied.mapped.contains(&(
            "blk.3.ffn_up_exps.weight".to_string(),
            "model.layers.3.block_sparse_moe.switch_mlp.up_proj.weight".to_string()
        )));
    }

    #[test]
    fn mlx_conventions_a_single_covered_name_is_renamed_and_others_are_kept() {
        let applied = GRANITE.apply(names());

        assert!(applied.mapped.contains(&(
            "blk.3.ffn_down_exps.weight".to_string(),
            "model.layers.3.block_sparse_moe.switch_mlp.down_proj.weight".to_string()
        )));
        assert!(applied.mapped.contains(&(
            "blk.3.attn_q.weight".to_string(),
            "model.layers.3.self_attn.q_proj.weight".to_string()
        )));
    }

    #[test]
    fn mlx_conventions_none_leaves_names_untouched() {
        let applied = QuantizedConventions::NONE.apply(names());

        assert_eq!(applied, names());
    }

    #[test]
    fn mlx_conventions_a_part_count_that_differs_from_the_rule_is_left_fused() {
        let mut three_way = names();
        three_way.fused.iter_mut().for_each(|item| item.of = 3);

        let applied = GRANITE.apply(three_way);

        assert_eq!(applied.fused.len(), 2);
    }

    #[test]
    fn mlx_conventions_router_is_quantized_at_eight_bits_at_the_export_group_size() {
        let base = AffineParams::new(4, 32).unwrap();

        let router = GRANITE
            .params_for("model.layers.0.block_sparse_moe.router.layer.weight", base)
            .unwrap();
        let attention = GRANITE
            .params_for("model.layers.0.self_attn.q_proj.weight", base)
            .unwrap();

        assert_eq!((router.bits(), router.group_size()), (8, 32));
        assert_eq!(attention, base);
    }

    #[test]
    fn mlx_conventions_overrides_list_only_modules_that_differ_from_the_base() {
        let base = AffineParams::default();
        let tensors = [
            "model.layers.0.block_sparse_moe.router.layer.scales",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.block_sparse_moe.router.layer.weight",
        ];

        let overrides = GRANITE.overrides(tensors, base);

        assert_eq!(
            overrides,
            vec![(
                "model.layers.0.block_sparse_moe.router.layer".to_string(),
                8
            )]
        );
    }
}
