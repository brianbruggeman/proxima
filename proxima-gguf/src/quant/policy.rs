//! Per-tensor precision policy: which [`GgmlType`] a tensor should be
//! stored at, chosen by its functional role rather than uniformly across
//! the whole checkpoint.
//!
//! Ground truth for [`PrecisionPolicy::llama_cpp_q4_k_s`]: llama.cpp's own
//! `Q4_K_S` quantizer applied to two real checkpoints (read via this
//! crate's parser, metadata only, never the tensor payload) --
//! `Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf` (995 tensors: `Q4_K` x769,
//! `F32` x65, `Q8_0` x64, `Q5_K` x64, `F16` x32, `Q6_K` x1) and
//! `openchat-3.5-1210.Q4_K_S.gguf` (291 tensors: `Q4_K` x217, `F32` x65,
//! `Q5_K` x8, `Q6_K` x1). Every role below is the *uniform* choice observed
//! for that role in at least one of the two models; where llama.cpp instead
//! varies by layer position (`attn_v`, part of `ffn_down` in the MoE model)
//! the majority type is used and the variation is documented on the field.
//!
//! This module intentionally is NOT a heuristic buried in a function: the
//! mapping is a plain data value ([`PrecisionPolicy`]), so a caller can
//! build one from measurement (this crate's own error-vs-bytes curve, see
//! `tests::real_file`) or from runtime access telemetry (e.g.
//! `proxima-tensor`'s `OperandAccess::distinct_elements` /
//! `total_elements`, which already identifies a cold MoE expert) instead of
//! the llama.cpp default. Applying the policy ([`PrecisionPolicy::target_for`])
//! is the only mechanism this module owns; re-encoding a tensor at the
//! chosen type is the caller's job (this crate's `quant` codecs already do
//! that half).

use crate::tensor::TensorInfo;
use crate::types::GgmlType;

/// A tensor's functional role within a transformer checkpoint, independent
/// of which layer or (for MoE) which expert it belongs to. Classified from
/// the tensor's GGUF name, which follows llama.cpp's own naming convention
/// (`blk.<layer>.<role>[.<expert>].weight`, or a handful of top-level names
/// with no `blk.` prefix at all).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum TensorRole {
    /// `token_embd.weight` — the input embedding table.
    TokenEmbd,
    /// `blk.N.attn_q.weight` — attention query projection.
    AttnQ,
    /// `blk.N.attn_k.weight` — attention key projection.
    AttnK,
    /// `blk.N.attn_v.weight` — attention value projection.
    AttnV,
    /// `blk.N.attn_output.weight` — attention output projection.
    AttnOutput,
    /// `blk.N.attn_norm.weight` — pre-attention layer norm.
    AttnNorm,
    /// `blk.N.ffn_gate_inp.weight` — MoE router (expert-selection gate).
    /// Checked before [`Self::FfnGate`]: the name is a superstring of it.
    FfnGateInp,
    /// `blk.N.ffn_gate[.E].weight` — FFN gate projection (SwiGLU gate half).
    FfnGate,
    /// `blk.N.ffn_up[.E].weight` — FFN up projection.
    FfnUp,
    /// `blk.N.ffn_down[.E].weight` — FFN down projection.
    FfnDown,
    /// `blk.N.ffn_norm.weight` — pre-FFN layer norm.
    FfnNorm,
    /// `output_norm.weight` — final layer norm before the output head.
    OutputNorm,
    /// `output.weight` — the output (unembedding) head.
    OutputWeight,
    /// Anything not matched above (e.g. rope frequency tables, biases some
    /// architectures carry) — a real tensor this policy has no opinion on
    /// yet, not an error.
    Other,
}

impl TensorRole {
    /// Classifies a tensor by its GGUF name. Substring matches, not exact
    /// equality, so a per-layer (`blk.3.attn_q.weight`) or per-expert
    /// (`blk.3.ffn_down.5.weight`) name lands on the same role as every
    /// other layer/expert's instance of it.
    #[must_use]
    pub fn classify(name: &str) -> Self {
        match name {
            "token_embd.weight" => return Self::TokenEmbd,
            "output.weight" => return Self::OutputWeight,
            "output_norm.weight" => return Self::OutputNorm,
            _ => {}
        }
        if name.contains("attn_norm") {
            Self::AttnNorm
        } else if name.contains("ffn_norm") {
            Self::FfnNorm
        } else if name.contains("attn_q") {
            Self::AttnQ
        } else if name.contains("attn_k") {
            Self::AttnK
        } else if name.contains("attn_v") {
            Self::AttnV
        } else if name.contains("attn_output") {
            Self::AttnOutput
        } else if name.contains("ffn_gate_inp") {
            Self::FfnGateInp
        } else if name.contains("ffn_gate") {
            Self::FfnGate
        } else if name.contains("ffn_up") {
            Self::FfnUp
        } else if name.contains("ffn_down") {
            Self::FfnDown
        } else {
            Self::Other
        }
    }
}

/// A per-role target-[`GgmlType`] assignment — the value a caller supplies
/// to decide what a tensor should be re-encoded as. Alloc-tier and
/// sans-IO: this type only decides; applying it
/// ([`Self::target_for`]/[`Self::target_for_role`]) is a pure lookup, and
/// actually re-encoding a tensor's bytes at the chosen type is left to
/// [`crate::quant`]'s codecs plus whatever owns the file IO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionPolicy {
    pub token_embd: GgmlType,
    pub attn_q: GgmlType,
    pub attn_k: GgmlType,
    pub attn_v: GgmlType,
    pub attn_output: GgmlType,
    pub attn_norm: GgmlType,
    pub ffn_gate_inp: GgmlType,
    pub ffn_gate: GgmlType,
    pub ffn_up: GgmlType,
    pub ffn_down: GgmlType,
    pub ffn_norm: GgmlType,
    pub output_norm: GgmlType,
    pub output_weight: GgmlType,
    pub other: GgmlType,
}

impl PrecisionPolicy {
    /// The policy this crate measured llama.cpp's real `Q4_K_S` quantizer
    /// use (see this module's doc comment for the two source checkpoints
    /// and exact tensor counts). Most roles sit at `Q4_K`; `attn_k`/`attn_v`
    /// are bumped to `Q8_0` in the MoE (Mixtral) checkpoint -- the majority
    /// observed across both models is `Q4_K` for `attn_k` (uniform in both)
    /// and `Q4_K` for `attn_v` (uniform in the dense 7B aside from 4 of 32
    /// early/mid layers bumped to `Q5_K`; entirely `Q8_0` in the 8-expert
    /// MoE model -- this default follows the dense model's majority since
    /// the MoE case is architecture-conditional, not a blanket rule).
    /// `attn_output` is uniformly `Q5_K` in the MoE checkpoint and uniformly
    /// `Q4_K` in the dense one; this default follows the dense majority for
    /// the same reason. `output.weight` (the lone tensor of its role) is
    /// `Q6_K` in both. Norms (`attn_norm`/`ffn_norm`/`output_norm`) and the
    /// MoE router (`ffn_gate_inp`) are never block-quantized by llama.cpp at
    /// this quant level -- norms stay `F32`, the router stays `F16`.
    #[must_use]
    pub const fn llama_cpp_q4_k_s() -> Self {
        Self {
            token_embd: GgmlType::Q4_K,
            attn_q: GgmlType::Q4_K,
            attn_k: GgmlType::Q4_K,
            attn_v: GgmlType::Q4_K,
            attn_output: GgmlType::Q4_K,
            attn_norm: GgmlType::F32,
            ffn_gate_inp: GgmlType::F16,
            ffn_gate: GgmlType::Q4_K,
            ffn_up: GgmlType::Q4_K,
            ffn_down: GgmlType::Q4_K,
            ffn_norm: GgmlType::F32,
            output_norm: GgmlType::F32,
            output_weight: GgmlType::Q6_K,
            other: GgmlType::Q4_K,
        }
    }

    /// The policy observed for the 8-expert MoE checkpoint specifically:
    /// `attn_k`/`attn_v` bumped all the way to `Q8_0` and `attn_output`
    /// uniformly bumped to `Q5_K` -- both real, both llama.cpp's own
    /// choice, just conditioned on `n_expert == 8` rather than universal.
    #[must_use]
    pub const fn llama_cpp_q4_k_s_moe_8_expert() -> Self {
        Self {
            attn_k: GgmlType::Q8_0,
            attn_v: GgmlType::Q8_0,
            attn_output: GgmlType::Q5_K,
            ..Self::llama_cpp_q4_k_s()
        }
    }

    /// Looks up the target type for an already-classified role.
    #[must_use]
    pub const fn target_for_role(&self, role: TensorRole) -> GgmlType {
        match role {
            TensorRole::TokenEmbd => self.token_embd,
            TensorRole::AttnQ => self.attn_q,
            TensorRole::AttnK => self.attn_k,
            TensorRole::AttnV => self.attn_v,
            TensorRole::AttnOutput => self.attn_output,
            TensorRole::AttnNorm => self.attn_norm,
            TensorRole::FfnGateInp => self.ffn_gate_inp,
            TensorRole::FfnGate => self.ffn_gate,
            TensorRole::FfnUp => self.ffn_up,
            TensorRole::FfnDown => self.ffn_down,
            TensorRole::FfnNorm => self.ffn_norm,
            TensorRole::OutputNorm => self.output_norm,
            TensorRole::OutputWeight => self.output_weight,
            TensorRole::Other => self.other,
        }
    }

    /// Classifies `tensor`'s role from its name, then looks up the target
    /// type. The whole "apply a policy to a tensor" mechanism: no IO, no
    /// allocation beyond what [`TensorRole::classify`]'s `&str` matching
    /// already does (none).
    #[must_use]
    pub fn target_for(&self, tensor: &TensorInfo) -> GgmlType {
        self.target_for_role(TensorRole::classify(&tensor.name))
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use arrayvec::ArrayVec;

    use super::*;

    fn tensor(name: &str, ggml_type: GgmlType) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dims: ArrayVec::new(),
            ggml_type,
            offset: 0,
        }
    }

    #[test]
    fn classify_matches_every_named_role_from_real_gguf_naming() {
        let cases: &[(&str, TensorRole)] = &[
            ("token_embd.weight", TensorRole::TokenEmbd),
            ("output.weight", TensorRole::OutputWeight),
            ("output_norm.weight", TensorRole::OutputNorm),
            ("blk.0.attn_norm.weight", TensorRole::AttnNorm),
            ("blk.0.ffn_norm.weight", TensorRole::FfnNorm),
            ("blk.0.attn_q.weight", TensorRole::AttnQ),
            ("blk.0.attn_k.weight", TensorRole::AttnK),
            ("blk.0.attn_v.weight", TensorRole::AttnV),
            ("blk.0.attn_output.weight", TensorRole::AttnOutput),
            ("blk.0.ffn_gate_inp.weight", TensorRole::FfnGateInp),
            ("blk.0.ffn_gate.weight", TensorRole::FfnGate),
            ("blk.0.ffn_gate.5.weight", TensorRole::FfnGate),
            ("blk.0.ffn_up.weight", TensorRole::FfnUp),
            ("blk.0.ffn_up.5.weight", TensorRole::FfnUp),
            ("blk.0.ffn_down.weight", TensorRole::FfnDown),
            ("blk.0.ffn_down.5.weight", TensorRole::FfnDown),
            ("rope_freqs.weight", TensorRole::Other),
        ];
        for &(name, expected) in cases {
            assert_eq!(TensorRole::classify(name), expected, "name={name}");
        }
    }

    #[test]
    fn ffn_gate_inp_is_not_misclassified_as_ffn_gate() {
        // the substring trap this module's classify order exists to avoid.
        assert_eq!(
            TensorRole::classify("blk.7.ffn_gate_inp.weight"),
            TensorRole::FfnGateInp
        );
    }

    #[test]
    fn llama_cpp_default_leaves_norms_and_router_unquantized() {
        let policy = PrecisionPolicy::llama_cpp_q4_k_s();
        assert_eq!(policy.attn_norm, GgmlType::F32);
        assert_eq!(policy.ffn_norm, GgmlType::F32);
        assert_eq!(policy.output_norm, GgmlType::F32);
        assert_eq!(policy.ffn_gate_inp, GgmlType::F16);
    }

    #[test]
    fn llama_cpp_default_picks_q6_k_for_the_output_head() {
        let policy = PrecisionPolicy::llama_cpp_q4_k_s();
        assert_eq!(policy.output_weight, GgmlType::Q6_K);
    }

    #[test]
    fn moe_8_expert_policy_bumps_attn_kv_and_output_above_the_dense_default() {
        let dense = PrecisionPolicy::llama_cpp_q4_k_s();
        let moe = PrecisionPolicy::llama_cpp_q4_k_s_moe_8_expert();
        assert_eq!(dense.attn_k, GgmlType::Q4_K);
        assert_eq!(moe.attn_k, GgmlType::Q8_0);
        assert_eq!(dense.attn_v, GgmlType::Q4_K);
        assert_eq!(moe.attn_v, GgmlType::Q8_0);
        assert_eq!(dense.attn_output, GgmlType::Q4_K);
        assert_eq!(moe.attn_output, GgmlType::Q5_K);
        // everything not explicitly overridden stays identical to the dense
        // default -- proves `..Self::llama_cpp_q4_k_s()` isn't silently
        // dropping fields.
        assert_eq!(dense.token_embd, moe.token_embd);
        assert_eq!(dense.ffn_gate, moe.ffn_gate);
        assert_eq!(dense.output_weight, moe.output_weight);
    }

    #[test]
    fn target_for_applies_classification_then_lookup() {
        let policy = PrecisionPolicy::llama_cpp_q4_k_s();
        let attn_v = tensor("blk.3.attn_v.weight", GgmlType::Q4_K);
        assert_eq!(policy.target_for(&attn_v), GgmlType::Q4_K);

        let embd = tensor("token_embd.weight", GgmlType::Q4_K);
        assert_eq!(policy.target_for(&embd), GgmlType::Q4_K);

        let norm = tensor("blk.3.attn_norm.weight", GgmlType::F32);
        assert_eq!(policy.target_for(&norm), GgmlType::F32);
    }

    #[test]
    fn a_mixed_precision_result_carries_the_intended_type_per_tensor() {
        // the deliverable this whole module exists for: applying one
        // policy to a heterogeneous batch of tensors produces a distinct
        // target type per tensor, not one uniform type for the batch.
        let policy = PrecisionPolicy::llama_cpp_q4_k_s();
        let tensors = [
            tensor("token_embd.weight", GgmlType::F32),
            tensor("blk.0.attn_norm.weight", GgmlType::F32),
            tensor("blk.0.ffn_gate_inp.weight", GgmlType::F32),
            tensor("output.weight", GgmlType::F32),
        ];
        let targets: alloc::vec::Vec<GgmlType> = tensors.iter().map(|tensor| policy.target_for(tensor)).collect();
        assert_eq!(
            targets,
            alloc::vec![GgmlType::Q4_K, GgmlType::F32, GgmlType::F16, GgmlType::Q6_K]
        );
    }

    #[test]
    fn other_role_falls_back_to_the_policys_other_field_not_a_panic() {
        let policy = PrecisionPolicy {
            other: GgmlType::F32,
            ..PrecisionPolicy::llama_cpp_q4_k_s()
        };
        let odd = tensor("rope_freqs.weight", GgmlType::F32);
        assert_eq!(policy.target_for(&odd), GgmlType::F32);
    }
}

// -- Real-data proof: apply the policy to every tensor of the real Mixtral
// checkpoint's directory (metadata only, never tensor payload bytes) and
// report per-role coverage plus where the coarse per-role default disagrees
// with llama.cpp's real (layer-conditional) choice. Opportunistic, same as
// `crate::tests::real_file`: skips cleanly when the host-local model cache
// is absent.
#[cfg(all(test, feature = "std"))]
mod real_file {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::io::{Read, Seek, SeekFrom};

    use crate::pipe::parse_complete;
    use crate::types::GgmlType;

    use super::{PrecisionPolicy, TensorRole};

    const MIXTRAL_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf";
    const OPENCHAT_PATH: &str =
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf";

    fn parse_metadata(path: &str, caps: &[usize]) -> Option<crate::pipe::ParsedGguf> {
        let path = std::path::Path::new(path);
        if !path.exists() {
            eprintln!("skipping: no host-local gguf fixture at {}", path.display());
            return None;
        }
        let mut file = std::fs::File::open(path).expect("open host-local gguf fixture");
        let mut header_buf = alloc::vec::Vec::new();
        for &cap in caps {
            header_buf.resize(cap, 0);
            file.seek(SeekFrom::Start(0)).expect("seek to file start");
            let read = file.read(&mut header_buf).expect("read gguf header region");
            header_buf.truncate(read);
            if let Ok(parsed) = parse_complete(&header_buf) {
                return Some(parsed);
            }
        }
        panic!("gguf metadata region for {} did not fit in the largest cap", path.display());
    }

    /// Applies [`PrecisionPolicy::llama_cpp_q4_k_s_moe_8_expert`] to every
    /// tensor in the real Mixtral directory and reports, per role, how many
    /// tensors the policy's target type matches the file's actual stored
    /// type. Uniform roles (`attn_norm`, `ffn_norm`, `output_norm`,
    /// `ffn_gate_inp`, `token_embd`, `output.weight`, `attn_k`, `attn_v`,
    /// `attn_output`) must hit 100%; `ffn_down` is llama.cpp's one
    /// layer-conditional exception (first `n_layer/8` layers bumped to
    /// `Q5_K` across every expert) and is asserted at its real majority
    /// (224/256 = 87.5%) rather than 100%, with the exact miss count
    /// reported so the gap is visible, not hidden.
    #[test]
    fn policy_matches_real_mixtral_tensor_types_by_role() {
        let Some(parsed) = parse_metadata(MIXTRAL_PATH, &[16 << 20, 64 << 20, 128 << 20]) else {
            return;
        };
        let policy = PrecisionPolicy::llama_cpp_q4_k_s_moe_8_expert();

        let mut role_total: alloc::collections::BTreeMap<TensorRole, u32> = alloc::collections::BTreeMap::new();
        let mut role_match: alloc::collections::BTreeMap<TensorRole, u32> = alloc::collections::BTreeMap::new();

        for tensor in &parsed.tensors {
            let role = TensorRole::classify(&tensor.name);
            *role_total.entry(role).or_default() += 1;
            if policy.target_for_role(role) == tensor.ggml_type {
                *role_match.entry(role).or_default() += 1;
            }
        }

        for (role, total) in &role_total {
            let matched = role_match.get(role).copied().unwrap_or(0);
            eprintln!("mixtral_policy_coverage role={role:?} matched={matched}/{total}");
        }

        let uniform_roles = [
            TensorRole::AttnNorm,
            TensorRole::FfnNorm,
            TensorRole::OutputNorm,
            TensorRole::FfnGateInp,
            TensorRole::TokenEmbd,
            TensorRole::OutputWeight,
            TensorRole::AttnK,
            TensorRole::AttnV,
            TensorRole::AttnOutput,
        ];
        for role in uniform_roles {
            let total = role_total.get(&role).copied().unwrap_or(0);
            let matched = role_match.get(&role).copied().unwrap_or(0);
            assert_eq!(matched, total, "role {role:?} expected 100% policy match, got {matched}/{total}");
        }

        let ffn_down_total = role_total.get(&TensorRole::FfnDown).copied().unwrap_or(0);
        let ffn_down_matched = role_match.get(&TensorRole::FfnDown).copied().unwrap_or(0);
        assert_eq!(ffn_down_total, 256, "mixtral has 32 layers x 8 experts of ffn_down");
        assert_eq!(
            ffn_down_matched, 224,
            "ffn_down policy match should be exactly the non-bumped majority (224/256), got {ffn_down_matched}"
        );
    }

    /// Same coverage check against the dense 7B checkpoint, using the plain
    /// (non-MoE) default: `attn_k` is uniform, `attn_output` and `ffn_down`
    /// each have a small layer-conditional minority llama.cpp bumps to
    /// `Q5_K`, reported rather than asserted at 100% for those two.
    #[test]
    fn policy_matches_real_openchat_tensor_types_by_role() {
        let Some(parsed) = parse_metadata(OPENCHAT_PATH, &[4 << 20, 16 << 20, 64 << 20]) else {
            return;
        };
        let policy = PrecisionPolicy::llama_cpp_q4_k_s();

        let mut role_total: alloc::collections::BTreeMap<TensorRole, u32> = alloc::collections::BTreeMap::new();
        let mut role_match: alloc::collections::BTreeMap<TensorRole, u32> = alloc::collections::BTreeMap::new();
        for tensor in &parsed.tensors {
            let role = TensorRole::classify(&tensor.name);
            *role_total.entry(role).or_default() += 1;
            if policy.target_for_role(role) == tensor.ggml_type {
                *role_match.entry(role).or_default() += 1;
            }
        }
        for (role, total) in &role_total {
            let matched = role_match.get(role).copied().unwrap_or(0);
            eprintln!("openchat_policy_coverage role={role:?} matched={matched}/{total}");
        }

        assert_eq!(
            role_match.get(&TensorRole::AttnK).copied().unwrap_or(0),
            role_total.get(&TensorRole::AttnK).copied().unwrap_or(0),
            "attn_k is uniform Q4_K in the dense 7B checkpoint"
        );
        assert_eq!(
            role_match.get(&TensorRole::AttnNorm).copied().unwrap_or(0),
            role_total.get(&TensorRole::AttnNorm).copied().unwrap_or(0)
        );
        assert_eq!(
            role_match.get(&TensorRole::OutputWeight).copied().unwrap_or(0),
            role_total.get(&TensorRole::OutputWeight).copied().unwrap_or(0),
            "output.weight is Q6_K in both checkpoints"
        );

        // no tensor should classify as Other in either real checkpoint --
        // every name this policy has seen from llama.cpp is covered.
        assert_eq!(role_total.get(&TensorRole::Other).copied().unwrap_or(0), 0);
    }

    /// Cross-model sanity: `output.weight`'s target type must agree between
    /// the MoE-conditioned and dense default policies (`Q6_K` in both real
    /// files), proving the two `PrecisionPolicy` constructors aren't
    /// silently diverging on a role neither of them was meant to change.
    #[test]
    fn output_weight_target_agrees_across_policy_variants() {
        let dense = PrecisionPolicy::llama_cpp_q4_k_s();
        let moe = PrecisionPolicy::llama_cpp_q4_k_s_moe_8_expert();
        assert_eq!(dense.output_weight, GgmlType::Q6_K);
        assert_eq!(moe.output_weight, GgmlType::Q6_K);
        assert_eq!(dense.target_for_role(TensorRole::OutputWeight), GgmlType::Q6_K);
    }
}
