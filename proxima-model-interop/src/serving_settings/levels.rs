use bon::Builder;
use conflaguration::Settings;
use serde::{Deserialize, Serialize};

use crate::serving::{AdmissionSchedule, ExpertResidencySchedule, PhaseSchedule};

#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[builder(derive(Clone, Debug))]
#[serde(default)]
pub struct AdmissionScheduleSettings {
    /// hard ceiling on `parallel_sequences`; 0 turns the check off
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub max_concurrent_requests: usize,
}

impl Default for AdmissionScheduleSettings {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl AdmissionScheduleSettings {
    pub(super) fn as_admission_schedule(&self) -> AdmissionSchedule {
        AdmissionSchedule {
            max_concurrent_requests: self.max_concurrent_requests,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[builder(derive(Clone, Debug))]
#[serde(default)]
pub struct PhaseScheduleSettings {
    /// `true` finishes a sequence's prefill before its decode loop starts; `false` asks to interleave them across sequences
    #[setting(default = true)]
    #[builder(default = true)]
    pub prefill_before_decode: bool,
}

impl Default for PhaseScheduleSettings {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl PhaseScheduleSettings {
    pub(super) fn as_phase_schedule(&self) -> PhaseSchedule {
        PhaseSchedule {
            prefill_before_decode: self.prefill_before_decode,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[builder(derive(Clone, Debug))]
#[serde(default)]
pub struct ExpertResidencyScheduleSettings {
    /// byte budget for each layer's own resident experts; 0 turns the per-layer cap off
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub per_layer_budget_bytes: u64,
}

impl Default for ExpertResidencyScheduleSettings {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl ExpertResidencyScheduleSettings {
    pub(super) fn as_expert_residency_schedule(&self) -> ExpertResidencySchedule {
        ExpertResidencySchedule {
            per_layer_budget_bytes: self.per_layer_budget_bytes,
        }
    }
}
