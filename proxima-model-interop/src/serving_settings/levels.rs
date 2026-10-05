use bon::Builder;
use conflaguration::Settings;
use serde::{Deserialize, Serialize};

use crate::serving::AdmissionSchedule;

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
