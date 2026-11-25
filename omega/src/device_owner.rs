//! The single-owner GPU device registry and its command vocabulary.
//!
//! One task owns the device and every plan registered on it; callers talk to it
//! through a bounded [`proxima_primitives::sync::mpsc`] queue of
//! [`OwnerCommand`]s and receive replies on [`proxima_primitives::sync::oneshot`]
//! channels. The registry is generic over the plan type the owner drives.

use std::sync::Arc;
use std::vec::Vec;

use proxima_primitives::sync::oneshot;
use proxima_tensor::NodeId;

use crate::{CudaDeviceSnapshot, CudaDriverError};

/// Service-scoped identity for a registered plan generation.
#[derive(Clone, Debug)]
pub struct PlanLease {
    service: Arc<()>,
    slotindex: usize,
    generation: u64,
}

impl PlanLease {
    /// Generation assigned by this service when the plan was registered.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlanRegistryError {
    #[error("cuda plan registry is full")]
    Full,
    #[error("cuda plan lease belongs to another service")]
    WrongService,
    #[error("cuda plan lease is stale")]
    Stale,
    #[error("cuda plan generation has been retired")]
    Retired,
    #[error("cuda plan already has the maximum number of accepted jobs")]
    InFlightCountExhausted,
    #[error("cuda owner command queue is full")]
    QueueFull,
    #[error("cuda owner stopped before registering the plan")]
    OwnerStopped,
    #[error("cuda service closed after execution failure")]
    ServiceClosed,
    #[error("cuda graph has more input nodes than the node id can represent")]
    ProgramTooLarge,
}

struct PlanSlot<Plan> {
    generation: u64,
    plan: Option<Plan>,
    in_flight: usize,
    retired: bool,
}

/// Proof that one job was accepted against a plan generation.
pub struct PlanJobLease {
    plan: PlanLease,
}

pub struct PlanRegistry<Plan> {
    service: Arc<()>,
    capacity: usize,
    slots: Vec<PlanSlot<Plan>>,
    free: Vec<usize>,
}

impl<Plan> PlanRegistry<Plan> {
    #[must_use]
    pub fn new(service: Arc<()>, capacity: usize) -> Self {
        Self {
            service,
            capacity,
            slots: Vec::with_capacity(capacity),
            free: Vec::with_capacity(capacity),
        }
    }

    pub fn register(&mut self, plan: Plan) -> Result<PlanLease, PlanRegistryError> {
        let slotindex = match self.free.pop() {
            Some(slotindex) => slotindex,
            None if self.slots.len() < self.capacity => {
                let slotindex = self.slots.len();
                self.slots.push(PlanSlot {
                    generation: 0,
                    plan: None,
                    in_flight: 0,
                    retired: false,
                });
                slotindex
            }
            None => return Err(PlanRegistryError::Full),
        };
        let Some(slot) = self.slots.get_mut(slotindex) else {
            return Err(PlanRegistryError::Stale);
        };
        slot.plan = Some(plan);
        slot.in_flight = 0;
        slot.retired = false;
        Ok(PlanLease {
            service: Arc::clone(&self.service),
            slotindex,
            generation: slot.generation,
        })
    }

    pub fn begin_job(&mut self, lease: &PlanLease) -> Result<PlanJobLease, PlanRegistryError> {
        if !Arc::ptr_eq(&self.service, &lease.service) {
            return Err(PlanRegistryError::WrongService);
        }
        let Some(slot) = self.slots.get_mut(lease.slotindex) else {
            return Err(PlanRegistryError::Stale);
        };
        if slot.generation != lease.generation || slot.plan.is_none() {
            return Err(PlanRegistryError::Stale);
        }
        if slot.retired {
            return Err(PlanRegistryError::Retired);
        }
        let Some(in_flight) = slot.in_flight.checked_add(1) else {
            return Err(PlanRegistryError::InFlightCountExhausted);
        };
        slot.in_flight = in_flight;
        Ok(PlanJobLease {
            plan: lease.clone(),
        })
    }

    pub fn resolve_job_mut(&mut self, job: &PlanJobLease) -> Result<&mut Plan, PlanRegistryError> {
        if !Arc::ptr_eq(&self.service, &job.plan.service) {
            return Err(PlanRegistryError::WrongService);
        }
        let Some(slot) = self.slots.get_mut(job.plan.slotindex) else {
            return Err(PlanRegistryError::Stale);
        };
        if slot.generation != job.plan.generation || slot.in_flight == 0 {
            return Err(PlanRegistryError::Stale);
        }
        slot.plan.as_mut().ok_or(PlanRegistryError::Stale)
    }

    pub fn complete_job(&mut self, job: PlanJobLease) -> Result<(), PlanRegistryError> {
        if !Arc::ptr_eq(&self.service, &job.plan.service) {
            return Err(PlanRegistryError::WrongService);
        }
        let Some(slot) = self.slots.get_mut(job.plan.slotindex) else {
            return Err(PlanRegistryError::Stale);
        };
        if slot.generation != job.plan.generation || slot.in_flight == 0 {
            return Err(PlanRegistryError::Stale);
        }
        slot.in_flight -= 1;
        let release = slot.retired && slot.in_flight == 0;
        if release {
            self.release_slot(job.plan.slotindex);
        }
        Ok(())
    }

    pub fn retire(&mut self, lease: &PlanLease) -> Result<(), PlanRegistryError> {
        if !Arc::ptr_eq(&self.service, &lease.service) {
            return Err(PlanRegistryError::WrongService);
        }
        let Some(slot) = self.slots.get_mut(lease.slotindex) else {
            return Err(PlanRegistryError::Stale);
        };
        if slot.generation != lease.generation || slot.plan.is_none() {
            return Err(PlanRegistryError::Stale);
        }
        slot.retired = true;
        let release = slot.in_flight == 0;
        if release {
            self.release_slot(lease.slotindex);
        }
        Ok(())
    }

    fn release_slot(&mut self, slotindex: usize) {
        let Some(slot) = self.slots.get_mut(slotindex) else {
            return;
        };
        slot.plan = None;
        slot.retired = false;
        if let Some(next_generation) = slot.generation.checked_add(1) {
            slot.generation = next_generation;
            self.free.push(slotindex);
        }
    }
}

/// Runs `execute` against the plan behind `lease`, bracketed by begin and complete.
pub fn execute_registered_plan<Plan, Output, Error>(
    registry: &mut PlanRegistry<Plan>,
    lease: &PlanLease,
    execute: impl FnOnce(&mut Plan) -> Result<Output, Error>,
) -> Result<Result<Output, Error>, PlanRegistryError> {
    let job = registry.begin_job(lease)?;
    let execution = match registry.resolve_job_mut(&job) {
        Ok(plan) => execute(plan),
        Err(error) => {
            registry.complete_job(job)?;
            return Err(error);
        }
    };
    registry.complete_job(job)?;
    Ok(execution)
}

/// Every message the device owner accepts.
///
/// `Graph` is the owned graph a registration plans, `Plan` the device plan an
/// owner registers directly, `Job` an admitted execution, `Completed` its
/// reply, and `GraphError` the failure of planning a graph.
pub enum OwnerCommand<Graph, Plan, Job, Completed, GraphError> {
    DeviceSnapshot {
        reply: oneshot::Sender<Result<CudaDeviceSnapshot, CudaDriverError>>,
    },
    RegisterGraph {
        // boxed: the owned graph is ~300 bytes and would size every command
        graph: Box<Graph>,
        reply: oneshot::Sender<Result<PlanLease, GraphError>>,
    },
    RegisterPlan {
        // boxed: the plan is ~1.2 KB and would size every command; the owner boxes it into its registered plan anyway
        plan: Box<Plan>,
        reply: oneshot::Sender<Result<PlanLease, PlanRegistryError>>,
    },
    RetirePlan {
        lease: PlanLease,
        reply: oneshot::Sender<Result<(), PlanRegistryError>>,
    },
    InvalidateResidents {
        nodes: Vec<NodeId>,
        reply: oneshot::Sender<Result<(), CudaDriverError>>,
    },
    ExecutePlan {
        job: Job,
        reply: oneshot::Sender<Completed>,
    },
}

#[cfg(test)]
mod tests {
    use super::{PlanRegistry, PlanRegistryError, execute_registered_plan};
    use std::sync::Arc;

    #[test]
    fn test_omega_async_plan_register_returns_service_lease() {
        fn assert_send<Type: Send>() {}
        assert_send::<crate::CudaPlan>();

        let service = Arc::new(());
        let mut registry = PlanRegistry::new(Arc::clone(&service), 2);
        let lease = registry.register(41_u64).expect("plan should register");
        let job = registry.begin_job(&lease).expect("valid lease");
        assert_eq!(*registry.resolve_job_mut(&job).expect("valid job"), 41);
        registry.complete_job(job).expect("job should complete");
        assert!(Arc::ptr_eq(&lease.service, &service));
    }

    #[test]
    fn test_omega_async_plan_register_rejects_stale_and_foreign_leases() {
        let service = Arc::new(());
        let mut registry = PlanRegistry::new(Arc::clone(&service), 1);
        let stale = registry.register(7_u64).expect("plan should register");
        registry.retire(&stale).expect("plan should retire");
        let replacement = registry.register(9_u64).expect("slot should be reused");
        assert_eq!(replacement.slotindex, stale.slotindex);
        assert_ne!(replacement.generation, stale.generation);
        assert_eq!(
            registry.begin_job(&stale).err(),
            Some(PlanRegistryError::Stale)
        );

        let mut other_service = PlanRegistry::<u64>::new(Arc::new(()), 1);
        assert_eq!(
            other_service.begin_job(&replacement).err(),
            Some(PlanRegistryError::WrongService)
        );
    }

    #[test]
    fn test_omega_async_plan_register_owner_access_is_serial() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(0_u64).expect("plan should register");
        let job = registry.begin_job(&lease).expect("job should start");
        *registry.resolve_job_mut(&job).expect("valid plan") += 1;
        *registry.resolve_job_mut(&job).expect("valid plan") += 1;
        assert_eq!(*registry.resolve_job_mut(&job).expect("valid plan"), 2);
        registry.complete_job(job).expect("job should complete");
    }

    #[test]
    fn test_omega_async_plan_retire_shape_miss_rejects_new_jobs() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(17_u64).expect("plan should register");
        registry
            .retire(&lease)
            .expect("shape miss should retire plan");
        assert_eq!(
            registry.begin_job(&lease).err(),
            Some(PlanRegistryError::Stale)
        );
    }

    #[test]
    fn test_omega_async_plan_retire_stream_invalidation_rejects_new_jobs() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(23_u64).expect("plan should register");
        registry
            .retire(&lease)
            .expect("stream invalidation should retire plan");
        assert_eq!(
            registry.begin_job(&lease).err(),
            Some(PlanRegistryError::Stale)
        );
        assert_eq!(
            registry.begin_job(&lease).err(),
            Some(PlanRegistryError::Stale)
        );
    }

    #[test]
    fn test_omega_async_plan_retire_keeps_accepted_job_until_completion() {
        let service = Arc::new(());
        let mut registry = PlanRegistry::new(Arc::clone(&service), 1);
        let lease = registry.register(31_u64).expect("plan should register");
        let accepted = registry.begin_job(&lease).expect("job should be accepted");
        registry.retire(&lease).expect("plan should retire");

        assert_eq!(
            registry.begin_job(&lease).err(),
            Some(PlanRegistryError::Retired)
        );
        assert_eq!(
            *registry
                .resolve_job_mut(&accepted)
                .expect("accepted job retains retired plan"),
            31
        );
        assert!(registry.slots[lease.slotindex].plan.is_some());

        registry
            .complete_job(accepted)
            .expect("accepted job should retire");
        assert!(registry.slots[lease.slotindex].plan.is_none());
        let replacement = registry.register(37_u64).expect("slot should be reused");
        assert_eq!(replacement.slotindex, lease.slotindex);
        assert_ne!(replacement.generation, lease.generation);
    }

    #[test]
    fn test_omega_async_cuda_stage_rejects_before_executor_call() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(5_u64).expect("plan should register");
        registry.retire(&lease).expect("plan should retire");
        let mut invoked = false;
        let result = execute_registered_plan(&mut registry, &lease, |_| {
            invoked = true;
            Ok::<_, &'static str>(())
        });

        assert_eq!(result, Err(PlanRegistryError::Stale));
        assert!(!invoked);
    }

    #[test]
    fn test_omega_async_cuda_stage_publishes_after_executor_finishes() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(13_u64).expect("plan should register");
        let mut completed = false;
        let result = execute_registered_plan(&mut registry, &lease, |_| {
            completed = true;
            Ok::<_, &'static str>(19_u64)
        })
        .expect("registry should resolve");

        assert!(completed);
        assert_eq!(result, Ok(19));
        assert_eq!(registry.slots[lease.slotindex].in_flight, 0);
    }

    #[test]
    fn test_omega_async_cuda_stage_preserves_executor_failure_and_retires_job() {
        let mut registry = PlanRegistry::new(Arc::new(()), 1);
        let lease = registry.register(29_u64).expect("plan should register");
        let result = execute_registered_plan(&mut registry, &lease, |_| {
            Err::<(), _>("fake driver failure")
        })
        .expect("registry should resolve");

        assert_eq!(result, Err("fake driver failure"));
        assert_eq!(registry.slots[lease.slotindex].in_flight, 0);
        let job = registry
            .begin_job(&lease)
            .expect("lease should remain live");
        assert_eq!(
            *registry.resolve_job_mut(&job).expect("job should resolve"),
            29
        );
        registry.complete_job(job).expect("job should complete");
    }
}
