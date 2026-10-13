use alloc::string::String;
use alloc::vec::Vec;
use core::future::Future;

use proxima_primitives::pipe::Pipe;

use super::LoadedModel;
use crate::cuda_async::CudaGpuService;
use crate::error::InteropError;
use crate::serving::ServingConfig;
use proxima_core::ProximaError;

/// A reusable generation Pipe over `(String, usize) -> (Vec<u32>, String, bool)`.
///
/// It captures a loaded model, a shared [`CudaGpuService`], and one serving
/// configuration so callers can compose whole generation requests with the
/// existing [`Pipe`] algebra. When a caller does not need composition, call
/// [`LoadedModel::generate_async_with_serving_config`] directly. Each graph
/// evaluation still passes through Proxima's
/// [`CreditGate`](proxima_primitives::pipe::CreditGate) and execution
/// Pipe before this request future receives its result.
pub struct CudaServingPipe<'pipe, 'file, 'config, Owner> {
    model: &'pipe LoadedModel<'file>,
    service: &'pipe CudaGpuService<Owner>,
    serving_config: ServingConfig<'config>,
}

impl<'pipe, 'file, 'config, Owner> CudaServingPipe<'pipe, 'file, 'config, Owner> {
    /// Bind one loaded model and serving configuration to a shared GPU owner.
    #[must_use]
    pub const fn new(
        model: &'pipe LoadedModel<'file>,
        service: &'pipe CudaGpuService<Owner>,
        serving_config: ServingConfig<'config>,
    ) -> Self {
        Self {
            model,
            service,
            serving_config,
        }
    }
}

impl<'pipe, 'file, 'config, Owner> Pipe for CudaServingPipe<'pipe, 'file, 'config, Owner>
where
    Owner: Future<Output = Result<(), ProximaError>> + Send,
{
    type In = (String, usize);
    type Out = (Vec<u32>, String, bool);
    type Err = InteropError;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            self.model
                .generate_async_with_serving_config(
                    &input.0,
                    input.1,
                    self.serving_config,
                    self.service,
                )
                .await
        }
    }
}
