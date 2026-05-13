//! Plugin registration trait surface. Used by external plugin crates to
//! register pipe factories without depending on the umbrella `proxima`
//! crate. `AppBuilder` in the umbrella implements this trait so the
//! existing `plugin::register(app_builder)` call sites keep working.

#![cfg(feature = "alloc")]

use proxima_core::ProximaError;

use crate::pipe::pipe_factory::DynPipeFactory;

/// Surface a plugin can register against. Equivalent to the relevant
/// subset of methods on `AppBuilder` in the umbrella `proxima` crate.
/// Adding methods here is breaking; keep the surface minimal.
pub trait PluginRegistry: Sized {
    /// Register a pipe factory under the name embedded in the factory.
    /// Returns the builder with the factory registered. Errors if a
    /// factory with the same name is already registered.
    ///
    /// # Errors
    /// Returns `ProximaError` if registration fails (duplicate name,
    /// invalid factory shape, etc.).
    fn with_upstream_factory(self, factory: DynPipeFactory) -> Result<Self, ProximaError>;
}
