// one binary instead of fifteen avoids linking the whole workspace fifteen times.
#[path = "units/background_pool.rs"]
mod background_pool;
#[path = "units/dx_audit.rs"]
mod dx_audit;
#[path = "units/loom_wake_path.rs"]
mod loom_wake_path;
#[path = "units/no_tokio_in_public_api.rs"]
mod no_tokio_in_public_api;
#[path = "units/prime_tokio_compat.rs"]
mod prime_tokio_compat;
#[path = "units/producer_graph_config.rs"]
mod producer_graph_config;
#[path = "units/producer_lifecycle_app_integration.rs"]
mod producer_lifecycle_app_integration;
#[path = "units/propagation_cross_core.rs"]
mod propagation_cross_core;
#[path = "units/propagation_spawn.rs"]
mod propagation_spawn;
#[path = "units/propagation_wake.rs"]
mod propagation_wake;
#[path = "units/scenario_schema.rs"]
mod scenario_schema;
#[path = "units/scenario_smoke.rs"]
mod scenario_smoke;
#[path = "units/settings_round_trip.rs"]
mod settings_round_trip;
#[path = "units/settings_to_app.rs"]
mod settings_to_app;
#[path = "units/thread_local_composition.rs"]
mod thread_local_composition;
