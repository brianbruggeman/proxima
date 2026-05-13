# boot-time — cold-start latency

Reference, not a lesson: "boot" is a decomposition, not one number. This
binary reports `boot_ns` (main-entry to the prime runtime being ready — the
proxima-specific component a microVM slice pays) and `first_task_ns` (the
first task's own execution, proving ready-for-compute). Run it many times,
fresh process each, for a cold-start distribution.

```rust
{{#include ../../../examples/boot_time.rs}}
```
