# floor — raw-prime baseline

Reference, not a lesson: what proxima's composable abstraction *costs*. A
minimal HTTP/1.1 static-`200 ok` responder on raw prime — the same runtime
proxima's own h1 stack runs on — with no `Pipe`, no `ListenProtocol`, no h1
codec. The gap between this and proxima's full h1 stack is the composable
overhead, runtime held constant.

```rust
{{#include ../../../examples/rust_floor_h1.rs}}
```
