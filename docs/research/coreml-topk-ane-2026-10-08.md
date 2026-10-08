# Core ML top-k, gather/scatter, and ANE placement: sourced facts

Status key: proven = opened source text; reported = secondary/paper claim; unsourced = no source found.

## Q1. Is `topk` in the MIL op set, and its constraints?
Yes. Status: proven.
- iOS15 source: `topk` is defined in `tensor_operation` of the iOS15 defs. `input_spec`: `x` (T), `k` (const, optional, int32), `axis` (const, optional, int32), `ascending` (const, optional, bool). Type inference raises `ValueError("K={} is greater than size of the given axis={}")` when `k > x_shape[axis]` (static axis size only). Source: https://raw.githubusercontent.com/apple/coremltools/main/coremltools/converters/mil/mil/ops/defs/iOS15/tensor_operation.py
- iOS17 source: same op re-registered with `opset_version=_IOS17_TARGET`; adds `sort`, `return_indices`, `output_indices_dtype`; `k` domain int8/int16/int32; `k` default 1; `k=-1` selects all elements; `axis` default -1. Source: https://raw.githubusercontent.com/apple/coremltools/main/coremltools/converters/mil/mil/ops/defs/iOS17/tensor_operation.py
- API reference lists `topk` under iOS15 and iOS17 tensor_operation: https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html
- Static k: `k` is declared `const=True`, which is the coremltools flag for a compile-time constant input. Reading of the flag is mine (unsourced as a prose statement).
- Max tensor size: unsourced. No size limit appears in either source file.

## Q2. Which compute unit runs top-k under cpuAndNeuralEngine?
Answer: no ANE-confirmed path for top-k in the sources; a paper says it runs on CPU. Status: reported.
- NPUMoE (arXiv 2604.18788): top-k routing runs on CPU. Sec 3.2: "expert routing (top-k/argmax), and token dispatch/merge operations (scatter, gather, weighted combine) execute on the CPU due to their dynamic indexing and irregular memory access patterns." Sec 1: the NPU "cannot perform arbitrary runtime dependent core MoE operations such as dynamic indexing, tensor reshaping, top-k (router) etc." Abstract: "several irregular operators e.g. top-k, scatter/gather etc. are not NPU-friendly." https://arxiv.org/html/2604.18788v1
- Reverse-engineered ANE account (arXiv 2606.22283, not Apple-documented): Sec 4.4: "On the M1 the top-k, sort, and dynamic-slice validators are all callable, yet the code generator rejects all three." Label: none given for this sentence. Same paper, Sec 4.5: rank and sort "have a bridge route that is rejected at code generation on the M1." Only M1 is cited for top-k; no statement for M2-M5 was found in the fetched text (the fetch was truncated; Appendix A not read in full). https://arxiv.org/html/2606.22283v1
- Apple (hollance/neural-engine, running-on-ane.md): "If possible, Core ML will run the entire model on the ANE." Unsupported layer: "it will switch to another processor." https://github.com/hollance/neural-engine/blob/master/docs/running-on-ane.md
- Apple placement is a preference: "computeUnits = .all" "does not guarantee the model will run on the ANE." Same source.
- Unexplained: a search snippet claimed top-k is native on M2+; I did not open a source that states it. Unsourced.

## Q3. Can gather/scatter with data-dependent indices run on the ANE?
Answer: scatter family is reported as unsupported on all families; gather is restricted to a narrow envelope; data-dependent gather_nd is not covered by any source found. Status: reported.
- Reverse-engineered ANE paper (arXiv 2606.22283) Sec 4.2: gather on M1 "takes a software path valid only for a batch of one, depth of one, and three-element index channel". Sec 4.7: compiler aborts outside that envelope rather than falling back (per the paper). Sec 4.2: "The confirmed cases, unsupported on every family from the M1 through the M5, are: ... the scatter family (scatter, scatter along axis, scatter ND)." Table 4.1: scatter "No path on any family." https://arxiv.org/html/2606.22283v1
- NPUMoE Sec 3.2: scatter and gather for token dispatch/merge run on CPU. https://arxiv.org/html/2604.18788v1
- gather_nd, data-dependent indexing: no statement found in either paper; unsourced.
- Apple's Core ML FAQ suggests writing a translation to existing MIL ops for missing ops (composites), not an ANE-placement guarantee: https://apple.github.io/coremltools/docs-guides/source/faqs.html

## Q4. How NPUMoE keeps expert FFNs on the ANE with static shapes
Source: NPUMoE (arXiv 2604.18788). Status: reported (paper's own design, not measured by me).
- Routing and gather are off-graph (CPU). Sec 4.2 step 4: "Gather routed tokens into expert slices." Sec 4.2: "This layout eliminates dynamic indexing inside the compiled graph: expert-to-token assignment is resolved before invocation." Step 6: runtime "discards padded rows, and cumulates the results back to their original token position."
- Fixed per-expert capacity: Sec 4.1, "each expert is bounded to a fixed token capacity per layer"; "replace dynamic per-expert token counts with estimated fixed capacities proportional to each expert's expected load." Frequent experts get higher tiers.
- Grouped static graph: Sec 1 and 4.2, "batches multiple experts into a single static, grouped dense FFN compute graph"; experts in one group share a capacity tier "to prevent dynamic shapes."
- Overflow: Sec 3.2 and 4.1, overflow tokens are pruned by an activation-based saliency score; overflow cannot be spilled because "Core ML determines the compute graph and device placement statically at compile time."
- Cold experts go to the CPU/GPU fallback path (Sec 1).
- Not stated: a per-token masking scheme for the expert path; dense compute of all experts. The paper does not claim either. https://arxiv.org/html/2604.18788v1

## Q5. Does MLComputePlan report compute device per op?
Yes, the API reports anticipated devices per operation. Status: proven for the API names, reported for the semantics.
- Python (coremltools): class `MLComputePlan`, docstring "Represents the plan for executing a model." Methods: `get_compute_device_usage_for_mlprogram_operation(...) -> Optional[MLComputePlanDeviceUsage]`, `get_compute_device_usage_for_neuralnetwork_layer(...)`, plus estimated cost. Source: https://raw.githubusercontent.com/apple/coremltools/main/coremltools/models/compute_plan.py
- Objective-C/Swift bindings (objc2-core-ml, generated from Apple headers): `computeDeviceUsageForMLProgramOperation`, `computeDeviceUsageForNeuralNetworkLayer`, `estimatedCostOfMLProgramOperation`, `modelStructure`, `loadContentsOfURL:configuration:completionHandler:`. Source: https://docs.rs/objc2-core-ml/latest/objc2_core_ml/struct.MLComputePlan.html
- WWDC24 session 10161 (~15:27): MLComputePlan "surfaces the model structure and runtime information for each operation, including the supported and preferred compute devices, operation support status, and estimated relative cost." The transcript does not name the Swift method and does not say which device each op actually ran on. https://developer.apple.com/videos/play/wwdc2024/10161/
- Swift name `deviceUsage(for:)`: unverified (search snippet only; Apple's method page returned 404 / title-only). Unsourced.
- macOS 14 availability: unsourced. The WWDC24 source is the earliest source found; I did not verify a macOS 14 availability annotation.

## Residual / could not verify
- Top-k placement on M2+ and the M2+ native list (secondary snippet only).
- gather_nd and data-dependent index placement on ANE.
- Swift method name and macOS 14 availability for MLComputePlan (Apple doc page not retrievable).
- No measurement was taken; this is web research only.
