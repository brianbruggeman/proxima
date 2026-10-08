# NPU lane census for omega (research only; no recommendations)

Date of research: 2026-10-08. Every claim carries a [n] pointing at the Sources list. "unsourced" = no primary source found this session.
Search/fetch tooling returned page-text summaries; PDF fetches failed (compressed streams), so paper numbers come from arXiv abstract/HTML pages.

## Owner's three Hexagon names, resolved

| Owner term | What it maps to | Source |
|---|---|---|
| "exagonrpc" | No project of that name found. Closest: Hexagon FastRPC (Qualcomm CPU<->DSP RPC; IDL compiled by QAIC into stub (CPU) + skel (DSP)). Mapping of the spoken name is a guess. | [1][2] |
| "hexagonmlir" | qualcomm/hexagon-mlir, "Hexagon-MLIR", MLIR compiler stack for Triton kernels and PyTorch (torch-mlir) on Hexagon NPUs; paper arXiv 2602.19762 | [3][4][5] |
| "raw-fastrpc" | FastRPC used directly: hand-written IDL + skel on cDSP, no QNN/SNPE. Used this way by llama.cpp ggml-hexagon (CPU lib libggml-hexagon + DSP lib libggml-htp-vNN) | [1][2][21] |

## Part 1. Lane table

| # | Lane | Hardware | Entry API | Kind | Dtypes / quantized weights | Shape constraints | Published dispatch floor | Shared mem / zero-copy | Concurrency w/ GPU |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Apple ANE via CoreML | Apple ANE | MLModel / MLComputeUnits.cpuAndNeuralEngine (Swift); coremltools ComputeUnit.CPU_AND_NE [6] | compiled model (ahead of time; placement decided at compile time [7]) | fp16 native [7][8]. Palettization 1/2/3/4/6/8-bit works best on ANE; W8A8 int8 gives latency gain on A17 Pro/M4; per-block int4 "works really well" on Mac GPU [6]. GGUF Q4_0/Q8_0/Q4_K blocks: no direct consumption stated; coremltools has its own formats (conversion implied, no source says GGUF) | static shapes in the ANE MoE paper [7]; other limits unsourced for the public path | no absolute CoreML number found. Private-path measurements in lane 2 | MLMultiArray/IOSurface path for CoreML: unsourced | ANE leaves GPU free: stated, unmeasured [8]; GPU->ANE sequential pipeline over shared IOSurface in [11] |
| 2 | Apple ANE non-CoreML | Apple ANE | private _ANEClient, _ANECompiler, _ANEInMemoryModel(Descriptor), _ANERequest, MIL IR [9][10][11]. Private, undocumented: not shippable via App Store (shippability: unsourced, private status itself sourced) | compiled kernel (MIL compiled to E5 microcode) [9] | fp16 native, +-65,504 range [8]. maderix README: INT8 W8A8 supported (1.88x over fp16; constexpr_affine_dequantize) [10]. Orion paper: "quantization (INT8/INT4) is not yet supported" and "INT8 saves only memory bandwidth, not compute cycles" [8] (two sources disagree) | fixed [1,C,1,S] fp16 IOSurface layout; min IOSurface ~49 KB; uniform allocation for multi-I/O; conv with ~32,000 channels rejected; ~119 compiles/process limit then silent failure; multi-input requests error 0x1d in maderix [8][10] | Orion: ~0.095 ms dispatch overhead (Sec 1) vs ~0.03 ms bare XPC+IOKit (Fig 10); ~2.3 ms IOSurface round trip per dispatch; ANE decode 5.76 ms/tok vs CPU 3.48 ms/tok (Table 9); recompile 4,200 -> 494 ms/step; ANECCompile ~30-80 ms/kernel [8] | IOSurface-backed tensors [8][10] | maderix: GPU prefill then ANE decode, sequential; no simultaneous claim [10]. Orion: no concurrent measurements [8] |
| 3 | Hexagon raw FastRPC + Hexagon SDK | Qualcomm Hexagon cDSP (HVX, HMX) | FastRPC: IDL -> QAIC -> stub/skel; remote_handle64_open; rpcmem (ION/dma-heap) [1][2][12] | raw kernel (hexagon-clang, HVX/HMX intrinsics: toolchain detail unsourced here) | llama.cpp ggml-hexagon: repacks Q4_0, Q8_0, MXFP4 into non-host buffers; Q8_0 in get/set-rows and flash-attn (PR 26501) [21][22]. HMX specifics: unsourced | one NPU session ~3.5 GB virtual address space (README); 4 GiB limit on V75/V79 per RFC #26227 [21][22] | ~100 us/FastRPC call counted in an nntrainer PR; ~82-92 us after optimization; 527.7-587.7 us for a heavy MoE call (marshalling) in nntrainer forks. These are community repos, not vendor numbers [13][14] | rpcmem / FASTRPC_MAP_STATIC maps a buffer once [15]; ION/dma-heap [1][12] | cDSP alongside Adreno: llama.cpp lists CPU, Adreno (OpenCL), Hexagon as separate devices [21]; HeteroLLM measures GPU+NPU concurrent [16] |
| 4 | Hexagon via QNN / AI Engine Direct | Hexagon HTP | QNN C API (model lib .so, context binary .bin, DLC) [17]; ONNX Runtime QNN EP [17] | compiled model/graph (context binary is HTP-specific, not SoC-agnostic) [17] | HTP supports quantized types (uint8/uint16 typical; MatMul on HTP: (u8,u8),(u8,u16),(u16,u8)); fp16 execution via enable_htp_fp16_precision; ExecuTorch QNN schemes 8a8w, 16a16w, 16a8w, 16a4w, 16a4w_block [17][18]. Block-quantized GGUF consumed directly: unsourced | ahead-of-time compiled, fixed context lengths (separate prefill and decode graphs) in rwkv-qualcomm and llm.npu [17][19]; formal static-shape statement: unsourced | Odin 3 (V79) INT8 execute 3.4-3.5 ms per graph (qnn-net-run, 20 runs; community repo) [14]. Vendor floor: unsourced | llm.npu: shared buffers between processors [19] | HeteroLLM: GPU-NPU sync ~400 us; concurrent bandwidth 43.3 -> 59.5 GB/s [16] |
| 5 | Hexagon via TVM | Hexagon DSP / simulator | tvm.target.hexagon('v68'...), Hexagon launcher (FastRPC based) [23][24] | compiled kernel / model (LLVM codegen; HVX via LLVM intrinsics) | HMX: not covered in sources found. Block-quant: unsourced | not stated in sources found | launcher "executes one layer at a time", "no performance optimizations" (PR 8986) [24]. No latency number found | rpcmem/ION via FastRPC [23] | unsourced |
| 6 | Hexagon via MLIR (hexagon-mlir) | Hexagon v73/v75/v79/v81 (HVX); matmul via Hexagon Kernel Library (HexKL, experimental, not valid on v81) [3][4][25] | Triton kernels and PyTorch via torch-mlir; compile + run scripts, device via ANDROID_HOST/ANDROID_SERIAL [25]. Launch mechanism: unsourced (user guide does not state) | compiled kernel ("mega-kernels" keeping data in TCM, DMA DDR<->TCM) [3] | paper results tables show float16/float32 only; int4/int8 not discussed; no end-to-end LLM results in the paper [5] | fixed example shapes; no explicit constraint stated [5] | none reported [5] | TCM + DMA [3] | unsourced |
| 7a | Intel NPU | Intel Core Ultra NPU | OpenVINO NPU plugin, Level Zero driver [26] | compiled model | INT4-FP16 symmetric channel-wise or group-wise (Core Ultra Series 1); NF4-FP16 enabled; INT8 weight-only unsupported (issue reports crash at decode) [26] | compiler rejects dynamic shapes and ReadValue/Assign state; state emulated with separate buffers; NPUW prefill chunk size default 1024 [26] | unsourced | Level Zero remote tensors [26] | unsourced |
| 7b | AMD XDNA (Ryzen AI) | XDNA / XDNA2 (32 cores, 4x8) | MLIR-AIE / IRON Python API -> final.xclbin + insts.txt [27] | raw kernel (hand-placed dataflow + AIE C++) | int8, int16, bf16 native; bfp16 on XDNA2 (8 values share 1 exponent); amd/IRON GEMM keeps B as bfp16 on AIE2P [27] | amd/IRON GEMM has M,K,N as runtime params, one xclbin for all shapes [27] | no vendor figure found; community xdna-engine: configures/token 366 -> 170 gave 74.6 -> 54.8 ms/token (Qwen3-0.6B) [27] | unsourced | unsourced |

## Per-lane notes

Lane 1 (ANE via CoreML)
- Apple's own Llama 3.1 CoreML post targets the GPU, not the ANE: M1 Max, macOS Sequoia 15.2 beta, "specifically target the GPU"; int4 block-wise (block size 32); 16 GB -> 4.2 GB; 0.19 / 1.25 / 16.26 / 33.67 tok/s for no-cache / KV-as-IO / KV-as-state / int4 [28].
- coremltools: palettization nbits in {1,2,3,4,6,8}; "weight palettization typically works the best on the Neural Engine"; W8A8 uses int8-int8 path on A17 Pro/M4 [6].
- NPUMoE (Apple Silicon): attention (QKV, softmax, stateful KV update) and expert FFN on NPU with static shapes via CoreML; routing/top-k/layerNorm/scatter-gather and cold experts on CPU; "ANE dequantizes INT8 to FP16 before computation"; GPU unused; ~1.2 GB compute graph on M2 Ultra "falls back to CPU" [7].

Lane 2 (ANE private)
- Orion (arXiv 2603.06728): 20 restrictions on MIL programs, 14 previously undocumented; M4 Max 170+ tok/s GPT-2 124M; 3.8x training speedup [8]. maderix/ANE: "15.8 TFLOPS FP16 (M4)" spec line; measured peaks 18.6 TOPS fp16, 35.1 TOPS int8 W8A8 [10]. Orion paper itself states GPU (MLX/Metal) "currently achieves higher absolute throughput for LLM inference than the ANE" [8].
- ane-infer (community) claims hybrid ANE+GPU+CPU, 32 tok/s [11]; claim from README, not verified.

Lane 3 (raw FastRPC)
- qualcomm/fastrpc: userspace lib for CPU<->DSP RPC; IDL via QAIC -> headers, stub, skel [1]. FastRPC user guide 80-N7039-2: ION allocator for contiguous memory; for IDL "in" params CPU flushes cache for the buffer [12]. Hexagon SDK docs (Technologies_FastRPC.html) ship only with the SDK: not read.
- llama.cpp Hexagon backend: devices HTP0..N; libggml-htp-v73/v75/v79/v81; HVX thread count is a runtime knob; sample logs ~21 ms/token Gemma (Android), Llama-3.2-1B Q4_0 ~169 tok/s pp128, ~52 tok/s tg64 (single runs) [21].
- nntrainer forks cite llama.cpp ggml-hexagon making 1 FastRPC call per forward pass (~0.2 ms dispatch) and dspqueue as the alternative transport [13][14].

Lane 4 (QNN)
- llm.npu (ASPLOS'25): built on QNN + MLLM, ~10K lines C/C++/asm; extra operators (KVCache, SiLU, RMSNorm, RoPE) implemented because QNN lacks them; chunk length 256 on Xiaomi 14; 22.4x avg prefill speedup, >1,000 tok/s prefill for billion-scale model; "Hexagon is the only mobile NPU with an open ISA" (authors' statement) [19].
- HeteroLLM/HeteroInfer (Snapdragon 8 Gen 3): NPU 34 TOPS INT8; FP16 figure is the authors' estimate (half); W4A16 not supported on NPU for decoding (footnote); graph preparation at length 135 = 408.4 ms (34.6% of latency); operand order [14336,4096]x[4096,K] ~6x faster than reversed; FFN-down NPU 0.5x-1.5x GPU [16].

Lane 5 (TVM)
- Requires Hexagon SDK >= 4.0.0 (README); runtime dir has hexagon_hvx.cc, hexagon_vtcm_pool.h, rpc/ [23]. RFC: ARM<->Hexagon via FastRPC, stub/skel libs [23]. Issue 17195 shows remote_handle64_open failure (0x72) on a skel URI [2].

Lane 6 (hexagon-mlir)
- Paper: lowering ends at "LLVM-IR with runtime calls"; multithreading via MLIR Async dialect; Flash Attention 4.7x vectorization speedup (float32, N_CTX=1024, DIM_HEAD=64, BLOCK_N=64); multithreading 2.28x (32K elems) to 3.95x (512K) [5]. LLVM DevMtg 2025 slides report LLM and Triton kernel demos (flash attention, softmax, argmax, matmul) [4]. Features: HVX vectorization, TCM, DMA, HexKL matmul (experimental) [3].

Lane 7
- Intel: LLMPipeline(ir,"NPU") with INT8 weight-only IR accepted then crashes (issue 35641) [26]. static-shape check must re-run after INT4 compression because decompression subgraph can reintroduce a dynamic dim [26].
- AMD: MLIR-AIR fused MHA for LLaMA 2: 834 us unfused -> 373 us fused (community aggregator citation) [27].

## Part 2a. Split by stage (front on NPU, expert FFN on GPU)

Searched: no paper found that places attention/router on an NPU and expert FFNs on a GPU and reports per-token handoff cost (search result stated this gap; I did not find one either). Nearest measured items:

- NPUMoE (arXiv 2604.18788): attention + hot-expert FFN on NPU, routing/cold experts on CPU, GPU idle. Latency 1.32x-5.55x lower, energy 1.81x-7.37x better, CPU cycles 1.78x-5.54x lower (Apple M-series, three MoE LLMs, four long-context workloads). CPU-NPU sync ">60% of runtime in worst case"; naive CoreML spends 4.44x-5.69x more energy on data communication. Grouping 8x32 vs 1x32 experts: 0.26 vs 0.31 ms/token. No per-token handoff in ms reported [7].
- HeteroLLM (arXiv 2501.14794), Snapdragon 8 Gen 3, dense LLMs (not MoE), GPU+NPU tensor partition: GPU explicit sync ~400 us fixed; "tens of microseconds" in prefill with polling; sleep granularity 80-100 us; 1.34x-6.02x speedup [16].
- llm.npu: NPU runs static-shape dense blocks; outlier tensors and attention-related ops on CPU/GPU; shared buffers synchronize intermediates [19]. A later paper notes HeteroLLM, like llm.npu, still runs attention on CPU/GPU [16-review].
- Orion: ANE round trip ~2.3 ms per dispatch via IOSurface; ANE decode 5.76 ms/tok vs CPU 3.48 [8].
- GPU-to-GPU attention/expert disaggregation (JANUS arXiv 2512.13525, up to 4.7x per-GPU throughput; MegaScale-Infer arXiv 2504.02263): different hardware class, listed only as the stage-split literature found [29].
- DraftExpert (arXiv 2607.24434): experts staged from Flash to mobile NPU; expert loading dominates decode [29].

## Part 2b. NPU/predictor-driven routing and expert prefetch

| Paper | Signal | Hit rate | Speedup | Source |
|---|---|---|---|---|
| Pre-gated MoE (ISCA 2024, Hwang et al.) | Learned pre-gate in block N selects experts for block N+1; input is the current block's activations (implied, Sec VI-D; not formally specified in text read). First block uses two gates; last block none | not reported as a hit rate; accuracy "comparable" to original, small degradation on some downstream tasks on larger models | latency avg 1.7x (max 1.9x) vs MoE-OnDemand, avg 42x (max 125x) vs MoE-Prefetch; throughput avg 1.5x (max 1.6x) vs OnDemand, 27x (max 55x) vs Prefetch; peak GPU memory 23% of GPU-only (Switch-Base configs) | [30] |
| MoE-Infinity (arXiv 2401.14361) | Request-level Expert Activation Matrix traces; cosine distance vs stored EAMs; summed/normalized with layer-proximity factor (1-(i-l)/L); drives prefetch and eviction | no percentages reported | 3.1-16.7x per-token latency (abstract) vs vLLM, Ollama, DeepSpeed, BrainStorm; conclusion states 2.7-13.7x; Table 1 DeepSeek-V2-Lite TPOT: 155 ms vs vLLM 485, DeepSpeed 737, Mixtral-Offloading 1250, Ollama 2590; RTX A5000 24 GB | [31] |
| EdgeMoE (arXiv 2308.14352) | Offline-profiled table keyed on expert activation status of the two previous MoE layers -> next-layer expert probability; preloads 1-3 experts/layer | one example probability 87.1%; top 20% of activation paths cover >99% of activations; overall hit rate not given in text | 1.19x-2.77x inference speedup vs memory-optimized baselines; 2.64x-3.03x vs IO-EXP on Jetson TX2; Jetson TX2, Raspberry Pi 4B, Xiaomi 14 | [32] |
| SiDA-MoE (MLSys 2024, arXiv 2310.18859) | Offline-trained hash (2-layer LSTM + sparse attention) from input token embeddings (not hidden states) predicts all activated experts per token | top-3: Switch-base-8 99.00/97.41/91.74% (SST2/MRPC/MultiRC); base-128 98.78/98.65/90.49% | up to 3.93x throughput (Switch-base-256, SST2), 72% latency reduction, 80% GPU memory saving, down to 1% performance drop; MultiRC gains 1.26x-1.57x | [33] |
| MoE-Beyond, ST-MoE | named in search results as later learned predictors / prefetchers; numbers not read | unread | ST-MoE: 1.5x over Pre-gated MoE (as reported by search summary, paper not opened) | [34] |

Note: none of these run the predictor on an NPU; all were measured on GPU/CPU/Jetson systems.

## Sources
[1] https://github.com/qualcomm/fastrpc
[2] https://github.com/apache/tvm/issues/17195
[3] https://github.com/qualcomm/hexagon-mlir
[4] https://llvm.org/devmtg/2025-10/slides/quick_talks/baskaran_slama.pdf
[5] https://arxiv.org/html/2602.19762v1
[6] https://apple.github.io/coremltools/docs-guides/source/opt-overview.html
[7] https://arxiv.org/html/2604.18788
[8] https://arxiv.org/html/2603.06728 (abstract: https://arxiv.org/abs/2603.06728)
[9] https://github.com/mechramc/Orion
[10] https://github.com/maderix/ANE
[11] https://github.com/thebasedcapital/ane-infer
[12] https://usermanual.wiki/Pdf/80N70392FASTRPCDEBUGGUIDE.399075392/html
[13] https://github.com/nntrainer/nntrainer/pull/4238
[14] https://github.com/dlwlzzero/nntrainer/issues/88 ; https://github.com/mayusi/odin3-npu-teardown/issues/1
[15] https://github.com/dlwlzzero/nntrainer/pull/52
[16] https://arxiv.org/html/2501.14794 (HeteroLLM)
[17] https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html ; https://github.com/MollySophia/rwkv-qualcomm
[18] https://docs.pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html
[19] https://arxiv.org/abs/2407.05858 (llm.npu)
[21] https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md
[22] https://github.com/ggml-org/llama.cpp/pull/26501
[23] https://github.com/apache/tvm/tree/main/src/runtime/hexagon ; https://discuss.tvm.apache.org/t/introducing-hexagon-backend/2421
[24] https://github.com/apache/tvm/pull/8986
[25] https://github.com/qualcomm/hexagon-mlir/blob/main/docs/user-guide.md
[26] https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html ; https://github.com/openvinotoolkit/openvino/issues/35641 ; https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md
[27] https://github.com/amd/IRON ; https://github.com/atassis/xdna-engine ; https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-isca-2025.pdf
[28] https://machinelearning.apple.com/research/core-ml-on-device-llama
[29] https://arxiv.org/pdf/2512.13525 ; https://arxiv.org/pdf/2504.02263 ; https://arxiv.org/html/2607.24434v1
[30] https://arxiv.org/html/2308.12066v3 (ISCA 2024 PDF: https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf)
[31] https://arxiv.org/html/2401.14361
[32] https://arxiv.org/html/2308.14352
[33] https://arxiv.org/html/2310.18859 ; https://arxiv.org/abs/2310.18859
[34] https://arxiv.org/pdf/2508.17137 ; https://www.alphaxiv.org/abs/2606.15453
