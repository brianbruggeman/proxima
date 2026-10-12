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

## Additional lanes (sourced 2026-10-11)

Sourced on 2026-10-11 from primary sources only (vendor docs, vendor repositories pinned to a commit, schema files, and one vendor datasheet). Each claim carries a [n] into the reference list at the end of this section; numbering continues from [34]. Pages fetched through a summarizing fetch tool are marked "summary" and must be re-read raw before a card cites them. "unsourced" = no primary source found.

Question 1: does AMD define a model format of its own? Answer from the sources: for the NPU, no AMD model interchange format is documented; the input is ONNX [37][38], and the compiled artifacts are an EP cache directory with no file extension and an `xclbin` for INT8 CNN models on PHX/HPT [38], plus an `xclbin` and instruction stream from the MLIR-AIE / IRON path [35][36]. For the GPU, MIGraphX defines its own serialized program container, `.mxr`, which is MessagePack by default [40][41]; its byte layout is unsourced beyond "MessagePack".

| lane | formats in | compiled artifact | runtime / incumbent | dtypes | status |
|---|---|---|---|---|---|
| AMD XDNA / XDNA2 via Ryzen AI (ONNX Runtime + Vitis AI EP) | ONNX; quantizer named as AMD Quark, PyTorch or ONNX models [37] | EP cache subdirectory named by cache key, no file extension; `context_model.onnx` when the EP context path is set; `vitisai_ep_report.json` is a report, not a model [38] | `import onnxruntime` with `'VitisAIExecutionProvider'` [38]; the Python package name for the Ryzen AI build is unsourced (the page names none) | INT8 CNN named in the xclbin option row [38]; bfloat16 compile flow for floating-point models [38, summary]; quantized-type list unsourced | sourced for the path; incumbent binary = onnxruntime with the VitisAI EP; no benchmark number sourced |
| AMD XDNA / XDNA2 via MLIR-AIE / IRON / Peano | Python design via `aie.iron` (`@iron.jit`) [35][36] | `xclbin` + instruction stream, compiled by Peano (LLVM for AIE) [35] | XRT: `libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms` installed from the AMD PPA [36]; the XRT CLI names are unsourced | int8 / int16 / bf16 per the census row 7b [27]; bfloat16 operators and bfp16 storage for B on AIE2P (GEMM) [36] | sourced for the artifact; no vendor throughput number sourced |
| AMD GPU via ROCm HIP | HIP C++ source: "HIP is a C++ Runtime API and Kernel Language" [39] | compiler path to LLVM IR and code objects: unsourced in this session | HIP runtime library name: unsourced | unsourced | status: unsourced beyond the HIP definition |
| AMD GPU via MIGraphX | ONNX parsed into a MIGraphX program [40] | `.mxr` serialized program: MessagePack default, JSON optional, via `migraphx::save` / `migraphx::load` [40][41] | `migraphx-driver` (docs and tool exist: `docs/migraphx-driver.rst`, `docs/dev/tools.rst:25`) [42] | unsourced for the program's dtypes | sourced for the container and the incumbent driver; the `.mxr` byte layout and stability are unsourced |
| Rockchip RKNN on RK3588 (Turing RK1 module) | rknn-toolkit2 examples cover caffe, darknet, onnx, pytorch, tensorflow, tflite [47]; the README states the flow "convert the trained model into an RKNN format model" [46] | `.rknn` model file produced by the converter [46] | `librknnrt.so` (path `rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so`) [51]; C API `rknn_api.h` [48]; incumbent binary `rknn_benchmark` (usage: `./rknn_benchmark xxx.rknn [input_data] [loop_count] [core_mask]`; core mask only on RK3588) [49][50] | `RKNN_TENSOR_FLOAT32, FLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, BOOL, INT4, BFLOAT16` in the API enum [48]; SoC spec: "6TOPS NPU ，triple core, support int4/int8/int16/FP16/BF16/TF32 acceleration" [43]; Turing RK1: "6 TOPS" [44, summary] | sourced for converter inputs, runtime, benchmark, dtypes, and TOPS. Open `.rknn` schema: none in the repo tree at 59a913d1 (no `.fbs` or schema file found) and the converter is not in the repo, so the `.rknn` layout is unsourced and must be treated as closed |
| NVIDIA Jetson via TensorRT (engine / plan file) | ONNX parsed by trtexec ("will parse your ONNX file, build a TensorRT plan file" [52]); TensorRT network built by API also measurable via trtexec [52] | serialized TensorRT engine (plan file, `--saveEngine`) [52] | `trtexec` (performance summary printed) [52]; the Jetson-side TensorRT build for each module is unsourced | unsourced | sourced for trtexec behavior; module membership sourced below [45]; Jetson TOPS and DLA support unsourced |
| Raspberry Pi CM4 | none; no NPU or accelerator format | none | none | none | sourced negative: CM4 is "Broadcom quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz" [54]; whole-word search for NPU / neural in the datasheet text returned 0 hits [54] |
| LiteRT `.tflite` and `.litertlm` | `.tflite` FlatBuffer, `file_identifier "TFL3"` [55]; `.litertlm` LLM bundle [56] | `.tflite`: the model table in `schema.fbs` [55]; `.litertlm`: container with a FlatBuffer header schema (sections, `SystemMetadata`, `TFLiteModel` and `TFLiteWeights` section types) [57], magic bytes "TFL3", "PK", or "LITERTLM" checked by the runtime [58] | LiteRT-LM CLI and Python tools `litert-lm-builder` / `litert-lm-peek` [59]; incumbent for LLM bundles = LiteRT-LM run with its backend | unsourced at the per-tensor level | sourced for names, schemas, and magic; the full `.litertlm` section layout beyond the header schema is unsourced |

Per-lane notes (2026-10-11):
- Compute module set for the Turing Pi 2 [45, summary of the official specs table]: Turing RK1 (M.2 NVMe PCIe 3.0 x4, mPCIe, SATA, USB 3, USB 2); Raspberry Pi CM4 (M.2 NVMe not supported; mPCIe, SATA, USB 3, USB 2); one combined Jetson row "Orin NX / Orin Nano / Xavier NX / TX2 NX" (NVMe PCIe 4.0 x4 / 4.0 x4 / 3.0 x4 / 3.0 x2 respectively); Jetson Nano (B01) (M.2 NVMe PCIe 2.0 x1, USB 2 only). The summary states that the Jetson NX and TX2 NX rows are present in the combined row, not absent.
- The CM4 and RK1 rows give the NPU question its answer: RK1 is the only module in this set with a sourced NPU ([43][44]); CM4 has none [54]; the Jetson modules have TensorRT as their sourced inference path only for trtexec's own behavior [52].
- AMD question 1: the Ryzen AI doc names `xclbin` only in the INT8 CNN option row, and names no MLIR-AIE or instruction stream [38]; the xclbin plus instruction stream appears only on the MLIR-AIE / IRON side [35][36]. The two AMD NPU paths therefore produce different artifacts and are not the same lane.

Not sourced this session (named so the next pass knows where to look): the ROCm HIP compiler output format and the LLVM IR lane; the HIP runtime library name; the Ryzen AI inference package name; any `.rknn` byte layout or schema; the RKNN per-operator dtype table (the header enum is sourced, the per-operator table in `doc/RKNNToolKit2_OP_Support-2.3.2.md` at 59a913d1 was located but not read); Jetson DLA support and Jetson TOPS; any Jetson-side TensorRT measurement on these modules; the LiteRT NPU delegate list; any benchmark number for any lane in this section.

References (continued from [34]):
[35] https://raw.githubusercontent.com/Xilinx/mlir-aie/766aca40deaf57b2a12b14e1e15985b878e7faae/README.md (lines 59-60: "`@iron.jit` compiles the design to an `xclbin` + instruction stream via the LLVM/MLIR-based Peano compiler"; line 70: Peano extends LLVM with the AI Engine target)
[36] https://raw.githubusercontent.com/amd/IRON/aae6262bc80636d75d9829ff106037225191e8b7/README.md (line 27: IRON "relies on language bindings around the MLIR-AIE dialect"; line 48: GEMM "B is stored as bfp16 on AIE2P"; line 127: XRT and driver packages)
[37] https://ryzenai.docs.amd.com/en/latest/index.html (summary: ONNX models run through ONNX Runtime; Vitis AI Execution Provider; AMD Quark named as the quantization toolkit for PyTorch and ONNX models)
[38] https://ryzenai.docs.amd.com/en/latest/modelrun.html (summary: `'VitisAIExecutionProvider'`; EP context model `context_model.onnx`; cache subdirectory named by cache key with no extension; `vitisai_ep_report.json`; xclbin option "only when running INT8 CNN models on PHX/HPT devices"; MLIR-AIE and instruction stream absent)
[39] https://raw.githubusercontent.com/ROCm/HIP/master/README.md (line 1 of the description: "HIP is a C++ Runtime API and Kernel Language that allows developers to create portable applications for AMD and NVIDIA GPUs from single source code")
[40] https://raw.githubusercontent.com/ROCm/AMDMIGraphX/793e480f95e8d55acffa00a6243c7beafe517c3d/examples/migraphx/cpp_parse_load_save/README.md (line 18: "serialized to MessagePack (.mxr) format"; line 36: "MessagePack is the default format")
[41] https://raw.githubusercontent.com/ROCm/AMDMIGraphX/793e480f95e8d55acffa00a6243c7beafe517c3d/src/include/migraphx/load_save.hpp (line 36: `std::string format = "msgpack";`; lines 39-49: load, load_buffer, save, save_buffer)
[42] https://raw.githubusercontent.com/ROCm/AMDMIGraphX/793e480f95e8d55acffa00a6243c7beafe517c3d/docs/dev/tools.rst (line 25: "Runs ``migraphx-driver roctx``"); tree entry `docs/migraphx-driver.rst`
[43] https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html (RK3588 specification: "6TOPS NPU ，triple core, support int4/int8/int16/FP16/BF16/TF32 acceleration")
[44] https://docs.turingpi.com/docs/turing-rk1-specs-and-io-ports (summary: "Rockchip RK3588 SoC"; "6 TOPS" of NPU; no precision list, no RKNN mention)
[45] https://docs.turingpi.com/docs/turing-pi2-specs-and-io-ports (summary of the compatibility table: RK1, CM4, combined Jetson Orin NX / Orin Nano / Xavier NX / TX2 NX row, Jetson Nano B01)
[46] https://raw.githubusercontent.com/airockchip/rknn-toolkit2/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/README.md (line 8: "convert the trained model into an RKNN format model, and then inference on the development board using the RKNN C API or Python API"; support list includes RK3588 Series)
[47] https://api.github.com/repos/airockchip/rknn-toolkit2/git/trees/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08?recursive=1 (tree: `rknn-toolkit2/examples/{caffe,darknet,onnx,pytorch,tensorflow,tflite}`)
[48] https://raw.githubusercontent.com/airockchip/rknn-toolkit2/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h (lines 175-186: `rknn_tensor_type` enum)
[49] https://raw.githubusercontent.com/airockchip/rknn-toolkit2/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/rknpu2/examples/rknn_benchmark/README.md (usage line; "Only RK3588 support core mask"; "make sure that the cpu/ddr/npu has been clocked to the highest frequency before testing")
[50] https://raw.githubusercontent.com/airockchip/rknn-toolkit2/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/rknpu2/examples/rknn_benchmark/src/rknn_benchmark.cpp (tree entry; source not read this session)
[51] https://api.github.com/repos/airockchip/rknn-toolkit2/git/trees/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08?recursive=1 (tree entry `rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so`; binary, not read)
[52] https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/benchmarking.html (summary: "trtexec will parse your ONNX file, build a TensorRT plan file, measure the performance of this plan file, and then print a performance summary"; plan file measured separately via trtexec)
[54] https://pip-assets.raspberrypi.com/categories/634-raspberry-pi-compute-module-4/documents/RP-008168-DS-4-cm4-datasheet.pdf (read with `pdftotext -layout`; line 165: "Broadcom quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz"; whole-word NPU / neural count 0)
[55] https://raw.githubusercontent.com/google-ai-edge/LiteRT/aeae743070b3be22533261e4fa5bd3342631f8ee/tflite/converter/schema/schema.fbs (line 33: `file_identifier "TFL3";`; line 35: `file_extension "tflite";`; line 1753: `table Model`)
[56] https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/03d3979f8d6091a8357b122ac8148c231c01f362/README.md (line 1 section: "LiteRT-LM is Google's **production-ready** orchestration layer to run LLMs with LiteRT, engineered for **high-performance**, **cross-platform** execution"; product page https://ai.google.dev/edge/litert-lm as linked there)
[57] https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/03d3979f8d6091a8357b122ac8148c231c01f362/schema/core/litertlm_header_schema.fbs (namespace `litert.lm.schema`; `SystemMetadata`; `AnySectionDataType` incl. `TFLiteModel`, `TFLiteWeights`, `LlmMetadataProto`; `SectionObject`)
[58] https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/03d3979f8d6091a8357b122ac8148c231c01f362/runtime/util/file_format_util.cc (lines 43-47: `TFL3`, `PK`, `LITERTLM` magic checks)
[59] https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/03d3979f8d6091a8357b122ac8148c231c01f362/docs/litert_lm_builder.md (`litert-lm-builder` build / unpack, `litert-lm-peek` inspect)

## Open paths for Rockchip and Qualcomm NPUs (sourced 2026-10-12)

Pins: linux master 24a19b95; mesa main 716eb010; qualcomm/hexagon-mlir main 178e173c. Quotes are verbatim from the cited line or page. No recommendations.

### Q1. Rockchip RK3588: open kernel driver and userspace

| Item | Finding | Ref |
|---|---|---|
| Kernel driver | `accel/rocket`, GPL-2.0-only. "The accel/rocket driver supports the Neural Processing Units (NPUs) inside some Rockchip SoCs such as the RK3588." "Hardware currently supported: * RK3588". Docs file absent at tag v6.17 (HTTP 404), present at tag v6.18 (HTTP 200) | [60][61] |
| Kernel scope | "This driver just powers the hardware on and off, allocates and maps buffers to the device and submits jobs to the frontend unit. Everything else is done in userspace, as a Gallium driver (also called rocket) that is part of the Mesa3D project." | [60] |
| Userspace | Mesa Gallium driver `src/gallium/drivers/rocket` (MIT), consumed by Teflon, Mesa's TFLite external delegate. Mesa 25.3.0 relnotes (2025-11-14): "rocket: Initial commit of a driver for Rockchip's NPU", "teflon: Link to the rocket driver" | [62][63] |
| Teflon hardware table | "Rocket / `RK3588 NPU` / `PINE64 QuartzPro64`"; delegate is "a shared library that the TensorFlow Lite runtime can load at startup". The doc's kernel line ("a mainline 6.7 (6.8 for the i.MX8MP) or greater kernel") is etnaviv-era text; the rocket minimum kernel is the v6.18 fact above | [62] |
| Ops | `rkt_ml_operation_supported` accepts only `PIPE_ML_OPERATION_TYPE_CONVOLUTION` and `PIPE_ML_OPERATION_TYPE_ADD` (add only when both inputs have `data == NULL`); `default: supported = false`. Convolution requires dilation 1x1 and `tensor_quantization_supported` on input, weight, bias, output; source comment: "Dilation and per-axis quantization not yet implemented" | [64] |
| Dtypes | Teflon tested-models table lists UINT8 for MobileNet V1, MobileNet V2, SSDLite MobileDet ("Fully supported"; QuartzPro64 ~18 ms, ~21 ms, ~48 ms; single figures from the doc). The full accepted dtype set lives in `tensor_quantization_supported`, not read: unsourced | [62][64] |
| What is submitted | A register command buffer, not a compiled blob. `struct drm_rocket_task { __u32 regcmd; __u32 regcmd_count; }`: "regcmd: Input: DMA address to NPU mapping of register command buffer"; `drm_rocket_job` carries a pointer to an array of tasks plus input and output BO handle arrays; `DRM_IOCTL_ROCKET_SUBMIT` takes `drm_rocket_submit`. Mesa packs each entry as `target << 48 \| value << 16 \| reg` (`emit_raw`) and issues `drmIoctl(screen->fd, DRM_IOCTL_ROCKET_SUBMIT, &submit)` | [64][65][66] |
| Kernel consumption | `rocket_pc_writel(core, BASE_ADDRESS, task->regcmd)` then a data-amount register from `(task->regcmd_count + 1) / 2 - 1`: the kernel points the hardware program-counter unit at the buffer; it does not parse entries | [67] |
| Register documentation | Mesa carries `registers.xml` (rules-ng schema `rules-ng.xsd`) plus `gen_header.py` in the rocket directory; the kernel doc names "chapter 36 in the RK3588 TRM" as the hardware description | [60][68] |
| Non-TFLite producer | The uapi takes DMA-mapped buffers plus a register-command array, so a producer that fills BOs and emits the 64-bit entries can submit without Teflon or TFLite. Entry encoding is from `rkt_regcmd.c`; the per-operation register sequences (`fill_first_regcmd` onward, weight layout in `rkt_coefs.c`) were not read beyond line 50, so register-map completeness for anything beyond conv and add is unsourced | [65][66][68] |
| Other SoCs | Kernel doc lists RK3588 only at the pinned commit | [60] |

### Q2. Qualcomm Hexagon NPU

| Path | Finding | Ref |
|---|---|---|
| QNN / AI Engine Direct | Closed SDK lane, carried from census row 4. LiteRT's Qualcomm page: "LiteRT supports Qualcomm AI Engine Direct (QNN) through the `CompiledModel` API for both AOT and on-device compilation." The page does not state whether the libraries are closed | [69] |
| FastRPC, mainline kernel | `drivers/misc/fastrpc.c`, GPL-2.0; Kconfig `QCOM_FASTRPC`: "Provides a communication mechanism that allows for clients to make remote method invocations across processor boundary to applications DSP processor." Depends on `ARCH_QCOM`, `RPMSG`. Source distinguishes `adsp` and `cdsp` domains; compatible list includes `qcom,sm8450`, `qcom,sa8775p`, `qcom,qcs8300`. The DSP-side skel is still built with the Hexagon toolchain (census row 3) | [70][71] |
| qcom accel driver | `drivers/accel/qaic` is "Qualcomm Cloud AI accelerators ... PCIe cards", depends on `PCI` and `MHI_BUS`: not the Snapdragon Hexagon NPU. No Snapdragon Hexagon accel driver was found by checking this Kconfig only; a full-tree search was not run | [72] |
| hexagon-mlir license | `LICENSE.txt`: "Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met"; the file contains the "Neither the name" clause (grep count 1), the BSD-3-Clause text. GitHub's repository listing states BSD-3-Clause | [73] |
| hexagon-mlir input IR | Paper: "PyTorch models convert to Linalg and other MLIR core ops using Torch-MLIR [1]. Similarly, Triton kernels convert to MLIR using triton-to-linalg"; working abstraction `linalg.generic` on tensors, then tiling, `scf.for`/`scf.forall`, MLIR Async dialect for threading, `linalg.pack/unpack`. Entry points are Triton and PyTorch; no ONNX, StableHLO or TOSA entry appears in the sources read | [74][75] |
| hexagon-mlir output | Module attribute `llvm.target_triple = "hexagon"`; paper: "LLVM-IR with runtime calls"; user guide: "see the generated LLVM-IR along with Hexagon assembly". Final binaries need Hexagon SDK 6.4.0.2, Hexagon Tools and HexKL downloaded from Qualcomm's software center | [74][76] |
| hexagon-mlir targets | Tested v73, v75, v79 (FAQ) and v73, v75, v79, v81 (user guide); "You do need access to a Qualcomm device with Hexagon NPU hardware to run the compiled kernels." Execution via `ANDROID_HOST` and `ANDROID_SERIAL`, or the Hexagon simulator with `RUN_ON_SIM=1`. Retail Snapdragon device versus dev board: unsourced (docs say "a Qualcomm device with Hexagon NPU" and give a support email for test-device access) | [76][77] |
| Dtypes in paper | Census row 6 records float16/float32 only in the results tables | [74] |

### Q3. LiteRT as an interchange

| Item | Finding | Ref |
|---|---|---|
| NPU vendor list | LiteRT NPU page lists five: Google Tensor, Qualcomm AI Engine Direct, MediaTek NeuroPilot, Intel OpenVINO, Samsung Exynos AI LiteCore. Google Tensor: "doesn't yet support on-device (JIT) compilation"; Intel: "Support AOT and on-device compilation execution through the `CompiledModel` API" | [78] |
| Input and flow | "use LiteRT AOT Compiler to compile your .tflite model to the supported SoCs"; unsupported subgraphs fall back to CPU or GPU (partial delegation) | [78] |
| Qualcomm | SoCs SM8850, SM8750, SM8650, SM8550, SM8475, SM8450; Bazel 7.4.1, Android API 34 | [69] |
| MediaTek | NeuroPilot is on the vendor list; the dedicated MediaTek page was not fetched: details unsourced | [78] |
| Rockchip | Not in the LiteRT list. Reached through Mesa Teflon, an external delegate for the TFLite runtime, not through LiteRT CompiledModel | [62][78] |
| Hub question | A `.tflite` file is a documented FlatBuffer (schema [55]) and is the stated input to the LiteRT AOT compiler and to Teflon. Not established by the sources: per-vendor op and dtype coverage; whether the vendor compiler plugins are open; Teflon coverage for an LLM graph (conv and add only per [64]). End-to-end coverage for a proxima graph via `.tflite` is unmeasured | [55][62][64][78] |

Not sourced this session: the full accepted dtype set of `tensor_quantization_supported`; register sequences for any rocket operation beyond `rkt_regcmd.c` lines 1-50; whether retail Snapdragon devices run hexagon-mlir output; hexagon-mlir launch mechanism; any Snapdragon Hexagon accel driver in mainline beyond `fastrpc.c`; LiteRT MediaTek page contents; open or closed status of the LiteRT vendor compiler plugins; any benchmark of any lane.

References (continued from [59]):
[60] https://raw.githubusercontent.com/torvalds/linux/24a19b953e2a3d0ea5a8c4951a0ab8f32720c907/Documentation/accel/rocket/index.rst (lines 5-15)
[61] https://raw.githubusercontent.com/torvalds/linux/v6.18/Documentation/accel/rocket/index.rst (HTTP 200); same path at tag v6.17 (HTTP 404)
[62] https://gitlab.freedesktop.org/mesa/mesa/-/raw/716eb0108e7d695393d865197de68fcdb5c223ca/docs/teflon.rst (hardware table, tested-models table, build and kernel lines)
[63] https://docs.mesa3d.org/relnotes/25.3.0.html (dated 2025-11-14; rocket and teflon entries)
[64] https://gitlab.freedesktop.org/mesa/mesa/-/raw/716eb0108e7d695393d865197de68fcdb5c223ca/src/gallium/drivers/rocket/rkt_ml.c (`rkt_ml_operation_supported` lines 275-310; `DRM_IOCTL_ROCKET_SUBMIT` line 548)
[65] https://raw.githubusercontent.com/torvalds/linux/24a19b953e2a3d0ea5a8c4951a0ab8f32720c907/include/uapi/drm/rocket_accel.h (lines 94-185)
[66] https://gitlab.freedesktop.org/mesa/mesa/-/raw/716eb0108e7d695393d865197de68fcdb5c223ca/src/gallium/drivers/rocket/rkt_regcmd.c (lines 1-50; SPDX MIT)
[67] https://raw.githubusercontent.com/torvalds/linux/24a19b953e2a3d0ea5a8c4951a0ab8f32720c907/drivers/accel/rocket/rocket_job.c (lines 92-99, 138-140)
[68] https://gitlab.freedesktop.org/api/v4/projects/mesa%2Fmesa/repository/tree?path=src/gallium/drivers/rocket&ref=main (entries `registers.xml`, `rules-ng.xsd`, `gen_header.py`, `rkt_coefs.c`, `rkt_regcmd.c`, `rkt_task.c`)
[69] https://developers.google.com/edge/litert/next/qualcomm (supported SoCs, requirements, QNN sentence)
[70] https://raw.githubusercontent.com/torvalds/linux/24a19b953e2a3d0ea5a8c4951a0ab8f32720c907/drivers/misc/Kconfig (lines 266-276)
[71] https://raw.githubusercontent.com/torvalds/linux/24a19b953e2a3d0ea5a8c4951a0ab8f32720c907/drivers/misc/fastrpc.c (lines 2486-2488, 2523-2527)
[72] https://raw.githubusercontent.com/torvalds/linux/24a19b953e2a3d0ea5a8c4951a0ab8f32720c907/drivers/accel/qaic/Kconfig (lines 1-12)
[73] https://raw.githubusercontent.com/qualcomm/hexagon-mlir/178e173c329322f79e88cd17847d9fad4e77b3e5/LICENSE.txt
[74] https://arxiv.org/pdf/2602.19762v1 (Torch-MLIR and triton-to-linalg passage; `llvm.target_triple = "hexagon"`; "LLVM-IR with runtime calls")
[75] https://raw.githubusercontent.com/qualcomm/hexagon-mlir/178e173c329322f79e88cd17847d9fad4e77b3e5/README.md ("complements our commercial toolchains")
[76] https://raw.githubusercontent.com/qualcomm/hexagon-mlir/178e173c329322f79e88cd17847d9fad4e77b3e5/docs/user-guide.md (lines 44-45, 91, 129, 292-297)
[77] https://raw.githubusercontent.com/qualcomm/hexagon-mlir/178e173c329322f79e88cd17847d9fad4e77b3e5/docs/faq.md (hardware section; `linalg-hexagon-opt`)
[78] https://developers.google.com/edge/litert/next/npu (vendor list, .tflite AOT flow, fallback)
