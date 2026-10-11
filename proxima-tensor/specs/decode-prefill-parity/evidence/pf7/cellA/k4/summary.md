### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 337.725 [CoV 0.7%] (336.763-341.541) |
| evaluate | 327.718 [CoV 0.7%] (326.309-330.814) |
| outside_evaluate | 10.453 [CoV 3.5%] (10.007-10.727) |
| apply_serving_config | 0.000 [CoV 31.3%] (0.000-0.000) |
| build_position_inputs | 0.454 [CoV 4.0%] (0.435-0.471) |
| named_blocks_span | 9.853 [CoV 3.5%] (9.422-10.104) |
| nb_weights_loop | 0.001 [CoV 5.3%] (0.001-0.001) |
| nb_device_kv_adopt | 8.718 [CoV 2.9%] (8.396-8.889) |
| nb_push_named_blocks | 0.003 [CoV 26.4%] (0.003-0.005) |
| nb_greedy_program | 1.112 [CoV 8.3%] (1.006-1.189) |
| append_plus_greedy | 0.001 [CoV 42.4%] (0.001-0.002) |
| outside_unattributed | 0.147 [CoV 1.9%] (0.145-0.150) |
| prepare | 32.271 [CoV 2.7%] (30.808-32.310) |
| gap_after_prepare | 0.464 [CoV 1.8%] (0.457-0.474) |
| pre_encode | 34.476 [CoV 2.5%] (33.387-35.084) |
| pipeline_compile | 25.274 [CoV 3.6%] (24.181-25.963) |
| encode_sum | 3.543 [CoV 0.8%] (3.508-3.565) |
| encode_to_commit | 0.003 [CoV 7.4%] (0.003-0.003) |
| commit_call_sum | 0.018 [CoV 11.2%] (0.016-0.020) |
| chunk1_queue | 46.957 [CoV 5.6%] (45.389-50.577) |
| chunk1_sched_to_gpu | 0.029 [CoV 16.4%] (0.023-0.033) |
| gpu_busy_sum | 210.999 [CoV 0.2%] (210.603-211.284) |
| inter_chunk_idle | 0.118 [CoV 5.9%] (0.109-0.122) |
| lead_idle_entry_to_gpu | 84.569 [CoV 2.4%] (82.387-86.466) |
| lead_idle_prepare_start_to_gpu | 115.834 [CoV 1.9%] (115.172-119.201) |
| tail | 0.170 [CoV 8.2%] (0.164-0.192) |
| post_wait_to_exit | 0.030 [CoV 6.1%] (0.028-0.031) |
| in_evaluate_unattributed | 0.276 [CoV 9.9%] (0.232-0.277) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 62.538 [CoV 0.7%] (61.778-62.561) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 213.955 [CoV 0.1%] (213.804-214.227) |
| evaluate | 202.318 [CoV 0.0%] (202.157-202.323) |
| outside_evaluate | 11.890 [CoV 5.2%] (10.876-11.926) |
| apply_serving_config | 0.000 [CoV 16.6%] (0.000-0.000) |
| build_position_inputs | 0.531 [CoV 1.8%] (0.520-0.539) |
| named_blocks_span | 11.271 [CoV 5.9%] (10.195-11.363) |
| nb_weights_loop | 0.001 [CoV 3.7%] (0.001-0.001) |
| nb_device_kv_adopt | 9.009 [CoV 1.1%] (8.848-9.024) |
| nb_push_named_blocks | 0.003 [CoV 21.3%] (0.003-0.004) |
| nb_greedy_program | 2.041 [CoV 36.5%] (1.122-2.458) |
| append_plus_greedy | 0.002 [CoV 7.6%] (0.002-0.002) |
| outside_unattributed | 0.147 [CoV 7.0%] (0.139-0.160) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.132 [CoV 2.6%] (0.131-0.137) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.978 [CoV 6.6%] (0.886-1.007) |
| encode_to_commit | 0.001 [CoV 20.7%] (0.001-0.001) |
| commit_call_sum | 0.017 [CoV 15.9%] (0.015-0.020) |
| chunk1_queue | 0.359 [CoV 8.2%] (0.326-0.384) |
| chunk1_sched_to_gpu | 0.095 [CoV 32.3%] (0.067-0.129) |
| gpu_busy_sum | 200.986 [CoV 0.0%] (200.947-201.074) |
| inter_chunk_idle | 0.126 [CoV 0.8%] (0.125-0.126) |
| lead_idle_entry_to_gpu | 0.691 [CoV 9.0%] (0.602-0.716) |
| lead_idle_prepare_start_to_gpu | 0.691 [CoV 9.0%] (0.602-0.716) |
| tail | 0.175 [CoV 10.0%] (0.168-0.202) |
| post_wait_to_exit | 0.027 [CoV 13.7%] (0.025-0.033) |
| in_evaluate_unattributed | 0.306 [CoV 3.8%] (0.287-0.307) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 94.290 [CoV 0.2%] (93.990-94.344) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.632 [CoV 2.0%] (5.460-5.670) |
| evaluate | 5.561 [CoV 2.0%] (5.399-5.610) |
| outside_evaluate | 0.058 [CoV 6.2%] (0.054-0.062) |
| apply_serving_config | 0.000 [CoV 87.3%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 4.9%] (0.001-0.001) |
| named_blocks_span | 0.017 [CoV 10.4%] (0.016-0.019) |
| nb_weights_loop | 0.001 [CoV 7.5%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 6.0%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 20.2%] (0.000-0.000) |
| outside_unattributed | 0.040 [CoV 5.1%] (0.037-0.040) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.069 [CoV 11.0%] (0.062-0.077) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.516 [CoV 4.4%] (0.483-0.526) |
| encode_to_commit | 0.000 [CoV 28.4%] (0.000-0.000) |
| commit_call_sum | 0.013 [CoV 7.0%] (0.012-0.014) |
| chunk1_queue | 0.063 [CoV 18.7%] (0.052-0.076) |
| chunk1_sched_to_gpu | 0.067 [CoV 3.6%] (0.065-0.070) |
| gpu_busy_sum | 5.031 [CoV 3.4%] (4.776-5.098) |
| inter_chunk_idle | 0.079 [CoV 4.7%] (0.078-0.085) |
| lead_idle_entry_to_gpu | 0.228 [CoV 6.7%] (0.226-0.254) |
| lead_idle_prepare_start_to_gpu | 0.228 [CoV 6.7%] (0.226-0.254) |
| tail | 0.137 [CoV 4.0%] (0.128-0.138) |
| post_wait_to_exit | 0.012 [CoV 10.3%] (0.011-0.013) |
| in_evaluate_unattributed | 0.056 [CoV 7.2%] (0.050-0.058) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 89.913 [CoV 0.6%] (88.977-90.003) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.665 [CoV 0.4%] (5.662-5.702) |
| evaluate | 5.606 [CoV 0.4%] (5.599-5.639) |
| outside_evaluate | 0.065 [CoV 7.7%] (0.062-0.072) |
| apply_serving_config | 0.000 [CoV 50.6%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 10.6%] (0.001-0.002) |
| named_blocks_span | 0.019 [CoV 8.5%] (0.019-0.022) |
| nb_weights_loop | 0.001 [CoV 6.8%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 7.5%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 25.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 0.0%] (0.000-0.000) |
| outside_unattributed | 0.044 [CoV 9.3%] (0.041-0.050) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.077 [CoV 9.0%] (0.076-0.089) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.582 [CoV 10.2%] (0.528-0.647) |
| encode_to_commit | 0.000 [CoV 7.6%] (0.000-0.000) |
| commit_call_sum | 0.013 [CoV 12.6%] (0.013-0.017) |
| chunk1_queue | 0.062 [CoV 6.2%] (0.060-0.067) |
| chunk1_sched_to_gpu | 0.065 [CoV 1.7%] (0.065-0.067) |
| gpu_busy_sum | 5.078 [CoV 0.8%] (5.034-5.111) |
| inter_chunk_idle | 0.082 [CoV 1.0%] (0.081-0.083) |
| lead_idle_entry_to_gpu | 0.247 [CoV 7.2%] (0.242-0.276) |
| lead_idle_prepare_start_to_gpu | 0.247 [CoV 7.2%] (0.242-0.276) |
| tail | 0.135 [CoV 2.3%] (0.133-0.139) |
| post_wait_to_exit | 0.013 [CoV 8.2%] (0.013-0.015) |
| in_evaluate_unattributed | 0.061 [CoV 7.6%] (0.059-0.068) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 89.330 [CoV 0.8%] (88.295-89.580) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k4/p1/telemetry.log | 337 | 213 (212-215) | 369.9 | 21.8 | 3035.3 | 1606.0 | 1624.0 | 3137.9 | 1709.5 | aff93fd561a93eff |
| k4/p2/telemetry.log | 341 | 213 (212-215) | 368.7 | 20.1 | 3129.2 | 1607.6 | 1624.0 | 3233.3 | 1712.3 | aff93fd561a93eff |
| k4/p3/telemetry.log | 336 | 214 (213-214) | 366.9 | 20.7 | 3116.1 | 1605.2 | 1624.0 | 3313.7 | 1708.7 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 213.0 ms, CoV of the per-process medians 0.27%
