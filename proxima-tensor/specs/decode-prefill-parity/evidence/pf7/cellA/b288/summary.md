### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 347.965 [CoV 1.6%] (338.460-348.480) |
| evaluate | 336.794 [CoV 1.5%] (328.514-337.714) |
| outside_evaluate | 10.766 [CoV 5.9%] (9.947-11.171) |
| apply_serving_config | 0.000 [CoV 20.9%] (0.000-0.000) |
| build_position_inputs | 0.463 [CoV 3.1%] (0.445-0.473) |
| named_blocks_span | 10.145 [CoV 6.1%] (9.354-10.561) |
| nb_weights_loop | 0.001 [CoV 17.8%] (0.001-0.001) |
| nb_device_kv_adopt | 9.111 [CoV 7.2%] (8.276-9.544) |
| nb_push_named_blocks | 0.004 [CoV 36.9%] (0.004-0.006) |
| nb_greedy_program | 1.006 [CoV 3.2%] (0.996-1.057) |
| append_plus_greedy | 0.001 [CoV 27.7%] (0.001-0.002) |
| outside_unattributed | 0.146 [CoV 0.7%] (0.145-0.147) |
| prepare | 32.780 [CoV 2.7%] (31.625-33.372) |
| gap_after_prepare | 0.537 [CoV 9.1%] (0.479-0.575) |
| pre_encode | 37.786 [CoV 8.6%] (33.737-40.073) |
| pipeline_compile | 27.959 [CoV 10.1%] (24.413-29.843) |
| encode_sum | 3.788 [CoV 6.8%] (3.487-3.992) |
| encode_to_commit | 0.003 [CoV 13.4%] (0.002-0.003) |
| commit_call_sum | 0.015 [CoV 19.7%] (0.015-0.021) |
| chunk1_queue | 50.193 [CoV 5.8%] (48.057-53.857) |
| chunk1_sched_to_gpu | 0.286 [CoV 71.2%] (0.038-0.311) |
| gpu_busy_sum | 208.972 [CoV 0.8%] (207.670-210.942) |
| inter_chunk_idle | 0.040 [CoV 1.4%] (0.039-0.041) |
| lead_idle_entry_to_gpu | 94.072 [CoV 6.2%] (85.010-95.580) |
| lead_idle_prepare_start_to_gpu | 127.390 [CoV 5.3%] (117.114-129.527) |
| tail | 0.167 [CoV 15.3%] (0.157-0.208) |
| post_wait_to_exit | 0.024 [CoV 41.8%] (0.023-0.046) |
| in_evaluate_unattributed | 0.224 [CoV 7.9%] (0.202-0.237) |
| chunk1_ops | 288.000 [CoV 0.0%] (288.000-288.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 60.055 [CoV 2.4%] (59.593-62.324) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 216.990 [CoV 0.3%] (215.866-217.286) |
| evaluate | 205.204 [CoV 0.1%] (205.079-205.294) |
| outside_evaluate | 11.235 [CoV 3.7%] (11.140-11.910) |
| apply_serving_config | 0.000 [CoV 7.0%] (0.000-0.000) |
| build_position_inputs | 0.486 [CoV 4.2%] (0.472-0.512) |
| named_blocks_span | 10.604 [CoV 3.7%] (10.550-11.258) |
| nb_weights_loop | 0.001 [CoV 17.4%] (0.001-0.002) |
| nb_device_kv_adopt | 9.084 [CoV 2.0%] (8.832-9.173) |
| nb_push_named_blocks | 0.003 [CoV 11.1%] (0.003-0.004) |
| nb_greedy_program | 2.095 [CoV 20.9%] (1.491-2.265) |
| append_plus_greedy | 0.002 [CoV 16.8%] (0.001-0.002) |
| outside_unattributed | 0.137 [CoV 1.3%] (0.135-0.138) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.140 [CoV 3.8%] (0.135-0.146) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.890 [CoV 6.8%] (0.863-0.981) |
| encode_to_commit | 0.001 [CoV 8.8%] (0.001-0.001) |
| commit_call_sum | 0.014 [CoV 11.0%] (0.012-0.014) |
| chunk1_queue | 2.413 [CoV 5.0%] (2.367-2.600) |
| chunk1_sched_to_gpu | 0.065 [CoV 13.5%] (0.052-0.068) |
| gpu_busy_sum | 201.337 [CoV 0.1%] (201.269-201.471) |
| inter_chunk_idle | 0.043 [CoV 5.1%] (0.040-0.044) |
| lead_idle_entry_to_gpu | 3.271 [CoV 4.4%] (3.240-3.511) |
| lead_idle_prepare_start_to_gpu | 3.271 [CoV 4.4%] (3.240-3.511) |
| tail | 0.176 [CoV 6.6%] (0.175-0.196) |
| post_wait_to_exit | 0.025 [CoV 7.3%] (0.023-0.026) |
| in_evaluate_unattributed | 0.283 [CoV 5.0%] (0.270-0.299) |
| chunk1_ops | 288.000 [CoV 0.0%] (288.000-288.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 93.127 [CoV 0.3%] (92.996-93.550) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.946 [CoV 2.3%] (5.839-6.109) |
| evaluate | 5.889 [CoV 2.3%] (5.768-6.042) |
| outside_evaluate | 0.072 [CoV 19.2%] (0.050-0.072) |
| apply_serving_config | 0.000 [CoV 53.0%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 19.5%] (0.001-0.002) |
| named_blocks_span | 0.025 [CoV 28.9%] (0.015-0.026) |
| nb_weights_loop | 0.001 [CoV 20.7%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 32.9%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 32.7%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 45.5%] (0.000-0.000) |
| outside_unattributed | 0.044 [CoV 13.9%] (0.035-0.045) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.088 [CoV 20.0%] (0.061-0.090) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.528 [CoV 9.2%] (0.449-0.530) |
| encode_to_commit | 0.000 [CoV 39.1%] (0.000-0.001) |
| commit_call_sum | 0.011 [CoV 34.8%] (0.006-0.012) |
| chunk1_queue | 0.141 [CoV 17.5%] (0.106-0.150) |
| chunk1_sched_to_gpu | 0.070 [CoV 1.2%] (0.070-0.071) |
| gpu_busy_sum | 5.077 [CoV 3.2%] (4.814-5.110) |
| inter_chunk_idle | 0.030 [CoV 1.1%] (0.030-0.030) |
| lead_idle_entry_to_gpu | 0.709 [CoV 8.8%] (0.610-0.717) |
| lead_idle_prepare_start_to_gpu | 0.709 [CoV 8.8%] (0.610-0.717) |
| tail | 0.139 [CoV 8.0%] (0.123-0.143) |
| post_wait_to_exit | 0.017 [CoV 28.5%] (0.009-0.017) |
| in_evaluate_unattributed | 0.067 [CoV 18.6%] (0.048-0.069) |
| chunk1_ops | 288.000 [CoV 0.0%] (288.000-288.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 83.157 [CoV 2.1%] (82.447-85.733) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.926 [CoV 2.2%] (5.824-6.078) |
| evaluate | 5.851 [CoV 2.2%] (5.753-6.007) |
| outside_evaluate | 0.069 [CoV 4.6%] (0.067-0.073) |
| apply_serving_config | 0.000 [CoV 25.0%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 2.7%] (0.002-0.002) |
| named_blocks_span | 0.022 [CoV 5.6%] (0.021-0.024) |
| nb_weights_loop | 0.001 [CoV 2.5%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 8.2%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 12.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 8.5%] (0.000-0.000) |
| outside_unattributed | 0.043 [CoV 9.1%] (0.042-0.050) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.086 [CoV 2.0%] (0.086-0.089) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.513 [CoV 6.7%] (0.511-0.574) |
| encode_to_commit | 0.000 [CoV 5.7%] (0.000-0.000) |
| commit_call_sum | 0.009 [CoV 12.4%] (0.009-0.011) |
| chunk1_queue | 0.120 [CoV 6.5%] (0.109-0.124) |
| chunk1_sched_to_gpu | 0.074 [CoV 0.4%] (0.074-0.074) |
| gpu_busy_sum | 4.861 [CoV 2.9%] (4.803-5.071) |
| inter_chunk_idle | 0.029 [CoV 1.2%] (0.029-0.030) |
| lead_idle_entry_to_gpu | 0.681 [CoV 4.6%] (0.677-0.734) |
| lead_idle_prepare_start_to_gpu | 0.681 [CoV 4.6%] (0.677-0.734) |
| tail | 0.136 [CoV 8.5%] (0.122-0.144) |
| post_wait_to_exit | 0.015 [CoV 8.3%] (0.014-0.016) |
| in_evaluate_unattributed | 0.066 [CoV 4.0%] (0.064-0.069) |
| chunk1_ops | 288.000 [CoV 0.0%] (288.000-288.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 82.775 [CoV 0.4%] (82.379-83.021) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| b288/p1/telemetry.log | 348 | 217 (215-218) | 378.4 | 18.8 | 3144.5 | 1606.1 | 1624.0 | 3343.4 | 1710.2 | aff93fd561a93eff |
| b288/p2/telemetry.log | 347 | 217 (216-217) | 380.6 | 18.2 | 3205.4 | 1605.5 | 1624.0 | 3310.0 | 1709.8 | aff93fd561a93eff |
| b288/p3/telemetry.log | 338 | 217 (214-219) | 383.1 | 19.6 | 3137.0 | 1606.9 | 1624.0 | 3326.7 | 1713.5 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 217.0 ms, CoV of the per-process medians 0.00%
