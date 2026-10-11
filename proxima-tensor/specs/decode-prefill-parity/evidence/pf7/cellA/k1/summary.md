### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 342.486 [CoV 0.3%] (341.452-343.720) |
| evaluate | 332.428 [CoV 0.3%] (331.682-333.750) |
| outside_evaluate | 9.969 [CoV 1.5%] (9.769-10.058) |
| apply_serving_config | 0.000 [CoV 46.7%] (0.000-0.001) |
| build_position_inputs | 0.458 [CoV 3.1%] (0.445-0.473) |
| named_blocks_span | 9.364 [CoV 1.3%] (9.186-9.418) |
| nb_weights_loop | 0.001 [CoV 13.3%] (0.001-0.001) |
| nb_device_kv_adopt | 8.270 [CoV 0.4%] (8.229-8.286) |
| nb_push_named_blocks | 0.004 [CoV 30.4%] (0.002-0.004) |
| nb_greedy_program | 1.107 [CoV 11.9%] (0.896-1.113) |
| append_plus_greedy | 0.002 [CoV 23.4%] (0.001-0.002) |
| outside_unattributed | 0.146 [CoV 9.0%] (0.137-0.164) |
| prepare | 31.752 [CoV 3.3%] (30.837-32.941) |
| gap_after_prepare | 0.437 [CoV 8.3%] (0.436-0.503) |
| pre_encode | 35.190 [CoV 7.0%] (33.578-38.465) |
| pipeline_compile | 25.158 [CoV 8.6%] (24.513-28.663) |
| encode_sum | 3.373 [CoV 1.9%] (3.351-3.469) |
| encode_to_commit | 0.003 [CoV 11.7%] (0.003-0.003) |
| commit_call_sum | 0.015 [CoV 13.3%] (0.012-0.016) |
| chunk1_queue | 49.275 [CoV 1.7%] (47.958-49.471) |
| chunk1_sched_to_gpu | 0.041 [CoV 23.6%] (0.035-0.056) |
| gpu_busy_sum | 212.148 [CoV 0.4%] (210.972-212.643) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 86.576 [CoV 3.0%] (86.571-91.166) |
| lead_idle_prepare_start_to_gpu | 120.015 [CoV 1.6%] (118.765-122.440) |
| tail | 0.187 [CoV 20.9%] (0.176-0.256) |
| post_wait_to_exit | 0.022 [CoV 4.4%] (0.021-0.023) |
| in_evaluate_unattributed | 0.064 [CoV 4.4%] (0.062-0.067) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 61.944 [CoV 0.7%] (61.379-62.276) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 216.201 [CoV 0.4%] (215.204-216.896) |
| evaluate | 205.131 [CoV 0.1%] (204.610-205.153) |
| outside_evaluate | 10.947 [CoV 0.6%] (10.945-11.055) |
| apply_serving_config | 0.000 [CoV 45.9%] (0.000-0.000) |
| build_position_inputs | 0.496 [CoV 6.4%] (0.444-0.498) |
| named_blocks_span | 10.378 [CoV 0.3%] (10.358-10.413) |
| nb_weights_loop | 0.001 [CoV 9.0%] (0.001-0.001) |
| nb_device_kv_adopt | 8.557 [CoV 5.9%] (8.401-9.358) |
| nb_push_named_blocks | 0.003 [CoV 3.9%] (0.003-0.003) |
| nb_greedy_program | 1.807 [CoV 16.1%] (1.407-1.939) |
| append_plus_greedy | 0.001 [CoV 10.5%] (0.001-0.001) |
| outside_unattributed | 0.128 [CoV 4.4%] (0.127-0.137) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.124 [CoV 2.4%] (0.119-0.124) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.801 [CoV 4.5%] (0.800-0.865) |
| encode_to_commit | 0.001 [CoV 86.0%] (0.001-0.004) |
| commit_call_sum | 0.013 [CoV 10.0%] (0.013-0.016) |
| chunk1_queue | 2.607 [CoV 4.9%] (2.497-2.751) |
| chunk1_sched_to_gpu | 0.076 [CoV 4.5%] (0.073-0.080) |
| gpu_busy_sum | 201.078 [CoV 0.1%] (200.706-201.138) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 3.631 [CoV 3.9%] (3.469-3.754) |
| lead_idle_prepare_start_to_gpu | 3.631 [CoV 3.9%] (3.469-3.754) |
| tail | 0.175 [CoV 6.2%] (0.175-0.194) |
| post_wait_to_exit | 0.023 [CoV 9.4%] (0.021-0.026) |
| in_evaluate_unattributed | 0.074 [CoV 7.0%] (0.073-0.083) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 93.263 [CoV 0.3%] (93.024-93.620) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 6.151 [CoV 1.1%] (6.101-6.236) |
| evaluate | 6.105 [CoV 1.1%] (6.053-6.183) |
| outside_evaluate | 0.050 [CoV 8.8%] (0.049-0.058) |
| apply_serving_config | 0.000 [CoV 35.1%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 14.6%] (0.001-0.001) |
| named_blocks_span | 0.014 [CoV 17.4%] (0.014-0.019) |
| nb_weights_loop | 0.001 [CoV 14.5%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 24.1%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 25.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 15.5%] (0.000-0.000) |
| outside_unattributed | 0.035 [CoV 5.7%] (0.034-0.038) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.063 [CoV 9.6%] (0.060-0.072) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.433 [CoV 3.8%] (0.426-0.458) |
| encode_to_commit | 0.000 [CoV 19.9%] (0.000-0.000) |
| commit_call_sum | 0.004 [CoV 13.5%] (0.004-0.005) |
| chunk1_queue | 0.162 [CoV 2.4%] (0.159-0.167) |
| chunk1_sched_to_gpu | 0.079 [CoV 1.2%] (0.079-0.081) |
| gpu_busy_sum | 5.155 [CoV 0.9%] (5.106-5.197) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 0.744 [CoV 3.2%] (0.735-0.781) |
| lead_idle_prepare_start_to_gpu | 0.744 [CoV 3.2%] (0.735-0.781) |
| tail | 0.149 [CoV 2.5%] (0.149-0.155) |
| post_wait_to_exit | 0.009 [CoV 15.2%] (0.008-0.011) |
| in_evaluate_unattributed | 0.047 [CoV 10.0%] (0.046-0.055) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 83.610 [CoV 0.7%] (82.593-83.646) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 6.191 [CoV 0.2%] (6.176-6.202) |
| evaluate | 6.132 [CoV 0.0%] (6.127-6.133) |
| outside_evaluate | 0.056 [CoV 9.3%] (0.051-0.062) |
| apply_serving_config | 0.000 [CoV 25.0%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 8.8%] (0.001-0.001) |
| named_blocks_span | 0.017 [CoV 9.4%] (0.015-0.018) |
| nb_weights_loop | 0.001 [CoV 7.9%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 9.7%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 21.8%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 13.5%] (0.000-0.000) |
| outside_unattributed | 0.037 [CoV 9.0%] (0.035-0.042) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.070 [CoV 11.5%] (0.064-0.080) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.455 [CoV 5.6%] (0.438-0.489) |
| encode_to_commit | 0.000 [CoV 18.3%] (0.000-0.000) |
| commit_call_sum | 0.005 [CoV 11.8%] (0.005-0.006) |
| chunk1_queue | 0.156 [CoV 1.8%] (0.152-0.158) |
| chunk1_sched_to_gpu | 0.080 [CoV 1.5%] (0.078-0.080) |
| gpu_busy_sum | 5.146 [CoV 0.4%] (5.125-5.170) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 0.774 [CoV 4.9%] (0.742-0.818) |
| lead_idle_prepare_start_to_gpu | 0.774 [CoV 4.9%] (0.742-0.818) |
| tail | 0.152 [CoV 2.3%] (0.148-0.155) |
| post_wait_to_exit | 0.010 [CoV 16.3%] (0.009-0.012) |
| in_evaluate_unattributed | 0.053 [CoV 10.3%] (0.048-0.059) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 83.039 [CoV 1.0%] (81.804-83.359) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k1/p1/telemetry.log | 343 | 216 (214-217) | 382.2 | 16.3 | 3039.3 | 1603.3 | 1625.0 | 3236.0 | 1706.9 | aff93fd561a93eff |
| k1/p2/telemetry.log | 341 | 216 (215-217) | 387.9 | 17.8 | 3170.5 | 1605.8 | 1625.0 | 3325.3 | 1709.4 | aff93fd561a93eff |
| k1/p3/telemetry.log | 342 | 215 (215-216) | 382.9 | 16.0 | 3137.8 | 1608.1 | 1625.0 | 3242.5 | 1711.9 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 216.0 ms, CoV of the per-process medians 0.27%
