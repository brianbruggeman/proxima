### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 345.804 [CoV 1.4%] (343.699-353.091) |
| evaluate | 335.654 [CoV 1.6%] (332.664-342.853) |
| outside_evaluate | 10.238 [CoV 4.7%] (10.150-11.035) |
| apply_serving_config | 0.000 [CoV 26.6%] (0.000-0.000) |
| build_position_inputs | 0.474 [CoV 2.9%] (0.459-0.487) |
| named_blocks_span | 9.585 [CoV 4.9%] (9.549-10.394) |
| nb_weights_loop | 0.001 [CoV 29.9%] (0.001-0.001) |
| nb_device_kv_adopt | 8.515 [CoV 5.0%] (8.501-9.270) |
| nb_push_named_blocks | 0.004 [CoV 35.7%] (0.003-0.005) |
| nb_greedy_program | 1.062 [CoV 3.9%] (1.014-1.095) |
| append_plus_greedy | 0.002 [CoV 37.0%] (0.001-0.002) |
| outside_unattributed | 0.153 [CoV 11.3%] (0.140-0.175) |
| prepare | 33.058 [CoV 1.6%] (32.603-33.664) |
| gap_after_prepare | 0.577 [CoV 10.2%] (0.490-0.597) |
| pre_encode | 37.850 [CoV 10.1%] (36.858-44.255) |
| pipeline_compile | 27.426 [CoV 12.4%] (26.299-33.006) |
| encode_sum | 3.933 [CoV 2.8%] (3.747-3.935) |
| encode_to_commit | 0.002 [CoV 8.0%] (0.002-0.003) |
| commit_call_sum | 0.017 [CoV 13.6%] (0.016-0.021) |
| chunk1_queue | 48.917 [CoV 3.2%] (47.699-50.854) |
| chunk1_sched_to_gpu | 0.046 [CoV 107.1%] (0.038-0.244) |
| gpu_busy_sum | 211.212 [CoV 0.4%] (210.391-212.126) |
| inter_chunk_idle | 0.122 [CoV 4.9%] (0.113-0.125) |
| lead_idle_entry_to_gpu | 90.339 [CoV 4.2%] (88.519-95.930) |
| lead_idle_prepare_start_to_gpu | 123.886 [CoV 3.5%] (121.718-130.171) |
| tail | 0.159 [CoV 6.6%] (0.152-0.173) |
| post_wait_to_exit | 0.037 [CoV 17.6%] (0.027-0.038) |
| in_evaluate_unattributed | 0.234 [CoV 5.3%] (0.232-0.255) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 61.079 [CoV 1.0%] (60.077-61.214) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 213.235 [CoV 0.2%] (213.167-213.945) |
| evaluate | 202.133 [CoV 0.1%] (202.008-202.376) |
| outside_evaluate | 11.291 [CoV 2.1%] (11.102-11.568) |
| apply_serving_config | 0.000 [CoV 16.6%] (0.000-0.000) |
| build_position_inputs | 0.472 [CoV 0.9%] (0.471-0.479) |
| named_blocks_span | 10.703 [CoV 2.1%] (10.502-10.949) |
| nb_weights_loop | 0.001 [CoV 3.3%] (0.001-0.001) |
| nb_device_kv_adopt | 8.729 [CoV 3.1%] (8.448-8.996) |
| nb_push_named_blocks | 0.003 [CoV 6.7%] (0.003-0.003) |
| nb_greedy_program | 2.128 [CoV 18.2%] (1.649-2.385) |
| append_plus_greedy | 0.002 [CoV 17.9%] (0.001-0.002) |
| outside_unattributed | 0.134 [CoV 1.1%] (0.131-0.134) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.129 [CoV 2.3%] (0.128-0.133) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.918 [CoV 3.6%] (0.862-0.920) |
| encode_to_commit | 0.001 [CoV 12.5%] (0.001-0.001) |
| commit_call_sum | 0.015 [CoV 5.6%] (0.015-0.016) |
| chunk1_queue | 0.364 [CoV 10.1%] (0.318-0.389) |
| chunk1_sched_to_gpu | 0.133 [CoV 25.4%] (0.091-0.154) |
| gpu_busy_sum | 200.875 [CoV 0.1%] (200.795-201.203) |
| inter_chunk_idle | 0.122 [CoV 2.4%] (0.122-0.127) |
| lead_idle_entry_to_gpu | 0.656 [CoV 4.3%] (0.650-0.703) |
| lead_idle_prepare_start_to_gpu | 0.656 [CoV 4.3%] (0.650-0.703) |
| tail | 0.152 [CoV 4.6%] (0.144-0.158) |
| post_wait_to_exit | 0.025 [CoV 4.4%] (0.023-0.025) |
| in_evaluate_unattributed | 0.272 [CoV 0.9%] (0.269-0.273) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 94.212 [CoV 0.1%] (94.059-94.289) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.457 [CoV 2.6%] (5.272-5.556) |
| evaluate | 5.395 [CoV 2.5%] (5.220-5.486) |
| outside_evaluate | 0.067 [CoV 14.9%] (0.053-0.072) |
| apply_serving_config | 0.000 [CoV 67.0%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 15.1%] (0.001-0.002) |
| named_blocks_span | 0.022 [CoV 21.9%] (0.016-0.024) |
| nb_weights_loop | 0.001 [CoV 12.1%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 17.7%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 12.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 34.5%] (0.000-0.000) |
| outside_unattributed | 0.041 [CoV 9.3%] (0.036-0.044) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.089 [CoV 18.5%] (0.065-0.093) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.538 [CoV 6.2%] (0.486-0.547) |
| encode_to_commit | 0.000 [CoV 33.3%] (0.000-0.000) |
| commit_call_sum | 0.021 [CoV 31.1%] (0.011-0.021) |
| chunk1_queue | 0.070 [CoV 29.8%] (0.042-0.078) |
| chunk1_sched_to_gpu | 0.069 [CoV 5.4%] (0.064-0.071) |
| gpu_busy_sum | 4.806 [CoV 1.0%] (4.757-4.857) |
| inter_chunk_idle | 0.080 [CoV 2.9%] (0.078-0.083) |
| lead_idle_entry_to_gpu | 0.282 [CoV 18.4%] (0.205-0.294) |
| lead_idle_prepare_start_to_gpu | 0.282 [CoV 18.4%] (0.205-0.294) |
| tail | 0.118 [CoV 10.8%] (0.115-0.140) |
| post_wait_to_exit | 0.017 [CoV 28.9%] (0.011-0.020) |
| in_evaluate_unattributed | 0.064 [CoV 14.8%] (0.051-0.069) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 88.710 [CoV 1.4%] (87.864-90.310) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.404 [CoV 1.3%] (5.302-5.440) |
| evaluate | 5.334 [CoV 1.2%] (5.246-5.376) |
| outside_evaluate | 0.055 [CoV 9.6%] (0.053-0.063) |
| apply_serving_config | 0.000 [CoV 65.8%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 10.3%] (0.001-0.001) |
| named_blocks_span | 0.016 [CoV 16.7%] (0.015-0.021) |
| nb_weights_loop | 0.001 [CoV 5.0%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 13.5%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 0.3%] (0.000-0.000) |
| outside_unattributed | 0.037 [CoV 5.7%] (0.036-0.041) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.069 [CoV 12.9%] (0.065-0.083) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.490 [CoV 4.1%] (0.488-0.525) |
| encode_to_commit | 0.000 [CoV 14.4%] (0.000-0.000) |
| commit_call_sum | 0.012 [CoV 15.2%] (0.011-0.015) |
| chunk1_queue | 0.052 [CoV 17.4%] (0.040-0.057) |
| chunk1_sched_to_gpu | 0.064 [CoV 3.1%] (0.063-0.067) |
| gpu_busy_sum | 4.801 [CoV 0.4%] (4.790-4.825) |
| inter_chunk_idle | 0.079 [CoV 1.7%] (0.077-0.079) |
| lead_idle_entry_to_gpu | 0.217 [CoV 12.6%] (0.201-0.256) |
| lead_idle_prepare_start_to_gpu | 0.217 [CoV 12.6%] (0.201-0.256) |
| tail | 0.119 [CoV 3.8%] (0.114-0.123) |
| post_wait_to_exit | 0.012 [CoV 20.2%] (0.010-0.015) |
| in_evaluate_unattributed | 0.051 [CoV 12.1%] (0.051-0.062) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 90.164 [CoV 0.8%] (88.971-90.368) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.364 [CoV 10.1%] (0.318-0.389) | 0.133 [CoV 25.4%] (0.091-0.154) | 9.396 [CoV 0.8%] (9.394-9.526) | n/a | 0.081 [CoV 5.5%] (0.075-0.084) |
| 2 | 86.000 [CoV 0.0%] (86.000-86.000) | 0.956 [CoV 14.3%] (0.834-1.110) | 8.857 [CoV 1.6%] (8.617-8.858) | 49.368 [CoV 0.2%] (49.367-49.511) | 0.042 [CoV 5.0%] (0.039-0.043) | 0.211 [CoV 5.9%] (0.209-0.232) |
| 3 | 116.000 [CoV 0.0%] (116.000-116.000) | 1.790 [CoV 17.2%] (1.505-2.126) | 57.278 [CoV 0.7%] (56.691-57.416) | 61.030 [CoV 0.2%] (60.860-61.089) | 0.043 [CoV 7.0%] (0.038-0.044) | 0.264 [CoV 4.8%] (0.249-0.274) |
| 4 | 157.000 [CoV 0.0%] (157.000-157.000) | 2.548 [CoV 16.0%] (2.218-3.045) | 117.178 [CoV 0.4%] (116.398-117.355) | 81.195 [CoV 0.0%] (81.175-81.207) | 0.043 [CoV 2.7%] (0.041-0.043) | 0.341 [CoV 2.8%] (0.339-0.357) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.052 [CoV 17.4%] (0.040-0.057) | 0.064 [CoV 3.1%] (0.063-0.067) | 0.240 [CoV 0.4%] (0.239-0.241) | n/a | 0.033 [CoV 15.1%] (0.031-0.040) |
| 2 | 86.000 [CoV 0.0%] (86.000-86.000) | 0.043 [CoV 9.9%] (0.038-0.046) | 0.209 [CoV 2.6%] (0.203-0.214) | 1.113 [CoV 0.4%] (1.109-1.119) | 0.022 [CoV 6.9%] (0.020-0.022) | 0.116 [CoV 2.7%] (0.115-0.121) |
| 3 | 116.000 [CoV 0.0%] (116.000-116.000) | 0.053 [CoV 7.8%] (0.047-0.054) | 1.182 [CoV 0.9%] (1.180-1.200) | 1.426 [CoV 0.4%] (1.420-1.431) | 0.028 [CoV 0.7%] (0.028-0.028) | 0.151 [CoV 2.8%] (0.151-0.158) |
| 4 | 157.000 [CoV 0.0%] (157.000-157.000) | 0.073 [CoV 1.4%] (0.071-0.073) | 2.416 [CoV 0.7%] (2.411-2.442) | 2.019 [CoV 0.4%] (2.018-2.032) | 0.028 [CoV 0.7%] (0.028-0.028) | 0.190 [CoV 3.5%] (0.189-0.201) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k4_enqueue_r2/p1/telemetry.log | 345 | 213 (212-214) | 358.0 | 18.7 | 3044.9 | 1609.3 | 1624.0 | 3244.9 | 1713.0 | aff93fd561a93eff |
| k4_enqueue_r2/p2/telemetry.log | 353 | 213 (213-214) | 364.3 | 20.8 | 3180.4 | 1610.5 | 1624.0 | 3329.1 | 1715.0 | aff93fd561a93eff |
| k4_enqueue_r2/p3/telemetry.log | 343 | 214 (213-214) | 365.6 | 19.5 | 3215.7 | 1610.3 | 1624.0 | 3351.4 | 1715.0 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 213.0 ms, CoV of the per-process medians 0.27%
