### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 357.765 [CoV 2.6%] (357.558-374.206) |
| evaluate | 347.062 [CoV 2.7%] (347.012-363.810) |
| outside_evaluate | 10.546 [CoV 1.5%] (10.396-10.703) |
| apply_serving_config | 0.000 [CoV 21.7%] (0.000-0.000) |
| build_position_inputs | 0.484 [CoV 1.6%] (0.475-0.490) |
| named_blocks_span | 9.902 [CoV 1.5%] (9.775-10.070) |
| nb_weights_loop | 0.002 [CoV 27.3%] (0.001-0.002) |
| nb_device_kv_adopt | 8.841 [CoV 1.2%] (8.725-8.933) |
| nb_push_named_blocks | 0.004 [CoV 31.9%] (0.003-0.006) |
| nb_greedy_program | 1.035 [CoV 3.9%] (1.022-1.100) |
| append_plus_greedy | 0.002 [CoV 12.2%] (0.002-0.002) |
| outside_unattributed | 0.147 [CoV 2.7%] (0.143-0.151) |
| prepare | 34.395 [CoV 3.9%] (32.256-34.698) |
| gap_after_prepare | 0.515 [CoV 4.1%] (0.491-0.534) |
| pre_encode | 48.728 [CoV 21.8%] (47.187-68.626) |
| pipeline_compile | 36.732 [CoV 28.9%] (35.402-57.727) |
| encode_sum | 3.858 [CoV 2.6%] (3.700-3.883) |
| encode_to_commit | 0.003 [CoV 8.9%] (0.003-0.003) |
| commit_call_sum | 0.019 [CoV 24.4%] (0.014-0.023) |
| chunk1_queue | 49.912 [CoV 2.3%] (48.711-50.989) |
| chunk1_sched_to_gpu | 0.050 [CoV 50.9%] (0.016-0.051) |
| gpu_busy_sum | 210.189 [CoV 0.1%] (209.814-210.192) |
| inter_chunk_idle | 0.044 [CoV 6.5%] (0.040-0.045) |
| lead_idle_entry_to_gpu | 101.795 [CoV 10.2%] (101.079-120.376) |
| lead_idle_prepare_start_to_gpu | 136.724 [CoV 6.8%] (136.268-153.146) |
| tail | 0.159 [CoV 5.5%] (0.154-0.172) |
| post_wait_to_exit | 0.034 [CoV 53.0%] (0.032-0.077) |
| in_evaluate_unattributed | 0.244 [CoV 15.5%] (0.238-0.312) |
| chunk1_ops | 96.000 [CoV 0.0%] (96.000-96.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 58.680 [CoV 2.5%] (56.170-58.751) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 214.363 [CoV 0.7%] (213.967-216.910) |
| evaluate | 203.316 [CoV 0.3%] (203.078-204.241) |
| outside_evaluate | 11.033 [CoV 8.6%] (10.897-12.670) |
| apply_serving_config | 0.000 [CoV 17.7%] (0.000-0.000) |
| build_position_inputs | 0.474 [CoV 6.6%] (0.471-0.528) |
| named_blocks_span | 10.424 [CoV 7.5%] (10.292-11.769) |
| nb_weights_loop | 0.001 [CoV 29.4%] (0.001-0.002) |
| nb_device_kv_adopt | 9.047 [CoV 4.9%] (9.028-9.822) |
| nb_push_named_blocks | 0.004 [CoV 18.7%] (0.004-0.005) |
| nb_greedy_program | 1.276 [CoV 42.0%] (1.254-2.481) |
| append_plus_greedy | 0.002 [CoV 11.8%] (0.002-0.002) |
| outside_unattributed | 0.141 [CoV 1.0%] (0.138-0.141) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.148 [CoV 15.7%] (0.145-0.190) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.993 [CoV 13.6%] (0.915-1.187) |
| encode_to_commit | 0.001 [CoV 5.7%] (0.001-0.001) |
| commit_call_sum | 0.021 [CoV 11.8%] (0.018-0.023) |
| chunk1_queue | 1.267 [CoV 14.1%] (1.021-1.350) |
| chunk1_sched_to_gpu | 0.077 [CoV 19.8%] (0.056-0.083) |
| gpu_busy_sum | 201.057 [CoV 0.3%] (201.035-201.947) |
| inter_chunk_idle | 0.042 [CoV 11.6%] (0.035-0.044) |
| lead_idle_entry_to_gpu | 1.740 [CoV 8.6%] (1.521-1.797) |
| lead_idle_prepare_start_to_gpu | 1.740 [CoV 8.6%] (1.521-1.797) |
| tail | 0.201 [CoV 57.8%] (0.173-0.466) |
| post_wait_to_exit | 0.028 [CoV 7.1%] (0.025-0.029) |
| in_evaluate_unattributed | 0.290 [CoV 1.7%] (0.287-0.296) |
| chunk1_ops | 96.000 [CoV 0.0%] (96.000-96.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 94.083 [CoV 0.6%] (93.102-94.091) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.581 [CoV 3.4%] (5.487-5.860) |
| evaluate | 5.509 [CoV 3.5%] (5.419-5.788) |
| outside_evaluate | 0.070 [CoV 2.3%] (0.069-0.072) |
| apply_serving_config | 0.000 [CoV 16.6%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 3.8%] (0.002-0.002) |
| named_blocks_span | 0.024 [CoV 4.8%] (0.023-0.025) |
| nb_weights_loop | 0.001 [CoV 6.5%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 14.7%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 13.3%] (0.000-0.000) |
| outside_unattributed | 0.044 [CoV 3.2%] (0.041-0.044) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.092 [CoV 3.3%] (0.088-0.094) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.513 [CoV 1.3%] (0.507-0.520) |
| encode_to_commit | 0.000 [CoV 9.9%] (0.000-0.001) |
| commit_call_sum | 0.015 [CoV 5.3%] (0.014-0.015) |
| chunk1_queue | 0.097 [CoV 10.4%] (0.088-0.109) |
| chunk1_sched_to_gpu | 0.069 [CoV 2.1%] (0.068-0.071) |
| gpu_busy_sum | 4.837 [CoV 3.7%] (4.775-5.113) |
| inter_chunk_idle | 0.028 [CoV 1.7%] (0.027-0.028) |
| lead_idle_entry_to_gpu | 0.410 [CoV 2.6%] (0.402-0.424) |
| lead_idle_prepare_start_to_gpu | 0.410 [CoV 2.6%] (0.402-0.424) |
| tail | 0.129 [CoV 2.9%] (0.124-0.132) |
| post_wait_to_exit | 0.017 [CoV 7.7%] (0.016-0.018) |
| in_evaluate_unattributed | 0.066 [CoV 1.8%] (0.065-0.067) |
| chunk1_ops | 96.000 [CoV 0.0%] (96.000-96.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 87.161 [CoV 0.3%] (87.155-87.581) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.714 [CoV 3.4%] (5.668-6.032) |
| evaluate | 5.647 [CoV 3.2%] (5.586-5.929) |
| outside_evaluate | 0.074 [CoV 6.0%] (0.073-0.081) |
| apply_serving_config | 0.000 [CoV 14.2%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 8.5%] (0.002-0.002) |
| named_blocks_span | 0.027 [CoV 5.0%] (0.026-0.028) |
| nb_weights_loop | 0.001 [CoV 3.9%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 14.3%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 20.2%] (0.000-0.000) |
| append_plus_greedy | 0.001 [CoV 19.0%] (0.000-0.001) |
| outside_unattributed | 0.045 [CoV 5.3%] (0.044-0.049) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.102 [CoV 7.6%] (0.095-0.110) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.542 [CoV 4.6%] (0.530-0.579) |
| encode_to_commit | 0.001 [CoV 10.5%] (0.001-0.001) |
| commit_call_sum | 0.018 [CoV 4.9%] (0.017-0.018) |
| chunk1_queue | 0.103 [CoV 24.4%] (0.091-0.143) |
| chunk1_sched_to_gpu | 0.069 [CoV 3.8%] (0.065-0.069) |
| gpu_busy_sum | 4.882 [CoV 1.1%] (4.839-4.945) |
| inter_chunk_idle | 0.028 [CoV 1.2%] (0.028-0.028) |
| lead_idle_entry_to_gpu | 0.433 [CoV 13.5%] (0.416-0.531) |
| lead_idle_prepare_start_to_gpu | 0.433 [CoV 13.5%] (0.416-0.531) |
| tail | 0.141 [CoV 9.4%] (0.135-0.162) |
| post_wait_to_exit | 0.021 [CoV 12.0%] (0.019-0.024) |
| in_evaluate_unattributed | 0.072 [CoV 7.3%] (0.069-0.080) |
| chunk1_ops | 96.000 [CoV 0.0%] (96.000-96.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 86.320 [CoV 2.3%] (83.123-86.701) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 96.000 [CoV 0.0%] (96.000-96.000) | 1.267 [CoV 14.1%] (1.021-1.350) | 0.077 [CoV 19.8%] (0.056-0.083) | 50.600 [CoV 1.3%] (50.421-51.657) | n/a | 0.257 [CoV 2.5%] (0.248-0.260) |
| 2 | 287.000 [CoV 0.0%] (287.000-287.000) | 3.095 [CoV 14.0%] (2.697-3.567) | 48.503 [CoV 1.1%] (48.159-49.247) | 150.565 [CoV 0.0%] (150.561-150.676) | 0.042 [CoV 11.6%] (0.035-0.044) | 0.743 [CoV 11.7%] (0.664-0.839) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 96.000 [CoV 0.0%] (96.000-96.000) | 0.103 [CoV 24.4%] (0.091-0.143) | 0.069 [CoV 3.8%] (0.065-0.069) | 1.199 [CoV 0.9%] (1.183-1.203) | n/a | 0.148 [CoV 5.2%] (0.144-0.159) |
| 2 | 287.000 [CoV 0.0%] (287.000-287.000) | 0.124 [CoV 10.5%] (0.119-0.145) | 0.871 [CoV 0.3%] (0.871-0.876) | 3.691 [CoV 1.2%] (3.664-3.748) | 0.028 [CoV 1.2%] (0.028-0.028) | 0.391 [CoV 3.0%] (0.383-0.407) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| b96_r2/p1/telemetry.log | 357 | 216 (215-218) | 389.3 | 21.6 | 3218.2 | 1617.9 | 1624.0 | 3346.1 | 1722.6 | aff93fd561a93eff |
| b96_r2/p2/telemetry.log | 374 | 215 (214-216) | 376.0 | 20.7 | 3240.3 | 1611.8 | 1624.0 | 3416.9 | 1716.2 | aff93fd561a93eff |
| b96_r2/p3/telemetry.log | 357 | 214 (213-215) | 373.7 | 20.2 | 3211.3 | 1613.0 | 1624.0 | 3314.0 | 1720.1 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 215.0 ms, CoV of the per-process medians 0.47%
