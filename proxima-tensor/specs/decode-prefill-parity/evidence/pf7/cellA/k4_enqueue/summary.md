### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 354.296 [CoV 14.0%] (353.889-447.434) |
| evaluate | 343.619 [CoV 14.2%] (342.818-435.235) |
| outside_evaluate | 11.070 [CoV 7.0%] (10.677-12.199) |
| apply_serving_config | 0.000 [CoV 27.0%] (0.000-0.000) |
| build_position_inputs | 0.472 [CoV 2.2%] (0.468-0.488) |
| named_blocks_span | 10.426 [CoV 7.3%] (10.055-11.551) |
| nb_weights_loop | 0.001 [CoV 29.0%] (0.001-0.002) |
| nb_device_kv_adopt | 9.226 [CoV 8.2%] (8.952-10.416) |
| nb_push_named_blocks | 0.007 [CoV 33.0%] (0.004-0.008) |
| nb_greedy_program | 1.104 [CoV 4.3%] (1.078-1.170) |
| append_plus_greedy | 0.002 [CoV 26.8%] (0.002-0.003) |
| outside_unattributed | 0.157 [CoV 8.0%] (0.148-0.173) |
| prepare | 33.125 [CoV 30.9%] (33.071-54.660) |
| gap_after_prepare | 0.528 [CoV 24.7%] (0.489-0.760) |
| pre_encode | 45.594 [CoV 48.7%] (44.549-97.948) |
| pipeline_compile | 34.739 [CoV 50.9%] (33.275-76.421) |
| encode_sum | 4.019 [CoV 33.0%] (3.862-6.719) |
| encode_to_commit | 0.003 [CoV 11.9%] (0.003-0.003) |
| commit_call_sum | 0.022 [CoV 16.5%] (0.020-0.027) |
| chunk1_queue | 49.940 [CoV 9.1%] (48.847-57.580) |
| chunk1_sched_to_gpu | 0.033 [CoV 188.3%] (-1.527-0.050) |
| gpu_busy_sum | 211.408 [CoV 1.2%] (207.480-212.134) |
| inter_chunk_idle | 0.128 [CoV 11.7%] (0.105-0.130) |
| lead_idle_entry_to_gpu | 97.320 [CoV 30.4%] (97.206-159.342) |
| lead_idle_prepare_start_to_gpu | 130.919 [CoV 30.5%] (130.819-214.761) |
| tail | 0.168 [CoV 160.2%] (0.166-6.356) |
| post_wait_to_exit | 0.033 [CoV 12.3%] (0.027-0.033) |
| in_evaluate_unattributed | 0.287 [CoV 153.6%] (0.241-6.477) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 59.739 [CoV 14.0%] (46.371-59.875) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 214.021 [CoV 0.8%] (213.144-216.359) |
| evaluate | 202.811 [CoV 0.2%] (202.383-203.062) |
| outside_evaluate | 11.569 [CoV 6.3%] (11.089-12.542) |
| apply_serving_config | 0.000 [CoV 12.6%] (0.000-0.000) |
| build_position_inputs | 0.485 [CoV 1.4%] (0.474-0.486) |
| named_blocks_span | 10.939 [CoV 5.9%] (10.466-11.756) |
| nb_weights_loop | 0.002 [CoV 17.2%] (0.001-0.002) |
| nb_device_kv_adopt | 9.419 [CoV 1.3%] (9.238-9.462) |
| nb_push_named_blocks | 0.004 [CoV 8.5%] (0.004-0.004) |
| nb_greedy_program | 1.934 [CoV 32.1%] (1.203-2.370) |
| append_plus_greedy | 0.002 [CoV 14.2%] (0.002-0.002) |
| outside_unattributed | 0.142 [CoV 30.8%] (0.142-0.234) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.147 [CoV 7.6%] (0.145-0.166) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 1.020 [CoV 6.5%] (0.993-1.122) |
| encode_to_commit | 0.001 [CoV 2.7%] (0.001-0.001) |
| commit_call_sum | 0.021 [CoV 4.0%] (0.020-0.022) |
| chunk1_queue | 0.387 [CoV 55.8%] (0.379-0.930) |
| chunk1_sched_to_gpu | 0.068 [CoV 11.3%] (0.057-0.072) |
| gpu_busy_sum | 201.017 [CoV 0.2%] (200.955-201.574) |
| inter_chunk_idle | 0.119 [CoV 0.7%] (0.119-0.120) |
| lead_idle_entry_to_gpu | 0.752 [CoV 22.2%] (0.727-1.065) |
| lead_idle_prepare_start_to_gpu | 0.752 [CoV 22.2%] (0.727-1.065) |
| tail | 0.172 [CoV 11.8%] (0.141-0.176) |
| post_wait_to_exit | 0.029 [CoV 43.8%] (0.029-0.059) |
| in_evaluate_unattributed | 0.264 [CoV 23.4%] (0.254-0.379) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 93.947 [CoV 0.4%] (93.715-94.391) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.465 [CoV 3.2%] (5.454-5.766) |
| evaluate | 5.387 [CoV 3.2%] (5.381-5.686) |
| outside_evaluate | 0.077 [CoV 11.4%] (0.067-0.085) |
| apply_serving_config | 0.000 [CoV 31.3%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 12.3%] (0.002-0.002) |
| named_blocks_span | 0.025 [CoV 11.7%] (0.024-0.030) |
| nb_weights_loop | 0.001 [CoV 15.4%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 26.0%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 36.9%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 46.5%] (0.000-0.001) |
| outside_unattributed | 0.048 [CoV 8.5%] (0.043-0.051) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.097 [CoV 7.1%] (0.091-0.105) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.552 [CoV 6.3%] (0.539-0.606) |
| encode_to_commit | 0.001 [CoV 15.4%] (0.000-0.001) |
| commit_call_sum | 0.021 [CoV 11.6%] (0.020-0.025) |
| chunk1_queue | 0.074 [CoV 11.1%] (0.068-0.085) |
| chunk1_sched_to_gpu | 0.068 [CoV 2.7%] (0.065-0.068) |
| gpu_busy_sum | 4.799 [CoV 2.0%] (4.786-4.957) |
| inter_chunk_idle | 0.082 [CoV 0.2%] (0.082-0.082) |
| lead_idle_entry_to_gpu | 0.302 [CoV 9.3%] (0.282-0.338) |
| lead_idle_prepare_start_to_gpu | 0.302 [CoV 9.3%] (0.282-0.338) |
| tail | 0.129 [CoV 8.5%] (0.128-0.148) |
| post_wait_to_exit | 0.019 [CoV 16.3%] (0.018-0.024) |
| in_evaluate_unattributed | 0.069 [CoV 13.9%] (0.066-0.085) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 87.490 [CoV 0.6%] (87.091-88.110) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.543 [CoV 1.4%] (5.473-5.631) |
| evaluate | 5.468 [CoV 1.4%] (5.395-5.547) |
| outside_evaluate | 0.075 [CoV 5.1%] (0.074-0.081) |
| apply_serving_config | 0.000 [CoV 20.9%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 6.9%] (0.002-0.002) |
| named_blocks_span | 0.026 [CoV 5.4%] (0.026-0.028) |
| nb_weights_loop | 0.001 [CoV 8.1%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 9.3%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 10.9%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 43.6%] (0.000-0.001) |
| outside_unattributed | 0.046 [CoV 3.1%] (0.046-0.048) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.098 [CoV 4.2%] (0.095-0.104) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.578 [CoV 8.1%] (0.561-0.652) |
| encode_to_commit | 0.001 [CoV 12.0%] (0.000-0.001) |
| commit_call_sum | 0.023 [CoV 1.1%] (0.022-0.023) |
| chunk1_queue | 0.075 [CoV 11.3%] (0.070-0.087) |
| chunk1_sched_to_gpu | 0.066 [CoV 0.6%] (0.066-0.066) |
| gpu_busy_sum | 4.832 [CoV 0.7%] (4.803-4.867) |
| inter_chunk_idle | 0.081 [CoV 1.2%] (0.081-0.083) |
| lead_idle_entry_to_gpu | 0.300 [CoV 7.4%] (0.293-0.335) |
| lead_idle_prepare_start_to_gpu | 0.300 [CoV 7.4%] (0.293-0.335) |
| tail | 0.123 [CoV 14.0%] (0.122-0.155) |
| post_wait_to_exit | 0.020 [CoV 8.7%] (0.020-0.023) |
| in_evaluate_unattributed | 0.070 [CoV 7.0%] (0.070-0.079) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 87.796 [CoV 1.4%] (85.769-87.926) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.387 [CoV 55.8%] (0.379-0.930) | 0.068 [CoV 11.3%] (0.057-0.072) | 9.387 [CoV 0.5%] (9.366-9.448) | n/a | 0.085 [CoV 2.0%] (0.085-0.088) |
| 2 | 86.000 [CoV 0.0%] (86.000-86.000) | 0.955 [CoV 35.8%] (0.913-1.663) | 8.712 [CoV 0.3%] (8.689-8.735) | 49.416 [CoV 0.1%] (49.345-49.479) | 0.041 [CoV 4.3%] (0.038-0.041) | 0.232 [CoV 2.8%] (0.224-0.237) |
| 3 | 116.000 [CoV 0.0%] (116.000-116.000) | 1.845 [CoV 25.6%] (1.635-2.625) | 57.069 [CoV 0.1%] (56.945-57.109) | 61.025 [CoV 0.1%] (60.946-61.026) | 0.041 [CoV 4.8%] (0.039-0.043) | 0.277 [CoV 5.6%] (0.263-0.294) |
| 4 | 157.000 [CoV 0.0%] (157.000-157.000) | 2.687 [CoV 12.6%] (2.399-3.080) | 116.972 [CoV 0.2%] (116.822-117.176) | 81.135 [CoV 0.0%] (81.128-81.144) | 0.040 [CoV 0.9%] (0.040-0.040) | 0.351 [CoV 15.7%] (0.348-0.454) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.075 [CoV 11.3%] (0.070-0.087) | 0.066 [CoV 0.6%] (0.066-0.066) | 0.245 [CoV 0.8%] (0.242-0.246) | n/a | 0.046 [CoV 10.4%] (0.045-0.055) |
| 2 | 86.000 [CoV 0.0%] (86.000-86.000) | 0.057 [CoV 6.7%] (0.052-0.059) | 0.214 [CoV 2.5%] (0.207-0.218) | 1.123 [CoV 0.6%] (1.116-1.130) | 0.024 [CoV 1.7%] (0.023-0.024) | 0.132 [CoV 3.8%] (0.130-0.140) |
| 3 | 116.000 [CoV 0.0%] (116.000-116.000) | 0.060 [CoV 11.3%] (0.056-0.070) | 1.176 [CoV 1.2%] (1.158-1.185) | 1.427 [CoV 0.6%] (1.422-1.440) | 0.028 [CoV 0.9%] (0.028-0.029) | 0.171 [CoV 5.4%] (0.168-0.186) |
| 4 | 157.000 [CoV 0.0%] (157.000-157.000) | 0.075 [CoV 7.3%] (0.071-0.082) | 2.394 [CoV 2.8%] (2.303-2.433) | 2.035 [CoV 0.7%] (2.017-2.044) | 0.029 [CoV 0.9%] (0.028-0.029) | 0.215 [CoV 5.6%] (0.214-0.236) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k4_enqueue/p1/telemetry.log | 354 | 213 (212-219) | 369.6 | 20.6 | 3148.3 | 1611.1 | 1624.0 | 3336.9 | 1716.7 | aff93fd561a93eff |
| k4_enqueue/p2/telemetry.log | 353 | 214 (213-215) | 377.2 | 21.5 | 3178.9 | 1610.7 | 1624.0 | 3363.5 | 1716.2 | aff93fd561a93eff |
| k4_enqueue/p3/telemetry.log | 447 | 216 (214-217) | 382.6 | 23.9 | 3220.9 | 1611.6 | 1624.0 | 3388.6 | 1715.1 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 214.0 ms, CoV of the per-process medians 0.71%
