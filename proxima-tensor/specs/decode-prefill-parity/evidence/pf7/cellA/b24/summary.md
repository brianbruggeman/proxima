### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 350.370 [CoV 1.3%] (349.954-358.317) |
| evaluate | 339.232 [CoV 1.5%] (338.984-348.034) |
| outside_evaluate | 10.970 [CoV 4.2%] (10.283-11.137) |
| apply_serving_config | 0.000 [CoV 61.5%] (0.000-0.001) |
| build_position_inputs | 0.483 [CoV 1.9%] (0.472-0.490) |
| named_blocks_span | 10.327 [CoV 4.5%] (9.645-10.505) |
| nb_weights_loop | 0.001 [CoV 55.3%] (0.001-0.003) |
| nb_device_kv_adopt | 9.275 [CoV 4.9%] (8.534-9.313) |
| nb_push_named_blocks | 0.005 [CoV 7.3%] (0.005-0.006) |
| nb_greedy_program | 1.085 [CoV 6.4%] (1.024-1.163) |
| append_plus_greedy | 0.002 [CoV 12.9%] (0.002-0.002) |
| outside_unattributed | 0.150 [CoV 5.9%] (0.147-0.164) |
| prepare | 34.001 [CoV 2.1%] (32.867-34.120) |
| gap_after_prepare | 0.517 [CoV 2.5%] (0.517-0.539) |
| pre_encode | 42.438 [CoV 10.6%] (40.163-49.136) |
| pipeline_compile | 31.142 [CoV 12.1%] (29.649-37.137) |
| encode_sum | 4.115 [CoV 8.7%] (3.556-4.179) |
| encode_to_commit | 0.003 [CoV 11.1%] (0.003-0.003) |
| commit_call_sum | 0.022 [CoV 11.0%] (0.019-0.023) |
| chunk1_queue | 49.348 [CoV 0.8%] (48.656-49.387) |
| chunk1_sched_to_gpu | 0.045 [CoV 18.9%] (0.036-0.053) |
| gpu_busy_sum | 211.696 [CoV 0.4%] (210.604-212.423) |
| inter_chunk_idle | 0.044 [CoV 1.0%] (0.044-0.044) |
| lead_idle_entry_to_gpu | 94.783 [CoV 4.6%] (92.182-100.738) |
| lead_idle_prepare_start_to_gpu | 128.166 [CoV 3.5%] (126.819-135.278) |
| tail | 0.157 [CoV 4.8%] (0.156-0.170) |
| post_wait_to_exit | 0.031 [CoV 11.6%] (0.029-0.036) |
| in_evaluate_unattributed | 0.217 [CoV 41.5%] (0.097-0.238) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 60.109 [CoV 1.0%] (59.284-60.493) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 213.467 [CoV 0.1%] (213.332-213.901) |
| evaluate | 202.175 [CoV 0.3%] (202.088-203.071) |
| outside_evaluate | 11.160 [CoV 1.8%] (11.129-11.494) |
| apply_serving_config | 0.000 [CoV 20.9%] (0.000-0.000) |
| build_position_inputs | 0.489 [CoV 2.9%] (0.468-0.496) |
| named_blocks_span | 10.559 [CoV 1.9%] (10.498-10.870) |
| nb_weights_loop | 0.001 [CoV 12.0%] (0.001-0.002) |
| nb_device_kv_adopt | 9.378 [CoV 4.7%] (8.945-9.827) |
| nb_push_named_blocks | 0.004 [CoV 4.9%] (0.004-0.004) |
| nb_greedy_program | 1.374 [CoV 10.6%] (1.157-1.416) |
| append_plus_greedy | 0.002 [CoV 1.3%] (0.002-0.002) |
| outside_unattributed | 0.137 [CoV 2.2%] (0.136-0.142) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.137 [CoV 2.1%] (0.135-0.141) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.903 [CoV 2.6%] (0.866-0.909) |
| encode_to_commit | 0.001 [CoV 14.9%] (0.001-0.001) |
| commit_call_sum | 0.023 [CoV 16.4%] (0.022-0.029) |
| chunk1_queue | 0.356 [CoV 9.9%] (0.345-0.413) |
| chunk1_sched_to_gpu | 0.097 [CoV 28.7%] (0.086-0.145) |
| gpu_busy_sum | 201.137 [CoV 0.2%] (201.118-201.857) |
| inter_chunk_idle | 0.042 [CoV 4.5%] (0.039-0.043) |
| lead_idle_entry_to_gpu | 0.739 [CoV 7.3%] (0.674-0.780) |
| lead_idle_prepare_start_to_gpu | 0.739 [CoV 7.3%] (0.674-0.780) |
| tail | 0.162 [CoV 17.2%] (0.161-0.215) |
| post_wait_to_exit | 0.025 [CoV 12.7%] (0.024-0.030) |
| in_evaluate_unattributed | 0.243 [CoV 42.4%] (0.102-0.254) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 94.291 [CoV 0.1%] (94.120-94.383) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.963 [CoV 3.1%] (5.649-5.970) |
| evaluate | 5.884 [CoV 3.1%] (5.583-5.907) |
| outside_evaluate | 0.068 [CoV 2.8%] (0.065-0.068) |
| apply_serving_config | 0.000 [CoV 13.5%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 4.4%] (0.002-0.002) |
| named_blocks_span | 0.022 [CoV 4.9%] (0.021-0.023) |
| nb_weights_loop | 0.001 [CoV 2.6%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 6.7%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 14.0%] (0.000-0.000) |
| outside_unattributed | 0.042 [CoV 2.5%] (0.042-0.044) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.088 [CoV 1.5%] (0.087-0.089) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.517 [CoV 1.1%] (0.511-0.523) |
| encode_to_commit | 0.000 [CoV 11.1%] (0.000-0.000) |
| commit_call_sum | 0.014 [CoV 8.1%] (0.012-0.014) |
| chunk1_queue | 0.068 [CoV 9.3%] (0.061-0.073) |
| chunk1_sched_to_gpu | 0.067 [CoV 2.1%] (0.065-0.068) |
| gpu_busy_sum | 5.110 [CoV 3.7%] (4.796-5.114) |
| inter_chunk_idle | 0.288 [CoV 2.8%] (0.280-0.296) |
| lead_idle_entry_to_gpu | 0.269 [CoV 1.3%] (0.263-0.270) |
| lead_idle_prepare_start_to_gpu | 0.269 [CoV 1.3%] (0.263-0.270) |
| tail | 0.126 [CoV 0.9%] (0.126-0.128) |
| post_wait_to_exit | 0.016 [CoV 4.2%] (0.015-0.016) |
| in_evaluate_unattributed | 0.063 [CoV 2.2%] (0.062-0.065) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 85.707 [CoV 0.4%] (85.380-86.131) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.750 [CoV 0.9%] (5.683-5.778) |
| evaluate | 5.674 [CoV 0.7%] (5.613-5.691) |
| outside_evaluate | 0.068 [CoV 7.6%] (0.066-0.076) |
| apply_serving_config | 0.000 [CoV 20.3%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 9.8%] (0.002-0.002) |
| named_blocks_span | 0.023 [CoV 11.0%] (0.023-0.027) |
| nb_weights_loop | 0.001 [CoV 6.5%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 18.4%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 13.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 21.5%] (0.000-0.000) |
| outside_unattributed | 0.042 [CoV 5.9%] (0.041-0.046) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.088 [CoV 8.7%] (0.087-0.101) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.513 [CoV 5.7%] (0.509-0.563) |
| encode_to_commit | 0.000 [CoV 9.2%] (0.000-0.001) |
| commit_call_sum | 0.013 [CoV 17.5%] (0.012-0.017) |
| chunk1_queue | 0.064 [CoV 18.2%] (0.062-0.085) |
| chunk1_sched_to_gpu | 0.067 [CoV 0.6%] (0.066-0.067) |
| gpu_busy_sum | 4.814 [CoV 0.2%] (4.808-4.827) |
| inter_chunk_idle | 0.296 [CoV 0.7%] (0.295-0.299) |
| lead_idle_entry_to_gpu | 0.270 [CoV 9.9%] (0.270-0.320) |
| lead_idle_prepare_start_to_gpu | 0.270 [CoV 9.9%] (0.270-0.320) |
| tail | 0.131 [CoV 3.7%] (0.128-0.138) |
| post_wait_to_exit | 0.017 [CoV 15.0%] (0.016-0.021) |
| in_evaluate_unattributed | 0.065 [CoV 7.2%] (0.064-0.073) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 84.957 [CoV 1.1%] (83.477-85.121) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.356 [CoV 9.9%] (0.345-0.413) | 0.097 [CoV 28.7%] (0.086-0.145) | 9.401 [CoV 3.2%] (9.388-9.929) | n/a | 0.080 [CoV 3.9%] (0.078-0.084) |
| 2 | 359.000 [CoV 0.0%] (359.000-359.000) | 2.920 [CoV 18.3%] (2.790-3.857) | 6.274 [CoV 3.3%] (6.219-6.608) | 191.740 [CoV 0.0%] (191.718-191.758) | 0.042 [CoV 4.5%] (0.039-0.043) | 0.817 [CoV 2.6%] (0.783-0.820) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.064 [CoV 18.2%] (0.062-0.085) | 0.067 [CoV 0.6%] (0.066-0.067) | 0.243 [CoV 0.6%] (0.241-0.244) | n/a | 0.042 [CoV 11.4%] (0.042-0.051) |
| 2 | 359.000 [CoV 0.0%] (359.000-359.000) | 0.130 [CoV 1.5%] (0.130-0.134) | 0.058 [CoV 2.5%] (0.056-0.059) | 4.567 [CoV 0.3%] (4.566-4.588) | 0.296 [CoV 0.7%] (0.295-0.299) | 0.469 [CoV 4.6%] (0.469-0.507) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| b24/p1/telemetry.log | 350 | 213 (213-213) | 369.8 | 18.8 | 3217.8 | 1607.6 | 1624.9 | 3336.1 | 1712.7 | aff93fd561a93eff |
| b24/p2/telemetry.log | 349 | 213 (213-214) | 371.1 | 19.3 | 3143.1 | 1608.2 | 1624.3 | 3336.8 | 1712.8 | aff93fd561a93eff |
| b24/p3/telemetry.log | 358 | 214 (213-215) | 404.3 | 20.8 | 3193.2 | 1612.3 | 1624.0 | 3321.1 | 1716.7 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 213.0 ms, CoV of the per-process medians 0.27%
