### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 339.434 [CoV 1.1%] (339.152-345.778) |
| evaluate | 329.259 [CoV 1.0%] (329.009-334.689) |
| outside_evaluate | 10.425 [CoV 5.7%] (9.893-11.089) |
| apply_serving_config | 0.000 [CoV 13.3%] (0.000-0.000) |
| build_position_inputs | 0.458 [CoV 1.5%] (0.447-0.460) |
| named_blocks_span | 9.811 [CoV 6.1%] (9.275-10.482) |
| nb_weights_loop | 0.001 [CoV 16.2%] (0.001-0.001) |
| nb_device_kv_adopt | 8.663 [CoV 7.4%] (8.162-9.441) |
| nb_push_named_blocks | 0.003 [CoV 49.6%] (0.002-0.006) |
| nb_greedy_program | 1.092 [CoV 4.6%] (1.021-1.116) |
| append_plus_greedy | 0.002 [CoV 32.5%] (0.001-0.002) |
| outside_unattributed | 0.158 [CoV 6.6%] (0.145-0.165) |
| prepare | 31.834 [CoV 0.9%] (31.416-31.974) |
| gap_after_prepare | 0.497 [CoV 1.7%] (0.497-0.512) |
| pre_encode | 34.760 [CoV 1.8%] (34.118-35.372) |
| pipeline_compile | 25.116 [CoV 1.5%] (24.788-25.551) |
| encode_sum | 4.394 [CoV 2.1%] (4.269-4.451) |
| encode_to_commit | 0.002 [CoV 3.6%] (0.002-0.003) |
| commit_call_sum | 0.027 [CoV 7.3%] (0.027-0.031) |
| chunk1_queue | 46.608 [CoV 5.4%] (46.136-50.813) |
| chunk1_sched_to_gpu | 0.028 [CoV 141.6%] (0.024-0.373) |
| gpu_busy_sum | 212.979 [CoV 0.6%] (210.960-213.278) |
| inter_chunk_idle | 0.564 [CoV 1.0%] (0.556-0.566) |
| lead_idle_entry_to_gpu | 84.508 [CoV 3.3%] (82.862-88.395) |
| lead_idle_prepare_start_to_gpu | 116.979 [CoV 2.2%] (115.208-120.308) |
| tail | 0.178 [CoV 14.5%] (0.175-0.225) |
| post_wait_to_exit | 0.039 [CoV 29.1%] (0.029-0.052) |
| in_evaluate_unattributed | 0.290 [CoV 3.0%] (0.277-0.293) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 16.000 [CoV 0.0%] (16.000-16.000) |
| busy_over_wall_pct | 62.151 [CoV 0.9%] (61.681-62.798) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 213.671 [CoV 0.1%] (213.374-213.944) |
| evaluate | 202.882 [CoV 0.1%] (202.509-203.052) |
| outside_evaluate | 10.865 [CoV 0.5%] (10.863-10.950) |
| apply_serving_config | 0.000 [CoV 44.6%] (0.000-0.000) |
| build_position_inputs | 0.442 [CoV 3.3%] (0.436-0.464) |
| named_blocks_span | 10.297 [CoV 0.3%] (10.288-10.351) |
| nb_weights_loop | 0.001 [CoV 7.9%] (0.001-0.001) |
| nb_device_kv_adopt | 8.365 [CoV 2.8%] (8.310-8.751) |
| nb_push_named_blocks | 0.003 [CoV 8.3%] (0.003-0.003) |
| nb_greedy_program | 1.908 [CoV 21.9%] (1.315-2.031) |
| append_plus_greedy | 0.002 [CoV 4.1%] (0.002-0.002) |
| outside_unattributed | 0.136 [CoV 1.0%] (0.134-0.136) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.130 [CoV 1.7%] (0.127-0.131) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 1.029 [CoV 3.1%] (1.004-1.068) |
| encode_to_commit | 0.001 [CoV 42.9%] (0.000-0.001) |
| commit_call_sum | 0.028 [CoV 5.0%] (0.028-0.030) |
| chunk1_queue | 0.367 [CoV 8.3%] (0.362-0.419) |
| chunk1_sched_to_gpu | 0.107 [CoV 19.0%] (0.080-0.118) |
| gpu_busy_sum | 201.250 [CoV 0.1%] (200.879-201.459) |
| inter_chunk_idle | 0.564 [CoV 3.2%] (0.535-0.568) |
| lead_idle_entry_to_gpu | 0.711 [CoV 2.1%] (0.686-0.713) |
| lead_idle_prepare_start_to_gpu | 0.711 [CoV 2.1%] (0.686-0.713) |
| tail | 0.169 [CoV 6.7%] (0.163-0.185) |
| post_wait_to_exit | 0.032 [CoV 16.2%] (0.027-0.037) |
| in_evaluate_unattributed | 0.263 [CoV 3.1%] (0.259-0.275) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 16.000 [CoV 0.0%] (16.000-16.000) |
| busy_over_wall_pct | 94.211 [CoV 0.1%] (94.144-94.385) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.865 [CoV 0.3%] (5.849-5.881) |
| evaluate | 5.810 [CoV 0.4%] (5.782-5.825) |
| outside_evaluate | 0.062 [CoV 11.1%] (0.056-0.069) |
| apply_serving_config | 0.000 [CoV 34.6%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 16.4%] (0.001-0.002) |
| named_blocks_span | 0.019 [CoV 16.9%] (0.017-0.023) |
| nb_weights_loop | 0.001 [CoV 14.1%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 31.6%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 29.5%] (0.000-0.001) |
| outside_unattributed | 0.040 [CoV 6.7%] (0.038-0.043) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.068 [CoV 13.8%] (0.061-0.080) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.603 [CoV 8.1%] (0.552-0.650) |
| encode_to_commit | 0.000 [CoV 15.0%] (0.000-0.000) |
| commit_call_sum | 0.019 [CoV 13.7%] (0.018-0.023) |
| chunk1_queue | 0.089 [CoV 3.0%] (0.086-0.091) |
| chunk1_sched_to_gpu | 0.061 [CoV 9.3%] (0.059-0.070) |
| gpu_busy_sum | 5.031 [CoV 0.6%] (5.001-5.061) |
| inter_chunk_idle | 0.309 [CoV 0.7%] (0.306-0.310) |
| lead_idle_entry_to_gpu | 0.258 [CoV 9.4%] (0.241-0.290) |
| lead_idle_prepare_start_to_gpu | 0.258 [CoV 9.4%] (0.241-0.290) |
| tail | 0.142 [CoV 1.2%] (0.141-0.144) |
| post_wait_to_exit | 0.018 [CoV 23.8%] (0.015-0.024) |
| in_evaluate_unattributed | 0.057 [CoV 14.4%] (0.052-0.069) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 16.000 [CoV 0.0%] (16.000-16.000) |
| busy_over_wall_pct | 85.799 [CoV 0.9%] (84.777-86.288) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.886 [CoV 0.6%] (5.831-5.902) |
| evaluate | 5.823 [CoV 0.7%] (5.760-5.838) |
| outside_evaluate | 0.068 [CoV 9.1%] (0.062-0.074) |
| apply_serving_config | 0.000 [CoV 54.0%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 10.6%] (0.002-0.002) |
| named_blocks_span | 0.022 [CoV 13.6%] (0.019-0.025) |
| nb_weights_loop | 0.001 [CoV 4.2%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 19.0%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 20.2%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 19.3%] (0.000-0.000) |
| outside_unattributed | 0.043 [CoV 6.2%] (0.041-0.046) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.079 [CoV 12.5%] (0.067-0.086) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.647 [CoV 7.2%] (0.589-0.680) |
| encode_to_commit | 0.000 [CoV 13.8%] (0.000-0.000) |
| commit_call_sum | 0.023 [CoV 10.2%] (0.021-0.026) |
| chunk1_queue | 0.091 [CoV 3.2%] (0.090-0.095) |
| chunk1_sched_to_gpu | 0.064 [CoV 10.6%] (0.059-0.072) |
| gpu_busy_sum | 5.029 [CoV 1.4%] (4.916-5.040) |
| inter_chunk_idle | 0.306 [CoV 1.1%] (0.301-0.306) |
| lead_idle_entry_to_gpu | 0.282 [CoV 10.2%] (0.254-0.311) |
| lead_idle_prepare_start_to_gpu | 0.282 [CoV 10.2%] (0.254-0.311) |
| tail | 0.142 [CoV 1.6%] (0.140-0.144) |
| post_wait_to_exit | 0.022 [CoV 16.1%] (0.017-0.023) |
| in_evaluate_unattributed | 0.067 [CoV 9.2%] (0.059-0.071) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 16.000 [CoV 0.0%] (16.000-16.000) |
| busy_over_wall_pct | 84.651 [CoV 0.9%] (84.191-85.639) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k16/p1/telemetry.log | 345 | 214 (213-214) | 368.4 | 25.0 | 3066.9 | 1615.5 | 1624.0 | 3267.8 | 1719.2 | aff93fd561a93eff |
| k16/p2/telemetry.log | 339 | 213 (213-214) | 368.7 | 24.7 | 3237.6 | 1615.6 | 1624.0 | 3340.2 | 1719.9 | aff93fd561a93eff |
| k16/p3/telemetry.log | 339 | 213 (213-215) | 368.0 | 22.9 | 3136.3 | 1609.6 | 1624.0 | 3241.1 | 1714.3 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 213.0 ms, CoV of the per-process medians 0.27%
