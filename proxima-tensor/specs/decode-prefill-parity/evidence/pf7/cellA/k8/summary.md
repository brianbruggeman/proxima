### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 336.134 [CoV 1.7%] (335.283-345.730) |
| evaluate | 326.246 [CoV 1.7%] (325.063-335.504) |
| outside_evaluate | 10.220 [CoV 1.9%] (9.888-10.225) |
| apply_serving_config | 0.000 [CoV 30.2%] (0.000-0.000) |
| build_position_inputs | 0.456 [CoV 2.7%] (0.445-0.470) |
| named_blocks_span | 9.603 [CoV 1.9%] (9.300-9.622) |
| nb_weights_loop | 0.001 [CoV 13.7%] (0.001-0.001) |
| nb_device_kv_adopt | 8.511 [CoV 1.8%] (8.320-8.624) |
| nb_push_named_blocks | 0.003 [CoV 26.4%] (0.003-0.005) |
| nb_greedy_program | 0.974 [CoV 5.7%] (0.959-1.065) |
| append_plus_greedy | 0.001 [CoV 14.5%] (0.001-0.002) |
| outside_unattributed | 0.141 [CoV 4.1%] (0.140-0.151) |
| prepare | 31.372 [CoV 2.7%] (30.615-32.340) |
| gap_after_prepare | 0.466 [CoV 14.3%] (0.408-0.542) |
| pre_encode | 34.061 [CoV 9.4%] (34.009-39.889) |
| pipeline_compile | 25.004 [CoV 9.5%] (24.907-29.313) |
| encode_sum | 4.059 [CoV 5.5%] (3.761-4.192) |
| encode_to_commit | 0.003 [CoV 6.2%] (0.002-0.003) |
| commit_call_sum | 0.023 [CoV 5.3%] (0.021-0.024) |
| chunk1_queue | 46.080 [CoV 0.2%] (45.967-46.136) |
| chunk1_sched_to_gpu | 0.030 [CoV 17.1%] (0.026-0.036) |
| gpu_busy_sum | 211.905 [CoV 0.8%] (209.917-213.207) |
| inter_chunk_idle | 0.279 [CoV 5.2%] (0.259-0.287) |
| lead_idle_entry_to_gpu | 82.614 [CoV 4.2%] (82.512-88.660) |
| lead_idle_prepare_start_to_gpu | 114.451 [CoV 3.8%] (113.535-121.541) |
| tail | 0.167 [CoV 31.9%] (0.148-0.263) |
| post_wait_to_exit | 0.031 [CoV 10.4%] (0.029-0.035) |
| in_evaluate_unattributed | 0.259 [CoV 11.6%] (0.221-0.279) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 62.609 [CoV 1.1%] (61.669-63.042) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 213.753 [CoV 0.2%] (213.559-214.329) |
| evaluate | 202.674 [CoV 0.1%] (202.589-202.946) |
| outside_evaluate | 11.079 [CoV 1.9%] (10.986-11.383) |
| apply_serving_config | 0.000 [CoV 17.3%] (0.000-0.000) |
| build_position_inputs | 0.525 [CoV 14.3%] (0.447-0.597) |
| named_blocks_span | 10.504 [CoV 1.4%] (10.339-10.637) |
| nb_weights_loop | 0.001 [CoV 12.0%] (0.001-0.001) |
| nb_device_kv_adopt | 9.046 [CoV 6.5%] (8.220-9.339) |
| nb_push_named_blocks | 0.003 [CoV 13.5%] (0.003-0.004) |
| nb_greedy_program | 2.013 [CoV 21.0%] (1.414-2.149) |
| append_plus_greedy | 0.002 [CoV 6.9%] (0.002-0.002) |
| outside_unattributed | 0.143 [CoV 4.6%] (0.135-0.148) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.131 [CoV 3.2%] (0.130-0.138) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.958 [CoV 2.7%] (0.916-0.963) |
| encode_to_commit | 0.001 [CoV 14.0%] (0.001-0.001) |
| commit_call_sum | 0.021 [CoV 7.8%] (0.019-0.022) |
| chunk1_queue | 0.357 [CoV 2.0%] (0.347-0.361) |
| chunk1_sched_to_gpu | 0.106 [CoV 52.2%] (0.049-0.158) |
| gpu_busy_sum | 201.208 [CoV 0.1%] (201.186-201.485) |
| inter_chunk_idle | 0.289 [CoV 3.7%] (0.274-0.294) |
| lead_idle_entry_to_gpu | 0.721 [CoV 5.2%] (0.668-0.740) |
| lead_idle_prepare_start_to_gpu | 0.721 [CoV 5.2%] (0.668-0.740) |
| tail | 0.170 [CoV 4.0%] (0.164-0.177) |
| post_wait_to_exit | 0.030 [CoV 7.9%] (0.027-0.032) |
| in_evaluate_unattributed | 0.283 [CoV 7.3%] (0.249-0.285) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 94.297 [CoV 0.0%] (94.267-94.353) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.834 [CoV 2.9%] (5.766-6.085) |
| evaluate | 5.772 [CoV 2.9%] (5.719-6.033) |
| outside_evaluate | 0.052 [CoV 7.8%] (0.052-0.060) |
| apply_serving_config | 0.000 [CoV 50.6%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 13.2%] (0.001-0.001) |
| named_blocks_span | 0.015 [CoV 7.7%] (0.015-0.017) |
| nb_weights_loop | 0.001 [CoV 9.1%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 9.4%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 21.8%] (0.000-0.000) |
| outside_unattributed | 0.035 [CoV 6.6%] (0.035-0.039) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.061 [CoV 9.3%] (0.060-0.071) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.509 [CoV 3.4%] (0.501-0.534) |
| encode_to_commit | 0.000 [CoV 9.2%] (0.000-0.000) |
| commit_call_sum | 0.018 [CoV 7.2%] (0.017-0.019) |
| chunk1_queue | 0.087 [CoV 1.5%] (0.085-0.088) |
| chunk1_sched_to_gpu | 0.051 [CoV 6.1%] (0.048-0.054) |
| gpu_busy_sum | 5.170 [CoV 3.4%] (5.106-5.445) |
| inter_chunk_idle | 0.168 [CoV 3.2%] (0.165-0.176) |
| lead_idle_entry_to_gpu | 0.233 [CoV 1.2%] (0.231-0.237) |
| lead_idle_prepare_start_to_gpu | 0.233 [CoV 1.2%] (0.231-0.237) |
| tail | 0.133 [CoV 3.6%] (0.130-0.139) |
| post_wait_to_exit | 0.012 [CoV 11.3%] (0.011-0.014) |
| in_evaluate_unattributed | 0.050 [CoV 7.4%] (0.049-0.056) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 88.489 [CoV 0.6%] (88.314-89.307) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.811 [CoV 0.2%] (5.789-5.816) |
| evaluate | 5.742 [CoV 0.3%] (5.723-5.752) |
| outside_evaluate | 0.069 [CoV 4.7%] (0.065-0.071) |
| apply_serving_config | 0.000 [CoV 34.6%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 9.8%] (0.001-0.002) |
| named_blocks_span | 0.021 [CoV 4.9%] (0.021-0.023) |
| nb_weights_loop | 0.001 [CoV 6.5%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 3.3%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 17.1%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 24.0%] (0.000-0.000) |
| outside_unattributed | 0.046 [CoV 6.3%] (0.042-0.048) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.084 [CoV 6.6%] (0.079-0.090) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.626 [CoV 4.1%] (0.598-0.649) |
| encode_to_commit | 0.000 [CoV 0.0%] (0.000-0.000) |
| commit_call_sum | 0.022 [CoV 5.3%] (0.021-0.024) |
| chunk1_queue | 0.080 [CoV 3.3%] (0.079-0.084) |
| chunk1_sched_to_gpu | 0.058 [CoV 5.5%] (0.057-0.064) |
| gpu_busy_sum | 5.078 [CoV 0.2%] (5.078-5.093) |
| inter_chunk_idle | 0.167 [CoV 0.8%] (0.166-0.168) |
| lead_idle_entry_to_gpu | 0.272 [CoV 3.4%] (0.266-0.284) |
| lead_idle_prepare_start_to_gpu | 0.272 [CoV 3.4%] (0.266-0.284) |
| tail | 0.134 [CoV 1.5%] (0.132-0.136) |
| post_wait_to_exit | 0.017 [CoV 6.5%] (0.016-0.018) |
| in_evaluate_unattributed | 0.066 [CoV 4.4%] (0.062-0.068) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 87.450 [CoV 0.3%] (87.046-87.518) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k8/p1/telemetry.log | 336 | 213 (213-214) | 369.8 | 21.9 | 3236.1 | 1605.5 | 1624.0 | 3434.9 | 1711.9 | aff93fd561a93eff |
| k8/p2/telemetry.log | 335 | 214 (213-214) | 369.3 | 22.4 | 3061.8 | 1608.6 | 1624.0 | 3261.0 | 1712.9 | aff93fd561a93eff |
| k8/p3/telemetry.log | 345 | 214 (212-215) | 369.7 | 22.2 | 3166.4 | 1609.9 | 1624.0 | 3349.3 | 1718.5 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 213.0 ms, CoV of the per-process medians 0.27%
