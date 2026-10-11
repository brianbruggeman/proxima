### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 345.776 [CoV 0.6%] (345.190-349.053) |
| evaluate | 335.589 [CoV 0.5%] (334.998-338.257) |
| outside_evaluate | 10.192 [CoV 3.4%] (10.187-10.796) |
| apply_serving_config | 0.000 [CoV 27.1%] (0.000-0.000) |
| build_position_inputs | 0.469 [CoV 3.0%] (0.460-0.488) |
| named_blocks_span | 9.572 [CoV 3.7%] (9.540-10.180) |
| nb_weights_loop | 0.001 [CoV 13.9%] (0.001-0.001) |
| nb_device_kv_adopt | 8.475 [CoV 4.0%] (8.410-9.032) |
| nb_push_named_blocks | 0.004 [CoV 8.0%] (0.004-0.004) |
| nb_greedy_program | 1.121 [CoV 4.7%] (1.042-1.139) |
| append_plus_greedy | 0.002 [CoV 15.2%] (0.002-0.002) |
| outside_unattributed | 0.153 [CoV 2.8%] (0.148-0.157) |
| prepare | 32.425 [CoV 0.5%] (32.271-32.592) |
| gap_after_prepare | 0.458 [CoV 5.3%] (0.452-0.498) |
| pre_encode | 38.134 [CoV 4.2%] (37.066-40.214) |
| pipeline_compile | 28.232 [CoV 3.4%] (27.379-29.325) |
| encode_sum | 3.924 [CoV 2.8%] (3.913-4.109) |
| encode_to_commit | 0.088 [CoV 30.1%] (0.078-0.134) |
| commit_call_sum | 0.044 [CoV 3.4%] (0.044-0.047) |
| chunk1_queue | 48.384 [CoV 4.0%] (45.857-49.648) |
| chunk1_sched_to_gpu | 0.050 [CoV 4.9%] (0.048-0.053) |
| gpu_busy_sum | 209.957 [CoV 1.2%] (206.822-211.584) |
| inter_chunk_idle | 4.323 [CoV 1.9%] (4.220-4.387) |
| lead_idle_entry_to_gpu | 90.461 [CoV 3.3%] (85.849-91.341) |
| lead_idle_prepare_start_to_gpu | 123.511 [CoV 2.4%] (118.772-124.063) |
| tail | 0.138 [CoV 9.7%] (0.124-0.151) |
| post_wait_to_exit | 0.028 [CoV 1.9%] (0.027-0.028) |
| in_evaluate_unattributed | 0.237 [CoV 3.0%] (0.229-0.244) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 60.151 [CoV 1.3%] (59.814-61.295) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 217.260 [CoV 0.6%] (216.288-218.764) |
| evaluate | 206.209 [CoV 0.4%] (205.940-207.423) |
| outside_evaluate | 11.050 [CoV 4.2%] (10.349-11.206) |
| apply_serving_config | 0.000 [CoV 44.6%] (0.000-0.000) |
| build_position_inputs | 0.464 [CoV 0.7%] (0.460-0.466) |
| named_blocks_span | 10.431 [CoV 4.4%] (9.744-10.604) |
| nb_weights_loop | 0.001 [CoV 33.3%] (0.001-0.002) |
| nb_device_kv_adopt | 8.924 [CoV 3.2%] (8.514-9.055) |
| nb_push_named_blocks | 0.003 [CoV 10.5%] (0.003-0.003) |
| nb_greedy_program | 1.799 [CoV 31.4%] (1.072-2.073) |
| append_plus_greedy | 0.001 [CoV 17.6%] (0.001-0.002) |
| outside_unattributed | 0.137 [CoV 1.3%] (0.135-0.139) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.123 [CoV 2.4%] (0.123-0.128) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 1.035 [CoV 3.1%] (1.017-1.079) |
| encode_to_commit | 0.045 [CoV 10.2%] (0.041-0.051) |
| commit_call_sum | 0.039 [CoV 13.9%] (0.036-0.047) |
| chunk1_queue | 0.297 [CoV 11.1%] (0.283-0.348) |
| chunk1_sched_to_gpu | 0.160 [CoV 18.9%] (0.134-0.195) |
| gpu_busy_sum | 201.164 [CoV 0.1%] (200.778-201.331) |
| inter_chunk_idle | 4.002 [CoV 8.0%] (3.996-4.580) |
| lead_idle_entry_to_gpu | 0.685 [CoV 10.9%] (0.675-0.817) |
| lead_idle_prepare_start_to_gpu | 0.685 [CoV 10.9%] (0.675-0.817) |
| tail | 0.139 [CoV 4.1%] (0.135-0.146) |
| post_wait_to_exit | 0.025 [CoV 4.8%] (0.024-0.026) |
| in_evaluate_unattributed | 0.258 [CoV 5.9%] (0.243-0.273) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 92.615 [CoV 0.2%] (92.591-92.949) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 6.082 [CoV 3.7%] (5.875-6.331) |
| evaluate | 6.007 [CoV 3.6%] (5.824-6.260) |
| outside_evaluate | 0.066 [CoV 10.5%] (0.056-0.068) |
| apply_serving_config | 0.000 [CoV 63.3%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 15.4%] (0.001-0.002) |
| named_blocks_span | 0.021 [CoV 19.1%] (0.015-0.022) |
| nb_weights_loop | 0.001 [CoV 7.2%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 23.1%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 15.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 34.6%] (0.000-0.000) |
| outside_unattributed | 0.043 [CoV 7.7%] (0.038-0.044) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.073 [CoV 14.1%] (0.058-0.077) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.553 [CoV 6.5%] (0.494-0.556) |
| encode_to_commit | 0.028 [CoV 15.8%] (0.021-0.028) |
| commit_call_sum | 0.021 [CoV 18.8%] (0.015-0.021) |
| chunk1_queue | 0.044 [CoV 17.7%] (0.035-0.050) |
| chunk1_sched_to_gpu | 0.065 [CoV 3.4%] (0.063-0.067) |
| gpu_busy_sum | 4.829 [CoV 3.6%] (4.776-5.106) |
| inter_chunk_idle | 0.671 [CoV 5.8%] (0.656-0.731) |
| lead_idle_entry_to_gpu | 0.252 [CoV 12.0%] (0.212-0.269) |
| lead_idle_prepare_start_to_gpu | 0.252 [CoV 12.0%] (0.212-0.269) |
| tail | 0.107 [CoV 3.9%] (0.102-0.110) |
| post_wait_to_exit | 0.015 [CoV 24.1%] (0.011-0.017) |
| in_evaluate_unattributed | 0.063 [CoV 12.1%] (0.051-0.064) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 81.440 [CoV 1.5%] (79.451-81.560) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.958 [CoV 1.3%] (5.908-6.060) |
| evaluate | 5.902 [CoV 1.4%] (5.849-6.006) |
| outside_evaluate | 0.056 [CoV 2.7%] (0.054-0.057) |
| apply_serving_config | 0.000 [CoV 35.1%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 1.9%] (0.001-0.001) |
| named_blocks_span | 0.016 [CoV 4.6%] (0.016-0.017) |
| nb_weights_loop | 0.001 [CoV 3.0%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 4.7%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 15.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 20.2%] (0.000-0.000) |
| outside_unattributed | 0.038 [CoV 1.9%] (0.037-0.038) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.061 [CoV 2.3%] (0.059-0.062) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.496 [CoV 0.5%] (0.492-0.497) |
| encode_to_commit | 0.021 [CoV 1.7%] (0.021-0.022) |
| commit_call_sum | 0.015 [CoV 0.2%] (0.015-0.015) |
| chunk1_queue | 0.036 [CoV 1.8%] (0.035-0.036) |
| chunk1_sched_to_gpu | 0.063 [CoV 2.1%] (0.061-0.064) |
| gpu_busy_sum | 4.844 [CoV 1.0%] (4.787-4.880) |
| inter_chunk_idle | 0.660 [CoV 0.7%] (0.658-0.667) |
| lead_idle_entry_to_gpu | 0.215 [CoV 0.3%] (0.214-0.215) |
| lead_idle_prepare_start_to_gpu | 0.215 [CoV 0.3%] (0.214-0.215) |
| tail | 0.104 [CoV 0.7%] (0.103-0.105) |
| post_wait_to_exit | 0.011 [CoV 5.2%] (0.010-0.011) |
| in_evaluate_unattributed | 0.052 [CoV 1.9%] (0.051-0.053) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 81.519 [CoV 0.3%] (81.172-81.649) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.297 [CoV 11.1%] (0.283-0.348) | 0.160 [CoV 18.9%] (0.134-0.195) | 9.507 [CoV 0.5%] (9.423-9.511) | n/a | 0.081 [CoV 2.2%] (0.078-0.082) |
| 2 | 86.000 [CoV 0.0%] (86.000-86.000) | 0.822 [CoV 23.0%] (0.803-1.184) | 0.077 [CoV 1.7%] (0.077-0.079) | 49.466 [CoV 0.3%] (49.325-49.666) | 1.049 [CoV 19.8%] (1.016-1.431) | 0.241 [CoV 1.2%] (0.236-0.241) |
| 3 | 116.000 [CoV 0.0%] (116.000-116.000) | 1.215 [CoV 4.6%] (1.152-1.263) | 0.076 [CoV 3.4%] (0.074-0.079) | 60.912 [CoV 0.1%] (60.857-60.916) | 1.438 [CoV 4.2%] (1.373-1.492) | 0.297 [CoV 6.9%] (0.291-0.331) |
| 4 | 157.000 [CoV 0.0%] (157.000-157.000) | 1.315 [CoV 5.4%] (1.301-1.433) | 0.074 [CoV 3.9%] (0.074-0.079) | 81.096 [CoV 0.1%] (81.078-81.290) | 1.545 [CoV 5.0%] (1.514-1.663) | 0.444 [CoV 10.5%] (0.391-0.483) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.036 [CoV 1.8%] (0.035-0.036) | 0.063 [CoV 2.1%] (0.061-0.064) | 0.242 [CoV 0.5%] (0.240-0.243) | n/a | 0.031 [CoV 1.1%] (0.030-0.031) |
| 2 | 86.000 [CoV 0.0%] (86.000-86.000) | 0.049 [CoV 2.6%] (0.047-0.049) | 0.056 [CoV 1.4%] (0.055-0.057) | 1.121 [CoV 1.4%] (1.109-1.140) | 0.205 [CoV 1.4%] (0.203-0.208) | 0.117 [CoV 0.9%] (0.116-0.118) |
| 3 | 116.000 [CoV 0.0%] (116.000-116.000) | 0.054 [CoV 1.9%] (0.053-0.055) | 0.059 [CoV 1.5%] (0.059-0.061) | 1.442 [CoV 0.9%] (1.420-1.444) | 0.220 [CoV 1.0%] (0.218-0.222) | 0.155 [CoV 0.5%] (0.154-0.155) |
| 4 | 157.000 [CoV 0.0%] (157.000-157.000) | 0.060 [CoV 3.8%] (0.059-0.063) | 0.060 [CoV 0.5%] (0.060-0.060) | 2.030 [CoV 1.9%] (2.017-2.090) | 0.226 [CoV 0.7%] (0.224-0.227) | 0.194 [CoV 0.9%] (0.192-0.196) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k4_wait_ping_r2/p1/telemetry.log | 345 | 218 (216-224) | 386.7 | 19.7 | 3038.8 | 1607.3 | 1624.0 | 3239.6 | 1712.5 | aff93fd561a93eff |
| k4_wait_ping_r2/p2/telemetry.log | 349 | 218 (216-218) | 376.4 | 18.3 | 3236.0 | 1609.4 | 1624.0 | 3419.7 | 1715.3 | aff93fd561a93eff |
| k4_wait_ping_r2/p3/telemetry.log | 345 | 216 (216-217) | 374.1 | 18.3 | 3129.9 | 1604.7 | 1624.0 | 3237.2 | 1708.7 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 217.0 ms, CoV of the per-process medians 0.53%
