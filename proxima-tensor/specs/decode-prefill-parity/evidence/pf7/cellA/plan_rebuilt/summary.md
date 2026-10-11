### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 354.997 [CoV 1.6%] (346.917-357.363) |
| evaluate | 344.009 [CoV 1.5%] (336.827-346.389) |
| outside_evaluate | 10.975 [CoV 4.8%] (10.090-10.988) |
| apply_serving_config | 0.000 [CoV 17.7%] (0.000-0.000) |
| build_position_inputs | 0.499 [CoV 8.7%] (0.455-0.542) |
| named_blocks_span | 10.264 [CoV 4.6%] (9.480-10.280) |
| nb_weights_loop | 0.001 [CoV 120.2%] (0.001-0.008) |
| nb_device_kv_adopt | 9.101 [CoV 5.2%] (8.340-9.184) |
| nb_push_named_blocks | 0.004 [CoV 38.0%] (0.003-0.006) |
| nb_greedy_program | 1.119 [CoV 3.2%] (1.063-1.129) |
| append_plus_greedy | 0.002 [CoV 11.5%] (0.001-0.002) |
| outside_unattributed | 0.152 [CoV 23.6%] (0.151-0.223) |
| prepare | 33.056 [CoV 1.0%] (32.875-33.525) |
| gap_after_prepare | 0.532 [CoV 3.4%] (0.504-0.537) |
| pre_encode | 41.846 [CoV 10.3%] (37.128-45.697) |
| pipeline_compile | 31.190 [CoV 11.7%] (27.311-34.534) |
| encode_sum | 3.585 [CoV 4.1%] (3.528-3.813) |
| encode_to_commit | 0.003 [CoV 8.3%] (0.003-0.004) |
| commit_call_sum | 0.014 [CoV 11.2%] (0.014-0.017) |
| chunk1_queue | 53.167 [CoV 3.0%] (52.070-55.200) |
| chunk1_sched_to_gpu | 0.307 [CoV 66.9%] (0.051-0.309) |
| gpu_busy_sum | 208.669 [CoV 0.8%] (208.532-211.451) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 100.643 [CoV 4.0%] (94.437-101.676) |
| lead_idle_prepare_start_to_gpu | 134.699 [CoV 3.0%] (128.030-135.055) |
| tail | 0.168 [CoV 9.8%] (0.145-0.176) |
| post_wait_to_exit | 0.024 [CoV 9.8%] (0.024-0.028) |
| in_evaluate_unattributed | 0.065 [CoV 22.3%] (0.064-0.093) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 59.170 [CoV 1.2%] (58.780-60.110) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 265.954 [CoV 1.0%] (264.897-270.039) |
| evaluate | 254.273 [CoV 1.1%] (253.617-258.818) |
| outside_evaluate | 11.062 [CoV 4.3%] (10.676-11.625) |
| apply_serving_config | 0.000 [CoV 22.9%] (0.000-0.000) |
| build_position_inputs | 0.468 [CoV 2.1%] (0.465-0.483) |
| named_blocks_span | 10.454 [CoV 4.4%] (10.074-11.003) |
| nb_weights_loop | 0.001 [CoV 13.7%] (0.001-0.001) |
| nb_device_kv_adopt | 8.644 [CoV 1.8%] (8.640-8.921) |
| nb_push_named_blocks | 0.003 [CoV 8.8%] (0.003-0.004) |
| nb_greedy_program | 1.744 [CoV 25.0%] (1.363-2.253) |
| append_plus_greedy | 0.002 [CoV 8.0%] (0.002-0.002) |
| outside_unattributed | 0.136 [CoV 3.5%] (0.129-0.138) |
| prepare | 34.189 [CoV 2.8%] (33.652-35.536) |
| gap_after_prepare | 0.543 [CoV 5.7%] (0.511-0.573) |
| pre_encode | 6.975 [CoV 4.1%] (6.605-7.162) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 2.734 [CoV 2.3%] (2.679-2.802) |
| encode_to_commit | 0.001 [CoV 8.3%] (0.001-0.001) |
| commit_call_sum | 0.011 [CoV 9.6%] (0.011-0.013) |
| chunk1_queue | 4.881 [CoV 2.5%] (4.806-5.045) |
| chunk1_sched_to_gpu | 0.073 [CoV 5.0%] (0.072-0.079) |
| gpu_busy_sum | 204.464 [CoV 1.1%] (203.221-207.588) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 14.949 [CoV 3.2%] (14.226-15.113) |
| lead_idle_prepare_start_to_gpu | 48.927 [CoV 2.8%] (48.374-50.978) |
| tail | 0.150 [CoV 7.3%] (0.148-0.169) |
| post_wait_to_exit | 0.022 [CoV 4.5%] (0.022-0.024) |
| in_evaluate_unattributed | 0.059 [CoV 26.3%] (0.058-0.090) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 77.549 [CoV 0.5%] (76.895-77.573) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.457 [CoV 0.9%] (5.442-5.531) |
| evaluate | 5.392 [CoV 0.7%] (5.380-5.451) |
| outside_evaluate | 0.060 [CoV 11.3%] (0.059-0.072) |
| apply_serving_config | 0.000 [CoV 103.6%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 21.9%] (0.001-0.002) |
| named_blocks_span | 0.019 [CoV 22.3%] (0.016-0.025) |
| nb_weights_loop | 0.001 [CoV 11.8%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 35.6%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 25.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 17.7%] (0.000-0.000) |
| outside_unattributed | 0.040 [CoV 6.5%] (0.039-0.044) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.071 [CoV 17.2%] (0.063-0.088) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.544 [CoV 8.9%] (0.514-0.611) |
| encode_to_commit | 0.000 [CoV 28.4%] (0.000-0.000) |
| commit_call_sum | 0.020 [CoV 9.5%] (0.020-0.024) |
| chunk1_queue | 0.042 [CoV 17.9%] (0.040-0.056) |
| chunk1_sched_to_gpu | 0.072 [CoV 5.4%] (0.069-0.077) |
| gpu_busy_sum | 4.799 [CoV 0.5%] (4.781-4.831) |
| inter_chunk_idle | 0.167 [CoV 5.2%] (0.163-0.180) |
| lead_idle_entry_to_gpu | 0.225 [CoV 14.3%] (0.214-0.278) |
| lead_idle_prepare_start_to_gpu | 0.225 [CoV 14.3%] (0.214-0.278) |
| tail | 0.118 [CoV 6.1%] (0.107-0.119) |
| post_wait_to_exit | 0.015 [CoV 22.9%] (0.013-0.020) |
| in_evaluate_unattributed | 0.058 [CoV 15.6%] (0.051-0.070) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 88.201 [CoV 1.1%] (86.899-88.855) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.503 [CoV 1.9%] (5.489-5.683) |
| evaluate | 5.441 [CoV 1.8%] (5.418-5.602) |
| outside_evaluate | 0.062 [CoV 9.0%] (0.056-0.067) |
| apply_serving_config | 0.000 [CoV 33.3%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 8.8%] (0.001-0.002) |
| named_blocks_span | 0.019 [CoV 14.2%] (0.017-0.022) |
| nb_weights_loop | 0.001 [CoV 5.0%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 13.4%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 13.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 17.4%] (0.000-0.000) |
| outside_unattributed | 0.040 [CoV 6.4%] (0.038-0.043) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.074 [CoV 10.7%] (0.069-0.085) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.547 [CoV 5.8%] (0.520-0.583) |
| encode_to_commit | 0.000 [CoV 20.8%] (0.000-0.000) |
| commit_call_sum | 0.022 [CoV 13.2%] (0.020-0.026) |
| chunk1_queue | 0.049 [CoV 19.1%] (0.044-0.063) |
| chunk1_sched_to_gpu | 0.070 [CoV 2.4%] (0.068-0.071) |
| gpu_busy_sum | 4.828 [CoV 1.4%] (4.827-4.941) |
| inter_chunk_idle | 0.164 [CoV 0.9%] (0.162-0.164) |
| lead_idle_entry_to_gpu | 0.229 [CoV 9.6%] (0.220-0.263) |
| lead_idle_prepare_start_to_gpu | 0.229 [CoV 9.6%] (0.220-0.263) |
| tail | 0.112 [CoV 2.2%] (0.110-0.115) |
| post_wait_to_exit | 0.016 [CoV 14.1%] (0.014-0.018) |
| in_evaluate_unattributed | 0.059 [CoV 8.9%] (0.053-0.064) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 88.211 [CoV 0.4%] (87.945-88.696) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 383.000 [CoV 0.0%] (383.000-383.000) | 4.881 [CoV 2.5%] (4.806-5.045) | 0.073 [CoV 5.0%] (0.072-0.079) | 204.464 [CoV 1.1%] (203.221-207.588) | n/a | 2.734 [CoV 2.3%] (2.679-2.802) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.049 [CoV 19.1%] (0.044-0.063) | 0.070 [CoV 2.4%] (0.068-0.071) | 0.243 [CoV 0.7%] (0.242-0.245) | n/a | 0.036 [CoV 12.1%] (0.032-0.041) |
| 2 | 17.000 [CoV 0.0%] (17.000-17.000) | 0.035 [CoV 18.2%] (0.032-0.045) | 0.317 [CoV 0.5%] (0.316-0.319) | 0.201 [CoV 1.2%] (0.201-0.205) | 0.021 [CoV 1.4%] (0.021-0.021) | 0.028 [CoV 7.2%] (0.027-0.031) |
| 3 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.028 [CoV 12.5%] (0.027-0.034) | 0.502 [CoV 0.9%] (0.500-0.509) | 0.347 [CoV 1.6%] (0.346-0.356) | 0.020 [CoV 7.9%] (0.018-0.021) | 0.038 [CoV 4.4%] (0.036-0.040) |
| 4 | 32.000 [CoV 0.0%] (32.000-32.000) | 0.027 [CoV 7.2%] (0.027-0.031) | 0.820 [CoV 1.3%] (0.819-0.839) | 0.401 [CoV 1.0%] (0.401-0.408) | 0.021 [CoV 0.3%] (0.021-0.021) | 0.048 [CoV 4.7%] (0.046-0.051) |
| 5 | 43.000 [CoV 0.0%] (43.000-43.000) | 0.031 [CoV 5.8%] (0.030-0.034) | 1.178 [CoV 1.1%] (1.175-1.199) | 0.559 [CoV 1.0%] (0.559-0.569) | 0.025 [CoV 0.6%] (0.025-0.025) | 0.064 [CoV 3.4%] (0.062-0.066) |
| 6 | 58.000 [CoV 0.0%] (58.000-58.000) | 0.035 [CoV 4.1%] (0.035-0.038) | 1.671 [CoV 1.1%] (1.666-1.702) | 0.723 [CoV 1.0%] (0.721-0.734) | 0.022 [CoV 0.6%] (0.022-0.022) | 0.081 [CoV 3.4%] (0.078-0.084) |
| 7 | 78.000 [CoV 0.0%] (78.000-78.000) | 0.042 [CoV 16.4%] (0.041-0.055) | 2.307 [CoV 0.6%] (2.296-2.325) | 1.003 [CoV 1.1%] (1.002-1.022) | 0.027 [CoV 0.9%] (0.027-0.027) | 0.107 [CoV 3.5%] (0.105-0.112) |
| 8 | 107.000 [CoV 0.0%] (107.000-107.000) | 0.056 [CoV 4.6%] (0.056-0.061) | 3.188 [CoV 0.9%] (3.169-3.228) | 1.353 [CoV 1.4%] (1.353-1.386) | 0.027 [CoV 1.0%] (0.026-0.027) | 0.137 [CoV 4.6%] (0.132-0.145) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| plan_rebuilt/p1/telemetry.log | 355 | 265 (262-266) | 414.0 | 29.1 | 3165.9 | 1558.3 | 1624.8 | 3335.7 | 1715.6 | aff93fd561a93eff |
| plan_rebuilt/p2/telemetry.log | 346 | 270 (263-271) | 422.5 | 30.0 | 3201.4 | 1556.6 | 1625.0 | 3365.1 | 1713.9 | aff93fd561a93eff |
| plan_rebuilt/p3/telemetry.log | 357 | 265 (261-266) | 413.7 | 29.4 | 3215.8 | 1562.6 | 1625.0 | 3321.1 | 1720.0 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 265.0 ms, CoV of the per-process medians 1.08%
