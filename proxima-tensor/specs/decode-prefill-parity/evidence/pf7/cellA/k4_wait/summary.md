### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 341.188 [CoV 0.7%] (340.711-345.215) |
| evaluate | 331.073 [CoV 0.6%] (330.638-334.357) |
| outside_evaluate | 10.116 [CoV 4.3%] (10.072-10.857) |
| apply_serving_config | 0.000 [CoV 19.9%] (0.000-0.000) |
| build_position_inputs | 0.456 [CoV 1.4%] (0.448-0.460) |
| named_blocks_span | 9.522 [CoV 4.4%] (9.471-10.240) |
| nb_weights_loop | 0.001 [CoV 11.2%] (0.001-0.001) |
| nb_device_kv_adopt | 8.418 [CoV 5.2%] (8.307-9.137) |
| nb_push_named_blocks | 0.004 [CoV 17.6%] (0.004-0.005) |
| nb_greedy_program | 1.082 [CoV 3.3%] (1.077-1.143) |
| append_plus_greedy | 0.002 [CoV 20.8%] (0.001-0.002) |
| outside_unattributed | 0.144 [CoV 6.7%] (0.140-0.159) |
| prepare | 30.660 [CoV 1.9%] (30.619-31.663) |
| gap_after_prepare | 0.449 [CoV 7.3%] (0.429-0.494) |
| pre_encode | 34.310 [CoV 5.9%] (33.841-37.639) |
| pipeline_compile | 25.046 [CoV 7.4%] (24.366-27.954) |
| encode_sum | 3.873 [CoV 1.5%] (3.827-3.943) |
| encode_to_commit | 0.003 [CoV 0.0%] (0.003-0.003) |
| commit_call_sum | 0.050 [CoV 6.9%] (0.045-0.052) |
| chunk1_queue | 45.648 [CoV 3.6%] (45.304-48.343) |
| chunk1_sched_to_gpu | 0.296 [CoV 69.0%] (0.062-0.405) |
| gpu_busy_sum | 211.777 [CoV 1.2%] (208.011-212.801) |
| inter_chunk_idle | 4.806 [CoV 2.6%] (4.643-4.886) |
| lead_idle_entry_to_gpu | 85.043 [CoV 2.4%] (82.181-86.200) |
| lead_idle_prepare_start_to_gpu | 117.200 [CoV 2.0%] (113.248-117.289) |
| tail | 0.191 [CoV 21.2%] (0.143-0.221) |
| post_wait_to_exit | 0.027 [CoV 8.8%] (0.025-0.030) |
| in_evaluate_unattributed | 0.274 [CoV 15.3%] (0.212-0.285) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 61.347 [CoV 1.1%] (61.052-62.371) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 217.285 [CoV 0.1%] (217.021-217.339) |
| evaluate | 206.429 [CoV 0.2%] (206.258-207.169) |
| outside_evaluate | 10.763 [CoV 2.3%] (10.671-11.137) |
| apply_serving_config | 0.000 [CoV 10.9%] (0.000-0.000) |
| build_position_inputs | 0.468 [CoV 4.2%] (0.464-0.501) |
| named_blocks_span | 10.200 [CoV 2.2%] (10.076-10.511) |
| nb_weights_loop | 0.001 [CoV 16.0%] (0.001-0.001) |
| nb_device_kv_adopt | 8.883 [CoV 3.9%] (8.442-9.124) |
| nb_push_named_blocks | 0.003 [CoV 15.1%] (0.003-0.004) |
| nb_greedy_program | 1.919 [CoV 30.4%] (1.077-1.979) |
| append_plus_greedy | 0.002 [CoV 5.3%] (0.001-0.002) |
| outside_unattributed | 0.137 [CoV 4.1%] (0.130-0.141) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.127 [CoV 1.2%] (0.125-0.127) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 1.100 [CoV 8.5%] (0.987-1.170) |
| encode_to_commit | 0.001 [CoV 15.5%] (0.001-0.001) |
| commit_call_sum | 0.039 [CoV 20.2%] (0.038-0.054) |
| chunk1_queue | 0.338 [CoV 7.2%] (0.338-0.381) |
| chunk1_sched_to_gpu | 0.169 [CoV 26.6%] (0.111-0.192) |
| gpu_busy_sum | 201.146 [CoV 0.1%] (201.106-201.608) |
| inter_chunk_idle | 4.129 [CoV 2.0%] (4.111-4.265) |
| lead_idle_entry_to_gpu | 0.710 [CoV 0.7%] (0.709-0.718) |
| lead_idle_prepare_start_to_gpu | 0.710 [CoV 0.7%] (0.709-0.718) |
| tail | 0.177 [CoV 14.0%] (0.175-0.223) |
| post_wait_to_exit | 0.026 [CoV 4.4%] (0.025-0.027) |
| in_evaluate_unattributed | 0.289 [CoV 8.3%] (0.249-0.289) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 92.785 [CoV 0.2%] (92.531-92.812) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 6.351 [CoV 0.5%] (6.321-6.383) |
| evaluate | 6.296 [CoV 0.6%] (6.259-6.329) |
| outside_evaluate | 0.056 [CoV 6.4%] (0.052-0.060) |
| apply_serving_config | 0.000 [CoV 35.1%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 6.9%] (0.001-0.001) |
| named_blocks_span | 0.016 [CoV 7.2%] (0.015-0.017) |
| nb_weights_loop | 0.001 [CoV 3.0%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 12.0%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 15.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 15.5%] (0.000-0.000) |
| outside_unattributed | 0.039 [CoV 6.1%] (0.035-0.040) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.061 [CoV 5.8%] (0.055-0.061) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.500 [CoV 3.9%] (0.481-0.520) |
| encode_to_commit | 0.000 [CoV 9.2%] (0.000-0.000) |
| commit_call_sum | 0.014 [CoV 8.1%] (0.013-0.015) |
| chunk1_queue | 0.057 [CoV 7.9%] (0.051-0.060) |
| chunk1_sched_to_gpu | 0.063 [CoV 0.3%] (0.062-0.063) |
| gpu_busy_sum | 5.172 [CoV 1.2%] (5.078-5.187) |
| inter_chunk_idle | 0.742 [CoV 2.1%] (0.737-0.766) |
| lead_idle_entry_to_gpu | 0.206 [CoV 4.0%] (0.205-0.220) |
| lead_idle_prepare_start_to_gpu | 0.206 [CoV 4.0%] (0.205-0.220) |
| tail | 0.129 [CoV 1.3%] (0.127-0.130) |
| post_wait_to_exit | 0.011 [CoV 8.8%] (0.010-0.012) |
| in_evaluate_unattributed | 0.051 [CoV 7.0%] (0.048-0.056) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 80.964 [CoV 0.6%] (80.248-81.151) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 6.367 [CoV 0.2%] (6.355-6.375) |
| evaluate | 6.297 [CoV 0.2%] (6.285-6.315) |
| outside_evaluate | 0.068 [CoV 5.8%] (0.062-0.069) |
| apply_serving_config | 0.000 [CoV 57.9%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 8.1%] (0.001-0.002) |
| named_blocks_span | 0.021 [CoV 8.2%] (0.019-0.023) |
| nb_weights_loop | 0.001 [CoV 7.9%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 9.1%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 36.7%] (0.000-0.000) |
| outside_unattributed | 0.042 [CoV 6.8%] (0.039-0.045) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.078 [CoV 11.3%] (0.065-0.082) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.557 [CoV 7.0%] (0.517-0.594) |
| encode_to_commit | 0.000 [CoV 18.3%] (0.000-0.000) |
| commit_call_sum | 0.021 [CoV 18.0%] (0.017-0.024) |
| chunk1_queue | 0.062 [CoV 6.1%] (0.062-0.069) |
| chunk1_sched_to_gpu | 0.065 [CoV 0.1%] (0.065-0.065) |
| gpu_busy_sum | 5.090 [CoV 1.7%] (4.958-5.119) |
| inter_chunk_idle | 0.762 [CoV 1.4%] (0.761-0.780) |
| lead_idle_entry_to_gpu | 0.259 [CoV 7.6%] (0.231-0.268) |
| lead_idle_prepare_start_to_gpu | 0.259 [CoV 7.6%] (0.231-0.268) |
| tail | 0.130 [CoV 1.4%] (0.128-0.132) |
| post_wait_to_exit | 0.015 [CoV 15.8%] (0.012-0.017) |
| in_evaluate_unattributed | 0.066 [CoV 8.3%] (0.057-0.066) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 79.424 [CoV 0.3%] (79.243-79.780) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k4_wait/p1/telemetry.log | 341 | 218 (217-218) | 388.5 | 20.0 | 3127.4 | 1606.4 | 1624.0 | 3230.2 | 1709.9 | aff93fd561a93eff |
| k4_wait/p2/telemetry.log | 345 | 217 (216-222) | 385.9 | 18.4 | 3198.6 | 1604.1 | 1624.0 | 3301.2 | 1708.9 | aff93fd561a93eff |
| k4_wait/p3/telemetry.log | 340 | 218 (217-219) | 386.4 | 19.9 | 3234.1 | 1605.3 | 1624.0 | 3340.9 | 1709.5 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 217.0 ms, CoV of the per-process medians 0.27%
