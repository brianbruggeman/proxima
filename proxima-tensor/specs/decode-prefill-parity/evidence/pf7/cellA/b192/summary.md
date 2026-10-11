### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 592.508 [CoV 31.2%] (353.100-679.171) |
| evaluate | 581.703 [CoV 31.9%] (341.790-668.454) |
| outside_evaluate | 10.806 [CoV 2.9%] (10.717-11.310) |
| apply_serving_config | 0.000 [CoV 10.0%] (0.000-0.000) |
| build_position_inputs | 0.487 [CoV 1.5%] (0.483-0.497) |
| named_blocks_span | 10.163 [CoV 3.1%] (10.064-10.666) |
| nb_weights_loop | 0.001 [CoV 10.2%] (0.001-0.001) |
| nb_device_kv_adopt | 9.052 [CoV 3.2%] (8.975-9.510) |
| nb_push_named_blocks | 0.006 [CoV 8.0%] (0.005-0.006) |
| nb_greedy_program | 1.081 [CoV 2.9%] (1.061-1.123) |
| append_plus_greedy | 0.002 [CoV 19.0%] (0.001-0.002) |
| outside_unattributed | 0.154 [CoV 2.3%] (0.153-0.160) |
| prepare | 33.205 [CoV 2.6%] (33.080-34.643) |
| gap_after_prepare | 0.524 [CoV 1.6%] (0.524-0.539) |
| pre_encode | 277.030 [CoV 71.0%] (42.143-333.316) |
| pipeline_compile | 117.227 [CoV 86.1%] (31.258-266.658) |
| encode_sum | 3.956 [CoV 4.9%] (3.879-4.253) |
| encode_to_commit | 0.004 [CoV 27.0%] (0.003-0.005) |
| commit_call_sum | 0.023 [CoV 13.3%] (0.019-0.025) |
| chunk1_queue | 54.536 [CoV 3.3%] (53.448-56.989) |
| chunk1_sched_to_gpu | 0.011 [CoV 138.4%] (-0.004-0.011) |
| gpu_busy_sum | 211.657 [CoV 1.2%] (207.262-211.757) |
| inter_chunk_idle | 0.045 [CoV 15.4%] (0.034-0.046) |
| lead_idle_entry_to_gpu | 335.007 [CoV 56.6%] (98.808-393.958) |
| lead_idle_prepare_start_to_gpu | 368.611 [CoV 50.1%] (133.976-427.701) |
| tail | 1.025 [CoV 149.6%] (0.248-12.622) |
| post_wait_to_exit | 0.030 [CoV 11.5%] (0.028-0.035) |
| in_evaluate_unattributed | 0.240 [CoV 165.9%] (0.236-16.393) |
| chunk1_ops | 192.000 [CoV 0.0%] (192.000-192.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 35.739 [CoV 35.2%] (31.164-58.698) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 244.241 [CoV 10.1%] (214.570-262.631) |
| evaluate | 232.130 [CoV 9.7%] (203.931-247.745) |
| outside_evaluate | 12.111 [CoV 51.6%] (10.639-25.772) |
| apply_serving_config | 0.000 [CoV 26.6%] (0.000-0.000) |
| build_position_inputs | 0.491 [CoV 68.0%] (0.475-1.417) |
| named_blocks_span | 11.417 [CoV 14.2%] (10.044-13.320) |
| nb_weights_loop | 0.002 [CoV 54.5%] (0.001-0.003) |
| nb_device_kv_adopt | 9.247 [CoV 11.7%] (8.884-11.014) |
| nb_push_named_blocks | 0.004 [CoV 7.7%] (0.004-0.005) |
| nb_greedy_program | 1.918 [CoV 44.4%] (1.086-2.806) |
| append_plus_greedy | 0.002 [CoV 38.1%] (0.002-0.004) |
| outside_unattributed | 0.147 [CoV 167.9%] (0.133-13.349) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.154 [CoV 9.2%] (0.136-0.164) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.879 [CoV 39.9%] (0.864-1.654) |
| encode_to_commit | 0.001 [CoV 17.2%] (0.001-0.001) |
| commit_call_sum | 0.021 [CoV 22.1%] (0.018-0.028) |
| chunk1_queue | 2.092 [CoV 40.7%] (1.805-3.722) |
| chunk1_sched_to_gpu | 0.068 [CoV 10.5%] (0.063-0.077) |
| gpu_busy_sum | 212.079 [CoV 3.2%] (201.036-213.075) |
| inter_chunk_idle | 0.043 [CoV 3.6%] (0.041-0.044) |
| lead_idle_entry_to_gpu | 2.773 [CoV 59.0%] (2.444-6.635) |
| lead_idle_prepare_start_to_gpu | 2.773 [CoV 59.0%] (2.444-6.635) |
| tail | 6.318 [CoV 86.7%] (0.167-8.596) |
| post_wait_to_exit | 0.051 [CoV 30.7%] (0.031-0.059) |
| in_evaluate_unattributed | 6.925 [CoV 83.9%] (0.258-8.587) |
| chunk1_ops | 192.000 [CoV 0.0%] (192.000-192.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 93.297 [CoV 5.2%] (85.397-93.708) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 9.121 [CoV 37.0%] (5.676-12.388) |
| evaluate | 8.957 [CoV 36.3%] (5.614-12.044) |
| outside_evaluate | 0.143 [CoV 49.7%] (0.070-0.211) |
| apply_serving_config | 0.000 [CoV 38.5%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 17.3%] (0.002-0.002) |
| named_blocks_span | 0.030 [CoV 77.3%] (0.024-0.092) |
| nb_weights_loop | 0.001 [CoV 12.8%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 13.2%] (0.002-0.002) |
| nb_greedy_program | 0.000 [CoV 27.0%] (0.000-0.000) |
| append_plus_greedy | 0.001 [CoV 51.0%] (0.000-0.001) |
| outside_unattributed | 0.054 [CoV 54.2%] (0.044-0.114) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.137 [CoV 52.4%] (0.095-0.261) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.778 [CoV 51.6%] (0.520-1.431) |
| encode_to_commit | 0.001 [CoV 29.1%] (0.001-0.001) |
| commit_call_sum | 0.019 [CoV 40.6%] (0.014-0.031) |
| chunk1_queue | 0.157 [CoV 38.8%] (0.119-0.251) |
| chunk1_sched_to_gpu | 0.069 [CoV 2.6%] (0.067-0.070) |
| gpu_busy_sum | 6.671 [CoV 20.4%] (4.808-7.243) |
| inter_chunk_idle | 0.033 [CoV 9.9%] (0.030-0.036) |
| lead_idle_entry_to_gpu | 0.877 [CoV 89.0%] (0.556-2.954) |
| lead_idle_prepare_start_to_gpu | 0.877 [CoV 89.0%] (0.556-2.954) |
| tail | 0.281 [CoV 104.1%] (0.130-1.110) |
| post_wait_to_exit | 0.027 [CoV 32.0%] (0.018-0.035) |
| in_evaluate_unattributed | 0.087 [CoV 58.1%] (0.067-0.192) |
| chunk1_ops | 192.000 [CoV 0.0%] (192.000-192.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 78.221 [CoV 18.4%] (58.740-84.998) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 7.862 [CoV 43.4%] (5.694-13.215) |
| evaluate | 7.490 [CoV 43.5%] (5.618-12.870) |
| outside_evaluate | 0.109 [CoV 34.9%] (0.072-0.148) |
| apply_serving_config | 0.001 [CoV 41.9%] (0.000-0.001) |
| build_position_inputs | 0.002 [CoV 18.4%] (0.002-0.002) |
| named_blocks_span | 0.033 [CoV 39.4%] (0.025-0.053) |
| nb_weights_loop | 0.001 [CoV 16.7%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 8.9%] (0.002-0.002) |
| nb_greedy_program | 0.000 [CoV 28.7%] (0.000-0.000) |
| append_plus_greedy | 0.001 [CoV 40.9%] (0.000-0.001) |
| outside_unattributed | 0.055 [CoV 15.3%] (0.044-0.060) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.135 [CoV 25.5%] (0.092-0.156) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.929 [CoV 33.5%] (0.522-1.059) |
| encode_to_commit | 0.001 [CoV 25.0%] (0.001-0.001) |
| commit_call_sum | 0.024 [CoV 45.7%] (0.014-0.036) |
| chunk1_queue | 0.129 [CoV 23.0%] (0.098-0.157) |
| chunk1_sched_to_gpu | 0.072 [CoV 5.4%] (0.070-0.078) |
| gpu_busy_sum | 5.006 [CoV 30.0%] (4.834-8.002) |
| inter_chunk_idle | 0.034 [CoV 13.1%] (0.029-0.038) |
| lead_idle_entry_to_gpu | 1.266 [CoV 42.7%] (0.535-1.353) |
| lead_idle_prepare_start_to_gpu | 1.266 [CoV 42.7%] (0.535-1.353) |
| tail | 0.251 [CoV 86.2%] (0.128-0.733) |
| post_wait_to_exit | 0.029 [CoV 25.7%] (0.018-0.031) |
| in_evaluate_unattributed | 0.097 [CoV 24.1%] (0.067-0.111) |
| chunk1_ops | 192.000 [CoV 0.0%] (192.000-192.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 70.046 [CoV 15.0%] (63.680-85.002) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 192.000 [CoV 0.0%] (192.000-192.000) | 2.092 [CoV 40.7%] (1.805-3.722) | 0.068 [CoV 10.5%] (0.063-0.077) | 113.457 [CoV 6.0%] (102.655-114.566) | n/a | 0.459 [CoV 3.4%] (0.452-0.482) |
| 2 | 191.000 [CoV 0.0%] (191.000-191.000) | 3.173 [CoV 48.5%] (2.637-6.254) | 112.209 [CoV 5.9%] (101.538-113.028) | 98.574 [CoV 0.1%] (98.474-98.623) | 0.043 [CoV 3.6%] (0.041-0.044) | 0.420 [CoV 64.2%] (0.409-1.148) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 192.000 [CoV 0.0%] (192.000-192.000) | 0.129 [CoV 23.0%] (0.098-0.157) | 0.072 [CoV 5.4%] (0.070-0.078) | 2.454 [CoV 28.8%] (2.382-3.865) | n/a | 0.432 [CoV 32.8%] (0.265-0.530) |
| 2 | 191.000 [CoV 0.0%] (191.000-191.000) | 0.104 [CoV 33.5%] (0.074-0.146) | 2.236 [CoV 27.2%] (2.188-3.449) | 2.510 [CoV 30.5%] (2.451-4.071) | 0.034 [CoV 13.1%] (0.029-0.038) | 0.395 [CoV 25.6%] (0.253-0.424) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| b192/p1/telemetry.log | 353 | 215 (214-216) | 371.3 | 19.2 | 3211.1 | 1609.4 | 1624.0 | 3356.7 | 1714.3 | aff93fd561a93eff |
| b192/p2/telemetry.log | 592 | 262 (245-369) | 796.7 | 13.6 | 3210.6 | 1612.6 | 1624.0 | 3321.7 | 1716.2 | aff93fd561a93eff |
| b192/p3/telemetry.log | 679 | 244 (226-272) | 540.7 | 17.7 | 3162.6 | 1617.6 | 1624.0 | 3321.4 | 1721.3 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 244.0 ms, CoV of the per-process medians 9.87%
