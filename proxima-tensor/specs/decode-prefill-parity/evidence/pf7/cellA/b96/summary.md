### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 487.815 [CoV 8.6%] (443.204-526.782) |
| evaluate | 477.026 [CoV 8.8%] (432.253-516.049) |
| outside_evaluate | 10.789 [CoV 1.1%] (10.733-10.952) |
| apply_serving_config | 0.000 [CoV 21.7%] (0.000-0.000) |
| build_position_inputs | 0.488 [CoV 0.3%] (0.486-0.489) |
| named_blocks_span | 10.146 [CoV 1.1%] (10.090-10.309) |
| nb_weights_loop | 0.001 [CoV 28.1%] (0.001-0.002) |
| nb_device_kv_adopt | 9.044 [CoV 1.2%] (8.989-9.203) |
| nb_push_named_blocks | 0.005 [CoV 9.2%] (0.005-0.006) |
| nb_greedy_program | 1.075 [CoV 0.1%] (1.074-1.076) |
| append_plus_greedy | 0.002 [CoV 8.7%] (0.002-0.002) |
| outside_unattributed | 0.153 [CoV 0.7%] (0.152-0.154) |
| prepare | 33.133 [CoV 45.9%] (33.037-68.867) |
| gap_after_prepare | 0.538 [CoV 2.0%] (0.537-0.556) |
| pre_encode | 175.652 [CoV 38.3%] (95.058-217.867) |
| pipeline_compile | 165.733 [CoV 41.1%] (84.004-206.802) |
| encode_sum | 3.941 [CoV 1.3%] (3.862-3.956) |
| encode_to_commit | 0.003 [CoV 1.9%] (0.003-0.003) |
| commit_call_sum | 0.023 [CoV 14.4%] (0.018-0.024) |
| chunk1_queue | 53.516 [CoV 0.9%] (52.871-53.771) |
| chunk1_sched_to_gpu | 0.039 [CoV 30.2%] (0.037-0.062) |
| gpu_busy_sum | 210.641 [CoV 1.1%] (207.027-211.193) |
| inter_chunk_idle | 0.043 [CoV 2.9%] (0.041-0.044) |
| lead_idle_entry_to_gpu | 232.290 [CoV 28.7%] (151.071-274.840) |
| lead_idle_prepare_start_to_gpu | 265.864 [CoV 16.6%] (220.495-308.510) |
| tail | 0.220 [CoV 7.8%] (0.202-0.236) |
| post_wait_to_exit | 0.029 [CoV 2.8%] (0.028-0.030) |
| in_evaluate_unattributed | 0.220 [CoV 18.2%] (0.212-0.292) |
| chunk1_ops | 96.000 [CoV 0.0%] (96.000-96.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 43.180 [CoV 9.6%] (39.300-47.651) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 214.721 [CoV 2.3%] (214.523-223.143) |
| evaluate | 203.413 [CoV 2.5%] (203.221-212.251) |
| outside_evaluate | 11.848 [CoV 4.5%] (11.124-12.144) |
| apply_serving_config | 0.000 [CoV 17.7%] (0.000-0.000) |
| build_position_inputs | 0.483 [CoV 1.7%] (0.471-0.487) |
| named_blocks_span | 11.234 [CoV 4.6%] (10.522-11.515) |
| nb_weights_loop | 0.001 [CoV 6.2%] (0.001-0.002) |
| nb_device_kv_adopt | 9.415 [CoV 0.8%] (9.310-9.455) |
| nb_push_named_blocks | 0.005 [CoV 8.7%] (0.004-0.005) |
| nb_greedy_program | 1.219 [CoV 5.5%] (1.162-1.296) |
| append_plus_greedy | 0.002 [CoV 3.1%] (0.002-0.002) |
| outside_unattributed | 0.137 [CoV 1.6%] (0.137-0.140) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.145 [CoV 2.3%] (0.140-0.146) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.904 [CoV 2.0%] (0.877-0.910) |
| encode_to_commit | 0.001 [CoV 16.4%] (0.001-0.001) |
| commit_call_sum | 0.021 [CoV 5.8%] (0.020-0.023) |
| chunk1_queue | 1.034 [CoV 16.4%] (1.032-1.357) |
| chunk1_sched_to_gpu | 0.071 [CoV 12.7%] (0.059-0.075) |
| gpu_busy_sum | 201.412 [CoV 2.4%] (201.232-209.820) |
| inter_chunk_idle | 0.045 [CoV 6.5%] (0.042-0.047) |
| lead_idle_entry_to_gpu | 1.542 [CoV 13.2%] (1.519-1.909) |
| lead_idle_prepare_start_to_gpu | 1.542 [CoV 13.2%] (1.519-1.909) |
| tail | 0.205 [CoV 7.9%] (0.196-0.228) |
| post_wait_to_exit | 0.027 [CoV 10.3%] (0.026-0.032) |
| in_evaluate_unattributed | 0.282 [CoV 5.8%] (0.268-0.301) |
| chunk1_ops | 96.000 [CoV 0.0%] (96.000-96.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 93.888 [CoV 0.1%] (93.676-93.893) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.892 [CoV 3.6%] (5.838-6.233) |
| evaluate | 5.811 [CoV 3.6%] (5.724-6.132) |
| outside_evaluate | 0.090 [CoV 5.9%] (0.082-0.091) |
| apply_serving_config | 0.000 [CoV 5.9%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 3.4%] (0.002-0.002) |
| named_blocks_span | 0.029 [CoV 3.9%] (0.028-0.030) |
| nb_weights_loop | 0.001 [CoV 2.1%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 15.4%] (0.002-0.002) |
| nb_greedy_program | 0.000 [CoV 14.2%] (0.000-0.000) |
| append_plus_greedy | 0.001 [CoV 5.9%] (0.001-0.001) |
| outside_unattributed | 0.055 [CoV 7.8%] (0.049-0.057) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.103 [CoV 2.3%] (0.103-0.107) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.570 [CoV 2.3%] (0.556-0.582) |
| encode_to_commit | 0.001 [CoV 7.1%] (0.001-0.001) |
| commit_call_sum | 0.017 [CoV 4.1%] (0.017-0.018) |
| chunk1_queue | 0.142 [CoV 16.8%] (0.111-0.156) |
| chunk1_sched_to_gpu | 0.068 [CoV 2.4%] (0.067-0.070) |
| gpu_busy_sum | 4.864 [CoV 0.6%] (4.830-4.889) |
| inter_chunk_idle | 0.029 [CoV 1.5%] (0.028-0.029) |
| lead_idle_entry_to_gpu | 0.503 [CoV 0.6%] (0.499-0.505) |
| lead_idle_prepare_start_to_gpu | 0.503 [CoV 0.6%] (0.499-0.505) |
| tail | 0.194 [CoV 14.2%] (0.165-0.220) |
| post_wait_to_exit | 0.022 [CoV 3.7%] (0.021-0.023) |
| in_evaluate_unattributed | 0.078 [CoV 2.8%] (0.076-0.080) |
| chunk1_ops | 96.000 [CoV 0.0%] (96.000-96.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 84.093 [CoV 0.2%] (84.047-84.416) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.656 [CoV 0.8%] (5.614-5.701) |
| evaluate | 5.582 [CoV 0.8%] (5.543-5.628) |
| outside_evaluate | 0.077 [CoV 5.6%] (0.073-0.081) |
| apply_serving_config | 0.000 [CoV 15.2%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 6.9%] (0.002-0.002) |
| named_blocks_span | 0.027 [CoV 6.4%] (0.025-0.029) |
| nb_weights_loop | 0.001 [CoV 8.1%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 15.2%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 10.9%] (0.000-0.000) |
| append_plus_greedy | 0.001 [CoV 14.5%] (0.001-0.001) |
| outside_unattributed | 0.047 [CoV 4.8%] (0.045-0.049) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.098 [CoV 1.8%] (0.095-0.098) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.537 [CoV 1.2%] (0.532-0.545) |
| encode_to_commit | 0.001 [CoV 7.8%] (0.001-0.001) |
| commit_call_sum | 0.016 [CoV 0.8%] (0.015-0.016) |
| chunk1_queue | 0.100 [CoV 2.5%] (0.096-0.101) |
| chunk1_sched_to_gpu | 0.067 [CoV 2.0%] (0.067-0.069) |
| gpu_busy_sum | 4.843 [CoV 0.4%] (4.825-4.862) |
| inter_chunk_idle | 0.028 [CoV 1.1%] (0.028-0.028) |
| lead_idle_entry_to_gpu | 0.429 [CoV 2.1%] (0.428-0.445) |
| lead_idle_prepare_start_to_gpu | 0.429 [CoV 2.1%] (0.428-0.445) |
| tail | 0.154 [CoV 10.0%] (0.136-0.166) |
| post_wait_to_exit | 0.020 [CoV 3.4%] (0.019-0.020) |
| in_evaluate_unattributed | 0.075 [CoV 3.2%] (0.072-0.077) |
| chunk1_ops | 96.000 [CoV 0.0%] (96.000-96.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 85.709 [CoV 0.7%] (85.461-86.554) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 96.000 [CoV 0.0%] (96.000-96.000) | 1.034 [CoV 16.4%] (1.032-1.357) | 0.071 [CoV 12.7%] (0.059-0.075) | 50.512 [CoV 9.6%] (50.420-59.384) | n/a | 0.259 [CoV 4.1%] (0.249-0.270) |
| 2 | 287.000 [CoV 0.0%] (287.000-287.000) | 2.785 [CoV 26.9%] (2.687-4.242) | 48.294 [CoV 7.0%] (48.270-54.418) | 150.622 [CoV 0.1%] (150.486-150.719) | 0.045 [CoV 6.5%] (0.042-0.047) | 0.633 [CoV 2.4%] (0.629-0.657) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 96.000 [CoV 0.0%] (96.000-96.000) | 0.100 [CoV 2.5%] (0.096-0.101) | 0.067 [CoV 2.0%] (0.067-0.069) | 1.182 [CoV 0.6%] (1.178-1.190) | n/a | 0.149 [CoV 2.1%] (0.147-0.153) |
| 2 | 287.000 [CoV 0.0%] (287.000-287.000) | 0.107 [CoV 1.2%] (0.106-0.108) | 0.886 [CoV 0.9%] (0.873-0.887) | 3.661 [CoV 0.4%] (3.643-3.672) | 0.028 [CoV 1.1%] (0.028-0.028) | 0.388 [CoV 1.1%] (0.381-0.389) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| b96/p1/telemetry.log | 443 | 215 (214-218) | 374.3 | 20.2 | 3211.9 | 1616.7 | 1624.0 | 3333.7 | 1721.4 | aff93fd561a93eff |
| b96/p2/telemetry.log | 526 | 223 (215-246) | 394.7 | 23.1 | 3186.4 | 1610.8 | 1624.0 | 3341.3 | 1716.4 | aff93fd561a93eff |
| b96/p3/telemetry.log | 487 | 216 (213-217) | 372.5 | 21.1 | 3225.2 | 1612.9 | 1624.0 | 3411.9 | 1718.4 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 216.0 ms, CoV of the per-process medians 2.00%
