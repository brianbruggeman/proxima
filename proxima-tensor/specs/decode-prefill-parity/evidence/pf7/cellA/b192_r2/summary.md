### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 345.849 [CoV 0.4%] (344.983-347.839) |
| evaluate | 335.434 [CoV 0.3%] (334.527-336.591) |
| outside_evaluate | 10.456 [CoV 4.4%] (10.415-11.249) |
| apply_serving_config | 0.000 [CoV 48.3%] (0.000-0.001) |
| build_position_inputs | 0.493 [CoV 7.0%] (0.484-0.550) |
| named_blocks_span | 9.819 [CoV 4.3%] (9.773-10.545) |
| nb_weights_loop | 0.001 [CoV 1.9%] (0.001-0.001) |
| nb_device_kv_adopt | 8.699 [CoV 5.2%] (8.560-9.429) |
| nb_push_named_blocks | 0.007 [CoV 49.6%] (0.003-0.008) |
| nb_greedy_program | 1.090 [CoV 5.3%] (1.085-1.191) |
| append_plus_greedy | 0.002 [CoV 28.2%] (0.001-0.002) |
| outside_unattributed | 0.150 [CoV 1.4%] (0.147-0.151) |
| prepare | 34.299 [CoV 3.3%] (32.612-34.713) |
| gap_after_prepare | 0.517 [CoV 9.7%] (0.501-0.598) |
| pre_encode | 36.821 [CoV 5.9%] (36.453-40.517) |
| pipeline_compile | 27.104 [CoV 6.8%] (26.166-29.793) |
| encode_sum | 3.663 [CoV 3.1%] (3.536-3.765) |
| encode_to_commit | 0.003 [CoV 2.4%] (0.003-0.003) |
| commit_call_sum | 0.016 [CoV 10.3%] (0.013-0.016) |
| chunk1_queue | 49.089 [CoV 1.3%] (48.012-49.098) |
| chunk1_sched_to_gpu | 0.047 [CoV 112.8%] (0.046-0.307) |
| gpu_busy_sum | 210.474 [CoV 1.0%] (208.800-212.823) |
| inter_chunk_idle | 0.043 [CoV 13.3%] (0.034-0.043) |
| lead_idle_entry_to_gpu | 88.984 [CoV 2.1%] (88.708-92.093) |
| lead_idle_prepare_start_to_gpu | 123.605 [CoV 2.2%] (122.112-127.307) |
| tail | 0.160 [CoV 5.8%] (0.149-0.167) |
| post_wait_to_exit | 0.027 [CoV 2.7%] (0.027-0.028) |
| in_evaluate_unattributed | 0.257 [CoV 12.3%] (0.218-0.279) |
| chunk1_ops | 192.000 [CoV 0.0%] (192.000-192.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 61.010 [CoV 1.3%] (60.028-61.536) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 214.877 [CoV 0.1%] (214.623-215.163) |
| evaluate | 203.756 [CoV 0.1%] (203.528-203.789) |
| outside_evaluate | 11.077 [CoV 2.6%] (10.938-11.499) |
| apply_serving_config | 0.000 [CoV 10.3%] (0.000-0.000) |
| build_position_inputs | 0.470 [CoV 1.6%] (0.467-0.481) |
| named_blocks_span | 10.476 [CoV 2.8%] (10.331-10.895) |
| nb_weights_loop | 0.001 [CoV 10.6%] (0.001-0.001) |
| nb_device_kv_adopt | 8.699 [CoV 1.5%] (8.488-8.717) |
| nb_push_named_blocks | 0.003 [CoV 26.3%] (0.002-0.004) |
| nb_greedy_program | 1.595 [CoV 32.5%] (1.277-2.384) |
| append_plus_greedy | 0.002 [CoV 3.0%] (0.002-0.002) |
| outside_unattributed | 0.139 [CoV 2.0%] (0.135-0.140) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.133 [CoV 2.0%] (0.131-0.136) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.871 [CoV 0.6%] (0.867-0.876) |
| encode_to_commit | 0.001 [CoV 17.6%] (0.001-0.001) |
| commit_call_sum | 0.016 [CoV 12.1%] (0.013-0.017) |
| chunk1_queue | 1.703 [CoV 12.9%] (1.602-2.040) |
| chunk1_sched_to_gpu | 0.076 [CoV 4.3%] (0.074-0.080) |
| gpu_busy_sum | 200.700 [CoV 0.0%] (200.698-200.744) |
| inter_chunk_idle | 0.040 [CoV 3.6%] (0.040-0.043) |
| lead_idle_entry_to_gpu | 2.351 [CoV 8.2%] (2.347-2.699) |
| lead_idle_prepare_start_to_gpu | 2.351 [CoV 8.2%] (2.347-2.699) |
| tail | 0.156 [CoV 7.8%] (0.136-0.156) |
| post_wait_to_exit | 0.025 [CoV 19.0%] (0.022-0.031) |
| in_evaluate_unattributed | 0.259 [CoV 6.4%] (0.249-0.281) |
| chunk1_ops | 192.000 [CoV 0.0%] (192.000-192.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 93.707 [CoV 0.1%] (93.592-93.747) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.521 [CoV 0.2%] (5.509-5.526) |
| evaluate | 5.457 [CoV 0.2%] (5.446-5.467) |
| outside_evaluate | 0.055 [CoV 2.8%] (0.055-0.058) |
| apply_serving_config | 0.000 [CoV 70.3%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 3.2%] (0.001-0.001) |
| named_blocks_span | 0.016 [CoV 6.1%] (0.016-0.017) |
| nb_weights_loop | 0.001 [CoV 5.2%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 6.5%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 15.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 12.2%] (0.000-0.000) |
| outside_unattributed | 0.038 [CoV 0.6%] (0.038-0.038) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.068 [CoV 6.2%] (0.064-0.072) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.471 [CoV 1.1%] (0.469-0.479) |
| encode_to_commit | 0.000 [CoV 23.9%] (0.000-0.000) |
| commit_call_sum | 0.007 [CoV 10.4%] (0.007-0.008) |
| chunk1_queue | 0.092 [CoV 16.6%] (0.069-0.094) |
| chunk1_sched_to_gpu | 0.074 [CoV 4.1%] (0.072-0.078) |
| gpu_busy_sum | 4.762 [CoV 0.2%] (4.756-4.771) |
| inter_chunk_idle | 0.029 [CoV 3.3%] (0.028-0.030) |
| lead_idle_entry_to_gpu | 0.476 [CoV 2.0%] (0.467-0.486) |
| lead_idle_prepare_start_to_gpu | 0.476 [CoV 2.0%] (0.467-0.486) |
| tail | 0.122 [CoV 2.2%] (0.118-0.122) |
| post_wait_to_exit | 0.010 [CoV 6.8%] (0.010-0.011) |
| in_evaluate_unattributed | 0.052 [CoV 4.2%] (0.051-0.055) |
| chunk1_ops | 192.000 [CoV 0.0%] (192.000-192.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 86.541 [CoV 0.4%] (86.233-86.911) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.603 [CoV 1.7%] (5.531-5.724) |
| evaluate | 5.537 [CoV 1.8%] (5.473-5.666) |
| outside_evaluate | 0.064 [CoV 9.3%] (0.054-0.064) |
| apply_serving_config | 0.000 [CoV 21.8%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 10.4%] (0.001-0.002) |
| named_blocks_span | 0.021 [CoV 14.1%] (0.016-0.021) |
| nb_weights_loop | 0.001 [CoV 8.7%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 19.1%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 10.4%] (0.000-0.000) |
| outside_unattributed | 0.041 [CoV 5.6%] (0.037-0.041) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.077 [CoV 9.6%] (0.067-0.082) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.490 [CoV 2.3%] (0.476-0.498) |
| encode_to_commit | 0.000 [CoV 18.4%] (0.000-0.000) |
| commit_call_sum | 0.009 [CoV 15.7%] (0.008-0.010) |
| chunk1_queue | 0.091 [CoV 2.6%] (0.088-0.092) |
| chunk1_sched_to_gpu | 0.074 [CoV 2.0%] (0.072-0.075) |
| gpu_busy_sum | 4.796 [CoV 2.1%] (4.773-4.959) |
| inter_chunk_idle | 0.029 [CoV 2.7%] (0.028-0.030) |
| lead_idle_entry_to_gpu | 0.501 [CoV 3.7%] (0.478-0.515) |
| lead_idle_prepare_start_to_gpu | 0.501 [CoV 3.7%] (0.478-0.515) |
| tail | 0.120 [CoV 3.2%] (0.114-0.121) |
| post_wait_to_exit | 0.013 [CoV 13.9%] (0.011-0.014) |
| in_evaluate_unattributed | 0.061 [CoV 9.3%] (0.052-0.062) |
| chunk1_ops | 192.000 [CoV 0.0%] (192.000-192.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 86.264 [CoV 0.5%] (85.704-86.595) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 192.000 [CoV 0.0%] (192.000-192.000) | 1.703 [CoV 12.9%] (1.602-2.040) | 0.076 [CoV 4.3%] (0.074-0.080) | 102.203 [CoV 0.0%] (102.131-102.213) | n/a | 0.437 [CoV 1.7%] (0.436-0.449) |
| 2 | 191.000 [CoV 0.0%] (191.000-191.000) | 2.542 [CoV 9.6%] (2.383-2.870) | 101.091 [CoV 0.1%] (101.078-101.204) | 98.494 [CoV 0.1%] (98.442-98.651) | 0.040 [CoV 3.6%] (0.040-0.043) | 0.431 [CoV 2.8%] (0.424-0.448) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 192.000 [CoV 0.0%] (192.000-192.000) | 0.091 [CoV 2.6%] (0.088-0.092) | 0.074 [CoV 2.0%] (0.072-0.075) | 2.361 [CoV 2.2%] (2.348-2.445) | n/a | 0.250 [CoV 2.6%] (0.241-0.253) |
| 2 | 191.000 [CoV 0.0%] (191.000-191.000) | 0.081 [CoV 2.0%] (0.079-0.082) | 2.230 [CoV 1.7%] (2.219-2.291) | 2.433 [CoV 1.7%] (2.421-2.500) | 0.029 [CoV 2.7%] (0.028-0.030) | 0.241 [CoV 2.1%] (0.234-0.244) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| b192_r2/p1/telemetry.log | 347 | 215 (214-216) | 367.1 | 17.8 | 3143.3 | 1615.2 | 1624.0 | 3245.9 | 1719.1 | aff93fd561a93eff |
| b192_r2/p2/telemetry.log | 345 | 214 (214-216) | 369.1 | 18.5 | 3205.8 | 1607.7 | 1624.0 | 3327.5 | 1713.9 | aff93fd561a93eff |
| b192_r2/p3/telemetry.log | 345 | 214 (214-215) | 373.6 | 18.2 | 3156.5 | 1614.9 | 1624.0 | 3340.2 | 1719.8 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 214.0 ms, CoV of the per-process medians 0.27%
