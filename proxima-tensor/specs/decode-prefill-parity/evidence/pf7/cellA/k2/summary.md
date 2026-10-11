### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 340.322 [CoV 1.5%] (336.257-346.686) |
| evaluate | 330.382 [CoV 1.4%] (326.300-335.501) |
| outside_evaluate | 9.957 [CoV 6.9%] (9.940-11.185) |
| apply_serving_config | 0.000 [CoV 11.1%] (0.000-0.000) |
| build_position_inputs | 0.477 [CoV 6.7%] (0.456-0.520) |
| named_blocks_span | 9.359 [CoV 7.0%] (9.325-10.514) |
| nb_weights_loop | 0.001 [CoV 32.2%] (0.001-0.002) |
| nb_device_kv_adopt | 8.355 [CoV 7.2%] (8.252-9.379) |
| nb_push_named_blocks | 0.003 [CoV 24.9%] (0.002-0.004) |
| nb_greedy_program | 1.084 [CoV 8.1%] (0.951-1.109) |
| append_plus_greedy | 0.001 [CoV 8.3%] (0.001-0.002) |
| outside_unattributed | 0.140 [CoV 4.7%] (0.137-0.150) |
| prepare | 32.226 [CoV 4.8%] (30.576-33.636) |
| gap_after_prepare | 0.479 [CoV 10.5%] (0.445-0.546) |
| pre_encode | 36.658 [CoV 3.4%] (35.818-38.321) |
| pipeline_compile | 26.635 [CoV 4.9%] (25.549-28.142) |
| encode_sum | 4.000 [CoV 11.3%] (3.395-4.247) |
| encode_to_commit | 0.003 [CoV 7.5%] (0.003-0.003) |
| commit_call_sum | 0.014 [CoV 2.8%] (0.013-0.014) |
| chunk1_queue | 48.409 [CoV 2.5%] (46.860-49.236) |
| chunk1_sched_to_gpu | 0.297 [CoV 11.3%] (0.249-0.310) |
| gpu_busy_sum | 208.970 [CoV 0.2%] (208.379-209.245) |
| inter_chunk_idle | 0.043 [CoV 11.9%] (0.037-0.047) |
| lead_idle_entry_to_gpu | 88.819 [CoV 3.2%] (85.828-91.522) |
| lead_idle_prepare_start_to_gpu | 121.525 [CoV 3.7%] (116.849-125.704) |
| tail | 0.175 [CoV 37.0%] (0.173-0.316) |
| post_wait_to_exit | 0.025 [CoV 6.9%] (0.023-0.027) |
| in_evaluate_unattributed | 0.237 [CoV 20.0%] (0.165-0.242) |
| chunk1_ops | 191.000 [CoV 0.0%] (191.000-191.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 61.230 [CoV 1.5%] (60.356-62.146) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 215.222 [CoV 0.4%] (214.418-216.135) |
| evaluate | 204.078 [CoV 0.1%] (203.641-204.200) |
| outside_evaluate | 11.027 [CoV 7.3%] (10.541-12.150) |
| apply_serving_config | 0.000 [CoV 0.0%] (0.000-0.000) |
| build_position_inputs | 0.468 [CoV 11.2%] (0.465-0.563) |
| named_blocks_span | 10.440 [CoV 7.0%] (9.965-11.430) |
| nb_weights_loop | 0.001 [CoV 11.0%] (0.001-0.001) |
| nb_device_kv_adopt | 8.502 [CoV 3.7%] (8.428-9.011) |
| nb_push_named_blocks | 0.003 [CoV 10.0%] (0.003-0.003) |
| nb_greedy_program | 1.946 [CoV 14.4%] (1.835-2.395) |
| append_plus_greedy | 0.002 [CoV 15.6%] (0.001-0.002) |
| outside_unattributed | 0.139 [CoV 7.3%] (0.129-0.150) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.137 [CoV 6.2%] (0.132-0.149) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.858 [CoV 4.5%] (0.845-0.919) |
| encode_to_commit | 0.001 [CoV 22.9%] (0.001-0.001) |
| commit_call_sum | 0.014 [CoV 4.5%] (0.014-0.015) |
| chunk1_queue | 1.663 [CoV 3.7%] (1.604-1.727) |
| chunk1_sched_to_gpu | 0.052 [CoV 24.1%] (0.038-0.062) |
| gpu_busy_sum | 201.318 [CoV 0.1%] (200.992-201.356) |
| inter_chunk_idle | 0.042 [CoV 2.2%] (0.041-0.043) |
| lead_idle_entry_to_gpu | 2.321 [CoV 3.4%] (2.218-2.374) |
| lead_idle_prepare_start_to_gpu | 2.321 [CoV 3.4%] (2.218-2.374) |
| tail | 0.178 [CoV 4.0%] (0.171-0.185) |
| post_wait_to_exit | 0.024 [CoV 13.0%] (0.022-0.028) |
| in_evaluate_unattributed | 0.289 [CoV 8.4%] (0.264-0.313) |
| chunk1_ops | 191.000 [CoV 0.0%] (191.000-191.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 93.728 [CoV 0.1%] (93.557-93.834) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.878 [CoV 2.4%] (5.699-5.974) |
| evaluate | 5.829 [CoV 2.5%] (5.636-5.917) |
| outside_evaluate | 0.061 [CoV 10.1%] (0.051-0.061) |
| apply_serving_config | 0.000 [CoV 65.8%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 12.9%] (0.001-0.001) |
| named_blocks_span | 0.018 [CoV 15.0%] (0.015-0.020) |
| nb_weights_loop | 0.001 [CoV 11.4%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 20.0%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 25.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 35.3%] (0.000-0.000) |
| outside_unattributed | 0.040 [CoV 8.2%] (0.035-0.041) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.076 [CoV 9.4%] (0.066-0.079) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.487 [CoV 6.8%] (0.451-0.517) |
| encode_to_commit | 0.000 [CoV 7.8%] (0.000-0.000) |
| commit_call_sum | 0.007 [CoV 14.7%] (0.007-0.009) |
| chunk1_queue | 0.109 [CoV 6.8%] (0.104-0.119) |
| chunk1_sched_to_gpu | 0.074 [CoV 2.0%] (0.073-0.076) |
| gpu_busy_sum | 5.101 [CoV 3.1%] (4.879-5.183) |
| inter_chunk_idle | 0.031 [CoV 5.0%] (0.029-0.031) |
| lead_idle_entry_to_gpu | 0.527 [CoV 6.3%] (0.474-0.532) |
| lead_idle_prepare_start_to_gpu | 0.527 [CoV 6.3%] (0.474-0.532) |
| tail | 0.142 [CoV 4.6%] (0.134-0.147) |
| post_wait_to_exit | 0.011 [CoV 17.9%] (0.009-0.013) |
| in_evaluate_unattributed | 0.058 [CoV 13.2%] (0.047-0.060) |
| chunk1_ops | 191.000 [CoV 0.0%] (191.000-191.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 86.450 [CoV 1.0%] (85.334-87.078) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.863 [CoV 1.1%] (5.804-5.937) |
| evaluate | 5.807 [CoV 1.1%] (5.749-5.875) |
| outside_evaluate | 0.063 [CoV 8.0%] (0.056-0.065) |
| apply_serving_config | 0.000 [CoV 25.0%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 8.9%] (0.001-0.001) |
| named_blocks_span | 0.019 [CoV 9.9%] (0.016-0.020) |
| nb_weights_loop | 0.001 [CoV 12.4%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 15.6%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 21.8%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 13.5%] (0.000-0.000) |
| outside_unattributed | 0.042 [CoV 7.9%] (0.037-0.043) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.076 [CoV 9.4%] (0.068-0.082) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.490 [CoV 10.2%] (0.469-0.568) |
| encode_to_commit | 0.000 [CoV 7.7%] (0.000-0.000) |
| commit_call_sum | 0.009 [CoV 7.0%] (0.008-0.009) |
| chunk1_queue | 0.103 [CoV 4.8%] (0.096-0.106) |
| chunk1_sched_to_gpu | 0.074 [CoV 0.5%] (0.074-0.075) |
| gpu_busy_sum | 5.110 [CoV 1.3%] (5.004-5.132) |
| inter_chunk_idle | 0.030 [CoV 1.3%] (0.030-0.031) |
| lead_idle_entry_to_gpu | 0.505 [CoV 6.5%] (0.480-0.546) |
| lead_idle_prepare_start_to_gpu | 0.505 [CoV 6.5%] (0.480-0.546) |
| tail | 0.136 [CoV 0.9%] (0.135-0.138) |
| post_wait_to_exit | 0.013 [CoV 10.5%] (0.010-0.013) |
| in_evaluate_unattributed | 0.059 [CoV 9.3%] (0.053-0.063) |
| chunk1_ops | 191.000 [CoV 0.0%] (191.000-191.000) |
| chunks | 2.000 [CoV 0.0%] (2.000-2.000) |
| busy_over_wall_pct | 85.918 [CoV 0.8%] (85.456-86.762) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k2/p1/telemetry.log | 346 | 216 (214-216) | 375.6 | 19.0 | 3151.4 | 1606.4 | 1624.0 | 3256.1 | 1710.2 | aff93fd561a93eff |
| k2/p2/telemetry.log | 336 | 215 (214-216) | 376.6 | 18.3 | 3133.4 | 1602.9 | 1624.0 | 3238.1 | 1706.7 | aff93fd561a93eff |
| k2/p3/telemetry.log | 340 | 214 (213-214) | 373.5 | 17.2 | 3132.4 | 1604.6 | 1624.0 | 3323.4 | 1708.2 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 215.0 ms, CoV of the per-process medians 0.47%
