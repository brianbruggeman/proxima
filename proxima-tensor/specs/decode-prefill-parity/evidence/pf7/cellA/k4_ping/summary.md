### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 345.337 [CoV 0.4%] (343.990-346.757) |
| evaluate | 335.119 [CoV 0.4%] (333.561-336.445) |
| outside_evaluate | 10.312 [CoV 1.0%] (10.218-10.428) |
| apply_serving_config | 0.000 [CoV 28.5%] (0.000-0.000) |
| build_position_inputs | 0.476 [CoV 1.5%] (0.465-0.479) |
| named_blocks_span | 9.678 [CoV 1.0%] (9.600-9.798) |
| nb_weights_loop | 0.001 [CoV 12.5%] (0.001-0.001) |
| nb_device_kv_adopt | 8.585 [CoV 1.0%] (8.538-8.701) |
| nb_push_named_blocks | 0.005 [CoV 23.9%] (0.003-0.005) |
| nb_greedy_program | 1.070 [CoV 1.9%] (1.036-1.071) |
| append_plus_greedy | 0.001 [CoV 26.0%] (0.001-0.002) |
| outside_unattributed | 0.151 [CoV 2.3%] (0.150-0.156) |
| prepare | 33.173 [CoV 0.3%] (33.112-33.337) |
| gap_after_prepare | 0.543 [CoV 7.0%] (0.496-0.570) |
| pre_encode | 40.817 [CoV 5.9%] (37.257-41.654) |
| pipeline_compile | 29.170 [CoV 5.6%] (27.438-30.684) |
| encode_sum | 3.813 [CoV 2.3%] (3.787-3.948) |
| encode_to_commit | 0.077 [CoV 12.4%] (0.073-0.092) |
| commit_call_sum | 0.014 [CoV 23.2%] (0.014-0.021) |
| chunk1_queue | 48.190 [CoV 5.0%] (45.455-50.281) |
| chunk1_sched_to_gpu | 0.265 [CoV 65.5%] (0.053-0.308) |
| gpu_busy_sum | 209.584 [CoV 0.7%] (208.539-211.575) |
| inter_chunk_idle | 0.118 [CoV 12.5%] (0.099-0.127) |
| lead_idle_entry_to_gpu | 90.559 [CoV 1.9%] (89.297-92.726) |
| lead_idle_prepare_start_to_gpu | 124.466 [CoV 1.3%] (123.014-126.334) |
| tail | 0.151 [CoV 0.6%] (0.151-0.152) |
| post_wait_to_exit | 0.029 [CoV 2.4%] (0.028-0.029) |
| in_evaluate_unattributed | 0.247 [CoV 4.3%] (0.231-0.251) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 60.624 [CoV 0.7%] (60.441-61.266) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 213.640 [CoV 0.2%] (213.076-213.725) |
| evaluate | 202.538 [CoV 0.1%] (202.109-202.541) |
| outside_evaluate | 11.161 [CoV 1.1%] (10.967-11.178) |
| apply_serving_config | 0.000 [CoV 16.6%] (0.000-0.000) |
| build_position_inputs | 0.466 [CoV 1.9%] (0.466-0.482) |
| named_blocks_span | 10.552 [CoV 1.2%] (10.345-10.575) |
| nb_weights_loop | 0.001 [CoV 15.8%] (0.001-0.001) |
| nb_device_kv_adopt | 9.004 [CoV 3.6%] (8.992-9.565) |
| nb_push_named_blocks | 0.003 [CoV 10.9%] (0.003-0.003) |
| nb_greedy_program | 1.540 [CoV 15.0%] (1.173-1.542) |
| append_plus_greedy | 0.002 [CoV 16.1%] (0.001-0.002) |
| outside_unattributed | 0.145 [CoV 4.9%] (0.134-0.146) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.148 [CoV 6.3%] (0.140-0.159) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.992 [CoV 2.4%] (0.968-1.015) |
| encode_to_commit | 0.065 [CoV 1.4%] (0.064-0.065) |
| commit_call_sum | 0.013 [CoV 14.5%] (0.013-0.016) |
| chunk1_queue | 0.357 [CoV 10.8%] (0.340-0.417) |
| chunk1_sched_to_gpu | 0.097 [CoV 8.2%] (0.084-0.098) |
| gpu_busy_sum | 200.788 [CoV 0.1%] (200.733-201.207) |
| inter_chunk_idle | 0.122 [CoV 0.5%] (0.122-0.123) |
| lead_idle_entry_to_gpu | 0.790 [CoV 4.0%] (0.742-0.800) |
| lead_idle_prepare_start_to_gpu | 0.790 [CoV 4.0%] (0.742-0.800) |
| tail | 0.181 [CoV 2.3%] (0.175-0.182) |
| post_wait_to_exit | 0.027 [CoV 3.1%] (0.027-0.029) |
| in_evaluate_unattributed | 0.277 [CoV 8.5%] (0.261-0.309) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 94.220 [CoV 0.1%] (94.186-94.400) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.449 [CoV 0.8%] (5.379-5.456) |
| evaluate | 5.381 [CoV 0.7%] (5.322-5.386) |
| outside_evaluate | 0.069 [CoV 10.3%] (0.059-0.072) |
| apply_serving_config | 0.000 [CoV 13.5%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 13.7%] (0.001-0.002) |
| named_blocks_span | 0.024 [CoV 22.6%] (0.017-0.027) |
| nb_weights_loop | 0.001 [CoV 9.3%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 32.6%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 12.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 33.1%] (0.000-0.000) |
| outside_unattributed | 0.043 [CoV 4.2%] (0.040-0.043) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.088 [CoV 11.2%] (0.072-0.088) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.544 [CoV 5.0%] (0.509-0.561) |
| encode_to_commit | 0.037 [CoV 27.0%] (0.023-0.039) |
| commit_call_sum | 0.015 [CoV 9.2%] (0.015-0.018) |
| chunk1_queue | 0.055 [CoV 18.3%] (0.044-0.063) |
| chunk1_sched_to_gpu | 0.072 [CoV 3.0%] (0.069-0.073) |
| gpu_busy_sum | 4.783 [CoV 0.2%] (4.773-4.790) |
| inter_chunk_idle | 0.080 [CoV 2.3%] (0.079-0.083) |
| lead_idle_entry_to_gpu | 0.300 [CoV 9.5%] (0.256-0.306) |
| lead_idle_prepare_start_to_gpu | 0.300 [CoV 9.5%] (0.256-0.306) |
| tail | 0.114 [CoV 5.8%] (0.113-0.125) |
| post_wait_to_exit | 0.018 [CoV 21.8%] (0.012-0.019) |
| in_evaluate_unattributed | 0.063 [CoV 11.3%] (0.056-0.070) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 87.988 [CoV 0.8%] (87.859-89.129) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.455 [CoV 2.9%] (5.427-5.716) |
| evaluate | 5.384 [CoV 2.8%] (5.360-5.637) |
| outside_evaluate | 0.067 [CoV 7.8%] (0.066-0.076) |
| apply_serving_config | 0.000 [CoV 10.9%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 7.7%] (0.002-0.002) |
| named_blocks_span | 0.023 [CoV 8.7%] (0.022-0.026) |
| nb_weights_loop | 0.001 [CoV 10.8%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 14.7%] (0.001-0.002) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 21.7%] (0.000-0.001) |
| outside_unattributed | 0.042 [CoV 6.7%] (0.042-0.047) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.087 [CoV 5.9%] (0.086-0.096) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.555 [CoV 5.0%] (0.543-0.597) |
| encode_to_commit | 0.036 [CoV 21.6%] (0.034-0.050) |
| commit_call_sum | 0.016 [CoV 11.3%] (0.016-0.019) |
| chunk1_queue | 0.054 [CoV 7.3%] (0.054-0.061) |
| chunk1_sched_to_gpu | 0.070 [CoV 0.4%] (0.070-0.070) |
| gpu_busy_sum | 4.794 [CoV 0.7%] (4.768-4.837) |
| inter_chunk_idle | 0.080 [CoV 2.4%] (0.079-0.083) |
| lead_idle_entry_to_gpu | 0.302 [CoV 8.3%] (0.298-0.345) |
| lead_idle_prepare_start_to_gpu | 0.302 [CoV 8.3%] (0.298-0.345) |
| tail | 0.115 [CoV 9.8%] (0.114-0.135) |
| post_wait_to_exit | 0.017 [CoV 8.9%] (0.016-0.019) |
| in_evaluate_unattributed | 0.065 [CoV 6.5%] (0.064-0.072) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 88.186 [CoV 1.1%] (86.487-88.239) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.357 [CoV 10.8%] (0.340-0.417) | 0.097 [CoV 8.2%] (0.084-0.098) | 9.395 [CoV 0.5%] (9.392-9.477) | n/a | 0.083 [CoV 1.8%] (0.081-0.084) |
| 2 | 86.000 [CoV 0.0%] (86.000-86.000) | 0.884 [CoV 20.2%] (0.825-1.189) | 8.736 [CoV 0.7%] (8.632-8.750) | 49.421 [CoV 0.2%] (49.261-49.479) | 0.042 [CoV 2.9%] (0.040-0.043) | 0.287 [CoV 3.4%] (0.283-0.301) |
| 3 | 116.000 [CoV 0.0%] (116.000-116.000) | 1.634 [CoV 11.5%] (1.608-1.967) | 57.069 [CoV 0.1%] (57.056-57.149) | 60.893 [CoV 0.2%] (60.871-61.052) | 0.042 [CoV 5.6%] (0.041-0.045) | 0.269 [CoV 1.9%] (0.264-0.274) |
| 4 | 157.000 [CoV 0.0%] (157.000-157.000) | 2.370 [CoV 9.5%] (2.315-2.749) | 116.907 [CoV 0.0%] (116.905-116.987) | 81.135 [CoV 0.1%] (81.048-81.207) | 0.040 [CoV 9.2%] (0.036-0.043) | 0.340 [CoV 7.8%] (0.339-0.387) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.054 [CoV 7.3%] (0.054-0.061) | 0.070 [CoV 0.4%] (0.070-0.070) | 0.241 [CoV 1.4%] (0.240-0.246) | n/a | 0.042 [CoV 10.4%] (0.041-0.049) |
| 2 | 86.000 [CoV 0.0%] (86.000-86.000) | 0.047 [CoV 8.4%] (0.047-0.054) | 0.203 [CoV 0.7%] (0.203-0.205) | 1.111 [CoV 0.8%] (1.106-1.124) | 0.023 [CoV 5.5%] (0.022-0.025) | 0.133 [CoV 5.8%] (0.130-0.145) |
| 3 | 116.000 [CoV 0.0%] (116.000-116.000) | 0.054 [CoV 8.7%] (0.053-0.062) | 1.176 [CoV 0.5%] (1.167-1.177) | 1.422 [CoV 0.7%] (1.416-1.434) | 0.028 [CoV 0.3%] (0.028-0.029) | 0.163 [CoV 5.4%] (0.161-0.178) |
| 4 | 157.000 [CoV 0.0%] (157.000-157.000) | 0.074 [CoV 2.6%] (0.073-0.077) | 2.393 [CoV 0.3%] (2.380-2.396) | 2.019 [CoV 0.8%] (2.008-2.039) | 0.028 [CoV 0.9%] (0.028-0.028) | 0.206 [CoV 4.0%] (0.205-0.221) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k4_ping/p1/telemetry.log | 346 | 213 (212-224) | 408.7 | 21.1 | 3093.5 | 1608.9 | 1624.0 | 3237.2 | 1713.6 | aff93fd561a93eff |
| k4_ping/p2/telemetry.log | 344 | 213 (213-216) | 365.3 | 21.4 | 3183.5 | 1613.8 | 1624.0 | 3299.4 | 1717.8 | aff93fd561a93eff |
| k4_ping/p3/telemetry.log | 345 | 213 (211-213) | 361.7 | 20.5 | 3142.7 | 1615.6 | 1624.0 | 3271.0 | 1719.3 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 213.0 ms, CoV of the per-process medians 0.00%
