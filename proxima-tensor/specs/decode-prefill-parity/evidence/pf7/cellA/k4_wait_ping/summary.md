### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 443.646 [CoV 20.0%] (428.899-606.444) |
| evaluate | 423.704 [CoV 20.7%] (415.971-590.864) |
| outside_evaluate | 15.580 [CoV 21.9%] (12.929-19.942) |
| apply_serving_config | 0.000 [CoV 17.7%] (0.000-0.000) |
| build_position_inputs | 0.692 [CoV 29.9%] (0.484-0.898) |
| named_blocks_span | 14.941 [CoV 22.3%] (12.080-18.860) |
| nb_weights_loop | 0.006 [CoV 59.6%] (0.001-0.006) |
| nb_device_kv_adopt | 13.691 [CoV 21.8%] (10.933-16.985) |
| nb_push_named_blocks | 0.008 [CoV 43.5%] (0.006-0.014) |
| nb_greedy_program | 1.219 [CoV 27.7%] (1.107-1.817) |
| append_plus_greedy | 0.002 [CoV 3.3%] (0.002-0.002) |
| outside_unattributed | 0.155 [CoV 10.1%] (0.153-0.182) |
| prepare | 89.097 [CoV 78.4%] (38.685-210.749) |
| gap_after_prepare | 0.572 [CoV 36.4%] (0.530-0.990) |
| pre_encode | 68.769 [CoV 18.5%] (55.325-80.567) |
| pipeline_compile | 57.768 [CoV 21.3%] (44.044-68.051) |
| encode_sum | 4.658 [CoV 109.2%] (4.550-28.156) |
| encode_to_commit | 0.132 [CoV 115.3%] (0.116-0.867) |
| commit_call_sum | 0.094 [CoV 40.6%] (0.089-0.175) |
| chunk1_queue | 52.107 [CoV 30.9%] (51.296-85.416) |
| chunk1_sched_to_gpu | 0.032 [CoV 173.8%] (-29.057-0.038) |
| gpu_busy_sum | 210.574 [CoV 0.6%] (208.666-210.843) |
| inter_chunk_idle | 28.264 [CoV 74.1%] (5.022-41.527) |
| lead_idle_entry_to_gpu | 128.848 [CoV 10.9%] (109.496-135.844) |
| lead_idle_prepare_start_to_gpu | 199.582 [CoV 37.4%] (175.101-340.127) |
| tail | 0.254 [CoV 162.5%] (0.141-9.229) |
| post_wait_to_exit | 0.039 [CoV 49.4%] (0.029-0.073) |
| in_evaluate_unattributed | 0.329 [CoV 36.6%] (0.240-0.496) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 47.465 [CoV 18.5%] (34.408-49.159) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 227.648 [CoV 8.7%] (225.616-262.722) |
| evaluate | 215.841 [CoV 8.5%] (213.203-247.735) |
| outside_evaluate | 12.602 [CoV 10.8%] (12.412-14.987) |
| apply_serving_config | 0.000 [CoV 34.6%] (0.000-0.000) |
| build_position_inputs | 0.492 [CoV 8.1%] (0.464-0.544) |
| named_blocks_span | 11.964 [CoV 11.0%] (11.811-14.299) |
| nb_weights_loop | 0.002 [CoV 29.8%] (0.002-0.003) |
| nb_device_kv_adopt | 9.950 [CoV 18.7%] (9.251-13.035) |
| nb_push_named_blocks | 0.004 [CoV 3.9%] (0.004-0.004) |
| nb_greedy_program | 1.492 [CoV 32.0%] (1.458-2.478) |
| append_plus_greedy | 0.002 [CoV 5.9%] (0.002-0.002) |
| outside_unattributed | 0.142 [CoV 3.3%] (0.135-0.144) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.148 [CoV 8.2%] (0.140-0.164) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 2.008 [CoV 110.5%] (1.367-10.548) |
| encode_to_commit | 0.075 [CoV 166.0%] (0.056-4.587) |
| commit_call_sum | 0.095 [CoV 150.2%] (0.088-1.884) |
| chunk1_queue | 0.430 [CoV 116.4%] (0.342-2.758) |
| chunk1_sched_to_gpu | 0.069 [CoV 37.1%] (0.056-0.112) |
| gpu_busy_sum | 208.671 [CoV 1.2%] (207.515-212.213) |
| inter_chunk_idle | 5.924 [CoV 33.8%] (4.471-8.711) |
| lead_idle_entry_to_gpu | 0.855 [CoV 132.6%] (0.744-8.619) |
| lead_idle_prepare_start_to_gpu | 0.855 [CoV 132.6%] (0.744-8.619) |
| tail | 0.243 [CoV 164.5%] (0.149-11.281) |
| post_wait_to_exit | 0.029 [CoV 4.9%] (0.029-0.032) |
| in_evaluate_unattributed | 0.297 [CoV 43.2%] (0.258-0.552) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 91.776 [CoV 3.6%] (86.238-91.977) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 10.297 [CoV 52.6%] (8.489-21.530) |
| evaluate | 10.218 [CoV 49.3%] (8.334-20.174) |
| outside_evaluate | 0.089 [CoV 45.2%] (0.089-0.183) |
| apply_serving_config | 0.000 [CoV 14.3%] (0.000-0.001) |
| build_position_inputs | 0.002 [CoV 5.1%] (0.002-0.002) |
| named_blocks_span | 0.031 [CoV 30.8%] (0.030-0.050) |
| nb_weights_loop | 0.001 [CoV 7.9%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 5.1%] (0.002-0.003) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.001 [CoV 11.3%] (0.001-0.001) |
| outside_unattributed | 0.057 [CoV 23.6%] (0.056-0.083) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.103 [CoV 16.6%] (0.099-0.133) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 1.085 [CoV 77.8%] (0.891-3.393) |
| encode_to_commit | 0.095 [CoV 31.3%] (0.061-0.118) |
| commit_call_sum | 0.064 [CoV 112.0%] (0.055-0.387) |
| chunk1_queue | 0.105 [CoV 44.7%] (0.073-0.176) |
| chunk1_sched_to_gpu | 0.069 [CoV 5.0%] (0.067-0.074) |
| gpu_busy_sum | 5.817 [CoV 53.7%] (5.497-13.268) |
| inter_chunk_idle | 1.550 [CoV 69.6%] (1.318-4.309) |
| lead_idle_entry_to_gpu | 0.572 [CoV 33.2%] (0.568-0.975) |
| lead_idle_prepare_start_to_gpu | 0.572 [CoV 33.2%] (0.568-0.975) |
| tail | 0.220 [CoV 72.3%] (0.176-0.620) |
| post_wait_to_exit | 0.030 [CoV 12.9%] (0.024-0.030) |
| in_evaluate_unattributed | 0.087 [CoV 6.0%] (0.086-0.096) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 71.252 [CoV 17.5%] (51.894-72.069) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 7.359 [CoV 65.8%] (6.644-19.837) |
| evaluate | 7.187 [CoV 64.2%] (6.555-18.977) |
| outside_evaluate | 0.088 [CoV 17.0%] (0.081-0.111) |
| apply_serving_config | 0.000 [CoV 10.0%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 5.2%] (0.002-0.002) |
| named_blocks_span | 0.030 [CoV 8.2%] (0.029-0.034) |
| nb_weights_loop | 0.001 [CoV 5.4%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.002 [CoV 8.2%] (0.002-0.003) |
| nb_greedy_program | 0.000 [CoV 9.0%] (0.000-0.000) |
| append_plus_greedy | 0.001 [CoV 11.1%] (0.001-0.001) |
| outside_unattributed | 0.054 [CoV 8.2%] (0.048-0.056) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.106 [CoV 9.9%] (0.099-0.120) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 1.043 [CoV 54.6%] (0.709-2.033) |
| encode_to_commit | 0.065 [CoV 49.5%] (0.064-0.142) |
| commit_call_sum | 0.054 [CoV 108.4%] (0.049-0.310) |
| chunk1_queue | 0.073 [CoV 31.9%] (0.069-0.119) |
| chunk1_sched_to_gpu | 0.067 [CoV 7.8%] (0.066-0.076) |
| gpu_busy_sum | 4.968 [CoV 26.7%] (4.865-7.605) |
| inter_chunk_idle | 1.195 [CoV 90.8%] (0.918-4.532) |
| lead_idle_entry_to_gpu | 0.483 [CoV 43.6%] (0.373-0.845) |
| lead_idle_prepare_start_to_gpu | 0.483 [CoV 43.6%] (0.373-0.845) |
| tail | 0.174 [CoV 114.6%] (0.132-1.046) |
| post_wait_to_exit | 0.026 [CoV 10.6%] (0.025-0.030) |
| in_evaluate_unattributed | 0.090 [CoV 8.6%] (0.080-0.095) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 4.000 [CoV 0.0%] (4.000-4.000) |
| busy_over_wall_pct | 67.877 [CoV 23.0%] (46.700-74.513) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### prefill step 0: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.430 [CoV 116.4%] (0.342-2.758) | 0.069 [CoV 37.1%] (0.056-0.112) | 16.054 [CoV 14.7%] (15.922-20.449) | n/a | 0.085 [CoV 2.3%] (0.084-0.088) |
| 2 | 86.000 [CoV 0.0%] (86.000-86.000) | 1.621 [CoV 40.5%] (0.949-2.250) | 0.061 [CoV 4.4%] (0.061-0.065) | 49.973 [CoV 2.0%] (49.630-51.512) | 2.490 [CoV 37.8%] (1.229-2.754) | 0.301 [CoV 143.1%] (0.259-4.268) |
| 3 | 116.000 [CoV 0.0%] (116.000-116.000) | 1.322 [CoV 7.0%] (1.272-1.455) | 0.064 [CoV 10.2%] (0.055-0.067) | 61.052 [CoV 0.1%] (60.946-61.060) | 1.641 [CoV 5.9%] (1.583-1.776) | 0.458 [CoV 35.2%] (0.453-0.804) |
| 4 | 157.000 [CoV 0.0%] (157.000-157.000) | 1.658 [CoV 8.0%] (1.450-1.683) | 0.061 [CoV 8.1%] (0.056-0.066) | 81.245 [CoV 0.1%] (81.150-81.287) | 1.891 [CoV 54.7%] (1.727-4.307) | 1.206 [CoV 83.3%] (0.509-3.085) |

### decode steps >= 2: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms

| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |
|---|---|---|---|---|---|---|
| 1 | 24.000 [CoV 0.0%] (24.000-24.000) | 0.073 [CoV 31.9%] (0.069-0.119) | 0.067 [CoV 7.8%] (0.066-0.076) | 0.252 [CoV 30.6%] (0.251-0.413) | n/a | 0.059 [CoV 8.1%] (0.052-0.061) |
| 2 | 86.000 [CoV 0.0%] (86.000-86.000) | 0.097 [CoV 21.8%] (0.073-0.115) | 0.056 [CoV 9.9%] (0.054-0.065) | 1.150 [CoV 26.0%] (1.131-1.745) | 0.347 [CoV 57.6%] (0.248-0.731) | 0.163 [CoV 36.6%] (0.160-0.291) |
| 3 | 116.000 [CoV 0.0%] (116.000-116.000) | 0.111 [CoV 7.3%] (0.099-0.113) | 0.063 [CoV 10.3%] (0.059-0.072) | 1.465 [CoV 26.2%] (1.433-2.222) | 0.396 [CoV 65.9%] (0.319-1.011) | 0.223 [CoV 42.7%] (0.204-0.423) |
| 4 | 157.000 [CoV 0.0%] (157.000-157.000) | 0.110 [CoV 9.1%] (0.103-0.123) | 0.060 [CoV 12.9%] (0.058-0.073) | 2.085 [CoV 30.6%] (2.047-3.397) | 0.414 [CoV 76.9%] (0.312-1.225) | 0.334 [CoV 27.6%] (0.255-0.444) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k4_wait_ping/p1/telemetry.log | 443 | 228 (218-233) | 426.2 | 23.2 | 3171.6 | 1606.6 | 1624.0 | 3335.2 | 1711.8 | aff93fd561a93eff |
| k4_wait_ping/p2/telemetry.log | 606 | 262 (240-292) | 891.8 | 14.2 | 3145.3 | 1616.5 | 1624.0 | 3335.1 | 1723.0 | aff93fd561a93eff |
| k4_wait_ping/p3/telemetry.log | 428 | 225 (219-235) | 439.3 | 23.3 | 3168.0 | 1608.2 | 1624.0 | 3300.9 | 1715.8 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 233.0 ms, CoV of the per-process medians 8.62%
