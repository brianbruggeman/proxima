### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 342.621 [CoV 0.8%] (339.568-344.758) |
| evaluate | 332.535 [CoV 0.7%] (329.551-334.116) |
| outside_evaluate | 10.086 [CoV 3.3%] (10.016-10.642) |
| apply_serving_config | 0.000 [CoV 28.7%] (0.000-0.000) |
| build_position_inputs | 0.458 [CoV 7.7%] (0.450-0.517) |
| named_blocks_span | 9.489 [CoV 3.1%] (9.417-9.973) |
| nb_weights_loop | 0.001 [CoV 63.8%] (0.001-0.003) |
| nb_device_kv_adopt | 8.305 [CoV 3.9%] (8.241-8.835) |
| nb_push_named_blocks | 0.003 [CoV 8.0%] (0.003-0.003) |
| nb_greedy_program | 1.156 [CoV 2.4%] (1.110-1.161) |
| append_plus_greedy | 0.001 [CoV 24.9%] (0.001-0.002) |
| outside_unattributed | 0.144 [CoV 3.6%] (0.140-0.150) |
| prepare | 32.366 [CoV 3.0%] (31.161-33.083) |
| gap_after_prepare | 0.515 [CoV 4.0%] (0.482-0.518) |
| pre_encode | 35.121 [CoV 4.3%] (33.074-36.011) |
| pipeline_compile | 25.369 [CoV 5.3%] (24.053-26.748) |
| encode_sum | 3.294 [CoV 1.3%] (3.264-3.346) |
| encode_to_commit | 0.096 [CoV 13.0%] (0.094-0.118) |
| commit_call_sum | 0.006 [CoV 25.8%] (0.006-0.009) |
| chunk1_queue | 50.370 [CoV 0.4%] (50.358-50.674) |
| chunk1_sched_to_gpu | 0.052 [CoV 6.3%] (0.049-0.056) |
| gpu_busy_sum | 210.422 [CoV 0.0%] (210.389-210.442) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 88.961 [CoV 1.6%] (87.167-89.871) |
| lead_idle_prepare_start_to_gpu | 121.844 [CoV 1.9%] (118.810-123.469) |
| tail | 0.180 [CoV 13.3%] (0.168-0.216) |
| post_wait_to_exit | 0.025 [CoV 11.8%] (0.022-0.028) |
| in_evaluate_unattributed | 0.062 [CoV 2.0%] (0.062-0.064) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 61.415 [CoV 0.8%] (61.025-61.974) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 216.089 [CoV 0.5%] (215.246-217.582) |
| evaluate | 204.865 [CoV 0.1%] (204.836-205.137) |
| outside_evaluate | 11.344 [CoV 6.7%] (10.883-12.387) |
| apply_serving_config | 0.000 [CoV 25.0%] (0.000-0.000) |
| build_position_inputs | 0.468 [CoV 7.5%] (0.450-0.519) |
| named_blocks_span | 10.761 [CoV 6.6%] (10.304-11.726) |
| nb_weights_loop | 0.001 [CoV 4.2%] (0.001-0.001) |
| nb_device_kv_adopt | 8.431 [CoV 5.0%] (8.405-9.173) |
| nb_push_named_blocks | 0.003 [CoV 19.3%] (0.002-0.003) |
| nb_greedy_program | 2.336 [CoV 12.2%] (1.951-2.482) |
| append_plus_greedy | 0.002 [CoV 16.8%] (0.001-0.002) |
| outside_unattributed | 0.129 [CoV 6.4%] (0.125-0.141) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.132 [CoV 5.2%] (0.121-0.133) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.783 [CoV 4.2%] (0.774-0.836) |
| encode_to_commit | 0.062 [CoV 0.4%] (0.062-0.063) |
| commit_call_sum | 0.005 [CoV 4.0%] (0.004-0.005) |
| chunk1_queue | 2.615 [CoV 3.8%] (2.537-2.735) |
| chunk1_sched_to_gpu | 0.075 [CoV 79.6%] (0.073-0.262) |
| gpu_busy_sum | 200.883 [CoV 0.0%] (200.859-200.977) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 3.781 [CoV 1.7%] (3.735-3.866) |
| lead_idle_prepare_start_to_gpu | 3.781 [CoV 1.7%] (3.735-3.866) |
| tail | 0.189 [CoV 3.6%] (0.180-0.193) |
| post_wait_to_exit | 0.023 [CoV 2.5%] (0.022-0.023) |
| in_evaluate_unattributed | 0.079 [CoV 4.2%] (0.076-0.083) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 93.154 [CoV 0.2%] (93.116-93.439) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.759 [CoV 0.6%] (5.721-5.791) |
| evaluate | 5.705 [CoV 0.7%] (5.663-5.742) |
| outside_evaluate | 0.058 [CoV 6.5%] (0.054-0.061) |
| apply_serving_config | 0.000 [CoV 70.3%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 13.2%] (0.001-0.001) |
| named_blocks_span | 0.017 [CoV 11.6%] (0.016-0.020) |
| nb_weights_loop | 0.001 [CoV 12.4%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 33.1%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 32.9%] (0.000-0.000) |
| outside_unattributed | 0.039 [CoV 3.3%] (0.037-0.039) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.067 [CoV 4.0%] (0.066-0.071) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.541 [CoV 2.1%] (0.524-0.545) |
| encode_to_commit | 0.030 [CoV 5.0%] (0.029-0.032) |
| commit_call_sum | 0.017 [CoV 3.3%] (0.017-0.018) |
| chunk1_queue | 0.073 [CoV 4.4%] (0.069-0.075) |
| chunk1_sched_to_gpu | 0.056 [CoV 5.6%] (0.056-0.062) |
| gpu_busy_sum | 5.109 [CoV 1.4%] (4.997-5.130) |
| inter_chunk_idle | 0.166 [CoV 1.9%] (0.163-0.169) |
| lead_idle_entry_to_gpu | 0.262 [CoV 5.7%] (0.251-0.281) |
| lead_idle_prepare_start_to_gpu | 0.262 [CoV 5.7%] (0.251-0.281) |
| tail | 0.131 [CoV 5.8%] (0.130-0.144) |
| post_wait_to_exit | 0.012 [CoV 10.3%] (0.012-0.014) |
| in_evaluate_unattributed | 0.055 [CoV 6.9%] (0.050-0.057) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 88.481 [CoV 0.9%] (87.262-88.629) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.807 [CoV 0.6%] (5.758-5.831) |
| evaluate | 5.747 [CoV 0.6%] (5.703-5.771) |
| outside_evaluate | 0.060 [CoV 4.7%] (0.058-0.063) |
| apply_serving_config | 0.000 [CoV 21.8%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 5.4%] (0.001-0.001) |
| named_blocks_span | 0.018 [CoV 7.9%] (0.017-0.020) |
| nb_weights_loop | 0.001 [CoV 2.9%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 15.4%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 12.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 10.2%] (0.000-0.000) |
| outside_unattributed | 0.039 [CoV 2.9%] (0.038-0.040) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.070 [CoV 5.4%] (0.066-0.074) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.552 [CoV 3.1%] (0.530-0.563) |
| encode_to_commit | 0.030 [CoV 3.7%] (0.030-0.032) |
| commit_call_sum | 0.018 [CoV 5.1%] (0.017-0.019) |
| chunk1_queue | 0.068 [CoV 1.3%] (0.067-0.069) |
| chunk1_sched_to_gpu | 0.058 [CoV 2.8%] (0.058-0.061) |
| gpu_busy_sum | 5.096 [CoV 0.8%] (5.071-5.149) |
| inter_chunk_idle | 0.167 [CoV 1.2%] (0.164-0.168) |
| lead_idle_entry_to_gpu | 0.262 [CoV 4.1%] (0.261-0.280) |
| lead_idle_prepare_start_to_gpu | 0.262 [CoV 4.1%] (0.261-0.280) |
| tail | 0.134 [CoV 0.2%] (0.134-0.135) |
| post_wait_to_exit | 0.013 [CoV 5.4%] (0.013-0.014) |
| in_evaluate_unattributed | 0.057 [CoV 7.3%] (0.054-0.062) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 88.109 [CoV 0.5%] (87.505-88.389) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k_default_ping/p1/telemetry.log | 342 | 217 (215-217) | 375.6 | 20.6 | 3138.6 | 1611.1 | 1625.0 | 3241.4 | 1714.6 | aff93fd561a93eff |
| k_default_ping/p2/telemetry.log | 344 | 216 (215-216) | 371.1 | 21.7 | 3043.2 | 1610.9 | 1625.0 | 3187.9 | 1717.1 | aff93fd561a93eff |
| k_default_ping/p3/telemetry.log | 339 | 215 (214-218) | 370.6 | 21.6 | 3118.2 | 1605.1 | 1625.0 | 3320.0 | 1709.0 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 216.0 ms, CoV of the per-process medians 0.46%
