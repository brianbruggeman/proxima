### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 340.565 [CoV 1.2%] (339.055-346.812) |
| evaluate | 330.252 [CoV 1.2%] (328.809-336.441) |
| outside_evaluate | 10.312 [CoV 0.6%] (10.245-10.371) |
| apply_serving_config | 0.000 [CoV 14.2%] (0.000-0.000) |
| build_position_inputs | 0.460 [CoV 3.0%] (0.452-0.479) |
| named_blocks_span | 9.674 [CoV 0.6%] (9.655-9.756) |
| nb_weights_loop | 0.001 [CoV 8.4%] (0.001-0.001) |
| nb_device_kv_adopt | 8.510 [CoV 0.8%] (8.479-8.616) |
| nb_push_named_blocks | 0.004 [CoV 21.9%] (0.003-0.004) |
| nb_greedy_program | 1.125 [CoV 2.0%] (1.117-1.160) |
| append_plus_greedy | 0.002 [CoV 11.5%] (0.001-0.002) |
| outside_unattributed | 0.153 [CoV 7.7%] (0.136-0.158) |
| prepare | 31.882 [CoV 2.3%] (30.704-31.995) |
| gap_after_prepare | 0.465 [CoV 5.5%] (0.426-0.472) |
| pre_encode | 36.312 [CoV 5.3%] (33.649-37.290) |
| pipeline_compile | 26.264 [CoV 5.8%] (24.634-27.691) |
| encode_sum | 3.429 [CoV 4.8%] (3.329-3.658) |
| encode_to_commit | 0.003 [CoV 18.7%] (0.002-0.003) |
| commit_call_sum | 0.011 [CoV 5.8%] (0.011-0.012) |
| chunk1_queue | 48.047 [CoV 4.9%] (47.665-52.063) |
| chunk1_sched_to_gpu | 0.055 [CoV 108.5%] (0.049-0.312) |
| gpu_busy_sum | 210.742 [CoV 0.8%] (209.400-212.712) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 88.114 [CoV 4.8%] (84.705-93.080) |
| lead_idle_prepare_start_to_gpu | 120.574 [CoV 4.0%] (115.834-125.434) |
| tail | 0.183 [CoV 3.9%] (0.181-0.195) |
| post_wait_to_exit | 0.022 [CoV 3.0%] (0.022-0.023) |
| in_evaluate_unattributed | 0.059 [CoV 2.3%] (0.059-0.062) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 61.486 [CoV 1.6%] (60.765-62.737) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 216.153 [CoV 0.4%] (215.082-216.542) |
| evaluate | 204.808 [CoV 0.1%] (204.448-204.901) |
| outside_evaluate | 10.806 [CoV 4.3%] (10.742-11.604) |
| apply_serving_config | 0.000 [CoV 43.4%] (0.000-0.000) |
| build_position_inputs | 0.450 [CoV 6.2%] (0.436-0.491) |
| named_blocks_span | 10.246 [CoV 4.4%] (10.164-11.006) |
| nb_weights_loop | 0.001 [CoV 12.6%] (0.001-0.001) |
| nb_device_kv_adopt | 8.585 [CoV 3.3%] (8.205-8.758) |
| nb_push_named_blocks | 0.002 [CoV 8.8%] (0.002-0.003) |
| nb_greedy_program | 2.037 [CoV 11.4%] (1.942-2.400) |
| append_plus_greedy | 0.002 [CoV 13.9%] (0.001-0.002) |
| outside_unattributed | 0.134 [CoV 5.9%] (0.128-0.144) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.125 [CoV 1.7%] (0.124-0.128) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.782 [CoV 2.4%] (0.774-0.809) |
| encode_to_commit | 0.001 [CoV 11.4%] (0.001-0.001) |
| commit_call_sum | 0.011 [CoV 15.0%] (0.009-0.013) |
| chunk1_queue | 2.479 [CoV 4.9%] (2.474-2.693) |
| chunk1_sched_to_gpu | 0.080 [CoV 12.0%] (0.065-0.082) |
| gpu_busy_sum | 200.986 [CoV 0.1%] (200.698-201.045) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 3.486 [CoV 5.2%] (3.458-3.792) |
| lead_idle_prepare_start_to_gpu | 3.486 [CoV 5.2%] (3.458-3.792) |
| tail | 0.178 [CoV 9.5%] (0.167-0.201) |
| post_wait_to_exit | 0.023 [CoV 6.8%] (0.022-0.025) |
| in_evaluate_unattributed | 0.074 [CoV 5.9%] (0.073-0.081) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 93.255 [CoV 0.3%] (92.833-93.304) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.787 [CoV 0.3%] (5.767-5.798) |
| evaluate | 5.738 [CoV 0.3%] (5.713-5.745) |
| outside_evaluate | 0.052 [CoV 1.7%] (0.051-0.052) |
| apply_serving_config | 0.000 [CoV 70.3%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 4.2%] (0.001-0.001) |
| named_blocks_span | 0.015 [CoV 1.8%] (0.015-0.015) |
| nb_weights_loop | 0.001 [CoV 6.5%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 13.0%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 0.0%] (0.000-0.000) |
| outside_unattributed | 0.036 [CoV 2.1%] (0.035-0.036) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.061 [CoV 1.4%] (0.061-0.063) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.508 [CoV 1.6%] (0.496-0.511) |
| encode_to_commit | 0.000 [CoV 0.0%] (0.000-0.000) |
| commit_call_sum | 0.015 [CoV 3.5%] (0.015-0.016) |
| chunk1_queue | 0.086 [CoV 1.6%] (0.086-0.089) |
| chunk1_sched_to_gpu | 0.051 [CoV 1.0%] (0.050-0.051) |
| gpu_busy_sum | 5.133 [CoV 0.3%] (5.112-5.138) |
| inter_chunk_idle | 0.170 [CoV 0.4%] (0.169-0.170) |
| lead_idle_entry_to_gpu | 0.237 [CoV 1.2%] (0.236-0.242) |
| lead_idle_prepare_start_to_gpu | 0.237 [CoV 1.2%] (0.236-0.242) |
| tail | 0.137 [CoV 1.4%] (0.135-0.139) |
| post_wait_to_exit | 0.011 [CoV 2.5%] (0.011-0.011) |
| in_evaluate_unattributed | 0.048 [CoV 2.0%] (0.048-0.050) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 88.583 [CoV 0.2%] (88.484-88.771) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.812 [CoV 0.6%] (5.766-5.830) |
| evaluate | 5.742 [CoV 0.5%] (5.711-5.766) |
| outside_evaluate | 0.061 [CoV 15.4%] (0.051-0.069) |
| apply_serving_config | 0.000 [CoV 50.0%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 17.4%] (0.001-0.002) |
| named_blocks_span | 0.018 [CoV 19.1%] (0.015-0.022) |
| nb_weights_loop | 0.001 [CoV 12.2%] (0.001-0.001) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.001 [CoV 13.9%] (0.001-0.001) |
| nb_greedy_program | 0.000 [CoV 13.5%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 32.6%] (0.000-0.000) |
| outside_unattributed | 0.041 [CoV 12.6%] (0.035-0.045) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.071 [CoV 17.4%] (0.060-0.085) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.554 [CoV 8.5%] (0.498-0.590) |
| encode_to_commit | 0.000 [CoV 20.8%] (0.000-0.000) |
| commit_call_sum | 0.018 [CoV 18.4%] (0.014-0.021) |
| chunk1_queue | 0.085 [CoV 1.0%] (0.084-0.086) |
| chunk1_sched_to_gpu | 0.053 [CoV 10.2%] (0.049-0.060) |
| gpu_busy_sum | 5.128 [CoV 0.4%] (5.094-5.129) |
| inter_chunk_idle | 0.167 [CoV 0.5%] (0.166-0.167) |
| lead_idle_entry_to_gpu | 0.248 [CoV 10.8%] (0.226-0.280) |
| lead_idle_prepare_start_to_gpu | 0.248 [CoV 10.8%] (0.226-0.280) |
| tail | 0.133 [CoV 0.9%] (0.133-0.135) |
| post_wait_to_exit | 0.014 [CoV 21.6%] (0.011-0.017) |
| in_evaluate_unattributed | 0.058 [CoV 16.2%] (0.047-0.065) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 88.285 [CoV 1.1%] (87.119-88.948) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)

| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |
|---|---|---|---|---|---|---|---|---|---|---|
| k_default_enqueue/p1/telemetry.log | 346 | 216 (216-217) | 379.9 | 21.4 | 3136.8 | 1606.9 | 1625.0 | 3248.4 | 1710.5 | aff93fd561a93eff |
| k_default_enqueue/p2/telemetry.log | 339 | 215 (214-215) | 367.0 | 18.6 | 3064.0 | 1606.7 | 1625.0 | 3166.7 | 1710.1 | aff93fd561a93eff |
| k_default_enqueue/p3/telemetry.log | 340 | 216 (215-216) | 374.8 | 19.6 | 3235.2 | 1607.4 | 1625.0 | 3340.1 | 1712.8 | aff93fd561a93eff |

warm ttft (gens 2-5) pooled over processes: median 216.0 ms, CoV of the per-process medians 0.27%
