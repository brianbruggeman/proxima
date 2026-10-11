### prefill step 0, generation 1 (cold)

N executions per process: [1, 1, 1] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 342.002 [CoV 4.6%] (338.606-367.954) |
| evaluate | 331.870 [CoV 4.7%] (328.577-357.697) |
| outside_evaluate | 10.131 [CoV 1.1%] (10.029-10.257) |
| apply_serving_config | 0.000 [CoV 15.7%] (0.000-0.000) |
| build_position_inputs | 0.467 [CoV 2.5%] (0.450-0.472) |
| named_blocks_span | 9.547 [CoV 0.9%] (9.422-9.591) |
| nb_weights_loop | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.002 [CoV 15.9%] (0.002-0.002) |
| outside_unattributed | 0.137 [CoV 22.0%] (0.132-0.193) |
| prepare | 31.604 [CoV 1.9%] (31.156-32.372) |
| gap_after_prepare | 0.466 [CoV 11.2%] (0.429-0.535) |
| pre_encode | 36.127 [CoV 32.0%] (33.896-58.692) |
| pipeline_compile | 26.706 [CoV 37.9%] (24.693-47.239) |
| encode_sum | 3.454 [CoV 2.7%] (3.359-3.548) |
| encode_to_commit | 0.002 [CoV 15.0%] (0.002-0.002) |
| commit_call_sum | 0.012 [CoV 17.5%] (0.011-0.015) |
| chunk1_queue | 49.215 [CoV 0.4%] (49.127-49.515) |
| chunk1_sched_to_gpu | 0.052 [CoV 57.5%] (0.049-0.126) |
| gpu_busy_sum | 211.057 [CoV 1.0%] (208.859-213.212) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 88.766 [CoV 14.6%] (86.638-111.802) |
| lead_idle_prepare_start_to_gpu | 120.351 [CoV 10.8%] (119.476-143.941) |
| tail | 0.193 [CoV 12.7%] (0.156-0.199) |
| post_wait_to_exit | 0.035 [CoV 32.9%] (0.031-0.055) |
| in_evaluate_unattributed | 0.233 [CoV 65.2%] (0.052-0.295) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 61.682 [CoV 3.6%] (57.945-61.712) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 33.000 [CoV 0.0%] (33.000-33.000) |

### prefill step 0, generations 2 to 5 (warm)

N executions per process: [4, 4, 4] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 216.751 [CoV 0.1%] (216.497-216.807) |
| evaluate | 204.959 [CoV 0.1%] (204.893-205.439) |
| outside_evaluate | 11.603 [CoV 1.7%] (11.296-11.648) |
| apply_serving_config | 0.000 [CoV 8.5%] (0.000-0.000) |
| build_position_inputs | 0.568 [CoV 4.1%] (0.532-0.575) |
| named_blocks_span | 10.904 [CoV 1.5%] (10.659-10.976) |
| nb_weights_loop | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.002 [CoV 6.1%] (0.002-0.002) |
| outside_unattributed | 0.129 [CoV 3.7%] (0.123-0.132) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.128 [CoV 1.3%] (0.127-0.130) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.834 [CoV 3.6%] (0.801-0.861) |
| encode_to_commit | 0.001 [CoV 28.1%] (0.001-0.001) |
| commit_call_sum | 0.012 [CoV 27.1%] (0.009-0.016) |
| chunk1_queue | 2.683 [CoV 2.0%] (2.640-2.748) |
| chunk1_sched_to_gpu | 0.077 [CoV 2.6%] (0.075-0.079) |
| gpu_busy_sum | 201.113 [CoV 0.1%] (200.933-201.505) |
| inter_chunk_idle | -0.000 [CoV 0.0%] (-0.000--0.000) |
| lead_idle_entry_to_gpu | 3.685 [CoV 1.1%] (3.662-3.744) |
| lead_idle_prepare_start_to_gpu | 3.685 [CoV 1.1%] (3.662-3.744) |
| tail | 0.186 [CoV 8.1%] (0.176-0.206) |
| post_wait_to_exit | 0.022 [CoV 2.2%] (0.022-0.023) |
| in_evaluate_unattributed | 0.109 [CoV 18.4%] (0.078-0.110) |
| chunk1_ops | 383.000 [CoV 0.0%] (383.000-383.000) |
| chunks | 1.000 [CoV 0.0%] (1.000-1.000) |
| busy_over_wall_pct | 93.202 [CoV 0.2%] (92.998-93.361) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generation 1

N executions per process: [22, 22, 22] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.699 [CoV 1.8%] (5.577-5.776) |
| evaluate | 5.652 [CoV 1.9%] (5.518-5.723) |
| outside_evaluate | 0.053 [CoV 0.8%] (0.052-0.053) |
| apply_serving_config | 0.000 [CoV 12.5%] (0.000-0.000) |
| build_position_inputs | 0.002 [CoV 4.2%] (0.001-0.002) |
| named_blocks_span | 0.018 [CoV 5.4%] (0.017-0.019) |
| nb_weights_loop | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 14.4%] (0.000-0.000) |
| outside_unattributed | 0.032 [CoV 3.6%] (0.032-0.034) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.076 [CoV 4.6%] (0.075-0.081) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.535 [CoV 1.2%] (0.529-0.542) |
| encode_to_commit | 0.000 [CoV 15.7%] (0.000-0.000) |
| commit_call_sum | 0.019 [CoV 3.0%] (0.019-0.020) |
| chunk1_queue | 0.090 [CoV 2.9%] (0.086-0.091) |
| chunk1_sched_to_gpu | 0.063 [CoV 3.2%] (0.060-0.064) |
| gpu_busy_sum | 5.033 [CoV 2.6%] (4.840-5.082) |
| inter_chunk_idle | 0.163 [CoV 1.7%] (0.162-0.167) |
| lead_idle_entry_to_gpu | 0.256 [CoV 2.7%] (0.253-0.266) |
| lead_idle_prepare_start_to_gpu | 0.256 [CoV 2.7%] (0.253-0.266) |
| tail | 0.138 [CoV 6.2%] (0.126-0.142) |
| post_wait_to_exit | 0.014 [CoV 4.6%] (0.014-0.015) |
| in_evaluate_unattributed | 0.057 [CoV 1.2%] (0.056-0.058) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 88.280 [CoV 0.6%] (87.396-88.465) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |

### decode steps >= 2, generations 2 to 5

N executions per process: [88, 88, 88] (median of the per-process medians, CoV and range over the 3 processes)

| quantity (ms unless named) | median [CoV] (min-max) |
|---|---|
| wall | 5.781 [CoV 0.2%] (5.761-5.785) |
| evaluate | 5.727 [CoV 0.2%] (5.713-5.733) |
| outside_evaluate | 0.060 [CoV 3.6%] (0.056-0.060) |
| apply_serving_config | 0.000 [CoV 50.0%] (0.000-0.000) |
| build_position_inputs | 0.001 [CoV 5.8%] (0.001-0.002) |
| named_blocks_span | 0.019 [CoV 4.1%] (0.018-0.020) |
| nb_weights_loop | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_device_kv_adopt | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_push_named_blocks | 0.000 [CoV 0.0%] (0.000-0.000) |
| nb_greedy_program | 0.000 [CoV 0.0%] (0.000-0.000) |
| append_plus_greedy | 0.000 [CoV 21.8%] (0.000-0.000) |
| outside_unattributed | 0.037 [CoV 6.3%] (0.035-0.039) |
| prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| gap_after_prepare | 0.000 [CoV 0.0%] (0.000-0.000) |
| pre_encode | 0.082 [CoV 3.7%] (0.077-0.082) |
| pipeline_compile | 0.000 [CoV 0.0%] (0.000-0.000) |
| encode_sum | 0.594 [CoV 7.6%] (0.563-0.653) |
| encode_to_commit | 0.000 [CoV 7.9%] (0.000-0.000) |
| commit_call_sum | 0.021 [CoV 5.1%] (0.020-0.023) |
| chunk1_queue | 0.076 [CoV 1.9%] (0.075-0.078) |
| chunk1_sched_to_gpu | 0.057 [CoV 4.1%] (0.057-0.061) |
| gpu_busy_sum | 5.086 [CoV 0.2%] (5.078-5.097) |
| inter_chunk_idle | 0.165 [CoV 0.4%] (0.165-0.166) |
| lead_idle_entry_to_gpu | 0.257 [CoV 4.0%] (0.247-0.268) |
| lead_idle_prepare_start_to_gpu | 0.257 [CoV 4.0%] (0.247-0.268) |
| tail | 0.131 [CoV 0.9%] (0.130-0.133) |
| post_wait_to_exit | 0.015 [CoV 5.0%] (0.015-0.017) |
| in_evaluate_unattributed | 0.063 [CoV 4.4%] (0.059-0.064) |
| chunk1_ops | 24.000 [CoV 0.0%] (24.000-24.000) |
| chunks | 8.000 [CoV 0.0%] (8.000-8.000) |
| busy_over_wall_pct | 88.071 [CoV 0.3%] (87.709-88.147) |
| lead_unattributed | 0.000 [CoV 0.0%] (0.000-0.000) |
| pipeline_misses | 0.000 [CoV 0.0%] (0.000-0.000) |
