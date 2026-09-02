"""Runtime thread/parallel-config verification, shared by inference_bench.py
and train_bench.py. Prints what torch actually has in effect immediately
before the timed loop and fails loudly if a single-thread request did not
take -- a script that REQUESTS one thread and one that GETS one thread must
not emit identical output when they diverge.
"""

from __future__ import annotations

import os

import torch

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def report_and_verify_threads(requested_threads: int) -> None:
    """Prints the actual thread/parallel config as seen by this process right
    now, then raises SystemExit if requested_threads == 1 but torch did not
    actually land on one thread.
    """
    actual_intra = torch.get_num_threads()
    actual_interop = torch.get_num_interop_threads()
    env_seen = {name: os.environ.get(name, "<unset>") for name in THREAD_ENV_VARS}

    print(f"torch.get_num_threads()={actual_intra} torch.get_num_interop_threads()={actual_interop}")
    print(f"env: {env_seen}")
    print(torch.__config__.parallel_info())

    if requested_threads == 1 and actual_intra != 1:
        raise SystemExit(f"requested single-thread but torch.get_num_threads()={actual_intra} != 1 -- refusing to report a number under an unverified config")


def report_load_average(label: str) -> None:
    load_1min, load_5min, load_15min = os.getloadavg()
    print(f"load average ({label}): {load_1min:.2f} {load_5min:.2f} {load_15min:.2f}")
