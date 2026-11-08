#!/usr/bin/env python3
"""Profile a real QueueDITL run to find actual bottlenecks.

Usage (from repo root):
    python scripts/profile_queue_ditl.py

Outputs a sorted cProfile table showing the top cumulative-time hot spots.
"""

import cProfile
import io
import pstats
from datetime import datetime
from pathlib import Path

import numpy as np
from rust_ephem import TLEEphemeris

from conops.config import MissionConfig
from conops.ditl.queue_ditl import QueueDITL

TLE = str(Path(__file__).parent.parent / "examples" / "example.tle")
TOP_N = 40


def build_ditl() -> QueueDITL:
    config = MissionConfig()
    config.constraint.ignore_roll = True

    ephem = TLEEphemeris(
        begin=datetime(2025, 12, 1, 0, 0, 0),
        end=datetime(2025, 12, 2, 0, 0, 0),
        step_size=60,
        tle=TLE,
    )

    ditl = QueueDITL(config=config, ephem=ephem)

    rng = np.random.default_rng(42)
    for i in range(500):
        ditl.queue.add(
            ra=rng.uniform(0, 360),
            dec=rng.uniform(-90, 90),
            exptime=1000,
            obsid=10000 + i,
        )

    return ditl


def main() -> None:
    print("Building DITL...", flush=True)
    ditl = build_ditl()

    print("Profiling calc()...", flush=True)
    pr = cProfile.Profile()
    pr.enable()
    ditl.calc()
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(TOP_N)

    print(buf.getvalue())

    # Also print tottime (self time) top hits — often more diagnostic
    buf2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=buf2).sort_stats(pstats.SortKey.TIME)
    ps2.print_stats(TOP_N)
    print("\n--- Sorted by self time ---")
    print(buf2.getvalue())


if __name__ == "__main__":
    main()
