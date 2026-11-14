#!/usr/bin/env python3
"""Benchmark the perf cost of "Preserve compound roll constraint semantics"
(commit 8e63db9).

That commit changed ``Constraint.roll_dependent_constraint`` from extracting
only ``BoresightOffsetConstraint`` leaves (fast, but wrong: drops
roll-independent siblings and breaks AND/OR/XOR/AtLeast semantics) to
preserving whole per-field subtrees (correct, but re-evaluates
roll-independent siblings at every one of the 360 angles ``roll_range()``
sweeps).

This script builds the same compound trees both ways and times
``roll_range()`` directly, to quantify the regression rather than guess at it.

Usage (from repo root):
    .venv/bin/python scripts/benchmark_roll_constraint_projection.py
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rust_ephem
from rust_ephem.constraints import AtLeastConstraint, OrConstraint

TLE = str(Path(__file__).parent.parent / "examples" / "example.tle")
N_SAMPLES = 200  # (ra, dec, time) samples per tree shape
N_REPEATS = 3


def _offset(yaw_deg: float) -> rust_ephem.constraints.BoresightOffsetConstraint:
    return rust_ephem.SunConstraint(min_angle=45.0).boresight_offset(yaw_deg=yaw_deg)


def build_ephem() -> rust_ephem.TLEEphemeris:
    return rust_ephem.TLEEphemeris(
        begin=datetime(2025, 12, 1, 0, 0, 0),
        end=datetime(2025, 12, 2, 0, 0, 0),
        step_size=60,
        tle=TLE,
    )


def make_trees(n_independent_siblings: int) -> tuple[object, object]:
    """Return (leaf_only, full_subtree) trees for one AtLeastConstraint field.

    Mirrors a realistic star-tracker-style field: two roll-dependent
    boresight offsets mixed with N roll-independent sun/moon/earth/orbit
    checks, all under a single AtLeastConstraint (as produced by
    StarTrackerConfiguration with >=2 trackers, or a hand-authored
    science/safety constraint).
    """
    boresights = [_offset(45.0), _offset(-45.0)]
    independents = [
        rust_ephem.MoonConstraint(min_angle=10.0),
        rust_ephem.SunConstraint(min_angle=45.0),
        rust_ephem.EarthLimbConstraint(min_angle=20.0),
        rust_ephem.OrbitPoleConstraint(min_angle=15.0),
    ]
    siblings = [
        independents[i % len(independents)] for i in range(n_independent_siblings)
    ]

    full_subtree = AtLeastConstraint(
        min_violated=1, constraints=[*boresights, *siblings]
    )
    # Pre-fix behavior: only the BoresightOffsetConstraint leaves survive,
    # OR-combined, roll-independent siblings dropped entirely.
    leaf_only = (
        OrConstraint(constraints=boresights) if len(boresights) > 1 else boresights[0]
    )
    return leaf_only, full_subtree


def time_roll_range(
    tree: object,
    ephem: rust_ephem.TLEEphemeris,
    ra_dec_time: list[tuple[float, float, "datetime"]],
) -> float:
    start = time.perf_counter()
    for _ in range(N_REPEATS):
        for ra, dec, t in ra_dec_time:
            tree.roll_range(time=t, ephemeris=ephem, target_ra=ra, target_dec=dec)
    return (time.perf_counter() - start) / N_REPEATS


def main() -> None:
    print(f"Building ephemeris and {N_SAMPLES} sample targets...", flush=True)
    ephem = build_ephem()
    rng = np.random.default_rng(42)
    idxs = rng.integers(0, len(ephem.timestamp), size=N_SAMPLES)
    samples = [
        (float(rng.uniform(0, 360)), float(rng.uniform(-90, 90)), ephem.timestamp[i])
        for i in idxs
    ]

    print(
        f"{'#independent siblings':>22} | {'leaf-only (old, s)':>19} | "
        f"{'full-subtree (new, s)':>22} | {'slowdown':>9}"
    )
    for n_siblings in (0, 1, 2, 4, 8):
        leaf_only, full_subtree = make_trees(n_siblings)
        t_leaf = time_roll_range(leaf_only, ephem, samples)
        t_full = time_roll_range(full_subtree, ephem, samples)
        slowdown = t_full / t_leaf if t_leaf > 0 else float("inf")
        print(
            f"{n_siblings:>22} | {t_leaf:>19.4f} | {t_full:>22.4f} | {slowdown:>8.2f}x"
        )


if __name__ == "__main__":
    main()
