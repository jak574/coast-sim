"""Compare Python and Rust constraint routing, plus complete slew calculation.

Run from the repository root with an existing planetary ephemeris cache::

    .venv/bin/python scripts/benchmark_slew_routing.py --out /tmp/slew-routing.json

The baseline Slew class is loaded from the revision before the Rust migration;
both implementations use the same interpreter, rust-ephem, constraints and orbit.
Ephemeris construction, object construction and warmup are excluded from timings.
Different waypoint outcomes are reported: the new planner can find detours where
the old planner fell back to an unsafe direct arc. These are latency comparisons,
not assertions that both implementations produce equivalent trajectories.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import rust_ephem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import conops.simulation.slew as slew_module  # noqa: E402
from conops.common import dtutcfromtimestamp  # noqa: E402
from conops.common.enums import SlewAlgorithm  # noqa: E402
from conops.common.vector import constraint_avoiding_waypoint  # noqa: E402
from conops.config import AttitudeControlSystem, Constraint  # noqa: E402
from conops.targets import PlanEntry  # noqa: E402


def legacy_slew_class(revision):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:conops/simulation/slew.py"],
        cwd=ROOT,
        text=True,
    )
    namespace = dict(vars(slew_module))
    exec(compile(source, f"{revision}:conops/simulation/slew.py", "exec"), namespace)
    namespace["Slew"].model_rebuild(_types_namespace={"PlanEntry": PlanEntry})
    return namespace["Slew"]


def radec(vector):
    return (
        float(np.degrees(np.arctan2(vector[1], vector[0])) % 360),
        float(np.degrees(np.arcsin(np.clip(vector[2], -1, 1)))),
    )


def time_pair(old, new, iterations, repeats):
    for _ in range(10):
        old()
        new()
    samples = {"python": [], "rust": []}
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(repeats):
            pair = [("python", old), ("rust", new)]
            if repeat % 2:
                pair.reverse()
            for name, call in pair:
                start = perf_counter_ns()
                for _ in range(iterations):
                    call()
                samples[name].append((perf_counter_ns() - start) / iterations / 1000)
    finally:
        if gc_enabled:
            gc.enable()
    return {name: statistics.median(values) for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--baseline", default="e33e6705^")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.days) < 1:
        parser.error("iterations, repeats and days must be positive")
    old_slew_class = legacy_slew_class(args.baseline)
    begin = datetime(2025, 12, 1, tzinfo=timezone.utc)
    ephem = rust_ephem.TLEEphemeris(
        begin=begin,
        end=begin + timedelta(days=args.days),
        step_size=60,
        tle=str(ROOT / "examples" / "example.tle"),
    )
    index = len(ephem.timestamp) // 2
    utime = ephem.timestamp[index].timestamp()
    sun = ephem.sun_pv.position[index] - ephem.gcrs_pv.position[index]
    sun /= np.linalg.norm(sun)
    tangent = np.cross(sun, [0.0, 0.0, 1.0])
    tangent /= np.linalg.norm(tangent)
    start = radec(0.5 * sun + np.sqrt(0.75) * tangent)
    end = radec(0.5 * sun - np.sqrt(0.75) * tangent)
    clear_start = radec(-0.5 * sun + np.sqrt(0.75) * tangent)
    clear_end = radec(-0.5 * sun - np.sqrt(0.75) * tangent)
    sun_constraint = rust_ephem.SunConstraint(min_angle=45)
    compound = (
        sun_constraint
        | rust_ephem.MoonConstraint(min_angle=10)
        | rust_ephem.EarthLimbConstraint(min_angle=10)
    )
    cases = [
        ("Sun: clear", sun_constraint, clear_start, clear_end),
        ("Sun: crossing", sun_constraint, start, end),
        ("Sun: blocked start", sun_constraint, radec(sun), end),
        ("Sun/Moon/Earth: crossing", compound, start, end),
    ]
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rust_ephem": importlib.metadata.version("rust-ephem"),
        "conops_source": slew_module.__file__,
        "baseline": subprocess.check_output(
            ["git", "rev-parse", args.baseline], cwd=ROOT, text=True
        ).strip(),
        "current": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "ephemeris_samples": len(ephem.timestamp),
        "iterations_per_batch": args.iterations,
        "batches": args.repeats,
        "summary": "median microseconds per call; alternating order; GC disabled",
    }
    print(json.dumps(metadata, indent=2), flush=True)
    print(
        f"{'Case / measurement':35} {'Python us':>11} {'Rust us':>11} {'Speedup':>9}",
        flush=True,
    )
    results = []
    for name, constraint, a, b in cases:

        def check(ra, dec, time):
            rounded_time = round(time / ephem.step_size) * ephem.step_size
            return bool(
                constraint.in_constraint(
                    ephemeris=ephem,
                    target_ra=ra,
                    target_dec=dec,
                    time=dtutcfromtimestamp(rounded_time),
                    target_roll=None,
                )
            )

        def python_route():
            return constraint_avoiding_waypoint(*a, *b, utime, check)

        def rust_route():
            return constraint.route_around(
                ephemeris=ephem,
                start_ra=a[0],
                start_dec=a[1],
                end_ra=b[0],
                end_dec=b[1],
                time=utime,
                target_roll=None,
            )

        acs = AttitudeControlSystem(
            slew_algorithm=SlewAlgorithm.CONSTRAINT_AVOIDING,
            slew_constraint=constraint,
        )
        kwargs = dict(
            ephem=ephem,
            constraint=Constraint(ephem=ephem),
            acs_config=acs,
            startra=a[0],
            startdec=a[1],
            endra=b[0],
            enddec=b[1],
            slewstart=utime,
        )
        old_slew = old_slew_class(**kwargs)
        new_slew = slew_module.Slew(**kwargs)
        old_slew.calc_slewtime()
        new_slew.calc_slewtime()
        outcome = {
            "python_waypoint": python_route(),
            "rust_waypoint": rust_route().waypoint,
            "python_segments": len(old_slew._slew_segments),
            "rust_segments": len(new_slew._slew_segments),
        }
        assert outcome["python_segments"] == (
            2 if outcome["python_waypoint"] is not None else 1
        )
        assert outcome["rust_segments"] == (
            2 if outcome["rust_waypoint"] is not None else 1
        )
        for measurement, old, new in [
            ("routing", python_route, rust_route),
            ("calc_slewtime", old_slew.calc_slewtime, new_slew.calc_slewtime),
        ]:
            medians = time_pair(old, new, args.iterations, args.repeats)
            speedup = medians["python"] / medians["rust"]
            print(
                f"{name + ' / ' + measurement:35} {medians['python']:11.2f} "
                f"{medians['rust']:11.2f} {speedup:8.2f}x",
                flush=True,
            )
            results.append(
                dict(
                    case=name,
                    measurement=measurement,
                    microseconds=medians,
                    speedup=speedup,
                    **outcome,
                )
            )
        print(
            f"  Executed segments: Python={outcome['python_segments']}, "
            f"Rust={outcome['rust_segments']}",
            flush=True,
        )
    if args.out:
        args.out.write_text(
            json.dumps(dict(metadata=metadata, results=results), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
