#!/usr/bin/env python
"""Accept planned and rolling runs of a scenario against a fresh dispatch run.

The scenario is one of COASTSim's standard benchmark scenarios (see
conops.benchmark.scenarios), so the check does not depend on any mission's
program. Every case first dispatches the scenario afresh and seeds the
planner with the plan that run executed. Four cases are run: one plan built
up front and a rolling-horizon run, each with the local-search and the CP-SAT
scheduler. Each case is checked for:

* value: the executed dynamic science value against the fresh dispatch run.
  CP-SAT must exceed it and local search must equal or exceed it;
* execution: no difference between the plan and what was flown;
* activities: every scheduled activity held for its whole length, starting
  inside its window, with no science collected during the hold;
* planning time: every plan, and every replan of a rolling run, built within
  the gate;
* repeatability: the same executed plan and value from separate processes.

A rolling run plans ``--horizon-hours`` ahead at each replan, 24 by default.
A horizon shorter than the windows the requests must be observed in leaves a
replan unable to tell two plans apart that differ later: on the ``program``
scenario, whose visit windows are ten hours long, rolling local search is
below dispatch with a 12 hour horizon and above it with 24.

Example:
    .venv/bin/python scripts/accept_planned_runs.py --scenario program --hours 24
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from conops import DITL  # noqa: E402
from conops.benchmark import BenchmarkScenario  # noqa: E402
from conops.benchmark.contenders import _dispatch, too_targets  # noqa: E402
from conops.benchmark.metrics import dynamic_science_value  # noqa: E402
from conops.benchmark.scenarios import SCENARIOS  # noqa: E402
from conops.common import ObsType  # noqa: E402
from conops.ditl import RollingHorizonDITL  # noqa: E402
from conops.ditl.plan_validator import entry_obstype  # noqa: E402
from conops.schedulers import LocalSearchScheduler  # noqa: E402

TLE = REPO_ROOT / "examples" / "example.tle"
CASES = [
    "planned:local_search",
    "planned:cp_sat",
    "rolling:local_search",
    "rolling:cp_sat",
]
STRICT = {"planned:cp_sat", "rolling:cp_sat"}
"""Cases whose value must exceed dispatch; the others must equal or exceed it."""


def _scheduler(kind: str, args: argparse.Namespace) -> tuple[type, dict[str, object]]:
    """The scheduler class for ``kind`` and repeatable search settings."""
    if kind == "local_search":
        return LocalSearchScheduler, {
            "time_limit": args.search_seconds,
            "max_iterations": args.iterations,
            "seed": args.seed,
        }
    from conops.schedulers import CpSatScheduler

    return CpSatScheduler, {
        "time_limit": args.search_seconds,
        "max_iterations": args.iterations,
        "solver_time_limit": args.solver_seconds,
        "solver_deterministic_time": args.deterministic_time,
        "workers": 1,
        "seed": args.seed,
    }


def _value(simulation) -> float:  # type: ignore[no-untyped-def]
    return float(
        dynamic_science_value(
            simulation.telemetry.housekeeping, simulation.plan, simulation.step_size
        )
    )


def _digest(simulation) -> str:  # type: ignore[no-untyped-def]
    """A digest of the plan the run executed."""
    rows = [
        (
            int(e.obsid),
            str(entry_obstype(e)),
            round(float(e.begin), 3),
            round(float(e.end), 3),
            round(float(e.exposure or 0), 3),
        )
        for e in sorted(simulation.plan, key=lambda e: (float(e.begin), int(e.obsid)))
    ]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def _activity_check(scenario: BenchmarkScenario, simulation) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """How many of the scenario's activities were held whole and inside
    their windows, and how many science collections fell inside a hold."""
    end = scenario.end.timestamp()
    due = [a for a in scenario.activities if a.latest_start + a.duration_seconds <= end]
    holds: list[tuple[str, float, float]] = []
    for entry in simulation.plan:
        if entry_obstype(entry) != ObsType.ACTIVITY:
            continue
        if entry.activity_hold_begin is None or entry.activity_hold_end is None:
            continue
        hold = (
            str(entry.name),
            float(entry.activity_hold_begin),
            float(entry.activity_hold_end),
        )
        if hold not in holds:  # a rolling run may list one hold in several entries
            holds.append(hold)
    whole = 0
    for activity in due:
        # Activities may share a name: match on the hold starting in the window.
        whole += int(
            any(
                name == activity.name
                and activity.begin - 1e-6 <= begin <= activity.latest_start + 1e-6
                and abs((hold_end - begin) - activity.duration_seconds) <= 1e-6
                for name, begin, hold_end in holds
            )
        )
    inside = 0
    for record in simulation.telemetry.housekeeping:
        if not (record.collection_seconds or 0) > 0 or record.calibration:
            continue
        at = record.timestamp.timestamp()
        inside += int(any(begin <= at < hold_end for _, begin, hold_end in holds))
    return {"due": len(due), "whole": whole, "science_in_hold": inside}


def run_case(case: str, args: argparse.Namespace) -> dict[str, object]:
    """Dispatch the scenario afresh, then run ``case`` seeded by that run."""
    mode, kind = case.split(":")
    scenario = SCENARIOS[args.scenario](TLE, hours=args.hours, seed=args.scenario_seed)
    dispatched = _dispatch(scenario)
    dispatch_value = _value(dispatched)
    scheduler, options = _scheduler(kind, args)
    config = scenario.make_config()
    targets = scenario.make_targets(config)
    began = time.perf_counter()
    if mode == "planned":
        targets = [*targets, *too_targets(config, scenario.toos)]
        builder = scheduler(
            config,
            targets,
            scenario.begin,
            scenario.end,
            activities=scenario.activities,
            start_plan=dispatched.plan,
            **options,
        )
        plan = builder.schedule()
        plan_seconds = [time.perf_counter() - began]
        simulation = DITL(
            config=config, plan=plan, begin=scenario.begin, end=scenario.end
        )
        simulation.step_size = builder.ctx.step_size
    else:
        hours = (scenario.end - scenario.begin).total_seconds() / 3600
        simulation = RollingHorizonDITL(
            config,
            targets,
            begin=scenario.begin,
            end=scenario.end,
            horizon=timedelta(hours=min(hours, args.horizon_hours)),
            replan_interval=timedelta(hours=min(hours, args.replan_hours)),
            scheduler=scheduler,
            scheduler_options=options,
            activities=scenario.activities,
            start_plan=dispatched.plan,
        )
        simulation.step_size = int(simulation.ephem.step_size)
        for too in scenario.toos:
            simulation.submit_too(**too.model_dump())
    if not simulation.calc():
        raise RuntimeError(f"{case}: the simulation did not complete")
    if mode == "rolling":
        plan_seconds = [float(r.planning_seconds) for r in simulation.replans]
    return {
        "case": case,
        "dispatch_value": dispatch_value,
        "value": _value(simulation),
        "mismatches": len(simulation.validate_plan_matches_execution()),
        "activities": _activity_check(scenario, simulation),
        "plans": len(plan_seconds),
        "longest_plan_seconds": max(plan_seconds),
        "total_plan_seconds": sum(plan_seconds),
        "digest": _digest(simulation),
    }


def _child(case: str, argv: list[str]) -> dict[str, object]:
    """Run one case in a process of its own and return its result."""
    done = subprocess.run(
        [sys.executable, __file__, *argv, "--child", case],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    for line in reversed(done.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])  # type: ignore[no-any-return]
    return {
        "case": case,
        "error": (done.stderr.strip().splitlines() or ["no output"])[-1],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="program")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--scenario-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=2029, help="search seed")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--search-seconds", type=float, default=600.0)
    parser.add_argument("--solver-seconds", type=float, default=600.0)
    parser.add_argument("--deterministic-time", type=float, default=20.0)
    parser.add_argument("--gate-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--horizon-hours",
        type=float,
        default=24.0,
        help="how far ahead each replan of a rolling run plans",
    )
    parser.add_argument(
        "--replan-hours",
        type=float,
        default=6.0,
        help="time between scheduled replans of a rolling run",
    )
    parser.add_argument("--repeats", type=int, default=3, help="processes per case")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=None)
    parser.add_argument("--child", choices=CASES, default=None, help=argparse.SUPPRESS)
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)

    if args.child is not None:
        print("RESULT " + json.dumps(run_case(args.child, args)))
        return 0

    cases = args.cases or CASES
    if importlib.util.find_spec("ortools") is None:
        cases = [c for c in cases if not c.endswith("cp_sat")]
        print("OR-Tools is not installed: the CP-SAT cases are left out")
    passed = True
    for case in cases:
        runs = [_child(case, raw) for _ in range(args.repeats)]
        errors = [r["error"] for r in runs if "error" in r]
        if errors:
            print(f"{case}: FAILED TO RUN: {errors[0]}")
            passed = False
            continue
        first = runs[0]
        gain = 100.0 * (first["value"] / first["dispatch_value"] - 1.0)  # type: ignore[operator]
        value_ok = (
            first["value"] > first["dispatch_value"]  # type: ignore[operator]
            if case in STRICT
            else first["value"] >= first["dispatch_value"]  # type: ignore[operator]
        )
        acts = first["activities"]
        checks = {
            "value": value_ok,
            "execution": first["mismatches"] == 0,
            "activities": acts["whole"] == acts["due"] and acts["science_in_hold"] == 0,  # type: ignore[index]
            "planning time": max(r["longest_plan_seconds"] for r in runs)  # type: ignore[type-var]
            <= args.gate_seconds,
            "repeatability": len({(r["digest"], r["value"]) for r in runs}) == 1,
        }
        passed = passed and all(checks.values())
        print(
            f"{case}: value {first['value']:.0f} against dispatch "
            f"{first['dispatch_value']:.0f} ({gain:+.2f}%); "
            f"mismatches {first['mismatches']}; "
            f"activities whole {acts['whole']}/{acts['due']}, "  # type: ignore[index]
            f"science in a hold {acts['science_in_hold']}; "  # type: ignore[index]
            f"{first['plans']} plan(s), longest "
            f"{max(r['longest_plan_seconds'] for r in runs):.1f} s; "  # type: ignore[type-var]
            f"{len({r['digest'] for r in runs})} distinct plan(s) in "
            f"{args.repeats} processes"
        )
        for name, ok in checks.items():
            print(f"    {'pass' if ok else 'FAIL'}  {name}")
    print("ACCEPTED" if passed else "NOT ACCEPTED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
