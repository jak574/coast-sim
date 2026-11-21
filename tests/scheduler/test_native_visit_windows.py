"""Native independent-window feasibility and selected-visit handoff."""

from datetime import timedelta

import numpy as np
import pytest

from conops import DITL
from conops.config import TargetConfig
from conops.schedulers import CpSatScheduler
from conops.schedulers.cpsat_scheduler import _ChunkModel, _Task
from conops.targets import Plan, Pointing, ScheduledActivity, VisitWindow

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target

pytest.importorskip("ortools")


@pytest.mark.parametrize(
    "first_visit,later_visit,key,later_expected",
    [
        (0, 1, (100, 0), True),
        (None, None, (100, 0), False),
        (0, 0, (100, 0), False),
        (None, None, (100, 1), True),
    ],
)
def test_later_visit_can_survive_an_unreachable_earlier_window(
    first_visit, later_visit, key, later_expected
):
    pose = (0.0, 0.0, 0.0)
    tasks = [
        _Task(None, (pose, pose), (0, 0)),
        _Task(None, (pose, pose), (0, 0), duration=600, inherits_attitude=True),
        _Task(
            (100, 0),
            (pose, pose),
            (0, 0),
            collection=(600, 600),
            window=(0, 2400),
            visit=first_visit,
            rate=10.0,
        ),
        _Task(
            key,
            (pose, pose),
            (1200, 1200),
            copy=1,
            collection=(600, 600),
            window=(0, 2400),
            visit=later_visit,
            rate=10.0,
        ),
    ]
    chunk = _ChunkModel(tasks, np.zeros((4, 4), dtype=np.int64), 60, 2400, {100: 1200})
    status, ending, chosen = chunk.solve(10, 1, 2029, 0, deterministic_time=20)
    assert status == "OPTIMAL" and ending == "optimal"
    assert chosen is not None
    assert bool(chosen) is later_expected
    assert sum(item.seconds for item in chosen) == (600 if later_expected else 0)


def test_native_solution_retains_the_selected_visit_window():
    pose = (0.0, 0.0, 0.0)
    tasks = [
        _Task(None, (pose, pose), (0, 0)),
        _Task(
            (100, 0),
            (pose, pose),
            (0, 0),
            collection=(600, 600),
            window=(0, 1200),
            visit=7,
            rate=10.0,
        ),
    ]
    chunk = _ChunkModel(tasks, np.zeros((2, 2), dtype=np.int64), 60, 1200, {100: 600})
    status, ending, chosen = chunk.solve(10, 1, 2029, 0, deterministic_time=20)
    assert status == "OPTIMAL" and ending == "optimal"
    assert chosen is not None
    assert len(chosen) == 1
    assert (chosen[0].obsid, chosen[0].visit, chosen[0].seconds) == (100, 7, 600)


def test_native_search_recovers_a_later_visit_after_a_blocked_window(monkeypatch):
    config = make_config(
        6, weights=TargetConfig(urgency_weight=0, completion_deficit_weight=0)
    )
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    config.battery.planning_level = None
    flexible = make_target(
        config, 200, 105, 10, merit=100, minutes=180, snapshot=60, ss_min=60
    )
    timed = Pointing(
        config=config,
        obsid=100,
        name="Independent visits",
        ra=105,
        dec=10,
        merit=90,
        ss_min=600,
        ss_max=600,
        exptime=1200,
        visit_windows=[
            VisitWindow(start=T0 + 600, deadline=T0 + 1200),
            VisitWindow(start=T0 + 2 * HOUR, deadline=T0 + 3 * HOUR),
        ],
    )
    inputs = [target.model_dump_json() for target in (flexible, timed)]
    required = ScheduledActivity(
        name="Required hold",
        begin=T0,
        duration_seconds=HOUR,
        fixed=True,
        obsid=997000,
    )
    scheduler = CpSatScheduler(
        config,
        [flexible, timed],
        BEGIN,
        BEGIN + timedelta(hours=6),
        activities=[required],
        include_passes=False,
        time_limit=0,
        max_iterations=0,
        workers=1,
        seed=2029,
        solver_time_limit=1200,
        solver_deterministic_time=20,
        chunk=timedelta(hours=6),
    )
    # The native model must find this gain without the exact-repair fallback.
    monkeypatch.setattr(scheduler, "_repair_window", lambda decoded: decoded)
    plan = scheduler.schedule()
    assert scheduler.initial_score == 3 * HOUR * 100
    assert scheduler.score == 3 * HOUR * 100 + 600 * 90
    assert any(scheduler.solver_chunks_used)
    [visit] = [entry for entry in plan if entry.obsid == 100]
    assert visit.visit == 1
    assert T0 + 2 * HOUR <= visit.collection_begin <= T0 + 3 * HOUR
    [hold] = [entry for entry in plan if entry.obsid == required.obsid]
    assert (hold.activity_hold_begin, hold.activity_hold_end) == (T0, T0 + HOUR)
    assert scheduler._complete_feasible()
    assert inputs == [target.model_dump_json() for target in (flexible, timed)]
    run = DITL(
        config,
        plan=Plan.model_validate_json(plan.model_dump_json()),
        begin=BEGIN,
        end=BEGIN + timedelta(hours=6),
    )
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
