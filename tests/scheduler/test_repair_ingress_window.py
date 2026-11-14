"""A tight science start still permits the preceding slew and setup."""

from datetime import timedelta

import pytest

from conops.config import ObservationTiming
from conops.schedulers import CpSatScheduler
from conops.schedulers.priority_scheduler import StartState
from conops.targets import VisitWindow

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target


@pytest.mark.parametrize("visit_window", [False, True])
@pytest.mark.parametrize("slack_seconds", [0, 180])
def test_repairs_tight_science_start_without_delaying_ingress(
    visit_window, slack_seconds
):
    config = make_config(24)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    config.spacecraft_bus.attitude_control.max_slew_rate = 0.01
    config.payload.observation_timing = ObservationTiming(setup_seconds=HOUR)
    long = make_target(
        config, 100, 105, 10, merit=100, minutes=8 * 60, snapshot=8 * 60, ss_min=8 * 60
    )
    timed = make_target(
        config,
        200,
        115,
        10,
        merit=90,
        minutes=60,
        snapshot=60,
        ss_min=60,
        earliest_start=T0 + 5 * HOUR,
        deadline=T0 + 5 * HOUR + slack_seconds,
    )
    if visit_window:
        timed.earliest_start = timed.deadline = None
        timed.visit_windows = [
            VisitWindow(start=T0 + 5 * HOUR, deadline=T0 + 5 * HOUR + slack_seconds)
        ]
    scheduler = CpSatScheduler(
        config,
        [long, timed],
        BEGIN,
        BEGIN + timedelta(hours=24),
        chunk=timedelta(hours=6),
        solver_time_limit=1200,
        solver_deterministic_time=20,
        workers=1,
        seed=2029,
        time_limit=0,
        max_iterations=0,
        include_passes=False,
        start_state=StartState(time=T0, attitude=(105, 10, 0)),
    )
    plan = scheduler.schedule()
    selected = [entry for entry in plan if int(entry.obsid) == 200]
    assert len(selected) == 1
    entry = selected[0]
    assert float(entry.begin) < T0 + 4 * HOUR
    assert T0 + 5 * HOUR <= entry.collection_begin <= T0 + 5 * HOUR + slack_seconds
    assert entry.collection_end - entry.collection_begin == HOUR
    assert entry.visit == (0 if visit_window else None)
    assert scheduler.score == 8 * HOUR * 100 + HOUR * 90
    assert not timed.visits_done
