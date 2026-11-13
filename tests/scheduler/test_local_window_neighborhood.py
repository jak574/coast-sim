"""A search trial can target a closing window without increasing its budget."""

from datetime import timedelta

import pytest

from conops.config import ObservationTiming
from conops.schedulers import LocalSearchScheduler
from conops.schedulers.priority_scheduler import StartState
from conops.targets import VisitWindow

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target


@pytest.mark.parametrize("visit_window", [False, True])
def test_one_insertion_trial_uses_the_closing_window(visit_window):
    config = make_config(24)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    config.spacecraft_bus.attitude_control.max_slew_rate = 0.01
    config.payload.observation_timing = ObservationTiming(setup_seconds=HOUR)
    long = make_target(
        config, 100, 105, 10, merit=100, minutes=480, snapshot=480, ss_min=480
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
        deadline=T0 + 5 * HOUR,
    )
    if visit_window:
        timed.earliest_start = timed.deadline = None
        timed.visit_windows = [VisitWindow(start=T0 + 5 * HOUR, deadline=T0 + 5 * HOUR)]
    scheduler = LocalSearchScheduler(
        config,
        [long, timed],
        BEGIN,
        BEGIN + timedelta(hours=24),
        seed=14,
        time_limit=60,
        max_iterations=1,
        include_passes=False,
        start_state=StartState(time=T0, attitude=(105, 10, 0)),
    )
    plan = scheduler.schedule()
    selected = [e for e in plan if e.obsid == 200]
    assert len(selected) == 1
    assert selected[0].collection_begin == T0 + 5 * HOUR
    assert selected[0].collection_end - selected[0].collection_begin == HOUR
    assert scheduler.score == 8 * HOUR * 100 + HOUR * 90
    assert scheduler.search_iterations == [1]
    assert scheduler.search_limits == ["iteration budget"]
