"""Window insertion must explore early ingress, not only the deadline's neighbours."""

import random
from datetime import timedelta

import pytest

from conops.config import ObservationTiming
from conops.schedulers import LocalSearchScheduler
from conops.schedulers.local_search_scheduler import _Snapshot
from conops.schedulers.priority_scheduler import StartState
from conops.targets import VisitWindow

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target


@pytest.mark.parametrize("visit_window", [False, True])
def test_window_insertion_can_start_a_long_ingress_before_several_snapshots(
    visit_window,
):
    config = make_config(36)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    acs = config.spacecraft_bus.attitude_control
    acs.settle_time = 0
    acs.max_slew_rate = 0.0005
    config.payload.observation_timing = ObservationTiming(setup_seconds=HOUR)
    ordinary = [
        make_target(config, 100 + i, 105, 10, minutes=60, snapshot=60, ss_min=60)
        for i in range(8)
    ]
    timed = make_target(
        config,
        200,
        115,
        10,
        minutes=60,
        snapshot=60,
        ss_min=60,
        earliest_start=T0 + 6 * HOUR,
        deadline=T0 + 8 * HOUR,
    )
    for target in [*ordinary, timed]:
        target.roll_deg = 0
    if visit_window:
        timed.earliest_start = timed.deadline = None
        timed.visit_windows = [VisitWindow(start=T0 + 6 * HOUR, deadline=T0 + 8 * HOUR)]
    scheduler = LocalSearchScheduler(
        config,
        [*ordinary, timed],
        BEGIN,
        BEGIN + timedelta(hours=36),
        include_passes=False,
        start_state=StartState(time=T0, attitude=(105, 10, 0)),
        time_limit=0,
        max_iterations=0,
    )
    scheduler._momentum_seed = None
    scheduler._owned_locked = None
    scheduler._start_from_priority_plan()
    current = scheduler._decode(
        [_Snapshot(obsid=t.obsid, seconds=HOUR) for t in ordinary]
    )
    assert len(current.final.timeline) == len(ordinary)
    baseline = scheduler._objective(current.final)
    earlier = scheduler._decode([_Snapshot(obsid=200, seconds=HOUR), *current.sequence])
    assert any(b.entry.obsid == 200 for b in earlier.final.timeline)
    recovered = []
    for seed in range(16):
        move = scheduler._window_insertion(current, random.Random(seed))
        assert move is not None
        decoded = scheduler._redecode(current, *move)
        visits = [b.entry for b in decoded.final.timeline if b.entry.obsid == 200]
        if visits:
            assert T0 + 6 * HOUR <= visits[0].collection_begin <= T0 + 8 * HOUR
            assert visits[0].collection_end - visits[0].collection_begin == HOUR
            assert len(decoded.final.timeline) == len(ordinary) + 1
            recovered.append(scheduler._objective(decoded.final))
    assert recovered, "No insertion trial reaches the legal earlier ingress"
    assert max(recovered) > baseline
