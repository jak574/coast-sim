"""An ingress start cannot substitute for the requested science-window index."""

from datetime import timedelta

from conops.config import ObservationTiming
from conops.schedulers import CpSatScheduler
from conops.schedulers.local_search_scheduler import _Snapshot
from conops.schedulers.priority_scheduler import StartState
from conops.targets import VisitWindow

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target


def scheduler_for_windows(*, reserve_first=False):
    config = make_config(12)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    config.spacecraft_bus.attitude_control.max_slew_rate = 0.01
    config.payload.observation_timing = ObservationTiming(setup_seconds=HOUR)
    target = make_target(config, 200, 115, 10, minutes=60, snapshot=60, ss_min=60)
    target.visit_windows = [
        VisitWindow(start=T0 + 4 * HOUR, deadline=T0 + 4 * HOUR),
        VisitWindow(start=T0 + 5 * HOUR, deadline=T0 + 5 * HOUR),
    ]
    if reserve_first:
        target.exptime = 2 * HOUR
    scheduler = CpSatScheduler(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=12),
        time_limit=0,
        max_iterations=0,
        include_passes=False,
        start_state=StartState(time=T0, attitude=(105, 10, 0)),
        reserved_seconds={200: HOUR} if reserve_first else None,
        reserved_visits={200: T0 + 5 * HOUR} if reserve_first else None,
        reserved_visit_windows={200: 0} if reserve_first else None,
    )
    scheduler._momentum_seed = None
    scheduler._owned_locked = None
    scheduler._start_from_priority_plan()
    return scheduler, target


def test_selected_window_survives_ingress_before_the_previous_deadline():
    scheduler, target = scheduler_for_windows()
    decoded = scheduler._decode([_Snapshot(obsid=200, seconds=HOUR, visit=1)])
    assert len(decoded.final.timeline) == 1
    entry = decoded.final.timeline[0].entry
    assert float(entry.begin) < target.visit_windows[0].deadline
    assert entry.visit == 1
    assert entry.collection_begin == target.visit_windows[1].start
    normalized = scheduler._normalized_placements(decoded)
    assert len(normalized.final.timeline) == 1
    assert normalized.final.timeline[0].entry.visit == 1
    assert normalized.final.timeline[0].entry.collection_begin == entry.collection_begin
    assert not target.visits_done


def test_explicit_window_cannot_bypass_a_committed_visit_reservation():
    scheduler, target = scheduler_for_windows(reserve_first=True)
    decoded = scheduler._decode([_Snapshot(obsid=200, seconds=HOUR, visit=0)])
    assert decoded.final.timeline == []
    assert not target.visits_done
