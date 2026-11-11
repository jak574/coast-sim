"""Exact continuation may recover a visit hidden by a native chunk boundary."""

from datetime import timedelta

from conops.config import ObservationCategory, ObservationTiming
from conops.schedulers import CpSatScheduler
from conops.schedulers.local_search_scheduler import _Decoded
from conops.targets import ScheduledActivity, VisitWindow

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target


class WithoutRepair(CpSatScheduler):
    def _repair_window(self, decoded: _Decoded) -> _Decoded:
        return decoded


def scenario(*, budget=None):
    categories = (
        []
        if budget is None
        else [
            ObservationCategory(
                name="science", obsid_min=0, obsid_max=1000, time_budget_seconds=budget
            )
        ]
    )
    config = make_config(24, categories=categories)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    config.payload.observation_timing = ObservationTiming(setup_seconds=HOUR)
    a = make_target(
        config, 100, 105, 10, merit=100, minutes=8 * 60, snapshot=8 * 60, ss_min=8 * 60
    )
    b = make_target(
        config,
        200,
        105,
        10,
        merit=90,
        minutes=60,
        snapshot=60,
        ss_min=60,
        earliest_start=T0 + 5 * HOUR,
        deadline=T0 + 6 * HOUR,
    )
    return config, [a, b]


def planner(kind=CpSatScheduler, *, budget=None, activities=()):
    config, targets = scenario(budget=budget)
    return kind(
        config,
        targets,
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
        activities=activities,
    )


def test_recovers_timed_visit_before_long_snapshot_with_unchanged_native_budget():
    old = planner(WithoutRepair)
    old.schedule()
    new = planner()
    plan = new.schedule()
    assert old.score == 8 * HOUR * 100
    assert new.score == 8 * HOUR * 100 + HOUR * 90
    assert [int(e.obsid) for e in plan] == [200, 100]
    assert new.solver_score == old.score
    assert new.solver_search_statuses == old.solver_search_statuses
    assert new.solver_search_limits == old.solver_search_limits
    assert new.solver_search_statuses and set(new.solver_search_statuses) == {"OPTIMAL"}
    assert new.solver_deterministic_time == 20 and new.chunk == timedelta(hours=6)
    assert new.workers == 1 and new.seed == 2029


def test_keeps_budget_instead_of_trading_away_more_valuable_whole_science():
    scheduler = planner(budget=8 * HOUR)
    plan = scheduler.schedule()
    assert scheduler.score == 8 * HOUR * 100
    assert [int(e.obsid) for e in plan] == [100]
    assert not scheduler._window_repaired


def test_keeps_mandatory_activity_whole_while_reordering_science():
    hold = ScheduledActivity(
        name="engineering", begin=T0 + 10 * HOUR, duration_seconds=HOUR, fixed=True
    )
    scheduler = planner(activities=[hold])
    plan = scheduler.schedule()
    assert scheduler.score == 8 * HOUR * 100 + HOUR * 90
    activities = [e for e in plan if e.activity_hold_begin is not None]
    assert len(activities) == 1
    assert activities[0].activity_hold_begin == hold.begin
    assert activities[0].activity_hold_end == hold.end
    for e in plan:
        if int(e.obsid) in (100, 200):
            assert float(e.end) <= hold.begin or float(e.begin) >= hold.end


def test_reused_scheduler_clears_repair_flag_and_keeps_input_history():
    scheduler = planner()
    before = [
        (t.exptime, t.collected_seconds, set(t.visits_done)) for t in scheduler.targets
    ]
    first = scheduler.schedule()
    assert scheduler._window_repaired
    second = scheduler.schedule()
    assert scheduler._window_repaired
    assert [e.model_dump(mode="json") for e in first] == [
        e.model_dump(mode="json") for e in second
    ]
    assert [
        (t.exptime, t.collected_seconds, set(t.visits_done)) for t in scheduler.targets
    ] == before


def test_recovers_an_unvisited_timed_window_without_mutating_history():
    config, targets = scenario()
    target = targets[1]
    target.earliest_start = None
    target.deadline = None
    target.visit_windows = [VisitWindow(start=T0 + 5 * HOUR, deadline=T0 + 6 * HOUR)]
    before = [w.model_dump(mode="json") for w in target.visit_windows]
    scheduler = CpSatScheduler(
        config,
        targets,
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
    )
    plan = scheduler.schedule()
    selected = [e for e in plan if int(e.obsid) == 200]
    assert len(selected) == 1 and selected[0].visit == 0
    assert (
        selected[0].collection_seconds_between(
            float(selected[0].begin), float(selected[0].end)
        )
        == HOUR
    )
    assert [w.model_dump(mode="json") for w in target.visit_windows] == before
    assert not target.visits_done
    assert scheduler.score == 8 * HOUR * 100 + HOUR * 90


def test_reserved_window_is_not_recovered_a_second_time():
    config, targets = scenario()
    target = targets[1]
    target.earliest_start = None
    target.deadline = None
    target.visit_windows = [
        VisitWindow(start=T0 + 5 * HOUR, deadline=T0 + 6 * HOUR),
        VisitWindow(start=T0 + 15 * HOUR, deadline=T0 + 16 * HOUR),
    ]
    target.exptime = 2 * HOUR
    scheduler = CpSatScheduler(
        config,
        targets,
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
        reserved_seconds={200: HOUR},
        reserved_visits={200: T0 + 6 * HOUR},
        reserved_visit_windows={200: 0},
    )
    plan = scheduler.schedule()
    selected = [e for e in plan if int(e.obsid) == 200]
    assert len(selected) == 1 and selected[0].visit == 1
    assert selected[0].collection_begin >= T0 + 15 * HOUR
    assert (
        selected[0].collection_seconds_between(
            float(selected[0].begin), float(selected[0].end)
        )
        == HOUR
    )
    assert not target.visits_done


def test_expired_deadline_does_not_start_a_repair():
    scheduler = planner()
    scheduler.targets[1].earliest_start = None
    scheduler.targets[1].deadline = T0 - 1
    plan = scheduler.schedule()
    assert [int(e.obsid) for e in plan] == [100]
    assert not scheduler._window_repaired
