"""A bounded random search must retain profitable exact window recovery."""

from datetime import timedelta

import pytest

from conops.schedulers import LocalSearchScheduler
from conops.targets import ScheduledActivity, VisitWindow

from .planning_scenario import BEGIN, HOUR, T0
from .test_deadline_repair import scenario


def planner(*, budget=None, iterations=0, activities=(), visit_window=False):
    config, targets = scenario(budget=budget)
    if visit_window:
        target = targets[1]
        target.earliest_start = None
        target.deadline = None
        target.visit_windows = [
            VisitWindow(start=T0 + 5 * HOUR, deadline=T0 + 6 * HOUR)
        ]
    return LocalSearchScheduler(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=24),
        include_passes=False,
        time_limit=5,
        max_iterations=iterations,
        seed=2029,
        activities=activities,
    )


@pytest.mark.parametrize("iterations", [0, 1])
@pytest.mark.parametrize("visit_window", [False, True])
def test_recovers_profitable_visit_without_spending_random_proposals(
    iterations, visit_window
):
    scheduler = planner(iterations=iterations, visit_window=visit_window)
    before = [
        (t.exptime, t.collected_seconds, set(t.visits_done)) for t in scheduler.targets
    ]
    plan = scheduler.schedule()

    assert [int(e.obsid) for e in plan] == [200, 100]
    assert scheduler.score == 8 * HOUR * 100 + HOUR * 90
    assert T0 + 5 * HOUR <= plan.entries[0].collection_begin <= T0 + 6 * HOUR
    assert scheduler.search_iterations == [iterations]
    assert scheduler.search_limits == ["iteration budget" if iterations else "disabled"]
    assert [
        (t.exptime, t.collected_seconds, set(t.visits_done)) for t in scheduler.targets
    ] == before


def test_recovery_keeps_category_budget_and_more_valuable_whole_visit():
    scheduler = planner(budget=8 * HOUR)
    plan = scheduler.schedule()
    assert [int(e.obsid) for e in plan] == [100]
    assert scheduler.score == 8 * HOUR * 100


def test_recovery_preserves_required_hold():
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
    for entry in plan:
        if int(entry.obsid) in (100, 200):
            assert float(entry.end) <= hold.begin or float(entry.begin) >= hold.end


def test_reused_instance_repeats_recovery_without_changing_requests():
    scheduler = planner()
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert [int(e.obsid) for e in first] == [200, 100]
    assert [e.model_dump(mode="json") for e in first] == [
        e.model_dump(mode="json") for e in second
    ]
    assert scheduler.search_iterations == [0]
    assert all(not target.visits_done for target in scheduler.targets)
    assert all(target.collected_seconds == 0 for target in scheduler.targets)
