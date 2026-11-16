"""Expired requests must not consume the local search's proposal budget."""

from datetime import timedelta

import pytest

from conops.schedulers import LocalSearchScheduler
from conops.schedulers.context import SchedulingContext
from conops.targets import VisitWindow

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target


def scheduler(expired, deadline_offset):
    config = make_config(3)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    targets = [
        make_target(config, 100, 105, 10, merit=100, minutes=60),
        make_target(
            config,
            200,
            105,
            10,
            merit=90,
            minutes=60,
            deadline=T0 + deadline_offset,
        ),
        *[
            make_target(
                config,
                300 + i,
                105,
                10,
                merit=1000,
                minutes=60,
                deadline=T0 - 1,
            )
            for i in range(expired)
        ],
    ]
    for target in targets:
        target.roll_deg = 0.0
    return LocalSearchScheduler(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=3),
        include_passes=False,
        time_limit=5,
        max_iterations=1,
        seed=8,
    )


@pytest.mark.parametrize("deadline_offset", [0, HOUR / 2])
def test_expired_requests_do_not_displace_a_feasible_search_proposal(
    monkeypatch, deadline_offset
):
    monkeypatch.setattr(
        SchedulingContext, "initial_attitude", lambda *args: (105.0, 10.0, 0.0)
    )
    ordinary = scheduler(0, deadline_offset)
    reference = ordinary.schedule()
    with_expired = scheduler(50, deadline_offset)
    plan = with_expired.schedule()

    # One insertion suffices: the short-window visit goes before flexible
    # science. Its deadline may precede the incumbent's end, or equal the
    # horizon start, and still admit this ordering.
    assert [int(entry.obsid) for entry in reference] == [200, 100]
    assert ordinary.score == HOUR * (100 + 90)
    assert [int(entry.obsid) for entry in plan] == [200, 100]
    assert with_expired.score == ordinary.score
    assert with_expired.search_iterations == ordinary.search_iterations == [1]
    assert with_expired.search_limits == ordinary.search_limits == ["iteration budget"]
    assert plan.entries[0].collection_begin <= T0 + deadline_offset


def test_live_visit_window_remains_eligible_for_search(monkeypatch):
    monkeypatch.setattr(
        SchedulingContext, "initial_attitude", lambda *args: (105.0, 10.0, 0.0)
    )
    planned = scheduler(0, HOUR / 2)
    target = planned.targets[1]
    target.deadline = None
    target.visit_windows = [VisitWindow(start=T0, deadline=T0 + HOUR / 2)]

    plan = planned.schedule()

    assert [int(entry.obsid) for entry in plan] == [200, 100]
    assert plan.entries[0].visit == 0
    assert planned.score == HOUR * (100 + 90)
