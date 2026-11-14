"""Flexible science must leave room for a window that closes in this chunk."""

from datetime import timedelta

import pytest

pytest.importorskip("ortools")

from conops.schedulers import CpSatScheduler  # noqa: E402
from conops.schedulers.context import SchedulingContext  # noqa: E402
from conops.targets import PlanEntry  # noqa: E402

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target  # noqa: E402


@pytest.mark.parametrize("chunk_hours", [2, 6])
@pytest.mark.parametrize("with_later_visit", [False, True])
def test_flexible_visit_can_move_into_the_gap_after_a_chunk(
    monkeypatch, chunk_hours, with_later_visit
):
    hours = 2 * chunk_hours + (2 if with_later_visit else 0)
    config = make_config(hours)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    windows = {
        100: [[T0, T0 + hours * HOUR]],
        101: [[T0, T0 + chunk_hours * HOUR]],
        102: [[T0 + 2 * chunk_hours * HOUR, T0 + hours * HOUR]],
    }

    def visibility(self, begin=None, end=None):
        self.windows = [list(window) for window in windows[int(self.obsid)]]
        return len(self.windows)

    # Supply the same physical visibility to request filtering and exact
    # entry placement. No request has an artificial deadline or visit window.
    monkeypatch.setattr(PlanEntry, "visibility", visibility)
    monkeypatch.setattr(
        SchedulingContext,
        "visibility_windows",
        lambda self, target: [tuple(w) for w in windows[int(target.obsid)]],
    )
    monkeypatch.setattr(
        SchedulingContext, "initial_attitude", lambda *args: (105.0, 10.0, 0.0)
    )
    targets = [
        make_target(
            config,
            100,
            105,
            10,
            merit=11,
            minutes=chunk_hours * 60,
            ss_min=chunk_hours * 60,
        ),
        make_target(
            config,
            101,
            105,
            10,
            merit=10,
            minutes=chunk_hours * 60,
            ss_min=chunk_hours * 60,
        ),
    ]
    if with_later_visit:
        targets.append(
            make_target(config, 102, 105, 10, merit=20, minutes=120, ss_min=120)
        )
    for target in targets:
        target.roll_deg = 0.0
    scheduler = CpSatScheduler(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=hours),
        chunk=timedelta(hours=chunk_hours),
        include_passes=False,
        workers=1,
        seed=1,
        solver_deterministic_time=1.0,
        solver_time_limit=5.0,
        max_iterations=0,
    )

    plan = scheduler.schedule()

    # Priority order runs the flexible observation first and loses 101.
    # Both fit when 101 runs first, with the later visit kept unchanged.
    expected = [101, 100, 102] if with_later_visit else [101, 100]
    assert [int(entry.obsid) for entry in plan.entries] == expected
    assert scheduler.score > scheduler.initial_score
    assert sum(
        entry.collection_seconds_between(entry.begin, entry.end)
        for entry in plan.entries
    ) == pytest.approx(hours * HOUR)
    for entry in plan.entries:
        begin, end = windows[int(entry.obsid)][0]
        assert begin <= entry.collection_begin < entry.collection_end <= end
    if with_later_visit:
        assert plan.entries[-1].collection_begin == T0 + 2 * chunk_hours * HOUR
    assert "wall clock" not in scheduler.solver_search_limits
