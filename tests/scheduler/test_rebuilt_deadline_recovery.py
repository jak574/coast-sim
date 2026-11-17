"""A successful recovery must not suppress recovery in a rebuilt plan."""

from datetime import timedelta

import pytest

from conops import DITL
from conops.schedulers import CpSatScheduler, LocalSearchScheduler
from conops.targets import Plan, ScheduledActivity

from .planning_scenario import BEGIN, HOUR, T0
from .test_deadline_repair import scenario


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
@pytest.mark.parametrize("budget", [None, 8 * HOUR])
def test_rebuilt_plan_can_recover_a_visit_after_an_earlier_success(
    monkeypatch, planner, budget
):
    config, targets = scenario(budget=budget)
    scheduler = planner(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=24),
        time_limit=0,
        max_iterations=0,
        include_passes=False,
    )
    if planner is CpSatScheduler:
        # Exercise recovery when the native search retains its priority seed.
        monkeypatch.setattr(scheduler, "_solve", lambda start: start)
    original_targets = [target.model_dump_json() for target in targets]
    firing = ScheduledActivity(
        name="Newly forecast wheel unload",
        begin=T0 + 4 * HOUR,
        duration_seconds=HOUR / 2,
        fixed=True,
        desaturation=True,
        obsid=997000,
    )
    builds = []

    def rebuild_with_firing(build):
        # A momentum pass can replace the science after an earlier firing.
        # Both calls use the real construction/recovery/finishing paths.
        first = build()
        builds.append([e.obsid for e in first])
        assert (200 in builds[0]) is (budget is None)
        scheduler.dumps = [firing]
        return build()

    monkeypatch.setattr(scheduler, "_plan_within_momentum", rebuild_with_firing)
    plan = scheduler.schedule()

    science = [e for e in plan if e.obsid in (100, 200)]
    assert {int(e.obsid) for e in science} == ({100, 200} if budget is None else {100})
    assert scheduler.score == 8 * HOUR * 100 + (HOUR * 90 if budget is None else 0)
    [hold] = [e for e in plan if e.desaturation]
    assert (hold.activity_hold_begin, hold.activity_hold_end) == (
        firing.begin,
        firing.end,
    )
    if budget is None:
        [visit] = [e for e in science if e.obsid == 200]
        assert T0 + 5 * HOUR <= visit.collection_begin <= T0 + 6 * HOUR
    assert scheduler.search_iterations == [0, 0]
    assert scheduler.search_limits == ["disabled", "disabled"]
    assert [target.model_dump_json() for target in targets] == original_targets
    assert scheduler._complete_feasible()
    run = DITL(
        config,
        plan=Plan.model_validate_json(plan.model_dump_json()),
        begin=BEGIN,
        end=BEGIN + timedelta(hours=24),
    )
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
