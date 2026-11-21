"""Retained science should still admit safe work in its unused time."""

from datetime import timedelta

import pytest

from conops import DITL
from conops.config import OnboardRecorder, ScienceData
from conops.schedulers import CpSatScheduler, LocalSearchScheduler

from .planning_scenario import BEGIN, MIN, T0, make_target
from .test_complete_incumbent import scenario


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
@pytest.mark.parametrize("recorder_blocks_visit", [False, True])
def test_retained_incumbent_fills_idle_time_without_moving_selected_science(
    monkeypatch, planner, recorder_blocks_visit
):
    config, target, seed = scenario()
    extra = make_target(
        config,
        200,
        105,
        10,
        minutes=5,
        snapshot=5,
        ss_min=5,
        earliest_start=T0 + 35 * MIN,
        deadline=T0 + 45 * MIN,
    )
    if recorder_blocks_visit:
        config.recorder = OnboardRecorder(capacity_gb=1, planning_limit=0.9)
        target.science_data = {"Default Instrument": ScienceData(rate_gbps=0)}
        extra.science_data = {"Default Instrument": ScienceData(rate_gbps=0.01)}
    original_seed = seed.model_dump_json()
    selected = [
        (e.obsid, e.collection_begin, e.collection_end)
        for e in seed
        if not e.desaturation
    ]
    holds = [
        (e.obsid, e.activity_hold_begin, e.activity_hold_end)
        for e in seed
        if e.desaturation
    ]
    scheduler = planner(
        config,
        [target, extra],
        BEGIN,
        BEGIN + timedelta(hours=1),
        start_plan=seed,
        include_passes=False,
        time_limit=0,
        max_iterations=0,
    )

    def weaker_proposal(build):
        scheduler._start_from_priority_plan()
        return scheduler._finish([])

    # Isolate the actual retained-seed branch from independent contenders.
    monkeypatch.setattr(scheduler, "_plan_within_momentum", weaker_proposal)
    monkeypatch.setattr(scheduler, "_priority_incumbent", lambda: None)
    monkeypatch.setattr(scheduler, "_independent_incumbent", lambda: None)
    plan = scheduler.schedule()

    assert scheduler.incumbent_retained
    assert all(
        any((e.obsid, e.collection_begin, e.collection_end) == visit for e in plan)
        for visit in selected
    )
    assert [
        (e.obsid, e.activity_hold_begin, e.activity_hold_end)
        for e in plan
        if e.desaturation
    ] == holds
    added = [e for e in plan if int(e.obsid) == extra.obsid]
    if recorder_blocks_visit:
        assert added == []
    else:
        assert len(added) == 1
        [entry] = added
        assert (
            entry.collection_seconds_between(float(entry.begin), float(entry.end))
            == 5 * MIN
        )
        assert T0 + 35 * MIN <= entry.collection_begin <= T0 + 45 * MIN
        assert scheduler.incumbent_score is not None
        assert scheduler.score > scheduler.incumbent_score
    assert seed.model_dump_json() == original_seed
    assert scheduler._complete_feasible()
    run = DITL(config, plan=plan, begin=BEGIN, end=BEGIN + timedelta(hours=1))
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
