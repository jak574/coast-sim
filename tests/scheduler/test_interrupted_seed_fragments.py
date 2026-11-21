"""Interrupted dispatch science must remain available to a seeded planner."""

from datetime import timedelta

import pytest

from conops import DITL
from conops.benchmark.contenders import _dispatch, too_targets
from conops.benchmark.metrics import dynamic_science_value
from conops.benchmark.scenarios import baseline
from conops.config import ObservationCategory, OnboardRecorder, ScienceData
from conops.schedulers import CpSatScheduler, LocalSearchScheduler
from conops.schedulers.local_search_scheduler import _Snapshot

from .planning_scenario import BEGIN, T0, make_config, make_target


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
def test_interrupted_dispatch_fragment_keeps_the_complete_seed_floor(
    monkeypatch, planner
):
    scenario = baseline("examples/example.tle", hours=6)
    dispatch = _dispatch(scenario)
    seed = dispatch.plan
    fragments = [
        e for e in seed if 0 < e.exposure < e.ss_min and float(e.end) in e.interruptions
    ]
    assert fragments
    original = seed.model_dump_json()
    interruptions = [e.interruptions for e in seed]
    expected = dynamic_science_value(
        dispatch.telemetry.housekeeping, seed, float(dispatch.step_size)
    )
    config = scenario.make_config()
    scheduler = planner(
        config,
        [*scenario.make_targets(config), *too_targets(config, scenario.toos)],
        scenario.begin,
        scenario.end,
        start_plan=seed,
        time_limit=60,
        max_iterations=100,
    )
    if planner is CpSatScheduler:
        monkeypatch.setattr(scheduler, "_solve", lambda start: start)
    incumbent = scheduler._complete_seed(seed)
    assert incumbent is not None
    assert incumbent.score == expected
    plan = scheduler.schedule()
    assert scheduler.score >= expected
    assert scheduler.search_iterations == [100]
    assert scheduler.search_limits == ["iteration budget"]
    run = DITL(config, plan=plan, begin=scenario.begin, end=scenario.end)
    run.step_size = scheduler.ctx.step_size
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    assert (
        dynamic_science_value(
            run.telemetry.housekeeping, run.plan, float(run.step_size)
        )
        >= expected
    )
    assert seed.model_dump_json() == original
    assert [e.interruptions for e in seed] == interruptions


@pytest.mark.parametrize(
    "case,accepted",
    [
        ("original interval", True),
        ("no interruption", False),
        ("released", False),
        ("moved", False),
        ("different end", False),
        ("resized", False),
        ("uninterruptible", False),
        ("budget", False),
        ("recorder", False),
    ],
)
def test_short_fragment_is_confined_to_its_original_interval_and_constraints(
    case, accepted
):
    category = ObservationCategory(name="Science", obsid_min=0, obsid_max=1000)
    config = make_config(2, categories=[category])
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    target = make_target(config, 100, 105, 10, minutes=10, ss_min=5)

    def builder():
        result = LocalSearchScheduler(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=2),
            include_passes=False,
            time_limit=0,
            max_iterations=0,
        )
        result._reserve_fixed()
        result._fixed = [result._copy_block(b) for b in result.timeline]
        result._initialize_decoder()
        return result

    original = builder()
    full = original._decode([_Snapshot(obsid=100, seconds=600, release=T0)])
    [entry] = [b.entry for b in full.final.timeline]
    cutoff = entry.collection_begin + 120
    if case == "uninterruptible":
        category.interruptible = False
    elif case == "budget":
        category.time_budget_seconds = 60
    elif case == "recorder":
        config.recorder = OnboardRecorder(capacity_gb=1, planning_limit=0.9)
        target.science_data = {"Default Instrument": ScienceData(rate_gbps=0.01)}
    scheduler = builder()
    snapshot = _Snapshot(
        obsid=100,
        seconds=60 if case == "resized" else 120,
        release=None if case == "released" else T0 + (60 if case == "moved" else 0),
        interrupted_window=None
        if case == "no interruption"
        else (entry.collection_begin, cutoff + (60 if case == "different end" else 0)),
    )
    decoded = scheduler._decode([snapshot])
    assert bool(decoded.final.timeline) is accepted
    assert target.ss_min == 300
    assert target.exptime == 600
    assert target.collected_seconds == 0
    if accepted:
        [block] = decoded.final.timeline
        assert block.end == cutoff
        assert block.entry.interruptions == (cutoff,)
        normalized = scheduler._normalized_placements(decoded)
        assert normalized.sequence[0].interrupted_window == (
            entry.collection_begin,
            cutoff,
        )
        assert normalized.final.score == decoded.final.score
