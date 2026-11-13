"""Finalized proposals must beat an independently feasible whole incumbent."""

from datetime import timedelta

import numpy as np
import pytest

from conops import DITL
from conops.config import (
    OnboardRecorder,
    ScienceData,
    StoredMomentumConfig,
    ThrusterDumpConfig,
)
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.schedulers.priority_scheduler import ActivitySchedulingError
from conops.targets import Plan, ScheduledActivity

from .planning_scenario import BEGIN, MIN, T0, make_config, make_target


def scenario():
    config = make_config(1)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.inertia_tensor_body_kg_m2 = np.diag([1000, 1200, 800])
    config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig(
        gravity_gradient_enabled=True,
        capacity_n_m_s=1000,
        planning_level=0.6,
        thruster_dump=ThrusterDumpConfig(duration_seconds=120),
    )
    target = make_target(
        config,
        100,
        105,
        10,
        minutes=20,
        snapshot=20,
        ss_min=5,
        earliest_start=T0 + 20 * MIN,
    )
    firing = ScheduledActivity(
        name="Incumbent firing",
        begin=T0 + 10 * MIN,
        duration_seconds=120,
        desaturation=True,
        obsid=997000,
    )
    shorter = target.model_copy(update={"ss_max": 10 * MIN})
    shorter.exptime = 10 * MIN
    seed = PriorityScheduler(
        config,
        [shorter],
        BEGIN,
        BEGIN + timedelta(hours=1),
        activities=[firing],
        include_passes=False,
    ).schedule()
    assert any(entry.desaturation for entry in seed)
    return config, target, seed


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
@pytest.mark.parametrize("better_proposal", [False, True])
def test_complete_incumbent_comparison_ignores_a_rebuilt_tail_score(
    monkeypatch, planner, better_proposal
):
    config, target, seed = scenario()
    original_seed = seed.model_dump_json()
    scheduler = planner(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=1),
        start_plan=seed,
        include_passes=False,
        time_limit=0,
    )

    def finalized_proposal(build):
        assert [(dump.begin, dump.end) for dump in scheduler.dumps] == [
            (T0 + 10 * MIN, T0 + 12 * MIN)
        ]
        scheduler._start_from_priority_plan()
        # Model a final momentum pass whose reported score concerns only its
        # unlocked tail. Complete executable science decides the winner.
        timeline = scheduler._greedy_timeline if better_proposal else []
        result = scheduler._finish(timeline)
        scheduler.score = -1 if better_proposal else 1e12
        return result

    monkeypatch.setattr(scheduler, "_plan_within_momentum", finalized_proposal)
    # Isolate the supplied-seed/final-proposal comparison from the separately
    # tested independent optimized contender.
    monkeypatch.setattr(scheduler, "_independent_incumbent", lambda: None)
    monkeypatch.setattr(scheduler, "_refine_settled_plan", lambda plan: plan)
    monkeypatch.setattr(scheduler, "_augment_selected_plan", lambda plan: plan)
    plan = scheduler.schedule()
    science = sum(
        entry.collection_seconds_between(float(entry.begin), float(entry.end))
        for entry in plan
        if not entry.desaturation
    )
    assert science == (20 if better_proposal else 10) * MIN
    assert scheduler.incumbent_retained is not better_proposal
    assert scheduler.incumbent_score > 0
    if better_proposal:
        assert scheduler.score > scheduler.incumbent_score
    else:
        assert scheduler.score == scheduler.incumbent_score
        assert [
            (entry.activity_hold_begin, entry.activity_hold_end)
            for entry in plan
            if entry.desaturation
        ] == [(T0 + 10 * MIN, T0 + 12 * MIN)]
    assert seed.model_dump_json() == original_seed
    run = DITL(config, plan=plan, begin=BEGIN, end=BEGIN + timedelta(hours=1))
    assert run.calc()
    assert run.validate_plan_matches_execution() == []


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
def test_incompatible_seed_support_does_not_displace_a_new_required_hold(planner):
    config, target, seed = scenario()
    required = ScheduledActivity(
        name="New required hold",
        begin=T0 + 10 * MIN,
        duration_seconds=50 * MIN,
        fixed=True,
        obsid=998000,
    )
    scheduler = planner(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=1),
        activities=[required],
        start_plan=seed,
        include_passes=False,
        time_limit=0,
    )
    plan = scheduler.schedule()
    [hold] = [entry for entry in plan if entry.obsid == required.obsid]
    assert hold.activity_hold_begin == required.begin
    assert hold.activity_hold_end == required.end
    assert not any(entry.desaturation for entry in plan)
    assert scheduler.incumbent_score is None


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
def test_a_failed_proposals_reservation_keeps_a_separately_feasible_seed(
    monkeypatch, planner
):
    config, target, seed = scenario()
    scheduler = planner(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=1),
        start_plan=seed,
        include_passes=False,
        time_limit=0,
    )

    def unreachable_proposal(build):
        raise ActivitySchedulingError(
            ScheduledActivity(name="Proposal dump", begin=T0, duration_seconds=120),
            "proposal leaves no ingress",
        )

    monkeypatch.setattr(scheduler, "_plan_within_momentum", unreachable_proposal)
    # This case specifically proves recovery to the supplied seed's support.
    monkeypatch.setattr(scheduler, "_independent_incumbent", lambda: None)
    plan = scheduler.schedule()
    assert scheduler.incumbent_retained
    assert any(entry.obsid == target.obsid for entry in plan)
    assert any(entry.desaturation for entry in plan)
    assert scheduler._complete_feasible()


@pytest.mark.parametrize("resource", ["recorder", "momentum"])
def test_a_seed_is_rechecked_against_the_actual_starting_resources(resource):
    config, target, seed = scenario()
    config.recorder = OnboardRecorder(capacity_gb=10, planning_limit=0.9)
    target.science_data = {"Default Instrument": ScienceData(rate_gbps=0.01)}
    options = dict(include_passes=False, time_limit=0)
    valid = LocalSearchScheduler(
        config, [target], BEGIN, BEGIN + timedelta(hours=1), **options
    )
    assert valid._complete_seed(seed) is not None
    if resource == "recorder":
        options["start_volume_gb"] = 9.0
    else:
        options["start_momentum_eci_n_m_s"] = (1001.0, 0.0, 0.0)
    changed = LocalSearchScheduler(
        config, [target], BEGIN, BEGIN + timedelta(hours=1), **options
    )
    assert changed._complete_seed(seed) is None


def test_complete_seed_cannot_reopen_a_committed_visit_window():
    from conops.schedulers.local_search_scheduler import _Snapshot

    from .test_selected_visit_decode import scheduler_for_windows

    original, _ = scheduler_for_windows()
    decoded = original._decode([_Snapshot(obsid=200, seconds=3600, visit=0)])
    [block] = decoded.final.timeline
    seed = Plan(entries=[block.entry])
    current, target = scheduler_for_windows(reserve_first=True)
    assert current._complete_seed(seed) is None
    assert not target.visits_done


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
def test_a_higher_value_proposal_cannot_discard_committed_science(monkeypatch, planner):
    config, target, seed = scenario()
    [locked] = [entry for entry in seed if entry.obsid == target.obsid]
    valuable = make_target(config, 101, 110, 10, merit=200, minutes=10, snapshot=10)
    alternative = PriorityScheduler(
        config, [valuable], BEGIN, BEGIN + timedelta(hours=1), include_passes=False
    )
    alternative.schedule()
    scheduler = planner(
        config,
        [target, valuable],
        BEGIN,
        BEGIN + timedelta(hours=1),
        locked=[locked],
        start_plan=seed,
        include_passes=False,
        time_limit=0,
    )

    def drops_commitment(build):
        scheduler._start_from_priority_plan()
        return scheduler._finish(alternative.timeline)

    monkeypatch.setattr(scheduler, "_plan_within_momentum", drops_commitment)
    monkeypatch.setattr(scheduler, "_refine_settled_plan", lambda plan: plan)
    monkeypatch.setattr(scheduler, "_augment_selected_plan", lambda plan: plan)
    plan = scheduler.schedule()
    assert scheduler.incumbent_retained
    assert any(
        entry.obsid == locked.obsid
        and entry.collection_begin == locked.collection_begin
        and entry.collection_end == locked.collection_end
        for entry in plan
    )
