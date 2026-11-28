"""Replanning keeps useful incumbents and owns mutable locked-entry metadata."""

from datetime import timedelta

import pytest

from conops.config import TargetConfig
from conops.ditl import RollingHorizonDITL
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.targets import Plan, PlanEntry, Pointing, TOORequest, VisitWindow

from .planning_scenario import BEGIN, MIN, PATCH, T0, make_config, make_target
from .test_activity_safety import _planner


def _case():
    config = make_config(2, weights=TargetConfig(urgency_weight=0))
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    filler = make_target(config, 1, *PATCH[0], merit=1000, minutes=40)
    timed = Pointing(
        config=config,
        obsid=2,
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        merit=100,
        ss_min=20 * MIN,
        ss_max=20 * MIN,
        visit_windows=[VisitWindow(start=T0 + 10 * MIN, deadline=T0 + 15 * MIN)],
    )
    impossible = make_target(config, 3, *PATCH[0], merit=2000, minutes=180)
    return config, [filler, timed], impossible


def _good_plan():
    config, requests, _ = _case()
    plan = LocalSearchScheduler(
        config,
        requests,
        BEGIN,
        BEGIN + timedelta(hours=2),
        include_passes=False,
        seed=3,
        time_limit=30,
        max_iterations=50,
    ).schedule()
    assert [int(e.obsid) for e in plan] == [2, 1]
    return plan


@pytest.mark.parametrize("external_seed", [False, True])
def test_infeasible_trigger_does_not_erase_a_feasible_timed_visit(external_seed):
    config, requests, impossible = _case()
    seed = (
        PriorityScheduler(
            config, requests, BEGIN, BEGIN + timedelta(hours=2), include_passes=False
        ).schedule()
        if external_seed
        else None
    )
    original = seed.model_dump(mode="json") if seed is not None else None
    run = RollingHorizonDITL(
        config,
        requests,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=2),
        horizon=timedelta(hours=2),
        replan_interval=timedelta(hours=2),
        scheduler=LocalSearchScheduler,
        include_passes=False,
        start_plan=seed,
        scheduler_options=dict(seed=3, time_limit=30, max_iterations=50),
    )
    run.step_size = 60
    run.submit_too(
        TOORequest(
            request=impossible,
            submit_time=T0 + MIN,
            deadline=T0 + 2 * 3600,
            interrupt=False,
        )
    )
    assert run.calc()
    assert len(run.replans) == 2
    assert run.validate_plan_matches_execution() == []
    assert (
        sum(
            h.collection_seconds or 0
            for h in run.telemetry.housekeeping
            if h.obsid == 2 and not h.calibration
        )
        == 20 * MIN
    )
    assert not any(int(e.obsid) == 3 for e in run.plan)
    assert all(r.search_iterations == [50] for r in run.replans)
    if seed is not None:
        assert seed.model_dump(mode="json") == original


def test_stronger_external_seed_beats_a_weaker_additional_start():
    config, requests, _ = _case()
    weak = PriorityScheduler(
        config, requests, BEGIN, BEGIN + timedelta(hours=2), include_passes=False
    ).schedule()
    strong = _good_plan()
    originals = [p.model_dump(mode="json") for p in [strong, weak]]
    scheduler = LocalSearchScheduler(
        config,
        requests,
        BEGIN,
        BEGIN + timedelta(hours=2),
        include_passes=False,
        max_iterations=0,
        start_plan=strong,
        additional_start_plans=[weak],
    )
    plan = scheduler.schedule()
    assert [int(e.obsid) for e in plan] == [2, 1]
    assert scheduler.seed_score == scheduler.score == 2520000
    assert [p.model_dump(mode="json") for p in [strong, weak]] == originals


@pytest.mark.parametrize("reason", ["expired", "completed"])
def test_incumbent_is_decoded_against_current_visit_history(reason):
    prior = _good_plan()
    original = prior.model_dump(mode="json")
    config, requests, _ = _case()
    begin = BEGIN
    if reason == "expired":
        begin += timedelta(minutes=20)
    else:
        requests[1].record_collection(T0 + 30 * MIN, 20 * MIN, visit=0)
    scheduler = LocalSearchScheduler(
        config,
        requests,
        begin,
        BEGIN + timedelta(hours=2),
        include_passes=False,
        max_iterations=0,
        additional_start_plans=[prior],
    )
    assert [int(e.obsid) for e in scheduler.schedule()] == [1]
    assert prior.model_dump(mode="json") == original


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
def test_locked_ingress_recalculation_preserves_the_input_plan(kind):
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    target = make_target(config, 700, 105, 10, minutes=10, snapshot=10, ss_min=10)
    entry = PlanEntry(
        obsid=700,
        ra=105,
        dec=10,
        begin=T0 + 10 * MIN,
        end=T0 + 30 * MIN,
        collection_begin=T0 + 20 * MIN,
        collection_end=T0 + 30 * MIN,
        exptime=10 * MIN,
        ss_min=10 * MIN,
        ss_max=10 * MIN,
    )
    source = Plan(entries=[entry])
    original = source.model_dump(mode="json")
    scheduler_type, options = _planner(kind)
    plan = scheduler_type(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=2),
        include_passes=False,
        locked=source.entries,
        **options,
    ).schedule()
    [placed] = [e for e in plan if int(e.obsid) == 700]
    assert float(placed.begin) != T0 + 10 * MIN
    assert float(placed.collection_begin) == T0 + 20 * MIN
    assert float(placed.collection_end) == T0 + 30 * MIN
    assert source.model_dump(mode="json") == original
    assert entry.config is None


@pytest.mark.parametrize("scheduler_type", [PriorityScheduler, LocalSearchScheduler])
def test_callable_scheduler_factory_remains_compatible(scheduler_type):
    config, requests, _ = _case()
    calls = []

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return scheduler_type(*args, **kwargs)

    options = (
        dict(max_iterations=10, time_limit=30)
        if scheduler_type is LocalSearchScheduler
        else {}
    )
    run = RollingHorizonDITL(
        config,
        requests,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=2),
        horizon=timedelta(hours=2),
        replan_interval=timedelta(hours=1),
        include_passes=False,
        scheduler=factory,
        scheduler_options=options,
    )
    run.step_size = 60
    assert run.calc()
    assert len(calls) == 2
    assert run.validate_plan_matches_execution() == []
