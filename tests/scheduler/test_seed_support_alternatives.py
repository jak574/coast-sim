"""Advisory seed dumps must not exclude better independent search results."""

from datetime import timedelta

import numpy as np
import pytest

from conops import DITL
from conops.config import StoredMomentumConfig, ThrusterDumpConfig
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.schedulers.priority_scheduler import ActivitySchedulingError
from conops.targets import ScheduledActivity

from .planning_scenario import BEGIN, MIN, T0, make_config, make_target


def support_scenario():
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
    ordinary = make_target(
        config, 100, 105, 10, merit=20, minutes=20, snapshot=10, ss_min=5
    )
    old_dump = ScheduledActivity(
        name="Earlier plan's dump",
        begin=T0 + 30 * MIN,
        duration_seconds=120,
        desaturation=True,
        obsid=997000,
    )
    seed = PriorityScheduler(
        config,
        [ordinary],
        BEGIN,
        BEGIN + timedelta(hours=1),
        activities=[old_dump],
        include_passes=False,
    ).schedule()
    urgent = make_target(
        config,
        101,
        110,
        10,
        merit=200,
        minutes=40,
        snapshot=40,
        ss_min=40,
        deadline=T0 + 15 * MIN,
    )
    return config, ordinary, urgent, seed


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
def test_seed_dump_cannot_hide_a_feasible_higher_value_visit(planner):
    config, ordinary, urgent, seed = support_scenario()
    original_seed = seed.model_dump_json()
    scheduler = planner(
        config,
        [ordinary, urgent],
        BEGIN,
        BEGIN + timedelta(hours=1),
        start_plan=seed,
        include_passes=False,
        time_limit=0,
    )
    plan = scheduler.schedule()
    science = sum(
        entry.collection_seconds_between(float(entry.begin), float(entry.end))
        for entry in plan
        if entry.obsid == urgent.obsid
    )
    assert science == 40 * MIN
    assert scheduler._complete_feasible()
    assert seed.model_dump_json() == original_seed
    run = DITL(config, plan=plan, begin=BEGIN, end=BEGIN + timedelta(hours=1))
    assert run.calc()
    assert run.validate_plan_matches_execution() == []


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
def test_failed_native_search_retains_a_stronger_feasible_priority_plan(
    planner, monkeypatch
):
    config, ordinary, urgent, seed = support_scenario()
    original_seed = seed.model_dump_json()
    end = BEGIN + timedelta(hours=1)
    priority = PriorityScheduler(
        config, [ordinary, urgent], BEGIN, end, include_passes=False
    ).schedule()
    scheduler = planner(
        config,
        [ordinary, urgent],
        BEGIN,
        end,
        start_plan=seed,
        include_passes=False,
        time_limit=0,
    )
    checked_priority = scheduler._complete_seed(priority)
    checked_seed = scheduler._complete_seed(seed)
    assert checked_priority is not None and checked_seed is not None
    assert checked_priority.score > checked_seed.score

    def failed_native(self, build):
        raise ActivitySchedulingError(
            ScheduledActivity(
                name="Unreachable proposed firing", begin=T0, duration_seconds=120
            ),
            "optimized proposal has no safe ingress",
        )

    # A failed optimized proposal must not invalidate the independently built
    # and checked priority baseline. The base priority planner stays real.
    monkeypatch.setattr(LocalSearchScheduler, "_plan_within_momentum", failed_native)
    plan = scheduler.schedule()
    assert (
        sum(
            entry.collection_seconds_between(float(entry.begin), float(entry.end))
            for entry in plan
            if entry.obsid == urgent.obsid
        )
        == 40 * MIN
    )
    assert scheduler.score >= checked_priority.score
    assert scheduler._complete_feasible()
    assert seed.model_dump_json() == original_seed
    run = DITL(config, plan=plan, begin=BEGIN, end=end)
    assert run.calc()
    assert run.validate_plan_matches_execution() == []


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
def test_weaker_priority_layout_is_optimized_before_being_rejected(planner):
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.inertia_tensor_body_kg_m2 = np.diag([1000, 1200, 800])
    config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig(
        gravity_gradient_enabled=True,
        capacity_n_m_s=1000,
        planning_level=0.6,
        thruster_dump=ThrusterDumpConfig(duration_seconds=120),
    )

    def target(obsid, merit, minutes, **options):
        return make_target(
            config,
            obsid,
            105,
            10,
            merit=merit,
            minutes=minutes,
            snapshot=minutes,
            ss_min=minutes,
            **options,
        )

    ordinary = target(100, 100, 10)
    second = target(101, 90, 10)
    early = target(102, 80, 10, deadline=T0 + 15 * MIN)
    later = target(103, 20, 20, earliest_start=T0 + 30 * MIN, deadline=T0 + 35 * MIN)
    long_visit = target(104, 60, 40, deadline=T0 + 15 * MIN)
    dump = ScheduledActivity(
        name="Advisory firing",
        begin=T0 + 30 * MIN,
        duration_seconds=120,
        desaturation=True,
        obsid=997000,
    )
    end = BEGIN + timedelta(hours=1, minutes=30)
    seed = PriorityScheduler(
        config,
        [ordinary, second, early.model_copy(update={"merit": 200, "fom": 200}), later],
        BEGIN,
        end,
        activities=[dump],
        include_passes=False,
    ).schedule()
    original_seed = seed.model_dump_json()
    requests = [ordinary, second, early, later, long_visit]
    options = {"include_passes": False, "seed": 7}
    if planner is LocalSearchScheduler:
        options.update(time_limit=30, max_iterations=500)
    else:
        options.update(
            time_limit=0,
            solver_time_limit=15,
            solver_deterministic_time=2,
            workers=1,
        )
    warm = planner(config, requests, BEGIN, end, start_plan=seed, **options)
    cold = planner(config, requests, BEGIN, end, **options)
    seed_trial = warm._complete_seed(seed)
    priority_plan = PriorityScheduler(
        config, requests, BEGIN, end, include_passes=False
    ).schedule()
    priority_trial = warm._complete_seed(priority_plan)
    assert seed_trial is not None and priority_trial is not None
    # Comparing only these initial scores rejects the layout that can fit
    # the long visit, before either native optimizer gets to improve it.
    assert seed_trial.score > priority_trial.score

    plan, independent_plan = warm.schedule(), cold.schedule()
    selected = warm._complete_seed(plan)
    independent = warm._complete_seed(independent_plan)
    assert selected is not None and independent is not None
    assert selected.score >= independent.score > seed_trial.score
    assert (
        sum(
            entry.collection_seconds_between(float(entry.begin), float(entry.end))
            for entry in plan
            if entry.obsid == long_visit.obsid
        )
        == 40 * MIN
    )
    assert warm._complete_feasible()
    assert seed.model_dump_json() == original_seed
    if planner is LocalSearchScheduler:
        assert warm.search_iterations == [500, 500]
        assert warm.search_limits == ["iteration budget", "iteration budget"]
    else:
        assert len(warm.solver_search_limits) >= len(cold.solver_search_limits)
        assert "wall clock" not in warm.solver_search_limits
    run = DITL(config, plan=plan, begin=BEGIN, end=end)
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
