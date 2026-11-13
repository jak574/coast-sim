"""Advisory seed dumps must not exclude a better independent priority layout."""

from datetime import timedelta

import numpy as np
import pytest

from conops import DITL
from conops.config import StoredMomentumConfig, ThrusterDumpConfig
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import ScheduledActivity

from .planning_scenario import BEGIN, MIN, T0, make_config, make_target


@pytest.mark.parametrize("planner", [LocalSearchScheduler, CpSatScheduler])
def test_seed_dump_cannot_hide_a_feasible_higher_value_visit(planner):
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
    original_seed = seed.model_dump_json()
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
