"""A dump before one visit must not strand it behind a later stale dump."""

from datetime import timedelta

import numpy as np
import pytest

from conops.config import StoredMomentumConfig
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.schedulers.momentum import MomentumForecast
from conops.schedulers.priority_scheduler import StartState
from conops.targets import Pointing
from conops.targets.sequence import ObservationSequence, SequenceStep

from .test_momentum_ledger import BEGIN, DUMP, T0, _config


@pytest.mark.parametrize("initial_fraction", [0.0, 0.7])
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
def test_dump_rebuilds_displaced_observations(
    monkeypatch, kind, sequence, initial_fraction
):
    config = _config()
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    origin = (80.0, -20.0, 0.0)
    duration = 600
    targets = [
        Pointing(
            config=config,
            obsid=obsid,
            ra=origin[0],
            dec=origin[1],
            roll_deg=0,
            merit=100 - obsid,
            deadline=T0 + (150 if obsid == 1 else duration + 360),
            exptime=duration,
            ss_min=duration,
            ss_max=duration,
            sequence=(
                ObservationSequence(
                    steps=[SequenceStep(name="science", duration_seconds=duration)]
                )
                if sequence
                else None
            ),
            visits=1,
        )
        for obsid in (1, 2)
    ]
    settings = dict(
        config=config,
        targets=targets,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=1),
        step_size=10,
        include_passes=False,
        start_state=StartState(time=T0, attitude=origin),
    )
    # Set the band from this orbit's real gravity-gradient loading, with
    # room for each visit and the firing after it, but not for both visits.
    empty = PriorityScheduler(**settings)
    empty.schedule()
    first = next(block for block in empty.timeline if block.entry.obsid == 1)
    assert empty._momentum is not None
    loading = empty._momentum.loading(first)
    empty._momentum.profile([first])
    direction = empty._momentum.tracker.momentum_eci_n_m_s
    assert direction is not None and loading > 0
    capacity = 1.7 * loading
    initial = direction / np.linalg.norm(direction) * (initial_fraction * capacity)
    config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig(
        gravity_gradient_enabled=True,
        capacity_n_m_s=capacity,
        planning_level=0.6 if initial_fraction else 0.3,
        thruster_dump=DUMP,
    )
    if kind == "priority":
        planner = PriorityScheduler
        options = {}
    elif kind == "local_search":
        planner = LocalSearchScheduler
        options = dict(max_iterations=0)
    else:
        # Test the shared momentum loop, independently of solver search.
        monkeypatch.setattr(CpSatScheduler, "_solve", lambda self, start: start)
        planner = CpSatScheduler
        options = dict(workers=1)
    builder = planner(
        **settings,
        **options,
        start_momentum_eci_n_m_s=tuple(float(value) for value in initial),
    )
    plan = builder.schedule()
    science = [entry for entry in plan if not entry.desaturation]
    assert {entry.obsid for entry in science} == {1, 2}
    assert (
        sum(
            entry.collection_seconds_between(entry.begin, entry.end)
            for entry in science
        )
        == 2 * duration
    )
    dumps = [entry for entry in plan if entry.desaturation]
    assert len(dumps) >= 2
    for dump in dumps:
        assert not any(
            entry.begin < dump.end and dump.begin < entry.end for entry in science
        )
    fresh = MomentumForecast(
        config,
        builder.ctx,
        start_attitude=origin,
        start_momentum_eci_n_m_s=tuple(float(value) for value in initial),
    )
    _, norms = fresh.profile(builder.timeline)
    assert norms.max() < capacity
