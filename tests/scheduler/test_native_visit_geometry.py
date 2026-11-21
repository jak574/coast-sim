"""The native route model must use each visit's declared science pose."""

from datetime import timedelta

import numpy as np
import pytest

from conops.config import TargetConfig
from conops.schedulers import CpSatScheduler
from conops.schedulers.cpsat_scheduler import _ChunkModel
from conops.targets import Pointing, VisitWindow
from tests.scheduler.planning_scenario import BEGIN, HOUR, T0, make_config

pytest.importorskip("ortools")


def test_native_candidates_use_the_visit_pointing_and_absolute_roll(monkeypatch):
    config = make_config(
        6, weights=TargetConfig(urgency_weight=0, completion_deficit_weight=0)
    )
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    config.battery.planning_level = None
    target = Pointing(
        config=config,
        obsid=100,
        name="Independent visits",
        ra=105,
        dec=10,
        roll_deg=0,
        merit=90,
        ss_min=600,
        ss_max=600,
        exptime=1200,
        visit_windows=[
            VisitWindow(start=T0 + 600, deadline=T0 + 1200),
            VisitWindow(
                start=T0 + 2 * HOUR, deadline=T0 + 3 * HOUR, ra=115, dec=20, roll_deg=25
            ),
        ],
    )
    captured = []
    native = _ChunkModel.__init__

    def record(self, tasks, *args, **kwargs):
        captured.extend(
            (task.visit, task.attitude_in, task.attitude_out)
            for task in tasks
            if task.obsid == 100
        )
        return native(self, tasks, *args, **kwargs)

    monkeypatch.setattr(_ChunkModel, "__init__", record)
    planner = CpSatScheduler(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=6),
        include_passes=False,
        time_limit=0,
        max_iterations=0,
        workers=1,
        seed=2029,
        solver_time_limit=10,
        solver_deterministic_time=1,
        chunk=timedelta(hours=6),
    )
    planner.schedule()
    wanted = target.visit_pointing(1).target_body_attitude(25)
    candidates = [
        (incoming, outgoing) for visit, incoming, outgoing in captured if visit == 1
    ]
    assert candidates
    for incoming, outgoing in candidates:
        np.testing.assert_allclose(incoming, wanted, rtol=0, atol=1e-9)
        np.testing.assert_allclose(outgoing, wanted, rtol=0, atol=1e-9)


def test_native_candidates_use_each_visits_visibility_span(monkeypatch):
    from conops.schedulers.context import SchedulingContext

    config = make_config(
        6, weights=TargetConfig(urgency_weight=0, completion_deficit_weight=0)
    )
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    config.battery.planning_level = None
    target = Pointing(
        config=config,
        obsid=100,
        name="Independent visibility",
        ra=105,
        dec=10,
        roll_deg=0,
        merit=90,
        ss_min=600,
        ss_max=600,
        exptime=1200,
        visit_windows=[
            VisitWindow(start=T0 + 600, deadline=T0 + 1200),
            VisitWindow(start=T0 + 2 * HOUR, deadline=T0 + 3 * HOUR, ra=115, dec=20),
        ],
    )

    def visibility(self, pointing):
        return (
            [(T0, T0 + HOUR)]
            if pointing.ra == 105
            else [(T0 + 2 * HOUR, T0 + 6 * HOUR)]
        )

    monkeypatch.setattr(SchedulingContext, "visibility_windows", visibility)
    captured = []
    native = _ChunkModel.__init__

    def record(self, tasks, *args, **kwargs):
        captured.extend((t.visit, t.window) for t in tasks if t.obsid == 100)
        return native(self, tasks, *args, **kwargs)

    monkeypatch.setattr(_ChunkModel, "__init__", record)
    planner = CpSatScheduler(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=6),
        include_passes=False,
        time_limit=0,
        max_iterations=0,
        workers=1,
        seed=2029,
        solver_time_limit=10,
        solver_deterministic_time=1,
        chunk=timedelta(hours=6),
    )
    planner.schedule()
    assert (0, (0, HOUR)) in captured
    assert (1, (2 * HOUR, 6 * HOUR)) in captured
