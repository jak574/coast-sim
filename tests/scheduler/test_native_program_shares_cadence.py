"""A dynamic program term can turn a zero origin rate into positive value."""

import numpy as np
import pytest
from ortools.sat.python import cp_model

from conops.schedulers._delivered_shares import NodeScience
from conops.schedulers.cpsat_scheduler import _ChunkModel, _Task


def test_dynamic_share_keeps_cadence_discount_when_origin_rate_is_zero():
    pose = (0.0, 0.0, 0.0)
    tasks = [
        _Task(None, (pose, pose), (0, 0)),
        _Task(
            None,
            (pose, pose),
            (0, 0),
            duration=600,
            science=NodeScience(program="B", fixed_seconds=600),
        ),
        _Task(
            (201, 0),
            (pose, pose),
            (600, 600),
            collection=(600, 600),
            window=(0, 1200),
            rate=0.0,
            late=0.0,
            due=0,
            cadence=600,
            spacing=600,
            science=NodeScience(
                program="A", allocated_share=0.5, weight=40.0, late=0.0005
            ),
        ),
    ]
    chunk = _ChunkModel(
        tasks,
        np.zeros((3, 3), dtype=np.int64),
        60,
        1200,
        {201: 600},
        delivered_history={"A": 600},
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 2
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    assert solver.boolean_value(chunk.present[2])
    assert solver.objective_value == pytest.approx(8400, rel=0, abs=1e-9)


def test_negative_dynamic_value_cannot_invent_cadence_lateness():
    pose = (0.0, 0.0, 0.0)
    tasks = [
        _Task(None, (pose, pose), (0, 0)),
        _Task(
            None,
            (pose, pose),
            (0, 0),
            duration=1800,
            science=NodeScience(program="A", fixed_seconds=1800),
        ),
        _Task(
            (201, 0),
            (pose, pose),
            (1800, 1800),
            collection=(600, 600),
            window=(0, 2400),
            rate=10.0,
            late=10 * 0.5 / 3600,
            due=1800,
            cadence=3600,
            spacing=3600,
            science=NodeScience(
                program="A", allocated_share=0.5, weight=40.0, late=0.5 / 3600
            ),
        ),
    ]
    chunk = _ChunkModel(
        tasks,
        np.zeros((3, 3), dtype=np.int64),
        60,
        2400,
        {201: 600},
        delivered_history={"B": 600},
    )
    chunk.model.add(chunk.present[2] == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 2
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    assert solver.value(chunk.lateness[2]) == 0
    assert solver.objective_value == -12000
