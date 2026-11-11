"""Settled task ingresses need not follow a new snapshot's departure grid."""

import numpy as np
import pytest
from ortools.sat.python import cp_model

from conops.schedulers.cpsat_scheduler import _ChunkModel, _Task, _TaskHint


def _chunk(*, inherits: bool, second_slew: int = 0) -> _ChunkModel:
    pose = (105.0, 10.0, 0.0)
    tasks = [
        _Task(None, (pose, pose), (0, 0)),
        _Task(None, (pose, pose), (101, 101), duration=20, inherits_attitude=inherits),
        _Task(None, (pose, pose), (180, 180), duration=20, inherits_attitude=inherits),
        _Task(
            (1, 0),
            (pose, pose),
            (300, 900),
            collection=(20, 20),
            window=(0, 900),
            rate=1,
        ),
    ]
    slews = np.zeros((4, 4), dtype=np.int64)
    slews[1, 2] = second_slew
    chunk = _ChunkModel(tasks, slews, 120, 1000, {1: 20})
    for arc in ((0, 1), (1, 2), (2, 3), (3, 0)):
        literal = chunk.arc_literals.get(arc)
        chunk.model.add(False if literal is None else literal == 1)
    return chunk


def _solver() -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 2029
    solver.parameters.max_deterministic_time = 1
    solver.parameters.max_time_in_seconds = 10
    return solver


@pytest.mark.parametrize("inherits", [False, True])
def test_fixed_holds_keep_their_exact_times_between_snapshot_steps(inherits):
    chunk = _chunk(inherits=inherits)
    solver = _solver()
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    assert solver.value(chunk.arrival[1]) == 101
    assert solver.value(chunk.arrival[2]) == 180
    assert solver.boolean_value(chunk.present[3])


@pytest.mark.parametrize("slew,feasible", [(59, True), (60, False)])
def test_fixed_task_ingress_still_reserves_the_full_slew(slew, feasible):
    # The first hold ends at 121, leaving exactly 59 s before the second.
    chunk = _chunk(inherits=False, second_slew=slew)
    solver = _solver()
    expected = cp_model.OPTIMAL if feasible else cp_model.INFEASIBLE
    assert solver.solve(chunk.model) == expected


@pytest.mark.parametrize("inherits", [False, True])
def test_fixed_task_hints_preserve_a_feasible_route_between_steps(inherits):
    chunk = _chunk(inherits=inherits)
    chunk.add_hint(
        {1: _TaskHint(0, 101, 0), 2: _TaskHint(2, 180, 0), 3: _TaskHint(2, 300, 20)}
    )
    solver = _solver()
    solver.parameters.fix_variables_to_their_hinted_value = True
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    assert solver.value(chunk.fixed_slew_start[2]) == 121
    assert solver.value(chunk.slew_step[2]) == 1
    assert solver.value(chunk.slew_step[3]) == 2


def test_new_snapshot_departures_remain_on_the_original_grid():
    pose = (105.0, 10.0, 0.0)
    tasks = [
        _Task(None, (pose, pose), (0, 0)),
        _Task(None, (pose, pose), (101, 101), duration=20, inherits_attitude=True),
        _Task((1, 0), (pose, pose), (180, 180), collection=(20, 20), window=(0, 200)),
    ]
    chunk = _ChunkModel(tasks, np.zeros((3, 3), dtype=np.int64), 120, 300, {1: 20})
    chunk.model.add(chunk.present[2] == 1)
    # Science cannot fit before the hold. After it ends at 121, the next
    # permitted new departure is 240, beyond the snapshot's arrival bound.
    solver = _solver()
    assert solver.solve(chunk.model) == cp_model.INFEASIBLE
