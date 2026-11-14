"""The native circuit carries a pose through holds that inherit attitude."""

import numpy as np
import pytest

from conops.schedulers.cpsat_scheduler import _ChunkModel, _Task


@pytest.mark.parametrize("holds", [1, 2])
@pytest.mark.parametrize("inherits", [False, True])
def test_hold_ingress_and_departure_follow_the_commanded_attitude(holds, inherits):
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    pose = (0.0, 0.0, 0.0)
    tasks = [
        _Task(None, (pose, pose), (0, 0)),
        _Task((1, 0), (pose, pose), (0, 0), collection=(10, 10), window=(0, 10)),
    ]
    for n in range(holds):
        tasks.append(
            _Task(
                None,
                (pose, pose),
                (10 + 10 * n, 10 + 10 * n),
                duration=10,
                inherits_attitude=inherits,
            )
        )
    arrival = 10 + 10 * holds + 7
    tasks.append(
        _Task(
            (2, 0),
            (pose, pose),
            (arrival, arrival),
            collection=(10, 10),
            window=(0, arrival + 10),
        )
    )
    last = len(tasks) - 1
    slews = np.full((len(tasks), len(tasks)), 99, dtype=np.int64)
    np.fill_diagonal(slews, 0)
    slews[0, 1] = 0
    # The actual predecessor needs seven seconds to the final observation.
    # The stale pose stored on the hold would need 99 seconds.
    slews[1, last] = 7
    chunk = _ChunkModel(tasks, slews, 1, arrival + 10, {1: 10, 2: 10})
    chain = [*range(len(tasks)), 0]
    for before, after in zip(chain, chain[1:]):
        literal = chunk.arc_literals.get((before, after))
        if literal is None:
            chunk.model.add(False)
        else:
            chunk.model.add(literal == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 2
    status = solver.solve(chunk.model)
    if not inherits:
        assert status == cp_model.INFEASIBLE
        return
    assert status == cp_model.OPTIMAL
    for hold in range(2, last):
        assert solver.value(chunk.inherited_pose[hold]) == 1
        assert solver.value(chunk.slew_step[hold]) == 10 + 10 * (hold - 2)
    assert solver.value(chunk.inherited_slew[(last - 1, last)]) == 7
    assert solver.value(chunk.arrival[last]) == arrival
