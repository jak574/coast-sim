"""Cadence follows visit times when visibility spans have another order."""

import numpy as np
import pytest

from conops.schedulers.cpsat_scheduler import _ChunkModel, _Task

pytest.importorskip("ortools")


def test_cadence_orders_independent_visits_by_their_arrival_windows():
    pose = (0.0, 0.0, 0.0)
    tasks = [
        _Task(None, (pose, pose), (0, 0)),
        _Task(
            (100, 60),
            (pose, pose),
            (120, 120),
            collection=(30, 30),
            window=(60, 360),
            visit=0,
            spacing=60,
            cadence=60,
            rate=10,
        ),
        _Task(
            (100, 0),
            (pose, pose),
            (240, 240),
            copy=1,
            collection=(30, 30),
            window=(0, 360),
            visit=1,
            spacing=60,
            cadence=60,
            rate=10,
        ),
    ]
    chunk = _ChunkModel(tasks, np.zeros((3, 3), dtype=np.int64), 1, 360, {100: 60})
    status, ending, selected = chunk.solve(10, 1, 2029, 0, deterministic_time=20)
    assert status == "OPTIMAL" and ending == "optimal"
    assert selected is not None
    assert sum(s.seconds for s in selected) == 60
