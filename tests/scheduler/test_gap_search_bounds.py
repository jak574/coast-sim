"""Later feasible departures and horizon-safe return forecasts."""

from datetime import timedelta

import numpy as np
import pytest

from conops.common.vector import (
    _quat_mul,
    attitude_to_quat,
    quat_to_attitude,
    quaternion_attitude_delta,
)
from conops.schedulers import PriorityScheduler
from conops.simulation.attitude import AttitudeTrajectory

from .planning_scenario import BEGIN, T0
from .planning_scenario import make_config as _config


@pytest.mark.parametrize("case", ["fast-body-axes", "fixed-settle"])
def test_the_fixed_pose_bound_never_exceeds_a_feasible_slew(case: str) -> None:
    config = _config(2, open_sky=case == "fast-body-axes")
    limits = config.spacecraft_bus.attitude_control
    if case == "fast-body-axes":
        limits.max_slew_rate_body = (0.01, 1.0, 1.0)
        limits.slew_acceleration_body = (0.01, 1.0, 1.0)
        limits.settle_time = 0.0
        first = (0.0, 0.0, 0.0)
        middle_q = _quat_mul(np.asarray((0.0, 0.0, 1.0, 0.0)), attitude_to_quat(*first))
        last_q = _quat_mul(np.asarray((0.0, 0.0, 0.0, 1.0)), middle_q)
        middle, last = quat_to_attitude(middle_q), quat_to_attitude(last_q)
        # Two executable fast-axis turns reach the same pose as a slow-axis turn.
        outbound = AttitudeTrajectory.turn(T0, first, middle, limits)
        inbound = AttitudeTrajectory.turn(outbound.end, middle, last, limits)
        feasible_duration = inbound.end - T0
        assert feasible_duration < 400.0
    else:
        limits.settle_time = 5.0
        first, last = (240.0, 20.0, 0.0), (110.0, -20.0, 0.0)
        distance, axis = quaternion_attitude_delta(*first, *last)
        feasible_duration = limits.slew_time(distance, axis)
    scheduler = PriorityScheduler(config, [], BEGIN, BEGIN + timedelta(hours=2))
    assert scheduler._shortest_slew_seconds(first, last) <= feasible_duration
