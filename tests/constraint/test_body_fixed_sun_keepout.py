"""A Sun keep-out about a body-fixed direction that is not the boresight is
configured as a hardware-safety constraint with a boresight offset. It depends
on roll, and it holds at targets and along slew paths in dispatch and planned
runs."""

from datetime import timedelta

import numpy as np
import pytest
from rust_ephem import SunConstraint

from conops import DITL, QueueDITL
from conops.common import ACSMode
from conops.common.vector import normal_to_boresight_offset_euler_deg
from conops.config import MissionConfig
from conops.schedulers import PriorityScheduler
from conops.targets import Pointing
from tests.scheduler.planning_scenario import BEGIN, make_config, make_target

HOURS = 12
END = BEGIN + timedelta(hours=HOURS)
# A body-fixed direction 39 degrees from the boresight (+X), toward +Y, and
# the smallest angle the Sun may make with it.
TILT = np.radians(39.0)
NORMAL = (float(np.cos(TILT)), float(np.sin(TILT)), 0.0)
MIN_SUN_ANGLE_DEG = 15.0


def _scenario(keep_out: bool) -> tuple[MissionConfig, list[Pointing]]:
    config = make_config(HOURS)
    # Slow slews, so that each one is sampled at several steps.
    config.spacecraft_bus.attitude_control.max_slew_rate = 0.2
    if keep_out:
        roll, pitch, yaw = normal_to_boresight_offset_euler_deg(NORMAL)
        config.constraint.safety_constraint = SunConstraint(
            min_angle=MIN_SUN_ANGLE_DEG
        ).boresight_offset(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)
    rng = np.random.default_rng(7)
    targets = [
        make_target(
            config,
            100 + index,
            float(rng.uniform(0, 360)),
            float(np.degrees(np.arcsin(rng.uniform(-1, 1)))),
            minutes=10,
        )
        for index in range(150)
    ]
    return config, targets


def _run(kind: str, keep_out: bool) -> DITL | QueueDITL:
    config, targets = _scenario(keep_out)
    run: DITL | QueueDITL
    if kind == "dispatch":
        run = QueueDITL(config=config, begin=BEGIN, end=END)
        for target in targets:
            run.queue.add_request(target)
    else:
        plan = PriorityScheduler(
            config, targets, BEGIN, END, include_passes=False
        ).schedule()
        run = DITL(config=config, plan=plan, begin=BEGIN, end=END)
    assert run.calc()
    return run


def _sun_angles(run: DITL | QueueDITL) -> tuple[list[float], list[ACSMode]]:
    """The Sun's angle from ``NORMAL`` and the ACS mode at every step."""
    angles: list[float] = []
    modes: list[ACSMode] = []
    for sample in run.telemetry.housekeeping:
        assert sample.sun_body_vector is not None and sample.acs_mode is not None
        cosine = float(np.dot(np.asarray(sample.sun_body_vector), np.asarray(NORMAL)))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
        modes.append(ACSMode(sample.acs_mode))
    return angles, modes


def _slews_entering_from_legal_ends(angles: list[float], modes: list[ACSMode]) -> int:
    """Slews that start and end outside the keep-out and enter it between."""
    count = 0
    index = 1
    while index < len(modes):
        if modes[index] != ACSMode.SLEWING:
            index += 1
            continue
        stop = index
        while stop < len(modes) and modes[stop] == ACSMode.SLEWING:
            stop += 1
        ends = (angles[index - 1], angles[min(stop, len(angles) - 1)])
        if min(ends) >= MIN_SUN_ANGLE_DEG > min(angles[index:stop]):
            count += 1
        index = stop
    return count


@pytest.mark.parametrize("kind", ["dispatch", "plan"])
def test_the_keep_out_holds_at_targets_and_along_slews(kind: str) -> None:
    # Without the keep-out the same program puts the Sun inside it, both
    # while observing and during a slew between two legal attitudes.
    angles, modes = _sun_angles(_run(kind, keep_out=False))
    inside = {m for a, m in zip(angles, modes, strict=True) if a < MIN_SUN_ANGLE_DEG}
    assert {ACSMode.SCIENCE, ACSMode.SLEWING} <= inside
    assert _slews_entering_from_legal_ends(angles, modes) > 0

    run = _run(kind, keep_out=True)
    angles, modes = _sun_angles(run)
    assert min(angles) >= MIN_SUN_ANGLE_DEG
    assert ACSMode.SAFE not in modes
    assert modes.count(ACSMode.SCIENCE) > 0
    assert run.validate_plan_matches_execution() == []
