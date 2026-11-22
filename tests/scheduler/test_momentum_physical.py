"""Momentum forecasts follow the physical trajectories ACS installs: committed
slews and holds, ground contacts, and moving-target tracking."""

from datetime import timedelta

import numpy as np
import pytest

from conops.common import ACSCommandType, ObsType
from conops.ditl import RollingHorizonDITL
from conops.schedulers.context import SchedulingContext
from conops.schedulers.momentum import MomentumForecast
from conops.schedulers.priority_scheduler import _Block
from conops.simulation.acs_command import ACSCommand
from conops.simulation.attitude import AttitudeExecutor, AttitudeTrajectory
from conops.simulation.passes import Pass
from conops.simulation.slew import Slew
from conops.targets import PlanEntry, track_from_table

from .momentum_scenario import BEGIN, T0
from .momentum_scenario import momentum_config as _config

INITIAL = (0.1, -0.2, 0.3)


def _contact(late=0, wait=0):
    config = _config()
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    gspass = Pass(
        config=config,
        station="Test station",
        obsid=900,
        begin=T0,
        length=600,
        utime=[T0 + k * 60 for k in range(11)],
        ra=[80 + k * 0.8 for k in range(11)],
        dec=[-20] * 11,
        roll=[10] * 11,
    )
    acquire = T0 + late
    begin = acquire - wait
    profile = list(zip(gspass.ra, gspass.dec, gspass.roll, strict=True))
    origin = gspass.attitude_for_profile_at(profile, acquire)
    assert origin is not None
    entry = PlanEntry(
        config=config,
        obstype=ObsType.GSP,
        obsid=gspass.obsid,
        begin=begin,
        end=gspass.end,
        contact_begin=acquire,
        contact_end=gspass.end,
        slewtime=0,
        ra=origin[0],
        dec=origin[1],
        roll=origin[2],
    )
    run = RollingHorizonDITL(
        config, [], begin=BEGIN, end=BEGIN + timedelta(seconds=720)
    )
    run.step_size = 10
    run._entry_passes = {id(entry): gspass}
    run.acs.ra, run.acs.dec, run.acs.roll = origin
    run.acs._executor = AttitudeExecutor(begin, origin)
    run.acs.passrequests.passes = [gspass]
    run.acs._start_pass(
        ACSCommand(command_type=ACSCommandType.START_PASS, execution_time=acquire),
        acquire,
    )
    motion = run.acs._executor._trajectory
    assert motion is not None and motion.end > entry.end
    context = SchedulingContext(
        config, BEGIN + timedelta(seconds=begin - T0), BEGIN + timedelta(seconds=720)
    )
    block = _Block(
        entry=entry,
        attitude_in=origin,
        attitude_out=motion.state(motion.end).attitude,
        ready=acquire,
        end=motion.end,
        gspass=gspass,
    )
    return config, context, block, motion, run


def _physical(run, motion, origin, begin, end, initial):
    # The ACS motion's start and physical stop are integration boundaries,
    # including a wait before acquisition and an off-grid terminal brake.
    cursor = begin
    for stop in sorted({motion.start, motion.end, end}):
        stop = min(stop, end)
        if stop <= cursor:
            continue
        initial = run._forecast_momentum(
            cursor,
            stop,
            lambda t: origin if t < motion.start else motion.state(t).attitude,
            start_eci_n_m_s=initial,
        ).end_eci_n_m_s
        cursor = stop
    return initial


def _tracking(idle_seconds=120, middle=84):
    config = _config()
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    origin = (80.0, -20.0, 10.0)
    entry = PlanEntry(
        config=config,
        obsid=702,
        obstype=ObsType.PPT,
        ra=80,
        dec=-20,
        roll=10,
        begin=T0,
        end=T0 + 1200,
        slewtime=0,
        track=track_from_table(
            [T0, T0 + 600, T0 + 1200, T0 + 3600],
            [80, middle, 88, 100],
            [-20] * 4,
        ),
    )
    limits = config.spacecraft_bus.attitude_control
    end = BEGIN + timedelta(seconds=1200 + idle_seconds)
    run = RollingHorizonDITL(config, [], begin=BEGIN, end=end)
    run.step_size = 10
    slew = Slew.idle_hold(config, *origin, T0)
    slew.obstype, slew.at, slew.obsid = ObsType.PPT, entry, entry.obsid
    ingress = AttitudeTrajectory.from_slew(slew, limits)
    motion = run.acs._tracking_after(slew, ingress, limits)
    assert motion is not None
    context = SchedulingContext(config, BEGIN, end)
    block = _Block(
        entry=entry,
        attitude_in=origin,
        attitude_out=motion.state(motion.end).attitude,
        ready=T0,
        end=motion.end,
    )
    return config, context, block, motion, run


@pytest.mark.parametrize("kind,peak", [("contact", 0.1), ("tracking", 0.05)])
def test_wheel_demand_includes_the_body_rate(kind, peak):
    config, context, block, motion, _ = _contact() if kind == "contact" else _tracking()
    momentum = config.spacecraft_bus.attitude_control.stored_momentum
    momentum.gravity_gradient_enabled = False
    momentum.capacity_n_m_s = 1
    momentum.slew_momentum_enabled = True
    forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=(0, 0, 0))
    times, demand = forecast.demand([block])
    inertia = np.asarray(config.spacecraft_bus.inertia_tensor_body_kg_m2)
    # No external torque and zero initial H: the wheels carry the body's I*omega.
    expected = [
        np.linalg.norm(inertia @ np.deg2rad(motion.state(t).angular_velocity_body))
        for t in times
    ]
    assert max(expected) > peak
    np.testing.assert_allclose(demand, expected, rtol=0, atol=1e-10)
    assert demand[-1] == pytest.approx(0)
    assert forecast.loading(block) >= max(expected) - 1e-10


@pytest.mark.parametrize("kind", ["contact", "tracking"])
def test_cache_distinguishes_paths_with_the_same_endpoints(kind, monkeypatch):
    if kind == "contact":
        config, context, first, _, _ = _contact()
        ra = first.gspass.ra.copy()
        ra[5] += 1
        second = first.model_copy(
            update={"gspass": first.gspass.model_copy(update={"ra": ra})}
        )
    else:
        config, context, first, _, _ = _tracking(middle=83)
        second = _tracking(middle=86)[2]
        second.entry.config = config
    forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=INITIAL)
    forecast.profile([first])
    first_h = forecast.tracker.momentum_eci_n_m_s.copy()
    with monkeypatch.context() as patch:

        def uncached_update(**kwargs):
            pytest.fail("An equivalent rebuilt block should reuse its integration")

        patch.setattr(forecast.tracker, "update", uncached_update)
        forecast.profile([first.model_copy(update={"entry": first.entry.model_copy()})])
    forecast.profile([second])
    second_h = forecast.tracker.momentum_eci_n_m_s.copy()
    fresh = MomentumForecast(config, context, start_momentum_eci_n_m_s=INITIAL)
    fresh.profile([second])
    np.testing.assert_array_equal(second_h, fresh.tracker.momentum_eci_n_m_s)
    assert np.linalg.norm(first_h - second_h) > 1e-4
