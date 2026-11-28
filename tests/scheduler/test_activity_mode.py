"""Scheduled activities are held in ACTIVITY mode, with the IDLE loads and
constraint scopes unless ACTIVITY is given its own."""

from datetime import timedelta

import pytest

from conops import DITL, QueueDITL
from conops.common import ACSMode, ObsType
from conops.config import (
    AttitudeConstraintScope,
    Instrument,
    MissionConfig,
    PowerDraw,
)
from conops.config.data_generator import HousekeepingData
from conops.config.instrument import InstrumentMode
from conops.ditl import RollingHorizonDITL
from conops.schedulers import PriorityScheduler
from conops.targets import Plan, ScheduledActivity

from .planning_scenario import BEGIN, MIN, T0, make_config


@pytest.mark.parametrize(
    "fields,mode",
    [
        ({}, ACSMode.ACTIVITY),
        ({"desaturation": True}, ACSMode.DESAT),
    ],
)
def test_activity_and_saved_entry_agree_on_hold_mode(fields, mode):
    activity = ScheduledActivity(
        name="Engineering", begin=T0, duration_seconds=600, obsid=998000, **fields
    )
    entry = activity.plan_entry(None, (105, 10, 0))
    restored = Plan.model_validate_json(Plan(entries=[entry]).model_dump_json())
    assert (
        activity.hold_mode == entry.activity_mode == restored[0].activity_mode == mode
    )


def test_activity_inherits_idle_constraint_scopes_unless_overridden():
    safety = AttitudeConstraintScope.HARDWARE_SAFETY
    assert MissionConfig().attitude_constraint_scopes_for_mode(ACSMode.ACTIVITY) == [
        safety
    ]
    inherited = MissionConfig(
        attitude_constraint_scopes={
            "IDLE": [safety, AttitudeConstraintScope.POWER_GENERATION]
        }
    )
    assert inherited.attitude_constraint_scopes_for_mode(ACSMode.ACTIVITY) == [
        safety,
        AttitudeConstraintScope.POWER_GENERATION,
    ]
    own = MissionConfig(attitude_constraint_scopes={"ACTIVITY": []})
    assert own.attitude_constraint_scopes_for_mode(ACSMode.ACTIVITY) == []


@pytest.mark.parametrize("in_eclipse", [False, True])
def test_activity_power_inherits_idle_with_explicit_overrides(in_eclipse):
    power = PowerDraw(
        nominal_power=50,
        eclipse_power=60,
        power_mode={ACSMode.IDLE: 100},
        eclipse_power_mode={ACSMode.IDLE: 110},
    )
    assert power.power(ACSMode.ACTIVITY, in_eclipse) == (110 if in_eclipse else 100)
    power.power_mode[ACSMode.ACTIVITY] = 200
    power.eclipse_power_mode[ACSMode.ACTIVITY] = 210
    assert power.power(ACSMode.ACTIVITY, in_eclipse) == (210 if in_eclipse else 200)
    assert power.power(99, in_eclipse) == (60 if in_eclipse else 50)


def test_activity_housekeeping_inherits_idle_with_explicit_override():
    data = HousekeepingData(rate_gbps=1, rate_mode={ACSMode.IDLE: 2})
    assert data.data_generated(10, mode=ACSMode.ACTIVITY) == 20
    data.rate_mode[ACSMode.ACTIVITY] = 3
    assert data.data_generated(10, mode=ACSMode.ACTIVITY) == 30


@pytest.mark.parametrize("kind", ["dispatch", "plan", "rolling"])
@pytest.mark.parametrize("override", [False, True])
def test_activity_mode_and_payload_load_execute_and_clear(kind, override, tmp_path):
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.power_draw = PowerDraw(
        nominal_power=50,
        power_mode={ACSMode.IDLE: 100, **({ACSMode.ACTIVITY: 200} if override else {})},
    )
    config.payload.instruments = [
        Instrument(
            name="camera",
            modes=[
                InstrumentMode(name="IDLE", power_w=10),
                InstrumentMode(name="SCIENCE", power_w=40),
                InstrumentMode(name="calibration", power_w=30, observing=False),
            ],
        )
    ]
    activity = ScheduledActivity(
        name="Engineering",
        begin=T0 + 10 * MIN,
        duration_seconds=10 * MIN,
        instrument_modes={"camera": "calibration"},
    )
    end = BEGIN + timedelta(hours=2)
    if kind == "dispatch":
        run = QueueDITL(config=config, begin=BEGIN, end=end, activities=[activity])
    elif kind == "rolling":
        run = RollingHorizonDITL(
            config,
            [],
            begin=BEGIN,
            end=end,
            activities=[activity],
            scheduler=PriorityScheduler,
            include_passes=False,
            horizon=timedelta(hours=1),
            replan_interval=timedelta(minutes=15),
        )
    else:
        plan = PriorityScheduler(
            config,
            [],
            BEGIN,
            end,
            activities=[activity],
            include_passes=False,
        ).schedule()
        run = DITL(
            config=config,
            plan=Plan.load(plan.save(tmp_path / "plan.json")),
            begin=BEGIN,
            end=end,
        )
    assert run.calc()
    [entry] = [e for e in run.plan if e.obstype == ObsType.ACTIVITY]
    samples = [
        h
        for h in run.telemetry.housekeeping
        if entry.activity_hold_begin <= h.timestamp.timestamp() < float(entry.end)
    ]
    assert samples
    assert {h.acs_mode for h in samples} == {ACSMode.ACTIVITY}
    assert all(h.power_bus == (200 if override else 100) for h in samples)
    assert all(h.power_payload == 30 for h in samples)
    assert all(h.instrument_modes == {"camera": "calibration"} for h in samples)
    assert run.validate_plan_matches_execution() == []
    assert not run.acs.activity_active
    assert run.acs.get_mode(T0 + 90 * MIN) == ACSMode.IDLE
    samples[0].acs_mode = ACSMode.IDLE
    run.mode[run.utime.index(samples[0].timestamp.timestamp())] = int(ACSMode.IDLE)
    assert run.validate_plan_matches_execution()
