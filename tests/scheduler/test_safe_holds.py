"""Power-positive SAFE holds use real geometry, loads and executed attitudes."""

from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from conops import DITL, QueueDITL
from conops.common import ACSMode, ObsType
from conops.common.vector import attitude_to_quat
from conops.config import (
    ActivityAttitude,
    Instrument,
    MissionConfig,
    SingleAxisSolarArrayDrive,
    SolarPanel,
    SolarPanelSet,
    SpacecraftFrame,
    ThrusterDumpConfig,
)
from conops.config.instrument import InstrumentMode
from conops.ditl import RollingHorizonDITL
from conops.schedulers import ActivitySchedulingError
from conops.simulation.acs import ACS
from conops.simulation.safe_hold import (
    SafeHoldUnavailableError,
    ecliptic_poles,
    power_positive_attitudes,
)
from conops.targets import Plan, Program, ScheduledActivity

from .planning_scenario import BEGIN, HOUR, MIN, T0, make_config, make_target
from .test_activity_safety import _planner

POLICY = ActivityAttitude(kind="power_positive", minimum_margin_w=100)


def configuration(hours=4, payload_frame=False):
    config = make_config(hours)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    if payload_frame:
        config.spacecraft_bus.frame = SpacecraftFrame(
            boresight=(0, 0, 1), sun_side=(1, 0, 0)
        )
    config.solar_panel = SolarPanelSet(
        panels=[
            SolarPanel(normal=config.spacecraft_bus.frame.sun_side, max_power=3000)
        ],
        conversion_efficiency=1,
    )
    config.spacecraft_bus.power_draw.power_mode[int(ACSMode.SAFE)] = 700
    config.payload.instruments = [
        Instrument(
            name="camera",
            modes=[
                InstrumentMode(name="IDLE", power_w=50),
                InstrumentMode(name="SCIENCE", power_w=100),
                InstrumentMode(name="SAFE", power_w=20),
                InstrumentMode(name="standby", power_w=70),
            ],
        )
    ]
    return config


def activity():
    return ScheduledActivity(
        name="Safe engineering hold",
        begin=T0 + HOUR,
        duration_seconds=HOUR,
        fixed=True,
        safe_hold=True,
        attitude=POLICY,
        instrument_modes={"camera": "standby"},
    )


@pytest.mark.parametrize("payload_frame", [False, True])
@pytest.mark.parametrize(
    "kind,rolling",
    [
        ("dispatch", False),
        ("priority", False),
        ("local_search", False),
        ("cp_sat", False),
        ("priority", True),
    ],
)
def test_safe_hold_executes_positive_and_returns_to_normal(
    kind, rolling, payload_frame, tmp_path
):
    config = configuration(payload_frame=payload_frame)
    hold = activity()
    end = BEGIN + timedelta(hours=4)
    targets = [make_target(config, 100, 105, 10, earliest_start=hold.end, minutes=20)]
    if kind == "dispatch":
        run = QueueDITL(config=config, begin=BEGIN, end=end, activities=[hold])
        for target in targets:
            run.queue.add_request(target)
    else:
        scheduler, options = _planner(kind)
        if rolling:
            run = RollingHorizonDITL(
                config,
                targets,
                begin=BEGIN,
                end=end,
                activities=[hold],
                scheduler=scheduler,
                scheduler_options=options,
                include_passes=False,
                horizon=timedelta(hours=3),
                replan_interval=timedelta(minutes=30),
            )
        else:
            plan = scheduler(
                config,
                targets,
                BEGIN,
                end,
                activities=[hold],
                include_passes=False,
                **options,
            ).schedule()
            run = DITL(
                config=config,
                plan=Plan.load(plan.save(tmp_path / "plan.json")),
                begin=BEGIN,
                end=end,
            )
    assert run.calc()
    [entry] = [e for e in run.plan if e.obstype == ObsType.ACTIVITY]
    assert entry.safe_hold and entry.activity_mode == ACSMode.SAFE
    assert entry.activity_hold_begin == hold.begin
    assert float(entry.end) == hold.end
    samples = [
        h
        for h in run.telemetry.housekeeping
        if hold.begin <= h.timestamp.timestamp() < hold.end
    ]
    assert len(samples) == 60
    assert {int(h.acs_mode) for h in samples} == {int(ACSMode.SAFE)}
    lit = [
        h
        for h in samples
        if not config.constraint.in_eclipse(0, 0, h.timestamp.timestamp())
    ]
    assert lit
    for hk in lit:
        assert hk.power_bus == 700
        assert hk.instrument_modes == {"camera": "standby"}
        assert hk.power_payload == 70
        generated = config.solar_panel.power(
            hk.timestamp.timestamp(),
            hk.ra,
            hk.dec,
            config.constraint.ephem,
            roll=hk.roll,
        )
        assert generated - hk.power_usage > POLICY.minimum_margin_w
    assert not run.acs.in_safe_mode
    assert not run.acs.scheduled_safe_hold
    assert int(run.telemetry.housekeeping[-1].acs_mode) != int(ACSMode.SAFE)
    assert any(
        int(h.acs_mode) == int(ACSMode.SCIENCE) and h.timestamp.timestamp() >= hold.end
        for h in run.telemetry.housekeeping
    )
    assert run.validate_plan_matches_execution() == []
    # The declared payload boresight, rather than physical body +X, is polar.
    pose = (entry.ra, entry.dec, entry.roll)
    pole_poses = [
        config.spacecraft_bus.frame.mounting.body_attitude(ra, dec, roll)
        for ra, dec in ecliptic_poles()
        for roll in range(360)
    ]
    assert any(
        np.allclose(attitude_to_quat(*pose), attitude_to_quat(*p), atol=1e-7)
        or np.allclose(attitude_to_quat(*pose), -attitude_to_quat(*p), atol=1e-7)
        for p in pole_poses
    )


@pytest.mark.parametrize("payload_frame", [False, True])
def test_fault_safe_mode_uses_the_same_whole_hold_policy_and_stays_latched(
    payload_frame,
):
    config = configuration(hours=10, payload_frame=payload_frame)
    config.spacecraft_bus.attitude_control.safe_mode_attitude = POLICY
    acs = ACS(config=config)
    acs.request_safe_mode(T0)
    acs.pointing(T0)
    assert acs.in_safe_mode
    arrival = acs.last_slew.slewend
    lit = []
    for t in range(int(T0), int(T0 + 10 * HOUR), MIN):
        ra, dec, roll, _ = acs.pointing(t)
        assert int(acs.get_mode(t)) == int(ACSMode.SAFE)
        if t < arrival or config.constraint.in_eclipse(0, 0, t):
            continue
        power = config.solar_panel.power(t, ra, dec, config.constraint.ephem, roll=roll)
        load = config.spacecraft_bus.power(ACSMode.SAFE) + config.payload.power(
            ACSMode.SAFE
        )
        lit.append(power - load)
    assert lit and min(lit) > POLICY.minimum_margin_w
    acs.request_battery_charge(T0 + 10 * HOUR, 10, 20, 30, 123)
    assert not acs.command_queue


def test_power_must_stay_positive_after_the_initial_sample():
    config = configuration()
    with patch.object(
        SolarPanelSet,
        "power",
        side_effect=lambda t, *args, **kw: 3000 if t < T0 + HOUR else 0,
    ):
        with pytest.raises(SafeHoldUnavailableError, match="every sunlit sample"):
            next(
                power_positive_attitudes(
                    config, POLICY, T0, T0 + 3 * HOUR, ACSMode.SAFE
                )
            )


def test_missing_power_is_an_explicit_failure_for_scheduled_and_fault_holds():
    config = configuration()
    config.solar_panel.panels = []
    scheduler, options = _planner("priority")
    with pytest.raises(ActivitySchedulingError, match="no solar panels"):
        scheduler(
            config,
            [],
            BEGIN,
            BEGIN + timedelta(hours=4),
            activities=[activity()],
            include_passes=False,
            **options,
        ).schedule()
    config.spacecraft_bus.attitude_control.safe_mode_attitude = POLICY
    acs = ACS(config=config)
    acs.request_safe_mode(T0)
    with pytest.raises(SafeHoldUnavailableError, match="no solar panels"):
        acs.pointing(T0)
    assert acs.in_safe_mode


def test_safe_hold_policy_roundtrips_and_invalid_combinations_fail(tmp_path):
    program = Program(begin=T0, end=T0 + 4 * HOUR, activities=[activity()])
    assert Program.load(program.save(tmp_path / "program.json")) == program
    config = configuration()
    config.spacecraft_bus.attitude_control.safe_mode_attitude = POLICY
    config.to_yaml_file(str(tmp_path / "config.yaml"))
    restored = MissionConfig.from_yaml_file(str(tmp_path / "config.yaml"))
    assert restored.spacecraft_bus.attitude_control.safe_mode_attitude == POLICY
    with pytest.raises(ValidationError, match="firing attitude"):
        ThrusterDumpConfig(attitude=POLICY)
    with pytest.raises(ValidationError):
        ScheduledActivity(
            name="invalid",
            begin=T0,
            duration_seconds=60,
            desaturation=True,
            safe_hold=True,
        )


def test_finite_drives_are_rejected_until_ingress_drive_forecasts_exist():
    config = configuration()
    config.solar_panel.panels[0].single_axis_drive = SingleAxisSolarArrayDrive(
        rotation_axis=(0, 0, 1),
        min_angle_deg=-90,
        max_angle_deg=90,
        max_rate_deg_per_s=1,
    )
    with pytest.raises(ValueError, match="body-mounted panels"):
        next(power_positive_attitudes(config, POLICY, T0, T0 + HOUR, ACSMode.SAFE))


def test_shadow_only_interval_does_not_claim_to_be_power_positive():
    config = configuration()
    with patch.object(type(config.constraint), "in_eclipse", return_value=True):
        with pytest.raises(SafeHoldUnavailableError, match="no sunlit samples"):
            next(power_positive_attitudes(config, POLICY, T0, T0 + HOUR, ACSMode.SAFE))


def test_ledger_counts_safe_load_only_from_hold_start_and_invalidates_mode_cache():
    from conops.schedulers.battery import BatteryLedger

    config = configuration()
    config.solar_panel.panels = []
    hold = activity().model_copy(update={"obsid": 998000})
    entry = hold.plan_entry(config, (105, 10, 25))
    entry.activity_hold_begin = hold.begin + 10 * MIN
    ledger = BatteryLedger(config, T0, T0 + 4 * HOUR, start_deficit_wh=0, step_size=MIN)
    addition = ledger.addition(entry)
    idle = config.spacecraft_bus.power(ACSMode.IDLE) + config.payload.power(
        ACSMode.IDLE, instrument_modes=hold.instrument_modes
    )
    safe = config.spacecraft_bus.power(ACSMode.SAFE) + config.payload.power(
        ACSMode.SAFE, instrument_modes=hold.instrument_modes
    )
    assert addition.amount == pytest.approx(idle / 6 + safe * 5 / 6)
    entry.safe_hold = False
    assert ledger.addition(entry).amount == pytest.approx(idle)


def test_fixed_fault_safe_pose_is_retained():
    from conops.common.vector import quaternion_attitude_delta

    config = configuration()
    pose = (105, 10, 25)
    config.spacecraft_bus.attitude_control.safe_mode_attitude = ActivityAttitude(
        kind="fixed", ra_deg=pose[0], dec_deg=pose[1], roll_deg=pose[2]
    )
    acs = ACS(config=config)
    acs.request_safe_mode(T0)
    acs.pointing(T0)
    for t in range(int(T0 + HOUR), int(T0 + 4 * HOUR), MIN):
        ra, dec, roll, _ = acs.pointing(t)
        assert quaternion_attitude_delta(ra, dec, roll, *pose)[0] < 1e-6
        assert acs.in_safe_mode


def test_geometry_must_stay_safe_after_the_initial_sample():
    config = configuration()
    with patch(
        "conops.simulation.safe_hold.attitude_constraint_name_for_scopes",
        side_effect=lambda constraint, scopes, ra, dec, t, **kw: (
            "blocked" if t > T0 else None
        ),
    ):
        with pytest.raises(SafeHoldUnavailableError, match="every sunlit sample"):
            next(power_positive_attitudes(config, POLICY, T0, T0 + HOUR, ACSMode.SAFE))


@pytest.mark.parametrize(
    "actual_mode,actual_roll,message",
    [(ACSMode.IDLE, 25, "mode_mismatch"), (ACSMode.SAFE, 30, "pointing_mismatch")],
)
def test_validator_checks_safe_hold_mode_and_whole_attitude(
    actual_mode, actual_roll, message
):
    from conops.ditl.plan_validator import PlanExecutionValidator

    config = configuration()
    entry = (
        activity()
        .model_copy(update={"obsid": 998000})
        .plan_entry(config, (105, 10, 25))
    )
    validator = PlanExecutionValidator(
        config=config,
        plan=Plan(entries=[entry]),
        utime=[entry.begin],
        ra=[105],
        dec=[10],
        roll=[actual_roll],
        obsid=[998000],
        mode=[actual_mode],
        end_time=entry.end,
        passes=[],
        attitude_violation_at=lambda *args: None,
    )
    assert any(message in mismatch.message for mismatch in validator.validate())
