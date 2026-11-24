"""A plan leaves the spacecraft somewhere after each of its entries. Where
that attitude is about to be crossed by a safety limit, the plan ends the
entry in time to slew away and holds a safe attitude, as dispatch does."""

from datetime import datetime, timedelta, timezone

import pytest
from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

from conops import DITL, create_ditl
from conops.common import ACSMode
from conops.config import (
    AttitudeControlSystem,
    Battery,
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
)
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.targets import Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
END = BEGIN + timedelta(hours=3)
STEP = 10
SAFE_HOLD = "Safe attitude hold"


def _config() -> MissionConfig:
    """The Earth's limb is a safety limit as well as a science one, so the
    attitude an observation leaves is unsafe as soon as its target sets."""
    config = MissionConfig(
        constraint=Constraint(
            sun_constraint=SunConstraint(min_angle=45),
            earth_constraint=EarthLimbConstraint(min_angle=10),
            safety_constraint=EarthLimbConstraint(min_angle=10),
        ),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000.0),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=0.5, slew_acceleration=0.25, settle_time=30.0
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.constraint.ephem = TLEEphemeris(
        tle="examples/example.tle", begin=BEGIN, end=END, step_size=STEP
    )
    return config


def _target(config: MissionConfig) -> Pointing:
    target = Pointing(
        config=config,
        obsid=1,
        name="Held",
        ra=80.0,
        dec=-20.0,
        merit=100.0,
        ss_min=600,
        ss_max=3 * 3600,
    )
    target.exptime = 3 * 3600
    return target


def _execute(config: MissionConfig, plan) -> DITL:  # type: ignore[no-untyped-def]
    run = DITL(config=config, plan=plan, begin=BEGIN, end=END)
    run.step_size = STEP
    assert run.calc()
    return run


def _science(plan) -> list:  # type: ignore[no-untyped-def]
    return [e for e in plan if int(e.obsid) == 1]


@pytest.mark.parametrize("scheduler", [PriorityScheduler, LocalSearchScheduler])
def test_a_plan_holds_a_safe_attitude_after_its_target_sets(scheduler) -> None:  # type: ignore[no-untyped-def]
    """Before, the plan ran its one observation until the target set, the
    run idled into the keep-out a step later and spent the rest of it in
    safe mode. Nothing could be placed after that observation either."""
    config = _config()
    options = {"time_limit": 0.5} if scheduler is LocalSearchScheduler else {}
    plan = scheduler(
        config, [_target(config)], BEGIN, END, step_size=STEP, **options
    ).schedule()
    science = _science(plan)
    holds = [e for e in plan if e.name == SAFE_HOLD]
    # The target is observed again an orbit later, from the safe hold.
    assert len(science) >= 2
    assert holds
    assert float(holds[0].begin) >= float(science[0].end)
    assert float(science[1].begin) >= float(holds[0].end)
    run = _execute(config, plan)
    assert ACSMode.SAFE not in {ACSMode(int(mode)) for mode in run.mode}
    assert run.validate_plan_matches_execution() == []


def test_the_plan_collects_about_what_dispatch_does() -> None:
    config = _config()
    plan = PriorityScheduler(
        config, [_target(config)], BEGIN, END, step_size=STEP
    ).schedule()
    planned = sum(float(e.exposure or 0) for e in _science(plan))
    dispatch_config = _config()
    dispatch = create_ditl(dispatch_config, [_target(dispatch_config)], BEGIN, END)
    dispatch.step_size = STEP
    assert dispatch.calc()
    dispatched = sum(float(e.exposure or 0) for e in _science(dispatch.plan))
    assert planned >= 0.95 * dispatched


def test_no_hold_is_added_where_the_attitude_can_be_held() -> None:
    """Without the safety limit the attitude an observation leaves can be
    held, and the plan is as it was."""
    config = _config()
    config.constraint.safety_constraint = None
    plan = PriorityScheduler(
        config, [_target(config)], BEGIN, END, step_size=STEP
    ).schedule()
    assert not [e for e in plan if e.name == SAFE_HOLD]
