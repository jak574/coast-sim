"""Where no starting pointing is given, the run starts at one its own
constraints allow."""

from datetime import datetime, timedelta, timezone

import pytest
from rust_ephem import SunConstraint, TLEEphemeris

from conops import create_ditl
from conops.common import ACSMode
from conops.common.vector import angular_separation
from conops.config import (
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    SolarPanelSet,
)
from conops.simulation.acs import ACS
from conops.targets import Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
END = BEGIN + timedelta(hours=1)
T0 = BEGIN.timestamp()
KEEP_OUT = 45.0


def _config() -> MissionConfig:
    """At this start the default pointing, away from the Earth, is 20
    degrees from the Sun, inside the 45 degree safety keep-out."""
    config = MissionConfig(
        constraint=Constraint(safety_constraint=SunConstraint(min_angle=KEEP_OUT)),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
    )
    config.constraint.ephem = TLEEphemeris(
        tle="examples/example.tle", begin=BEGIN, end=END, step_size=10
    )
    return config


def _sun_angle(config: MissionConfig, ra: float, dec: float) -> float:
    ephem = config.constraint.ephem
    assert ephem is not None
    return float(
        angular_separation(
            ra, dec, float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0])
        )
    )


def test_the_default_pointing_is_inside_the_keep_out_here() -> None:
    config = _config()
    ephem = config.constraint.ephem
    assert ephem is not None
    default = (
        (180 + float(ephem.earth_ra_deg[0])) % 360,
        -float(ephem.earth_dec_deg[0]),
    )
    assert _sun_angle(config, *default) < KEEP_OUT


def test_a_run_given_no_pointing_starts_outside_the_keep_out() -> None:
    config = _config()
    acs = ACS(config=config)
    ra, dec, _ = acs._advance_attitude(T0).state.attitude
    assert _sun_angle(config, ra, dec) >= KEEP_OUT


def test_a_pointing_given_explicitly_is_left_as_it_is() -> None:
    config = _config()
    ephem = config.constraint.ephem
    assert ephem is not None
    acs = ACS(config=config)
    acs.ra = float(ephem.sun_ra_deg[0])
    acs.dec = float(ephem.sun_dec_deg[0])
    ra, dec, _ = acs._advance_attitude(T0).state.attitude
    assert _sun_angle(config, ra, dec) == pytest.approx(0.0, abs=1e-6)


def test_the_run_observes_and_does_not_start_in_safe_mode() -> None:
    """Before, the run began inside the keep-out, had no time to slew out
    of it before the first step was checked, and spent the whole run in
    safe mode."""
    config = _config()
    target = Pointing(
        config=config,
        obsid=1,
        name="Target",
        ra=80.0,
        dec=-20.0,
        merit=100.0,
        ss_min=600,
        ss_max=3600,
    )
    target.exptime = 3600
    ditl = create_ditl(config, [target], BEGIN, END)
    ditl.step_size = 10
    assert ditl.calc()
    modes = {ACSMode(int(mode)) for mode in ditl.mode}
    assert ACSMode.SAFE not in modes
    assert ACSMode.SCIENCE in modes
    assert ditl.validate_plan_matches_execution() == []
