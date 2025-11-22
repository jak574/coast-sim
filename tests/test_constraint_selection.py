"""Tests for constraint selection in ACS based on mode."""

from unittest.mock import Mock, patch

import pytest

from conops.acs import ACS
from conops.battery import Battery
from conops.common import ACSMode
from conops.config import Config
from conops.constraint import Constraint
from conops.groundstation import GroundStationRegistry
from conops.instrument import Payload
from conops.solar_panel import SolarPanelSet
from conops.spacecraft_bus import SpacecraftBus


@pytest.fixture
def bus_constraint():
    """Create a mock constraint for spacecraft bus."""
    constraint = Mock(spec=Constraint)
    ephem = Mock()
    ephem.step_size = 60
    # Mock earth and sun arrays
    earth_mock = Mock()
    earth_mock.ra = Mock(deg=0.0)
    earth_mock.dec = Mock(deg=0.0)
    ephem.earth = [earth_mock]
    ephem.index = Mock(return_value=0)
    constraint.ephem = ephem
    constraint.in_eclipse = Mock(return_value=False)
    constraint.inoccult = Mock(return_value=False)
    return constraint


@pytest.fixture
def payload_constraint():
    """Create a mock constraint for payload."""
    constraint = Mock(spec=Constraint)
    ephem = Mock()
    ephem.step_size = 60
    # Mock earth and sun arrays
    earth_mock = Mock()
    earth_mock.ra = Mock(deg=0.0)
    earth_mock.dec = Mock(deg=0.0)
    ephem.earth = [earth_mock]
    ephem.index = Mock(return_value=0)
    constraint.ephem = ephem
    constraint.in_eclipse = Mock(return_value=False)
    constraint.inoccult = Mock(return_value=False)
    return constraint


@pytest.fixture
def config_with_separate_constraints(bus_constraint, payload_constraint):
    """Create a config with separate bus and payload constraints."""
    spacecraft_bus = Mock(spec=SpacecraftBus)
    spacecraft_bus.constraint = bus_constraint
    spacecraft_bus.attitude_control = Mock()
    spacecraft_bus.attitude_control.predict_slew = Mock(
        return_value=(0.0, (Mock(), Mock()))
    )
    spacecraft_bus.attitude_control.slew_time = Mock(return_value=10.0)

    payload = Mock(spec=Payload)
    payload.constraint = payload_constraint

    solar_panel = Mock(spec=SolarPanelSet)
    solar_panel.optimal_charging_pointing = Mock(return_value=(45.0, 23.5))

    battery = Mock(spec=Battery)
    ground_stations = Mock(spec=GroundStationRegistry)

    # Use bus constraint as the top-level constraint for initialization
    config = Config(
        name="Test Config",
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=bus_constraint,
        ground_stations=ground_stations,
    )
    return config


class TestConstraintSelection:
    """Test constraint selection based on ACS mode."""

    def test_acs_stores_bus_and_payload_constraints(
        self, config_with_separate_constraints, bus_constraint, payload_constraint
    ):
        """Test that ACS stores references to both bus and payload constraints."""
        with patch("conops.acs.PassTimes"):
            acs = ACS(
                constraint=bus_constraint, config=config_with_separate_constraints
            )

            assert acs.bus_constraint is bus_constraint
            assert acs.payload_constraint is payload_constraint

    def test_normal_mode_uses_payload_constraint(
        self, config_with_separate_constraints, bus_constraint, payload_constraint
    ):
        """Test that normal science mode uses payload constraint."""
        with patch("conops.acs.PassTimes"):
            acs = ACS(
                constraint=bus_constraint, config=config_with_separate_constraints
            )
            acs.acsmode = ACSMode.SCIENCE

            # Simulate pointing call which triggers constraint selection
            acs.pointing(1000.0)

            # Should be using payload constraint for science operations
            assert acs.constraint is payload_constraint

    def test_safe_mode_uses_bus_constraint(
        self, config_with_separate_constraints, bus_constraint, payload_constraint
    ):
        """Test that SAFE mode uses spacecraft bus constraint."""
        with patch("conops.acs.PassTimes"):
            acs = ACS(
                constraint=bus_constraint, config=config_with_separate_constraints
            )

            # Enter safe mode
            acs.request_safe_mode(1000.0)
            acs.pointing(1000.0)

            # Should be using bus constraint in safe mode
            assert acs.in_safe_mode
            assert acs.constraint is bus_constraint

    def test_slewing_mode_uses_payload_constraint(
        self, config_with_separate_constraints, bus_constraint, payload_constraint
    ):
        """Test that SLEWING mode uses payload constraint (not bus)."""
        with patch("conops.acs.PassTimes"):
            acs = ACS(
                constraint=bus_constraint, config=config_with_separate_constraints
            )

            # Set up a slew
            acs.current_slew = Mock()
            acs.current_slew.is_slewing = Mock(return_value=True)
            acs.current_slew.obstype = "PPT"

            acs.pointing(1000.0)

            # Even when slewing, should use payload constraint for science slews
            assert acs.constraint is payload_constraint

    def test_fallback_to_bus_constraint_when_no_payload(self):
        """Test that ACS falls back to bus constraint if no payload constraint exists."""
        bus_constraint = Mock(spec=Constraint)
        ephem = Mock()
        ephem.step_size = 60
        # Mock earth and sun arrays
        earth_mock = Mock()
        earth_mock.ra = Mock(deg=0.0)
        earth_mock.dec = Mock(deg=0.0)
        ephem.earth = [earth_mock]
        ephem.index = Mock(return_value=0)
        bus_constraint.ephem = ephem
        bus_constraint.in_eclipse = Mock(return_value=False)
        bus_constraint.inoccult = Mock(return_value=False)

        spacecraft_bus = Mock(spec=SpacecraftBus)
        spacecraft_bus.constraint = bus_constraint
        spacecraft_bus.attitude_control = Mock()

        payload = Mock(spec=Payload)
        payload.constraint = None  # No payload constraint

        solar_panel = Mock(spec=SolarPanelSet)
        battery = Mock(spec=Battery)
        ground_stations = Mock(spec=GroundStationRegistry)

        config = Config(
            spacecraft_bus=spacecraft_bus,
            solar_panel=solar_panel,
            payload=payload,
            battery=battery,
            constraint=bus_constraint,
            ground_stations=ground_stations,
        )

        with patch("conops.acs.PassTimes"):
            acs = ACS(constraint=bus_constraint, config=config)
            acs.acsmode = ACSMode.SCIENCE

            acs.pointing(1000.0)

            # Should fall back to bus constraint
            assert acs.constraint is bus_constraint
