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
def base_constraint():
    """Create a mock base constraint for config."""
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
def config_with_payload_constraint(base_constraint, payload_constraint):
    """Create a config with base constraint and payload override."""
    spacecraft_bus = Mock(spec=SpacecraftBus)
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

    config = Config(
        name="Test Config",
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=base_constraint,
        ground_stations=ground_stations,
    )
    return config


class TestConstraintSelection:
    """Test constraint selection based on ACS mode."""

    def test_acs_stores_base_and_payload_constraints(
        self, config_with_payload_constraint, base_constraint, payload_constraint
    ):
        """Test that ACS stores references to both base and payload constraints."""
        with patch("conops.acs.PassTimes"):
            acs = ACS(constraint=base_constraint, config=config_with_payload_constraint)

            assert acs.base_constraint is base_constraint
            assert acs.payload_constraint is payload_constraint

    def test_science_mode_uses_payload_constraint(
        self, config_with_payload_constraint, base_constraint, payload_constraint
    ):
        """Test that science mode uses payload constraint when defined."""
        with patch("conops.acs.PassTimes"):
            acs = ACS(constraint=base_constraint, config=config_with_payload_constraint)
            acs.acsmode = ACSMode.SCIENCE

            # Simulate pointing call which triggers constraint selection
            acs.pointing(1000.0)

            # Should be using payload constraint for science operations
            assert acs.constraint is payload_constraint

    def test_non_science_mode_uses_base_constraint(
        self, config_with_payload_constraint, base_constraint, payload_constraint
    ):
        """Test that non-science modes use base constraint."""
        with patch("conops.acs.PassTimes"):
            acs = ACS(constraint=base_constraint, config=config_with_payload_constraint)

            # Directly test the _select_active_constraint method
            # Test PASS mode
            acs.acsmode = ACSMode.PASS
            acs._select_active_constraint(1000.0)
            assert acs.constraint is base_constraint

            # Test SLEWING mode
            acs.acsmode = ACSMode.SLEWING
            acs._select_active_constraint(1000.0)
            assert acs.constraint is base_constraint

            # Test CHARGING mode
            acs.acsmode = ACSMode.CHARGING
            acs._select_active_constraint(1000.0)
            assert acs.constraint is base_constraint

            # Test SAA mode
            acs.acsmode = ACSMode.SAA
            acs._select_active_constraint(1000.0)
            assert acs.constraint is base_constraint

    def test_science_mode_without_payload_constraint(self):
        """Test that science mode uses base constraint when no payload constraint exists."""
        base_constraint = Mock(spec=Constraint)
        ephem = Mock()
        ephem.step_size = 60
        # Mock earth and sun arrays
        earth_mock = Mock()
        earth_mock.ra = Mock(deg=0.0)
        earth_mock.dec = Mock(deg=0.0)
        ephem.earth = [earth_mock]
        ephem.index = Mock(return_value=0)
        base_constraint.ephem = ephem
        base_constraint.in_eclipse = Mock(return_value=False)
        base_constraint.inoccult = Mock(return_value=False)

        spacecraft_bus = Mock(spec=SpacecraftBus)
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
            constraint=base_constraint,
            ground_stations=ground_stations,
        )

        with patch("conops.acs.PassTimes"):
            acs = ACS(constraint=base_constraint, config=config)
            acs.acsmode = ACSMode.SCIENCE

            acs.pointing(1000.0)

            # Should use base constraint
            assert acs.constraint is base_constraint
