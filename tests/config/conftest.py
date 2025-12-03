"""Test fixtures for config subsystem tests."""

import pytest

from conops import (
    Battery,
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    Payload,
    SolarPanelSet,
    SpacecraftBus,
)
from conops.config.recorder import OnboardRecorder


@pytest.fixture
def minimal_config():
    name = "Test Config"
    spacecraft_bus = SpacecraftBus()
    solar_panel = SolarPanelSet()
    payload = Payload()
    battery = Battery()
    constraint = Constraint()
    ground_stations = GroundStationRegistry()
    recorder = OnboardRecorder()

    config = MissionConfig(
        name=name,
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=constraint,
        ground_stations=ground_stations,
        recorder=recorder,
    )

    return {
        "config": config,
        "spacecraft_bus": spacecraft_bus,
        "solar_panel": solar_panel,
        "payload": payload,
        "battery": battery,
        "constraint": constraint,
        "ground_stations": ground_stations,
        "recorder": recorder,
    }
