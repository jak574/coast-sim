from __future__ import annotations

from pydantic import BaseModel

from .battery import Battery
from .constraint import Constraint
from .groundstation import GroundStationRegistry
from .instrument import Payload
from .solar_panel import SolarPanelSet
from .spacecraft_bus import SpacecraftBus


class Config(BaseModel):
    """
    Configuration class for the spacecraft and its subsystems.

    Constraints can be defined at the spacecraft_bus and/or payload level.
    The top-level constraint field is maintained for backward compatibility.
    """

    name: str = "Default Config"
    spacecraft_bus: SpacecraftBus
    solar_panel: SolarPanelSet
    payload: Payload
    battery: Battery
    constraint: Constraint | None = None
    ground_stations: GroundStationRegistry
