from __future__ import annotations

from pydantic import BaseModel, model_validator

from .battery import Battery
from .constraint import Constraint
from .groundstation import GroundStationRegistry
from .instrument import Payload
from .solar_panel import SolarPanelSet
from .spacecraft_bus import SpacecraftBus

# Rebuild models to resolve forward references
SpacecraftBus.model_rebuild()
Payload.model_rebuild()


class Config(BaseModel):
    """
    Configuration class for the spacecraft and its subsystems.

    Constraints can be defined at the spacecraft_bus and/or payload level.
    If only one constraint is defined, it will be used as the default.
    The top-level constraint field is maintained for backward compatibility.
    """

    name: str = "Default Config"
    spacecraft_bus: SpacecraftBus
    solar_panel: SolarPanelSet
    payload: Payload
    battery: Battery
    constraint: Constraint | None = None
    ground_stations: GroundStationRegistry

    @model_validator(mode="after")
    def validate_constraints(self) -> Config:
        """Ensure at least one constraint is defined and set up defaults.

        If only one constraint exists (either in spacecraft_bus, payload, or Config),
        use that as the default constraint.
        """
        bus_constraint = self.spacecraft_bus.constraint
        payload_constraint = self.payload.constraint
        config_constraint = self.constraint

        # Count how many constraints are defined
        defined_constraints = [
            c
            for c in [bus_constraint, payload_constraint, config_constraint]
            if c is not None
        ]

        if not defined_constraints:
            raise ValueError(
                "At least one constraint must be defined (in spacecraft_bus, payload, or Config)"
            )

        # If only one constraint is defined, use it as the default everywhere
        if len(defined_constraints) == 1:
            default_constraint = defined_constraints[0]
            if self.spacecraft_bus.constraint is None:
                self.spacecraft_bus.constraint = default_constraint
            if self.payload.constraint is None:
                self.payload.constraint = default_constraint
            if self.constraint is None:
                self.constraint = default_constraint
        else:
            # Multiple constraints defined - ensure backward compatibility
            # If config.constraint is defined, use it as fallback
            if config_constraint is not None:
                if self.spacecraft_bus.constraint is None:
                    self.spacecraft_bus.constraint = config_constraint
                if self.payload.constraint is None:
                    self.payload.constraint = config_constraint
            else:
                # No config.constraint, so use spacecraft_bus or payload as available
                if self.constraint is None:
                    # Use spacecraft_bus constraint as the default if available
                    self.constraint = bus_constraint or payload_constraint

        return self
