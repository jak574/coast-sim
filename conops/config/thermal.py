from pydantic import Field, model_validator

from ._base import ConfigModel
from .power import PowerDraw


class Heater(ConfigModel):
    """Simple model of a spacecraft heater that draws power depending on the mode.

    Heaters typically draw more power during eclipse when there is no solar heating,
    requiring active thermal management to maintain temperature.
    """

    name: str = Field(description="Name/identifier for the heater")
    power_draw: PowerDraw = Field(
        description="Power draw characteristics for the heater"
    )

    def power(self, mode: int | None = None, in_eclipse: bool = False) -> float:
        """Get the heater power in the given mode and eclipse state.

        Args:
            mode: Operational mode (None for nominal)
            in_eclipse: Whether spacecraft is in eclipse

        Returns:
            Heater power draw in watts
        """
        return self.power_draw.power(mode, in_eclipse=in_eclipse)


class ThermalNode(ConfigModel):
    """A part of the spacecraft with a heat capacity and one temperature that
    does not itself radiate to space, such as a mirror, a structure or an
    instrument.

    It exchanges heat with other nodes, and with the radiators that have a
    thermal mass, through :class:`ThermalCoupling`.
    """

    name: str = Field(description="Name of the node, unique among nodes and radiators")
    thermal_mass_j_per_k: float = Field(
        gt=0.0, allow_inf_nan=False, description="Heat capacity of the node, in J/K"
    )
    temperature_k: float = Field(
        default=293.0,
        gt=0.0,
        allow_inf_nan=False,
        description="Temperature at the start of a run, in Kelvin",
    )
    heat_load_w: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Constant heat generated in the node, in watts",
    )
    subsystem: str | None = Field(
        default=None,
        description=(
            "Whose electrical power heats the node: 'spacecraft_bus', 'payload' "
            "or the name of an instrument. Required when "
            "subsystem_power_fraction is above zero"
        ),
    )
    subsystem_power_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the electrical power of ``subsystem`` that becomes "
            "heat in the node, added to heat_load_w at each step"
        ),
    )

    @model_validator(mode="after")
    def _power_fraction_needs_a_subsystem(self) -> "ThermalNode":
        if self.subsystem_power_fraction > 0.0 and self.subsystem is None:
            raise ValueError("subsystem_power_fraction needs a subsystem")
        return self

    def internal_heat_w(self, subsystem_power_w: float) -> float:
        """Heat generated in the node, in watts, when its subsystem draws
        ``subsystem_power_w`` of electrical power."""
        return self.heat_load_w + self.subsystem_power_fraction * subsystem_power_w


class ThermalCoupling(ConfigModel):
    """A conductive path between two nodes.

    The heat flow from the first to the second is ``conductance_w_per_k``
    times their temperature difference.
    """

    between: tuple[str, str] = Field(
        description=(
            "Names of the two nodes joined. Each is a ThermalNode or a radiator "
            "that has a thermal mass"
        )
    )
    conductance_w_per_k: float = Field(
        gt=0.0, allow_inf_nan=False, description="Conductance of the path, in W/K"
    )

    @model_validator(mode="after")
    def _joins_two_different_nodes(self) -> "ThermalCoupling":
        if self.between[0] == self.between[1]:
            raise ValueError("a thermal coupling joins two different nodes")
        return self


class ThermalNetwork(ConfigModel):
    """Interior thermal nodes and the conductive paths among them and the
    radiators that have a thermal mass."""

    nodes: list[ThermalNode] = Field(default_factory=list)
    couplings: list[ThermalCoupling] = Field(default_factory=list)
    start_at_equilibrium: bool = Field(
        default=False,
        description=(
            "Start a run at the steady temperatures of the first step's "
            "attitude and power, not at the configured temperatures. A group "
            "of coupled nodes that contains no radiator able to emit has no "
            "steady temperature and keeps its configured temperatures"
        ),
    )
