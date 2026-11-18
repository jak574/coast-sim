"""Temperatures of the radiators that have a thermal mass, over one run.

A radiator with ``thermal_mass_j_per_k`` set is one thermal node: its heat
capacity stores the difference between the heat it receives (internal heat
and absorbed Sun and Earth flux) and the heat it emits, and its temperature
changes accordingly. The configuration holds the node's properties; the
temperatures of a run are held here, so that simulations sharing one
configuration do not share temperatures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..config.radiator import Radiator, RadiatorConfiguration

Exposure = tuple[float, float]
"""Sun and Earth exposure of one radiator, each a fraction from 0 to 1."""


class RadiatorThermalModel:
    """Integrates the temperature of every radiator that has a thermal mass.

    ``advance`` is called once per simulation step with the exposures and
    subsystem powers of that step. The inputs of a step are held constant
    until the next one, so the temperature reported at a step is the result
    of the inputs of the steps before it.
    """

    def __init__(self, radiators: Sequence[Radiator]) -> None:
        self._radiators = list(radiators)
        self._temperature_k = {
            index: radiator.radiator_temperature_k
            for index, radiator in enumerate(self._radiators)
            if radiator.thermal_mass_j_per_k is not None
        }
        self._utime: float | None = None
        self._heat_w: dict[int, float] = {}
        self._exposures: list[Exposure] = []

    @classmethod
    def for_radiators(
        cls, configuration: RadiatorConfiguration
    ) -> RadiatorThermalModel | None:
        """The model for a configuration, or None where no radiator has a
        thermal mass."""
        model = cls(configuration.radiators)
        return model if model._temperature_k else None

    @property
    def temperatures_k(self) -> dict[str, float]:
        """Current temperature of each radiator that has a thermal mass, by
        radiator name."""
        return {
            self._radiators[index].name: temperature
            for index, temperature in self._temperature_k.items()
        }

    def advance(
        self,
        utime: float,
        exposures: Sequence[Exposure],
        subsystem_power_w: Mapping[str, float],
    ) -> None:
        """Move the temperatures to ``utime`` under the inputs of the last
        step, then hold this step's inputs for the next call.

        ``exposures`` has one entry per configured radiator, in configuration
        order. ``subsystem_power_w`` gives the electrical power of each
        subsystem by name; a subsystem that is absent draws nothing.
        """
        if len(exposures) != len(self._radiators):
            raise ValueError(
                f"{len(exposures)} radiator exposures were given for "
                f"{len(self._radiators)} radiators"
            )
        if self._utime is not None:
            seconds = utime - self._utime
            for index, temperature in self._temperature_k.items():
                sun, earth = self._exposures[index]
                self._temperature_k[index] = self._radiators[index].temperature_after(
                    temperature, seconds, self._heat_w[index], sun, earth
                )
        self._utime = utime
        self._exposures = list(exposures)
        self._heat_w = {
            index: self._radiators[index].internal_heat_w(
                subsystem_power_w.get(self._radiators[index].subsystem, 0.0)
            )
            for index in self._temperature_k
        }

    def heat_dissipation_w(self, exposures: Sequence[Exposure]) -> float:
        """Net heat flow of all radiators, in watts: those with a thermal mass
        at their current temperature, the others at their fixed temperature."""
        return sum(
            radiator.heat_dissipation_w(sun, earth, self._temperature_k.get(index))
            for index, (radiator, (sun, earth)) in enumerate(
                zip(self._radiators, exposures, strict=True)
            )
        )
