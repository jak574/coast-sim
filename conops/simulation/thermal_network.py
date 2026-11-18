"""Temperatures of a run's thermal nodes.

A thermal node has a heat capacity and one temperature. Two kinds exist: a
radiator with ``thermal_mass_j_per_k`` set, which absorbs Sun and Earth flux
and emits to space, and an interior :class:`~conops.config.ThermalNode`,
which does neither. Nodes exchange heat through the conductances of
:class:`~conops.config.ThermalCoupling`. The configuration holds the nodes'
properties; the temperatures of a run are held here, so that simulations
sharing one configuration do not share temperatures.

Each node obeys::

    C dT/dt = internal + absorbed + sum_j G_ij (T_j - T) - emitted(T)

A radiator that is coupled to nothing is advanced by
:meth:`~conops.config.Radiator.temperature_after`. Coupled nodes are advanced
together by the backward Euler method, solved by Newton's iteration, which is
stable for any step length and any conductance.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ..config.radiator import (
    BUS_SUBSYSTEM,
    PAYLOAD_SUBSYSTEM,
    STEFAN_BOLTZMANN_W_PER_M2_K4,
    Radiator,
)
from ..config.thermal import ThermalNetwork, ThermalNode

Exposure = tuple[float, float, float]
"""Sun, Earth infrared and albedo exposure of one radiator (see
:meth:`~conops.config.Radiator.earth_loads`)."""

_FloatArray = npt.NDArray[np.float64]

# A step is split so that no part is longer than this fraction of the shortest
# time constant among the coupled nodes, up to a fixed number of parts.
_MAX_TIME_CONSTANT_FRACTION = 0.25
_MAX_SUBSTEPS = 64
_NEWTON_ITERATIONS = 100
_CONVERGED_K = 1e-9
# Newton's iteration may not lower a temperature below this fraction of its
# value in one iteration, which keeps every temperature positive.
_MIN_NEWTON_RATIO = 0.5


class ThermalNetworkModel:
    """Integrates the temperature of every thermal node of a spacecraft.

    ``advance`` is called once per simulation step with the radiator
    exposures and the electrical powers of that step. The inputs of a step
    are held constant until the next one, so the temperature reported at a
    step is the result of the inputs of the steps before it.
    """

    def __init__(
        self,
        radiators: Sequence[Radiator],
        network: ThermalNetwork,
        power_sources: Collection[str] = (BUS_SUBSYSTEM, PAYLOAD_SUBSYSTEM),
    ) -> None:
        self._radiators = list(radiators)
        self._network = network
        # Node order: the radiators that have a thermal mass, then the
        # interior nodes.
        self._radiator_index = [
            index
            for index, radiator in enumerate(self._radiators)
            if radiator.thermal_mass_j_per_k is not None
        ]
        self._interior: list[ThermalNode] = list(network.nodes)
        thermal_radiators = [self._radiators[i] for i in self._radiator_index]
        self._names = [r.name for r in thermal_radiators] + [
            node.name for node in self._interior
        ]
        self._sources: list[tuple[str | None, float, float]] = [
            (r.subsystem, r.subsystem_power_fraction, r.heat_load_w)
            for r in thermal_radiators
        ] + [
            (node.subsystem, node.subsystem_power_fraction, node.heat_load_w)
            for node in self._interior
        ]
        for name, (subsystem, fraction, _) in zip(
            self._names, self._sources, strict=True
        ):
            if fraction > 0.0 and subsystem not in power_sources:
                raise ValueError(
                    f"thermal node {name!r} takes heat from {subsystem!r}, which "
                    f"is not one of {sorted(power_sources)}"
                )

        self._capacity = np.array(
            [float(r.thermal_mass_j_per_k or 0.0) for r in thermal_radiators]
            + [node.thermal_mass_j_per_k for node in self._interior],
            dtype=np.float64,
        )
        self._temperature = np.array(
            [r.radiator_temperature_k for r in thermal_radiators]
            + [node.temperature_k for node in self._interior],
            dtype=np.float64,
        )
        self._radiative = np.array(
            [
                r.radiating_area_m2 * r.emissivity * STEFAN_BOLTZMANN_W_PER_M2_K4
                for r in thermal_radiators
            ]
            + [0.0] * len(self._interior),
            dtype=np.float64,
        )
        self._sink4 = np.array(
            [r.sink_temperature_k**4 for r in thermal_radiators]
            + [0.0] * len(self._interior),
            dtype=np.float64,
        )

        count = len(self._names)
        position = {name: index for index, name in enumerate(self._names)}
        # The matrix L with (L T)_i = sum_j G_ij (T_i - T_j)
        self._laplacian = np.zeros((count, count), dtype=np.float64)
        group = list(range(count))

        def root(index: int) -> int:
            while group[index] != index:
                index = group[index]
            return index

        for coupling in network.couplings:
            first, second = (position[name] for name in coupling.between)
            conductance = coupling.conductance_w_per_k
            self._laplacian[first, first] += conductance
            self._laplacian[second, second] += conductance
            self._laplacian[first, second] -= conductance
            self._laplacian[second, first] -= conductance
            group[root(first)] = root(second)
        members: dict[int, list[int]] = {}
        for index in range(count):
            members.setdefault(root(index), []).append(index)
        # Radiators coupled to nothing, and the groups advanced together.
        self._alone = [
            m[0]
            for m in members.values()
            if len(m) == 1 and m[0] < len(self._radiator_index)
        ]
        self._groups = [
            np.array(m, dtype=np.intp)
            for m in members.values()
            if not (len(m) == 1 and m[0] < len(self._radiator_index))
        ]

        self._utime: float | None = None
        self._gain_w = np.zeros(count, dtype=np.float64)
        self._internal_w = np.zeros(count, dtype=np.float64)
        self._exposures: list[Exposure] = []

    @classmethod
    def for_spacecraft(
        cls,
        radiators: Sequence[Radiator],
        network: ThermalNetwork,
        power_sources: Collection[str] = (BUS_SUBSYSTEM, PAYLOAD_SUBSYSTEM),
    ) -> ThermalNetworkModel | None:
        """The model for a spacecraft, or None where it has no thermal node."""
        if not network.nodes and all(r.thermal_mass_j_per_k is None for r in radiators):
            return None
        return cls(radiators, network, power_sources)

    @property
    def radiator_temperatures_k(self) -> dict[str, float]:
        """Current temperature of each radiator that has a thermal mass, by
        radiator name."""
        split = len(self._radiator_index)
        return dict(
            zip(self._names[:split], self._temperature[:split].tolist(), strict=True)
        )

    @property
    def node_temperatures_k(self) -> dict[str, float]:
        """Current temperature of each interior node, by node name."""
        split = len(self._radiator_index)
        return dict(
            zip(self._names[split:], self._temperature[split:].tolist(), strict=True)
        )

    def advance(
        self,
        utime: float,
        exposures: Sequence[Exposure],
        power_w: Mapping[str, float],
    ) -> None:
        """Move the temperatures to ``utime`` under the inputs of the last
        step, then hold this step's inputs for the next call.

        ``exposures`` has one entry per configured radiator, in configuration
        order. ``power_w`` gives the electrical power of each heat source by
        name; a source that is absent draws nothing.

        The first call moves nothing. Where the network starts at
        equilibrium, it sets the steady temperatures of its inputs.
        """
        if len(exposures) != len(self._radiators):
            raise ValueError(
                f"{len(exposures)} radiator exposures were given for "
                f"{len(self._radiators)} radiators"
            )
        if self._utime is not None:
            self._step(utime - self._utime)
        self._exposures = list(exposures)
        for index, (subsystem, fraction, constant_w) in enumerate(self._sources):
            source_w = power_w.get(subsystem, 0.0) if subsystem is not None else 0.0
            self._internal_w[index] = constant_w + fraction * source_w
        self._gain_w = self._internal_w.copy()
        for index, radiator_index in enumerate(self._radiator_index):
            self._gain_w[index] += self._radiators[radiator_index].absorbed_power_w(
                *self._exposures[radiator_index]
            )
        if self._utime is None and self._network.start_at_equilibrium:
            self._settle()
        self._utime = utime

    def heat_dissipation_w(self, exposures: Sequence[Exposure]) -> float:
        """Net heat flow of all radiators, in watts: those with a thermal mass
        at their current temperature, the others at their fixed temperature."""
        temperature = dict(
            zip(self._radiator_index, self._temperature.tolist(), strict=False)
        )
        return sum(
            radiator.heat_dissipation_w(
                sun, earth, temperature.get(index), albedo_exposure=albedo
            )
            for index, (radiator, (sun, earth, albedo)) in enumerate(
                zip(self._radiators, exposures, strict=True)
            )
        )

    def _step(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        for index in self._alone:
            radiator_index = self._radiator_index[index]
            self._temperature[index] = self._radiators[
                radiator_index
            ].temperature_after(
                float(self._temperature[index]),
                seconds,
                float(self._internal_w[index]),
                *self._exposures[radiator_index],
            )
        for group in self._groups:
            temperature = self._temperature[group]
            rate = (
                np.diag(self._laplacian)[group]
                + 4.0 * self._radiative[group] * temperature**3
            ) / self._capacity[group]
            parts = min(
                _MAX_SUBSTEPS,
                max(
                    1,
                    int(np.ceil(rate.max() * seconds / _MAX_TIME_CONSTANT_FRACTION)),
                ),
            )
            for _ in range(parts):
                temperature = self._solve(group, temperature, seconds / parts)
            self._temperature[group] = temperature

    def _settle(self) -> None:
        """Set the steady temperatures of the held inputs, where they exist."""
        for index in self._alone:
            radiator_index = self._radiator_index[index]
            steady = self._radiators[radiator_index].equilibrium_temperature_k(
                float(self._internal_w[index]), *self._exposures[radiator_index]
            )
            if np.isfinite(steady):
                self._temperature[index] = steady
        for group in self._groups:
            if np.any(self._radiative[group] > 0.0):
                self._temperature[group] = self._solve(
                    group, self._temperature[group], None
                )

    def _solve(
        self, group: npt.NDArray[np.intp], start: _FloatArray, seconds: float | None
    ) -> _FloatArray:
        """The temperatures of ``group`` a backward Euler step of ``seconds``
        after ``start``, or the steady temperatures where ``seconds`` is None."""
        laplacian = self._laplacian[np.ix_(group, group)]
        radiative = self._radiative[group]
        storage = (
            np.zeros(len(group), dtype=np.float64)
            if seconds is None
            else self._capacity[group] / seconds
        )
        temperature = start.copy()
        for _ in range(_NEWTON_ITERATIONS):
            residual = (
                storage * (temperature - start)
                + laplacian @ temperature
                + radiative * (temperature**4 - self._sink4[group])
                - self._gain_w[group]
            )
            jacobian = laplacian + np.diag(storage + 4.0 * radiative * temperature**3)
            updated = np.maximum(
                temperature - np.linalg.solve(jacobian, residual),
                _MIN_NEWTON_RATIO * temperature,
            )
            converged = bool(np.max(np.abs(updated - temperature)) < _CONVERGED_K)
            temperature = updated
            if converged:
                break
        return temperature
