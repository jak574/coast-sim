"""Telemetry limit requirement: a recorded quantity stays within limits."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field, field_validator, model_validator

from ..common.enums import ACSMode
from .base import (
    Requirement,
    RequirementEvidence,
    RequirementResult,
    RequirementViolation,
)

if TYPE_CHECKING:
    from ..ditl.telemetry import Housekeeping


class TelemetryLimitRequirement(Requirement):
    """A quantity in the housekeeping telemetry stays at or above a minimum,
    at or below a maximum, or both, at every step of a run.

    ``quantity`` is the name of a housekeeping field, such as
    ``battery_level`` or ``recorder_fill_fraction``. For a field that holds a
    mapping, such as ``radiator_temperatures_k``, ``key`` selects the entry.
    Samples where the quantity is not recorded are not checked.

    Consecutive samples outside the limits make one violation, reported at
    its first sample with the value furthest outside the limit.
    """

    kind: ClassVar[str] = "telemetry_limit"

    quantity: str = Field(description="The housekeeping field checked")
    key: str | None = Field(
        default=None,
        description="The entry checked, for a field that holds a mapping",
    )
    minimum: float | None = Field(
        default=None, allow_inf_nan=False, description="The least value allowed"
    )
    maximum: float | None = Field(
        default=None, allow_inf_nan=False, description="The greatest value allowed"
    )
    modes: list[str] | None = Field(
        default=None,
        description=(
            "The ACS modes, by name, in which the limits apply; None for every mode"
        ),
    )

    @field_validator("modes")
    @classmethod
    def _modes_are_acs_modes(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        unknown = [name for name in value if name not in ACSMode.__members__]
        if unknown:
            raise ValueError(f"unknown ACS modes {unknown}")
        return value

    @model_validator(mode="after")
    def _has_a_limit(self) -> TelemetryLimitRequirement:
        if self.minimum is None and self.maximum is None:
            raise ValueError("a telemetry limit needs a minimum or a maximum")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum is above maximum")
        return self

    @property
    def _label(self) -> str:
        return self.quantity if self.key is None else f"{self.quantity}[{self.key!r}]"

    def _value(self, sample: Housekeeping) -> float | None:
        if not hasattr(sample, self.quantity):
            raise ValueError(
                f"requirement {self.name!r}: the telemetry has no quantity "
                f"{self.quantity!r}"
            )
        value: object = getattr(sample, self.quantity)
        if self.key is not None:
            value = value.get(self.key) if isinstance(value, Mapping) else None
        if value is None or not isinstance(value, (int, float)):
            return None
        return float(value)

    def _applies(self, sample: Housekeeping) -> bool:
        if self.modes is None:
            return True
        return (
            sample.acs_mode is not None and ACSMode(sample.acs_mode).name in self.modes
        )

    def _excess(self, value: float) -> tuple[float, float] | None:
        """How far ``value`` is outside the limits and the limit it breaks;
        None inside them."""
        if self.minimum is not None and value < self.minimum:
            return self.minimum - value, self.minimum
        if self.maximum is not None and value > self.maximum:
            return value - self.maximum, self.maximum
        return None

    def evaluate(self, evidence: RequirementEvidence) -> RequirementResult:
        if evidence.housekeeping is None:
            return self.not_evaluated("needs the telemetry of a run")
        checked = 0
        violations: list[RequirementViolation] = []
        # The violation in progress: first time, samples, worst value, its
        # excess and the limit it breaks.
        run: tuple[float, int, float, float, float] | None = None

        def close() -> None:
            nonlocal run
            if run is None:
                return
            start, samples, worst, _, limit = run
            side = "below" if worst < limit else "above"
            began = datetime.fromtimestamp(start, tz=timezone.utc).isoformat()
            violations.append(
                self.violation(
                    start,
                    f"{self._label} reaches {worst:.6g}, {side} the limit of "
                    f"{limit:.6g}, over {samples} sample(s) from {began}",
                    measured=worst,
                    limit=limit,
                )
            )
            run = None

        for sample in evidence.housekeeping:
            value = self._value(sample) if self._applies(sample) else None
            if value is None:
                close()
                continue
            checked += 1
            outside = self._excess(value)
            if outside is None:
                close()
                continue
            excess, limit = outside
            if run is None:
                run = (sample.timestamp.timestamp(), 1, value, excess, limit)
            elif excess > run[3]:
                run = (run[0], run[1] + 1, value, excess, limit)
            else:
                run = (run[0], run[1] + 1, run[2], run[3], run[4])
        close()
        return self.result(checked, violations)
