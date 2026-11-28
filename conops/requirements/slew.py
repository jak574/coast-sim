"""Slew time requirement: how long a slew of up to a given great-circle angle
may take, settled, as a mission's requirements state it.

The ACS model flies each slew by its own rates and accelerations; the
requirement is a check on what it flies. Each plan entry records the
great-circle angle between the pointings its slew joins beside the rotation
the slew actually makes (``slewdist``), which the roll lengthens, since
requirements are written in great-circle degrees.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config._base import ConfigModel
from .base import Requirement, RequirementEvidence, RequirementResult

if TYPE_CHECKING:
    from ..targets.plan_entry import PlanEntry

BEYOND_THE_TABLE = "beyond the table"
"""The class of a slew wider than the widest class."""

_ANGLE_TOLERANCE_DEG = 1e-9


class SlewClass(ConfigModel):
    """A class of slews: a slew of up to a great-circle angle completes,
    settled, within a time."""

    name: str = Field(description="The class's name, such as 'small' or 'large'")
    max_great_circle_deg: float = Field(
        gt=0.0,
        le=180.0,
        description=(
            "The widest slew of the class, as the great-circle angle between "
            "the pointings it joins, in degrees"
        ),
    )
    max_seconds: float = Field(
        gt=0.0, description="The most a slew of the class may take, settled, in seconds"
    )


class SlewClassSummary(BaseModel):
    """What a run's slews did against one slew class."""

    model_config = ConfigDict(frozen=True)

    name: str
    max_great_circle_deg: float | None
    """The class's widest slew; None beyond the table."""
    max_seconds: float | None
    """The class's allowance; None beyond the table."""
    slews: int
    longest_seconds: float
    """The longest slew of the class, settled."""
    violations: int


def _slews(entries: Iterable[PlanEntry]) -> list[PlanEntry]:
    return [e for e in entries if float(e.slewtime) > 0.0]


class SlewTimeRequirement(Requirement):
    """Every slew completes, settled, within the time its class allows.

    A slew belongs to the narrowest class wide enough for its great-circle
    angle. A slew wider than the widest class belongs to none and is not
    checked.
    """

    kind: ClassVar[str] = "slew_time"

    name: str = Field(
        default="Slew time", description="The requirement's name, unique in a mission"
    )
    classes: list[SlewClass] = Field(
        description="The slew classes, held in ascending angle", min_length=1
    )

    @field_validator("classes")
    @classmethod
    def _classes_in_ascending_angle(cls, value: list[SlewClass]) -> list[SlewClass]:
        angles = [c.max_great_circle_deg for c in value]
        if len(set(angles)) != len(angles):
            raise ValueError("slew classes must have distinct angles")
        if len({c.name for c in value}) != len(value):
            raise ValueError("slew classes must have distinct names")
        return sorted(value, key=lambda c: c.max_great_circle_deg)

    def slew_class(self, great_circle_deg: float) -> SlewClass | None:
        """The class a slew of ``great_circle_deg`` falls in: the narrowest
        class wide enough for it; None for a slew wider than the widest."""
        for slew_class in self.classes:
            if (
                great_circle_deg
                <= slew_class.max_great_circle_deg + _ANGLE_TOLERANCE_DEG
            ):
                return slew_class
        return None

    def evaluate(self, evidence: RequirementEvidence) -> RequirementResult:
        slews = _slews(evidence.entries)
        violations = []
        for entry in slews:
            angle = float(entry.great_circle_deg)
            seconds = float(entry.slewtime)
            slew_class = self.slew_class(angle)
            if slew_class is None or seconds <= slew_class.max_seconds:
                continue
            obsid = int(entry.obsid) if entry.obsid is not None else None
            violations.append(
                self.violation(
                    float(entry.begin),
                    f"slew of {angle:.1f} deg into {obsid} takes {seconds:.0f} s, "
                    f"over the {slew_class.max_seconds:.0f} s of its class "
                    f"{slew_class.name!r} (up to "
                    f"{slew_class.max_great_circle_deg:.0f} deg)",
                    obsid=obsid,
                    measured=seconds,
                    limit=slew_class.max_seconds,
                )
            )
        return self.result(len(slews), violations)

    def summarise(self, entries: Iterable[PlanEntry]) -> list[SlewClassSummary]:
        """One row per class, in ascending angle, with the slews of the class,
        the longest of them and how many broke the allowance; a last row for
        slews wider than the widest class, where there are any."""
        by_class: dict[str, list[PlanEntry]] = {c.name: [] for c in self.classes}
        beyond: list[PlanEntry] = []
        for entry in _slews(entries):
            slew_class = self.slew_class(float(entry.great_circle_deg))
            if slew_class is None:
                beyond.append(entry)
            else:
                by_class[slew_class.name].append(entry)
        rows = [
            SlewClassSummary(
                name=slew_class.name,
                max_great_circle_deg=slew_class.max_great_circle_deg,
                max_seconds=slew_class.max_seconds,
                slews=len(slews),
                longest_seconds=max((float(e.slewtime) for e in slews), default=0.0),
                violations=sum(
                    1 for e in slews if float(e.slewtime) > slew_class.max_seconds
                ),
            )
            for slew_class, slews in zip(self.classes, by_class.values(), strict=True)
        ]
        if beyond:
            rows.append(
                SlewClassSummary(
                    name=BEYOND_THE_TABLE,
                    max_great_circle_deg=None,
                    max_seconds=None,
                    slews=len(beyond),
                    longest_seconds=max(float(e.slewtime) for e in beyond),
                    violations=0,
                )
            )
        return rows
