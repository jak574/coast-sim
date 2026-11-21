"""The requirements framework: what a requirement is, what it is checked
against, and what a check reports.

A :class:`Requirement` is a statement about a mission that a plan or a run
either meets or violates, such as "a slew of up to 30 degrees completes within
300 s" or "the battery stays above 40% charge". Requirements are configured in
``MissionConfig.requirements``. They are checks on what the simulation does,
not constraints on it: a scheduler does not change a plan to meet one.

A requirement is evaluated against :class:`RequirementEvidence`, the plan
entries and, after a run, the telemetry. It returns a
:class:`RequirementResult` that lists each :class:`RequirementViolation`.

New kinds of requirement are added by subclassing :class:`Requirement`; see
its documentation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from ..config._base import ConfigModel

if TYPE_CHECKING:
    from ..config.config import MissionConfig
    from ..ditl.telemetry import Housekeeping
    from ..targets.plan_entry import PlanEntry

KIND_KEY = "kind"
"""The key that names a requirement's kind in a configuration file."""

_REGISTRY: dict[str, type[Requirement]] = {}


@dataclass(frozen=True)
class RequirementEvidence:
    """What a requirement is checked against.

    ``housekeeping`` is None for a plan that has not been executed. A
    requirement that needs telemetry then reports that it was not evaluated.
    """

    config: MissionConfig
    entries: Sequence[PlanEntry]
    housekeeping: Sequence[Housekeeping] | None = None


class RequirementViolation(BaseModel):
    """One place where a requirement was not met."""

    model_config = ConfigDict(frozen=True)

    requirement: str
    """The name of the requirement violated."""
    utime: float
    """When the violation happened, or began, as a Unix time."""
    message: str
    obsid: int | None = None
    """The observation concerned, where there is one."""
    measured: float | None = None
    """The value that broke the limit, where the requirement has one."""
    limit: float | None = None
    """The limit it broke."""

    def __str__(self) -> str:
        return self.message


class RequirementResult(BaseModel):
    """The outcome of checking one requirement."""

    model_config = ConfigDict(frozen=True)

    requirement: str
    kind: str
    description: str = ""
    evaluated: bool = True
    """False where the evidence the requirement needs was absent."""
    note: str = ""
    """Why the requirement was not evaluated."""
    checked: int = 0
    """How many items, such as slews or telemetry samples, were examined."""
    violations: tuple[RequirementViolation, ...] = ()

    @property
    def status(self) -> Literal["met", "violated", "not evaluated"]:
        if not self.evaluated:
            return "not evaluated"
        return "violated" if self.violations else "met"


class RequirementReport(BaseModel):
    """The outcome of checking every configured requirement."""

    model_config = ConfigDict(frozen=True)

    results: tuple[RequirementResult, ...] = ()

    @property
    def violations(self) -> list[RequirementViolation]:
        """Every violation, in time order."""
        return sorted(
            (v for result in self.results for v in result.violations),
            key=lambda violation: violation.utime,
        )

    @property
    def met(self) -> bool:
        """True where every requirement was evaluated and none was violated."""
        return all(result.status == "met" for result in self.results)

    def result(self, requirement: str) -> RequirementResult:
        """The result of the requirement of that name."""
        for result in self.results:
            if result.requirement == requirement:
                return result
        raise KeyError(requirement)

    def __str__(self) -> str:
        if not self.results:
            return "No requirements configured"
        lines = [
            f"{'Requirement':<32} {'Status':<14} {'Checked':>8} {'Violations':>10}"
        ]
        lines.extend(
            f"{result.requirement:<32} {result.status:<14} {result.checked:>8} "
            f"{len(result.violations):>10}"
            for result in self.results
        )
        return "\n".join(lines)


class Requirement(ConfigModel):
    """A statement about the mission that a plan or a run meets or violates.

    To add a kind of requirement, subclass this, name the kind and implement
    :meth:`evaluate`::

        class MaximumIdleFraction(Requirement):
            kind: ClassVar[str] = "maximum_idle_fraction"

            maximum: float

            def evaluate(self, evidence: RequirementEvidence) -> RequirementResult:
                samples = evidence.housekeeping
                if samples is None:
                    return self.not_evaluated("needs the telemetry of a run")
                idle = sum(s.acs_mode == ACSMode.IDLE for s in samples)
                fraction = idle / max(1, len(samples))
                violations = []
                if fraction > self.maximum:
                    violations.append(
                        self.violation(
                            samples[0].timestamp.timestamp(),
                            f"idle for {fraction:.0%} of the run",
                            measured=fraction,
                            limit=self.maximum,
                        )
                    )
                return self.result(len(samples), violations)

    Defining the subclass registers its ``kind``, after which a configuration
    file can name it: ``{"kind": "maximum_idle_fraction", "name": ...,
    "maximum": 0.1}``. The module that defines the subclass has to be imported
    before such a file is loaded.
    """

    kind: ClassVar[str] = ""
    """The name of this kind of requirement in configuration files. Empty for
    a base class that is not itself a requirement."""

    name: str = Field(description="The requirement's name, unique in a mission")
    description: str = Field(
        default="", description="The requirement as the mission states it"
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        kind = cls.__dict__.get("kind", "")
        if not isinstance(kind, str) or not kind:
            return
        existing = _REGISTRY.get(kind)
        # Defining a class again, as a notebook does when a cell is run again,
        # replaces it; a different class may not take a kind already in use.
        if existing is not None and (existing.__module__, existing.__qualname__) != (
            cls.__module__,
            cls.__qualname__,
        ):
            raise TypeError(
                f"requirement kind {kind!r} is already used by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _REGISTRY[kind] = cls

    @model_validator(mode="before")
    @classmethod
    def _accept_own_kind(cls, data: object) -> object:
        if not isinstance(data, Mapping) or KIND_KEY not in data:
            return data
        if data[KIND_KEY] != cls.kind:
            raise ValueError(
                f"{cls.__name__} is of kind {cls.kind!r}, not {data[KIND_KEY]!r}"
            )
        return {key: value for key, value in data.items() if key != KIND_KEY}

    @model_serializer(mode="wrap")
    def _write_kind_first(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        fields: dict[str, object] = handler(self)
        return {KIND_KEY: self.kind, **fields}

    def evaluate(self, evidence: RequirementEvidence) -> RequirementResult:
        """Check the requirement against a plan and, where given, a run."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement evaluate()"
        )

    def violation(
        self,
        utime: float,
        message: str,
        *,
        obsid: int | None = None,
        measured: float | None = None,
        limit: float | None = None,
    ) -> RequirementViolation:
        """A violation of this requirement."""
        return RequirementViolation(
            requirement=self.name,
            utime=utime,
            message=message,
            obsid=obsid,
            measured=measured,
            limit=limit,
        )

    def result(
        self, checked: int, violations: Sequence[RequirementViolation]
    ) -> RequirementResult:
        """The result of an evaluation that examined ``checked`` items."""
        return RequirementResult(
            requirement=self.name,
            kind=self.kind,
            description=self.description,
            checked=checked,
            violations=tuple(violations),
        )

    def not_evaluated(self, note: str) -> RequirementResult:
        """The result where the evidence the requirement needs is absent."""
        return RequirementResult(
            requirement=self.name,
            kind=self.kind,
            description=self.description,
            evaluated=False,
            note=note,
        )


def requirement_kinds() -> dict[str, type[Requirement]]:
    """The registered kinds of requirement, by kind name."""
    return dict(_REGISTRY)


def parse_requirement(value: object) -> Requirement:
    """A requirement from its configuration: an instance as it is, or a
    mapping whose ``kind`` names a registered kind."""
    if isinstance(value, Requirement):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("a requirement is a mapping with a 'kind'")
    kind = value.get(KIND_KEY)
    requirement_class = _REGISTRY.get(kind) if isinstance(kind, str) else None
    if requirement_class is None:
        raise ValueError(
            f"unknown requirement kind {kind!r}; known kinds: {sorted(_REGISTRY)}"
        )
    return requirement_class.model_validate(value)
