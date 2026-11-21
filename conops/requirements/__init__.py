"""Mission requirements: statements a plan or a run meets or violates, how
they are configured, and how they are checked.

See :mod:`conops.requirements.base` for the framework and for how to add a
kind of requirement.
"""

from .base import (
    Requirement,
    RequirementEvidence,
    RequirementReport,
    RequirementResult,
    RequirementViolation,
    parse_requirement,
    requirement_kinds,
)
from .evaluate import check_requirements
from .slew import BEYOND_THE_TABLE, SlewClass, SlewClassSummary, SlewTimeRequirement
from .telemetry import TelemetryLimitRequirement

__all__ = [
    "BEYOND_THE_TABLE",
    "Requirement",
    "RequirementEvidence",
    "RequirementReport",
    "RequirementResult",
    "RequirementViolation",
    "SlewClass",
    "SlewClassSummary",
    "SlewTimeRequirement",
    "TelemetryLimitRequirement",
    "check_requirements",
    "parse_requirement",
    "requirement_kinds",
]
