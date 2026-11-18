"""Checking the requirements a mission configures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .base import RequirementEvidence, RequirementReport

if TYPE_CHECKING:
    from ..config.config import MissionConfig
    from ..ditl.telemetry import Housekeeping
    from ..targets.plan_entry import PlanEntry


def check_requirements(
    config: MissionConfig,
    entries: Sequence[PlanEntry],
    housekeeping: Sequence[Housekeeping] | None = None,
) -> RequirementReport:
    """Check every requirement of ``config`` against a plan's entries and,
    where given, the housekeeping telemetry of the run that executed it.

    Without telemetry, the requirements that need it are reported as not
    evaluated.
    """
    evidence = RequirementEvidence(
        config=config, entries=list(entries), housekeeping=housekeeping
    )
    return RequirementReport(
        results=tuple(
            requirement.evaluate(evidence) for requirement in config.requirements
        )
    )
