"""The telemetry limit requirement: a recorded quantity stays within limits."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from conops import MissionConfig, QueueDITL
from conops.common import ACSMode
from conops.ditl.telemetry import Housekeeping
from conops.requirements import (
    RequirementEvidence,
    TelemetryLimitRequirement,
)
from tests.scheduler.planning_scenario import BEGIN, make_config

START = datetime(2025, 11, 1, tzinfo=timezone.utc)


def _samples(values: list[float | None], **fields: object) -> list[Housekeeping]:
    return [
        Housekeeping(
            timestamp=START + timedelta(minutes=index),
            battery_level=value,
            acs_mode=ACSMode.SCIENCE if index % 2 else ACSMode.IDLE,
            **fields,
        )
        for index, value in enumerate(values)
    ]


def _check(requirement: TelemetryLimitRequirement, samples: list[Housekeeping] | None):
    return requirement.evaluate(
        RequirementEvidence(config=MissionConfig(), entries=[], housekeeping=samples)
    )


def test_consecutive_samples_outside_the_limit_are_one_violation() -> None:
    requirement = TelemetryLimitRequirement(
        name="Battery", quantity="battery_level", minimum=0.4
    )
    result = _check(requirement, _samples([0.9, 0.39, 0.3, 0.35, 0.5, None, 0.2]))
    assert result.checked == 6
    first, second = result.violations
    assert first.utime == (START + timedelta(minutes=1)).timestamp()
    assert first.measured == 0.3 and first.limit == 0.4
    assert "3 sample(s)" in first.message and "below" in first.message
    assert second.measured == 0.2
    assert result.status == "violated"
    assert _check(requirement, _samples([0.4, 0.8])).status == "met"


def test_a_maximum_and_a_mode_filter() -> None:
    requirement = TelemetryLimitRequirement(
        name="Battery in science",
        quantity="battery_level",
        maximum=0.8,
        modes=["SCIENCE"],
    )
    # Odd samples are in SCIENCE; the even ones are not checked.
    result = _check(requirement, _samples([0.99, 0.7, 0.99, 0.9]))
    assert result.checked == 2
    [violation] = result.violations
    assert violation.measured == 0.9 and "above" in violation.message


def test_a_mapping_field_is_checked_by_key() -> None:
    requirement = TelemetryLimitRequirement(
        name="Radiator",
        quantity="radiator_temperatures_k",
        key="Bus",
        minimum=250.0,
        maximum=300.0,
    )
    samples = [
        Housekeeping(timestamp=START, radiator_temperatures_k={"Bus": 280.0}),
        Housekeeping(
            timestamp=START + timedelta(minutes=1),
            radiator_temperatures_k={"Bus": 305.0, "Other": 100.0},
        ),
        Housekeeping(timestamp=START + timedelta(minutes=2)),
    ]
    result = _check(requirement, samples)
    assert result.checked == 2
    [violation] = result.violations
    assert violation.measured == 305.0 and violation.limit == 300.0
    assert "radiator_temperatures_k['Bus']" in violation.message


def test_what_cannot_be_checked_is_reported() -> None:
    requirement = TelemetryLimitRequirement(
        name="Battery", quantity="battery_level", minimum=0.4
    )
    assert _check(requirement, None).status == "not evaluated"
    unknown = TelemetryLimitRequirement(name="Typo", quantity="batery", minimum=0.4)
    with pytest.raises(ValueError, match="batery"):
        _check(unknown, _samples([0.5]))
    with pytest.raises(ValidationError, match="minimum or a maximum"):
        TelemetryLimitRequirement(name="x", quantity="battery_level")
    with pytest.raises(ValidationError, match="above maximum"):
        TelemetryLimitRequirement(
            name="x", quantity="battery_level", minimum=2, maximum=1
        )
    with pytest.raises(ValidationError, match="unknown ACS modes"):
        TelemetryLimitRequirement(
            name="x", quantity="battery_level", minimum=0, modes=["SCIENCE", "NAP"]
        )


def test_a_run_is_checked_against_its_limits() -> None:
    config = make_config(2)
    config.requirements = [
        TelemetryLimitRequirement(
            name="Battery floor", quantity="battery_level", minimum=0.0
        ),
        TelemetryLimitRequirement(
            name="Battery ceiling", quantity="battery_level", maximum=0.0
        ),
    ]
    run = QueueDITL(config=config, begin=BEGIN, end=BEGIN + timedelta(hours=2))
    assert run.calc()
    report = run.check_requirements()
    assert report.result("Battery floor").status == "met"
    ceiling = report.result("Battery ceiling")
    assert ceiling.status == "violated"
    assert ceiling.checked == len(run.telemetry.housekeeping)
