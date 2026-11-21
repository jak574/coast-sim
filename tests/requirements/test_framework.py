"""The requirements framework: kinds are registered by subclassing, a mission
file names them, results report what was checked, and a new kind needs no
change to the library."""

from datetime import timedelta
from typing import ClassVar

import pytest
from pydantic import ValidationError

from conops import MissionConfig, QueueDITL
from conops.common import ACSMode
from conops.requirements import (
    Requirement,
    RequirementEvidence,
    RequirementReport,
    RequirementResult,
    SlewClass,
    SlewTimeRequirement,
    TelemetryLimitRequirement,
    check_requirements,
    parse_requirement,
    requirement_kinds,
)
from tests.scheduler.planning_scenario import BEGIN, PATCH, make_config, make_target

CLASSES = [{"name": "any", "max_great_circle_deg": 180.0, "max_seconds": 900.0}]


class MinimumScienceFraction(Requirement):
    """A requirement defined outside the library: at least a fraction of a
    run's steps are spent observing."""

    kind: ClassVar[str] = "test_minimum_science_fraction"

    minimum: float

    def evaluate(self, evidence: RequirementEvidence) -> RequirementResult:
        samples = evidence.housekeeping
        if samples is None:
            return self.not_evaluated("needs the telemetry of a run")
        science = sum(sample.acs_mode == ACSMode.SCIENCE for sample in samples)
        fraction = science / len(samples)
        violations = []
        if fraction < self.minimum:
            violations.append(
                self.violation(
                    samples[0].timestamp.timestamp(),
                    f"observing for {fraction:.0%} of the run",
                    measured=fraction,
                    limit=self.minimum,
                )
            )
        return self.result(len(samples), violations)


def test_the_built_in_kinds_are_registered() -> None:
    kinds = requirement_kinds()
    assert kinds["slew_time"] is SlewTimeRequirement
    assert kinds["telemetry_limit"] is TelemetryLimitRequirement
    assert "" not in kinds


def test_a_requirement_is_read_by_its_kind_and_written_with_it() -> None:
    requirement = parse_requirement({"kind": "slew_time", "classes": CLASSES})
    assert isinstance(requirement, SlewTimeRequirement)
    assert requirement.name == "Slew time"
    written = requirement.model_dump(mode="json")
    assert list(written)[0] == "kind" and written["kind"] == "slew_time"
    assert parse_requirement(written) == requirement
    assert parse_requirement(requirement) is requirement
    with pytest.raises(ValueError, match="unknown requirement kind"):
        parse_requirement({"kind": "no_such_kind", "name": "x"})
    with pytest.raises(ValueError, match="mapping"):
        parse_requirement("slew_time")
    with pytest.raises(ValidationError, match="telemetry_limit"):
        SlewTimeRequirement.model_validate(
            {"kind": "telemetry_limit", "classes": CLASSES}
        )


def test_a_kind_cannot_be_taken_by_another_class() -> None:
    with pytest.raises(TypeError, match="already used"):

        class Other(Requirement):
            kind: ClassVar[str] = "slew_time"

    class Abstract(Requirement):
        """Without a kind, a subclass is a base class and is not registered."""

    assert Abstract not in requirement_kinds().values()
    with pytest.raises(NotImplementedError):
        Abstract(name="x").evaluate(
            RequirementEvidence(config=MissionConfig(), entries=[])
        )


@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_a_mission_file_keeps_its_requirements(tmp_path, suffix: str) -> None:
    config = MissionConfig(
        requirements=[
            {"kind": "slew_time", "classes": CLASSES},
            TelemetryLimitRequirement(
                name="Battery", quantity="battery_level", minimum=0.4
            ),
            MinimumScienceFraction(name="Efficiency", minimum=0.5),
        ]
    )
    path = str(tmp_path / f"mission.{suffix}")
    if suffix == "json":
        config.to_json_file(path)
        loaded = MissionConfig.from_json_file(path)
    else:
        config.to_yaml_file(path)
        loaded = MissionConfig.from_yaml_file(path)
    assert loaded.requirements == config.requirements
    assert [type(r) for r in loaded.requirements] == [
        SlewTimeRequirement,
        TelemetryLimitRequirement,
        MinimumScienceFraction,
    ]


def test_requirement_names_are_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        MissionConfig(
            requirements=[
                {"kind": "slew_time", "classes": CLASSES},
                {"kind": "slew_time", "classes": CLASSES},
            ]
        )


def test_slew_classes_configured_on_the_acs_are_moved() -> None:
    """A file written before requirements had their own section."""
    old = {"spacecraft_bus": {"attitude_control": {"slew_requirements": CLASSES}}}
    config = MissionConfig.model_validate(old)
    [requirement] = config.requirements
    assert isinstance(requirement, SlewTimeRequirement)
    assert requirement.classes == [SlewClass(**CLASSES[0])]
    assert old["spacecraft_bus"]["attitude_control"] == {"slew_requirements": CLASSES}
    empty = {"spacecraft_bus": {"attitude_control": {"slew_requirements": []}}}
    assert MissionConfig.model_validate(empty).requirements == []


def test_a_plan_alone_leaves_telemetry_requirements_unevaluated() -> None:
    config = make_config(2)
    config.requirements = [
        SlewTimeRequirement(classes=[SlewClass(**CLASSES[0])]),
        TelemetryLimitRequirement(
            name="Battery", quantity="battery_level", minimum=0.4
        ),
    ]
    report = check_requirements(config, [])
    assert [r.status for r in report.results] == ["met", "not evaluated"]
    assert not report.met and report.violations == []
    assert report.result("Battery").note
    with pytest.raises(KeyError):
        report.result("absent")
    assert "not evaluated" in str(report)
    assert str(RequirementReport()) == "No requirements configured"


def test_a_run_checks_a_requirement_defined_outside_the_library() -> None:
    config = make_config(3)
    config.requirements = [
        MinimumScienceFraction(name="Easy", minimum=0.2),
        MinimumScienceFraction(name="Impossible", minimum=0.999),
    ]
    run = QueueDITL(config=config, begin=BEGIN, end=BEGIN + timedelta(hours=3))
    for index, (ra, dec) in enumerate(PATCH):
        run.queue.add_request(make_target(config, 100 + index, ra, dec, minutes=40))
    assert run.calc()
    report = run.check_requirements()
    assert report.result("Easy").status == "met"
    assert report.result("Easy").checked == len(run.telemetry.housekeeping)
    impossible = report.result("Impossible")
    assert impossible.status == "violated"
    [violation] = impossible.violations
    assert violation.limit == 0.999 and violation.measured < 0.999
    assert not report.met
