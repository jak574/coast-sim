"""The slew time requirement: its classes, the great-circle angle on each
entry, and the checks the runs, the planners and the statistics make against
it."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import QueueDITL, create_ditl
from conops.requirements import (
    BEYOND_THE_TABLE,
    SlewClass,
    SlewTimeRequirement,
    check_requirements,
)
from conops.schedulers import PriorityScheduler
from conops.simulation.slew import Slew
from conops.targets import PlanEntry
from tests.scheduler.planning_scenario import BEGIN, PATCH, T0
from tests.scheduler.planning_scenario import make_config as _config
from tests.scheduler.planning_scenario import make_target as _target

HOURS = 4
END = BEGIN + timedelta(hours=HOURS)
TABLE = [
    SlewClass(name="small", max_great_circle_deg=30.0, max_seconds=300.0),
    SlewClass(name="large", max_great_circle_deg=180.0, max_seconds=1500.0),
]
STRICT = [
    SlewClass(name="small", max_great_circle_deg=30.0, max_seconds=10.0),
    SlewClass(name="large", max_great_circle_deg=180.0, max_seconds=20.0),
]


class TestConfiguration:
    def test_the_classes_are_sorted_and_distinct(self) -> None:
        requirement = SlewTimeRequirement(classes=list(reversed(TABLE)))
        assert [c.name for c in requirement.classes] == ["small", "large"]
        assert requirement.slew_class(10.0) is requirement.classes[0]
        assert requirement.slew_class(30.0) is requirement.classes[0]
        assert requirement.slew_class(90.0) is requirement.classes[1]
        assert SlewTimeRequirement(classes=TABLE[:1]).slew_class(90.0) is None
        with pytest.raises(ValidationError, match="distinct angles"):
            SlewTimeRequirement(
                classes=[
                    SlewClass(name="a", max_great_circle_deg=30, max_seconds=1),
                    SlewClass(name="b", max_great_circle_deg=30, max_seconds=2),
                ]
            )
        with pytest.raises(ValidationError, match="distinct names"):
            SlewTimeRequirement(
                classes=[
                    SlewClass(name="a", max_great_circle_deg=30, max_seconds=1),
                    SlewClass(name="a", max_great_circle_deg=60, max_seconds=2),
                ]
            )
        with pytest.raises(ValidationError):
            SlewClass(name="wide", max_great_circle_deg=200.0, max_seconds=1.0)
        with pytest.raises(ValidationError):
            SlewTimeRequirement(classes=[])

    def test_a_roll_alone_has_no_great_circle_angle(self) -> None:
        config = _config(HOURS)
        slew = Slew(config=config)
        slew.startra, slew.startdec, slew.startroll = 10.0, 10.0, 0.0
        slew.endra, slew.enddec, slew.endroll = 10.0, 10.0, 90.0
        assert slew.great_circle_deg == pytest.approx(0.0)
        slew.endra, slew.enddec = 10.0, 40.0
        assert slew.great_circle_deg == pytest.approx(30.0)


def _entry(
    config, begin: float, slewtime: float, degrees: float, obsid: int = 1
) -> PlanEntry:  # type: ignore[no-untyped-def]
    return PlanEntry(
        config=config,
        ra=10.0,
        dec=10.0,
        begin=begin,
        end=begin + slewtime + 600.0,
        slewtime=int(slewtime),
        slewdist=degrees + 5.0,
        great_circle_deg=degrees,
        obsid=obsid,
        exptime=600,
    )


class TestChecks:
    def test_violations_and_the_summary(self) -> None:
        config = _config(HOURS)
        requirement = SlewTimeRequirement(classes=TABLE)
        config.requirements = [requirement]
        entries = [
            _entry(config, T0, 200.0, 20.0, 1),
            _entry(config, T0 + 1000, 400.0, 25.0, 2),
            _entry(config, T0 + 2000, 1000.0, 120.0, 3),
            _entry(config, T0 + 3000, 0.0, 0.0, 4),
        ]
        report = check_requirements(config, entries)
        result = report.result("Slew time")
        assert result.status == "violated" and result.checked == 3
        [violation] = report.violations
        assert violation.obsid == 2
        assert violation.measured == 400.0 and violation.limit == 300.0
        assert "over the 300 s" in str(violation)
        rows = requirement.summarise(entries)
        assert [(r.name, r.slews, r.violations) for r in rows] == [
            ("small", 2, 1),
            ("large", 1, 0),
        ]
        assert rows[0].longest_seconds == 400.0
        config.requirements = []
        assert check_requirements(config, entries).results == ()

    def test_a_slew_wider_than_the_table_is_reported_apart(self) -> None:
        config = _config(HOURS)
        requirement = SlewTimeRequirement(
            classes=[
                SlewClass(name="small", max_great_circle_deg=30.0, max_seconds=300.0)
            ]
        )
        config.requirements = [requirement]
        entries = [_entry(config, T0, 900.0, 100.0)]
        rows = requirement.summarise(entries)
        assert rows[-1].name == BEYOND_THE_TABLE and rows[-1].slews == 1
        assert check_requirements(config, entries).met


def _slewing_config(table):  # type: ignore[no-untyped-def]
    config = _config(HOURS)
    config.requirements = [SlewTimeRequirement(classes=table)]
    return config


def _targets(config):  # type: ignore[no-untyped-def]
    return [
        _target(config, 100 + i, ra, dec, merit=90 - i, minutes=40, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH)
    ]


class TestRuns:
    def test_dispatch_records_the_angle_and_reports_against_the_table(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = _slewing_config(STRICT)
        ditl = create_ditl(config, _targets(config), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 60
        assert ditl.calc()
        slews = [
            e for e in ditl.plan if e.obstype.name == "AT" and float(e.slewtime) > 0
        ]
        assert slews
        for entry in slews:
            assert 0.0 <= float(entry.great_circle_deg) <= float(entry.slewdist) + 1e-6
        assert any(float(e.great_circle_deg) > 0.0 for e in slews)
        report = ditl.check_requirements()
        violations = report.violations
        assert violations and not report.met
        assert all(v.requirement == "Slew time" for v in violations)
        assert ditl.validate_plan_matches_execution() == []
        rows = ditl.slew_class_summary()
        assert sum(r.violations for r in rows) == len(violations)
        ditl.print_statistics()
        printed = capsys.readouterr().out
        assert "REQUIREMENTS" in printed
        assert "SLEWS BY REQUIREMENT CLASS" in printed
        generous = _slewing_config(TABLE)
        ditl = create_ditl(generous, _targets(generous), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 60
        assert ditl.calc()
        assert ditl.check_requirements().met

    def test_the_planner_logs_the_slews_over_their_class(self) -> None:
        config = _slewing_config(STRICT)
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=60
        )
        plan = scheduler.schedule()
        assert any(float(e.great_circle_deg) > 0.0 for e in plan)
        assert any(
            "Requirement Slew time:" in e.description for e in scheduler.log.events
        )
        assert check_requirements(config, list(plan)).violations
        generous = _slewing_config(TABLE)
        scheduler = PriorityScheduler(
            generous, _targets(generous), BEGIN, END, step_size=60
        )
        scheduler.schedule()
        assert not any(
            "Requirement Slew time:" in e.description for e in scheduler.log.events
        )


class TestRollingRun:
    def test_the_rolling_run_reports_its_slews_against_the_table(self) -> None:
        from conops.config.scheduler import SchedulerMode
        from conops.ditl import RollingHorizonDITL

        config = _slewing_config(STRICT)
        config.scheduler.mode = SchedulerMode.ROLLING
        ditl = create_ditl(config, _targets(config), BEGIN, END)
        assert isinstance(ditl, RollingHorizonDITL)
        ditl.step_size = 60
        assert ditl.calc()
        violations = ditl.check_requirements().violations
        assert violations
        assert sum(r.violations for r in ditl.slew_class_summary()) == len(violations)
