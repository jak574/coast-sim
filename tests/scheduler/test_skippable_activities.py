"""An activity that serves an instrument, such as its calibration, is skipped
where that instrument observes nothing for long enough around it."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import DITL, QueueDITL, RollingHorizonDITL, create_ditl
from conops.config import MissionConfig, ObservationGroup, ObservationGroups
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.targets import Plan, Pointing, ScheduledActivity

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0, make_config, make_target

HOURS = 8
END = BEGIN + timedelta(hours=HOURS)
ORDER = [3, 1, 2]
FILLER = 100
CALIBRATION = "Lamp calibration"
OTHER = "Spectrograph"
"""An instrument no target observes with."""


def _config() -> MissionConfig:
    config = make_config(HOURS)
    config.observation_groups = ObservationGroups(
        groups=[
            ObservationGroup(
                name="campaign", members=ORDER, consecutive=True, window_start=T0
            )
        ]
    )
    return config


def _targets(config: MissionConfig) -> list[Pointing]:
    """Three members of one whole 40 minute visit each, and a filler."""
    members = [
        make_target(config, k, *PATCH[k], merit=m, minutes=40)
        for k, m in zip((1, 2, 3), (60, 50, 40), strict=True)
    ]
    filler = make_target(config, FILLER, *PATCH[4], merit=30, minutes=120, snapshot=20)
    return [*members, filler]


def _calibration(
    instrument: str | None = OTHER, unused: float | None = 1.5 * HOUR
) -> ScheduledActivity:
    """Twenty minutes, due an hour in: inside the two hours the group takes."""
    return ScheduledActivity(
        name=CALIBRATION,
        begin=T0 + 60 * MIN,
        duration_seconds=20 * MIN,
        for_instrument=instrument,
        skip_if_unused_seconds=unused,
    )


def _held(plan: Plan) -> bool:
    return any(entry.name == CALIBRATION for entry in plan)


def _group_gaps(plan: Plan) -> list[float]:
    science = sorted(
        (e for e in plan if e.obstype.name == "AT" and int(e.obsid) in ORDER),
        key=lambda e: float(e.begin),
    )
    return [
        float(later.begin) - float(earlier.end)
        for earlier, later in zip(science, science[1:], strict=False)
    ]


def _dispatch(activity: ScheduledActivity) -> QueueDITL:
    config = _config()
    ditl = create_ditl(config, _targets(config), BEGIN, END, activities=[activity])
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = 60
    assert ditl.calc()
    return ditl


def _planner(kind: str, config: MissionConfig, activity: ScheduledActivity):  # type: ignore[no-untyped-def]
    targets = _targets(config)
    common = {"step_size": 60, "activities": [activity]}
    if kind == "priority":
        return PriorityScheduler(config, targets, BEGIN, END, **common)
    if kind == "local_search":
        return LocalSearchScheduler(
            config, targets, BEGIN, END, time_limit=2.0, **common
        )
    pytest.importorskip("ortools")
    from conops.schedulers import CpSatScheduler

    return CpSatScheduler(config, targets, BEGIN, END, solver_time_limit=2.0, **common)


KINDS = ["priority", "local_search", "cp_sat"]


class TestTheRule:
    def test_the_instrument_and_the_interval_are_given_together(self) -> None:
        with pytest.raises(ValidationError):
            _calibration(instrument=None)
        with pytest.raises(ValidationError):
            _calibration(unused=None)

    def test_a_dump_or_a_safe_hold_cannot_be_skipped(self) -> None:
        for kind in ("desaturation", "safe_hold"):
            with pytest.raises(ValidationError):
                ScheduledActivity(
                    name="x",
                    begin=T0,
                    duration_seconds=600,
                    for_instrument=OTHER,
                    skip_if_unused_seconds=HOUR,
                    **{kind: True},
                )

    def test_skipped_where_its_start_is_inside_a_long_enough_stretch(self) -> None:
        activity = _calibration()
        assert activity.skipped_in(T0, T0 + 2 * HOUR)
        # Too short a stretch, and a stretch the start is not inside.
        assert not activity.skipped_in(T0, T0 + HOUR + 20 * MIN)
        assert not activity.skipped_in(T0 + 61 * MIN, T0 + 4 * HOUR)

    def test_an_activity_without_the_rule_is_never_skipped(self) -> None:
        activity = _calibration(instrument=None, unused=None)
        assert not activity.skippable
        assert not activity.skipped_in(T0, T0 + 10 * HOUR)


class TestDispatch:
    def test_skipped_through_a_group_that_does_not_use_the_instrument(self) -> None:
        ditl = _dispatch(_calibration())
        assert not _held(ditl.plan)
        assert [a.name for a in ditl.skipped_activities] == [CALIBRATION]
        assert any("skipped" in e.description for e in ditl.log.events)
        # The group runs on without stopping for it.
        assert max(_group_gaps(ditl.plan)) < 15 * MIN

    def test_held_where_the_stretch_is_too_short(self) -> None:
        ditl = _dispatch(_calibration(unused=6 * HOUR))
        assert _held(ditl.plan)
        assert ditl.skipped_activities == []

    def test_held_where_the_group_uses_the_instrument(self) -> None:
        config = _config()
        used = config.payload.instruments[0].name
        targets = _targets(config)
        for target in targets:
            target.instrument_name = used
        ditl = create_ditl(
            config, targets, BEGIN, END, activities=[_calibration(instrument=used)]
        )
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 60
        assert ditl.calc()
        assert _held(ditl.plan)
        assert ditl.skipped_activities == []


class TestPlanners:
    @pytest.mark.parametrize("kind", KINDS)
    def test_skipped_through_a_group_that_does_not_use_the_instrument(
        self, kind: str
    ) -> None:
        config = _config()
        scheduler = _planner(kind, config, _calibration())
        plan = scheduler.schedule()
        assert not _held(plan)
        assert [a.name for a in scheduler.skipped_activities] == [CALIBRATION]
        assert max(_group_gaps(plan)) < 15 * MIN
        run = DITL(config=config, plan=plan, begin=BEGIN, end=END)
        run.step_size = 60
        assert run.calc()
        assert run.validate_plan_matches_execution() == []

    @pytest.mark.parametrize("kind", KINDS)
    def test_held_where_the_stretch_is_too_short(self, kind: str) -> None:
        scheduler = _planner(kind, _config(), _calibration(unused=6 * HOUR))
        assert _held(scheduler.schedule())
        assert scheduler.skipped_activities == []


def test_a_rolling_run_skips_it_and_reports_it() -> None:
    config = _config()
    ditl = RollingHorizonDITL(
        config,
        _targets(config),
        begin=BEGIN,
        end=END,
        horizon=timedelta(hours=4),
        replan_interval=timedelta(hours=1),
        activities=[_calibration()],
    )
    ditl.step_size = 60
    assert ditl.calc()
    assert not _held(ditl.plan)
    assert [a.name for a in ditl.skipped_activities] == [CALIBRATION]
    assert max(_group_gaps(ditl.plan)) < 15 * MIN


def test_a_rolling_run_does_not_offer_a_skipped_activity_again() -> None:
    """A replan can begin after a skipped activity's start window has closed
    and before the activity would have ended. The activity was skipped, so
    it is not reserved then: reserving it would fail, its window being
    closed."""
    config = _config()
    late = ScheduledActivity(
        name=CALIBRATION,
        begin=T0 + 50 * MIN,
        duration_seconds=40 * MIN,
        for_instrument=OTHER,
        skip_if_unused_seconds=1.5 * HOUR,
    )
    ditl = RollingHorizonDITL(
        config,
        _targets(config),
        begin=BEGIN,
        end=END,
        horizon=timedelta(hours=4),
        replan_interval=timedelta(hours=1),
        activities=[late],
    )
    ditl.step_size = 60
    assert ditl.calc()
    assert not _held(ditl.plan)
    assert [a.name for a in ditl.skipped_activities] == [CALIBRATION]
