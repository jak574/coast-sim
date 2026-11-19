"""An observation cut short, a short remainder, and a group's deadline:
dispatch and the planners treat each the same way."""

from datetime import timedelta

import pytest

from conops import QueueDITL, create_ditl
from conops.config import (
    MissionConfig,
    ObservationGroup,
    ObservationGroups,
    TargetConfig,
)
from conops.config.observation_categories import ObservationCategory
from conops.schedulers import PriorityScheduler
from conops.targets import Pointing, TargetQueue, visit_windows
from conops.targets.plan_entry import is_rounding_remainder, snapshot_min_seconds

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 12
STEP = 60
END = BEGIN + timedelta(hours=HOURS)
VISIT = 10
STARTS = [T0 + 1 * HOUR, T0 + 4 * HOUR, T0 + 7 * HOUR]
WINDOWS = visit_windows(STARTS, 30 * MIN)
CADENCE = 2 * HOUR


def _cadence_config() -> MissionConfig:
    return _config(
        HOURS,
        weights=TargetConfig(cadence_weight=1.0),
        categories=[
            ObservationCategory(
                name="Monitoring",
                obsid_min=900,
                obsid_max=999,
                cadence_seconds=CADENCE,
            )
        ],
    )


def _timed_target(config: MissionConfig) -> Pointing:
    return Pointing(
        config=config,
        obsid=900,
        name="V900",
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        merit=90.0,
        ss_min=VISIT * MIN,
        ss_max=VISIT * MIN,
        visit_windows=list(WINDOWS),
    )


def _dispatch(config: MissionConfig, targets: list[Pointing]) -> QueueDITL:
    ditl = create_ditl(config, targets, BEGIN, END)
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = STEP
    return ditl


class TestCadenceAfterACut:
    def test_dispatch_takes_a_cut_visit_again_in_its_window(self) -> None:
        """A target of opportunity cuts the first visit of a cadence target.
        The visit is taken again in its window: the cut attempt is not a
        visit to wait a cadence after."""
        config = _cadence_config()
        fillers = [
            _target(config, 1 + k, *PATCH[1 + k], merit=50.0, minutes=180, snapshot=60)
            for k in range(3)
        ]
        ditl = _dispatch(config, [_timed_target(config), *fillers])
        cut_at = STARTS[0] + 6 * MIN
        ditl.submit_too(
            obsid=30001,
            ra=PATCH[0][0] + 2.0,
            dec=PATCH[0][1],
            merit=1000.0,
            exptime=5 * MIN,
            deadline=cut_at + 30 * MIN,
            name="Urgent",
            submit_time=cut_at,
        )
        assert ditl.calc()
        queued = next(t for t in ditl.queue.targets if int(t.obsid) == 900)
        assert queued.interruptions
        assert queued.visits_done == {0, 1, 2}
        assert queued.done

    def test_a_cut_remainder_is_due_and_a_finished_visit_waits(self) -> None:
        config = _cadence_config()
        target = _target(config, 901, *PATCH[1], minutes=60, snapshot=20)
        queue = TargetQueue(config=config, ephem=config.constraint.ephem)
        queue.append(target)
        target.record_collection(T0 + 10 * MIN, 5 * MIN)
        assert not queue._cadence_due(target, T0 + 11 * MIN)
        target.record_interruption(T0 + 10 * MIN)
        assert target.last_observation_cut
        assert queue._cadence_due(target, T0 + 11 * MIN)
        # The remainder is observed; the next visit waits a cadence again.
        target.record_collection(T0 + 30 * MIN, 15 * MIN)
        assert not target.last_observation_cut
        assert not queue._cadence_due(target, T0 + 31 * MIN)
        assert queue._cadence_due(target, T0 + 30 * MIN + CADENCE)

    def test_the_priority_planner_places_a_cut_remainder_without_the_wait(
        self,
    ) -> None:
        def first_start(cut: bool) -> float:
            config = _cadence_config()
            target = _target(config, 901, *PATCH[1], minutes=60, snapshot=20)
            target.record_collection(T0 - 5 * MIN, 5 * MIN)
            if cut:
                target.record_interruption(T0 - 5 * MIN)
            plan = PriorityScheduler(
                config, [target], BEGIN, END, step_size=STEP, include_passes=False
            ).schedule()
            return min(float(e.begin) for e in plan if int(e.obsid) == 901)

        assert first_start(cut=False) >= T0 - 5 * MIN + CADENCE
        assert first_start(cut=True) < T0 + CADENCE / 2


class TestInterruptedVisit:
    def test_a_cut_before_collecting_leaves_the_earlier_visit(self) -> None:
        """Visit 0 ends without an interruption. An observation for visit 1
        is cut before it collects: visit 0 stays visited."""
        config = _config(HOURS)
        target = _timed_target(config)
        target.record_collection(STARTS[0] + 4 * MIN, 3 * MIN)
        assert target.visits_done == {0}
        exposure = target.exptime
        target.record_interruption(STARTS[1] + 2 * MIN, began=STARTS[1])
        assert target.visits_done == {0}
        assert target.exptime == exposure
        assert target.interruptions == (STARTS[1] + 2 * MIN,)

    def test_a_cut_after_collecting_reopens_the_visit(self) -> None:
        config = _config(HOURS)
        target = _timed_target(config)
        target.record_collection(STARTS[0] + 4 * MIN, 3 * MIN)
        target.record_interruption(STARTS[0] + 4 * MIN, began=STARTS[0])
        assert target.visits_done == set()
        assert target.exptime == 3 * VISIT * MIN


class TestRoundingRemainder:
    def test_most_of_a_short_request_is_not_rounding(self) -> None:
        """A 100 s request cut after 45 s at a 60 s step still has 55 s to
        observe; it is not done."""
        config = _config(HOURS)
        target = _target(config, 1, *PATCH[0], minutes=100 / 60, snapshot=100 / 60)
        target.record_collection(T0 + 45.0, 45.0, step=STEP)
        assert target.exptime == pytest.approx(55.0)
        assert not target.done
        assert snapshot_min_seconds(target, STEP) == pytest.approx(55.0)

    def test_the_end_of_a_long_request_is_rounding(self) -> None:
        config = _config(HOURS)
        target = _target(config, 1, *PATCH[0], minutes=20)
        target.record_collection(T0 + 1170.0, 1170.0, step=STEP)
        assert target.exptime == 0
        assert snapshot_min_seconds(target, STEP) == target.ss_min

    def test_the_rule(self) -> None:
        assert is_rounding_remainder(30.0, 60.0, 1200.0)
        assert not is_rounding_remainder(30.0, 60.0, 200.0)
        assert not is_rounding_remainder(60.0, 60.0, 1200.0)
        assert is_rounding_remainder(30.0, 60.0, None)
        assert not is_rounding_remainder(30.0, 0.0, None)


class TestGroupDeadline:
    @pytest.mark.parametrize("where", ["none", "target", "group"])
    def test_collection_begins_by_the_groups_deadline(self, where: str) -> None:
        """A deadline too soon to slew by keeps the target from being
        observed, whether it is the target's own or its group's."""
        config = _config(HOURS)
        deadline = T0 + 60.0
        target = _target(
            config,
            1,
            *PATCH[0],
            merit=90.0,
            deadline=deadline if where == "target" else None,
        )
        if where == "group":
            config.observation_groups = ObservationGroups(
                groups=[
                    ObservationGroup(name="bin", members=[1], window_deadline=deadline)
                ]
            )
        ditl = _dispatch(config, [target])
        assert ditl.calc()
        collected = [
            e for e in ditl.plan if int(e.obsid) == 1 and e.collection_begin is not None
        ]
        if where == "none":
            assert float(collected[0].collection_begin) > deadline
        else:
            assert collected == []
