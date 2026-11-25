"""A consecutive observation group: its targets are observed in the order
listed, with no science of another target between the first and the last of
them, while a thruster dump may fall between two of them."""

from datetime import timedelta

import numpy as np
import pytest

from conops import DITL, QueueDITL, RollingHorizonDITL, create_ditl
from conops.common import ObsType
from conops.config import (
    MissionConfig,
    ObservationGroup,
    ObservationGroups,
    StoredMomentumConfig,
)
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.schedulers.priority_scheduler import StartState
from conops.targets import Plan, PlanEntry, Pointing, TOORequest, VisitWindow

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0, make_config, make_target
from .test_momentum_ledger import DUMP
from .test_momentum_ledger import _config as _momentum_config

HOURS = 8
END = BEGIN + timedelta(hours=HOURS)
ORDER = [3, 1, 2]
FILLER = 100


def _config(order: list[int] | None = None) -> MissionConfig:
    config = make_config(HOURS)
    config.observation_groups = ObservationGroups(
        groups=[
            ObservationGroup(name="campaign", members=order or ORDER, consecutive=True)
        ]
    )
    return config


def _targets(
    config: MissionConfig,
    merits: tuple[float, float, float] = (60, 50, 40),
    filler: float = 55,
) -> list[Pointing]:
    """Members 1, 2 and 3, observed in the order 3, 1, 2, and a filler that
    outranks two of them."""
    members = [
        make_target(config, k, *PATCH[k], merit=m, minutes=40, snapshot=20, ss_min=5)
        for k, m in zip((1, 2, 3), merits, strict=True)
    ]
    other = make_target(
        config, FILLER, *PATCH[4], merit=filler, minutes=300, snapshot=20
    )
    return [*members, other]


def _science(plan: Plan) -> list[PlanEntry]:
    return sorted(
        (entry for entry in plan if entry.obstype.name == "AT"),
        key=lambda entry: float(entry.begin),
    )


def _order(plan: Plan) -> list[int]:
    """The targets in the order observed, each named once per run of entries."""
    order: list[int] = []
    for entry in _science(plan):
        if not order or order[-1] != int(entry.obsid):
            order.append(int(entry.obsid))
    return order


def _assert_consecutive(plan: Plan, members: list[int]) -> None:
    """The members present are in order, and nothing else is between them."""
    order = _order(plan)
    positions = [k for k, obsid in enumerate(order) if obsid in members]
    assert positions, "no member was observed"
    run = order[positions[0] : positions[-1] + 1]
    assert run == [obsid for obsid in members if obsid in run]


def _dispatch(config: MissionConfig, targets: list[Pointing]) -> QueueDITL:
    ditl = create_ditl(config, targets, BEGIN, END)
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = 60
    assert ditl.calc()
    return ditl


def _planner(kind: str, config: MissionConfig, targets: list[Pointing]):  # type: ignore[no-untyped-def]
    if kind == "priority":
        return PriorityScheduler(config, targets, BEGIN, END, step_size=60)
    if kind == "local_search":
        return LocalSearchScheduler(
            config, targets, BEGIN, END, step_size=60, time_limit=2.0
        )
    pytest.importorskip("ortools")
    from conops.schedulers import CpSatScheduler

    return CpSatScheduler(
        config, targets, BEGIN, END, step_size=60, solver_time_limit=2.0
    )


KINDS = ["priority", "local_search", "cp_sat"]


class TestConfiguration:
    def test_a_group_is_not_consecutive_unless_asked(self) -> None:
        plain = ObservationGroup(name="bin", members=[1, 2])
        groups = ObservationGroups(
            groups=[plain, ObservationGroup(name="c", members=[9, 7], consecutive=True)]
        )
        assert not plain.consecutive
        assert [group.name for group in groups.consecutive] == ["c"]
        assert groups.position(7) == 1 and groups.position(9) == 0
        assert groups.position(1) is None and groups.position(50) is None
        loaded = ObservationGroups.model_validate_json(groups.model_dump_json())
        assert loaded.groups[1].consecutive


class TestDispatch:
    def test_members_are_observed_in_order_with_nothing_between(self) -> None:
        config = _config()
        ditl = _dispatch(config, _targets(config))
        order = _order(ditl.plan)
        # The filler outranks the first member, 3, so it goes first; once
        # 3 has begun, the group finishes before the filler returns.
        assert order[0] == FILLER
        start = order.index(3)
        assert order[start : start + 3] == ORDER
        _assert_consecutive(ditl.plan, ORDER)
        for obsid in ORDER:
            target = next(t for t in ditl.queue.targets if int(t.obsid) == obsid)
            assert target.done

    def test_without_the_group_the_filler_goes_between(self) -> None:
        config = make_config(HOURS)
        order = _order(_dispatch(config, _targets(config)).plan)
        assert order[:2] == [1, FILLER]

    def test_a_member_waits_for_the_members_before_it(self) -> None:
        # Member 3 may not begin before the run ends, so the group never
        # begins: members 1 and 2 are not observed, and the filler is.
        config = _config()
        targets = _targets(config, merits=(90, 90, 90))
        closed = next(t for t in targets if int(t.obsid) == 3)
        closed.earliest_start = T0 + 2 * HOURS * HOUR
        ditl = _dispatch(config, targets)
        assert _order(ditl.plan) == [FILLER]

    def test_the_group_ends_at_a_member_whose_deadline_passes(self) -> None:
        config = _config()
        targets = _targets(config, merits=(90, 90, 90))
        # Member 1, second in the order, must begin before the run does.
        late = next(t for t in targets if int(t.obsid) == 1)
        late.deadline = T0 - HOUR
        ditl = _dispatch(config, targets)
        order = _order(ditl.plan)
        assert order == [3, FILLER]
        held = ditl.queue.consecutive_holds(T0 + 4 * HOUR)
        assert "can no longer be observed" in held[2]

    def test_a_target_of_opportunity_waits_for_the_group(self) -> None:
        config = _config()
        ditl = create_ditl(config, _targets(config, merits=(90, 90, 90)), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 60
        ditl.submit_too(
            TOORequest(
                obsid=900,
                name="Transient",
                ra=PATCH[3][0],
                dec=PATCH[3][1],
                merit=1000.0,
                exptime=10 * MIN,
                submit_time=T0 + 30 * MIN,
            )
        )
        assert ditl.calc()
        order = _order(ditl.plan)
        assert order[:4] == [*ORDER, 900]


class TestPlanners:
    @pytest.mark.parametrize("kind", KINDS)
    def test_the_plan_keeps_the_order_with_nothing_between(self, kind: str) -> None:
        config = _config()
        targets = _targets(config)
        plan = _planner(kind, config, targets).schedule()
        _assert_consecutive(plan, ORDER)
        assert [obsid for obsid in _order(plan) if obsid in ORDER] == ORDER
        assert FILLER in _order(plan)
        run = DITL(config=config, plan=plan, begin=BEGIN, end=END)
        run.step_size = 60
        assert run.calc()
        assert run.validate_plan_matches_execution() == []

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_group_is_placed_before_other_science(self, kind: str) -> None:
        # The filler outranks every member, and the group still goes first.
        config = _config()
        targets = _targets(config, merits=(40, 40, 40), filler=90)
        plan = _planner(kind, config, targets).schedule()
        assert _order(plan)[:4] == [*ORDER, FILLER]

    def test_a_group_begins_where_it_has_room_to_finish(self) -> None:
        # An hour of committed science lies 80 minutes into the run. The
        # time before it holds one member but not all three, so the group
        # begins after it, and the filler takes the time before.
        config = _config()
        begin = T0 + 80 * MIN
        locked = PlanEntry(
            config=config,
            name="committed",
            ra=PATCH[4][0],
            dec=PATCH[4][1],
            roll=0.0,
            obsid=500,
            obstype=ObsType.AT,
            begin=begin,
            end=begin + HOUR,
            collection_begin=begin,
            collection_end=begin + HOUR,
            ss_min=60,
        )
        plan = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=60, locked=[locked]
        ).schedule()
        order = _order(plan)
        assert order[order.index(500) :][:4] == [500, *ORDER]
        assert order[0] == FILLER

    def test_the_group_ends_at_a_member_that_cannot_be_placed(self) -> None:
        config = _config()
        targets = _targets(config, merits=(90, 90, 90))
        late = next(t for t in targets if int(t.obsid) == 1)
        late.deadline = T0 - HOUR
        scheduler = PriorityScheduler(config, targets, BEGIN, END, step_size=60)
        plan = scheduler.schedule()
        assert _order(plan) == [3, FILLER]
        assert {int(t.obsid) for t in scheduler.unplaced} >= {2}

    def test_a_group_under_way_goes_on_before_anything_else(self) -> None:
        # The first member was observed before this plan; the next member
        # is placed first although the filler outranks it.
        config = _config()
        targets = _targets(config, merits=(40, 40, 40), filler=90)
        first = next(t for t in targets if int(t.obsid) == 3)
        first.done = True
        plan = PriorityScheduler(config, targets, BEGIN, END, step_size=60).schedule()
        assert _order(plan)[:3] == [1, 2, FILLER]


@pytest.mark.parametrize("consecutive", [True, False])
def test_a_dump_falls_between_two_members(consecutive: bool) -> None:
    """The wheels hold one observation but not two, so a dump is reserved
    after the first. In a consecutive group the second member follows the
    dump, and the target outside the group, which outranks it, comes after."""
    config = _momentum_config()
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    origin = (80.0, -20.0, 0.0)
    duration = 600
    merits = {1: 90.0, 2: 10.0, 3: 50.0}
    targets = [
        Pointing(
            config=config,
            obsid=obsid,
            ra=origin[0],
            dec=origin[1],
            roll_deg=0,
            merit=merit,
            exptime=duration,
            ss_min=duration,
            ss_max=duration,
        )
        for obsid, merit in merits.items()
    ]
    settings = dict(
        config=config,
        targets=targets,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=1),
        step_size=10,
        include_passes=False,
        start_state=StartState(time=T0, attitude=origin),
    )
    empty = PriorityScheduler(**settings)  # type: ignore[arg-type]
    empty.schedule()
    first = next(block for block in empty.timeline if block.entry.obsid == 1)
    assert empty._momentum is not None
    loading = empty._momentum.loading(first)
    assert loading > 0
    config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig(
        gravity_gradient_enabled=True,
        capacity_n_m_s=1.7 * loading,
        planning_level=0.3,
        thruster_dump=DUMP,
    )
    config.observation_groups = ObservationGroups(
        groups=[ObservationGroup(name="campaign", members=[1, 2], consecutive=True)]
        if consecutive
        else []
    )
    plan = PriorityScheduler(**settings).schedule()  # type: ignore[arg-type]
    entries = sorted(plan, key=lambda entry: float(entry.begin))
    names = [
        "dump" if getattr(entry, "desaturation", False) else int(entry.obsid)
        for entry in entries
    ]
    science = [name for name in names if name != "dump"]
    assert "dump" in names
    if consecutive:
        assert science == [1, 2, 3]
        assert "dump" in names[names.index(1) : names.index(2)]
    else:
        assert science == [1, 3, 2]
    assert np.isfinite(float(entries[-1].end))


def test_a_rolling_run_goes_on_with_a_group_across_replans() -> None:
    """The group takes longer than the time between replans. Each replan
    places the members still to observe first, so the run observes the
    group in order with nothing between."""
    config = _config()
    targets = _targets(config, merits=(40, 40, 40), filler=90)
    ditl = RollingHorizonDITL(
        config,
        targets,
        begin=BEGIN,
        end=END,
        horizon=timedelta(hours=3),
        replan_interval=timedelta(hours=1),
    )
    ditl.step_size = 60
    assert ditl.calc()
    assert len(ditl.replans) >= 3
    assert _order(ditl.plan)[:4] == [*ORDER, FILLER]
    assert ditl.validate_plan_matches_execution() == []


def test_a_rolling_run_holds_the_order_when_the_first_member_must_wait() -> None:
    """The first member cannot begin inside the first plans' horizon. No
    plan may place a later member ahead of it: the run observes the group
    in order once the first member can begin."""
    config = _config()
    targets = _targets(config, merits=(40, 40, 40), filler=20)
    first = next(t for t in targets if int(t.obsid) == ORDER[0])
    first.earliest_start = T0 + 4 * HOUR
    ditl = RollingHorizonDITL(
        config,
        targets,
        begin=BEGIN,
        end=END,
        horizon=timedelta(hours=3),
        replan_interval=timedelta(hours=1),
    )
    ditl.step_size = 60
    assert ditl.calc()
    order = _order(ditl.plan)
    members = [obsid for obsid in order if obsid in ORDER]
    assert members == ORDER[: len(members)]
    assert members, "the group is observed once its first member can begin"
    _assert_consecutive(ditl.plan, members)


@pytest.mark.parametrize("kind", KINDS)
def test_members_wait_for_a_member_the_plan_does_not_place(kind: str) -> None:
    """A member held back from a plan is still unfinished, so the members
    after it are not placed ahead of it."""
    config = _config()
    targets = _targets(config, merits=(90, 90, 90))
    scheduler = _planner(kind, config, targets)
    scheduler.held = frozenset({ORDER[0]})
    plan = scheduler.schedule()
    assert _order(plan) == [FILLER]
    assert {int(t.obsid) for t in scheduler.unplaced} >= set(ORDER[1:])


def test_members_follow_a_member_finished_before_the_plan() -> None:
    """A member with no exposure left does not hold the members after it."""
    config = _config()
    targets = _targets(config, merits=(90, 90, 90))
    first = next(t for t in targets if int(t.obsid) == ORDER[0])
    first.done = True
    plan = PriorityScheduler(config, targets, BEGIN, END, step_size=60).schedule()
    assert _order(plan)[:2] == ORDER[1:]


TIMED = 200
WINDOW = (T0 + 30 * MIN, T0 + 60 * MIN)


def _timed_against_group(config: MissionConfig, minutes: int = 20) -> list[Pointing]:
    """The three members, two hours of science together and observable from
    the start, and one visit of ``minutes`` that has to begin between 30
    and 60 minutes in and ranks below every member."""
    members = _targets(config)[:3]
    timed = Pointing(
        config=config,
        obsid=TIMED,
        name="Timed",
        ra=PATCH[4][0],
        dec=PATCH[4][1],
        merit=30,
        fom=30,
        ss_min=minutes * MIN,
        ss_max=minutes * MIN,
        visit_windows=[VisitWindow(start=WINDOW[0], deadline=WINDOW[1])],
    )
    return [*members, timed]


def _assert_window_kept(plan: Plan, minutes: int = 20) -> None:
    """The timed visit was collected whole, beginning in its window, and the
    group ran in order after it with nothing between its members."""
    timed = [e for e in _science(plan) if int(e.obsid) == TIMED]
    assert sum(float(e.exposure or 0) for e in timed) == minutes * MIN
    assert WINDOW[0] <= float(timed[0].collection_begin) <= WINDOW[1]
    _assert_consecutive(plan, ORDER)
    order = _order(plan)
    assert order.index(TIMED) < order.index(ORDER[0])
    for obsid in ORDER:
        collected = sum(float(e.exposure or 0) for e in plan if int(e.obsid) == obsid)
        assert collected == 40 * MIN


class TestTimeBoundRequestsKeepTheirWindows:
    """A consecutive group with no window of its own can begin later; a
    request whose window would close while the group runs cannot. The group
    waits for it."""

    def test_dispatch_holds_the_group_for_a_closing_window(self) -> None:
        config = _config()
        ditl = _dispatch(config, _timed_against_group(config))
        _assert_window_kept(ditl.plan)
        assert any(
            "has no window and waits for" in e.description for e in ditl.log.events
        )

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_planners_begin_the_group_after_the_window(self, kind: str) -> None:
        config = _config()
        targets = _timed_against_group(config)
        scheduler = _planner(kind, config, targets)
        plan = scheduler.schedule()
        _assert_window_kept(plan)
        run = DITL(config=config, plan=plan, begin=BEGIN, end=END)
        run.step_size = 60
        assert run.calc()
        assert run.validate_plan_matches_execution() == []

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_planners_leave_room_for_a_visit_longer_than_its_window(
        self, kind: str
    ) -> None:
        """The visit is an hour and has to begin within half an hour, so it
        is still collecting after its window has closed. The group begins
        after the visit can have ended, not just after the window closes:
        the visit would otherwise have no room and be lost."""
        config = _config()
        targets = _timed_against_group(config, minutes=60)
        plan = _planner(kind, config, targets).schedule()
        _assert_window_kept(plan, minutes=60)

    def test_a_group_with_a_window_of_its_own_is_not_held(self) -> None:
        """Its window says when it runs, so it begins at once, as before."""
        config = _config()
        config.observation_groups.groups[0].window_start = T0
        ditl = _dispatch(config, _timed_against_group(config))
        order = _order(ditl.plan)
        assert order[0] == ORDER[0]
        assert not any(
            "has no window and waits for" in e.description for e in ditl.log.events
        )


def _gaps_in_group(plan: Plan) -> list[float]:
    """The idle time, in seconds, between each two science entries from the
    group's first observation to its last."""
    science = [e for e in _science(plan) if int(e.obsid) in ORDER]
    return [
        float(later.begin) - float(earlier.end)
        for earlier, later in zip(science, science[1:], strict=False)
    ]


def test_a_group_under_way_does_not_wait_for_a_request_it_holds_back() -> None:
    """A time-bound request ranked below the members has to begin while the
    group runs. The group holds it back, so its members must not give way to
    it: they would wait for a request that cannot be observed, and the group
    would stand idle until the request's window had closed."""
    config = _config()
    # The group's own window starts it at once, before the timed window.
    config.observation_groups.groups[0].window_start = T0
    # Each member is one whole visit of 40 minutes, so it cannot be fitted
    # into the time before the timed request's latest start.
    members = [
        make_target(config, k, *PATCH[k], merit=m, minutes=40)
        for k, m in zip((1, 2, 3), (60, 50, 40), strict=True)
    ]
    late = (T0 + 50 * MIN, T0 + 110 * MIN)
    timed = Pointing(
        config=config,
        obsid=TIMED,
        name="Timed",
        ra=PATCH[4][0],
        dec=PATCH[4][1],
        merit=30,
        fom=30,
        ss_min=20 * MIN,
        ss_max=20 * MIN,
        visit_windows=[VisitWindow(start=late[0], deadline=late[1])],
    )
    ditl = _dispatch(config, [*members, timed])
    _assert_consecutive(ditl.plan, ORDER)
    for obsid in ORDER:
        collected = sum(
            float(e.exposure or 0) for e in ditl.plan if int(e.obsid) == obsid
        )
        assert collected == 40 * MIN
    # Slews and settling between the members, and no wait for the timed visit.
    assert max(_gaps_in_group(ditl.plan)) < 15 * MIN
