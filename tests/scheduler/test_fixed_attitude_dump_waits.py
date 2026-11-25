"""A thruster dump at a fixed firing attitude that the Earth's limb blocks for
part of every orbit, in dispatch and in a plan, and safe mode beginning during
an activity that holds an attitude of its own."""

from datetime import timedelta

import pytest
from rust_ephem import EarthLimbConstraint

from conops import create_ditl
from conops.common import ObsType
from conops.config import (
    ActivityAttitude,
    MissionConfig,
    StoredMomentumConfig,
    ThrusterDumpConfig,
)
from conops.schedulers import PriorityScheduler
from conops.targets import ScheduledActivity

from .planning_scenario import (
    BEGIN,
    MIN,
    PATCH,
    T0,
    make_config,
    make_target,
    replay,
)

FIRING = ActivityAttitude(kind="fixed", ra_deg=80, dec_deg=-20, roll_deg=0)
START = BEGIN + timedelta(minutes=70)
"""The firing attitude is blocked from here for about half an hour."""


def config_with_dump(attitude: ActivityAttitude = FIRING) -> MissionConfig:
    config = make_config(8, open_sky=True)
    config.constraint.safety_constraint = EarthLimbConstraint(min_angle=10)
    bus = config.spacecraft_bus
    bus.inertia_tensor_body_kg_m2 = ((1000, 0, 0), (0, 1200, 0), (0, 0, 800))
    bus.attitude_control.stored_momentum = StoredMomentumConfig(
        gravity_gradient_enabled=True,
        capacity_n_m_s=10,
        planning_level=0.6,
        initial_momentum_body_n_m_s=(7, 0, 0),
        thruster_dump=ThrusterDumpConfig(duration_seconds=600, attitude=attitude),
    )
    return config


def dispatch(config, begin, hours, snapshot=20, activities=None):
    target = make_target(config, 1, 100, 0, minutes=600, snapshot=snapshot)
    run = create_ditl(
        config,
        [target],
        begin,
        begin + timedelta(hours=hours),
        activities=activities or [],
    )
    run.step_size = 60
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    return run


def plan(config, begin, hours, activities=None):
    target = make_target(config, 1, 100, 0, minutes=600, snapshot=20)
    scheduler = PriorityScheduler(
        config,
        [target],
        begin,
        begin + timedelta(hours=hours),
        step_size=60,
        activities=activities or [],
    )
    return scheduler, scheduler.schedule()


def dump_minutes(entries, begin) -> list[int]:
    return [
        round((e.activity_hold_begin - begin.timestamp()) / 60)
        for e in entries
        if e.desaturation
    ]


def science_minutes(entries) -> float:
    return (
        sum(
            e.collection_end - e.collection_begin
            for e in entries
            if e.obsid == 1 and e.collection_begin is not None
        )
        / 60
    )


# ── Dispatch ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("snapshot", [20, 5])
def test_dispatch_fires_one_dump_and_observes_while_it_waits(snapshot):
    """While the dump waits for its firing attitude the wheels stay above
    the planning level. Before, each observation refused in that time added
    another dump: eight or eleven fired and science stopped."""
    run = dispatch(config_with_dump(), START, 3, snapshot)
    [fired] = dump_minutes(run.plan, START)
    assert fired > 0
    first = min(e.collection_begin for e in run.plan if e.obsid == 1)
    assert first < START.timestamp() + fired * 60
    assert science_minutes(run.plan) > 120


@pytest.mark.parametrize("minutes", [20, 30])
def test_dispatch_leaves_a_dump_it_cannot_fire_before_the_run_ends(minutes):
    """The firing attitude stays blocked to the end of the run, with the
    wheels well under their capacity. Before, the run stopped with an
    ``ActivitySchedulingError``."""
    run = dispatch(config_with_dump(), START, minutes / 60, snapshot=5)
    assert dump_minutes(run.plan, START) == []
    assert science_minutes(run.plan) > 0
    events = [e.description for e in run.log.events]
    assert sum("no thruster dump before" in d for d in events) == 1


@pytest.mark.parametrize("due", [30, 36])
def test_dispatch_keeps_the_waiting_dump_clear_of_a_fixed_activity(due):
    """An activity with a fixed start and attitude is due where the firing
    attitude first becomes allowed. Before, the dump was moved onto it and
    the activity could not be reserved."""
    hold = ScheduledActivity(
        name="Hold",
        begin=START.timestamp() + due * 60,
        duration_seconds=300,
        fixed=True,
        attitude=ActivityAttitude(kind="fixed", ra_deg=100, dec_deg=0, roll_deg=0),
    )
    run = dispatch(config_with_dump(), START, 3, activities=[hold])
    [held] = [e for e in run.plan if e.name == "Hold"]
    assert held.activity_hold_begin == hold.begin
    [fired] = dump_minutes(run.plan, START)
    assert fired * 60 + START.timestamp() >= hold.begin + hold.duration_seconds


# ── Plans ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("minutes", [20, 38])
def test_a_plan_leaves_a_dump_it_cannot_fire_before_the_horizon(minutes):
    """Before, planning stopped with an ``ActivitySchedulingError``."""
    scheduler, entries = plan(config_with_dump(), START, minutes / 60)
    assert dump_minutes(entries, START) == []
    assert any(
        "the dump is left to the next plan" in e.description
        for e in scheduler.log.events
    )


@pytest.mark.parametrize("minutes", [25, 26, 27])
def test_a_plan_fires_at_arrival_where_the_attitude_holds_for_the_firing(minutes):
    """The firing attitude is allowed for the firing from when the slew
    arrives. Before, it was also asked to be allowed for a slew's time
    longer, and the dump was put off by most of an orbit behind a hold."""
    begin = BEGIN + timedelta(minutes=minutes)
    _, entries = plan(config_with_dump(), begin, 3)
    [fired] = dump_minutes(entries, begin)
    assert fired < 30
    assert not any(e.name == "Hold before thruster dump" for e in entries)
    replay(config_with_dump(), entries, begin=begin, end=begin + timedelta(hours=3))


@pytest.mark.parametrize("due", [26, 29, 32])
def test_a_plan_moves_the_firing_past_a_mandatory_activity(due):
    """A mission activity with a fixed start overlaps the delayed firing or
    the approach to it. Before, that one firing time was the only one tried,
    and planning stopped with an ``ActivitySchedulingError``."""
    config = config_with_dump()
    hold = ScheduledActivity(
        name="Hold",
        begin=START.timestamp() + due * 60,
        duration_seconds=300,
        fixed=True,
    )
    scheduler, entries = plan(config, START, 3, activities=[hold])
    [held] = [e for e in entries if e.name == "Hold"]
    assert held.activity_hold_begin == hold.begin
    [fired] = dump_minutes(entries, START)
    assert fired * 60 + START.timestamp() >= hold.begin + hold.duration_seconds
    assert any(
        "a later firing is looked for" in e.description for e in scheduler.log.events
    )
    replay(config, entries, begin=START, end=START + timedelta(hours=3))


# ── Safe mode during an activity ─────────────────────────────────────────


@pytest.mark.parametrize("fault", [18, 19, 20, 21, 30, 50])
def test_safe_mode_during_a_fixed_attitude_activity_ends_its_hold(monkeypatch, fault):
    """Safe mode begins while the activity slews to its attitude (the slew
    starts at 18 minutes) or holds it. The rest of the hold is not flown, and
    the entry ends there. Before, the entry was recorded as held to its end
    and the run failed its plan and execution check."""
    config = make_config(2, open_sky=True)
    activity = ScheduledActivity(
        name="Hold",
        begin=T0 + 20 * MIN,
        duration_seconds=40 * MIN,
        attitude=ActivityAttitude(kind="fixed", ra_deg=150, dec_deg=20, roll_deg=0),
        fixed=True,
    )
    target = make_target(config, 100, *PATCH[0], minutes=60, snapshot=20)
    run = create_ditl(
        config, [target], BEGIN, BEGIN + timedelta(hours=2), activities=[activity]
    )
    run.step_size = 60
    management = type(run.fault_management)
    check = management.check

    def forced(self, housekeeping, acs):
        check(self, housekeeping=housekeeping, acs=acs)
        if housekeeping.timestamp.timestamp() >= T0 + fault * MIN:
            self.safe_mode_requested = True

    monkeypatch.setattr(management, "check", forced)
    assert run.calc()
    assert run.acs.in_safe_mode
    assert run.validate_plan_matches_execution() == []
    [entry] = [e for e in run.plan if e.obstype == ObsType.ACTIVITY]
    assert not entry.activity_finishes
    assert float(entry.end) == T0 + (fault + 1) * MIN
    assert any("Safe mode began during" in e.description for e in run.log.events)
