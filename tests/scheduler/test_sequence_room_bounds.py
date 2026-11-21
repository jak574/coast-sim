"""Whole sequences must fit without rebuilding impossible late visits."""

from datetime import timedelta
from math import inf

import pytest

from conops.config import ObservationTiming
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.schedulers.priority_scheduler import StartState
from conops.targets import ObservationSequence, SequenceStep

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target


def scenario(durations=(4 * HOUR,) * 3, setup=600.0, cleanup=300.0, opening=20 * HOUR):
    config = make_config(24)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.attitude_control.settle_time = 0
    config.payload.observation_timing = ObservationTiming(
        setup_seconds=setup, cleanup_seconds=cleanup, handoff_seconds=0.125
    )
    target = make_target(
        config,
        100,
        105,
        10,
        minutes=sum(durations) / 60,
        earliest_start=T0 + opening,
        deadline=T0 + 23 * HOUR,
    )
    target.roll_deg = 0
    target.sequence = ObservationSequence(
        steps=[
            SequenceStep(
                name=f"s{i}",
                duration_seconds=duration,
                setup=i % 2 == 0,
                science=i % 3 != 1,
            )
            for i, duration in enumerate(durations)
        ]
    )
    target.exptime = target.sequence.science_seconds
    target.ss_min = target.ss_max = target.exptime
    return config, target


@pytest.mark.parametrize(
    "kind", [PriorityScheduler, LocalSearchScheduler, CpSatScheduler]
)
def test_impossible_sequence_skips_visit_construction(kind, monkeypatch):
    config, target = scenario()

    def forbidden(*args, **kwargs):
        pytest.fail("A twelve-hour sequence cannot fit after hour twenty")

    monkeypatch.setattr(PriorityScheduler, "_visit", forbidden)
    options = (
        {}
        if kind is PriorityScheduler
        else dict(time_limit=0, max_iterations=0, seed=2029)
    )
    if kind is CpSatScheduler:
        options.update(
            chunk=timedelta(hours=6),
            solver_time_limit=30,
            solver_deterministic_time=20,
            workers=1,
        )
    scheduler = kind(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=24),
        include_passes=False,
        **options,
    )
    assert not scheduler.schedule().entries


@pytest.mark.parametrize(
    "durations", [(1e-7,) * 16, (1.1, 2.3, 3.7), (HOUR, 2 * HOUR, 3 * HOUR)]
)
@pytest.mark.parametrize("setup", [0.0, 0.0000001, 0.3, 600.125])
def test_lower_bound_does_not_exceed_actual_repeated_clock_additions(durations, setup):
    config, target = scenario(durations=durations, setup=setup, cleanup=0.1, opening=0)
    start = T0 + 0.1234567
    timing = config.payload.observation_timing
    # The true no-motion clock, following the sequence's actual operation
    # order. This deliberately retains each IEEE rounding at a large epoch.
    actual = start + timing.setup_seconds
    for index, step in enumerate(target.sequence.steps):
        if index > 0 and step.setup:
            actual += timing.setup_seconds
        actual += step.duration_seconds
    actual += timing.post_collection_seconds
    assert PriorityScheduler._minimum_sequence_end(target, timing, start) <= actual


@pytest.mark.parametrize("setup", [0.0, 0.3, 600.125])
def test_exact_horizon_boundary_preserves_executable_sequence(setup, monkeypatch):
    config, target = scenario(
        durations=(HOUR, 2 * HOUR, 3 * HOUR), setup=setup, opening=0
    )
    options = dict(
        include_passes=False, start_state=StartState(time=T0, attitude=(105, 10, 0))
    )
    baseline = PriorityScheduler(
        config, [target], BEGIN, BEGIN + timedelta(hours=24), **options
    ).schedule()
    assert len(baseline.entries) == 3
    end = baseline.entries[-1].end
    baseline_entries = [entry.model_dump(mode="json") for entry in baseline]
    bounded = PriorityScheduler(
        config, [target], BEGIN, BEGIN + timedelta(seconds=end - T0), **options
    ).schedule()
    assert [entry.model_dump(mode="json") for entry in bounded] == baseline_entries
    # A bounded run remains equivalent when pruning is disabled, including
    # its actual slew, roll, collection and engineering timing fields.
    monkeypatch.setattr(
        PriorityScheduler, "_minimum_sequence_end", staticmethod(lambda *args: -inf)
    )
    control = PriorityScheduler(
        config, [target], BEGIN, BEGIN + timedelta(seconds=end - T0), **options
    ).schedule()
    assert [entry.model_dump(mode="json") for entry in control] == baseline_entries


@pytest.mark.parametrize(
    "kind", [PriorityScheduler, LocalSearchScheduler, CpSatScheduler]
)
def test_calibration_duration_also_bounds_gap_before_mandatory_hold(kind, monkeypatch):
    from conops.targets import ScheduledActivity

    config, target = scenario(opening=0)
    # Only eight science hours but twelve physical hours: the ten-hour gap
    # is large enough for the exposure and too small for the whole visit.
    hold = ScheduledActivity(
        name="engineering", begin=T0 + 10 * HOUR, duration_seconds=14 * HOUR, fixed=True
    )

    def forbidden(*args, **kwargs):
        pytest.fail("The whole visit cannot precede the required hold")

    monkeypatch.setattr(PriorityScheduler, "_visit", forbidden)
    options = (
        {}
        if kind is PriorityScheduler
        else dict(time_limit=0, max_iterations=0, seed=2029)
    )
    if kind is CpSatScheduler:
        options.update(
            chunk=timedelta(hours=6),
            solver_time_limit=30,
            solver_deterministic_time=20,
            workers=1,
        )
    scheduler = kind(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=24),
        activities=[hold],
        include_passes=False,
        **options,
    )
    plan = scheduler.schedule()
    assert len(plan.entries) == 1
    entry = plan.entries[0]
    assert entry.activity_hold_begin == hold.begin
    assert entry.activity_hold_end == hold.end
