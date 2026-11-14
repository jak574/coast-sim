"""Reuse expensive sequence geometry without reusing executed request history."""

from datetime import timedelta

import pytest
from pydantic import PrivateAttr

from conops.config import MissionConfig, ObservationTiming
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import ObservationSequence, Pointing, SequenceStep, VisitWindow

from .planning_scenario import BEGIN, PATCH, T0, make_config
from .test_instrument_modes import _moded_config


def target(config: MissionConfig) -> Pointing:
    return Pointing(
        config=config,
        obsid=750,
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        visits=3,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="step", duration_seconds=60, science=False),
                SequenceStep(name="step", duration_seconds=120),
            ]
        ),
    )


def planner(config: MissionConfig, request: Pointing) -> PriorityScheduler:
    return PriorityScheduler(config, [request], BEGIN, BEGIN + timedelta(hours=2))


@pytest.mark.parametrize(
    "kind", [PriorityScheduler, LocalSearchScheduler, CpSatScheduler]
)
def test_repeated_sequence_attempts_construct_fewer_steps_with_exact_plans(
    monkeypatch: pytest.MonkeyPatch, kind: type[PriorityScheduler]
) -> None:
    class Uncached(kind):
        def _sequence_pointing(self, request: Pointing, index: int) -> Pointing:
            return request.step_pointing(index)

    original = Pointing.step_pointing
    calls: list[int] = []

    def counted(self: Pointing, index: int) -> Pointing:
        calls.append(index)
        return original(self, index)

    monkeypatch.setattr(Pointing, "step_pointing", counted)
    options: dict = {}
    if kind is LocalSearchScheduler:
        options = dict(max_iterations=10, time_limit=60, seed=1234)
    elif kind is CpSatScheduler:
        options = dict(
            max_iterations=0,
            solver_time_limit=3,
            solver_deterministic_time=0.01,
            workers=1,
            seed=1234,
        )
    counts = []
    plans = []
    for scheduler_type in (Uncached, kind):
        config = make_config(2)
        config.payload.observation_timing = ObservationTiming()
        request = target(config)
        scheduler = scheduler_type(
            config, [request], BEGIN, BEGIN + timedelta(hours=2), **options
        )
        calls.clear()
        plan = scheduler.schedule()
        counts.append(len(calls))
        plans.append([entry.model_dump(mode="json") for entry in plan])
        assert request.collected_seconds == 0 and not request.visits_done
    assert plans[0] == plans[1]
    assert plans[0]
    assert counts[1] < counts[0]


def test_public_step_factory_still_returns_independent_pointings() -> None:
    request = target(make_config(2))
    a, b = request.step_pointing(0), request.step_pointing(0)
    assert a is not b and a.windows == b.windows
    a.windows.clear()
    assert b.windows


def test_repeated_step_template_preserves_factory_values_and_parent_history() -> None:
    config = make_config(2)
    request = target(config)
    scheduler = planner(config, request)
    a = scheduler._sequence_pointing(request, 1)
    b = scheduler._sequence_pointing(request, 1)
    fresh = request.step_pointing(1)
    assert a is b and a.model_dump(mode="json") == fresh.model_dump(mode="json")
    assert a.windows == fresh.windows
    assert request.collected_seconds == 0 and not request.visits_done


def test_duplicate_step_names_keep_instrument_modes_and_durations_separate() -> None:
    config = _moded_config(2)
    request = target(config)
    request.sequence = ObservationSequence(
        steps=[
            SequenceStep(
                name="same",
                duration_seconds=60,
                science=False,
                instrument_name="camera",
                instrument_modes={"camera": "IDLE"},
            ),
            SequenceStep(
                name="same",
                duration_seconds=120,
                instrument_name="spectrograph",
                instrument_modes={"spectrograph": "SCIENCE"},
            ),
        ]
    )
    scheduler = planner(config, request)
    a = scheduler._sequence_pointing(request, 0)
    b = scheduler._sequence_pointing(request, 1)
    assert a is not b
    assert (a.instrument_name, a.instrument_modes, a.ss_max, a.calibration) == (
        "camera",
        {"camera": "IDLE"},
        60,
        True,
    )
    assert (b.instrument_name, b.instrument_modes, b.ss_max, b.calibration) == (
        "spectrograph",
        {"spectrograph": "SCIENCE"},
        120,
        False,
    )


def test_window_pointings_and_roll_requests_remain_separate() -> None:
    config = make_config(2)
    request = target(config)
    request.visit_windows = [
        VisitWindow(start=T0, deadline=T0 + 600, ra=105, dec=10, roll_deg=10),
        VisitWindow(start=T0 + 1200, deadline=T0 + 1800, ra=110, dec=10, roll_deg=20),
    ]
    scheduler = planner(config, request)
    visits = [request.visit_pointing(index) for index in (0, 1)]
    steps = [scheduler._sequence_pointing(visit, 1) for visit in visits]
    assert steps[0] is not steps[1]
    assert (steps[0].ra, steps[0].dec, steps[0].roll_deg, steps[0].visit) == (
        105,
        10,
        10,
        0,
    )
    assert (steps[1].ra, steps[1].dec, steps[1].roll_deg, steps[1].visit) == (
        110,
        10,
        20,
        1,
    )
    for visit, step in zip(visits, steps, strict=True):
        fresh = visit.step_pointing(1)
        assert step.model_dump(mode="json") == fresh.model_dump(mode="json")
        assert step.windows == fresh.windows


def test_new_pass_rebuilds_steps_with_changed_timing_and_metadata() -> None:
    config = make_config(2)
    request = target(config)
    scheduler = planner(config, request)
    scheduler.schedule()
    a = scheduler._sequence_pointing(request, 1)
    request.sequence = ObservationSequence(
        steps=[
            SequenceStep(name="updated", duration_seconds=240),
            SequenceStep(name="changed", duration_seconds=300, setup=True),
        ]
    )
    request.merit = 123
    request.exptime = 3 * 540
    config.payload.observation_timing = ObservationTiming(setup_seconds=90)
    scheduler._requests()
    b = scheduler._sequence_pointing(request, 1)
    assert a is not b
    assert b.ss_max == 300 and b.merit == 123
    fresh = request.step_pointing(1)
    assert b.model_dump(mode="json") == fresh.model_dump(mode="json")
    assert b.observation_timing == fresh.observation_timing


@pytest.mark.parametrize(
    "field,value",
    [("ra", 110), ("dec", 15), ("roll_deg", 15), ("instrument_name", "spectrograph")],
)
def test_changed_pointing_roll_or_instrument_cannot_reuse_the_old_step(
    field: str, value: float | str
) -> None:
    config = _moded_config(2)
    request = target(config)
    request.instrument_name = "camera"
    scheduler = planner(config, request)
    old = scheduler._sequence_pointing(request, 1)
    setattr(request, field, value)
    changed = scheduler._sequence_pointing(request, 1)
    fresh = request.step_pointing(1)
    assert changed is not old
    assert changed.model_dump(mode="json") == fresh.model_dump(mode="json")
    assert changed.windows == fresh.windows


def test_custom_pointing_factories_keep_their_fresh_call_behavior() -> None:
    class CustomPointing(Pointing):
        _calls: int = PrivateAttr(default=0)

        def step_pointing(self, index: int) -> Pointing:
            self._calls += 1
            return super().step_pointing(index)

    config = make_config(2)
    request = CustomPointing(
        config=config,
        obsid=751,
        ra=105,
        dec=10,
        sequence=ObservationSequence(
            steps=[SequenceStep(name="a", duration_seconds=60)]
        ),
    )
    scheduler = planner(config, request)
    a = scheduler._sequence_pointing(request, 0)
    b = scheduler._sequence_pointing(request, 0)
    assert a is not b and request._calls == 2


@pytest.mark.parametrize("field", ["ra", "dec", "roll_deg", "roll_offset_deg"])
def test_reuse_preserves_signed_zero_in_serialized_coordinates(field: str) -> None:
    config = make_config(2)
    request = target(config)
    scheduler = planner(config, request)
    setattr(request, field, 0.0)
    positive = scheduler._sequence_pointing(request, 1)
    setattr(request, field, -0.0)
    negative = scheduler._sequence_pointing(request, 1)
    fresh = request.step_pointing(1)
    assert negative is not positive
    assert negative.model_dump_json() == fresh.model_dump_json()
