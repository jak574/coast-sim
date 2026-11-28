"""Reproduce horizon-origin urgency being frozen onto a later observation."""

from datetime import timedelta

import pytest

from conops.config import TargetConfig
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import Pointing
from conops.targets.merit import MeritModel
from tests.scheduler.planning_scenario import BEGIN, PATCH, T0, make_config


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_recorded_value_uses_the_planned_decision_time(kind):
    config = make_config(
        3, weights=TargetConfig(urgency_weight=100, urgency_timescale_seconds=60)
    )
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    target = Pointing(
        config=config,
        obsid=1,
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        merit=100,
        ss_min=600,
        ss_max=600,
        earliest_start=T0 + 1800,
        deadline=T0 + 3600,
    )
    scheduler, options = {
        "priority": (PriorityScheduler, {}),
        "local": (LocalSearchScheduler, dict(time_limit=0, max_iterations=0)),
        "cp_sat": (
            CpSatScheduler,
            dict(
                time_limit=0,
                solver_time_limit=0.1,
                solver_deterministic_time=0.001,
                workers=1,
            ),
        ),
    }[kind]
    plan = scheduler(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=3),
        include_passes=False,
        **options,
    ).schedule()
    [entry] = [e for e in plan if int(e.obsid) == 1]
    assert float(entry.begin) > T0
    expected = (
        MeritModel(config)
        .value_terms(target, float(entry.begin), delivered_shares={})
        .value
    )
    assert entry.merit == pytest.approx(expected), dict(
        recorded=entry.merit,
        shared_model_at_planned_time=expected,
        horizon_origin_value=MeritModel(config)
        .value_terms(target, T0, delivered_shares={})
        .value,
        planned_decision_time=float(entry.begin),
    )


def _native_urgency_case(*, earliness=0.0, budget=1, allocated=False):
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    import numpy as np

    from conops.schedulers.cpsat_scheduler import _ChunkModel, _Span, _Task, _Urgency

    pose = (0.0, 0.0, 0.0)
    urgency = _Urgency(
        values=(100, 500, 1000),
        scale=1000,
        weight=100,
        earliness=earliness,
        allocation_bonus=1 if allocated else 0,
    )
    tasks = [
        _Task(None, (pose, pose), (0, 0)),
        _Task(
            (1, 0),
            (pose, pose),
            (0, 2),
            collection=(1, 1),
            window=(0, 3),
            urgency=urgency,
            spans=[_Span((1, 0), (2, 2), 0)] if allocated else [],
        ),
        _Task((2, 0), (pose, pose), (0, 0), collection=(3, 3), window=(0, 3), rate=20),
    ]
    chunk = _ChunkModel(
        tasks,
        np.zeros((3, 3), dtype=np.int64),
        1,
        3,
        {1: budget, 2: 3},
        allocated_left={(1, 0): 1} if allocated else {},
    )
    assert chunk.model.validate() == ""
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 2
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    return solver, chunk


@pytest.mark.parametrize(
    "discount,expected,urgent", [(0, 100, True), (0.125, 75, True), (0.25, 60, False)]
)
def test_native_objective_chooses_by_value_at_the_decision(discount, expected, urgent):
    solver, chunk = _native_urgency_case(earliness=discount)
    assert solver.objective_value == pytest.approx(expected)
    assert solver.boolean_value(chunk.present[1]) is urgent
    if urgent:
        assert solver.value(chunk.slew_step[1]) == 2


def test_absent_candidate_earns_no_urgency_value():
    solver, chunk = _native_urgency_case(budget=0)
    assert not solver.boolean_value(chunk.present[1])
    assert solver.value(chunk.urgency_counted[1]) == 0
    assert solver.objective_value == 60


def test_allocated_seconds_receive_the_same_time_dependent_urgency():
    solver, chunk = _native_urgency_case(allocated=True)
    assert solver.boolean_value(chunk.present[1])
    assert solver.objective_value == 200


def test_native_rates_follow_the_shared_function_with_bounded_integer_error():
    config = make_config(
        3, weights=TargetConfig(urgency_weight=100, urgency_timescale_seconds=60)
    )
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    target = Pointing(
        config=config,
        obsid=1,
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        merit=100,
        ss_min=600,
        ss_max=600,
        earliest_start=T0 + 1800,
        deadline=T0 + 3600,
    )
    scheduler = CpSatScheduler(
        config, [target], BEGIN, BEGIN + timedelta(hours=3), include_passes=False
    )
    scheduler._reserve_fixed()
    [request] = scheduler._requests()
    urgency = scheduler._candidate_urgency(
        request,
        T0,
        T0 + 3600,
        (T0 + 1800, T0 + 3600),
        (T0, T0 + 10800),
        None,
        600,
        3000,
        0,
    )
    assert urgency is not None
    model = MeritModel(config)
    for index, value in enumerate(urgency.values):
        expected = (
            model.value_terms(target, T0 + index * scheduler.ctx.step_size).urgency
            / 100
        )
        assert -1e-12 <= expected - value / urgency.scale <= 1 / urgency.scale + 1e-12


@pytest.mark.parametrize("allocation_factor", [1.0, 1.5])
def test_identical_decisions_use_exactly_the_shared_term_order(allocation_factor):
    config = make_config(
        3, weights=TargetConfig(urgency_weight=100, urgency_timescale_seconds=1000)
    )
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    target = Pointing(
        config=config,
        obsid=1,
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        merit=100,
        ss_min=600,
        ss_max=600,
        deadline=T0 + 20001,
    )
    scheduler = PriorityScheduler(
        config, [target], BEGIN, BEGIN + timedelta(hours=3), include_passes=False
    )
    scheduler._reserve_fixed()
    [request] = scheduler._requests()
    request.value_factor = allocation_factor
    request.steady_value *= allocation_factor
    at = T0 + 578
    entry = target.model_copy(
        update=dict(
            begin=at,
            end=at + 600,
            collection_begin=at,
            collection_end=at + 600,
            windows=[list(w) for w in request.windows],
        )
    )
    window = next(w for w in entry.windows if w[0] <= at < w[1])
    expected = MeritModel(config).value_terms(
        target, at, visibility_window=window, delivered_shares={}
    )
    expected = expected.model_copy(
        update={"allocation": (allocation_factor - 1) * expected.value}
    )
    assert scheduler._value(request, {}, entry) == expected.value


def _delivered_share_case(kind, *, future=True, budget=1200):
    from conops.common import ObsType
    from conops.config.observation_categories import ObservationCategory
    from conops.targets import PlanEntry
    from tests.scheduler.planning_scenario import make_target

    config = make_config(
        3,
        weights=TargetConfig(completion_deficit_weight=100),
        categories=[
            ObservationCategory(name="A", obsid_min=100, obsid_max=200, time_share=0.5),
            ObservationCategory(
                name="B",
                obsid_min=200,
                obsid_max=300,
                time_share=0.5,
                time_budget_seconds=budget,
            ),
        ],
    )
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    past = make_target(config, 100, *PATCH[0], minutes=10, snapshot=10)
    past.record_collection(T0 - 600, 600)
    reserved = make_target(config, 200, *PATCH[0], minutes=10, snapshot=10)
    target = make_target(
        config,
        201,
        *PATCH[0],
        minutes=10,
        snapshot=10,
        earliest_start=None if future else T0 + 2700,
    )
    at = T0 + (7200 if future else 600)
    locked = PlanEntry(
        config=config,
        name="Reserved science",
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        obsid=200,
        obstype=ObsType.AT,
        begin=at,
        end=at + 600,
        collection_begin=at,
        collection_end=at + 600,
        ss_min=60,
        merit=75,
    )
    cls, options = {
        "priority": (PriorityScheduler, {}),
        "local": (LocalSearchScheduler, dict(max_iterations=0, time_limit=0)),
        "cp_sat": (
            CpSatScheduler,
            dict(
                solver_time_limit=1,
                solver_deterministic_time=0.01,
                workers=1,
                time_limit=0,
            ),
        ),
    }[kind]
    scheduler = cls(
        config,
        [past, reserved, target],
        BEGIN,
        BEGIN + timedelta(hours=3),
        locked=[locked],
        include_passes=False,
        **options,
    )
    return scheduler, target, locked


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
@pytest.mark.parametrize("future", [True, False])
@pytest.mark.parametrize("budget", [900, 1200])
def test_delivered_merit_and_future_budget_reservations_are_separate(
    kind, future, budget
):
    scheduler, target, locked = _delivered_share_case(
        kind, future=future, budget=budget
    )
    plan = scheduler.schedule()
    [reservation] = [entry for entry in plan if int(entry.obsid) == 200]
    assert reservation.collection_begin == locked.collection_begin
    assert reservation.collection_end == locked.collection_end
    assert reservation.merit == locked.merit == 75
    entries = [entry for entry in plan if int(entry.obsid) == 201]
    if budget == 900:
        # The future 600 seconds still reserve the category's budget. The
        # remaining 300 seconds cannot admit a whole 600-second snapshot.
        assert not entries
        return
    [entry] = entries
    shares = {"A": 1.0} if future else {"A": 0.5, "B": 0.5}
    expected = (
        MeritModel(scheduler.config)
        .value_terms(target, float(entry.begin), delivered_shares=shares)
        .value
    )
    assert entry.merit == expected
    assert (float(entry.end) <= float(locked.begin)) is future
    assert sum(e.collection_seconds_between(e.begin, e.end) for e in plan) == 1200


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_values_follow_chronology_when_later_science_is_placed_first(kind):
    from tests.scheduler.planning_scenario import make_target

    scheduler, _, _ = _delivered_share_case(kind)
    config = scheduler.config
    late = make_target(
        config,
        201,
        *PATCH[0],
        merit=100,
        minutes=10,
        snapshot=10,
        earliest_start=T0 + 7200,
    )
    early = make_target(
        config, 202, *PATCH[0], merit=50, minutes=10, snapshot=10, deadline=T0 + 3600
    )
    scheduler.locked = []
    scheduler.targets = [scheduler.targets[0], late, early]
    plan = scheduler.schedule()
    [first, second] = sorted(plan, key=lambda entry: float(entry.begin))
    assert (int(first.obsid), int(second.obsid)) == (202, 201)
    assert (
        first.merit
        == MeritModel(config)
        .value_terms(early, float(first.begin), delivered_shares={"A": 1.0})
        .value
    )
    assert (
        second.merit
        == MeritModel(config)
        .value_terms(late, float(second.begin), delivered_shares={"A": 0.5, "B": 0.5})
        .value
    )


def test_native_candidate_value_excludes_science_reserved_for_later():
    scheduler, target, _ = _delivered_share_case("cp_sat")
    scheduler._reserve_fixed()
    scheduler._start_from_priority_plan()
    state = scheduler._initial_state()
    tasks = scheduler._candidates(
        state, dict(state.remaining), T0, T0 + 3600, T0 + 7200, {}
    )
    [task] = [task for task in tasks if task.obsid == int(target.obsid)]
    expected = (
        MeritModel(scheduler.config)
        .value_terms(target, T0, delivered_shares={"A": 1.0})
        .value
    )
    assert task.rate == expected
