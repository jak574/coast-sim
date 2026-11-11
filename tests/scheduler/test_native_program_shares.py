"""Exercise delivered shares through the integrated native constructor."""

import numpy as np
import pytest
from ortools.sat.python import cp_model

from conops.config.observation_categories import ObservationCategory
from conops.schedulers._delivered_shares import NodeScience
from conops.schedulers.cpsat_scheduler import _ChunkModel, _Span, _Task, _Urgency
from conops.targets.merit import MeritModel

POSE = (0.0, 0.0, 0.0)


def rate(base, program, history):
    delivered = sum(history.values())
    shares = (
        {p: seconds / delivered for p, seconds in history.items()} if delivered else {}
    )
    category = ObservationCategory(
        name=program, obsid_min=1, obsid_max=2, time_share=0.5
    )
    return base + 40 * MeritModel.completion_deficit(category, shares)


def solve(chunk):
    assert chunk.model.validate() == ""
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 2
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    return solver


def scenario(
    history,
    *,
    hold=False,
    budget=600,
    allocated=False,
    urgent=0.0,
    discount=0.0,
    offset=0.0,
):
    first = _Task(
        (1, 0),
        (POSE, POSE),
        (0, 0),
        collection=(600, 600),
        window=(0, 1260),
        rate=rate(120, "A", history),
        science=NodeScience(program="A", allocated_share=0.5, weight=40),
    )
    tasks = [_Task(None, (POSE, POSE), (0, 0)), first]
    if hold:
        tasks.append(
            _Task(None, (POSE, POSE), (600, 600), duration=60, inherits_attitude=True)
        )
    arrival = 660 if hold else 600
    ids = {}
    for obsid, program, base in ((2, "A", 105), (3, "B", 100)):
        ids[obsid] = len(tasks)
        value = rate(base, program, history)
        urgency = (
            _Urgency(values=(1,), scale=1, weight=urgent)
            if obsid == 2 and urgent
            else None
        )
        # No start-time lookup is needed for a constant urgency; represent all
        # legal start steps so the normal native element remains in range.
        if urgency is not None:
            urgency = urgency.model_copy(
                update={"values": tuple([1] * (arrival // 60 + 1))}
            )
        tasks.append(
            _Task(
                (obsid, 0),
                (POSE, POSE),
                (arrival, arrival),
                collection=(600, 600),
                window=(0, arrival + 600),
                rate=value,
                earliness=value * discount,
                offset=value * offset,
                spans=[_Span((obsid, 0), (arrival, arrival), value)]
                if allocated
                else [],
                urgency=urgency,
                science=NodeScience(
                    program=program,
                    allocated_share=0.5,
                    weight=40,
                    earliness=discount,
                    offset=offset,
                    allocation_bonus=1 if allocated else 0,
                ),
            )
        )
    chunk = _ChunkModel(
        tasks,
        np.zeros((len(tasks), len(tasks)), dtype=np.int64),
        60,
        arrival + 600,
        {1: 600, 2: 600, 3: budget},
        allocated_left={(2, 0): 600, (3, 0): 600} if allocated else {},
        delivered_history=history,
    )
    chunk.model.add(chunk.present[1] == 1)
    return chunk, ids, arrival


@pytest.mark.parametrize(
    "history", [{}, {"A": 600}, {"B": 600}, {"A": 1200.25, "B": 600.125}, {"C": 900.5}]
)
@pytest.mark.parametrize("hold", [False, True])
@pytest.mark.parametrize("budget", [0, 600])
def test_second_native_choice_uses_delivered_history(history, hold, budget):
    chunk, ids, _ = scenario(history, hold=hold, budget=budget)
    solver = solve(chunk)
    updated = dict(history)
    updated["A"] = updated.get("A", 0) + 600
    values = {2: rate(105, "A", updated), 3: rate(100, "B", updated)}
    choices = [2] if budget == 0 else [2, 3]
    chosen = next(
        obsid for obsid in choices if solver.boolean_value(chunk.present[ids[obsid]])
    )
    assert values[chosen] == max(values[k] for k in choices)
    assert solver.objective_value == pytest.approx(
        600 * (rate(120, "A", history) + values[chosen]), abs=0.1
    )
    assert chunk.delivered_variables is not None


@pytest.mark.parametrize("fixed_before", [False, True])
@pytest.mark.parametrize("science", [False, True])
def test_fixed_science_is_counted_at_its_real_position(fixed_before, science):
    start = 600 if fixed_before else 0
    fixed_start = 0 if fixed_before else 1200
    fixed_node = (
        NodeScience(program="A", fixed_seconds=600) if science else NodeScience()
    )
    tasks = [
        _Task(None, (POSE, POSE), (0, 0)),
        _Task(
            None,
            (POSE, POSE),
            (fixed_start, fixed_start),
            duration=600,
            science=fixed_node,
        ),
    ]
    for obsid, program, base in ((2, "A", 105), (3, "B", 100)):
        tasks.append(
            _Task(
                (obsid, 0),
                (POSE, POSE),
                (start, start),
                collection=(600, 600),
                window=(0, 1800),
                rate=rate(base, program, {}),
                science=NodeScience(program=program, allocated_share=0.5, weight=40),
            )
        )
    chunk = _ChunkModel(
        tasks,
        np.zeros((4, 4), dtype=np.int64),
        60,
        1800,
        {2: 600, 3: 600},
        delivered_history={},
    )
    solver = solve(chunk)
    chosen = 3 if fixed_before and science else 2
    assert solver.boolean_value(chunk.present[chosen])
    history = {"A": 600} if fixed_before and science else {}
    assert solver.objective_value == pytest.approx(
        600 * rate(100 if chosen == 3 else 105, "B" if chosen == 3 else "A", history)
    )


@pytest.mark.parametrize("bonus", [10.0, 50.0])
def test_existing_native_urgency_is_preserved(bonus):
    chunk, ids, _ = scenario({}, urgent=bonus)
    solver = solve(chunk)
    chosen = 2 if 85 + bonus > 120 else 3
    assert solver.boolean_value(chunk.present[ids[chosen]])
    assert solver.objective_value == pytest.approx(84000 + 600 * max(85 + bonus, 120))


def test_allocated_seconds_receive_dynamic_completion_value():
    chunk, ids, _ = scenario({}, allocated=True)
    solver = solve(chunk)
    assert solver.boolean_value(chunk.present[ids[3]])
    assert solver.objective_value == pytest.approx(84000 + 600 * 120 * 2)


@pytest.mark.parametrize("offset", [0.0, -0.1])
def test_deadline_discount_applies_to_the_new_dynamic_term(offset):
    chunk, ids, arrival = scenario({}, discount=0.0005, offset=offset)
    solver = solve(chunk)
    assert solver.boolean_value(chunk.present[ids[3]])
    assert solver.objective_value == pytest.approx(
        84000 + 600 * 120 * (1 + offset - 0.0005 * arrival)
    )


@pytest.mark.parametrize("hold", [False, True])
@pytest.mark.parametrize("allocated", [False, True])
def test_extended_route_hint_is_feasible_when_every_hint_is_fixed(hold, allocated):
    from conops.schedulers.cpsat_scheduler import _TaskHint

    chunk, ids, arrival = scenario(
        {"A": 600.25, "B": 300.125}, hold=hold, allocated=allocated
    )
    hint = {1: _TaskHint(0, 0, 600)}
    if hold:
        hint[2] = _TaskHint(10, 600, 0)
    hint[ids[3]] = _TaskHint(arrival // 60, arrival, 600)
    chunk.add_hint(hint)
    variables = chunk.model.proto.solution_hint.vars
    assert len(variables) == len(set(variables)), (
        "A native variable must receive only one hint"
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 2
    solver.parameters.fix_variables_to_their_hinted_value = True
    assert solver.solve(chunk.model) == cp_model.OPTIMAL
    assert solver.boolean_value(chunk.present[ids[3]])
