"""Route-dependent delivered-share correction for a native CP-SAT chunk.

Private integration work. The helper preserves the original objective, then
replaces only its chunk-origin program fractions at each native decision.
History and fixed science are represented in milliseconds; fractions have a
bounded integer scale. The scheduler's exact decoder remains authoritative.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ortools.sat.python import cp_model

    from conops.schedulers.cpsat_scheduler import _ChunkModel, _TaskHint

Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Share = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class NodeScience(BaseModel):
    model_config = ConfigDict(frozen=True)
    program: str | None = None
    fixed_seconds: Nonnegative = 0.0
    allocated_share: Share | None = None
    weight: Nonnegative = 0.0
    earliness: Nonnegative = 0.0
    offset: float = Field(default=0.0, allow_inf_nan=False)
    late: Nonnegative = 0.0
    allocation_bonus: Nonnegative = 0.0

    @model_validator(mode="after")
    def require_program(self) -> NodeScience:
        if (
            self.fixed_seconds or self.allocated_share is not None
        ) and self.program is None:
            raise ValueError("Science and an allocated share require a program")
        return self


class DeliveredShares(BaseModel):
    model_config = ConfigDict(frozen=True)
    nodes: tuple[NodeScience, ...]
    history: dict[str, Nonnegative] = Field(default_factory=dict)
    time_scale: int = Field(default=1000, ge=1)
    fraction_scale: int = Field(default=1_000_000, ge=1)


Variable = TypeVar("Variable")


class ProgramCounters(BaseModel, Generic[Variable]):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    context: DeliveredShares
    before: dict[tuple[int, str], Variable]
    after: dict[tuple[int, str], Variable]
    fractions: dict[int, Variable]
    products: dict[str, Variable]
    factors: dict[str, tuple[Variable, Variable]]
    predecessors: dict[int, Variable]
    fraction_scale: int
    time_scale: int


def original_objective(model: cp_model.CpModel) -> list[cp_model.ObjLinearExprT]:
    """Read public proto fields without losing urgency or allocation terms."""
    proto = model.proto
    if proto.has_floating_point_objective():
        original = proto.floating_point_objective
        assert original.maximize
        return [
            original.offset,
            *[
                c * model.get_int_var_from_proto_index(i)
                for i, c in zip(original.vars, original.coeffs, strict=True)
            ],
        ]
    original = proto.objective
    scale = original.scaling_factor or 1.0
    return [
        scale * original.offset,
        *[
            scale * c * model.get_int_var_from_proto_index(i)
            for i, c in zip(original.vars, original.coeffs, strict=True)
        ],
    ]


def add_delivered_shares(
    chunk: _ChunkModel, context: DeliveredShares
) -> ProgramCounters[cp_model.IntVar]:
    from ortools.sat.python import cp_model

    tasks, model, nodes = chunk.tasks, chunk.model, context.nodes
    assert len(nodes) == len(tasks) and nodes[0] == NodeScience()
    assert all(
        not task.optional or node.fixed_seconds == 0
        for task, node in zip(tasks, nodes, strict=True)
    )
    programs = sorted(
        set(context.history)
        | {node.program for node in nodes if node.program is not None}
    )
    initial = {
        p: math.floor(context.history.get(p, 0.0) * context.time_scale)
        for p in programs
    }
    fixed = [math.floor(node.fixed_seconds * context.time_scale) for node in nodes]
    total_bound = (
        sum(initial.values())
        + sum(fixed)
        + sum(t.collection[1] * context.time_scale for t in tasks)
    )
    scale = min(context.fraction_scale, (2**60) // max(1, total_bound))
    if scale < 1:
        raise ValueError("Delivered science exceeds the native integer range")
    terms = original_objective(model)
    before: dict[tuple[int, str], cp_model.IntVar] = {}
    after: dict[tuple[int, str], cp_model.IntVar] = {}
    for program in programs:
        bound = initial[program] + sum(
            fixed[k] + tasks[k].collection[1] * context.time_scale
            for k, node in enumerate(nodes)
            if node.program == program
        )
        before[(0, program)] = after[(0, program)] = model.new_constant(
            initial[program]
        )
        for k, task in enumerate(tasks[1:], start=1):
            before[(k, program)] = model.new_int_var(
                0, bound, f"delivered_before{k}_{program}"
            )
            after[(k, program)] = model.new_int_var(
                0, bound, f"delivered_after{k}_{program}"
            )
            added: cp_model.LinearExpr | int = 0
            if nodes[k].program == program:
                added = (
                    context.time_scale * chunk.counted[k] if task.optional else fixed[k]
                )
            model.add(after[(k, program)] == before[(k, program)] + added)
            if task.optional:
                model.add(before[(k, program)] == 0).only_enforce_if(~chunk.present[k])
    # An element lookup carries each program through the selected predecessor.
    # Absent tasks use a separate zero state; they must not inherit the nonzero
    # origin history merely because they have no incoming circuit arc.
    predecessors = {}
    absent = len(tasks)
    for j, task in enumerate(tasks[1:], start=1):
        predecessor = predecessors[j] = model.new_int_var(
            0, absent, f"delivered_predecessor{j}"
        )
        incoming = [
            i * arc
            for (i, successor), arc in chunk.arc_literals.items()
            if successor == j
        ]
        if task.optional:
            incoming.append(absent * (1 - chunk.present[j]))
        model.add(predecessor == sum(incoming))
        for program in programs:
            model.add_element(
                predecessor,
                [after[(i, program)] for i in range(len(tasks))] + [0],
                before[(j, program)],
            )
    fractions: dict[int, cp_model.IntVar] = {}
    products: dict[str, cp_model.IntVar] = {}
    factors: dict[str, tuple[cp_model.IntVar, cp_model.IntVar]] = {}
    for k, task in enumerate(tasks):
        node = nodes[k]
        if not task.optional or node.allocated_share is None or node.weight == 0:
            continue
        denominator = model.new_int_var(
            1, max(1, total_bound), f"delivered_denominator{k}"
        )
        model.add_max_equality(denominator, [sum(before[(k, p)] for p in programs), 1])
        fraction = fractions[k] = model.new_int_var(0, scale, f"delivered_fraction{k}")
        assert node.program is not None
        model.add_division_equality(
            fraction, scale * before[(k, node.program)], denominator
        )
        # This is the exact initial floating share used to produce task.rate,
        # rather than its rounded native history approximation.
        prior_total = sum(context.history.values())
        origin = (
            context.history.get(node.program, 0.0) / prior_total if prior_total else 0.0
        )

        def product(
            a: cp_model.IntVar, b: cp_model.IntVar, bound: int, name: str
        ) -> cp_model.IntVar:
            if bound >= 2**60:
                raise ValueError(
                    "Delivered-share objective product exceeds the native integer range"
                )
            result = model.new_int_var(0, bound, name)
            model.add_multiplication_equality(result, [a, b])
            products[name] = result
            factors[name] = (a, b)
            return result

        def correction(
            seconds: cp_model.IntVar, bound: int, suffix: str, *, discount: bool
        ) -> list[cp_model.ObjLinearExprT]:
            valued = product(
                fraction, seconds, scale * bound, f"delivered_value{k}_{suffix}"
            )
            terms: list[cp_model.ObjLinearExprT] = [
                node.weight * origin * seconds,
                -node.weight / scale * valued,
            ]
            if not discount:
                return terms
            result: list[cp_model.ObjLinearExprT] = [
                (1 + node.offset) * term for term in terms
            ]
            if node.earliness:
                delayed = product(
                    seconds,
                    chunk.arrival[k],
                    bound * task.arrival[1],
                    f"delivered_seconds_arrival{k}_{suffix}",
                )
                priced = product(
                    valued,
                    chunk.arrival[k],
                    scale * bound * task.arrival[1],
                    f"delivered_value_arrival{k}_{suffix}",
                )
                result += [
                    -node.weight * origin * node.earliness * delayed,
                    node.weight / scale * node.earliness * priced,
                ]
            if node.late:
                if k not in chunk.lateness:
                    raise ValueError(
                        "Cadence lateness must be modelled before program-share correction"
                    )
                delayed = product(
                    seconds,
                    chunk.lateness[k],
                    bound * task.cadence,
                    f"delivered_seconds_late{k}_{suffix}",
                )
                priced = product(
                    valued,
                    chunk.lateness[k],
                    scale * bound * task.cadence,
                    f"delivered_value_late{k}_{suffix}",
                )
                result += [
                    -node.weight * origin * node.late * delayed,
                    node.weight / scale * node.late * priced,
                ]
            return result

        terms.extend(
            correction(chunk.counted[k], task.collection[1], "science", discount=True)
        )
        for n, _span in enumerate(task.spans):
            # Match the existing native allocation component, which is added
            # independently of lateness; exact decoding still has final say.
            terms.extend(
                node.allocation_bonus * term
                for term in correction(
                    chunk.span_seconds[(k, n)],
                    task.collection[1],
                    f"allocation{n}",
                    discount=False,
                )
            )
    model.maximize(sum(terms))
    assert model.validate() == ""
    return ProgramCounters[cp_model.IntVar](
        context=context,
        before=before,
        after=after,
        fractions=fractions,
        products=products,
        factors=factors,
        fraction_scale=scale,
        time_scale=context.time_scale,
        predecessors=predecessors,
    )


def hint_delivered_shares(chunk: _ChunkModel, hint: Mapping[int, _TaskHint]) -> None:
    """Extend the original hint with the counters of the same chosen route."""
    state = chunk.delivered_variables
    if state is None or any(
        not task.optional and k not in hint for k, task in enumerate(chunk.tasks) if k
    ):
        return
    model = chunk.model
    values = dict(
        zip(
            model.proto.solution_hint.vars,
            model.proto.solution_hint.values,
            strict=True,
        )
    )

    def add(variable: cp_model.IntVar, value: int) -> None:
        model.add_hint(variable, value)
        values[variable.index] = value

    programs = sorted(
        state.context.history.keys()
        | {n.program for n in state.context.nodes if n.program is not None}
    )
    history = {
        p: math.floor(state.context.history.get(p, 0.0) * state.time_scale)
        for p in programs
    }
    included = set(hint)
    previous = 0
    for k, task_hint in hint.items():
        node = state.context.nodes[k]
        add(state.predecessors[k], previous)
        for program in programs:
            add(state.before[(k, program)], history[program])
        if k in state.fractions:
            assert node.program is not None
            add(
                state.fractions[k],
                state.fraction_scale
                * history[node.program]
                // max(1, sum(history.values())),
            )
        if node.program is not None:
            history[node.program] += (
                task_hint.collection * state.time_scale
                if chunk.tasks[k].optional
                else math.floor(node.fixed_seconds * state.time_scale)
            )
        for program in programs:
            add(state.after[(k, program)], history[program])
        previous = k
    for k in range(1, len(chunk.tasks)):
        if k in included:
            continue
        add(state.predecessors[k], len(chunk.tasks))
        for program in programs:
            add(state.before[(k, program)], 0)
            add(state.after[(k, program)], 0)
        if k in state.fractions:
            add(state.fractions[k], 0)
    for name, variable in state.products.items():
        a, b = state.factors[name]
        if a.index in values and b.index in values:
            add(variable, values[a.index] * values[b.index])
