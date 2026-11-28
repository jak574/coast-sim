"""CpSatScheduler(local_search_first=True): the local search runs before
the chunks, which are then solved from the searched plan."""

from datetime import timedelta

import pytest

pytest.importorskip("ortools")

from pydantic import ValidationError  # noqa: E402

from conops import DITL  # noqa: E402
from conops.config.scheduler import SchedulerConfig, SchedulerKind  # noqa: E402
from conops.schedulers import CpSatScheduler  # noqa: E402
from conops.schedulers.local_search_scheduler import _Decoded  # noqa: E402
from conops.targets import Plan, Pointing  # noqa: E402

from .planning_scenario import BEGIN, CONSTRAINED, PATCH  # noqa: E402
from .planning_scenario import make_config as _config  # noqa: E402
from .planning_scenario import make_target as _target  # noqa: E402

HOURS = 6
ITERATIONS = 40


def _targets() -> list[Pointing]:
    config = _config(HOURS)
    return [
        _target(config, 100 + i, ra, dec, merit=90 - 7 * i, minutes=60, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
    ]


class _Recording(CpSatScheduler):
    """Records the order of the two stages and the score each starts from."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.stages: list[tuple[str, float]] = []

    def _search(self, start: _Decoded) -> _Decoded:
        self.stages.append(("search", self._objective(start.final)))
        return super()._search(start)

    def _solve(self, start: _Decoded) -> _Decoded:
        self.stages.append(("chunks", self._objective(start.final)))
        return super()._solve(start)


def _scheduler(targets: list[Pointing], *, first: bool) -> _Recording:
    config = targets[0].config
    assert config is not None
    return _Recording(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=HOURS),
        solver_deterministic_time=1.0,
        solver_time_limit=60.0,
        workers=1,
        seed=1,
        time_limit=3600.0,
        max_iterations=ITERATIONS,
        local_search_first=first,
        include_passes=False,
    )


def test_by_default_the_chunks_are_solved_before_the_search() -> None:
    scheduler = _scheduler(_targets(), first=False)
    scheduler.schedule()
    assert [stage for stage, _ in scheduler.stages] == ["chunks", "search"]
    assert scheduler.stages[0][1] == pytest.approx(scheduler.start_score)


def test_the_search_runs_first_and_the_chunks_start_from_its_result() -> None:
    scheduler = _scheduler(_targets(), first=True)
    scheduler.schedule()
    assert [stage for stage, _ in scheduler.stages] == ["search", "chunks"]
    searched_from, solved_from = (score for _, score in scheduler.stages)
    assert searched_from == pytest.approx(scheduler.start_score)
    # The chunks start from what the search returned, which is never below
    # what it started from, and the search does not run a second time.
    assert solved_from >= searched_from
    assert scheduler.search_iterations == [ITERATIONS]
    assert scheduler.score >= scheduler.initial_score


def test_the_plan_executes_as_planned() -> None:
    targets = _targets()
    scheduler = _scheduler(targets, first=True)
    plan = scheduler.schedule()
    config = targets[0].config
    assert config is not None
    ditl = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=Plan.model_validate_json(plan.model_dump_json()),
        begin=BEGIN,
        end=BEGIN + timedelta(hours=HOURS),
    )
    ditl.step_size = 60
    assert ditl.calc()
    assert ditl.validate_plan_matches_execution() == []


def test_the_setting_is_a_cp_sat_setting() -> None:
    settings = SchedulerConfig(kind=SchedulerKind.CP_SAT, local_search_first=True)
    assert settings.scheduler_options()["local_search_first"] is True
    assert (
        "local_search_first"
        not in SchedulerConfig(kind=SchedulerKind.CP_SAT).scheduler_options()
    )
    with pytest.raises(ValidationError, match="local_search_first"):
        SchedulerConfig(kind=SchedulerKind.LOCAL_SEARCH, local_search_first=True)
