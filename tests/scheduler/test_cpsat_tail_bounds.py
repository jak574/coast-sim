"""CpSatScheduler and the plan after a chunk: a cadence visit in a chunk is
kept a cadence clear of the request's next visit after it, and a remainder
whose full snapshot was not placed does not discard the chunk's result."""

from datetime import timedelta

import numpy as np
import pytest

pytest.importorskip("ortools")

from ortools.sat.python import cp_model  # noqa: E402

from conops.config import MissionConfig, TargetConfig  # noqa: E402
from conops.config.observation_categories import ObservationCategory  # noqa: E402
from conops.schedulers import CpSatScheduler  # noqa: E402
from conops.schedulers.cpsat_scheduler import _ChunkModel, _Task  # noqa: E402
from conops.schedulers.local_search_scheduler import (  # noqa: E402
    _Decoded,
    _Snapshot,
    _State,
)
from conops.schedulers.priority_scheduler import _collection_seconds  # noqa: E402
from conops.targets import Pointing  # noqa: E402

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0  # noqa: E402
from .planning_scenario import make_config as _config  # noqa: E402
from .planning_scenario import make_target as _target  # noqa: E402

ORIGIN = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
MONITOR = ObservationCategory(
    name="Monitor", obsid_min=1, obsid_max=2, cadence_seconds=HOUR
)
HOURS = 6


def _monitor_config() -> MissionConfig:
    return _config(HOURS, weights=TargetConfig(cadence_weight=50), categories=[MONITOR])


def _monitored(config: MissionConfig) -> list[Pointing]:
    """A cadence target wanting four ten-minute visits, and filler."""
    monitor = _target(config, 1, *PATCH[0], merit=60, minutes=40, snapshot=10)
    filler = [
        _target(config, 10 + i, ra, dec, merit=40, minutes=60, snapshot=20)
        for i, (ra, dec) in enumerate(PATCH[1:])
    ]
    return [monitor, *filler]


class TestCadenceBoundInTheModel:
    @staticmethod
    def _solve(next_visit: int | None) -> tuple[bool, int]:
        """Solve one cadence visit of 300 s in a window of 3000 s, with the
        request's next visit after the chunk at ``next_visit``: whether the
        visit is taken, and when it arrives."""
        window = (0, 3000)
        visit = _Task(
            (1, 0.0),
            ORIGIN,
            (window[0], window[1] - 300),
            spacing=HOUR,
            collection=(300, 300),
            window=window,
            rate=1.0,
            # A later arrival is worth more, so only a bound keeps it early.
            earliness=-1e-4,
            next_visit=next_visit,
        )
        tasks = [_Task(None, ORIGIN, (0, 0)), visit]
        chunk = _ChunkModel(tasks, np.zeros((2, 2)), 60, 10_000, {1: 300})
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1
        solver.parameters.max_time_in_seconds = 10.0
        assert solver.solve(chunk.model) == cp_model.OPTIMAL
        return bool(solver.value(chunk.present[1])), int(solver.value(chunk.arrival[1]))

    def test_without_a_next_visit_the_visit_may_arrive_late(self) -> None:
        present, arrival = self._solve(None)
        assert present
        assert arrival == 2700

    def test_the_visit_ends_a_cadence_before_the_next_visit(self) -> None:
        """Arrival, 300 s of collection and an hour's spacing end by the
        next visit at 4500 s, so the visit arrives by 600 s."""
        present, arrival = self._solve(4500)
        assert present
        assert arrival + 300 + HOUR <= 4500
        assert arrival == 600

    def test_a_visit_that_cannot_end_a_cadence_before_is_left_out(self) -> None:
        present, _ = self._solve(3000)
        assert not present


class TestCadenceBoundInTheScheduler:
    def test_the_next_visits_are_the_first_of_each_cadence_request(self) -> None:
        config = _monitor_config()
        scheduler = CpSatScheduler(
            config,
            _monitored(config),
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            solver_time_limit=0.5,
            workers=1,
            seed=1,
        )
        scheduler._start_from_priority_plan()
        after = _Decoded.model_construct(
            sequence=[
                _Snapshot(obsid=10, seconds=20 * MIN, release=T0 + 4 * HOUR),
                _Snapshot(obsid=1, seconds=10 * MIN),
                _Snapshot(obsid=1, seconds=10 * MIN, release=T0 + 4.5 * HOUR),
                _Snapshot(obsid=1, seconds=10 * MIN, release=T0 + 5.5 * HOUR),
            ],
            states=[],
        )
        # The filler has no cadence; a snapshot with no release has no time.
        assert scheduler._next_visits(after) == {1: T0 + 4.5 * HOUR}
        assert scheduler._next_visits(None) == {}

    def test_a_chunks_candidates_are_bounded_by_the_visit_after_it(self) -> None:
        """The first chunk's candidates of the cadence request have, as
        their bound, the release of the request's first visit after the
        chunk, in seconds from the chunk's origin. No other candidate has
        a bound."""
        chunks: list[tuple[list[_Task], _State, _Decoded | None]] = []

        class Capture(CpSatScheduler):
            def _solve_chunk(self, state, *args, **kwargs):  # type: ignore[no-untyped-def]
                continuation = args[5] if len(args) > 5 else kwargs.get("continuation")
                chunks.append(([], state, continuation))
                return super()._solve_chunk(state, *args, **kwargs)

            def _candidates(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                tasks = super()._candidates(*args, **kwargs)
                chunks[-1][0].extend(tasks)
                return tasks

        config = _monitor_config()
        scheduler = Capture(
            config,
            _monitored(config),
            BEGIN,
            BEGIN + timedelta(hours=HOURS),
            solver_time_limit=1.0,
            workers=1,
            seed=1,
            include_passes=False,
        )
        scheduler.schedule()

        tasks, state, continuation = chunks[0]
        assert continuation is not None
        later = [
            s.release
            for s in continuation.sequence
            if s.obsid == 1 and s.release is not None
        ]
        assert later, "the plan after the first chunk has a visit of the request"
        origin, _ = scheduler._chunk_origin(state)
        visits = [task for task in tasks if task.obsid == 1]
        assert visits
        assert {task.next_visit for task in visits} == {
            int(np.floor(later[0] - origin))
        }
        assert all(task.next_visit is None for task in tasks if task.obsid != 1)


class TestRemainderRetry:
    X, Y = 100, 200

    def _scheduler(self) -> CpSatScheduler:
        """A request of 15 minutes taken in 10-minute snapshots, whose last
        5 minutes are a remainder below ``ss_min``, and another request."""
        config = _config(4)
        targets = [
            _target(config, self.X, *PATCH[0], minutes=15, snapshot=10),
            _target(config, self.Y, *PATCH[1], minutes=20, snapshot=20),
        ]
        scheduler = CpSatScheduler(
            config,
            targets,
            BEGIN,
            BEGIN + timedelta(hours=4),
            solver_time_limit=0.5,
            workers=1,
            seed=1,
            include_passes=False,
        )
        scheduler._start_from_priority_plan()
        return scheduler

    def _window(
        self, scheduler: CpSatScheduler, solved: list[_Snapshot]
    ) -> tuple[_Decoded, bool]:
        """Solve the whole horizon as one chunk of an empty plan, with the
        solver's answer replaced by ``solved``."""
        scheduler._solve_chunk = lambda *args, **kwargs: list(solved)  # type: ignore[method-assign]
        empty = scheduler._decode([])
        return scheduler._solve_window(empty, T0, T0 + 4 * HOUR, 1.0)

    @staticmethod
    def _science(decoded: _Decoded) -> dict[int, float]:
        science: dict[int, float] = {}
        for block in decoded.final.timeline:
            if not block.is_fixed:
                obsid = int(block.entry.obsid)
                science[obsid] = science.get(obsid, 0.0) + _collection_seconds(block)
        return science

    def test_a_remainder_alone_is_not_a_plan(self) -> None:
        """The case the retry is for: decoded as it is, the solver's answer
        has a remainder with no full snapshot after it and scores minus
        infinity."""
        scheduler = self._scheduler()
        decoded = scheduler._decode(
            [
                _Snapshot(obsid=self.X, seconds=5 * MIN, release=T0, remainder=True),
                _Snapshot(obsid=self.Y, seconds=20 * MIN, release=T0),
            ]
        )
        assert decoded.final.owed == {self.X}

    def test_the_chunk_is_kept_without_the_remainder(self) -> None:
        scheduler = self._scheduler()
        plan, used = self._window(
            scheduler,
            [
                _Snapshot(obsid=self.X, seconds=5 * MIN, release=T0),
                _Snapshot(obsid=self.Y, seconds=20 * MIN, release=T0),
            ],
        )
        assert used
        assert plan.final.owed == frozenset()
        assert self._science(plan) == {self.Y: 20 * MIN}
        assert scheduler._objective(plan.final) > 0

    def test_a_remainder_with_its_full_snapshot_is_kept_whole(self) -> None:
        scheduler = self._scheduler()
        plan, used = self._window(
            scheduler,
            [
                _Snapshot(obsid=self.X, seconds=5 * MIN, release=T0),
                _Snapshot(obsid=self.X, seconds=10 * MIN, release=T0),
                _Snapshot(obsid=self.Y, seconds=20 * MIN, release=T0),
            ],
        )
        assert used
        assert self._science(plan) == {self.X: 15 * MIN, self.Y: 20 * MIN}
