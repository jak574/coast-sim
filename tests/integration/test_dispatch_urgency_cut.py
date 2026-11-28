"""A protected opening cuts collection without changing another target's deadline."""

import pytest

from conops.config import TargetConfig

from .test_merit_dispatch import BEGIN, HOUR, _run


@pytest.mark.parametrize("protect_opening", [False, True])
def test_protected_request_does_not_inflate_another_targets_urgency(protect_opening):
    first_visits = []
    for opens_after in (HOUR, 2 * HOUR):
        filler = {
            "obsid": 10,
            "ra": 20.0,
            "dec": 0.0,
            "merit": 100.0,
            "exptime": 3 * HOUR,
            "ss_min": 300,
            "ss_max": 3 * HOUR,
            "deadline": BEGIN + 4 * HOUR,
        }
        timed = {
            "obsid": 11,
            "ra": 40.0,
            "dec": 0.0,
            "merit": 200.0,
            "exptime": 600,
            "ss_min": 600,
            "ss_max": 600,
            "earliest_start": BEGIN + opens_after,
            "deadline": BEGIN + opens_after + 600,
        }

        def prepare(run):
            run.queue.targets[1].protect_opening = protect_opening

        run = _run(
            [filler, timed],
            weights=TargetConfig(urgency_weight=60.0, urgency_timescale_seconds=HOUR),
            prepare=prepare,
            step_size=60,
        )
        first = next(entry for entry in run.plan if entry.obsid == 10)
        request = next(entry for entry in run.plan if entry.obsid == 11)
        assert first.collection_end <= request.begin
        assert request.collection_begin <= timed["deadline"]
        first_visits.append(first)

    earlier, later = first_visits
    assert earlier.begin == later.begin
    assert earlier.collection_end < later.collection_end
    assert earlier.merit == later.merit
    assert earlier.merit == pytest.approx(
        100.0 + 60.0 * HOUR / (BEGIN + 4 * HOUR - float(earlier.begin))
    )
