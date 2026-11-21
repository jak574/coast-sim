"""Retained plans identify science windows independently of ingress times."""

import pytest

from conops.schedulers.local_search_scheduler import _Snapshot
from conops.targets import Plan

from .planning_scenario import HOUR
from .test_selected_visit_decode import scheduler_for_windows


@pytest.mark.parametrize("reload", [False, True])
def test_seed_decode_preserves_the_selected_visit_window(reload, tmp_path):
    scheduler, target = scheduler_for_windows()
    original = scheduler._decode([_Snapshot(obsid=200, seconds=HOUR, visit=1)])
    entry = original.final.timeline[0].entry
    plan = Plan(entries=[entry])
    if reload:
        path = tmp_path / "seed.json"
        plan.save(path)
        plan = Plan.load(path)
    seeded = scheduler._decode_plan(plan)
    assert len(seeded.final.timeline) == 1
    restored = seeded.final.timeline[0].entry
    assert restored.visit == entry.visit == 1
    assert restored.collection_begin == entry.collection_begin
    assert not target.visits_done


def test_seed_visit_cannot_bypass_a_committed_window():
    scheduler, target = scheduler_for_windows(reserve_first=True)
    decoded = scheduler._decode([_Snapshot(obsid=200, seconds=HOUR, visit=1)])
    entry = decoded.final.timeline[0].entry.model_copy(update={"visit": 0})
    seeded = scheduler._decode_plan(Plan(entries=[entry]))
    assert seeded.final.timeline == []
    assert not target.visits_done
