"""Safe-roll candidates retain their order and cannot be corrupted by callers."""

import pytest

from conops.simulation.acs import ACS


@pytest.mark.parametrize("optimal", [12.5, 372.5, -347.5])
def test_safe_roll_candidate_order(optimal):
    assert ACS._idle_safe_roll_candidates(optimal) == [
        12.5,
        27.5,
        357.5,
        42.5,
        342.5,
        57.5,
        327.5,
        72.5,
        312.5,
        87.5,
        297.5,
        102.5,
        282.5,
        117.5,
        267.5,
        132.5,
        252.5,
        147.5,
        237.5,
        162.5,
        222.5,
        177.5,
        207.5,
        192.5,
    ]


def test_repeated_candidates_reuse_the_calculation_and_return_fresh_lists():
    cached = ACS._cached_idle_safe_roll_candidates
    cached.cache_clear()
    first = ACS._idle_safe_roll_candidates(12.5)
    expected = list(first)
    first[0] = -1
    first.append(-2)
    second = ACS._idle_safe_roll_candidates(12.5)
    assert second == expected
    assert cached.cache_info().misses == 1
    assert cached.cache_info().hits == 1


def test_changed_grid_does_not_use_stale_candidates(monkeypatch):
    import conops.simulation.acs as module

    assert len(ACS._idle_safe_roll_candidates(12.5)) == 24
    monkeypatch.setattr(module, "IDLE_SAFE_ATTITUDE_GRID_STEP_DEG", 30)
    result = ACS._idle_safe_roll_candidates(12.5)
    assert len(result) == 12
    assert result[:3] == [12.5, 42.5, 342.5]
