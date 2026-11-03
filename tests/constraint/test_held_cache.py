"""Overlapping held-attitude scans evaluate only missing cache entries."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops.config import Constraint


@pytest.fixture
def held_check():
    constraint = Constraint()
    constraint.ephem = Mock()
    check = Mock()
    check.in_constraint_batch.side_effect = lambda **kw: np.array(
        [[int(t.timestamp()) % 120 == 0 for t in kw["times"]]], dtype=bool
    )
    check.in_constraint.return_value = False
    return constraint, check


@pytest.mark.parametrize("roll", [None, 12.5])
def test_overlapping_hold_evaluates_only_new_times(held_check, roll):
    constraint, check = held_check
    with constraint.held_over([0.0, 60.0, 120.0]):
        assert constraint._cached_check("sensor", 10, 20, 0, check, roll)
    with constraint.held_over([60.0, 120.0, 180.0]):
        assert not constraint._cached_check("sensor", 10, 20, 180, check, roll)
        assert constraint._cached_check("sensor", 10, 20, 120, check, roll)

    calls = check.in_constraint_batch.call_args_list
    assert [[t.timestamp() for t in c.kwargs["times"]] for c in calls] == [
        [0.0, 60.0, 120.0],
        [180.0],
    ]
    assert calls[-1].kwargs["target_rolls"] == (None if roll is None else [roll])
    check.in_constraint.assert_not_called()


def test_repeated_times_are_evaluated_once(held_check):
    constraint, check = held_check
    with constraint.held_over([0.0, 60.0, 60.0, 120.0]):
        assert constraint._cached_check("sensor", 10, 20, 0, check)
    assert [
        t.timestamp() for t in check.in_constraint_batch.call_args.kwargs["times"]
    ] == [0, 60, 120]


def test_outside_a_cached_hold_uses_only_the_scalar_check(held_check):
    constraint, check = held_check
    with constraint.held_over([0.0, 60.0]):
        constraint._cached_check("sensor", 10, 20, 0, check)
    check.reset_mock()
    with constraint.held_over([0.0, 60.0]):
        assert not constraint._cached_check("sensor", 10, 20, 180, check)
    check.in_constraint_batch.assert_not_called()
    check.in_constraint.assert_called_once()


def test_clearing_cache_rechecks_the_entire_hold(held_check):
    constraint, check = held_check
    with constraint.held_over([0.0, 60.0]):
        constraint._cached_check("sensor", 10, 20, 0, check)
        constraint.clear_cache()
        constraint._cached_check("sensor", 10, 20, 0, check)
    assert [
        len(c.kwargs["times"]) for c in check.in_constraint_batch.call_args_list
    ] == [2, 2]
