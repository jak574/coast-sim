"""Overlapping held-attitude scans evaluate only missing cache entries."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pytest
from rust_ephem import SunConstraint, TLEEphemeris

from conops.common import angular_separation
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


@pytest.mark.parametrize(
    "first, second",
    [
        ((10.001, 20, 12.5), (10.009, 20, 12.5)),
        ((10, 20.001, 12.5), (10, 20.009, 12.5)),
        ((10, 20, 12.501), (10, 20, 12.509)),
    ],
)
def test_distinct_poses_in_one_rounded_bucket_refresh_the_whole_hold(
    held_check, first, second
):
    constraint, check = held_check
    check.in_constraint_batch.side_effect = lambda **kw: np.full(
        (1, len(kw["times"])),
        (kw["target_ras"][0], kw["target_decs"][0], kw["target_rolls"][0]) == second,
    )
    with constraint.held_over([0.0, 60.0]):
        assert not constraint._cached_check("sensor", *first[:2], 0, check, first[2])
    with constraint.held_over([60.0, 120.0]):
        assert constraint._cached_check("sensor", *second[:2], 120, check, second[2])
        assert constraint._cached_check("sensor", *second[:2], 60, check, second[2])
    assert [
        t.timestamp() for t in check.in_constraint_batch.call_args.kwargs["times"]
    ] == [
        60,
        120,
    ]


def test_reused_exact_samples_restore_values_overwritten_by_a_nearby_pose(held_check):
    constraint, check = held_check
    check.in_constraint_batch.side_effect = lambda **kw: np.full(
        (1, len(kw["times"])), kw["target_ras"][0] > 10.005
    )
    for ra, times in [
        (10.001, [0.0, 60.0]),
        (10.009, [60.0, 120.0]),
        (10.001, [0.0, 60.0, 180.0]),
    ]:
        with constraint.held_over(times):
            constraint._cached_check("sensor", ra, 20, times[-1], check)
    assert not constraint._cached_check("sensor", 10.001, 20, 60, check)
    assert [
        t.timestamp() for t in check.in_constraint_batch.call_args.kwargs["times"]
    ] == [180]


def test_distinct_times_in_one_rounded_bucket_keep_full_batch_order(held_check):
    constraint, check = held_check
    check.in_constraint_batch.side_effect = lambda **kw: np.array(
        [[t.timestamp() > 0.5 for t in kw["times"]]]
    )
    with constraint.held_over([0.1, 0.9, 60.0]):
        assert constraint._cached_check("sensor", 10, 20, 0.1, check)
    with constraint.held_over([0.9, 0.1, 120.0]):
        constraint._cached_check("sensor", 10, 20, 120, check)
        assert not constraint._cached_check("sensor", 10, 20, 0.9, check)
    assert [
        t.timestamp()
        for t in check.in_constraint_batch.call_args_list[0].kwargs["times"]
    ] == [0.1, 0.9, 60.0]


def test_overlapping_hold_refreshes_a_real_sun_cone_boundary():
    begin = datetime(2025, 11, 1, tzinfo=timezone.utc)
    ephem = TLEEphemeris(
        tle="examples/example.tle",
        begin=begin,
        end=begin + timedelta(minutes=3),
        step_size=60,
    )
    ras = (10.001, 10.009)
    angles = [
        angular_separation(ra, 20, ephem.sun_ra_deg[1], ephem.sun_dec_deg[1])
        for ra in ras
    ]
    sun = SunConstraint(min_angle=sum(angles) / 2)
    constraint = Constraint(sun_constraint=sun, ephem=ephem)
    times = [begin.timestamp() + i * 60 for i in range(3)]
    expected = [
        sun.in_constraint(
            ephemeris=ephem, target_ra=ra, target_dec=20, time=ephem.timestamp[1]
        )
        for ra in ras
    ]
    assert expected[0] != expected[1]
    with constraint.held_over(times[:2]):
        constraint.in_sun(ras[0], 20, times[0])
    with constraint.held_over(times[1:]):
        constraint.in_sun(ras[1], 20, times[2])
        assert constraint.in_sun(ras[1], 20, times[1]) == expected[1]
