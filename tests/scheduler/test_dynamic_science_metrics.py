"""Executed dynamic values stay distinct from original input priorities."""

from datetime import timedelta

import pytest

from conops.benchmark import BenchmarkResult, dispatch, format_results, run_benchmark
from conops.benchmark.metrics import dynamic_science_value
from conops.common import ObsType
from conops.ditl.telemetry import Housekeeping
from conops.targets import PlanEntry

from .planning_scenario import BEGIN, T0
from .test_benchmark import _scenario


def entry(value, *, start=0, duration=60, obsid=1, calibration=False):
    return PlanEntry(
        obsid=obsid,
        ra=105,
        dec=10,
        begin=T0 + start,
        end=T0 + start + duration,
        slewtime=0,
        obstype=ObsType.AT,
        collection_begin=T0 + start,
        collection_end=T0 + start + duration,
        merit=value,
        calibration=calibration,
    )


def sample(seconds, *, start=0, obsid=1, calibration=False):
    return Housekeeping(
        timestamp=BEGIN + timedelta(seconds=start),
        obsid=obsid,
        collection_seconds=seconds,
        calibration=calibration,
    )


def test_actual_seconds_receive_the_selected_dynamic_value():
    assert dynamic_science_value([sample(17.5)], [entry(1300)], 60) == 22750


def test_revisits_receive_their_own_values():
    entries = [entry(100), entry(500, start=120)]
    records = [sample(20), sample(30, start=120)]
    assert dynamic_science_value(records, entries, 60) == 17000


def test_calibration_and_uncollected_planned_time_earn_no_value():
    entries = [entry(2000, calibration=True), entry(500, start=60)]
    records = [sample(60, calibration=True), sample(20, start=60)]
    assert dynamic_science_value(records, entries, 60) == 10000


def test_dynamic_value_can_reverse_original_priority_order():
    # A base-100 request gained urgency; a base-500 request did not.
    urgent = dynamic_science_value([sample(10)], [entry(1100)], 60)
    ordinary = dynamic_science_value([sample(10)], [entry(500)], 60)
    assert urgent > ordinary


def test_collection_starting_mid_step_receives_only_recorded_seconds():
    assert dynamic_science_value([sample(15)], [entry(300, start=30)], 60) == 4500


def test_negative_dynamic_values_are_preserved():
    assert dynamic_science_value([sample(10)], [entry(-3)], 60) == -30


def test_missing_dynamic_provenance_is_rejected():
    with pytest.raises(ValueError, match="unambiguous executed dynamic merit"):
        dynamic_science_value([sample(10, start=120)], [entry(300)], 60)


def test_mixed_values_within_one_telemetry_sample_are_not_guessed():
    entries = [entry(100, duration=30), entry(500, start=30, duration=30)]
    with pytest.raises(ValueError, match="unambiguous executed dynamic merit"):
        dynamic_science_value([sample(40)], entries, 60)


@pytest.mark.parametrize("step", [0, -1, float("inf"), float("nan")])
def test_invalid_sample_duration_is_rejected(step):
    with pytest.raises(ValueError, match="step_size"):
        dynamic_science_value([], [], step)


def test_legacy_result_has_no_fabricated_dynamic_value():
    result = BenchmarkResult(contender="old", scenario="s", weighted_science=1000)
    assert result.dynamic_science is None
    table = format_results([result])
    assert "dynamic value" in table and "fixed value" in table
    assert table.splitlines()[2].split()[2] == "-"


def test_benchmark_includes_the_dispatchers_dynamic_value():
    scenario = _scenario()
    original = scenario.make_config

    def dynamic_config():
        config = original()
        config.targets.urgency_weight = 500
        config.targets.urgency_timescale_seconds = 3600
        return config

    scenario.make_config = dynamic_config
    [result] = run_benchmark(scenario, [dispatch()])
    assert result.error is None
    assert result.dynamic_science is not None
    assert result.dynamic_science > result.weighted_science


def test_missing_value_provenance_is_reported_as_a_failed_benchmark(monkeypatch):
    from conops.benchmark import harness

    def missing(*args):
        raise ValueError("missing executed dynamic merit")

    monkeypatch.setattr(harness, "dynamic_science_value", missing)
    [result] = run_benchmark(_scenario(), [dispatch()])
    assert result.error == "ValueError: missing executed dynamic merit"
    assert result.dynamic_science is None
