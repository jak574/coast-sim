"""Exact science answers survive harmless rebinding, preserving lookup order."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from rust_ephem import SunConstraint

import conops.config.constraint as constraint_module
from conops.config import Constraint, SpacecraftBus, SpacecraftFrame


@pytest.fixture
def sample_check():
    constraint = Constraint(
        safety_constraint=SunConstraint(min_angle=45),
        science_constraint=SunConstraint(min_angle=30),
    )
    constraint.ephem = Mock()
    constraint._body_frame_source = SpacecraftBus()
    check = Mock()
    check.model_dump_json.side_effect = lambda: (
        constraint.science_constraint.model_dump_json()
    )
    check.in_constraint.side_effect = lambda **kw: kw["target_ra"] >= 10.005
    assert constraint.body_safety_constraint is not None
    return constraint, check


def rebind(constraint):
    constraint.invalidate_combined_constraint_cache()
    assert constraint.body_safety_constraint is not None


def science(constraint, check, ra=10.001, dec=20.001, time=120.001, roll=30.001):
    return constraint._cached_check("explicit_science", ra, dec, time, check, roll)


def test_identical_geometry_reuses_native_answers_after_rebinding(sample_check):
    constraint, check = sample_check
    for _ in range(3):
        assert not science(constraint, check)
        rebind(constraint)
    assert check.in_constraint.call_count == 1


def test_rounded_lookup_still_takes_precedence_over_exact_samples(sample_check):
    constraint, check = sample_check
    assert not science(constraint, check, ra=10.001)
    rebind(constraint)
    assert science(constraint, check, ra=10.009)
    rebind(constraint)
    assert not science(constraint, check, ra=10.001)
    # Keep the established rounded-lookup result, even though this exact pose
    # previously had a different native answer in another lookup epoch.
    assert not science(constraint, check, ra=10.009)
    assert check.in_constraint.call_count == 2


def test_explicit_clear_forces_a_fresh_native_answer(sample_check):
    constraint, check = sample_check
    check.in_constraint.side_effect = [False, True]
    assert not science(constraint, check)
    constraint.clear_cache()
    assert science(constraint, check)
    assert check.in_constraint.call_count == 2


def test_changed_geometry_does_not_reuse_an_old_sample(sample_check):
    constraint, check = sample_check
    check.in_constraint.side_effect = lambda **kw: (
        constraint.science_constraint.min_angle > 45
    )
    assert not science(constraint, check)
    rebind(constraint)
    assert not science(constraint, check)
    assert check.in_constraint.call_count == 1
    constraint.science_constraint.min_angle = 60
    rebind(constraint)
    assert science(constraint, check)
    assert check.in_constraint.call_count == 2


def test_in_place_backend_change_is_checked_before_the_next_body_refresh(sample_check):
    constraint, check = sample_check
    check.in_constraint.side_effect = lambda **kw: (
        constraint.science_constraint.min_angle > 45
    )
    assert not science(constraint, check)
    rebind(constraint)
    constraint.science_constraint.min_angle = 60
    assert science(constraint, check)
    assert check.in_constraint.call_count == 2


def test_changed_frame_does_not_reuse_an_old_sample(sample_check):
    constraint, check = sample_check
    check.in_constraint.side_effect = lambda **kw: (
        constraint._body_frame_source.frame.boresight[0] == 1
    )
    assert science(constraint, check)
    rebind(constraint)
    assert science(constraint, check)
    assert check.in_constraint.call_count == 1
    constraint._body_frame_source.frame = SpacecraftFrame(
        boresight=(0, 0, 1), sun_side=(1, 0, 0)
    )
    assert constraint.body_safety_constraint is not None
    assert not science(constraint, check)
    assert check.in_constraint.call_count == 2


@pytest.mark.parametrize("dimension", ["ra", "dec", "time", "roll"])
def test_native_sample_keys_keep_fractional_inputs_exact(sample_check, dimension):
    constraint, check = sample_check
    backend_name = {"ra": "target_ra", "dec": "target_dec", "roll": "target_roll"}.get(
        dimension
    )

    def result(**kw):
        value = kw[backend_name] if backend_name else kw["time"].timestamp()
        return (
            value
            > {"ra": 10.005, "dec": 20.005, "time": 120.005, "roll": 30.005}[dimension]
        )

    check.in_constraint.side_effect = result
    assert not science(constraint, check)
    rebind(constraint)
    changed = {
        dimension: {"ra": 10.009, "dec": 20.009, "time": 120.009, "roll": 30.009}[
            dimension
        ]
    }
    assert science(constraint, check, **changed)
    rebind(constraint)
    assert not science(constraint, check)
    assert check.in_constraint.call_count == 2


def test_new_backend_identity_never_uses_an_old_answer(sample_check):
    constraint, check = sample_check
    assert not science(constraint, check)
    rebind(constraint)
    other = Mock()
    other.model_dump_json.return_value = check.model_dump_json()
    other.in_constraint.return_value = True
    assert science(constraint, other)
    other.in_constraint.assert_called_once()


def test_unserializable_geometry_still_works_without_retaining_samples(sample_check):
    constraint, check = sample_check
    constraint.sun_constraint = Mock()
    rebind(constraint)
    assert not science(constraint, check)
    rebind(constraint)
    assert not science(constraint, check)
    assert check.in_constraint.call_count == 2


def test_retained_samples_have_a_bounded_lru_capacity(sample_check, monkeypatch):
    constraint, check = sample_check
    monkeypatch.setattr(constraint_module, "_SCIENCE_SAMPLE_CACHE_SIZE", 2)
    for ra in (10, 20, 30):
        science(constraint, check, ra=ra)
    rebind(constraint)
    science(constraint, check, ra=10)  # Evicted; this must call the backend again.
    science(constraint, check, ra=30)  # Still retained.
    assert check.in_constraint.call_count == 4
    assert len(constraint._science_samples.entries) == 2


def test_deepcopy_drops_samples_without_copying_native_owners(sample_check):
    constraint, _ = sample_check

    class Backend:
        def model_dump_json(self):
            return "{}"

        def in_constraint(self, **kwargs):
            return False

        def __deepcopy__(self, memo):
            raise AssertionError("Do not copy opaque cached backend owners")

    assert not science(constraint, Backend())
    assert constraint._science_samples.entries
    copied = deepcopy(constraint)
    assert not copied._science_samples.entries
    assert copied._science_samples is not constraint._science_samples


def test_subclass_clear_callback_keeps_its_zero_argument_contract():
    calls = []

    class CustomConstraint(Constraint):
        def clear_cache(self):
            calls.append(True)
            super().clear_cache()

    constraint = CustomConstraint(safety_constraint=SunConstraint(min_angle=45))
    constraint._body_frame_source = SpacecraftBus()
    assert constraint.body_safety_constraint is not None
    rebind(constraint)
    assert len(calls) == 2


def test_wrapped_clear_callback_is_not_bypassed(sample_check, monkeypatch):
    constraint, _ = sample_check
    original = Constraint.clear_cache
    calls = []

    def wrapped(self):
        calls.append(True)
        original(self)

    monkeypatch.setattr(Constraint, "clear_cache", wrapped)
    rebind(constraint)
    assert len(calls) == 1


def test_unchanged_science_can_reuse_a_sample_before_lazy_hardware_binding(
    sample_check,
):
    constraint, check = sample_check
    assert not science(constraint, check)
    rebind(constraint)
    constraint.invalidate_combined_constraint_cache()
    assert constraint._body_safety_key is None
    assert not science(constraint, check)
    assert check.in_constraint.call_count == 1
    # A science lookup must not move the hardware refresh and rounded-cache
    # clearing boundary earlier than the original evaluator would.
    assert constraint._body_safety_key is None


def test_frame_changes_are_detected_before_a_lazy_hardware_refresh(sample_check):
    constraint, check = sample_check
    check.in_constraint.side_effect = lambda **kw: (
        constraint._body_frame_source.frame.boresight[0] == 1
    )
    assert science(constraint, check)
    rebind(constraint)
    constraint._body_frame_source.frame = SpacecraftFrame(
        boresight=(0, 0, 1), sun_side=(1, 0, 0)
    )
    assert not science(constraint, check)
    assert check.in_constraint.call_count == 2


@pytest.mark.parametrize("dimension", ["ra", "dec", "roll"])
def test_native_sample_reuse_distinguishes_signed_zero(sample_check, dimension):
    from math import copysign

    constraint, check = sample_check
    backend_name = {"ra": "target_ra", "dec": "target_dec", "roll": "target_roll"}[
        dimension
    ]
    check.in_constraint.side_effect = lambda **kw: copysign(1.0, kw[backend_name]) < 0
    assert not science(constraint, check, **{dimension: 0.0})
    rebind(constraint)
    assert science(constraint, check, **{dimension: -0.0})
    rebind(constraint)
    assert not science(constraint, check, **{dimension: 0.0})
    assert check.in_constraint.call_count == 2
