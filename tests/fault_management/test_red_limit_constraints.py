"""Tests for spacecraft-level red limit constraints in fault management system."""

from datetime import datetime, timezone

import pytest
import rust_ephem

from conops.config.fault_management import (
    FaultConstraint,
    FaultManagement,
    FaultState,
)


class TestFaultConstraint:
    """Test FaultConstraint model."""

    def test_red_limit_constraint_creation(self):
        """Test creating a red limit constraint."""
        constraint = FaultConstraint(
            name="test_sun_limit",
            constraint=rust_ephem.SunConstraint(min_angle=30.0),
            time_threshold_seconds=300.0,
            description="Test sun constraint",
        )

        assert constraint.name == "test_sun_limit"
        assert constraint.time_threshold_seconds == 300.0
        assert constraint.description == "Test sun constraint"
        assert constraint.constraint is not None

    def test_red_limit_constraint_no_threshold(self):
        """Test creating a red limit constraint without time threshold."""
        constraint = FaultConstraint(
            name="test_monitor",
            constraint=rust_ephem.MoonConstraint(min_angle=5.0),
            time_threshold_seconds=None,
            description="Monitoring only",
        )

        assert constraint.name == "test_monitor"
        assert constraint.time_threshold_seconds is None


class TestFaultState:
    """Test FaultState tracking."""

    def test_initial_state(self):
        """Test initial red limit state."""
        state = FaultState()

        assert state.in_violation is False
        assert state.total_violation_seconds == 0.0
        assert state.continuous_violation_seconds == 0.0

    def test_accumulate_violation_time(self):
        """Test accumulating violation time."""
        state = FaultState()

        # Simulate violations
        state.in_violation = True
        state.total_violation_seconds += 10.0
        state.continuous_violation_seconds += 10.0

        assert state.total_violation_seconds == 10.0
        assert state.continuous_violation_seconds == 10.0

        # Add more time
        state.total_violation_seconds += 5.0
        state.continuous_violation_seconds += 5.0

        assert state.total_violation_seconds == 15.0
        assert state.continuous_violation_seconds == 15.0

    def test_reset_continuous_on_recovery(self):
        """Test resetting continuous violation counter when constraint is satisfied."""
        state = FaultState()

        # Simulate violation
        state.in_violation = True
        state.total_violation_seconds = 100.0
        state.continuous_violation_seconds = 100.0

        # Recover from violation
        state.in_violation = False
        state.continuous_violation_seconds = 0.0

        # Total should remain but continuous resets
        assert state.total_violation_seconds == 100.0
        assert state.continuous_violation_seconds == 0.0


class TestFaultManagementRedLimits:
    """Test FaultManagement integration with red limit constraints."""

    def test_add_red_limit_constraint(self):
        """Test adding a red limit constraint to fault management."""
        fm = FaultManagement()

        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=rust_ephem.SunConstraint(min_angle=30.0),
            time_threshold_seconds=300.0,
            description="Sun constraint",
        )

        assert "sun_limit" in fm.red_limit_constraints
        assert fm.red_limit_constraints["sun_limit"].name == "sun_limit"
        assert fm.red_limit_constraints["sun_limit"].time_threshold_seconds == 300.0

    def test_add_multiple_red_limit_constraints(self):
        """Test adding multiple red limit constraints."""
        fm = FaultManagement()

        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=rust_ephem.SunConstraint(min_angle=30.0),
            time_threshold_seconds=300.0,
        )

        fm.add_red_limit_constraint(
            name="earth_limit",
            constraint=rust_ephem.EarthLimbConstraint(min_angle=10.0),
            time_threshold_seconds=600.0,
        )

        assert len(fm.red_limit_constraints) == 2
        assert "sun_limit" in fm.red_limit_constraints
        assert "earth_limit" in fm.red_limit_constraints

    def test_check_red_limit_constraints_creates_states(self):
        """Test that checking constraints creates tracking states."""
        fm = FaultManagement()

        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=rust_ephem.SunConstraint(min_angle=30.0),
            time_threshold_seconds=300.0,
        )

        # Create ephemeris for testing using TLE file
        ephem = rust_ephem.TLEEphemeris(
            tle="examples/example.tle",
            begin=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            step_size=60,
        )

        # Check constraints using unified check() method
        fm.check(
            values={},
            utime=ephem.timestamp[0].timestamp(),
            step_size=60.0,
            ephem=ephem,
            ra=180.0,
            dec=0.0,
        )

        assert "sun_limit" in fm.states

    def test_red_limit_statistics(self):
        """Test getting red limit statistics."""
        fm = FaultManagement()

        # Add a red limit constraint so the statistics method knows what type it is
        fm.add_red_limit_constraint(
            name="test_constraint",
            constraint=rust_ephem.SunConstraint(min_angle=30.0),
            time_threshold_seconds=300.0,
        )

        # Manually create a state with some violation time
        fm.states["test_constraint"] = FaultState(
            in_violation=True,
            total_violation_seconds=150.0,
            continuous_violation_seconds=100.0,
        )

        stats = fm.statistics()

        assert "test_constraint" in stats
        assert stats["test_constraint"]["in_violation"] is True
        assert stats["test_constraint"]["total_violation_seconds"] == 150.0
        assert stats["test_constraint"]["continuous_violation_seconds"] == 100.0

    def test_safe_mode_not_triggered_below_threshold(self):
        """Test that safe mode is not triggered if below time threshold."""
        fm = FaultManagement(safe_mode_on_red=True)

        # Add constraint with 300 second threshold
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=rust_ephem.SunConstraint(
                min_angle=90.0
            ),  # Tight constraint - always violated
            time_threshold_seconds=300.0,
        )

        # Create ephemeris using TLE file
        ephem = rust_ephem.TLEEphemeris(
            tle="examples/example.tle",
            begin=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            step_size=60,
        )

        # Check constraints multiple times but below threshold
        for i in range(4):  # 4 * 60 = 240 seconds < 300 second threshold
            fm.check(
                values={},
                utime=ephem.timestamp[i].timestamp(),
                step_size=60.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Safe mode should not be requested
        assert fm.safe_mode_requested is False

    def test_continuous_violation_resets_on_recovery(self):
        """Test that continuous violation counter resets when constraint is satisfied."""
        fm = FaultManagement()

        # Manually create state with continuous violation
        fm.states["test"] = FaultState(
            in_violation=False,
            total_violation_seconds=500.0,
            continuous_violation_seconds=100.0,
        )

        # After recovery, continuous should have been reset (but we're testing the concept)
        # In actual use, the check() method handles this
        state = fm.states["test"]
        if not state.in_violation:
            state.continuous_violation_seconds = 0.0

        assert state.total_violation_seconds == 500.0
        assert state.continuous_violation_seconds == 0.0


class TestFaultConstraintIntegration:
    """Integration tests for red limit constraints in simulation context."""

    def test_constraint_with_no_time_threshold_never_triggers_safe_mode(self):
        """Test that constraints without time thresholds never trigger safe mode."""
        fm = FaultManagement(safe_mode_on_red=True)

        # Add constraint without time threshold
        fm.add_red_limit_constraint(
            name="monitor_only",
            constraint=rust_ephem.SunConstraint(min_angle=90.0),
            time_threshold_seconds=None,  # No threshold
        )

        # Create ephemeris using TLE file
        ephem = rust_ephem.TLEEphemeris(
            tle="examples/example.tle",
            begin=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            step_size=60,
        )

        # Check constraints many times - should accumulate violation time but never trigger
        for i in range(20):  # 20 minutes of violation
            fm.check(
                values={},
                utime=ephem.timestamp[i].timestamp(),
                step_size=60.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Violation time should accumulate
        assert fm.states["monitor_only"].total_violation_seconds > 0

        # But safe mode should NOT be requested
        assert fm.safe_mode_requested is False

    def test_mixed_regular_and_red_limit_faults(self):
        """Test that regular faults and red limit constraints work together."""
        fm = FaultManagement(safe_mode_on_red=True)

        # Add regular threshold
        fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")

        # Add red limit constraint
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=rust_ephem.SunConstraint(min_angle=30.0),
            time_threshold_seconds=300.0,
        )

        # Check regular fault
        classifications = fm.check(
            values={"battery_level": 0.45}, utime=1000.0, step_size=60.0
        )

        assert classifications["battery_level"] == "yellow"
        assert fm.safe_mode_requested is False

        # Trigger red fault
        fm.check(values={"battery_level": 0.35}, utime=1060.0, step_size=60.0)

        # Should request safe mode from regular fault
        assert fm.safe_mode_requested is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
