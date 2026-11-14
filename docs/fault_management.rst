Fault Management
================

Overview
--------

Fault management watches the spacecraft's state during a simulation and responds when something goes out of limits. It compares configured telemetry parameters against a yellow (warning) threshold and a red (critical) threshold, records how long each parameter spends in each state, and can put the spacecraft into safe mode when a red condition occurs.

Key Features
------------

Fault management does the following:

* It monitors any number of spacecraft parameters at once. Each parameter has a yellow (warning) limit and a red (critical) limit.
* A threshold can be of the "below" type or the "above" type, depending on whether low or high values are the fault.
* It checks spacecraft red limit constraints. These are health and safety pointing constraints, and they trigger safe mode after a set time in violation.
* It accumulates the time spent in the yellow and red states, and the time spent in constraint violations.
* It triggers safe mode on a red condition or a sustained constraint violation. Safe mode is irreversible.
* It validates threshold names: each must match a predefined ``Housekeeping`` field.

Configuration
-------------

Programmatic Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create and configure fault management using the ``FaultManagement`` class. A parameter threshold is added with ``add_threshold`` and a red limit constraint with ``add_red_limit_constraint``. Both are described in the sections that follow.

.. code-block:: python

   from conops.common import ACSMode
   from conops.config.fault_management import FaultManagement
   import rust_ephem

   # Create fault management system
   fm = FaultManagement()

   # Add parameter thresholds
   fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
    fm.add_threshold("power_usage", yellow=450.0, red=500.0, direction="above")
    fm.add_threshold("recorder_fill_fraction", yellow=0.8, red=0.95, direction="above")

   # Add mode-specific threshold (power usage only checked in SCIENCE mode)
   fm.add_threshold(
       "power_usage",
       yellow=450.0,
       red=500.0,
       direction="above",
       acs_modes=[ACSMode.SCIENCE]
   )

   # Add spacecraft red limit constraints
   fm.add_red_limit_constraint(
       name="spacecraft_sun_limit",
       constraint=rust_ephem.SunConstraint(min_angle=30.0),
       time_threshold_seconds=300.0,  # 5 minutes
       description="Prevent thermal damage from prolonged sun exposure"
   )

   fm.add_red_limit_constraint(
       name="spacecraft_earth_limit",
       constraint=rust_ephem.EarthLimbConstraint(min_angle=10.0),
       time_threshold_seconds=600.0,  # 10 minutes
       description="Prevent stray light contamination"
   )

   # Add monitoring-only constraint (no safe mode trigger)
   fm.add_red_limit_constraint(
       name="spacecraft_moon_monitor",
       constraint=rust_ephem.MoonConstraint(min_angle=5.0),
       time_threshold_seconds=None,  # No automatic safe mode
       description="Monitor moon proximity (informational only)"
   )

Threshold Parameters
^^^^^^^^^^^^^^^^^^^^

Each threshold requires:

* ``name``: Unique identifier for the parameter (must match Housekeeping attribute name)
* ``yellow``: Warning threshold value
* ``red``: Critical threshold value (must be more severe than yellow)
* ``direction``: Either ``"below"`` or ``"above"``

  * ``"below"``: Fault triggered when value ≤ threshold (e.g., battery level)
  * ``"above"``: Fault triggered when value ≥ threshold (e.g., temperature, power)

A threshold also accepts these optional settings:

* ``acs_modes``: The ACS modes in which the threshold is checked (see the next section). By default it is checked in all modes.
* ``triggers_safe_mode``: Whether a RED classification triggers safe mode. ``add_threshold`` defaults to ``True``. Set it to ``False`` for a threshold that should alert without triggering safe mode.
* ``safe_mode_delay_seconds``: The continuous time in RED required before safe mode is triggered. The default of 0 triggers safe mode as soon as the parameter is RED.

ACS Mode Filtering
^^^^^^^^^^^^^^^^^^

Thresholds can be restricted to specific Attitude Control System (ACS) modes using the ``acs_modes`` parameter. This allows different fault policies for different operational modes.

For example:

- Only check high power draw during SCIENCE mode
- Check recorder fill level in all modes except SAFE mode
- Monitor battery level in all modes (default behavior)

The following example configures thresholds with ACS modes:

.. code-block:: python

   from conops.common import ACSMode
   from conops.config.fault_management import FaultManagement

   fm = FaultManagement()

   # Check battery in all modes (default)
   fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")

   # Only check high power usage during science operations
   fm.add_threshold(
       "power_usage",
       yellow=450.0,
       red=500.0,
       direction="above",
       acs_modes=[ACSMode.SCIENCE]
   )

   # Check recorder fill limits in multiple modes
   fm.add_threshold(
       "recorder_fill_fraction",
       yellow=0.8,
       red=0.95,
       direction="above",
       acs_modes=[ACSMode.SCIENCE, ACSMode.SLEW, ACSMode.SETTLE]
   )

Valid threshold names are the predefined ``Housekeeping`` fields. Commonly used fields include:

* ``battery_level``
* ``power_usage`` / ``power_bus`` / ``power_payload``
* ``panel_illumination``
* ``recorder_fill_fraction`` / ``recorder_alert``
* ``sun_angle_deg`` / ``earth_angle_deg`` / ``moon_angle_deg``
* ``star_tracker_functional_count``: the number of trackers not in a soft constraint zone. Its threshold is configured automatically when star trackers are present.
* ``star_tracker_hard_violations``: the number of trackers inside a HARD_KEEPOUT zone. Its threshold is configured automatically when star trackers are present.
* ``radiator_hard_violations``: the number of radiators violating a hard keepout constraint. Its threshold is configured automatically when radiators are present.
* ``telescope_hard_violations``: a 0/1 flag for the telescope hard constraint. Its threshold is configured automatically when a telescope hard constraint exists.

During fault checking, the current ACS mode is determined from:

1. ``housekeeping.acs_mode`` (preferred)
2. ``acs.acsmode`` (fallback)

Thresholds are only evaluated when the current mode is in the ``acs_modes`` list. Omit ``acs_modes`` (or set to ``null``) to check in all modes.

Spacecraft Red Limit Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to threshold-based monitoring, you can define spacecraft-level red limit constraints for health and safety. A red limit constraint is a pointing constraint, such as a minimum angle from the Sun, that the spacecraft should not violate for longer than a set time. These constraints are typically looser than instrument constraints, which are optimized for data quality. They exist purely to protect spacecraft hardware.

Red limit constraints can be added programmatically:

.. code-block:: python

   from conops.config.fault_management import FaultManagement
   import rust_ephem

   fm = FaultManagement()

   # Add spacecraft sun constraint - thermal protection
   fm.add_red_limit_constraint(
       name="spacecraft_sun_limit",
       constraint=rust_ephem.SunConstraint(min_angle=30.0),
       time_threshold_seconds=300.0,  # 5 minutes
       description="Prevent thermal damage from prolonged sun exposure"
   )

   # Add spacecraft earth constraint - stray light protection
   fm.add_red_limit_constraint(
       name="spacecraft_earth_limit",
       constraint=rust_ephem.EarthLimbConstraint(min_angle=10.0),
       time_threshold_seconds=600.0,  # 10 minutes
       description="Prevent stray light contamination"
   )

   # Add monitoring-only constraint (no safe mode trigger)
   fm.add_red_limit_constraint(
       name="spacecraft_moon_monitor",
       constraint=rust_ephem.MoonConstraint(min_angle=5.0),
       time_threshold_seconds=None,  # No automatic safe mode
       description="Monitor moon proximity (informational only)"
   )

Red Limit Constraint Parameters
""""""""""""""""""""""""""""""""

Each red limit constraint requires:

* ``name``: Unique identifier for the constraint
* ``constraint``: rust_ephem constraint definition (sun, earth_limb, moon, etc.)
* ``time_threshold_seconds``: Maximum continuous time in violation before triggering safe mode (set to ``null`` for monitoring only)
* ``description``: Human-readable description of the constraint purpose

Programmatic Usage
------------------

Creating Fault Management
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.common import ACSMode
   from conops.config.fault_management import FaultManagement
   import rust_ephem

   # Create fault management system
   fm = FaultManagement()

   # Add thresholds programmatically
   fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
    fm.add_threshold("power_usage", yellow=450.0, red=500.0, direction="above")
    fm.add_threshold("recorder_fill_fraction", yellow=0.8, red=0.95, direction="above")

   # Add mode-specific threshold
   fm.add_threshold(
       "power_usage",
       yellow=450.0,
       red=500.0,
       direction="above",
       acs_modes=[ACSMode.SCIENCE]
   )

   # Add spacecraft red limit constraints
   fm.add_red_limit_constraint(
       name="spacecraft_sun_limit",
       constraint=rust_ephem.SunConstraint(min_angle=30.0),
       time_threshold_seconds=300.0,  # 5 minutes
       description="Prevent thermal damage from prolonged sun exposure"
   )

   fm.add_red_limit_constraint(
       name="spacecraft_earth_limit",
       constraint=rust_ephem.EarthLimbConstraint(min_angle=10.0),
       time_threshold_seconds=600.0,  # 10 minutes
       description="Prevent stray light contamination"
   )

   # Add monitoring-only constraint (no safe mode trigger)
   fm.add_red_limit_constraint(
       name="spacecraft_moon_monitor",
       constraint=rust_ephem.MoonConstraint(min_angle=5.0),
       time_threshold_seconds=None,  # No automatic safe mode
       description="Monitor moon proximity (informational only)"
   )

Checking Parameters
^^^^^^^^^^^^^^^^^^^

Call ``check()`` each simulation cycle to evaluate monitored parameters and red limit constraints:

.. code-block:: python

   from datetime import datetime, timezone
   from conops.ditl.telemetry import Housekeeping

   # Create housekeeping telemetry packet
   hk = Housekeeping(
       timestamp=datetime.now(tz=timezone.utc),
       battery_level=battery.battery_level,
       power_usage=power_system.total_draw,
       recorder_fill_fraction=data_system.buffer_usage_fraction,
       ra=current_pointing_ra,
       dec=current_pointing_dec,
       acs_mode=spacecraft_acs.acsmode
   )

   # Once per run, snapshot policies into a fresh evaluator/report.
   run = fm.new_run()
   # Check parameters and constraints on every timestep.
   classifications = run.check(housekeeping=hk, acs=spacecraft_acs)

   # classifications = {"battery_level": "yellow", "power_usage": "nominal", ...}
   # Red limit constraints are checked automatically using housekeeping data

Retrieving Statistics
^^^^^^^^^^^^^^^^^^^^^

Get accumulated time in each fault state and constraint violation statistics:

.. code-block:: python

   stats = run.statistics()

   # For threshold-based parameters:
   # {
   #     "battery_level": {
   #         "yellow_seconds": 120.0,
   #         "red_seconds": 0.0,
   #         "current": "yellow"
   #     },
    #     "power_usage": {
   #         "yellow_seconds": 45.0,
   #         "red_seconds": 30.0,
   #         "current": "red"
   #     }
   # }

   # For red limit constraints:
   # {
   #     "spacecraft_sun_limit": {
   #         "in_violation": False,
   #         "total_violation_seconds": 180.0,
   #         "continuous_violation_seconds": 0.0
   #     }
   # }

Separating Statistics by Type
""""""""""""""""""""""""""""""

To separate threshold-based and constraint-based statistics:

.. code-block:: python

   stats = run.statistics()

   # Get red limit constraint stats
   constraint_stats = {
       name: data for name, data in stats.items()
       if any(c.name == name for c in fm.red_limit_constraints)
   }

   threshold_stats = {
       name: data for name, data in stats.items()
       if any(t.name == name for t in fm.thresholds)
   }

Integration with QueueDITL
--------------------------

When fault management is configured, the ``QueueDITL`` simulation loop runs it automatically. It checks parameters after each power update:

.. code-block:: python

   from conops.config import MissionConfig
   from conops.ditl import QueueDITL

   # Load config with fault_management section
   config = MissionConfig.from_json("config_with_fault_management.json")

   # Initialize defaults (adds battery_level, recorder_fill_fraction,
   # star_tracker_functional_count, star_tracker_hard_violations,
   # radiator_hard_violations, and telescope_hard_violations
   # thresholds if not already present)
   config.init_fault_management_defaults()

   # Run simulation
   ditl = QueueDITL(config=config, queue=target_queue, ephem=ephem, begin=begin, end=end)
   ditl.calc()

   # Check fault statistics after simulation
   stats = ditl.fault_management.statistics()
   print(f"Fault statistics: {stats}")

Safe Mode Behavior
------------------

Safe mode is triggered by either of two conditions: a parameter reaching the RED state, or a red limit constraint exceeding its time threshold. The trigger only acts when ``safe_mode_on_red`` is ``true``, which is the default.

Safe mode is then entered in two steps:

1. The ``safe_mode_requested`` flag is set to ``True``.
2. The QueueDITL loop detects the flag and enqueues an ``ENTER_SAFE_MODE`` command.

Once the spacecraft is in safe mode:

* Safe mode cannot be exited. The operation is irreversible.
* The spacecraft points its solar panels at the Sun for maximum power.
* The command queue is cleared, so all pending commands are discarded.
* The system operates in a minimal power configuration.

Red Limit Constraint Triggering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Red limit constraints track continuous violation time. The constraint must be violated continuously for the entire ``time_threshold_seconds`` duration before safe mode is triggered. If the constraint is satisfied, even briefly, the continuous violation counter resets to zero.

Counting continuous time means that a brief violation during a slew or maneuver does not trigger safe mode, which gives operations a reasonable grace period. A sustained violation that could cause hardware damage does trigger it.

Set ``time_threshold_seconds`` to ``null`` to create monitoring-only constraints that track violations but never trigger safe mode.

Star Tracker Monitoring
^^^^^^^^^^^^^^^^^^^^^^^

The ACS exposes two separate star-tracker telemetry fields:

* ``star_tracker_functional_count``: ``num_trackers - soft_violation_count``. It decreases when a tracker
  enters a soft constraint zone, where pointing quality is degraded. Hard violations do not affect this count.
* ``star_tracker_hard_violations``: Count of trackers inside a HARD_KEEPOUT zone. It is always non-negative,
  and a non-zero value means a tracker is in a health-and-safety exclusion region.

``init_fault_management_defaults()`` adds both thresholds automatically when star trackers are configured.
You can override the hard-violation policy, for example to trigger a safehold instead of just alerting.
To do so, add your threshold before calling ``init_fault_management_defaults()``. The default is skipped
when a threshold of that name already exists:

.. code-block:: python

     from conops.config.fault_management import FaultManagement

     fm = FaultManagement(safe_mode_on_red=True)

     # Override default: trigger safe mode after 30 s of continuous hard violation
     fm.add_threshold(
             "star_tracker_hard_violations",
             yellow=0.5,
             red=0.5,
             direction="above",
             triggers_safe_mode=True,
             safe_mode_delay_seconds=30.0,
     )

JSON configuration equivalent:

.. code-block:: json

     {
         "fault_management": {
             "safe_mode_on_red": true,
             "thresholds": [
                 {
                     "name": "star_tracker_hard_violations",
                     "yellow": 0.5,
                     "red": 0.5,
                     "direction": "above",
                     "triggers_safe_mode": true,
                     "safe_mode_delay_seconds": 30.0
                 }
             ]
         }
     }

The threshold values ``yellow=0.5`` and ``red=0.5`` work because the violation count is an integer: ``0 < 0.5``
(nominal) and ``1 >= 0.5`` (RED). Setting yellow and red to the same value means any violation goes
straight to RED without a separate YELLOW warning.

Radiator and Telescope Hard Monitoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same pattern applies to the other hard keepout families:

* ``radiator_hard_violations``: Count of radiators inside a hard keepout zone. Its threshold is added
  automatically when radiators are configured (``has_radiators``).
* ``telescope_hard_violations``: 0 or 1, according to whether the telescope hard constraint is currently
  violated. Its threshold is added automatically when a telescope hard constraint exists in the payload
  configuration.

Both default thresholds use the same parameters as the ``star_tracker_hard_violations`` default:
``direction="above"``, ``yellow=red=0.5``, ``triggers_safe_mode=False``. To override one, for example to
enable a safehold, add your own threshold before calling ``init_fault_management_defaults()``:

.. code-block:: python

     fm.add_threshold(
             "radiator_hard_violations",
             yellow=0.5,
             red=0.5,
             direction="above",
             triggers_safe_mode=True,
             safe_mode_delay_seconds=60.0,
     )

Example Configuration File
---------------------------

Complete example configurations are available in:

* ``examples/example_config_with_fault_management.json`` - Threshold-based monitoring
* ``examples/example_config_with_red_limits.json`` - Red limit constraints
* ``examples/example_spacecraft_red_limits.py`` - Programmatic red limit configuration

The threshold-based example demonstrates monitoring of:

* ``battery_level``: Warning at 50%, critical at 40%
* ``power_usage``: Warning at 450W, critical at 500W
* ``recorder_fill_fraction``: Warning at 80%, critical at 95%

The red limit example demonstrates spacecraft health and safety constraints:

* ``spacecraft_sun_limit``: 30° exclusion zone, 5 minute threshold (thermal protection)
* ``spacecraft_earth_limit``: 10° exclusion zone, 10 minute threshold (stray light protection)

Event Log
---------

All significant fault management transitions are recorded in an in-memory ``events`` list on ``FaultManagementRun``. Each entry is a ``FaultEvent`` with these fields:

* ``utime``: Unix timestamp (float)
* ``event_type``: One of ``threshold_transition``, ``constraint_violation``, ``safe_mode_trigger``
* ``name``: Threshold / constraint name
* ``cause``: Human-readable description
* ``metadata``: Optional contextual dict. The keys vary by event and may include the current value, thresholds, RA/Dec and violation durations.

Example:

.. code-block:: python

    # After running run.check(...)
    for evt in run.events:
         print(evt)  # Uses concise __str__ representation

Filtering events:

.. code-block:: python

    safe_mode_events = [e for e in run.events if e.event_type == "safe_mode_trigger"]
    sun_constraint_events = [e for e in run.events if e.name == "spacecraft_sun_limit"]

Run ownership and migration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fault management is split into a policy and a report. The policy is
``config.fault_management``, which contains only thresholds, red-limit constraints,
and the global response policy. The report is a ``FaultManagementRun``, which holds
the runtime fields produced by one run: ``states``, ``events``, and
``safe_mode_requested``. The runtime fields are not part of the mission YAML/JSON.
If an older configuration file contains them, remove them, and save fault reports
separately.

Both DITL implementations allocate a fresh report at each ``calc()`` call:

* The ACS and simulation share that report as ``ditl.fault_management``.
* Prior reports remain in ``ditl.fault_runs``, including reports from failed
  executions.
* Each report has a unique ``run_id``, so reruns covering identical dates are
  distinct.
* Policies are copied at run start, so later config edits cannot rewrite old
  reports.

Do not clear historical events to start another run.

.. code-block:: python

    ditl.calc()
    first = ditl.fault_management
    ditl.calc()
    second = ditl.fault_management
    assert first.run_id != second.run_id
    reports = {run.run_id: run.model_dump(mode="json") for run in ditl.fault_runs}

When separate simulation objects share a config, each simulation owns its own
report list. Collect their reports explicitly when comparing runs. Fault plots
accept a ``FaultManagementRun``, e.g. ``plot_fault_management_timeline(first)``.

A subsequent ``calc()`` on the same simulation resets the ACS execution state,
which includes the SAFE latch, pending commands and maneuvers. It also resets the
execution telemetry. Its initial attitude is the previously executed attitude.
This is not a full scenario replay, because battery, recorder and
target-consumption state are not reset. Construct a new simulation with fresh
scenario inputs when an identical initial physical state is required. Within a
run, SAFE remains irreversible.

Code that read or evaluated faults on ``config.fault_management`` should be
changed to use ``ditl.fault_management``. Standalone callers, which run fault
management without a DITL, call ``policy.new_run()`` once per execution and retain
the returned report. Do not call ``new_run()`` on every timestep.

Housekeeping Schema and New Metrics
-----------------------------------

Threshold names must match predefined ``Housekeeping`` fields.

To monitor a new metric, first add it to the ``Housekeeping`` model in code, and then add thresholds for the new field. Arbitrary extra fields in ``Housekeeping(...)`` are not accepted.

API Reference
-------------

See :mod:`conops.fault_management` for detailed API documentation.

Best Practices
--------------

Threshold-Based Monitoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Set each yellow threshold so that it gives an early warning before the critical limit.
* Test the threshold values to check that they don't cause premature safe mode triggers.
* Review the accumulated yellow and red time after simulations.
* Always include ``battery_level`` monitoring for power-critical missions.

Red Limit Constraints
^^^^^^^^^^^^^^^^^^^^^^

* Make red limits less restrictive than instrument (science) constraints.
* Design the constraints around hardware protection: thermal limits, detector saturation, etc.
* Choose time thresholds that allow transient violations during normal operations (slews, etc.).
* Test the time thresholds to verify that they don't trigger during routine maneuvers.
* To track violations without triggering safe mode, use monitoring mode by setting ``time_threshold_seconds: null``.

ACS Mode Filtering
^^^^^^^^^^^^^^^^^^^

* Use ``acs_modes`` to check parameters only in the modes where they matter operationally.
* Monitor high power usage during SCIENCE mode when needed.
* Check recorder fill limits in the operational modes where data accumulation matters.
* Monitor battery levels in all modes, because power is always critical.
* Verify that thresholds behave correctly during mode changes.

General
^^^^^^^

* Consider setting ``safe_mode_on_red: false`` for analysis runs where you want to observe fault behavior without intervention.
* Use thresholds for subsystem health (battery, power, recorder fill) and red limits for pointing safety.
