Quick Start Guide
=================

This guide walks through a first Day-In-The-Life (DITL) simulation and then
introduces the parts of COASTSim that the simulation uses. A DITL simulation
steps a spacecraft through a period of operations and records what it does.

The example below loads a default configuration, computes an orbit from a TLE
file, fills the observation queue with random targets, runs the simulation and
plots the results:

.. code-block:: python

    from datetime import datetime, timedelta
    from matplotlib import pyplot as plt
    import numpy as np
    from rust_ephem import TLEEphemeris

    from conops import MissionConfig, QueueDITL
    from conops.visualization import plot_ditl_timeline, plot_sky_pointing

    # Load configuration - use the default MissionConfig for this example
    config = MissionConfig()

    # Set simulation period
    begin = datetime(2025, 11, 1)
    end = begin + timedelta(days=1)

    # Compute orbit ephemeris
    ephemeris = TLEEphemeris(
        begin=begin,
        end=end,
        step_size=60,
        tle="examples/example.tle",
    )

    # Create DITL object
    # Field-of-regard telemetry is disabled by default.
    # Enable with calculate_field_of_regard=True when needed.
    ditl = QueueDITL(config=config, ephem=ephemeris)

    # Add 1000 random observations to the observation queue
    for i in range(1000):
        ditl.queue.add(
            ra=np.random.uniform(0, 360),
            dec=np.random.uniform(-90, 90),
            exptime=1000,
            obsid=10000 + i,
        )


    # Run DITL simulation
    ditl.calc()

    # Analyze results visually and statistically
    _ = plot_sky_pointing(ditl, figsize=(10, 5))
    plt.show()
    _ = plot_ditl_timeline(ditl, figsize=(8, 6))
    plt.show()
    ditl.print_statistics()

Logging DITL Events
-------------------

A DITL simulation does not print what happens during a run. It records
structured events in a log, which you can read through ``ditl.log``. To keep
the logs of many runs, attach a ``DITLLogStore``, which saves the events to an
SQLite file as they are logged and lets you query them later by run:

.. code-block:: python

   # Inspect events collected during the run
   for event in ditl.log:
      print(event.timestamp, event.event_type, event.description)

   # Persist logs for many runs using the standard library (optional)
   from conops.ditl import DITLLogStore
   store = DITLLogStore("ditl_logs.sqlite")
   ditl.log.run_id = "my-run-001"
   ditl.log.store = store
   # Events are persisted as they are logged; you can also bulk flush
   ditl.log.flush_to_store()

   # Later, query events by run
   events = store.fetch_events("my-run-001", event_type="PASS")

   store.close()

Configuration-Based Approach
----------------------------

The spacecraft is described by a ``MissionConfig`` object. You can start from
the default configuration and change individual parameters:

.. code-block:: python

   from conops.config import MissionConfig

   # Create a default configuration
   config = MissionConfig()

   # Customize parameters as needed
   config.spacecraft_bus.power_draw.nominal_power = 75.0  # in Watts
   config.solar_panel.panels[0].max_power = 600.0  # in Watts

``MissionConfig`` holds all the spacecraft parameters that a simulation uses.
You can save it to a JSON file with ``config.to_json_file("filename.json")``,
either to store your configuration or to edit it by hand. Here is an example of
a JSON configuration file that defines spacecraft parameters:

.. code-block:: json

   {
       "name": "My Space Telescope",
       "spacecraft_bus": {
           "power_draw": {
               "nominal_power": 50.0,
               "peak_power": 300.0
           },
           "attitude_control": {
               "slew_acceleration": 0.01,
               "max_slew_rate": 1.0
           }
       },
       "solar_panel": {
           "panels": [
               {
                   "name": "Main Solar Panel",
                   "gimbled": false,
                   "normal": [0.0, 1.0, 0.0],
                   "max_power": 500.0,
                   "conversion_efficiency": 0.9
               }
           ]
       },
       "payload": {
           "instruments": [
               {
                   "name": "Telescope Camera",
                   "power_draw": {
                       "nominal_power": 100.0,
                       "peak_power": 150.0
                   }
               }
           ]
       }
   }

Load the file to get a configuration object back:

.. code-block:: python

   from conops.config import MissionConfig

   config = MissionConfig.from_json_file("my_spacecraft_config.json")

Key Components
--------------

Ephemeris
~~~~~~~~~

COASTSim relies on the `rust-ephem` library to compute the spacecraft orbit,
and it accepts any ephemeris that library supports. For example, to use TLEs:

.. code-block:: python

   from rust_ephem import TLEEphemeris
   from datetime import datetime, timedelta

   begin = datetime(2025, 11, 1)
   end = begin + timedelta(days=1)
   ephemeris = TLEEphemeris(tle="spacecraft.tle", begin=begin, end=end)


Queue Scheduler
~~~~~~~~~~~~~~~

The `QueueDITL` class is the most commonly used DITL simulator. It implements a
simple queue-based scheduling algorithm, in which targets are scheduled in order
of merit. Targets of equal merit are placed in a random order, so each
simulation run can yield different results. This helps with Monte Carlo
analyses.

Here is an example of adding targets to the observation queue:

.. code-block:: python

    ditl = QueueDITL(config=config, ephem=ephemeris)
    ditl.queue.add(
        ra=266,
        dec=-29,
        exptime=1000,
        obsid=10000,
        merit=55
    )


Choosing How to Schedule
~~~~~~~~~~~~~~~~~~~~~~~~

Queue dispatch is one of three ways COASTSim can decide what to observe. A
configuration's ``scheduler`` section chooses between them, and ``create_ditl``
builds the matching simulation from a list of targets:

* ``dispatch`` (the default): a `QueueDITL` picks the next target each time the
  spacecraft is free.
* ``planned``: a scheduler builds one plan for the whole run, which a `DITL`
  executes.
* ``rolling``: a `RollingHorizonDITL` follows a plan and rebuilds it as the run
  goes on, as a mission planned on the ground does.

In ``planned`` and ``rolling`` modes, ``kind`` chooses the scheduler that builds
the plans: ``priority`` (the default), ``local_search`` or ``cp_sat``.

.. code-block:: python

    from conops import Pointing, create_ditl
    from conops.config import SchedulerConfig

    config.scheduler = SchedulerConfig(mode="rolling", kind="local_search", time_limit=5)

    target = Pointing(config=config, ra=266, dec=-29, obsid=10000, merit=55)
    target.exptime = 1000

    ditl = create_ditl(config, [target], begin, end, ephem=ephemeris)
    ditl.calc()

The same section can be saved with the rest of the configuration:

.. code-block:: yaml

    scheduler:
      mode: rolling
      kind: local_search
      time_limit: 5

See :doc:`configuration` for every setting, and :doc:`planning` for what each
mode and scheduler does.


Constraints
~~~~~~~~~~~

Spacecraft pointing constraints are defined with constraint classes from the
`rust-ephem` library. For example, to create Sun and Moon angle constraints:

.. code-block:: python

   from rust_ephem import SunConstraint, MoonConstraint

   sun_constraint = SunConstraint(min_angle=45.0)
   moon_constraint = MoonConstraint(min_angle=30.0)

These can then be added to MissionConfig:

.. code-block:: python

   from conops.config import MissionConfig, Constraint

   config = MissionConfig()
   config.constraint = Constraint(
       sun_constraint=sun_constraint,
       moon_constraint=moon_constraint
   )

`rust-ephem` can combine constraints with logical operations (AND, OR, NOT) to
express more complex conditions. For example, to avoid pointing close to the
Sun only when it is daytime:

.. code-block:: python

   from conops.config import MissionConfig, Constraint
   from rust_ephem import SunConstraint, MoonConstraint, EclipseConstraint

   config = MissionConfig()
   config.constraint = Constraint(
       sun_constraint=SunConstraint(min_angle=45.0) & ~EclipseConstraint(),
       moon_constraint=MoonConstraint(min_angle=30.0)
   )

This ``sun_constraint`` applies only when the spacecraft is not in eclipse,
that is, when it is not in the Earth's shadow.

Module Structure
----------------

COASTSim is organized into several key modules:

* `conops.common`: Shared utilities, enums (ACSMode, ChargeState, ACSCommandType), and common functions
* `conops.config`: Configuration classes for spacecraft components (battery, solar panels, instruments, etc.)
* `conops.targets`: Target management classes (Pointing, Queue, Plan, PlanEntry)
* `conops.schedulers`: Schedulers (PriorityScheduler, SequentialScheduler, QueueScheduler)
* `conops.simulation`: Core simulation components (ACS, slews, ground station passes, SAA, emergency charging, etc.)
* `conops.ditl`: Day-In-The-Life simulation classes (DITL, DITLMixin, QueueDITL, RollingHorizonDITL) and ``create_ditl``
* `conops.visualization`: Visualization utilities for plotting results (sky pointing, timelines, etc.)

All classes are available directly from the ``conops`` package:

.. code-block:: python

   from conops import (
       MissionConfig, ACS, DITL, QueueDITL, Pointing, Queue,
       SequentialScheduler, ACSMode, ACSCommandType
   )

Next Steps
----------

* The :doc:`examples` page has detailed use cases.
* The :doc:`api/modules` page is the complete API reference.
* The ``examples/`` directory has Jupyter notebooks with complete workflows.
