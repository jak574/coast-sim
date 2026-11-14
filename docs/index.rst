COASTSim: ConOps Astronomical Space Telescope Simulator
========================================================

COASTSim is a Python module for simulating the Concept of Operations (ConOps)
of space telescopes and astronomical spacecraft missions. Mission planners and
engineers can use it to simulate Day-In-The-Life (DITL) scenarios, evaluate
spacecraft performance, optimize observation schedules, and validate operational
constraints before launch.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   configuration
   telemetry
   examples
   visualization
   sky_pointing_visualization
   spacecraft_3d_visualization
   communications
   data_management
   fault_management
   attitude_execution
   attitude_exports
   momentum
   radiator_shadowing
   target_of_opportunity
   planning
   plan_serialization
   api/modules
   contributing
   README

Features
--------

The features are listed below by the part of the simulation they belong to.

The spacecraft and its subsystems:

* The spacecraft bus simulation models power systems, attitude control, and thermal management.
* Power budget analysis covers solar panel modeling, battery management, and emergency charging scenarios.
* The attitude control system model simulates slews, pointing accuracy, and settle time.
* Star tracker support provides multi-tracker attitude determination, with configurable hard/soft constraints and functional-count monitoring.
* Radiator thermal modelling covers body-mounted radiator geometry, hard keep-out constraints, net heat-flow metrics driven by Sun/Earth exposure, and inter-component panel shadowing.
* Instrument modeling supports multi-instrument configurations with power and pointing requirements.
* Data management simulates onboard data storage, with generation and downlink modeling.
* Fault management provides extensible parameter monitoring with yellow/red thresholds and automatic safe mode.

The orbit and the constraints that follow from it:

* Orbit propagation computes a TLE-based ephemeris and tracks the orbit.
* Constraint management handles Sun, Moon, Earth limb, and custom geometric constraints.
* South Atlantic Anomaly (SAA) avoidance handles the radiation belt constraint.
* Ground station pass modeling calculates communication windows and plans data downlink.
* The instantaneous field of regard, the accessible sky solid angle, can optionally be computed at each simulation timestep.

Planning and running a simulation:

* Observation planning provides target queue management and scheduling algorithms.
* DITL generation simulates a day-in-the-life timeline.
* Plan serialisation saves and reloads observation plans as portable, version-stamped JSON files, with optional TLE provenance and executed attitude/orbit-state sidecars.

Outputs:

* The telemetry system provides structured telemetry data models for simulation analysis and visualization.
* The 3-D spacecraft visualizer draws an interactive, rotatable 3-D model of the spacecraft from its MissionConfig. The model shows the bus, telescope, solar panels, radiators, and star trackers, with body-frame axes and normal vectors.

Quick Example
-------------

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
