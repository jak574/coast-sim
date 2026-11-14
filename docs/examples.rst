Examples
========

COASTSim includes example Jupyter notebooks in the ``examples/`` directory. Each notebook
works through one feature or workflow from start to finish. This page lists the notebooks,
explains how to run them, and then describes common uses of the package.

Example Notebooks
-----------------

Example Spacecraft DITL
~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Example_Spacecraft_DITL.ipynb``

This notebook builds a spacecraft configuration in Python and runs a complete DITL
simulation with it. It covers:

* Custom spacecraft configuration
* Power modeling (battery and solar panels)
* Attitude control system simulation
* Observation scheduling with constraints
* Ground station pass calculations
* South Atlantic Anomaly (SAA) avoidance
* Timeline visualization and analysis

The notebook follows this outline:

.. code-block:: python

   # Configure spacecraft components
   battery = Battery(watthour=1000, max_depth_of_discharge=0.4)
   solar_panels = SolarPanelSet(panels=[...])
   acs = AttitudeControlSystem(
       settle_time=10, max_slew_rate=0.3, slew_acceleration=0.01
   )

   # Set up instruments
   payload = Payload(instruments=[...])

   # Create spacecraft bus
   spacecraft_bus = SpacecraftBus(power_draw=power_draw, attitude_control=acs)

   # Collect the components in a mission configuration
   config = MissionConfig(
       spacecraft_bus=spacecraft_bus,
       solar_panel=solar_panels,
       payload=payload,
       battery=battery,
       constraint=constraint,
   )

   # Run simulation
   ditl = QueueDITL(config=config, ephem=ephemeris, begin=begin, end=end, queue=queue)
   ditl.calc()

Example DITL from JSON
~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Example_DITL_from_JSON.ipynb``

This notebook shows a shorter workflow in which the spacecraft is described in a JSON
configuration file:

* Loading spacecraft configuration from JSON
* Quick DITL setup and execution
* Basic analysis and visualization

Because the configuration lives in a file, this approach suits rapid prototyping and
testing different configurations:

.. code-block:: python

   from conops import MissionConfig, QueueDITL
   from conops.visualization import plot_ditl_telemetry

   # Load the spacecraft from the configuration file
   config = MissionConfig.from_json_file("example_config.json")
   # The ephemeris is not stored in the file, so set it here
   config.constraint.ephem = ephemeris

   # Run simulation
   ditl = QueueDITL(config=config, ephem=ephemeris, begin=begin, end=end, queue=queue)
   ditl.calc()

   # Analyze
   plot_ditl_telemetry(ditl)

Quick DITL Example
~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Quick_DITL_Example.ipynb``

This notebook runs a basic DITL simulation that starts from the default
``MissionConfig``:

* Adding a telescope, star trackers and a radiator to the default configuration
* Filling the queue of a ``QueueDITL`` with random targets and running it
* Sky pointing, timeline, ACS mode and data management plots
* The event log and the fault management timeline
* Saving the plan to disk

Ground-Station Passes with Three Antenna Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Antenna_Types_Example.ipynb``

This notebook compares fixed, omni, and gimballed antennas over the same day in LEO:

* Booking passes before execution with the same ``PassTimes`` API
* Comparing spacecraft tracking, science collection, downlink, and recorder volume
* Showing gimbal repoint outages and the effect of restricted antenna reach
* Checking station scheduling probability and preserving saved bookings for replay

See :doc:`communications` and :doc:`planning` for reference.

Dynamic Merit and Targets of Opportunity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Dynamic_Merit_Example.ipynb``

This notebook demonstrates how the queue scheduler chooses its next target, turning on
each dynamic merit term in turn:

* Urgency, which lets the scheduler observe a request with a short window that it
  would miss if it followed priority order alone
* Cadence, which revisits monitoring targets on schedule
* Completion deficit, which keeps programs near their allocated share of time
* Targets of Opportunity, which are submitted as ordinary short-window requests. The
  notebook shows when one interrupts the current observation and when it waits

See :doc:`configuration` (Dynamic Merit) and :doc:`target_of_opportunity` for reference.

Rolling-Horizon Replanning
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Rolling_Horizon_DITL_Example.ipynb``

This notebook simulates a ground-planned mission with ``RollingHorizonDITL``:

* Planning 12 hours ahead and replanning every 6 hours, with a commit lead time
* A Target of Opportunity that triggers a rapid replan and interrupts an observation
* What each replan kept, dropped and added, and how long it took
* Checking that the simulation executed the plan it followed
* A comparison with ``QueueDITL`` dispatch on the same targets

See :doc:`planning` (Rolling-horizon replanning) for reference.

Local Search Planning
~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Local_Search_Scheduler_Example.ipynb``

This notebook compares ``LocalSearchScheduler`` with ``PriorityScheduler`` on 100 random
targets:

* A one-day plan from each scheduler. Each plan is executed to check that it runs
  exactly as planned
* How each plan divides the day between science, slewing and idle time, with the two
  timelines side by side
* Merit-weighted science through the day, and which targets gained or lost time
* How the result and the planning time grow with the number of changes tried
* A benchmark of ``RollingHorizonDITL`` runs with each scheduler over several target sets

See :doc:`planning` (Improving plans by local search) for reference.

CP-SAT Planning
~~~~~~~~~~~~~~~

**Location**: ``examples/CP_SAT_Scheduler_Example.ipynb``

This notebook compares ``CpSatScheduler`` with ``LocalSearchScheduler`` on 100 random
targets, using ``PriorityScheduler`` as the baseline. It needs the ``cpsat`` extra to
be installed.

* A one-day plan from each scheduler, with the same planning time allowed for each.
  Each plan is executed to check that it runs exactly as planned
* Science, merit-weighted science, idle time and targets completed
* The gain in each of CP-SAT's chunks, and through the day
* Which targets gained or lost time
* How each scheduler's result grows with planning time
* A benchmark of ``RollingHorizonDITL`` runs with each scheduler over several target sets

See :doc:`planning` (Optimizing plans with CP-SAT) for reference.

Planning Within the Recorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Recorder_Constraint_Example.ipynb``

This notebook plans a day for a spacecraft whose recorder is too small for everything
the scheduler would like to observe. The spacecraft has a 32 Gb recorder and generates
housekeeping data. Its target list holds fifty-two faint targets and eight bright ones.
Each bright target has a single 20-minute visit that produces 5 Gb.

* Two one-day plans, one made with the recorder's planning limit and one without it.
  Both are executed. With the limit, the recorder stays within it and the bright
  visits are rationed around the ground contacts. Without the limit, the recorder
  fills, data is lost and the spacecraft enters safe mode
* The scheduler's ledger, which lists what each contact downlinks and the recorder's
  peak volume before the next contact
* Which targets are observed under the limit
* How local search fills the time the recorder left idle. It trades observations on
  merit-weighted science while staying within the limit
* How CP-SAT does the same with the recorder included in its model
* How a lower ``planning_limit`` keeps margin
* Rolling-horizon replans that start from the recorder's actual volume

See :doc:`planning` (Planning within the recorder) and :doc:`data_management` for
reference.

Planning Within the Battery
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Battery_Constraint_Example.ipynb``

This notebook plans a day for a space telescope whose battery limits how much it can
observe in eclipse. The telescope has a side-mounted solar array, which generates
power while the telescope observes in sunlight. Its battery cannot supply a whole
eclipse of observing.

* Two one-day plans, one made with the battery's planning level and one without it.
  Both are executed. With the planning level, the battery dips to that level through
  each eclipse and no further. Without it, the battery is run past its discharge
  limit in every eclipse
* The scheduler's ledger, which lists the lowest battery level in each orbit, the
  energy that charging in the idle sunlit time puts back, and the science that
  waited for charge
* Local search trading snapshots on merit while keeping to the same level
* CP-SAT with the battery included in its model, which trades eclipse science on
  energy directly
* How a higher ``planning_level`` keeps margin
* Rolling-horizon replans that start from the battery's actual charge

See :doc:`planning` (Planning within the battery) for reference.

Stored Momentum from the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Momentum_Tracking_Example.ipynb``

The environment applies small torques to a spacecraft, and the reaction wheels store
the momentum those torques build up. This notebook tracks that stored momentum over
a day for two torque sources, gravity gradient and solar radiation pressure:

* The surfaces that sunlight acts on. The notebook starts from the placeholder model,
  which is built from the bus, the array and the aperture, and shows how a mission's
  mass properties replace it
* A day of dispatch. The telemetry gives the stored momentum and each source's share
  of it, and shows the eclipses, where sunlight contributes nothing
* The same day spent holding one attitude, compared with the varied schedule. At a
  fixed attitude both torques are secular, so the momentum keeps growing, while the
  varied schedule averages the torques
* The tracker run on its own with sunlight as the only source, which shows how the
  centre of mass sets sunlight's share

See :doc:`configuration` (Stored momentum) and :doc:`telemetry` for reference.

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/example_config.json``

This is a complete JSON configuration file that shows all available parameters:

* Spacecraft bus configuration
* Solar panel specifications
* Instrument definitions
* Battery parameters
* Attitude control system settings
* Ground station network
* Constraint definitions

Load it with ``MissionConfig.from_json_file``. The same directory holds
``examples/example_config.yaml``, an annotated configuration file in YAML, which is
loaded with ``MissionConfig.from_yaml_file``.

Example TLE File
~~~~~~~~~~~~~~~~

**Location**: ``examples/example.tle``

This is a sample Two-Line Element (TLE) file for testing orbit propagation.

Running the Examples
--------------------

To run the example notebooks:

.. code-block:: bash

   cd examples
   jupyter notebook

Then open any of the ``.ipynb`` files in your browser.

Common Use Cases
----------------

The sections below describe the kinds of study COASTSim is typically used for and
list what you can examine in each.

Mission Planning
~~~~~~~~~~~~~~~~

Use COASTSim to evaluate operational scenarios before launch:

* Test different target scheduling strategies
* Validate power budget over multiple orbits
* Assess ground station coverage
* Optimize observation efficiency

Performance Analysis
~~~~~~~~~~~~~~~~~~~~

Assess how the spacecraft performs over a simulated period:

* Power generation vs. consumption analysis
* Thermal constraints during different operational modes
* Data downlink capacity and scheduling
* Slew time and observation efficiency

Constraint Validation
~~~~~~~~~~~~~~~~~~~~~

Verify that the observing constraints are satisfied:

* Sun angle avoidance
* Moon exclusion zones
* Earth limb constraints
* South Atlantic Anomaly avoidance
* Custom geometric constraints

Schedule Optimization
~~~~~~~~~~~~~~~~~~~~~

Test different scheduling algorithms:

* Priority-based scheduling
* Greedy vs. optimal schedulers
* Target visibility windows
* Constraint satisfaction

Data Management
~~~~~~~~~~~~~~~

Configure and simulate onboard data storage, generation, and downlink:

.. code-block:: python

   from datetime import datetime

   from rust_ephem import TLEEphemeris

   from conops import MissionConfig, QueueDITL
   from conops.config import (
       BandCapability,
       DataGeneration,
       GroundStation,
       GroundStationRegistry,
       Instrument,
       OnboardRecorder,
       Payload,
       PowerDraw,
       ScienceData,
   )

   # Configure instruments with data generation
   camera = Instrument(
       name="High-Res Camera",
       power_draw=PowerDraw(nominal_power=100),
       data_generation=DataGeneration(science=ScienceData(rate_gbps=0.0005))  # 0.5 Mbps while collecting
   )

   spectrometer = Instrument(
       name="Spectrometer",
       power_draw=PowerDraw(nominal_power=50),
       data_generation=DataGeneration(science=ScienceData(per_observation_gb=2.0))  # 2 Gb per obs
   )

   payload = Payload(instruments=[camera, spectrometer])

   # Configure onboard data recorder
   recorder = OnboardRecorder(
       name="Solid State Recorder",
       capacity_gb=128.0,  # 128 Gigabits capacity
       yellow_threshold=0.7,  # 70% full warning
       red_threshold=0.9  # 90% full critical
   )

   # Configure ground station with downlink capability
   station = GroundStation(
       code="GND",
       name="Ground Station",
       latitude_deg=35.0,
       longitude_deg=-106.0,
       bands=[
           BandCapability(band="X", downlink_rate_mbps=100.0)  # 100 Mbps downlink
       ],
   )

   # Add to configuration
   config = MissionConfig(
       payload=payload,
       recorder=recorder,
       ground_stations=GroundStationRegistry(stations=[station]),
       # ... other config
   )

   # Run simulation - data automatically generated and downlinked
   ephem = TLEEphemeris(
       tle="examples/example.tle",
       begin=datetime(2025, 12, 1),
       end=datetime(2025, 12, 2),
       step_size=60,
   )
   ditl = QueueDITL(config=config, ephem=ephem)
   for i in range(50):
       ditl.queue.add(ra=7.2 * i, dec=-60.0 + 2.4 * i, obsid=10000 + i, exptime=1000)
   ditl.calc()

   # Analyze data management
   import matplotlib.pyplot as plt

   plt.figure(figsize=(12, 8))

   # Plot recorder fill level
   plt.subplot(3, 1, 1)
   plt.plot(ditl.utime, ditl.recorder_fill_fraction)
   plt.ylabel('Recorder Fill Fraction')
   plt.title('Onboard Data Recorder Status')

   # Plot data generation
   plt.subplot(3, 1, 2)
   plt.plot(ditl.utime, ditl.data_generated_gb)
   plt.ylabel('Data Generated (Gb)')

   # Plot data downlink
   plt.subplot(3, 1, 3)
   plt.plot(ditl.utime, ditl.data_downlinked_gb)
   plt.ylabel('Data Downlinked (Gb)')
   plt.xlabel('Time')

   plt.tight_layout()
   plt.show()

The example uses the following data management behaviour:

* An instrument with rate-based generation produces data continuously at the specified
  rate in Gbps.
* An instrument with per-observation generation produces a fixed amount of data for
  each observation.
* The recorder limits the stored volume to its capacity automatically and applies its
  alert thresholds.
* Data is downlinked automatically during ground station passes.
* The fault management system monitors the recorder fill level.

Creating Your Own Examples
---------------------------

To build your own simulation, follow these steps:

1. Start with the example configuration as a template
2. Modify spacecraft parameters to match your mission
3. Define your target list or observation queue
4. Set appropriate constraints for your mission
5. Run the simulation over your desired time period
6. Analyze and visualize the results

For detailed API documentation, see :doc:`api/modules`.
