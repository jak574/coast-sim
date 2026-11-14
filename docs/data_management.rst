Data Management
===============

COASTSim simulates how a spacecraft generates data, stores it on board, and downlinks it to
the ground. Mission planners can use this to evaluate:

* Data recorder sizing requirements
* Downlink pass frequency and duration needs
* Data generation rates from science instruments
* Storage capacity utilization over mission timelines
* Alert thresholds for recorder fullness

Overview
--------

The data management system has three parts, each described in its own section below:

1. `OnboardRecorder`_ simulates the spacecraft's solid-state recorder (SSR) or other data
   storage device.
2. `DataGeneration`_ models how the bus and each instrument produce data, and how the
   observed target changes the amount.
3. `Downlink Operations`_ simulates the transfer of data to the ground during scheduled
   ground-station passes (see :doc:`communications`).

A DITL simulation uses all three together to track the flow of data throughout the
spacecraft's operations.

OnboardRecorder
---------------

The :class:`~conops.config.recorder.OnboardRecorder` class simulates an onboard data storage
device with configurable capacity and alert thresholds.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from conops.config import OnboardRecorder

   # Create a recorder with 128 Gigabit capacity
   recorder = OnboardRecorder(
       name="Solid State Recorder",
       capacity_gb=128.0,          # Maximum capacity in Gigabits
       current_volume_gb=0.0,      # Starting empty
       yellow_threshold=0.7,       # Warning at 70% full
       red_threshold=0.9           # Critical at 90% full
   )

Key Features
~~~~~~~~~~~~

Capacity Management
^^^^^^^^^^^^^^^^^^^

The recorder never holds more than its configured capacity. Data generated while the
recorder is full is lost, which simulates data that cannot be stored.

.. code-block:: python

   # Add data to recorder
   data_stored = recorder.add_data(10.5)  # Returns amount actually stored

   # Check current state
   print(f"Current volume: {recorder.current_volume_gb} Gb")
   print(f"Fill fraction: {recorder.get_fill_fraction()}")
   print(f"Available space: {recorder.available_capacity()} Gb")

Alert System
^^^^^^^^^^^^

The recorder reports one of three alert levels, based on its fill fraction:

.. list-table::
   :header-rows: 1
   :widths: 15 15 40 30

   * - Level
     - Color
     - Fill fraction
     - Meaning
   * - 0
     - Green
     - Below the yellow threshold
     - Normal operations
   * - 1
     - Yellow
     - At or above the yellow threshold
     - Warning state
   * - 2
     - Red
     - At or above the red threshold
     - Critical state

.. code-block:: python

   alert_level = recorder.get_alert_level()
   if alert_level == 2:
       print("CRITICAL: Recorder nearly full!")
   elif alert_level == 1:
       print("WARNING: Recorder filling up")

Data Removal
^^^^^^^^^^^^

During scheduled ground-station passes, data is removed from the recorder to
simulate downlink:

.. code-block:: python

   # Remove data during downlink
   data_downlinked = recorder.remove_data(25.0)  # Returns amount actually removed
   print(f"Downlinked {data_downlinked} Gb")

DataGeneration
--------------

COASTSim models two kinds of data. They are configured separately because different things
determine how much of each is generated.

Science data
~~~~~~~~~~~~

Science data depends on *what is observed*, and it is generated only while the spacecraft
is observing. It is described by :class:`~conops.config.ScienceData`, which has two fields:

* ``rate_gbps`` is the rate at which data is generated for every second of collection.
* ``per_observation_gb`` is a fixed amount that each observation generates whatever its
  length, such as a frame readout or headers.

Each instrument sets the science data it generates by default. A target can set its own
values for each instrument (see `Target Science Data`_ below).

Housekeeping data
~~~~~~~~~~~~~~~~~

Housekeeping data is the engineering telemetry: temperatures, voltages, attitude and
status. It depends on *what the spacecraft is doing*. It is generated at all times,
whatever is being observed. It is described by :class:`~conops.config.HousekeepingData`,
which has two fields:

* ``rate_gbps`` is the rate that applies at all times.
* ``rate_mode`` sets a different rate in particular ACS modes. A mode can be given by name
  or by number. This works in the same way that :class:`~conops.config.PowerDraw` sets
  power by mode.

The dispatch scheduler also has to allow for the housekeeping data generated while the
spacecraft waits between observations. The spacecraft can spend that time in any of
several modes, so dispatch reserves for it at the highest of the nominal, ``IDLE``,
``CHARGING`` and ``SAA`` rates. This value is given by
:meth:`~conops.config.MissionConfig.housekeeping_while_waiting`. See :doc:`planning`.

Configuring the bus and the instruments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spacecraft bus generates housekeeping data only, so its ``data_generation`` is a
``HousekeepingData``. An instrument generates both kinds, so its ``data_generation`` is a
:class:`~conops.config.DataGeneration` with a ``science`` part and a ``housekeeping`` part.

.. code-block:: python

   from conops.config import (
       DataGeneration,
       HousekeepingData,
       Instrument,
       MissionConfig,
       Payload,
       ScienceData,
       SpacecraftBus,
   )

   config = MissionConfig(
       spacecraft_bus=SpacecraftBus(
           # 1 Mbps of housekeeping, 4 Mbps of diagnostics in safe mode
           data_generation=HousekeepingData(
               rate_gbps=0.001, rate_mode={"SAFE": 0.004}
           ),
       ),
       payload=Payload(
           instruments=[
               # 0.5 Gbps while collecting, and its own housekeeping
               Instrument(
                   name="camera",
                   data_generation=DataGeneration(
                       science=ScienceData(rate_gbps=0.5),
                       housekeeping=HousekeepingData(rate_gbps=0.0002),
                   ),
               ),
               # 5 Gb for each observation, whatever its length; off in the SAA
               Instrument(
                   name="spectrometer",
                   data_generation=DataGeneration(
                       science=ScienceData(per_observation_gb=5.0),
                       housekeeping=HousekeepingData(
                           rate_gbps=0.0001, rate_mode={"SAA": 0.0}
                       ),
                   ),
               ),
           ]
       ),
   )

Computing the data generated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each part computes its own data with a ``data_generated`` method:

* ``ScienceData.data_generated`` takes the number of seconds an observation has collected.
  It returns the rate multiplied by those seconds, plus the fixed amount.
* ``HousekeepingData.data_generated`` takes the length of an interval and, optionally, the
  ACS mode during it. It returns the rate in that mode multiplied by the length.
* An instrument's ``DataGeneration.data_generated`` adds the two. If it is given only a
  duration, it assumes the observation collects for the whole of it.

.. code-block:: python

   camera = config.payload.instruments[0]
   camera.data_generation.science.data_generated(60)      # 0.5 * 60 = 30 Gb
   camera.data_generation.housekeeping.data_generated(60)  # 0.0002 * 60 = 0.012 Gb
   camera.data_generation.data_generated(60)              # 30.012 Gb

   spectrometer = config.payload.instruments[1]
   spectrometer.data_generation.science.data_generated(60)   # 5 Gb, whatever the length
   spectrometer.data_generation.data_generated(60)           # 5 + 0.0001 * 60 = 5.006 Gb

The same calculation is available at higher levels.
:meth:`Payload.data_generated <conops.config.Payload.data_generated>`,
:meth:`SpacecraftBus.data_generated <conops.config.SpacecraftBus.data_generated>` and
:meth:`MissionConfig.data_generated <conops.config.MissionConfig.data_generated>` each
add up what their generators produce over an interval, given the ACS mode and the seconds
collected. The ``housekeeping_generated`` and ``science_generated`` methods give the two
parts separately:

.. code-block:: python

   from conops.common.enums import ACSMode

   # A 60 s step in safe mode
   config.housekeeping_generated(60, mode=ACSMode.SAFE)
   # An observation that has collected for 45 s
   config.science_generated(45)
   # Both: an observation collecting for 45 s of a 60 s interval
   config.data_generated(60, mode=ACSMode.SCIENCE, collection_seconds=45)

For many simulations only the science rate matters. As a shorthand, an instrument's
science fields may be given directly to ``DataGeneration``, and they become its ``science``
part. For example, write ``DataGeneration(rate_gbps=0.5)`` in code, or
``data_generation: {rate_gbps: 0.5}`` in a configuration file.

Planning within the recorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The schedulers keep the recorder within ``recorder.planning_limit``. This is a fraction
of the capacity, and it defaults to the red threshold. As a scheduler builds a plan, it
works out how the recorder fills with the data described above and empties during the
reserved contacts. It places a science observation only where the recorder stays within the
limit at every peak. See :doc:`planning`.

Target Science Data
~~~~~~~~~~~~~~~~~~~

The amount of data an observation generates depends on the target. A bright source, for
example, fills a photon-counting detector faster than a faint one. To model this, give the
target a ``science_data`` field. It maps instrument names to the
:class:`~conops.config.ScienceData` that each instrument generates while observing that
target. For a named instrument, the target's values replace the instrument's own
``rate_gbps`` and ``per_observation_gb``. An instrument that the target does not name
generates its own default science data. Housekeeping data is unchanged.

.. code-block:: python

   from conops.config import ScienceData

   queue.add(
       ra=83.63,
       dec=22.01,
       obsid=1001,
       name="Crab",
       science_data={
           "camera": ScienceData(rate_gbps=1.2),
           "spectrometer": ScienceData(per_observation_gb=12.0),
       },
   )

   ditl.submit_too(
       obsid=2001, ra=180.0, dec=45.0, merit=10000.0, exptime=3600,
       name="GRB 250101A",
       science_data={"camera": ScienceData(rate_gbps=2.0)},
   )

Schedulers copy a target's ``science_data`` onto its plan entries, and it is saved
with the plan. If a target is created with a configuration and its ``science_data`` uses a
name that matches no configured instrument, an error is raised.

Data generation during a simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A DITL simulation advances in time steps, and each step adds data to the recorder.

Housekeeping data is generated for the whole step, at the rate for the step's ACS mode.

Science data is handled per observation. The simulation keeps a running total of what the
observation in progress has generated so far, which is its rate for the seconds it has
collected plus its fixed amount. Each step adds only the increase in that total. The fixed
amount is therefore counted once per observation, however many steps the observation
spans. A later visit to the same target is a new observation and generates the fixed amount
again.

Downlink Operations
-------------------

Data is downlinked automatically during ground station passes. The downlink rate
is configured in the ground station's antenna parameters.

Ground Station Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from conops.config import GroundStation, Antenna, GroundStationRegistry

   # Configure station with X-band downlink at 100 Mbps
   station = GroundStation(
       code="GND",
       name="Ground Station",
       latitude_deg=35.0,
       longitude_deg=-106.0,
       antenna=Antenna(
           bands=["X"],
           max_data_rate_mbps=100.0  # 100 Mbps = 0.1 Gbps
       )
   )

   registry = GroundStationRegistry(stations=[station])

Downlink Calculation
~~~~~~~~~~~~~~~~~~~~

During a ground station pass, each step of a DITL simulation does the following:

1. It identifies the active pass.
2. It takes the rate the spacecraft gets from the station, which is given by
   :func:`~conops.config.groundstation.effective_downlink_rate_mbps`.
3. It converts the rate to Gbps by dividing by 1000.
4. It calculates the data volume for the step as ``rate × seconds in contact``.
5. It removes that amount from the recorder, at the same time as it adds what the step
   generates.

.. code-block:: python

   # Example: 100 Mbps for 60 seconds
   downlink_rate_gbps = 100.0 / 1000.0  # 0.1 Gbps
   time_step = 60  # seconds
   data_downlinked = downlink_rate_gbps * time_step  # 6.0 Gb

Housekeeping data continues to be generated during a contact, and the link carries it
along with the data that is already stored. Within a step, the simulation therefore applies
generation and downlink together, using :meth:`~conops.config.OnboardRecorder.transfer`,
instead of adding one and then removing the other. This has three consequences:

* The recorder's volume changes at the net rate of generation minus downlink.
* The recorder cannot overflow while the link removes data faster than it arrives.
* Data generated while the recorder is empty goes straight down the link.

A contact may begin or end partway through a step. In that case the downlink is credited
only for the seconds in contact. Data generated before the contact begins is stored first,
and data generated after it ends stays on the recorder. Because of this, the result is the
same whatever the step size.

Ground Station Pass Plan Entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting with version 0.2.0, :class:`~conops.ditl.queue_ditl.QueueDITL` automatically
creates ``GSP`` (Ground Station Pass) entries for passes that require spacecraft
tracking. These activities appear alongside science observations in the exported
plan. Booked passes that require no spacecraft tracking are not given entries. They are
stored in the plan's ``ground_contacts`` field.

Pass Deconfliction
^^^^^^^^^^^^^^^^^^

When multiple ground stations are visible simultaneously (overlapping pass windows),
COASTSim automatically selects the pass with the highest expected data volume:

.. code-block:: python

   # Selection metric: downlink_rate × pass_duration
   # Prefers: higher data volume > longer pass > earlier start time

Overlapping opportunities that are not selected are logged but not executed. This
prevents accidental "tail contacts", where a second pass starts during an ongoing pass.

Pass Metadata
^^^^^^^^^^^^^

Each GSP entry includes:

* ``station``: Ground station code (e.g., ``"SGS"``, ``"TRO"``)
* ``contact_begin``: Actual pass start time
* ``contact_end``: Actual pass end time
* ``slewtime``: Time spent slewing to point at the station

The GSP entry's ``begin`` time marks when the spacecraft reserves the pass window, which
may include slew preparation. The ``contact_begin`` time marks when data downlink actually
starts.

Example
^^^^^^^

.. code-block:: python

   from conops import QueueDITL

   ditl = QueueDITL(config=config, ephem=ephem, queue=queue)
   ditl.calc()

   # Find ground station passes in the plan
   for entry in ditl.plan:
       if entry.obstype == ObsType.GSP:
           downlink_time = entry.contact_end - entry.contact_begin
           rate = config.ground_stations.get(entry.station).get_overall_max_downlink()
           volume = rate * downlink_time / 1000.0  # Convert to Gb
           print(f"{entry.station}: {downlink_time}s at {rate} Mbps = {volume:.1f} Gb")

See :doc:`plan_serialization` for complete details on GSP entry format and
:doc:`communications` for pass selection behavior.

Integration with DITL
----------------------

The data management system is automatically integrated into DITL simulations.

Configuration
~~~~~~~~~~~~~

Add the recorder to your spacecraft configuration:

.. code-block:: python

   from conops.config import MissionConfig

   config = MissionConfig(
       name="Science Mission",
       spacecraft_bus=spacecraft_bus,
       solar_panel=solar_panel,
       payload=payload,
       battery=battery,
       constraint=constraint,
       ground_stations=ground_stations,
       recorder=recorder  # Add recorder here
   )

Telemetry Output
~~~~~~~~~~~~~~~~

During a simulation, DITL records data management telemetry in its structured telemetry
object:

.. code-block:: python

   from conops.ditl import QueueDITL
   from conops import MissionConfig

   config = MissionConfig()
   ditl = QueueDITL(config=config)
   ditl.calc()

   # Access data management telemetry through structured telemetry system
   telemetry = ditl.telemetry

   # Housekeeping data (recorded at each timestep)
   recorder_volume = telemetry.housekeeping.recorder_volume_gb      # Current volume at each timestep
   fill_fraction = telemetry.housekeeping.recorder_fill_fraction    # Fill level (0-1) at each timestep
   alert_level = telemetry.housekeeping.recorder_alert              # Alert level (0/1/2) at each timestep

   # Payload data (recorded when data is generated)
   data_events = telemetry.data  # List of PayloadData records
   data_sizes = [pd.data_size_gb for pd in data_events]

The same quantities are also available as arrays directly on the DITL object. This older
form of access is still supported:

.. code-block:: python

   # Legacy access (still supported)
   recorder_volume = ditl.recorder_volume_gb
   fill_fraction = ditl.recorder_fill_fraction
   alert_level = ditl.recorder_alert
   data_generated = ditl.data_generated_gb
   data_downlinked = ditl.data_downlinked_gb

Visualization Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from conops.visualization import plot_data_management_telemetry

   # Use the built-in plotting function
   fig, axes = plot_data_management_telemetry(ditl)
   plt.show()

   # Or create custom plots using telemetry data
   fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

   # Get timestamps for plotting
   times = telemetry.housekeeping.timestamp

   # Recorder volume
   axes[0].plot(times, telemetry.housekeeping.recorder_volume_gb, 'b-')
   axes[0].set_ylabel('Volume (Gb)')
   axes[0].set_title('Onboard Recorder Status')
   axes[0].grid(True)

   # Fill fraction with thresholds
   axes[1].plot(times, telemetry.housekeeping.recorder_fill_fraction, 'g-', label='Fill Level')
   axes[1].axhline(y=0.7, color='orange', linestyle='--', label='Yellow Threshold')
   axes[1].axhline(y=0.9, color='red', linestyle='--', label='Red Threshold')
   axes[1].set_ylabel('Fill Fraction')
   axes[1].legend()
   axes[1].grid(True)

   # Data generation events
   if telemetry.data:
       data_times = [pd.timestamp for pd in telemetry.data]
       data_sizes = [pd.data_size_gb for pd in telemetry.data]
       axes[2].bar(data_times, data_sizes, width=0.01, alpha=0.7, color='purple')
       axes[2].set_ylabel('Generated (Gb)')
       axes[2].set_title('Data Generation Events')
       axes[2].grid(True)

   # Alert levels
   colors = ['green', 'orange', 'red']  # 0=normal, 1=warning, 2=critical
   for i, alert in enumerate(telemetry.housekeeping.recorder_alert):
       if alert > 0:  # Only plot non-zero alerts
           axes[3].scatter(times[i], alert, c=colors[alert], s=20, alpha=0.8)
   axes[3].set_ylabel('Alert Level')
   axes[3].set_title('Recorder Alerts')
   axes[3].set_yticks([0, 1, 2])
   axes[3].set_yticklabels(['Normal', 'Warning', 'Critical'])
   axes[3].grid(True)

   plt.tight_layout()
   plt.show()

Fault Management Integration
-----------------------------

The recorder's fill fraction is automatically monitored by the fault management system
when configured.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from conops.config import FaultManagement

   fault_mgmt = FaultManagement()

   config = MissionConfig(
       # ... other config ...
       recorder=recorder,
       fault_management=fault_mgmt
   )

   # Initialize fault management with default thresholds
   config.init_fault_management_defaults()

This automatically adds a ``recorder_fill_fraction`` threshold using the recorder's
yellow and red threshold values. The fault management system will:

* Monitor the fill fraction at each timestep
* Track state transitions (GREEN → YELLOW → RED)
* Log threshold violations
* Optionally trigger safe mode on RED alerts (if configured)

Manual Threshold Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also manually configure recorder thresholds:

.. code-block:: python

   fault_mgmt.add_threshold(
       name="recorder_fill_fraction",
       yellow=0.75,      # Custom yellow threshold
       red=0.95,         # Custom red threshold
       direction="above"  # Alert when value goes above thresholds
   )

Use Cases and Examples
----------------------

Sizing the Onboard Recorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determine the minimum recorder capacity needed for your mission:

.. code-block:: python

   # Run simulation with different recorder sizes
   capacities = [32, 64, 128, 256]  # Gigabits

   for capacity in capacities:
       recorder = OnboardRecorder(capacity_gb=capacity)
       config.recorder = recorder

       ditl = DITL(config=config)
       ditl.calc()

       max_fill = max(ditl.recorder_fill_fraction)
       print(f"Capacity: {capacity} Gb, Max Fill: {max_fill:.1%}")

       if max_fill < 0.9:
           print(f"✓ {capacity} Gb is sufficient")
           break

Evaluating Downlink Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determine how many ground station passes are needed:

.. code-block:: python

   # Track cumulative data
   cumulative_generated = 0
   cumulative_downlinked = 0

   for i in range(len(ditl.utime)):
       cumulative_generated += ditl.data_generated_gb[i]
       cumulative_downlinked += ditl.data_downlinked_gb[i]

   print(f"Total data generated: {cumulative_generated:.1f} Gb")
   print(f"Total data downlinked: {cumulative_downlinked:.1f} Gb")
   print(f"Net accumulation: {cumulative_generated - cumulative_downlinked:.1f} Gb")

   # Calculate required passes
   passes = len([p for p in ditl.executed_passes.passes])
   print(f"Ground station passes: {passes}")

Optimizing Observation Schedules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Balance science observations with data downlink availability:

.. code-block:: python

   # Analyze periods when recorder is full
   full_periods = []
   for i, fill in enumerate(ditl.recorder_fill_fraction):
       if fill >= 1.0:
           full_periods.append(ditl.utime[i])

   if full_periods:
       print(f"WARNING: Recorder full for {len(full_periods)} timesteps")
       print("Consider:")
       print("  - Adding more ground station passes")
       print("  - Increasing downlink rate")
       print("  - Reducing instrument data rates")
       print("  - Increasing recorder capacity")

Best Practices
--------------

1. **Start with Conservative Estimates**

   Begin with generous recorder capacity and downlink capabilities, then optimize
   based on simulation results.

2. **Monitor Alert Levels**

   Pay attention to yellow and red alerts during simulation. Persistent red alerts
   indicate insufficient downlink capability or excessive data generation.

3. **Account for Growth**

   Size your recorder with margin for:

   * Mission lifetime degradation
   * Potential instrument mode additions
   * Missed passes due to weather or station outages

4. **Balance Generation and Downlink**

   The ideal scenario maintains recorder fill between 30-70%, providing buffer
   for both data accumulation and unexpected gaps in downlink opportunities.

5. **Validate with Different Scenarios**

   Test your configuration with:

   * Best case: All passes succeed
   * Worst case: 20-30% of passes fail
   * Realistic case: Mix of successful and failed passes

Troubleshooting
---------------

Recorder Frequently Full
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Red alerts, data loss, fill fraction at 100%

**Solutions**:

* Increase recorder capacity
* Add more ground station passes
* Increase antenna downlink rate
* Reduce instrument data generation rates
* Adjust observation scheduling to prioritize downlink windows

Recorder Always Empty
~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Fill fraction near 0%, unused capacity

**Solutions**:

* Reduce recorder capacity (weight/cost savings)
* Reduce number of ground station contacts
* Consider lower downlink rates
* Optimize pass scheduling

Insufficient Downlink
~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Recorder steadily filling, not enough passes to clear data

**Solutions**:

* Calculate required downlink rate: ``total_data / mission_duration``
* Add ground stations at different longitudes for more frequent passes
* Increase antenna data rate
* Consider data compression (simulate with reduced generation rates)

API Reference
-------------

For detailed API documentation, see:

* :class:`~conops.config.recorder.OnboardRecorder`
* :class:`~conops.config.instrument.DataGeneration`
* :class:`~conops.config.instrument.Instrument`
* :class:`~conops.config.instrument.Payload`
* :class:`~conops.config.groundstation.Antenna`
* :class:`~conops.ditl.ditl.DITL`
* :class:`~conops.ditl.queue_ditl.QueueDITL`

Independent computer storage
----------------------------

Some spacecraft keep data on two computers, a payload computer and a spacecraft computer,
each with its own storage. Use ``MissionConfig.data_storage`` to model this. Each named
store has its own recorder, its own data sources and its own share of the downlink.

Keep using ``recorder`` for existing missions with one shared recorder. A configuration
that supplies both ``recorder`` and ``data_storage`` is rejected.

Units and capacities
~~~~~~~~~~~~~~~~~~~~

All storage quantities ending in ``_gb`` are **decimal gigabits**, not gigabytes. For
example, 8 TB installed is 64,000 Gbit.

Installed hardware capacity and usable recorder capacity are separate quantities. A store's
``installed_capacity_gb`` is the first, and its recorder's ``capacity_gb`` is the second.

Unresolved inputs
~~~~~~~~~~~~~~~~~

Hardware details are often not known when a configuration is first written. An inventory
may therefore leave some inputs unresolved by setting them to ``null``::

    data_storage:
      stores:
        payload_computer:
          owner: payload
          installed_capacity_gb: 64000
          recorder: null
        spacecraft_computer:
          owner: spacecraft_bus
          recorder: null
      bus_housekeeping: spacecraft_computer
      instrument_science:
        Optical Imager: payload_computer
      instrument_housekeeping:
        Optical Imager: null
      downlinks: null

This configuration can be loaded and inspected. Planning and simulation raise
an error listing the missing inputs. A null recorder never means unlimited
storage, and it never takes the 32-Gbit default capacity of the single-recorder
model. Inspect ``config.data_storage.unresolved(config)`` before running a campaign.

Recorder views are cached during planning and simulation. Constructors revalidate
the configuration and rebuild those views for a new run. If you use
``config.recorder_views()`` directly after editing nested configuration, call
``config.recorder_views(refresh=True)`` first.

Defining each store
~~~~~~~~~~~~~~~~~~~

To resolve the storage inputs, do the following:

* For every store, replace ``recorder: null`` with an ``OnboardRecorder`` definition. It
  must contain an explicit, confirmed usable ``capacity_gb`` and an initial
  ``current_volume_gb``. A recorder defined in a store is rejected if either field is
  omitted.
* Keep the usable capacity within the installed capacity. A usable capacity that exceeds
  the installed capacity is rejected.
* Set ``planning_limit`` to the fraction of the capacity available for planning. For
  example, ``0.8`` reserves 20 percent. An explicit ``planning_limit: null`` disables that
  store's planning constraint, just as in the single-recorder model. Physical simulation
  still caps the store's storage.
* Route every instrument's science and housekeeping to a named store.

Target-level science-rate and fixed-per-observation overrides retain their existing
meaning (see `Target Science Data`_).

Downlink reservations
~~~~~~~~~~~~~~~~~~~~~

Configure direct downlink reservations only after confirming the hardware path.
For example, if a station and the spacecraft both support Ka::

    downlinks:
      - station: SVAL
        band: Ka
        shares:
          payload_computer: 0.95
          spacecraft_computer: 0.05

The following rules apply to downlink reservations:

* The physical link rate is the smaller of the station and spacecraft rates in that band.
* Shares are positive fractions totaling at most one. They divide the physical link rate
  between the stores. A store does not receive a whole link of its own.
* Reservations are fixed, so the bandwidth reserved for an empty store remains unused.
* A station has one active downlink in this model.
* ``downlinks: []`` explicitly means no downlink, whereas ``null`` means the path is
  unresolved.
* Stations absent from the list remove no data, though they may still provide uplink
  contacts.

Calibration and activity frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration and activity frame sizes are fixed amounts of data. The store that receives a
fixed amount is found from the instruments named for it, through ``instrument_science``:

* A calibration step without its own modes inherits the observation's instruments.
* The instruments must identify one store for each fixed amount. Use separate entries or
  budget steps when frames belong to different stores.
* An activity naming no instrument uses the bus store.

Scheduling
~~~~~~~~~~

The schedulers plan each store separately. Spare capacity and contact credit never
move between stores.

* Dispatch, priority, local-search and CP-SAT scheduling check every store's own planning
  limit.
* Rolling replanning carries a mapping of store names to projected volumes.
* The long-range allocator budgets each store's data within its own planning limit and its
  own reserved downlink, starting from the volume on each store. It reports each store's
  budget in ``Allocation.recorders[name]``.

Execution uses the same source routing and downlink reservations as planning.

Telemetry and fault thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-store telemetry is in ``Housekeeping.recorders[name]``. It includes ``volume_gb``,
``capacity_gb``, ``fill_fraction``, ``alert`` and cumulative ``generated_gb``,
``downlinked_gb`` and ``lost_gb``.

The scalar recorder telemetry of the single-recorder model is still produced. The scalar
volume is the total over all stores. The scalar fill fraction and alert are those of the
worst store.

Default fault thresholds use ``recorders.<name>.fill_fraction``. Data-management plots show
the independent stores and their limits.

Limits of the model
~~~~~~~~~~~~~~~~~~~

This is a volume model. It does not model inter-computer staging, file delivery
acknowledgments, retransmission queues or the age of individual products. If
science must first be copied into spacecraft storage, leave its downlink path
unresolved until transfer throughput, staging space and copy/delete semantics
are implemented. A low fill fraction alone does not establish delivery latency
or completeness. Science generation rates, file overheads and link efficiencies
must also be supplied for a meaningful mission budget.
