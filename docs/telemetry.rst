Telemetry System
================

COASTSim records the state of the spacecraft during a DITL simulation as telemetry. The telemetry
is stored in typed data models, so each value has a named field with a known type and unit. This page
describes those models and shows how to read, plot, and export the data they hold.

Overview
--------

The telemetry system has three parts:

1. A housekeeping record is a timestamped snapshot of the spacecraft state. The simulation writes
   one at each timestep.
2. A payload data record describes one data generation event during an observation.
3. The telemetry container holds the housekeeping records and the payload data records of one
   simulation.

Earlier versions of COASTSim stored telemetry as separate parallel arrays on the DITL object.
Those arrays still work (see `Migration from Legacy Arrays`_). The typed records give better type
safety, easier data access, and more structured analysis.

Housekeeping Data
-----------------

Housekeeping records capture the complete spacecraft state at each simulation timestep.
The example below builds one record by hand to show the most commonly used fields.

.. code-block:: python

   from datetime import datetime

   from conops import ACSMode
   from conops.ditl.telemetry import Housekeeping

   # Example housekeeping record
   hk = Housekeeping(
       timestamp=datetime(2025, 1, 1, 12, 0, 0),
       ra=45.0,                    # Right ascension in degrees
       dec=-23.5,                  # Declination in degrees
       roll=0.0,                   # Roll angle in degrees
       acs_mode=ACSMode.SCIENCE,   # ACS mode
       panel_illumination=0.85,    # Solar panel illumination (0-1)
       power_usage=125.0,          # Total power usage in W
       power_bus=45.0,             # Spacecraft bus power in W
       power_payload=80.0,         # Payload power in W
       battery_level=0.75,         # Battery state of charge (0-1)
       charge_state=1,             # Charging state (0=discharging, 1=charging)
       battery_alert=0,            # Battery alert level (0/1/2)
       obsid=1001,                 # Current observation ID
       recorder_volume_gb=25.3,    # Data volume in recorder (Gb)
       recorder_fill_fraction=0.2, # Recorder fill level (0-1)
       recorder_alert=0,           # Recorder alert level (0/1/2)
       sun_angle_deg=75.3,         # Angular distance to Sun in degrees
       earth_angle_deg=62.1,       # Angular distance to Earth in degrees
       moon_angle_deg=118.5,       # Angular distance to Moon in degrees
       in_constraint=None,         # Name of violated constraint, or None
   )

Time Alignment with the Ephemeris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a simulation executes a plan, it looks up the Sun and orbital state for each
record in the ephemeris by the record's UTC timestamp. The simulation does not
assume that its first step corresponds to the first ephemeris sample. The values
therefore stay correct in two cases where counting array positions from index
zero would give the wrong sample:

- the simulation starts partway through a longer ephemeris;
- the execution step is coarser than the ephemeris sampling step.

Stored-momentum integration uses the same time-aligned position and environmental
samples.

Science and Calibration Collection Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two fields record how much of each step was spent collecting, and for what purpose:

- ``collection_seconds`` is the collection duration that was executed within the
  step. It excludes setup and cleanup time.
- ``calibration`` is ``True`` when that collection belongs to the calibration step
  of a sequence, and ``False`` otherwise.

To measure science time, sum ``collection_seconds`` over the records where
``calibration=False``. Calibration collection is left out of that sum, but it
still consumes time, power and any configured calibration data. The benchmark
reports the two durations separately as ``science_hours`` and
``calibration_hours``.

Telemetry saved before the ``calibration`` flag existed loads with the flag set
to ``False``, so its calibration time would be counted as science. Regenerate
such sequence telemetry before using it for a science-only acceptance comparison.

Housekeeping Fields
~~~~~~~~~~~~~~~~~~~

**Attitude and Pointing**
   - ``timestamp``: UTC timestamp of the record
   - ``ra``: Right ascension in degrees
   - ``dec``: Declination in degrees
   - ``roll``: Roll angle in degrees. It is the right-handed physical rotation of the
     spacecraft about body ``+X``. At zero roll, body ``+Z`` aligns with projected
     celestial north. At zero RA/Dec, a roll of ``+90`` degrees places body ``+Y``
     along inertial ``+Z``.
   - ``roll_offset_deg``: Signed offset from the solar-optimal roll, in the range
     [-180, 180). It uses the same right-handed convention as ``roll``. The value is
     meaningful only during SCIENCE mode.
   - ``quat_w``: Attitude quaternion scalar component (w)
   - ``quat_x``: Attitude quaternion vector component (x)
   - ``quat_y``: Attitude quaternion vector component (y)
   - ``quat_z``: Attitude quaternion vector component (z)

The quaternion is an inertial-to-body coordinate transform, whereas ``roll`` describes
the physical rotation of the spacecraft. The two therefore have opposite signs for
the same physical attitude: at ``RA=0`` and ``Dec=0`` the roll component of the
quaternion applies the inverse rotation ``R_x(-roll)``.

**ACS State**
   - ``acs_mode``: Current ACS mode (SCIENCE, SLEWING, SAFE, SAA, etc.). The mode is DESAT while the spacecraft holds for a thruster momentum dump.
   - ``gravity_gradient_torque_body_n_m``: Gravity-gradient torque vector in spacecraft body coordinates (N m), or ``None`` when tracking is disabled
   - ``disturbance_torques_body_n_m``: Torque vector of each tracked torque source in body coordinates (N m), keyed by source name (``gravity_gradient``, ``solar_radiation_pressure``, ``residual_dipole``, ``magnetorquer_unloading``), or ``None`` when tracking is disabled
   - ``disturbance_torque_body_n_m``: Total torque vector of the tracked sources in body coordinates (N m), or ``None`` when tracking is disabled
   - ``geomagnetic_field_body_t``: Geomagnetic field in body coordinates (T). It is recorded when magnetorquers or a residual dipole are tracked, and is ``None`` otherwise
   - ``stored_momentum_body_n_m_s``: Accumulated stored-momentum vector in spacecraft body coordinates (N m s), or ``None`` when tracking is disabled
   - ``stored_momentum_norm_n_m_s``: Coordinate-invariant norm of accumulated stored momentum (N m s), or ``None`` when tracking is disabled
   - ``actuator_momentum_body_n_m_s``: Momentum held by the wheels, in body coordinates (N m s), or ``None`` when tracking is disabled. While the spacecraft turns, the body itself carries momentum :math:`I\omega`, and the wheels hold the stored momentum less that amount. When the spacecraft is at rest, the wheels hold all of the stored momentum
   - ``actuator_momentum_norm_n_m_s``: Norm of the momentum held by the wheels (N m s), or ``None`` when tracking is disabled

**Power System**
   - ``panel_illumination``: Solar panel illumination fraction (0-1)
   - ``solar_array_drive_angles``: Current finite angles of the single-axis solar
     array drives. Each entry is identified by the configured panel index and name
   - ``solar_array_drive_angles_deg``: Compatibility property that exposes only the
     angles, in the order the driven panels are configured
   - ``instrument_modes``: The mode of each instrument that defines modes at this
     timestep, keyed by instrument name. It is ``None`` when no instrument defines
     modes. The mode is resolved in the same way as the instrument's power draw.
     An instrument that the current observation uses is in the mode the observation
     gives it. Every other instrument is in ``IDLE`` while the ACS is in ``SCIENCE``
     mode. In any other ACS mode it is in its own mode of the same name as the ACS
     mode, such as ``SLEWING``, if it defines one, and in ``IDLE`` if it does not.
     The ``instrument_mode_hours()`` method on a DITL sums these values
     into hours per instrument and mode, and ``print_statistics()`` reports the
     totals
   - ``power_usage``: Total spacecraft power consumption in W
   - ``power_bus``: Spacecraft bus power consumption in W
   - ``power_payload``: Payload power consumption in W
   - ``battery_level``: Battery state of charge (0-1)
   - ``charge_state``: Battery charging state (0=discharging, 1=charging)
   - ``battery_alert``: Battery alert level (0=normal, 1=warning, 2=critical)

**Data Management**
   - ``obsid``: Current observation ID
   - ``recorder_volume_gb``: Current data volume in recorder (Gb)
   - ``recorder_fill_fraction``: Recorder fill fraction (0-1)
   - ``recorder_alert``: Recorder alert level (0=normal, 1=warning, 2=critical)

**Constraint Geometry**
   - ``sun_angle_deg``: Angular distance from current pointing to Sun in degrees
   - ``earth_angle_deg``: Angular distance from current pointing to Earth in degrees
   - ``moon_angle_deg``: Angular distance from current pointing to Moon in degrees
   - ``in_constraint``: Name of the constraint currently being violated (e.g. ``"Sun"``,
     ``"Earth Limb"``, ``"Moon"``, ``"Panel"``, ``"Anti-Sun"``, ``"ST Hard"``, ``"ST Soft"``),
     or ``None`` when no constraint is active.
   - ``for_solid_angle_sr``: Instantaneous field-of-regard solid angle in steradians.
     This value is optional and is ``None`` unless FOR calculation is enabled.
   - ``in_eclipse``: Whether spacecraft is in eclipse

**Star Tracker, Telescope, and Radiator State**
   - ``star_tracker_hard_violations``: Number of star trackers in hard keep-out zones
   - ``star_tracker_soft_violations``: Whether any star tracker is in a soft-degradation zone
   - ``star_tracker_functional_count``: Number of star trackers currently counted as functional
   - ``telescope_hard_violations``: 1 if the telescope hard constraint is violated, 0 if it is not
   - ``radiator_hard_violations``: Number of radiators violating hard keep-out constraints
   - ``radiator_sun_exposure``: Area-weighted Sun exposure for all configured radiators (0-1)
   - ``radiator_earth_exposure``: Area-weighted Earth exposure for all configured radiators (0-1)
   - ``radiator_heat_dissipation_w``: Total first-order net radiator heat flow (W). A positive value means the radiators reject heat. A negative value means they absorb a net external radiative load

The radiator metrics are continuous measures of performance. A low or negative value
does not indicate a hardware failure: the pointing alone can reduce heat rejection,
or even cause temporary net heat absorption. See :doc:`radiator_shadowing` for how
these values are computed.

Payload Data
------------

Payload data records capture data generation events during observations.

.. code-block:: python

   from conops.ditl.telemetry import PayloadData
   from datetime import datetime

   # Example payload data record
   pd = PayloadData(
       timestamp=datetime(2025, 1, 1, 12, 15, 0),
       data_size_gb=2.5  # Size of data generated in Gb
   )

Telemetry Container
-------------------

The :class:`~conops.ditl.telemetry.Telemetry` class provides a structured container for
both housekeeping and payload data records.

.. code-block:: python

   from conops.ditl.telemetry import Telemetry, HousekeepingList, Housekeeping, PayloadData

   # Create telemetry container
   telemetry = Telemetry(
       housekeeping=HousekeepingList([hk1, hk2, hk3]),  # List of housekeeping records
       data=[pd1, pd2]  # List of payload data records
   )

Accessing Telemetry Data
~~~~~~~~~~~~~~~~~~~~~~~~~

Each housekeeping field is available as an attribute of ``telemetry.housekeeping``,
which returns that field's values for all records as a list. Payload data records
are read by iterating over ``telemetry.data``:

.. code-block:: python

   # Access housekeeping data
   timestamps = telemetry.housekeeping.timestamp
   ra_values = telemetry.housekeeping.ra
   dec_values = telemetry.housekeeping.dec
   acs_modes = telemetry.housekeeping.acs_mode
   battery_levels = telemetry.housekeeping.battery_level

   # Angular distances and constraint geometry
   sun_angles = telemetry.housekeeping.sun_angle_deg
   earth_angles = telemetry.housekeeping.earth_angle_deg
   moon_angles = telemetry.housekeeping.moon_angle_deg
   constraints = telemetry.housekeeping.in_constraint  # list of str|None

   # Star tracker, telescope, and radiator hard keepout health
   hard_violations = telemetry.housekeeping.star_tracker_hard_violations
   functional_count = telemetry.housekeeping.star_tracker_functional_count
   telescope_violations = telemetry.housekeeping.telescope_hard_violations
   radiator_violations = telemetry.housekeeping.radiator_hard_violations

   # Access payload data
   data_timestamps = [pd.timestamp for pd in telemetry.data]
   data_sizes = [pd.data_size_gb for pd in telemetry.data]

Field Extraction Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~conops.ditl.telemetry.Housekeeping` class provides utility methods for
extracting fields from multiple records:

.. code-block:: python

   from conops.ditl.telemetry import Housekeeping

   # Extract single field from multiple records
   timestamps = Housekeeping.extract_field(housekeeping_records, 'timestamp')
   battery_levels = Housekeeping.extract_field(housekeeping_records, 'battery_level')

   # Extract multiple fields at once
   fields = Housekeeping.extract_fields(housekeeping_records, ['ra', 'dec', 'acs_mode'])

Integration with DITL
----------------------

During DITL simulation, telemetry data is automatically recorded and stored in the
``ditl.telemetry`` attribute.

.. code-block:: python

   from datetime import datetime

   from rust_ephem import TLEEphemeris

   from conops.ditl import QueueDITL
   from conops import MissionConfig

   config = MissionConfig()
   ephem = TLEEphemeris(
       tle="examples/example.tle",
       begin=datetime(2025, 1, 1),
       end=datetime(2025, 1, 2),
       step_size=60,
   )
   # FOR calculation is disabled by default.
   # Enable it only when this telemetry is needed.
   ditl = QueueDITL(config=config, ephem=ephem, calculate_field_of_regard=True)
   ditl.calc()  # Run simulation

   # Access telemetry data
   telemetry = ditl.telemetry
   housekeeping = telemetry.housekeeping
   payload_data = telemetry.data

   # Analyze results
   print(f"Simulation covered {len(housekeeping)} timesteps")
   print(f"Generated {len(payload_data)} data records")
   print(f"Average battery level: {sum(housekeeping.battery_level) / len(housekeeping):.2f}")

Visualization with Telemetry
-----------------------------

The COASTSim visualization functions all read from the structured telemetry.
Pass them the DITL object after the simulation has run:

.. code-block:: python

   from conops.visualization import plot_ditl_telemetry, plot_data_management_telemetry

   # Plot basic telemetry timeline
   fig, axes = plot_ditl_telemetry(ditl)

   # Plot data management telemetry
   fig2, axes2 = plot_data_management_telemetry(ditl)

Exporting Telemetry Data
-------------------------

To analyze telemetry in another tool, build a table from the fields you need and
write it to a file. The example below uses pandas to write CSV files:

.. code-block:: python

   import pandas as pd

   # Convert housekeeping to DataFrame
   hk_df = pd.DataFrame({
       'timestamp': housekeeping.timestamp,
       'ra': housekeeping.ra,
       'dec': housekeeping.dec,
       'acs_mode': housekeeping.acs_mode,
       'battery_level': housekeeping.battery_level,
       'recorder_fill_fraction': housekeeping.recorder_fill_fraction
   })

   # Convert payload data to DataFrame
   pd_df = pd.DataFrame({
       'timestamp': [pd.timestamp for pd in payload_data],
       'data_size_gb': [pd.data_size_gb for pd in payload_data]
   })

   # Export to CSV
   hk_df.to_csv('housekeeping.csv', index=False)
   pd_df.to_csv('payload_data.csv', index=False)

Migration from Legacy Arrays
-----------------------------

Existing code may read telemetry from parallel arrays on the DITL object (``ditl.ra``, ``ditl.dec``, etc.).
Each array has an equivalent field on ``ditl.telemetry.housekeeping``:

.. code-block:: python

   # Legacy access (still works)
   ra_values = ditl.ra
   dec_values = ditl.dec
   battery_levels = ditl.batterylevel

   # New structured access
   ra_values = ditl.telemetry.housekeeping.ra
   dec_values = ditl.telemetry.housekeeping.dec
   battery_levels = ditl.telemetry.housekeeping.battery_level

The legacy arrays are maintained for backward compatibility. New code should use the
telemetry fields, which provide better type safety and more structured data access.
