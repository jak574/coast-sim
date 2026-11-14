Plan Serialisation
==================

COASTSim can save an observation plan produced by a DITL run to a portable JSON file and
reload it later. :class:`~conops.targets.Plan` and :class:`~conops.targets.PlanEntry`
are directly serializable `Pydantic v2`_ models. A completed DITL run also attaches
executed attitude and orbit-state timelines, which :meth:`~conops.targets.Plan.save`
writes as sibling JSON files.

Overview
--------

The primary models are:

* :class:`~conops.targets.PlanEntry` is one spacecraft activity, such as an observation or a
  period of ground-station tracking.
* :class:`~conops.targets.Plan` is the top-level plan. It holds the entries, the file
  metadata, and the optional execution timeseries.

These two models can also be imported under their older names,
:class:`~conops.targets.plan_schema.PlanSchema` and
:class:`~conops.targets.plan_schema.PlanEntrySchema`, so that existing code keeps working.
``PlanSchema.from_plan(plan)`` is still available for callers that need it. New code should
use ``Plan`` and ``PlanEntry`` directly.

Quick Start
-----------

Save a plan after a DITL run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest approach is to call :meth:`~conops.targets.Plan.save` directly on the plan:

.. code-block:: python

   # `ditl` is a QueueDITL (or similar) instance that has already been run
   saved_path = ditl.plan.save("plan_20251201.json")
   print(f"Saved to {saved_path}")

The file metadata is available on the ``Plan`` before it is written:

.. code-block:: python

  print(f"Saving {ditl.plan.num_entries} entries (version {ditl.plan.version})")
  ditl.plan.save("plan_20251201.json")

Load it back
~~~~~~~~~~~~

:meth:`~conops.targets.Plan.load` is a class method on :class:`~conops.targets.Plan`. It
returns a :class:`~conops.targets.Plan` with the metadata and entries that were saved:

.. code-block:: python

  from conops.targets import Plan

  plan = Plan.load("plan_20251201.json")
  print(plan.version)          # plan-file revision (integer)
  print(plan.num_entries)      # number of plan entries
  print(plan.entries[0].name)  # first target name

The JSON file does not contain the mission configuration or the ephemeris. A plan only
refers to them while a simulation is running, so they are left out of the file on purpose.

If existing code needs the loaded object to be a ``PlanSchema``, the compatibility class can
load the same file:

.. code-block:: python

   from conops.targets import PlanSchema

   legacy_schema = PlanSchema.load("plan_20251201.json")

Execute a loaded plan
~~~~~~~~~~~~~~~~~~~~~

:class:`~conops.ditl.ditl.DITL` executes a plan. It gives the loaded plan its configuration
and ephemeris again when the plan is passed to the constructor, or when it is assigned
before :meth:`~conops.ditl.ditl.DITL.calc` is called:

.. code-block:: python

  ditl = DITL(config=config, ephem=ephem, plan=plan)
  ditl.calc()

:class:`~conops.ditl.ditl.DITL` commands each entry when the simulation reaches the entry's
``begin`` time. What the spacecraft then does depends on the entry's type:

* For a science entry (``PPT``, ``AT`` or ``TOO``), the spacecraft slews to the target and
  collects until ``end``.
* For a ground-station tracking entry (``GSP``), the spacecraft slews onto the pass tracking
  profile. The contact starts at ``contact_begin`` and ends at ``end``.
* For a charging entry (``CHARGE``), the spacecraft charges until ``end``.
* For a scheduled activity entry (``ACTIVITY``), the spacecraft slews to the entry's
  attitude if it is elsewhere and holds there until ``end``.
* An entry of any other type is logged and skipped.

``DITL`` predicts the passes from the configured ground stations and matches them to the
``GSP`` entries by station and contact window. A predicted pass that is not in the plan is
skipped.

Some booked pass windows need no spacecraft tracking. These are stored in the plan's
``ground_contacts`` field instead of as entries. ``DITL`` uses them for downlink without
adding activities to the spacecraft timeline.

Executing a plan saved by :class:`~conops.ditl.queue_ditl.QueueDITL` with the same
configuration and ephemeris reproduces that run's timeline.  To list where execution differed
from the plan:

.. code-block:: python

  for mismatch in ditl.validate_plan_matches_execution():
      print(mismatch)

``DITL`` returns the mismatches without raising an error. This differs from
:class:`~conops.ditl.queue_ditl.QueueDITL`, which raises when its own plan does not match
its execution.

Round-trip via model_validate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Plan`` can also be validated from another plan object, because the model reads the
fields it needs from the object's attributes:

.. code-block:: python

  plan = Plan.model_validate(ditl.plan, from_attributes=True)

JSON File Format
----------------

The JSON file holds the plan's metadata fields first, followed by the list of entries.

.. code-block:: json

   {
     "version": 3,
     "coast_sim_version": "0.9.0",
     "created_at": "2025-12-01T00:00:00+00:00",
     "start": "2025-12-01T00:00:00+00:00",
     "end": "2025-12-01T23:59:00+00:00",
     "num_entries": 42,
     "attitude_timeseries_file": "plan_20251201_attitude_timeseries.json",
     "orbit_state_timeseries_file": "plan_20251201_orbit_state_timeseries.json",
     "roll_axis": "+X",
     "roll_convention": "right_handed_body_rotation",
     "roll_reference_axis": "+Z",
     "roll_reference": "projected_celestial_north",
     "metadata": {
       "ephemeris": {
         "source": "TLE",
         "norad_id": 25544
       }
     },
     "entries": [
       {
         "name": "TEST_001",
         "ra": 83.82,
         "dec": -5.39,
         "roll": 0.0,
         "begin": "2025-12-01T00:00:00+00:00",
         "end": "2025-12-01T00:16:40+00:00",
         "merit": 95.0,
         "slewtime": 120,
         "insaa": 0,
         "obsid": 1001,
         "obstype": "AT",
         "slewdist": 10.3,
         "ss_min": 300.0,
         "ss_max": 1000000.0,
         "exptime": 880,
         "exporig": 1000,
         "isat": false,
         "done": true,
         "exposure": 880
       },
       {
         "name": "SGS_PASS",
         "ra": 120.0,
         "dec": 45.0,
         "roll": 0.0,
         "begin": "2025-12-01T00:18:00+00:00",
         "end": "2025-12-01T00:28:00+00:00",
         "merit": 101.0,
         "slewtime": 120,
         "insaa": 0,
         "obsid": 65535,
         "obstype": "GSP",
         "slewdist": 5.2,
         "ss_min": 0.0,
         "ss_max": 480.0,
         "exptime": 480,
         "exporig": 600,
         "isat": false,
         "done": true,
         "exposure": 480,
         "station": "SGS",
         "contact_begin": "2025-12-01T00:20:00+00:00",
         "contact_end": "2025-12-01T00:28:00+00:00",
         "track_start_ra": 120.0,
         "track_start_dec": 45.0,
         "track_end_ra": 231.67,
         "track_end_dec": -0.38
       }
     ]
   }

Metadata Fields
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``version``
     - int
     - Integer plan-file revision counter.  Starts at 0 and is incremented
       automatically each time a new plan is saved to the same directory for
       the same time window (see :ref:`auto-versioning` below).
   * - ``coast_sim_version``
     - string
     - COASTSim package version that produced the file (e.g. ``"0.9.0"``).
   * - ``created_at``
     - string
     - ISO-8601 UTC timestamp of when the :class:`~conops.targets.Plan` instance was created
       (not updated by :meth:`~conops.targets.Plan.save`).
   * - ``start``
     - string
     - ISO-8601 UTC timestamp of the first entry's ``begin`` time
       (``"1970-01-01T00:00:00+00:00"`` if the plan is empty).
   * - ``end``
     - string
     - ISO-8601 UTC timestamp of the last entry's ``end`` time
       (``"1970-01-01T00:00:00+00:00"`` if the plan is empty).
   * - ``num_entries``
     - int
     - Total number of entries in the file.
   * - ``ground_contacts``
     - array | null
     - Booked passes that require no spacecraft tracking. Each row is
       ``[begin_unix_seconds, end_unix_seconds, station_code]``. An empty
       array reserves no station time. If the field is absent, the bookings are
       predicted from the configuration. Bookings saved in the file are replayed
       as they are, without a new scheduling draw.
   * - ``attitude_timeseries_file``
     - string | null
     - Filename (no path) of the sibling attitude-timeseries JSON file written alongside
       this plan, or ``null`` / absent when no timeseries was exported.
       See :ref:`attitude-timeseries` for the file format.
   * - ``orbit_state_timeseries_file``
     - string | null
     - Filename (no path) of the sibling GCRS orbit-state JSON file written alongside this
       plan, or ``null`` / absent when no orbit state was exported.
       See :ref:`orbit-state-timeseries` for the file format.
   * - ``roll_axis`` / ``roll_convention``
     - string
     - By default, plan roll is a right-handed physical spacecraft rotation about
       body ``+X``. An entry that points a mounted science instrument overrides this
       default in its own ``target_attitude.pointing``.
   * - ``roll_reference_axis`` / ``roll_reference``
     - string
     - At zero roll, body ``+Z`` aligns with celestial north projected into the
       plane normal to the boresight. At a celestial pole, where that projection
       is undefined, COAST uses the analytical continuation of its RA/Dec Euler
       sequence.
   * - ``metadata``
     - object | null
     - Optional record of how the plan was produced. COASTSim reserves the ``ephemeris``
       object for typed ephemeris details (see `Plan Provenance`_). Callers may add their
       own JSON-compatible metadata keys.

Initialization State
~~~~~~~~~~~~~~~~~~~~

``metadata.ephemeris.initialization_state`` provides a single Cartesian state
for initializing an external simulation before plan execution. The block
contains:

* an explicit UTC epoch;
* the frame (``GCRS``) and the origin (Earth center);
* the position in meters and the velocity in meters per second;
* the propagator, gravity model, and SGP4 operation mode reported by the source
  ``TLEEphemeris``.

The last group of fields records where the state came from, and callers cannot
override them. Producers should generate a dedicated ephemeris sample at the
requested initialization epoch instead of extrapolating plan-start orbital
elements.

Entry Fields
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``name``
     - string
     - Human-readable target name (e.g. ``"Crab Nebula"``).
   * - ``ra``
     - float
     - Science-target right ascension in degrees (J2000). For a mounted telescope,
       this is the direction of the instrument ``+X`` line of sight, which can differ
       from spacecraft body ``+X``.
   * - ``dec``
     - float
     - Declination in degrees (J2000).
   * - ``roll``
     - float
     - Right-handed roll in degrees. A value of ``-1`` means the roll is unset. By
       default the roll is about spacecraft body ``+X``. When ``instrument_name``
       identifies a mounted telescope, the roll is about that instrument's ``+X``
       science boresight.
   * - ``instrument_name``
     - string | null
     - Telescope whose frame defines ``ra``, ``dec``, and ``roll``. Omitted for
       targets that use the default body-``+X`` pointing.
   * - ``spacecraft_attitude``
     - array[float, float, float] | null
     - RA, Dec, and roll of the physical spacecraft body ``+X`` axis, derived from the
       pointing of a mounted science instrument. Omitted when it is identical to
       ``ra``/``dec``/``roll``.
   * - ``begin``
     - string
     - Start of the observation window (ISO-8601 UTC).
   * - ``end``
     - string
     - End of the observation window (ISO-8601 UTC).
   * - ``merit``
     - float
     - Scheduler merit/priority figure of merit.
   * - ``slewtime``
     - int
     - Slew duration in seconds.
   * - ``insaa``
     - int
     - Time spent in the South Atlantic Anomaly during the window (seconds).
   * - ``obsid``
     - int
     - Numeric observation identifier.
   * - ``obstype``
     - string
     - Observation type. Valid values: ``"AT"`` (Astronomical Target), ``"PPT"`` (Preprogrammed Target),
       ``"TOO"`` (Target of Opportunity), ``"SAFE"`` (Safe mode pointing), ``"CHARGE"`` (Emergency charging),
       ``"GSP"`` (Ground Station Pass), ``"IDLE"`` (Idle hold), ``"ACTIVITY"`` (Scheduled
       non-science activity).
   * - ``slewdist``
     - float
     - Angular slew distance in degrees.
   * - ``great_circle_deg``
     - float
     - Great-circle angle between the pointings the slew joins, in degrees.
   * - ``sun_vector_change_deg``
     - float
     - How far the slew moves the Sun's direction in the body frame, in degrees. The value
       is zero when no thermal settling table is configured.
   * - ``settle_seconds``
     - float
     - Settling time that is included in ``slewtime``, in seconds.
   * - ``ss_min``
     - float
     - Minimum length of one snapshot of the target, in seconds. A snapshot is one
       unbroken period of collection. For ``GSP`` entries the value is ``0``.
   * - ``ss_max``
     - float
     - Maximum length of one snapshot of the target, in seconds. For ``GSP`` entries
       the value is the contact duration.
   * - ``exptime``
     - float | null
     - Exposure time in seconds (may be shorter than original if interrupted).
   * - ``exporig``
     - float | null
     - Originally requested exposure time in seconds.
   * - ``isat``
     - bool
     - Flag that the ACS sets to ``true`` on a target it builds for a slew whose type is
       not ``PPT``. It is ``false`` otherwise. Only pointing entries have this field.
   * - ``done``
     - bool
     - ``true`` if the observation completed successfully.
   * - ``exposure``
     - int
     - Net science exposure time in seconds. For ``AT``/``PPT``/``TOO`` entries it is
       ``collection_end - collection_begin - insaa`` when the entry has collection
       bounds, and ``end - begin - slewtime - insaa`` when it does not. The collection
       bounds ``collection_begin`` and ``collection_end`` are optional ISO-8601 UTC
       timestamps. They are kept separately because the entry's ``end`` also includes
       cleanup and handoff time. For ``GSP`` entries, ``exposure`` is the actual
       downlink contact duration, computed as ``contact_end − max(contact_begin, begin)``.
   * - ``station``
     - string | null
     - Ground station code for ``GSP`` (Ground Station Pass) entries. Only present for
       ``obstype="GSP"``. Identifies which station is used for the commanded pass.
   * - ``contact_begin``
     - string | null
     - ISO-8601 UTC timestamp of when the ground station contact window begins. Only present
       for ``obstype="GSP"``. This is the actual pass start time. The entry's ``begin`` is
       earlier when the reservation includes slew preparation.
   * - ``contact_end``
     - string | null
     - ISO-8601 UTC timestamp of when the ground station contact window ends. Only present
       for ``obstype="GSP"``. Typically matches ``end``.
   * - ``track_start_ra``
     - float | null
     - Ground-station tracking right ascension at ``contact_begin`` in degrees. Only present
       for ``obstype="GSP"``. For GSP entries, ``ra`` uses the same pass-start convention.
   * - ``track_start_dec``
     - float | null
     - Ground-station tracking declination at ``contact_begin`` in degrees. Only present
       for ``obstype="GSP"``. For GSP entries, ``dec`` uses the same pass-start convention.
   * - ``track_start_roll``
     - float | null
     - Ground-station tracking roll at ``contact_begin`` in degrees. Only present
       for ``obstype="GSP"``. For GSP entries, ``roll`` uses the same pass-start convention.
   * - ``track_end_ra``
     - float | null
     - Ground-station tracking right ascension used by ACS at pass end, in degrees. This is
       derived from the final tracking sample and matches ``Pass.attitude_at(contact_end)``. Only
       present for ``obstype="GSP"``. Use this with ``track_end_dec`` when inspecting slews
       after a pass.
   * - ``track_end_dec``
     - float | null
     - Ground-station tracking declination used by ACS at pass end, in degrees. This is
       derived from the final tracking sample and matches ``Pass.attitude_at(contact_end)``. Only
       present for ``obstype="GSP"``. Use this with ``track_end_ra`` when inspecting slews
       after a pass.
   * - ``track_end_roll``
     - float | null
     - Ground-station tracking roll used by ACS at pass end, in degrees. This is derived
       from the final tracking sample and matches ``Pass.attitude_at(contact_end)``.
   * - ``station_lat_deg`` / ``station_lon_deg`` / ``station_alt_m``
     - float | null
     - Ground-station geodetic latitude and longitude in degrees, and altitude in metres.
       Present for exported ``GSP`` entries when the station definition provides them.
   * - ``target_attitude``
     - object | null
     - Commanded fixed target attitude for ``AT``, ``PPT``, and ``TOO`` entries. It is
       ``null`` for entries whose attitude is tracked dynamically, such as ``GSP``.

       The object contains a GCRS-to-COAST-body quaternion in ``wxyz`` order and the
       RA/Dec/roll inputs used to generate it.

       The rotation states its convention explicitly with
       ``quaternion_product: "hamilton"`` and ``vector_action: "q_v_q_conjugate"``. This
       means an inertial vector ``v`` is transformed into body coordinates as
       ``q * v * conjugate(q)``. Consumers using ``conjugate(q) * v * q`` must conjugate
       the exported quaternion.

       The pointing metadata records ``roll_convention: "right_handed_body_rotation"``.
       For mounted instruments it also records ``instrument_name``, ``boresight_body``,
       and ``roll_reference_body``.

Ground Station Pass (GSP) Entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ground station pass entry (``obstype="GSP"``) is a period in which the spacecraft
tracks a ground station. Where a science entry records an observation, a GSP entry records
the reservation and execution of a data downlink pass.

How GSP entries are created
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~conops.ditl.queue_ditl.QueueDITL` creates a GSP entry when the spacecraft enters
a ground station visibility window. It creates no GSP entries while the spacecraft is in
SAFE mode.

When several ground stations are visible at the same time, COASTSim selects the pass with
the highest expected data volume (downlink rate × duration). The overlapping opportunities
that it drops are logged but are not exported to the plan.

Times
^^^^^

The ``begin`` time marks when the spacecraft reserves the window, which may include slew
preparation. The ``contact_begin`` time marks the actual start of the ground station
contact.

The ``station``, ``contact_begin``, and ``contact_end`` fields are only present for GSP
entries. They are ``null`` or omitted for other observation types.

Attitude
^^^^^^^^

A GSP entry tracks the ground station through the contact, so its attitude changes during
the pass. The generic ``ra``, ``dec``, and ``roll`` fields give the attitude at the start
of the pass and match ``track_start_ra``, ``track_start_dec``, and ``track_start_roll``.
Use ``track_end_ra``, ``track_end_dec``, and ``track_end_roll`` to inspect the spacecraft
attitude at the end of the pass. The end fields are derived from the final tracking sample,
which matches what the ACS ``Pass.attitude_at(contact_end)`` returns.

Visualization
^^^^^^^^^^^^^

GSP entries are excluded from the science observation bands in
:func:`~conops.visualization.mpl.ditl_timeline.plot_ditl_timeline` and
:func:`~conops.visualization.plotly.ditl_timeline.plot_ditl_timeline_plotly`.
Ground station passes are still shown in the timeline's dedicated "Ground Contact" row.

Example GSP entry
^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "name": "TRO_PASS",
     "obstype": "GSP",
     "station": "TRO",
     "begin": "2025-12-01T12:00:00+00:00",
     "contact_begin": "2025-12-01T12:02:00+00:00",
     "contact_end": "2025-12-01T12:12:00+00:00",
     "end": "2025-12-01T12:12:00+00:00",
     "track_start_ra": 120.0,
     "track_start_dec": 45.0,
     "track_start_roll": 12.4,
     "track_end_ra": 231.67,
     "track_end_dec": -0.38,
     "track_end_roll": 18.2,
     "slewtime": 120,
     "exptime": 600,
     "obsid": 65535
   }

In this example, the spacecraft begins reserving the pass window at 12:00:00 (``begin``),
uses 2 minutes for slew preparation (``slewtime``), and the actual ground station contact
runs from 12:02:00 to 12:12:00 (``contact_begin`` to ``contact_end``).

Plan Provenance
~~~~~~~~~~~~~~~

A plan can record which orbit data it was built from, so that someone reading the file
later can reproduce it. This record is kept under the plan's ``metadata.ephemeris`` object,
and two helper functions write to it. Both helpers preserve metadata keys that are already
present.

Use :func:`~conops.targets.attach_tle_plan_metadata` to store the TLE that was used to
create a plan. The helper records the source, TLE filename and lines, epoch, NORAD ID, and
TLE mean elements:

.. code-block:: python

   from conops.targets import attach_tle_plan_metadata

   # tle_record is the rust_ephem.TLERecord used to build the ephemeris.
   attach_tle_plan_metadata(
       ditl.plan,
       tle_record,
       tle_file="examples/example.tle",
   )
   ditl.plan.save("plan_20251201.json")

Use :func:`~conops.targets.attach_osculating_elements_metadata` separately to record
instantaneous osculating elements derived from the propagated GCRS position and velocity.
The requested epoch must match exactly one ephemeris timestamp.
:class:`~conops.ditl.ditl.DITL` and :class:`~conops.ditl.queue_ditl.QueueDITL` call
``attach_osculating_elements_metadata`` automatically, at the simulation start epoch, when
they attach the orbit-state timeseries.

The resulting ``metadata.ephemeris`` object keeps the two element sets distinct:

* ``tle_mean_elements`` is associated with the TLE epoch.
* ``osculating_elements`` includes the selected state epoch, ``frame: "GCRS"``,
  ``origin: "Earth center"``, and the gravitational parameter used for the conversion.

Element angles are normalized to ``[0, 360)`` degrees.

The conversion from the Cartesian state to elements is done by rust-ephem, which also
selects the central-body gravitational parameter. Its current default is the WGS-72 Earth
value used by SGP4. COASTSim does not select or implement a gravity model. It serializes
the exact value returned by rust-ephem as ``GravitationalParameter_m3_s2`` so consumers can
reproduce the conversion.

When ``DITL`` and ``QueueDITL`` attach the orbit-state timeseries, they also record
``metadata.ephemeris.earth_orientation``. This describes the exact JPL EOP2 text loaded by
rust-ephem's UT1 and polar-motion providers. Each provider reports its availability, source
URL, SHA-256, whether the text came from a download or from the cache, and whether a stale
cache was used as a fallback. A provider that is unused or unavailable is still listed, with
``available: false``.

For validation and programmatic construction, use
:class:`~conops.targets.PlanMetadata` and :class:`~conops.targets.EphemerisMetadata`.
``Plan.metadata`` itself remains a JSON-compatible dictionary so that mission tools can add
their own provenance keys alongside the typed ``ephemeris`` object.

.. _auto-versioning:

Auto-versioning
~~~~~~~~~~~~~~~

:meth:`~conops.targets.Plan.save` chooses the version number itself when its ``path``
argument is a directory or ends with ``/``. It scans the directory for existing files
matching ``plan_<start>_<end>_v<N>.json`` and sets ``version`` to ``max(N) + 1``, or to
``0`` if no matching files exist.  Saving to an explicit file path leaves ``version``
unchanged.

.. _attitude-timeseries:

Attitude Timeseries
~~~~~~~~~~~~~~~~~~~

Both :class:`~conops.ditl.ditl.DITL` and :class:`~conops.ditl.queue_ditl.QueueDITL`
automatically attach the executed spacecraft attitude timeline to the plan at the end of
``calc()``/``run()``.  When the plan is then saved, this timeseries is written to a sibling
JSON file alongside the main plan file.

The sibling file is placed in the same directory as the plan and named
``<plan_stem>_attitude_timeseries.json``.  For example, if the plan is saved as
``plan_20251201T000000Z_20251201T235900Z_v3.json`` the attitude file is
``plan_20251201T000000Z_20251201T235900Z_v3_attitude_timeseries.json``.

The plan file records the sibling filename in the ``attitude_timeseries_file`` metadata
field so that consumers can locate it without scanning the directory.

Attitude Timeseries File Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "version": 0,
     "coast_sim_version": "0.9.0",
     "created_at": "2025-12-01T00:00:00+00:00",
     "plan_file": "plan_20251201T000000Z_20251201T235900Z_v3.json",
     "plan_version": 3,
     "plan_start": "2025-12-01T00:00:00+00:00",
     "plan_end": "2025-12-01T23:59:00+00:00",
     "frame": "GCRS",
     "body_frame": "COAST_BODY",
     "roll_axis": "+X",
     "roll_convention": "right_handed_body_rotation",
     "roll_reference_axis": "+Z",
     "roll_reference": "projected_celestial_north",
     "representation": "quaternion",
     "direction": "inertial_to_body",
     "order": "wxyz",
     "quaternion_product": "hamilton",
     "vector_action": "q_v_q_conjugate",
     "num_samples": 2,
     "samples": [
       {
         "utime": 1748736000.0,
         "timestamp": "2025-12-01T00:00:00+00:00",
         "ra": 83.82,
         "dec": -5.39,
         "roll": 0.0,
         "mode": "SCIENCE",
         "obsid": 1001,
         "quat_w": 0.9998,
         "quat_x": 0.0050,
         "quat_y": -0.0120,
         "quat_z": 0.0150
       },
       {
         "utime": 1748736060.0,
         "timestamp": "2025-12-01T00:01:00+00:00",
         "ra": 83.82,
         "dec": -5.39,
         "roll": 0.0,
         "mode": "SCIENCE",
         "obsid": 1001,
         "quat_w": 0.9998,
         "quat_x": 0.0051,
         "quat_y": -0.0121,
         "quat_z": 0.0151
       }
     ]
   }

Attitude Timeseries Metadata Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``version``
     - int
     - Schema format version (currently always ``0``).
   * - ``coast_sim_version``
     - string
     - COASTSim package version that produced the file.
   * - ``created_at``
     - string
     - ISO-8601 UTC timestamp of when the file was created.
   * - ``plan_file``
     - string | null
     - Filename of the associated plan JSON file.
   * - ``plan_version``
     - int | null
     - ``version`` value of the associated plan.
   * - ``plan_start``
     - string | null
     - ISO-8601 UTC timestamp matching the plan's ``start`` field.
   * - ``plan_end``
     - string | null
     - ISO-8601 UTC timestamp matching the plan's ``end`` field.
   * - ``frame`` / ``body_frame``
     - string
     - Source and destination frames for every sample quaternion: ``GCRS`` and
       ``COAST_BODY``.
   * - ``roll_axis`` / ``roll_convention``
     - string
     - Roll is a right-handed physical body rotation about ``+X``. At zero
       RA/Dec, ``+90`` degrees places body ``+Y`` along inertial ``+Z``.
   * - ``roll_reference_axis`` / ``roll_reference``
     - string
     - At zero roll, body ``+Z`` aligns with projected celestial north. The
       celestial-pole fallback follows the same RA/Dec Euler sequence as the
       exported quaternion.
   * - ``representation`` / ``direction`` / ``order``
     - string
     - Every sample uses an ``inertial_to_body`` quaternion in ``wxyz`` component order.
   * - ``quaternion_product`` / ``vector_action``
     - string
     - Every sample uses the Hamilton product and transforms inertial vectors into body
       coordinates as ``q * v * conjugate(q)``.
   * - ``num_samples``
     - int
     - Number of attitude samples in ``samples``.

Attitude Sample Fields
^^^^^^^^^^^^^^^^^^^^^^

Each element of ``samples`` is an :class:`~conops.targets.plan_schema.AttitudeSampleSchema`:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Field
     - Type
     - Description
   * - ``utime``
     - float
     - Unix timestamp (seconds since epoch, UTC).
   * - ``timestamp``
     - string
     - The same time as ``utime``, written as an ISO-8601 UTC timestamp.
   * - ``ra``
     - float | null
     - Right ascension of the spacecraft boresight in degrees (J2000).
   * - ``dec``
     - float | null
     - Declination of the spacecraft boresight in degrees (J2000).
   * - ``roll``
     - float | null
     - Right-handed physical spacecraft rotation about body ``+X``, in degrees.
   * - ``mode``
     - string | null
     - ACS mode name at this sample (e.g. ``"SCIENCE"``, ``"SLEWING"``, ``"PASS"``).
   * - ``obsid``
     - int | null
     - Observation ID active at this sample.
   * - ``quat_w``
     - float | null
     - Spacecraft attitude quaternion W component.
   * - ``quat_x``
     - float | null
     - Spacecraft attitude quaternion X component.
   * - ``quat_y``
     - float | null
     - Spacecraft attitude quaternion Y component.
   * - ``quat_z``
     - float | null
     - Spacecraft attitude quaternion Z component.

Loading the Attitude Timeseries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.targets import AttitudeTimeseriesSchema, Plan
   from pathlib import Path

   plan = Plan.load("plan_20251201T000000Z_20251201T235900Z_v3.json")

   if plan.attitude_timeseries_file:
       plan_dir = Path("plan_20251201T000000Z_20251201T235900Z_v3.json").parent
       timeseries_path = plan_dir / plan.attitude_timeseries_file
       raw = __import__("json").loads(timeseries_path.read_text())
       timeseries = AttitudeTimeseriesSchema.model_validate(raw)
       print(f"Loaded {timeseries.num_samples} attitude samples")

.. _orbit-state-timeseries:

Orbit-State Timeseries
~~~~~~~~~~~~~~~~~~~~~~

When the ephemeris provides GCRS position and velocity samples, completed
:class:`~conops.ditl.ditl.DITL` and :class:`~conops.ditl.queue_ditl.QueueDITL` runs attach an
orbit-state timeline to the plan. :meth:`~conops.targets.Plan.save` writes it beside the plan
as ``<plan_stem>_orbit_state_timeseries.json`` and stores that basename in
``orbit_state_timeseries_file``.

The file has the same common metadata fields as the attitude timeseries file (``version``,
``coast_sim_version``, ``created_at``, ``plan_file``, ``plan_version``, ``plan_start``,
``plan_end``, and ``num_samples``). It also declares its coordinate conventions:
``frame`` is ``"GCRS"``, ``origin`` is ``"Earth center"``, positions are in kilometres, and
velocities are in kilometres per second. Each sample contains:

.. code-block:: json

   {
     "utime": 1748736000.0,
     "timestamp": "2025-12-01T00:00:00+00:00",
     "position_km": [6524.834, 6862.875, 6448.296],
     "velocity_km_s": [4.902, 5.533, -1.976]
   }

Load the file through :class:`~conops.targets.OrbitStateTimeseriesSchema`:

.. code-block:: python

   import json
   from pathlib import Path

   from conops.targets import OrbitStateTimeseriesSchema, Plan

   plan_path = Path("plan_20251201T000000Z_20251201T235900Z_v3.json")
   plan = Plan.load(plan_path)
   if plan.orbit_state_timeseries_file:
       raw = json.loads(
           (plan_path.parent / plan.orbit_state_timeseries_file).read_text()
       )
       orbit_states = OrbitStateTimeseriesSchema.model_validate(raw)
       print(f"Loaded {orbit_states.num_samples} orbit-state samples")

Programs
--------

A :class:`~conops.targets.Program` describes what a run is asked to schedule, in the same
way that a ``Plan`` describes what a run executed. It is one JSON file, written for a
window of time, that holds the pool of observation requests, the Targets of Opportunity
(ToOs) with their triggers, the scheduled activities and the observation groups.

A program has the following fields.

``requests``
    The observation requests. Each one is a :class:`~conops.targets.Pointing` or a
    :class:`~conops.targets.MovingPointing`. Each request is saved with a ``kind``, and it
    is read back as the class that its ``kind`` names (see `Typed entries`_).

    :meth:`~conops.targets.Program.save` writes a request without its execution fields.
    These are the fields that a run fills in, such as ``begin``, ``end``, ``slewtime``,
    ``roll`` and ``done``. The full list is
    :attr:`~conops.targets.PlanEntry.EXECUTION_FIELDS`. They are left out so that the file
    holds the request as it was asked for, without the state of any particular run.

    A moving target's track is written to the file if the track was given, or was looked
    up, before the save. A moving target that names a body and has no track has its track
    looked up when its visibility is first computed. ``create_ditl`` computes it while it
    queues the targets for a dispatch run or builds the plan for a planned run. A rolling
    run computes it when it builds a plan during the run. ``Program.bind`` looks tracks up
    only when it is called with ``refresh_tracks=True``. ``create_ditl`` calls it without
    that argument.

``toos``
    The :class:`~conops.targets.TOORequest` records. Each one holds a request of any kind,
    together with its ``submit_time``, its ``deadline`` and whether it may ``interrupt``
    the observation in progress. A ``TOORequest`` can also be built from the flat fields
    ``obsid``, ``ra``, ``dec``, ``merit``, ``exptime``, ``name``, ``science_data`` and
    ``instrument_modes``, in which case it holds a plain pointing. The ``executed`` flag
    records whether a run has taken the ToO up. It is run state and is not written to the
    file.

``activities``
    The :class:`~conops.targets.ScheduledActivity` records. To build a repeating series
    of activities, use :func:`~conops.targets.activity.periodic_activities`.

``observation_groups``
    The :class:`~conops.config.ObservationGroups`. A group names requests by their obsids,
    so the groups are stored with the program that holds those requests. A run reads the
    groups from the configuration, and :meth:`~conops.targets.Program.bind` copies them
    there.

``begin`` and ``end``
    The window the program is written for. Every time in the file is absolute. As in a
    ``Plan``, times are ISO-8601 strings on disk and Unix seconds in memory.

``version``, ``coast_sim_version``, ``created_at``, ``name``, ``description`` and the roll convention fields
    These have the same meaning as in a ``Plan``.

.. code-block:: python

   from conops import create_ditl
   from conops.targets import Program

   program = Program(begin=begin, end=end, requests=requests, toos=toos,
                     activities=activities, observation_groups=groups)
   program.save("program.json")

   loaded = Program.load("program.json")
   ditl = create_ditl(config, program=loaded)   # binds, queues, submits the ToOs

Pass a program to ``create_ditl`` with ``program=`` in place of the ``targets=`` and
``activities=`` arguments. ``create_ditl`` raises an error if a program is given together
with either of them. It binds the program to the configuration and submits the program's
ToOs to the run, so there is no need to submit them separately.

A planned run builds its whole plan before execution starts and cannot take ToOs.
``create_ditl`` therefore raises an error when the scheduler mode is planned and the
program contains ToOs. Use dispatch or rolling replanning for a program with ToOs.

``QueueDITL.submit_too`` and ``RollingHorizonDITL.submit_too`` accept a ``TOORequest`` as
well as the flat fields.

Typed entries
~~~~~~~~~~~~~

Every saved entry carries a ``kind`` field that names its class:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``kind``
     - Class
   * - ``entry``
     - A plain :class:`~conops.targets.PlanEntry`
   * - ``pointing``
     - :class:`~conops.targets.Pointing`
   * - ``moving_pointing``
     - :class:`~conops.targets.MovingPointing`

``Plan.load``, ``Program.load`` and ``TargetList`` read each entry as the class its
``kind`` names. A plan saved after a run therefore reloads with the same entry classes,
and with the request fields those classes hold: ``visit_windows``, ``sequence``,
``earliest_start``, ``deadline``, ``roll_offset_deg``, ``protect_opening``, ``visits``
and a moving target's ``body``. A plan that is used as a start plan needs these fields.

A plan file written by a version of COASTSim that did not yet save a ``kind`` is still
accepted. Its entries are read as plain entries.

A plan's ``start`` and ``end`` are taken from its scheduled entries. A request that was
never scheduled has no ``begin`` or ``end`` and does not affect them.

Backward Compatibility
----------------------

:meth:`~conops.targets.Plan.save` creates any missing parent directories
automatically, so you can pass a nested path without creating it first.

:meth:`~conops.targets.Plan.load` accepts files written by older versions of
COASTSim. It handles the differences as follows:

* Fields not present in the file (e.g. ``created_at``, ``num_entries``) are filled with
  schema defaults. ``num_entries`` is always recomputed from the actual entry list after
  loading.
* Older files may store an entry's times (``begin``, ``end``, ``contact_begin``,
  ``contact_end``, ``collection_begin`` and ``collection_end``) as numeric Unix timestamps
  (float/int) where current files use ISO-8601 strings. Both forms are accepted. In memory
  an entry's times are floats holding Unix seconds, as they were before. Only the on-disk
  format changed to ISO-8601.
* The plan-level ``start`` and ``end`` are not read from the file, whichever form they are
  stored in. They are computed from the entries after loading and are reported as ISO-8601
  strings.
* Older files may store ``version`` as a semantic-version string (e.g. ``"0.1.3"``). Such
  a value is coerced to ``0``.
* Older files must already use the field names documented above (including ``exporig``).
  There is currently no automatic renaming or aliasing of deprecated keys.

``Plan`` does not set ``from_attributes`` in its model configuration, so
``model_validate`` accepts plain dicts by default. To validate an object that exposes the
expected attributes, pass ``from_attributes=True`` in the call, as ``PlanSchema.from_plan``
does.

API Reference
-------------

.. automethod:: conops.targets.plan.Plan.save
   :no-index:

.. automethod:: conops.targets.plan.Plan.load
   :no-index:

.. autoclass:: conops.targets.plan_schema.PlanEntrySchema
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: conops.targets.plan_schema.PlanSchema
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: conops.targets.plan_schema.AttitudeTimeseriesSchema
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: conops.targets.plan_schema.AttitudeSampleSchema
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: conops.targets.plan_schema.OrbitStateTimeseriesSchema
  :members:
  :undoc-members:
  :show-inheritance:
  :no-index:

.. autoclass:: conops.targets.plan_schema.OrbitStateSampleSchema
  :members:
  :undoc-members:
  :show-inheritance:
  :no-index:

.. _Pydantic v2: https://docs.pydantic.dev/latest/
