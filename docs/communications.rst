Communications System Configuration
====================================

Overview
--------

The communications system configuration describes how the spacecraft talks to the
ground: the frequency bands it carries, the uplink and downlink rate on each band,
and the type of antenna and how it is pointed. The simulation uses it to decide
when the spacecraft is in contact with a ground station and how much data it
downlinks there.

Key Features
------------

The configuration covers the following:

* Frequency bands: UHF, S, X, Ka, Ku, L, C and K.
* Antenna types: omnidirectional, fixed and gimbaled.
* Data rates: each frequency band has its own uplink rate and downlink rate.
* Default rates: a band is given the standard rates for that band when none are specified.
* Antenna pointing: how the antenna is mounted and what pointing it requires.
* Polarization: linear and circular options.
* Ground stations: each station has its own downlink capability on each band.

Components
----------

BandCapability
^^^^^^^^^^^^^^

A ``BandCapability`` defines the data rates on one frequency band. It has these fields:

* ``band``: Frequency band identifier (``"UHF"``, ``"S"``, ``"X"``, ``"Ka"``, ``"Ku"``, ``"L"``, ``"C"``, ``"K"``)
* ``uplink_rate_mbps``: Uplink data rate in Mbps (≥ 0)
* ``downlink_rate_mbps``: Downlink data rate in Mbps (≥ 0)

Standard defaults
"""""""""""""""""

When only the ``band`` is specified, the standard rates for that band are applied:

.. list-table::
   :header-rows: 1

   * - Band
     - Uplink (Mbps)
     - Downlink (Mbps)
   * - S
     - 2.0
     - 10.0
   * - X
     - 10.0
     - 150.0
   * - Ka
     - 20.0
     - 300.0
   * - Ku
     - 5.0
     - 50.0
   * - L
     - 0.5
     - 1.0
   * - C
     - 2.0
     - 20.0
   * - K
     - 15.0
     - 200.0

UHF has no generic default rate; specify mission-specific uplink and downlink
rates explicitly. Explicitly specified values override all defaults.

Rates that vary with range
""""""""""""""""""""""""""

A band's downlink rate can vary with the range from the spacecraft to the
station. ``downlink_rate_vs_range_km`` is a table of ``(range_km, rate_mbps)``
rows in increasing range. The rate is interpolated linearly between rows, and
beyond the first and last rows it is held at that row's rate.

Either side of a link may carry a table, the spacecraft's band or the ground
station's. The effective rate at a range is the lower of the two sides' rates at
that range, on the best band.

The table is used where the range is known, which is in the contacts of an omni
or gimballed antenna. Those contacts take the rate at each step's range, so a
contact at low elevation downlinks less than one overhead. Each contact span
reports its time-weighted mean rate. A body-fixed antenna's passes use
``downlink_rate_mbps`` throughout.

AntennaPointing
^^^^^^^^^^^^^^^

An ``AntennaPointing`` defines how the antenna is mounted and pointed. There are
three antenna types:

* ``OMNI``: Omnidirectional antenna (no pointing requirement)
* ``FIXED``: Body-fixed antenna with defined pointing direction
* ``GIMBALED``: Steerable antenna with angular range

It has these fields:

* ``antenna_type``: Type of antenna mounting
* ``fixed_boresight_body``: Fixed antenna boresight as a unit vector in spacecraft body frame.
  The default ``(-1, 0, 0)`` keeps the tracking convention that ground-station
  passes used in earlier versions.
* ``fixed_azimuth_deg`` / ``fixed_elevation_deg``: Deprecated metadata fields. Use
  ``fixed_boresight_body`` for GSP attitude generation. Nonzero values are accepted
  with a deprecation warning and do not affect generated GSP attitudes.
* ``gimbal_range_deg``: Angular range for gimbaled antennas (from boresight)
* ``repoint_threshold_deg``: For a gimballed antenna, the pointing allowance. It is
  the radius of a cap about the antenna's pointing, and the tracked station is kept
  inside that cap: the antenna is re-pointed ahead of the station where the station
  would leave it. The default is None, which means no tracking repoints.
* ``repoint_interval_vs_range_km``: Used instead of the threshold. It gives the
  interval between repoints, in seconds, against the spacecraft's distance from the
  Earth's centre, as ``(range_km, seconds)`` rows. The table is interpolated at the
  range at each repoint.
* ``repoint_seconds``: Seconds a repoint takes (default 0). Nothing is downlinked
  during a repoint, and an observation of a category that forbids repoints cannot
  collect during one.

The three repoint fields apply to a ``GIMBALED`` antenna only: an omni antenna has no
direction to re-point and a body-fixed one moves with the spacecraft. `Antenna
repoints`_ below describes how they are used.

The rest of this section describes how ground contacts are chosen. That depends
on the antenna type and, for omni and gimballed antennas, on the communications
system's ``contact_policy``.

Continuous contacts
"""""""""""""""""""

For omni and gimballed antennas, ``contact_policy="continuous"`` is the
communications default. Under this policy the link is handed over from one
station to the next, and overlapping station windows are all kept.

:class:`~conops.simulation.PassTimes` retains all available visibility windows,
including their overlapping tails and windows shorter than eight minutes. It then
chooses one station at a time, as follows:

* It selects the station with the highest instantaneous effective downlink rate.
  Ties are broken by the highest elevation and then by station code, so the
  choice is stable.
* It tracks that station until the station sets or its rate becomes zero, and
  then hands over to the best usable station.
* A faster station rising does not interrupt the current link.

A chain of overlapping windows therefore supplies continuous network coverage
instead of being reduced to whole, mutually exclusive passes.

Both ``PassTimes.predict()`` (the whole ephemeris) and ``PassTimes.get()``
(the requested days) produce the selected station reservations under this policy.
The work is divided between two classes. The pass scheduler owns availability
draws, station selection and trimming.
:class:`~conops.simulation.StationCoverage` consumes the scheduler's reservations
and applies link rates and the acquisition, handover and repoint outages. Calling
the scheduler directly therefore produces the same handovers used by planning and
execution.

Station availability
""""""""""""""""""""

A station being visible does not mean the spacecraft has access to it. Each
ground station has a ``schedule_probability``, which is drawn once per station
visibility window. Zero rejects the window; one accepts it. ``random_seed``
reproduces the draws.

Availability is decided once over the full ephemeris, and the decision is shared
by dispatch and every replan. Repeated daily queries on the same pass scheduler
retain a visibility window's first availability decision across midnight,
including a rejected window. A new scheduler makes its own decisions. Replaying
saved bookings makes no new draws.

Scheduled passes
""""""""""""""""

Set ``contact_policy="scheduled"`` to use whole-pass booking instead.
:class:`~conops.simulation.PassTimes` then applies its eight-minute minimum
and selects whole non-overlapping passes. It prefers passes by downlink volume,
then duration, then earlier start. Range-dependent effective rates contribute to
that volume.

Body-fixed antennas always use this pass-booking path. Their passes have
spacecraft tracking profiles and ``PASS`` activities that pause science, and the
contact policy does not change their behavior. Only fixed antennas require
``schedule_for_tracking`` eligibility.

Acquisition and handover
""""""""""""""""""""""""

Both policies for omni and gimballed antennas select one station at a time and
use the same acquisition, handover and repointing accounting:

* Every contact starts ``acquisition_seconds`` after selection.
* A direct switch from one station to another adds ``handover_seconds``.
* Nothing is downlinked during those outages.
* A fresh acquisition after a genuine availability gap pays no handover.

Rates use the bands shared by station and spacecraft at their range at each step.
Execution also checks a gimballed antenna's reach from the actual body pose,
whereas planning takes its selected station as reachable.

Contacts saved in a plan
""""""""""""""""""""""""

The recorder ledgers, dispatch and planners consume the same selected
contact timeline and its outages. Plans save the timeline's exact raw
``(begin, end, station)`` segments in ``ground_contacts``, before acquisition
and repoint costs.

Replaying a plan restores those segments without selecting again or repeating
availability draws, including when the seed or contact policy has changed. An
empty list restores no contacts. A plan saved earlier therefore retains its
original bookings; rebuild from the program to use a new policy. Plans without
``ground_contacts`` predict from the configuration.

Antenna repoints
""""""""""""""""

A gimballed antenna is re-pointed at each acquisition and handover, and within a
contact as the spacecraft moves along its orbit. An omni antenna is never
re-pointed, so it has no repoint events.

While an attitude is held, the station's direction moves in the body frame exactly
as it moves across the sky. Tracking repoint times are therefore calculated from
the station geometry from the contact's start. One of two settings decides when
the tracking repoints come:

* With ``repoint_threshold_deg`` set, the antenna leads the station. At each
  repoint it is pointed at the centre of the longest arc of the station's coming
  track that fits within a cap of that radius. The station enters the cap on one
  side and leaves on the other, and the next repoint comes where the arc no longer
  fits. Taking each arc as long as it can be gives the fewest repoints for the
  allowance, about half as many as re-pointing at the station each time it moves
  the allowance away.
* With ``repoint_interval_vs_range_km`` set instead, repoints follow the table's
  interval at the spacecraft's range from the Earth's centre, and each repoint
  points at the station.

A repoint takes ``repoint_seconds``, during which nothing is downlinked, so a
contact is split around it. A repoint at acquisition or handover lasts as long as
the acquisition or handover latency.

Each repoint records the antenna's pointing and how far it turned. The simulation
logs each repoint as an ``ANTENNA`` event with the station and the observation in
progress. ``print_statistics`` reports the repoints by kind, with the mean and
largest turn.

An observation category's ``antenna_repoints`` setting says what a repoint during
one of its observations means:

* ``allowed`` (the default).
* ``reported``: the repoint is counted and logged.
* ``forbidden``: the schedulers cut the category's visibility windows at each
  repoint, so snapshots are placed between repoints. Dispatch neither starts such
  an observation during a repoint nor lets it run into the next one (see
  :doc:`planning`).

Reporting
"""""""""

A configuration without a communications system behaves as one with a body-fixed
antenna does.

An omni or gimballed antenna has a link coverage model
(:class:`~conops.simulation.StationCoverage`). With coverage, a DITL's
``coverage`` holds the model, its data telemetry shows the downlink, and
``print_statistics`` reports the number of contacts and handovers and the
fraction of the run in contact.

The timeline plots show the contacts in their ``Ground Contact`` row, coloured by
station. On a run up to two days long there is one bar per span. On a longer run a
span of minutes is narrower than a pixel, so there is one bar per hour with a
contact, and the bar is as dark as the hour's fraction in contact against the
run's fullest hour.

Polarization
^^^^^^^^^^^^

``Polarization`` defines the antenna's polarization type. The options are:

* ``LINEAR_HORIZONTAL``: Linear horizontal polarization
* ``LINEAR_VERTICAL``: Linear vertical polarization
* ``CIRCULAR_RIGHT``: Right-hand circular polarization (RHCP)
* ``CIRCULAR_LEFT``: Left-hand circular polarization (LHCP)
* ``DUAL``: Supports multiple polarizations

CommunicationsSystem
^^^^^^^^^^^^^^^^^^^^

``CommunicationsSystem`` is the main configuration class for the communications
system. It has these fields:

* ``name``: Descriptive name for the communications system
* ``band_capabilities``: List of supported frequency bands with data rates
* ``antenna_pointing``: Antenna pointing configuration
* ``pointing_accuracy_deg``: Maximum pointing error to maintain good signal (degrees)
* ``polarization``: Antenna polarization type
* ``contact_policy``: ``continuous`` (default) for station handovers without
  dropping overlaps, or ``scheduled`` for whole-pass booking. Fixed antennas
  always use tracking passes.
* ``acquisition_seconds``: Seconds to acquire the link during passes without
  spacecraft tracking (default 0). No data is downlinked during acquisition.
* ``handover_seconds``: Seconds to switch the link straight from one ground station to
  another during passes without spacecraft tracking (default 0). The handover time is
  added on top of acquisition, and nothing is downlinked during it.

It has these methods:

* ``get_band(band: str)``: Get capability for a specific band
* ``get_downlink_rate(band: str)``: Get downlink rate in Mbps
* ``get_uplink_rate(band: str)``: Get uplink rate in Mbps
* ``can_communicate(pointing_error_deg: float)``: Check if communication is possible

Configuration Examples
----------------------

Python Mission Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.common import AntennaType, Polarization
   from conops.config import (
       CommunicationsSystem,
       BandCapability,
       AntennaPointing,
   )

   # LEO satellite with body-fixed -X S-band (using defaults)
   leo_comms = CommunicationsSystem(
       name="LEO S-band",
       band_capabilities=[
           BandCapability(band="S")  # Uses defaults: 2.0 Mbps up, 10.0 Mbps down
       ],
       antenna_pointing=AntennaPointing(
           antenna_type=AntennaType.FIXED,
           fixed_boresight_body=(-1.0, 0.0, 0.0),
       ),
       polarization=Polarization.CIRCULAR_RIGHT,
       pointing_accuracy_deg=10.0,
   )

   # High-rate X-band with gimbaled antenna (custom rates)
   xband_comms = CommunicationsSystem(
       name="High-Rate X-band",
       band_capabilities=[
           BandCapability(
               band="X",
               uplink_rate_mbps=10.0,
               downlink_rate_mbps=300.0,
           )
       ],
       antenna_pointing=AntennaPointing(
           antenna_type=AntennaType.GIMBALED,
           gimbal_range_deg=60.0,
       ),
       polarization=Polarization.CIRCULAR_RIGHT,
       pointing_accuracy_deg=3.0,
   )

   # Check communication capability
   if xband_comms.can_communicate(pointing_error_deg=2.5):
       rate = xband_comms.get_downlink_rate("X")
       print(f"Can downlink at {rate} Mbps")

JSON Configuration
^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "name": "S/X Dual Band",
     "band_capabilities": [
       {
         "band": "S",
         "uplink_rate_mbps": 2.0,
         "downlink_rate_mbps": 10.0
       },
       {
         "band": "X",
         "uplink_rate_mbps": 0.0,
         "downlink_rate_mbps": 150.0
       }
     ],
     "antenna_pointing": {
       "antenna_type": "gimbaled",
       "gimbal_range_deg": 45.0
     },
     "polarization": "dual",
     "pointing_accuracy_deg": 5.0
   }

Ground Station Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ground station's band capabilities are stored directly on the ``GroundStation`` object:

.. code-block:: python

   from conops.config import GroundStation, BandCapability

   # Ground station with X-band capability
   station = GroundStation(
       code="SGS",
       name="Singapore Ground Station",
       latitude_deg=1.3521,
       longitude_deg=103.8198,
       bands=[
           BandCapability(band="X", downlink_rate_mbps=100.0)
       ],
       gain_db=45.0
   )

   # Access methods
   supported = station.supported_bands()  # ["X"]
   rate = station.get_downlink_rate("X")  # 100.0
   max_rate = station.get_overall_max_downlink()  # 100.0

Integration with Spacecraft Bus
--------------------------------

The communications system is set as the ``communications`` field of the spacecraft bus configuration:

.. code-block:: python

   from conops.config import BandCapability, CommunicationsSystem, SpacecraftBus

   bus = SpacecraftBus(
       name="My Spacecraft",
       communications=CommunicationsSystem(
           name="X-band Comms",
           band_capabilities=[
               BandCapability(band="X", downlink_rate_mbps=150.0)
           ]
       ),
   )

Common Configurations
---------------------

The notebook ``examples/Antenna_Types_Example.ipynb`` is a complete example. It
runs the same day in low Earth orbit with a body-fixed, an omnidirectional and a
gimbaled antenna, and compares the passes and the downlink of the three.

Use Cases
---------

Data Management
^^^^^^^^^^^^^^^

The communications configuration is used to:

* Calculate data transfer rates during ground station passes
* Determine data volume that can be downlinked per pass
* Manage onboard data storage requirements

Pass Definition
^^^^^^^^^^^^^^^

The configuration affects pass planning:

* ``pointing_accuracy_deg`` determines if spacecraft pointing is adequate
* ``antenna_type`` affects whether spacecraft must point at ground station
* Gimbaled antennas may allow communication without spacecraft slewing

Link Budget Analysis
^^^^^^^^^^^^^^^^^^^^

The data rates and pointing requirements can be used for link budget
calculations, for planning communication windows and for optimizing data
transfer.

Data Rate Matching
^^^^^^^^^^^^^^^^^^

During ground station passes, the effective downlink rate is calculated as follows:

1. For each band that the ground station and the spacecraft have in common,
   compute ``min(GS downlink rate, SC downlink rate)``.
2. Select the maximum of those rates across all common bands.
3. That rate is the limit on data downlink for the link.

Example:

.. code-block:: python

   # Ground station: X-band @ 100 Mbps
   # Spacecraft: X-band @ 150 Mbps
   # Effective rate: 100 Mbps (GS limited)

   # Ground station: S @ 50 Mbps, X @ 200 Mbps
   # Spacecraft: S @ 10 Mbps, X @ 150 Mbps
   # Effective rate: 150 Mbps (X-band, SC limited)

Ground Station Pass (GSP) Plan Entries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When :class:`~conops.ditl.queue_ditl.QueueDITL` executes a Day-In-The-Life simulation,
spacecraft tracking activities for ground-station passes are exported as ``GSP``
entries in the observation plan. Pass bookings without tracking are stored in
``ground_contacts``.

A GSP entry records:

* which ground station is used (``station`` field);
* the actual pass start and end times (``contact_begin``, ``contact_end``);
* the preparation time, which is the slew time to point at the station before
  contact begins.

The following rules decide which passes get an entry:

* GSP entries are created only for passes the spacecraft actually commands and executes.
* When multiple stations are simultaneously visible, the overlapping pass
  opportunities are deconflicted automatically. The system selects the pass with
  the highest expected data volume (downlink rate × duration).
* Passes are not commanded, and no GSP entry is created, when the spacecraft is in SAFE mode.

GSP entries are excluded from science observation timelines in visualization plots.

Example workflow:

.. code-block:: python

   from datetime import datetime, timedelta

   from rust_ephem import TLEEphemeris

   from conops import MissionConfig, QueueDITL
   from conops.common import ObsType

   # A configuration with ground stations and a body-fixed antenna
   config = MissionConfig.from_json_file("my_config.json")

   begin = datetime(2025, 11, 1)
   end = begin + timedelta(days=1)
   ephem = TLEEphemeris(
       begin=begin, end=end, step_size=60, tle="examples/example.tle"
   )

   # Run DITL with ground station passes. Targets are added to ditl.queue
   # as in the quickstart.
   ditl = QueueDITL(config=config, ephem=ephem)
   ditl.calc()

   # Export plan with GSP entries
   ditl.plan.save("observation_plan.json")

   # GSP entries appear in the saved plan with obstype="GSP"
   for entry in ditl.plan:
       if entry.obstype == ObsType.GSP:
           print(f"Ground pass: {entry.station} from "
                 f"{entry.contact_begin} to {entry.contact_end}")

See :doc:`plan_serialization` for details on the GSP entry format and serialization.

Implementation Notes
--------------------

* Fixed-antenna pointing. Ground-station-pass profiles use the unit body-frame vector in
  ``fixed_boresight_body``. The default ``(-1, 0, 0)`` keeps the tracking convention
  of earlier versions. ``fixed_azimuth_deg`` and ``fixed_elevation_deg`` are retained
  only as deprecated metadata and do not steer the generated profile.
* Omni antennas. ``can_communicate()`` always returns ``True`` for an omni antenna,
  regardless of pointing.
* Multiple bands. A system can support multiple bands simultaneously (e.g., S-band
  for commands, X-band for science data).
* Data rates. Set a rate to 0.0 if the band doesn't support uplink or downlink in
  that direction.
* Standard defaults. The standard rates are applied automatically when only the band
  name is specified. Provide explicit rates to override them.

Validation
----------

All configuration parameters are validated using Pydantic:

* Data rates must be non-negative
* Angles must be within valid ranges
* Band types must be from supported list

See the tests in ``tests/config/test_communications.py`` for examples.

API Reference
-------------

For detailed API documentation, see:

* :py:class:`conops.config.BandCapability`
* :py:class:`conops.config.CommunicationsSystem`
* :py:class:`conops.config.AntennaPointing`
* :py:class:`conops.common.AntennaType`
* :py:class:`conops.common.Polarization`
* :py:class:`conops.config.GroundStation`
