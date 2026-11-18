Defining a Science Program
==========================

A science program is the set of observations requested of an observatory over a
period of time, such as a day or a week of operations. It is the scheduler's input:
a pool of observation requests, usually more than can fit, from which a schedule is
built.

In COASTSim a program is a :class:`~conops.targets.Program` object, which saves to
one JSON file. Besides the observation requests, it contains the other inputs to a
schedule for the same period: Targets of Opportunity (ToOs) submitted part-way
through, engineering activities during which no science is observed, and groups of
requests that share a window or a time budget.

This page explains what goes into a program, why each part exists, and how to write
one, with an example that runs. The detailed behaviour of each part is described in
:doc:`planning`, and the file format in :doc:`plan_serialization`.

Where a program fits
--------------------

A simulation has three inputs, and it helps to keep them apart:

* The **mission configuration** (:class:`~conops.config.MissionConfig`) describes
  the spacecraft: its power system, attitude control, instruments, constraints and
  ground stations. It changes rarely. See :doc:`configuration`.
* The **ephemeris** gives the orbit over the window.
* The **program** lists the observations and activities requested for that
  window. It changes with every planning cycle.

A run produces a :class:`~conops.targets.Plan`, the record of what was actually
observed. A program is the request and a plan is the result. The two files are
deliberately alike, so a request can be followed from the program into the plan by
its observation ID.

Without a program, a run's inputs are passed separately: a list of targets, a list
of activities, ToOs submitted one at a time, and groups set on the configuration.
That works in a script, but nothing records which inputs belonged together. Keeping
them in a program has three practical uses:

* The same file can be run in dispatch, planned and rolling modes, or with different
  spacecraft configurations, and the results compared knowing the inputs were
  identical.
* The file can be written by another tool, such as a proposal database or a survey
  planner, and read by COASTSim without any Python.
* The file can be kept with the results as the record of what was asked for.

.. note::

   The word "program" is also used for a category's ``program`` setting, which
   names the allocation whose share of observing time the category's targets count
   towards (see `Observation IDs and categories`_). That is a label on
   categories in the mission configuration. On this page, "program" means the
   :class:`~conops.targets.Program` object unless a category's setting is named.

What a program contains
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Field
     - Holds
     - Purpose
   * - ``begin``, ``end``
     - The window, as datetimes, ISO-8601 strings or Unix seconds
     - The period the program was written for. Every time in a program is absolute,
       so a program is only meaningful for its own window.
   * - ``requests``
     - :class:`~conops.targets.Pointing` and
       :class:`~conops.targets.MovingPointing` objects
     - The pool of observations the scheduler chooses from. Not every request has
       to be observed: a program normally asks for more than fits.
   * - ``toos``
     - :class:`~conops.targets.TOORequest` objects
     - Requests that become known only at a given time during the run.
   * - ``activities``
     - :class:`~conops.targets.ScheduledActivity` objects
     - Blocks of time with no science, such as maintenance or calibration.
   * - ``observation_groups``
     - :class:`~conops.config.ObservationGroups`
     - Sets of requests that share a window or a time budget, or are preferred one
       after another.
   * - ``name``, ``description``
     - Text
     - Labels for people. They do not affect the run.

Only ``begin`` and ``end`` are required. The other fields default to empty.

Observation IDs and categories
------------------------------

Every request has an ``obsid``, an integer that identifies it in the program, in
the plan, in the event log and in the statistics. Give every request its own
observation ID. A program is not checked for duplicates when it is loaded, and two
requests with the same ID cannot be told apart in the results. ToOs and activities
have observation IDs too, which should not repeat those of the requests. Scheduled
activities are numbered from 998000 when they are not given an ID, so keep science
IDs below that.

A request's observation ID also determines its **category**. A category is
a range of observation IDs defined in the mission configuration, and it has the
settings that apply to a whole kind of science: its cadence, its share of observing
time, its setup and cleanup times, and whether its observations may be interrupted.
A program does not define categories. It only uses observation IDs that fall in
them. It is therefore worth choosing an ID range for each kind of science before
writing any requests, for example 10000 to 19999 for a survey and 20000 to 29999 for
monitoring. See the observation categories section of :doc:`configuration`.

Writing requests
----------------

A request is a :class:`~conops.targets.Pointing`: a position on the sky, how much
science is wanted there, and how much it is worth. The fields that matter most are:

* ``ra`` and ``dec``, in degrees.
* ``merit``, the request's base priority. A higher merit is observed first. Only
  the relative values matter. Where the configuration enables dynamic merit terms
  such as cadence or urgency, those terms are added to the base merit, so leave
  gaps between priority bands that are wider than the dynamic terms.
* ``exptime``, the total science time wanted, in seconds.
* ``ss_min`` and ``ss_max``, the shortest and longest single snapshot, in seconds.

The exposure is normally longer than one snapshot can be, because a target in low
Earth orbit is hidden by the Earth for part of each orbit. The scheduler therefore
collects the exposure in several **snapshots**, each between ``ss_min`` and
``ss_max`` seconds long, until the exposure is complete. A request of 2400 seconds
with an ``ss_max`` of 1200 is observed in at least two snapshots, which need not be
consecutive.

.. code-block:: python

   from conops.targets import Pointing

   field = Pointing(
       config=config, obsid=10000, name="field 10000", ra=105, dec=10,
       merit=60, ss_min=300, ss_max=1200,
   )
   field.exptime = 2400

Requests can have further conditions. Each is optional, and each is described in
full in :doc:`planning`.

**A window.** ``earliest_start`` and ``deadline`` limit when the request's science
collection may begin. Use them for an observation that is only useful in part of
the program's window, such as one coordinated with another observatory.

**Set times.** A request with ``visit_windows`` is visited once in each window, and
each visit is taken whole. Use this for monitoring at fixed epochs. The helper
:func:`~conops.targets.cadence_windows` builds evenly spaced windows, and
:func:`~conops.targets.visit_windows` builds windows at times you list. For a plain
pointing, one visit is one snapshot of ``ss_max`` seconds.

.. code-block:: python

   from conops.targets import cadence_windows

   # Three ten-minute visits six hours apart. Each may start up to two hours
   # after its nominal time.
   monitor = Pointing(
       config=config, obsid=20000, name="variable star", ra=105, dec=0,
       merit=80, ss_max=600,
       visit_windows=cadence_windows(
           begin + timedelta(hours=1), 6 * 3600, count=3, slack_seconds=2 * 3600
       ),
   )

This differs from a category's ``cadence_seconds``, which spaces visits by a
minimum interval without fixing when they happen. Use visit windows when the times
themselves matter, and a category cadence when only the spacing does.

**A sequence of steps.** A request with a ``sequence`` is observed as a fixed
series of steps, such as a reference star followed by the target. A step can have
its own pointing, duration and instrument, and can be marked as collecting no
science. One visit of the sequence is always taken whole, and ``visits`` is the
number of visits requested.

.. code-block:: python

   from conops.targets import ObservationSequence, SequenceStep

   calibrated = Pointing(
       config=config, obsid=30000, name="calibrated field", ra=110, dec=20,
       merit=90, deadline=begin + timedelta(hours=18),
       sequence=ObservationSequence(
           steps=[
               SequenceStep(name="reference", duration_seconds=300, science=False),
               SequenceStep(name="target", duration_seconds=1200),
           ]
       ),
   )

**A moving target.** A Solar System body is a
:class:`~conops.targets.MovingPointing`, given either a ``track`` of positions or
the name of a ``body`` whose track is looked up. See the moving targets section of
:doc:`planning`.

Targets of Opportunity
----------------------

A ToO is a request that is added to the run at its ``submit_time`` and is not
available to the scheduler before then. It models an alert that arrives during
operations. A :class:`~conops.targets.TOORequest` contains an ordinary request and
adds:

* ``submit_time``, when the ToO is added. Before this time it is not in the queue
  and no plan includes it.
* ``deadline``, the latest time its science collection may begin.
* ``interrupt``, whether it may cut short the observation in progress. It defaults
  to true.

.. code-block:: python

   from conops.targets import TOORequest

   transient = Pointing(
       config=config, obsid=90000, name="transient", ra=115, dec=0,
       merit=1000, ss_min=600, ss_max=1200,
   )
   transient.exptime = 1200
   too = TOORequest(
       request=transient,
       submit_time=begin + timedelta(hours=9),
       deadline=begin + timedelta(hours=13),
   )

A ToO is ranked by merit like any other request, so give it a merit above the
science it is meant to displace. Putting ToOs in the program, in place of
submitting them from a script, means a run can be repeated with exactly the same
alerts. See :doc:`target_of_opportunity`.

Scheduled activities
--------------------

A :class:`~conops.targets.ScheduledActivity` is a block of time with a start and a
duration during which no science is observed. The schedulers reserve it first and
fit science around it. Use activities for anything that stops science at a known
time: a maintenance window, a calibration that needs the
spacecraft held still, or an engineering test.

.. code-block:: python

   from conops.targets import ScheduledActivity

   maintenance = ScheduledActivity(
       name="Maintenance", begin=begin + timedelta(hours=20), duration_seconds=1800
   )

By default the spacecraft holds the attitude it already has. An activity can ask for
a particular attitude, name instrument modes, generate data, or be allowed to start
a little late so that an observation in progress can finish (``slide_seconds``).
:func:`~conops.targets.activity.periodic_activities` builds a repeating series. See
the scheduled activities section of :doc:`planning`.

Observation groups
------------------

An :class:`~conops.config.ObservationGroup` names a set of requests by their
observation IDs and gives them something in common:

* ``window_start`` and ``window_deadline`` narrow when any member may be observed.
* ``time_budget_seconds`` caps the science the members collect together.
* ``prefer_contiguous``, on by default, raises a member's rank once another member
  has just been observed, so the members tend to be observed one after another.

Use a group for requests that only make sense together, such as the tiles of a
mosaic or the targets of one redshift bin. Groups are stored in the program, not in
the mission configuration file, because they refer to the program's own observation
IDs.

.. code-block:: python

   from conops.config import ObservationGroup, ObservationGroups

   groups = ObservationGroups(
       groups=[ObservationGroup(name="mosaic", members=[10000, 10001, 10002])]
   )

A complete example
------------------

The example below builds a one-day program from the pieces above, saves it, loads
it back and runs it. It uses the spacecraft of the standard benchmark scenarios,
with three categories.

.. code-block:: python

   from datetime import datetime, timedelta, timezone

   from conops import create_ditl
   from conops.benchmark.scenarios import mission_config
   from conops.config import (
       ObservationCategory,
       ObservationGroup,
       ObservationGroups,
   )
   from conops.targets import (
       ObservationSequence,
       Pointing,
       Program,
       ScheduledActivity,
       SequenceStep,
       TOORequest,
       cadence_windows,
   )

   begin = datetime(2025, 11, 1, tzinfo=timezone.utc)
   end = begin + timedelta(days=1)
   config = mission_config(
       "examples/example.tle", begin, end, seed=1, stations=False,
       categories=[
           ObservationCategory(name="Survey", obsid_min=10000, obsid_max=20000),
           ObservationCategory(name="Monitoring", obsid_min=20000, obsid_max=30000),
           ObservationCategory(name="Calibrated", obsid_min=30000, obsid_max=40000),
       ],
   )

   # Three survey fields, each 40 minutes in snapshots of 5 to 20 minutes.
   survey = []
   for obsid, (ra, dec) in enumerate([(105, 10), (110, 10), (115, 10)], 10000):
       field = Pointing(
           config=config, obsid=obsid, name=f"field {obsid}", ra=ra, dec=dec,
           merit=60, ss_min=300, ss_max=1200,
       )
       field.exptime = 2400
       survey.append(field)

   # A star visited three times, six hours apart.
   monitor = Pointing(
       config=config, obsid=20000, name="variable star", ra=105, dec=0,
       merit=80, ss_max=600,
       visit_windows=cadence_windows(
           begin + timedelta(hours=1), 6 * 3600, count=3, slack_seconds=2 * 3600
       ),
   )

   # A field observed after a reference, in the first 18 hours.
   calibrated = Pointing(
       config=config, obsid=30000, name="calibrated field", ra=110, dec=20,
       merit=90, deadline=begin + timedelta(hours=18),
       sequence=ObservationSequence(
           steps=[
               SequenceStep(name="reference", duration_seconds=300, science=False),
               SequenceStep(name="target", duration_seconds=1200),
           ]
       ),
   )

   # An alert that arrives nine hours into the day.
   transient = Pointing(
       config=config, obsid=90000, name="transient", ra=115, dec=0,
       merit=1000, ss_min=600, ss_max=1200,
   )
   transient.exptime = 1200

   program = Program(
       name="Example program",
       description="A survey, a monitored star and a calibrated field for one day",
       begin=begin,
       end=end,
       requests=[*survey, monitor, calibrated],
       toos=[
           TOORequest(
               request=transient,
               submit_time=begin + timedelta(hours=9),
               deadline=begin + timedelta(hours=13),
           )
       ],
       activities=[
           ScheduledActivity(
               name="Maintenance",
               begin=begin + timedelta(hours=20),
               duration_seconds=1800,
           )
       ],
       observation_groups=ObservationGroups(
           groups=[ObservationGroup(name="mosaic", members=[10000, 10001, 10002])]
       ),
   )
   program.save("program.json")

   loaded = Program.load("program.json")
   ditl = create_ditl(config, program=loaded, begin=begin, end=end)
   ditl.calc()
   ditl.print_statistics()

In the default dispatch mode this run observes the calibrated field first, then the
survey fields with the star's first visit between them, the star again at 07:00 and
13:00, the transient at 09:00 when it is submitted, and the maintenance block at
20:00.

The program file
----------------

:meth:`~conops.targets.Program.save` writes JSON. The example's file begins like
this, with most requests left out:

.. code-block:: json

   {
     "version": 0,
     "name": "Example program",
     "description": "A survey, a monitored star and a calibrated field for one day",
     "begin": "2025-11-01T00:00:00+00:00",
     "end": "2025-11-02T00:00:00+00:00",
     "requests": [
       {
         "kind": "pointing",
         "name": "field 10000",
         "ra": 105.0,
         "dec": 10.0,
         "obsid": 10000,
         "merit": 60.0,
         "ss_min": 300.0,
         "ss_max": 1200.0,
         "exptime": 2400.0
       }
     ],
     "toos": [
       {
         "request": {
           "kind": "pointing",
           "name": "transient",
           "ra": 115.0,
           "dec": 0.0,
           "obsid": 90000,
           "merit": 1000.0,
           "ss_min": 600.0,
           "ss_max": 1200.0,
           "exptime": 1200.0
         },
         "submit_time": "2025-11-01T09:00:00+00:00",
         "deadline": "2025-11-01T13:00:00+00:00",
         "interrupt": true
       }
     ],
     "activities": [
       {
         "name": "Maintenance",
         "begin": "2025-11-01T20:00:00+00:00",
         "duration_seconds": 1800.0
       }
     ],
     "observation_groups": {
       "groups": [{"name": "mosaic", "members": [10000, 10001, 10002]}]
     }
   }

A file written by hand or by another tool needs only the fields shown. The rules
are:

* Every time is written as an ISO-8601 string. A file may also give times as Unix
  seconds, and a string without a time zone is read as UTC.
* Each request has a ``kind``: ``pointing`` or ``moving_pointing``. A request
  without one is read as a pointing.
* Fields left out take their defaults.
* An unknown field in a group is an error. Anywhere else in the file an unknown
  field is ignored without a message, so a misspelt key silently leaves its setting
  at the default. Check the spelling of hand-written files against the saved
  example.
* The file holds requests as they were asked for. The fields a run fills in, such
  as ``begin``, ``end``, ``slewtime`` and ``done``, are not written, so saving a
  program after a run does not write that run's state into the file.

The complete field list, and how the request classes are told apart, is in
:doc:`plan_serialization`.

Running a program
-----------------

Pass the program to :func:`~conops.ditl.create_ditl` as ``program=``. This replaces
the ``targets=`` and ``activities=`` arguments, and giving a program together with
either of them is an error. ``create_ditl`` then does the following:

1. It binds the program: every request is given the configuration and the
   ephemeris, and the program's groups are copied to the configuration.
2. It queues or plans the requests, as the configuration's ``scheduler`` section
   selects.
3. It reserves the activities.
4. It submits the ToOs, so they need not be submitted separately.

The run's window comes from the ``begin`` and ``end`` arguments of ``create_ditl``,
or from the ephemeris where they are not given. It is not taken from the program,
so pass the program's own window, as the example does, unless you intend to run
only part of it.

How the requests are used depends on the scheduler mode:

* **Dispatch**, the default, picks the best visible request each time the
  spacecraft is free. ToOs join the queue when they are submitted.
* **Planned** builds one plan for the whole window before the run starts. A plan
  built in advance does not include a ToO submitted later, so ``create_ditl`` raises an error for a
  planned run whose program contains ToOs. Remove them, or use another mode.
* **Rolling** builds a plan and rebuilds it as the run goes, including when a ToO
  arrives.

See :doc:`planning` for the modes and the schedulers.

To use a program without ``create_ditl``, call
:meth:`~conops.targets.Program.bind` yourself and pass ``program.requests`` to a
scheduler. A request loaded from a file has no configuration or ephemeris until it
is bound, and its visibility cannot be computed before then.

Common mistakes
---------------

* **Observation IDs outside every category.** Such requests fall into the default
  category and get none of the settings intended for them, such as a cadence or a
  time share.
* **A minimum snapshot longer than the target is visible.** A request whose ``ss_min`` is
  longer than any visibility window is never observed. Check the target's
  visibility before choosing snapshot lengths.
* **Setting both a window and visit windows.** A request with ``visit_windows``
  does not also set ``earliest_start`` or ``deadline``. Each visit window already
  bounds its own visit.
* **Reusing a program for another window.** All times are absolute. To run another
  day, write a new program with new times.
* **A ToO with low merit.** A ToO is selected only if its merit is higher than the
  other requests', and it interrupts only if its merit is higher than that of the
  observation in progress.
* **Expecting the whole pool to be observed.** A program normally holds more than
  fits. The plan shows what was observed, and a planning scheduler's ``unplaced``
  attribute lists the requests it could not fit.

Where to go next
----------------

* :doc:`planning` describes how each kind of request, activity and group is
  scheduled, in every mode.
* :doc:`target_of_opportunity` describes how ToOs are selected and when they
  interrupt.
* :doc:`configuration` describes observation categories, dynamic merit and the
  ``scheduler`` section.
* :doc:`plan_serialization` gives the file formats of programs and plans.
