Offline Planning
================

Overview
--------

COASTSim runs a scheduler in one of three ways:

* **Dispatch** (closed loop): :class:`~conops.ditl.queue_ditl.QueueDITL` asks a
  target queue for the next target each time the spacecraft is free. This is the
  default way to simulate a queue-scheduled mission.
* **Planning** (open loop): a scheduler builds a whole
  :class:`~conops.targets.Plan` up front, and :class:`~conops.ditl.ditl.DITL`
  executes it.
* **Rolling-horizon replanning** (closed loop):
  :class:`~conops.ditl.rolling_ditl.RollingHorizonDITL` follows a plan and rebuilds
  it from the spacecraft's actual state as time passes, as a ground-planned mission
  does. See `Rolling-horizon replanning`_.

Over runs of days to months, a long-range allocator can decide on which day each
request is observed, and dispatch or rolling replanning follow it (see
`Long-range allocation`_).

A mission configuration's ``scheduler`` section selects the mode and scheduler, and
:func:`~conops.ditl.create_ditl` builds the matching simulation (see
:doc:`configuration`). The classes below can also be used directly.

:class:`~conops.schedulers.PriorityScheduler` is the planning engine. It reserves
ground contacts first, then takes requests in priority order and places each
snapshot in the earliest slot that still fits, without moving anything already
placed. It is fast and deterministic, and its plans are traceable to the priority
order. Because placements are never revisited, an early placement can block a
better packing. :class:`~conops.schedulers.LocalSearchScheduler` starts from its plan
and improves it (see `Improving plans by local search`_), and
:class:`~conops.schedulers.CpSatScheduler` optimizes it with a constraint solver (see
`Optimizing plans with CP-SAT`_).

Building and executing a plan
-----------------------------

The example below builds a one-day plan for two targets with
``PriorityScheduler`` and executes it with ``DITL``.

.. code-block:: python

   from datetime import datetime, timedelta, timezone

   from rust_ephem import TLEEphemeris

   from conops import DITL, MissionConfig, PriorityScheduler
   from conops.targets import Pointing

   begin = datetime(2025, 11, 1, tzinfo=timezone.utc)
   end = begin + timedelta(days=1)
   config = MissionConfig.from_json_file("examples/example_config.json")
   config.constraint.ephem = TLEEphemeris(
       tle="examples/example.tle", begin=begin, end=end, step_size=60
   )

   targets = []
   for obsid, (ra, dec, merit) in enumerate([(105, 10, 90), (30, 40, 60)], 10000):
       target = Pointing(
           config=config, obsid=obsid, ra=ra, dec=dec, merit=merit,
           ss_min=300, ss_max=1200,
       )
       target.exptime = 2400  # split into snapshots of up to ss_max
       targets.append(target)

   scheduler = PriorityScheduler(config, targets, begin, end)
   plan = scheduler.schedule()
   print(f"{len(plan)} entries; unplaced: {[t.obsid for t in scheduler.unplaced]}")

   ditl = DITL(config=config, plan=plan, begin=begin, end=end)
   ditl.step_size = scheduler.ctx.step_size
   ditl.calc()
   assert ditl.validate_plan_matches_execution() == []

Execute the plan with the same configuration, ephemeris, horizon and step size that
it was built with. The plan can also be saved and loaded (see
:doc:`plan_serialization`) before execution.

The rest of this section describes the options a target can carry into a plan.

Restricting science to sunlight
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``config.constraint.science_in_eclipse=False`` to restrict observations to
sunlit windows. An observation category can override the setting with its own
``science_in_eclipse``. Dispatch, priority, local-search and CP-SAT plans share this
policy, and it also covers Targets of Opportunity in a rolling run and whole
sequence visits. Natural and injected eclipses both apply. A stored plan that
conflicts with the current policy must be rebuilt before execution. See the
science-in-eclipse policy in :doc:`configuration`.

Limiting when a target is observed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A target's ``earliest_start`` and ``deadline`` limit when its science collection may
begin. Each can be given as Unix seconds, a datetime or an ISO-8601 string.

Keeping a run's inputs in one file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The requests, the Targets of Opportunity, the scheduled activities and the
observation groups that a run schedules can be kept together in one file as a
:class:`~conops.targets.Program`. Give the program to ``create_ditl`` as
``program=`` (see :doc:`plan_serialization`).

Targets observed at set times
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A target that is observed at set times is one request with ``visit_windows``. This
is a list of :class:`~conops.targets.VisitWindow` objects. Each window gives the
earliest and the latest time at which one visit's collection may begin. The windows
must be in time order and must not overlap.

A target with windows does not also set ``earliest_start`` or ``deadline``. Those
two fields bound the start of any snapshot, whereas a window bounds the start of one
whole visit.

The target is visited once per window, in order, and each visit is taken whole. For
a plain pointing, a visit is one snapshot of ``ss_max`` seconds, and the target's
``ss_min`` is set to match. For a target with a sequence, a visit is one visit of
its sequence. Unless the target's exposure is given, it is the science of all its
visits.

Two helpers build the list of windows:

* :func:`~conops.targets.visit_windows` builds windows that open at given times,
  each staying open for a slack.
* :func:`~conops.targets.cadence_windows` builds windows from a fixed first epoch,
  an interval and a count.

With these, a star visited at log-spaced times is one request, and so is a field
revisited every six hours.

Each scheduler treats the windows in its own way:

* The planners place each visit within its window and skip a window that no
  placement can meet.
* Local search and CP-SAT keep every visit in its window.
* Dispatch selects the target only while an unvisited window is open. Dispatch also
  looks ahead to timed requests (see `Timed requests in dispatch`_). For that
  look-ahead, a window that is still to open counts as a request opening.

A plan entry's ``visit`` says which window the entry observes in. The target records
the windows in which its visits have collected science, and each visit is credited
to its entry's window. A visit that runs past the start of the next window is
therefore still counted in its own window. Windows need only be spaced so that each
visit can begin within its own window.

A window can give its visit a pointing and a roll request of its own:

* ``ra`` and ``dec`` set where the visit points, for a visit that points somewhere
  other than the target's position, as the visits of a moving target do.
* ``roll_deg`` or ``roll_offset_deg`` replaces the target's roll request for that
  visit.

The visit is then observed at the window's pointing with the window's roll, and it
is visible when that pointing is visible. This holds in dispatch and in every
planner. The steps of a sequence visit are taken from the same pointing and roll. A
window that sets none of these fields observes the target's own request.

A visit can be cut short by a target of opportunity or by an emergency dump. While
its window is still open, the visit is taken again whole. The window counts as
visited only once a whole visit's science has been collected in it, although the
science collected before the cut still counts in the totals. If the window closes
before the visit can be taken again, the window is missed, dispatch logs the miss,
and the target goes on to its later windows.

Moving targets
^^^^^^^^^^^^^^

A target that moves across the sky, such as a Solar System body, is a
:class:`~conops.targets.MovingPointing`. There are two ways to say where it is:

* Set ``track`` to the target's position at a series of times, as a list of
  :class:`~conops.targets.TrackPoint`. :func:`~conops.targets.track_from_table`
  builds the list from a table of times, right ascensions and declinations. The
  position is interpolated between the track's points.
* Set ``body`` to the name or NAIF id of a Solar System body. The track is then
  looked up over the ephemeris through rust_ephem, from JPL Horizons by default or
  from a SPICE kernel (``spice_kernel``, ``use_horizons=False``). The lookup happens
  when the target's visibility is first computed, and the looked-up track travels
  with the pointing from then on.

A program bound with ``refresh_tracks`` looks up each body's track again. Where the
lookup fails, the program keeps the track it was given, so a pasted track stands in
when the service is down or refuses the body. The tracks that were looked up are
saved with the program.

The target's visibility is computed along the track. Dispatch and the planners both
point the spacecraft at the position where the slew to the target ends. To find it,
the slew is timed first to the target's position at the start of the slew and then
to the target's position at that slew's end. A planned entry carries that position,
and the executor takes up the track from there.

While the target is observed, the attitude follows the track at the track's rate, in
the same way that a ground contact's profile is followed, and it comes to rest when
the observation ends. The ACS reports ``SCIENCE`` throughout. The plan validator
expects the track's position at each sample, and the planners' battery ledger counts
the attitude along the track.

A plan entry records both the position at the slew's end and the track itself. A
plan replayed without the track observes the entry as a fixed pointing.

Asteroids are named by their NAIF/SPK IDs. The `extended NAIF numbering scheme
<https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/naif_ids.html>`_
assigns numbered asteroids IDs based on 20,000,000 plus the asteroid number, so
Hektor is ``body="20000624"``. The number ``624`` on its own is Hektor's permanent
asteroid number, but it is also a moon's NAIF ID. COASTSim preserves the identifier
through lookup and program serialization. The upstream ``rust_ephem`` adapter
translates asteroid NAIF IDs into the `Horizons designation-search syntax
<https://ssd-api.jpl.nasa.gov/doc/horizons.html>`_, so programs do not need to carry
that service-specific syntax.

The translation is included in ``rust_ephem`` 0.16.1, which is the oldest release
COASTSim installs with, so live small-body lookup needs no further setup. To retain a
track for offline execution, fetch it once using
``Program.bind(refresh_tracks=True)`` and save the program.

Visibility over part of the ephemeris
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calling ``visibility()`` on a pointing without bounds computes visibility over the
whole ephemeris. Queue intake and the planners use it this way.

``Pointing.visibility(begin=..., end=...)`` evaluates only the selected interval.
Bounds accept UTC datetimes, ISO strings or Unix seconds. The interval is evaluated
on the original ephemeris grid, with the adjacent samples included to preserve
boundary semantics.

Bounded queries accumulate the windows they find on the request:

* A query contained in an interval that is already known reuses those results.
* A wider or disjoint query evaluates only the missing intervals, with a grid-sample
  halo around each interval.
* Changes to pointing, track, roll, ephemeris or configuration invalidate the
  bounded results.
* Unknown intervals are not assumed visible.

Calling ``visibility()`` without bounds afterwards restores the whole-ephemeris
behavior.

For a moving target that names a ``body``, the live body lookup follows the
backend's whole-ephemeris ``get_body`` API, so the bounds limit only the constraint
evaluation. Saved tracks require no live lookup.

Timing budgets for one target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A target's ``own_timing`` replaces its category's setup, cleanup and handoff budgets
for that target alone, calibration steps included. For example, a long dark exposure
after one deep field is a cleanup step in that target's own timing, with its own
mode and frames, while the other targets of the category keep the category's
budgets. The planners, dispatch and the executed plan all take the timing from the
target, and the plan records it on the entry.

Instrument modes
^^^^^^^^^^^^^^^^

A target's ``instrument_modes`` names the instruments its observation uses and the
mode each is in, by name, such as ``{"camera": "SCIENCE"}``. A sequence step can
name its own modes. Instruments that are not named idle for the observation. A
target that names no modes uses the instrument it names with ``instrument_name``,
or every instrument with modes, in ``SCIENCE``.

The modes set two things:

* They set what the payload draws while the target is observed, in both DITLs and
  in the battery ledger the planners keep. A request that uses one instrument of
  several therefore costs the battery less than one that uses them all.
* They set what the observation generates on the recorder. Each instrument used
  generates its mode's science data, or the instrument's own where the mode sets
  none. An instrument with modes that is not used generates nothing (see the
  instrument modes under :doc:`configuration`).

The modes a target names, both its own and each sequence step's, are checked against
the payload when the target is made. The check includes the payload's
``concurrency`` rules on what may observe at once (see :doc:`configuration`). A
target that names no modes is checked later, when it is added to the dispatch queue
or given to a scheduler, and it is checked as using every instrument with modes.

The plan records the modes on each entry. A Target of Opportunity names its modes
with the ``instrument_modes`` argument of ``submit_too``.

Roll
^^^^

A target is observed at the power-optimal instrument roll unless it asks for
another. Where the science constraint depends on the roll, as a stray-light patch
written as a field-of-view polygon does (see :doc:`configuration`), the optimum is
the best power roll that keeps the constraint clear, when any roll does.

A target asks for another roll in one of two ways:

* ``roll_offset_deg`` is added to the power-optimal roll.
* ``roll_deg`` is the roll itself.

For a dispatch target, these are given as ``queue.add(..., roll_offset_deg=...)`` or
``roll_deg=``.

Dispatch and the planners observe the target at the requested roll where the
constraints allow it, and the plan entry's ``roll`` records it. The ACS's
``max_roll_from_optimum_deg`` bounds both kinds of request. An offset beyond it is a
configuration error. A target whose absolute roll is beyond it at some time is not
selected or placed at that time, but it may be later, as the optimal roll moves.

How a plan is built
-------------------

A plan is built in four steps. The steps are summarized in this list, and the
subsections after it give the detail.

1. **Locked entries** (the ``locked`` argument) keep their collection windows and are
   placed first. Only the slews into them are recomputed. Use this to pin
   placements and re-plan around them.
2. **Ground passes** are predicted from the configured ground stations, with the
   configuration's random seed, so ``DITL`` predicts the same passes when it executes
   the plan. Each pass is reserved on the first tracking profile the spacecraft can
   reach safely. Pass ``include_passes=False`` to plan science only.

   **Scheduled activities** (the ``activities`` argument, a list of
   :class:`~conops.targets.ScheduledActivity`) are reserved the same way. Each is a
   fixed block from its start for its duration, so no science is placed across it.
   See `Scheduled activities`_.
3. **Requests** are sorted by merit value at the start of the horizon. The merit
   value is the base merit plus the urgency, cadence and completion-deficit terms
   (see :doc:`configuration`). A request whose deadline is close gains urgency and
   so is placed ahead of flexible requests with higher base merit. Cadence and
   completion deficit follow the plan as it is built (see `Cadence and program
   shares in plans`_).

   Each request is split into snapshots of up to ``ss_max`` seconds and never
   shorter than ``ss_min``, until its ``exptime`` is used or it no longer fits.
   There is one exception to the ``ss_min`` limit. Once any of the request is
   placed, or the target has collected anything, its remainder is placed however
   short it is, in the same way that dispatch finishes such a target. A remainder
   shorter than the simulation step is treated as rounding and ends the request.
4. **Each snapshot** goes in the earliest slot where every check ``DITL`` and the plan
   validator will apply passes:

   * the slew starts on a simulation step, after the previous entry ends;
   * the slew path clears the ``SLEWING`` attitude-constraint scopes;
   * the target is visible when the slew starts (ACS will not start it otherwise),
     or, with ``slew_before_visible`` set in the ACS configuration, when the slew
     arrives;
   * the held attitude clears the ``SCIENCE`` scopes for the whole observation,
     including setup, cleanup and handoff time; an observation that would run into
     a constraint is shortened, but never below ``ss_min``;
   * any wait at the previous attitude clears the ``IDLE`` scopes;
   * collection starts no earlier than the target's ``earliest_start`` and by its
     ``deadline``;
   * the next entry can still be reached on time, with its own slew and wait
     checked the same way. When the next entry is a scheduled activity, no slew is
     made to it, and the snapshot's attitude must be one that ``IDLE`` holds
     through the activity.

Requests left with at least ``ss_min`` of exposure unplanned are listed in
``scheduler.unplaced``. Every placement and rejection is logged to ``scheduler.log``.
The input targets are not modified.

Moving snapshots later to make room
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A request whose snapshots straddle a time its target is hidden gets its next
snapshot the moment the target rises. By default ACS starts a slew only once its
target is visible, so the only way to be on the target at that moment is to stay on
it while it is hidden. Anything else placed in that time would get to the target
late.

The scheduler therefore moves snapshots to use that time. If a new snapshot would
fit a gap, but the science snapshot after the gap could then not be reached in time,
the later snapshot is moved later to make room. Up to three more snapshots packed
behind it are moved with it, as long as the entry after them can still be reached.

Moving a snapshot can cut it short where its target's visibility ends. That is
allowed only if the new snapshot's value-weighted seconds exceed those lost.
Exposure cut short is left for the request to plan elsewhere.

The same move fills the time before a scheduled activity. The activity holds the
attitude of whatever precedes it, so science placed before it changes where the
science after it slews from. Under a long thermal settle, that later science can no
longer be reached in time. It is then moved later, and the activity holds the new
snapshot's attitude.

Some entries are limited in how they move:

* Ground passes and locked entries are never moved.
* A cadence target's visit never moves earlier than it was. It also moves no later
  than leaves its next visit a cadence after it, so the target's visits stay a
  cadence apart.
* A visit in a window of its own moves only if it is the target's latest.
* A cadence visit placed at the opening of its visibility window is otherwise an
  anchor that nothing can precede, since a slew into it from another attitude would
  start before the target is visible.

Slewing before a target is visible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting ``slew_before_visible`` in the ACS configuration lets a slew start early
enough to arrive as its target rises, rather than waiting for it to rise. That saves
a slew's time each time a target rises, and the snapshot after a gap no longer needs
to move to make room.

Scheduled activities
^^^^^^^^^^^^^^^^^^^^

A scheduled activity is a block of time in which no science is observed. A
maintenance window, a calibration that needs the spacecraft still and an engineering
block are all activities. Each is a :class:`~conops.targets.ScheduledActivity` with
a start and a duration. The spacecraft holds one attitude for the block in ``IDLE``
mode. The mode is ``DESAT`` for a thruster momentum dump (see `Thruster momentum
dumps`_) and ``SAFE`` for an engineering safe hold (see `Engineering safe holds and
power-positive attitudes`_).

The planners take the activities in their ``activities`` argument and reserve each
one as a fixed block, as they reserve ground passes. ``DITL`` commands the plan's
activity entries, and the plan validator checks that no science, contact or charge
ran during one. Dispatch takes the same list (see `Scheduled activities in
dispatch`_).

An activity's ``instrument_modes`` names what it uses, such as a calibration lamp's
instrument, and that draw is counted during the hold. An activity that names none
draws every instrument's idle power. Its ``data_gb`` is what the activity puts on
the recorder when it begins, such as its frames. That data is reported as generated
at that step, and the recorder ledger and the rolling replan's forecast count it as
well.

:func:`~conops.targets.activity.periodic_activities` builds a list of activities at
a fixed interval from a first start. It takes either a count of activities or a
time, in which case it builds as many as begin by that time. Each activity has the
same modes, data, ``fixed`` and slide.

Mandatory activities reserve their windows before optional ground contacts. A
contact can be acquired later, or omitted when it cannot fit. If an activity cannot
be reserved, the planners raise :class:`~conops.schedulers.ActivitySchedulingError`
instead of returning a plan that omits it. The error names the mandatory activity,
its start window and the failed reservation. An activity reserved at or beyond an
internal horizon's end is still deferred to a later plan.

Fault safe mode is never scheduled. It is entered through fault management alone.
An engineering safe hold uses ``SAFE`` only for its own length and does not latch it.

The attitude a scheduled activity holds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unless it asks for an attitude of its own (see `Requesting an attitude for an
activity`_), an activity holds the attitude of whatever precedes it in the plan. The
science before it ends at the block with no slew back, the block holds that
science's attitude, and the science after it slews from there. After tracking a
moving target, the handoff uses the final resting attitude after braking, so the
activity does not command a return to the body's arrival position.

This limits what can be placed before an activity. A snapshot whose attitude cannot
be held through the block must leave time for a safe ingress slew. A snapshot from
which the activity, or the block after the activity, cannot be reached is not placed
before it.

With nothing before it, the activity holds the attitude the plan starts from, or the
one that the fixed block before it leaves. That attitude may not be safe to hold
through the activity and the idle time after it. The planner then searches a
deterministic list of safe-hold candidates, which it shares with dispatch:

* the initial idle attitude;
* nearby directions;
* array pointing;
* Earth-opposite and Sun-opposite directions;
* a sky grid;
* roll alternatives, where roll affects safety.

A candidate must be safe for the whole hold and reachable by a path-validated slew,
with time for the following block. Before the first command, the initial pose must
also remain safe until ingress. The planner commands the ingress slew before the
first keep-out violation and validates the destination through the wait and the
whole hold. If no candidate fits, the planners raise ``ActivitySchedulingError``.

When an activity needs an ingress slew, its entry begins with that ingress, and
``activity_hold_begin`` marks the start of the full calibration after ingress. A
sliding hold (see `Sliding and fixed activities`_) retains its full duration after
arrival, and its settled time and changed predecessor are rechecked.

Planned entries mark ``activity_requires_stationary``. Validation then rejects
slewing during the full hold. It also rejects ``SAFE`` there, unless the activity is
an engineering safe hold. Entries without the mark, which come from
older plans and from flexible dispatch holds, are not checked this way and retain
their existing recovery semantics.

Requesting an attitude for an activity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ScheduledActivity.attitude`` can request an explicit body pose with
:class:`~conops.config.activity_attitude.ActivityAttitude`. Its default
``kind="inherited"`` keeps the inherited and safe-hold policy described above. A
``kind="fixed"`` pose supplies ``ra_deg``, ``dec_deg`` and ``roll_deg`` in the same
GCRS physical body-attitude convention as plan entries. This fixes the pose. The
separate ``fixed`` activity flag fixes the activity's start time (see `Sliding and
fixed activities`_).

Dispatch prepares the required ingress as the activity's start window approaches.
Its look-ahead includes the current observation's cleanup and handoff budget,
including when collection has not started yet. What dispatch does to the current
observation depends on how far it has got:

* Dispatch can cancel a pending collection before it begins.
* Once science has executed, only future collection is shortened, and the
  observation's full teardown is paid before ingress.
* Already collected science is never reclassified as cleanup to fit the hold.

The planners reserve the ingress before the full hold, and search a sliding window
for a feasible time at the requested pose. An unsafe required pose is reported
explicitly and is not replaced with another attitude. Replans retain the validated
pose and hold, and the next observation departs directly from it.

``periodic_activities`` accepts the same ``attitude`` argument, and ``Program`` JSON
preserves it.

Engineering safe holds and power-positive attitudes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A recoverable engineering safe hold sets ``safe_hold=True``. Its stationary part
uses ACS ``SAFE`` power, constraint scopes and instrument modes. It returns to
scheduling at its end without latching fault safe mode. ``periodic_activities``
accepts ``safe_hold`` too. To select a power-positive hold, use::

   attitude = ActivityAttitude(
       kind="power_positive", minimum_margin_w=100,
       prefer_ecliptic_poles=True,
   )
   hold = ScheduledActivity(
       name="Engineering safe hold", begin=start, duration_seconds=36000,
       safe_hold=True, attitude=attitude,
   )

The selector first tries the mean J2000 ecliptic poles for the declared spacecraft
boresight, with one-degree rolls. It then tries a deterministic coarse sky/roll
grid. A candidate must pass two tests:

* It must clear the hold mode's constraints at every ephemeris sample, including
  eclipse.
* It must generate more than the bus plus payload load and ``minimum_margin_w`` at
  every sunlit sample.

Eclipse still drains the battery. An entirely shadowed hold cannot establish a
sunlit power margin.

The selector has these limits:

* It is a sampled candidate search. It does not prove that no continuous solution
  exists.
* It currently supports fixed body-mounted arrays. Finite panel drives are
  explicitly rejected because their ingress drive-state forecast is not yet
  included.
* Thruster firing poses remain inherited or fixed. The power-positive policy does
  not define thruster geometry.

The planners also check ingress and surrounding reservations. Dispatch checks one
pose over the union of possible holds in a sliding window, and conservatively
rejects a window that needs different poses. Infeasibility raises
``ActivitySchedulingError``. The requested policy is not silently changed and the
hold is not omitted.

Sliding and fixed activities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two options control when an activity begins relative to its nominal start.

An activity that cannot move, such as a maintenance window, is ``fixed``. It begins
at its start. If a slew is under way then, the slew finishes inside the activity.

An activity with ``slide_seconds`` may begin up to that long after its start. The
activity keeps its length wherever it begins. Dispatch treats such an activity as
follows:

* An observation under way at the activity's start runs on, and the activity begins
  when the observation ends. This includes a target of opportunity's observation
  and an indivisible visit.
* If the observation is still under way at the activity's latest start, the activity
  begins then and ends the observation.
* An observation selected before the activity may run until the activity's latest
  start. A response whose slew would cross the activity's start is therefore
  observed before the activity.
* Nothing starts while the activity waits.

The planners let the science before a sliding activity run into its window, and hold
the activity from when that science ends, within the window. The block is first
reserved at its latest start. As the plan is built, the block settles to follow
whatever is placed before it, and the time after its settled end is open to what
follows. Where its window permits, a sliding hold starts early enough to finish
before the physical run ends.

An activity can come due while another activity is under way. Whether it is
``fixed`` or not, it begins when the other activity ends and keeps its length.
Dispatch begins it then. The planners reserve it after the activity already on the
timeline, and settle it back to its own start where that activity slid and ended
sooner.

Scheduled activities in dispatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dispatch takes the same list of activities as the planners. When an activity
begins, dispatch ends the observation under way and marks it interrupted so that it
is revisited. It holds the attitude it has until the activity ends, in ``IDLE``, or
in ``DESAT`` for a dump and ``SAFE`` for an engineering safe hold.
Dispatch also starts no observation whose slew or collection would run into the next
activity.

A slew may be under way when an activity is due, such as a slew into a charging
attitude. The activity is then delayed to the slew's end, and it runs its full
length from there. A ``fixed`` activity is not delayed (see `Sliding and fixed
activities`_).

Activities in partial runs
^^^^^^^^^^^^^^^^^^^^^^^^^^

A run can cover only part of the time that a list of activities spans. For a partial
dispatch run:

* An activity is expired when its latest start plus its duration is at or before the
  run's beginning, and it is not executed.
* Activities starting at or after the run's end do not block dispatch either.
* Activity obsids are assigned from the full input list and remain unchanged when
  the run's interval excludes earlier activities.

An activity can also be under way when a partial run begins. Dispatch and the
planners resume a fixed activity that began earlier, and hold only until its
original end. A sliding activity is handled without earlier execution state. If its
window is still open, it begins a whole hold at the run's start. If the window has
closed, the run assumes the latest permitted start and resumes the remaining hold.

These assumptions do not reconstruct earlier attitude, recorder contents or
momentum. The run's configured initial conditions still apply. A saved plan carries
the settled hold bounds when an exact earlier reservation is available.

Activities in rolling replans
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An activity reserved within a rolling planning horizon keeps its full hold even when
it ends beyond that horizon. Replans commit the remaining hold with the activity, so
science cannot use its tail and its data is not generated again. The physical
simulation end and the ephemeris end still bound the hold. An internal planning
cutoff does not.

A replan handles an activity according to its state:

* An uncommitted activity whose nominal start has passed is retained while its slide
  window remains open. The replan reserves a new full hold after the committed work,
  on an execution step, and does not treat the activity as resumed.
* Committed and completed activities are excluded by identity so they are not
  repeated.
* An unstarted activity whose window has already closed is reported with
  ``ActivitySchedulingError``.

Hold metadata on activity entries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An activity's plan entry can cover less than the full hold, for example in a partial
run. The entry therefore carries both sets of bounds, and fields that relate them:

* ``activity_hold_begin`` and ``activity_hold_end`` bound the full hold.
* ``begin`` and ``end`` bound the portion in this run.
* ``activity_partial`` reports missing time at either boundary.
* ``activity_finishes`` says whether the interval reaches the full hold's end.

A resumed activity logs ``resumes`` and has no new ``fixed_data_gb``, because its
start data is not generated again. A hold cut short at the run end logs ``stops
before completion``. A partial dump unloads the wheels only if its remaining hold
reaches the full end. That rule applies in execution and in both the planner's and
the rolling replan's momentum forecasts. Older plans without hold metadata continue
to use their entry bounds as the full hold bounds.

Emergency recharge during an activity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This subsection applies to dispatch. A battery alert can be raised while an activity
block is under way, and dispatch must then decide whether the block can finish
before the spacecraft recharges.

The alert can wait for the block when a step-by-step forecast keeps the battery
above the physical discharge floor through the remaining hold, any eclipse wait, and
the actual slew to useful charging. The forecast includes the block's modes, panel
generation, saturation and executed finite-drive state. It does not mutate the live
drive or charging state. The alert threshold is distinct from this safety floor.

If the block cannot finish safely and a valid emergency charging manoeuvre can be
dispatched, dispatch closes the block's executed segment before that manoeuvre. The
segment's ``activity_interruption`` records the time, the reason
``emergency_recharge``, the unexecuted seconds and the disposition. What happens
after the battery alert clears depends on the kind of activity:

* An ordinary hold resumes its remaining duration.
* A dump restarts its full firing duration, so partial firing never resets the
  wheels.

Resumed segments carry ``activity_resumed=True``, retain the original activity ID
and do not generate the once-only data again. ``activity_partial`` identifies the
fragments. The final fragment can also have ``activity_finishes=True``.

A block can also come due while emergency charging is already under way:

* The block waits, with its full duration retained.
* A critically low battery is checked before beginning a new block or generating its
  start data.
* Deferral and resumption may exceed the original start window. This is logged as a
  safety override.

A paused block waits through an eclipse that ends charging early, until recharge has
cleared the alert. It also waits through a ground contact already under way. If the
physical run ends first, interruption metadata retains the unfinished work, and no
zero-length execution entry is invented.

Charging and its actual ingress are exported as separate ``CHARGE`` entries, with
their actual instrument modes. Saved execution plans replay those fragments and
charges. Planned runs continue to execute only their recorded commands and do not
add unplanned charges.

Thruster momentum dumps
^^^^^^^^^^^^^^^^^^^^^^^

A thruster momentum dump is an activity that the schedulers add themselves. Dispatch
calls for one when the stored momentum reaches the planning level of the wheels'
capacity (see the stored momentum under :doc:`configuration`). The planners reserve
one where their forecast reaches that level (see `Planning within the wheels`_). The
dump's entry is marked ``desaturation``, and the wheels are unloaded when it ends.

Planning faster when many requests cannot fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most planning time in a crowded horizon goes to requests that end up unplaced. A
snapshot fits a gap, but the entry after it cannot then be reached, so each later
start in the gap is tried in turn, each with a full search for the slew onward.

``successor_retries`` limits how many later starts are tried after that happens
before the gap is given up (3 by default). This plans much faster in a crowded
horizon, but can miss a fit late in a gap. Set the scheduler argument to None to try
every start. In a mission configuration's ``scheduler`` section the same setting is
written ``successor_retries: unlimited``, because a null value there means the
default. Rolling replans use the default unless the configuration or
``scheduler_options`` sets it.

What the gap search reuses
^^^^^^^^^^^^^^^^^^^^^^^^^^

A gap is searched for many requests, each from the attitude held before it, and each
request is tried from many starts a step apart. Much of that work repeats, and the
scheduling context (see `The scheduling context`_) keeps the results that repeat.

Held attitudes and visibility:

* The context keeps where a held attitude first violates a mode over a span, stored
  by the span.
* The context keeps each target's visibility windows. The scheduler keeps those of
  each candidate snapshot, by target, pointing and, where the roll enters them,
  roll. The constraints are therefore evaluated over the ephemeris once per target
  and not once per candidate, which matters most for a moving target.

Slews:

* With the quaternion slew algorithm, the context keeps the path of each slew
  between two attitudes. The path is predicted once and retimed for every start.
  With a thermal settling table, only the settle, which follows the Sun, is computed
  per start.
* A constraint-avoiding path is decided at the slew's start rounded to the ephemeris
  step. It is therefore predicted once per pair of attitudes and step, and retimed
  within the step.
* The samples along each arc that a constraint-avoiding path checks are evaluated in
  one call to ``rust_ephem``. The steps of a held attitude are evaluated the same
  way when the context scans a hold.
* A slew is built without validating the configuration it holds.

The battery ledger:

* It reads eclipse at its grid steps from the horizon, which it samples once.
* It remembers the bus's and payload's draw by mode, eclipse and instrument modes.
* It samples a slew's path at all of an addition's steps together, as the constraint
  check along a slew does.
* Where no panel is driven, it computes the panels' power at all of an addition's
  steps at once.

The local search decodes a trial plan for every change it tries, so these savings
decide how many changes a search of a given length tries. The CP-SAT scheduler
computes the slew table of a chunk's tasks for every pair at once. It decodes the
rest of the plan it started from once per chunk, and carries the decoding to the next
chunk while chunks keep that plan.

Starting a rolling run from a plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A rolling run can start from a plan. Pass the plan as
``RollingHorizonDITL(start_plan=...)``, or as
``create_ditl(config, ..., start_plan=...)`` for a planned or rolling run. The
local-search or CP-SAT scheduler of each replan is then given the entries of that
plan that lie inside the horizon it plans, as a ``start_plan`` the scheduler
improves on. A dispatch run's executed plan is the natural plan to start from. Each
replan record says how many entries seeded it (``seed_entries``).

Local-search and CP-SAT replans also consider the previous rolling plan's
uncommitted suffix. It is decoded again against the actual handoff state,
remaining exposure, completed and committed windows, budgets and mandatory
activities. A feasible incumbent can therefore survive a replan whose fresh
search would lose it. The external dispatch seed remains a separate contender.

Passing ``additional_start_plans`` explicitly in ``scheduler_options`` overrides
this automatic incumbent, and an empty tuple disables it. Automatic seed injection
applies to optimizer classes, which leaves callable scheduler factories compatible.

Cadence and program shares in plans
-----------------------------------

The cadence and completion-deficit merit terms depend on what has already been
observed. :class:`~conops.ditl.QueueDITL` evaluates them against what it has
executed. A planning scheduler has not executed anything yet, so it evaluates them
against the plan it is building, added to the science already collected. Time budgets
are counted against the plan in the same way. The subsections below describe each
rule. When neither ``completion_deficit_weight`` nor ``cadence_weight`` is set, the
completion-deficit and cadence rules do not change how a plan is built.

Completion deficit
^^^^^^^^^^^^^^^^^^

The completion-deficit term applies when ``completion_deficit_weight`` is set and
programs have a ``time_share``. Each program's share then counts the science its
targets have collected (``collected_seconds``) plus what the plan has given them so
far. The scheduler re-ranks the requests after every snapshot, so a program that is
falling behind its share moves up.

A program's share at a given snapshot counts only the science that comes before that
snapshot:

* The values recorded on the plan's entries, and the exact decoding of a plan into a
  timeline, use the collection that precedes each snapshot's decision, taken in
  chronological order.
* Science locked later in the horizon does not count as already delivered. Its
  exposure still reserves its target and the category and group budgets immediately.
* Each local-search state carries forward the science delivered by the part of the
  plan before it.
* CP-SAT solves the horizon in chunks (see `Optimizing plans with CP-SAT`_). It uses
  the delivered shares as they stand at the start of each chunk, and it scores the
  proposals it decodes against the collection that actually precedes them.

Observation groups
^^^^^^^^^^^^^^^^^^

A target can belong to an observation group (``observation_groups``, see
:doc:`configuration`). The group changes how the target is planned in three ways:

* The group's window narrows the target's earliest start and deadline.
* The group's time budget caps the target's snapshots, alongside the budget of the
  target's category.
* Where the group prefers contiguity, the priority-first scheduler goes on with the
  group's other targets once it has placed one of them. The group's targets are
  therefore planned one after another, as dispatch observes them.

The local-search and CP-SAT planners start from that order and keep the window and
the budget.

Time budgets
^^^^^^^^^^^^

A category with ``time_budget_seconds`` collects no more than that many seconds over
the run. The budget already used includes what the category's targets had collected
before the plan. Each snapshot is capped to what the plan leaves of the budget, and
nothing more is placed once less than a minimum snapshot is left. A visit of a
sequence (see `Observation sequences`_) is placed only whole. Dispatch and the
long-range allocator honour the same cap.

Cadence
^^^^^^^

With ``cadence_weight`` set, a target in a category with ``cadence_seconds`` is
visited no sooner than that long after its last visit. The last visit can be one
already in the plan or one already collected (``last_collection_time``). Keeping
visits this far apart means every visit has the full cadence value.

Dispatch follows the same rule. It does not select such a target until a cadence
after its last visit, so the target is not revisited as soon as nothing outranks it.

The planners apply the spacing as follows:

* The priority-first scheduler and local search place each visit at the earliest fit
  once it is due.
* CP-SAT keeps the same spacing between a target's candidates. With an explicit
  positive ``earliness_weight``, CP-SAT also discounts a visit that waits after it
  falls due. A visit falls due a cadence after the visit before it. The discount
  grows with the time the visit waits and reaches ``earliness_weight`` of the visit's
  value when the visit is a whole cadence late, which is the most it can be. The
  default applies no value discount.

Committed observations in rolling replans
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When :class:`~conops.ditl.RollingHorizonDITL` replans, some observations are already
committed but not yet fully collected. These count too:

* Their exposure counts towards their program's share (``reserved_seconds``).
* Their collection end counts as the target's last visit (``reserved_visits``), so a
  new plan does not revisit a target straight after a committed visit.

For requests with visit windows, the index of the last window that is still committed
is handed to the new plan as well (``reserved_visit_windows``). Planners combine it
with the history of completed visits, so a later commanded visit is not placed a
second time merely because an earlier visit has already completed.

The index is recorded because the window cannot always be worked out from times. A
sequence's calibration or slews can run past the opening of the next window, and
subtracting the science duration from the visit's finish does not then identify the
window reliably. Recorded indices preserve the identity of such a sequence. Callers
that use a scheduler on its own and provide only ``reserved_visits`` retain the
time-based fallback.

Only whole committed visits reserve a window. A pending attempt that is cut short can
still be followed by a whole retry. Passing an explicit empty window map says that
none of the committed attempts will complete a window.

Windowed visits cut by a Target of Opportunity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a ToO cuts a windowed visit, planning reopens that window for a whole retry
while the window is still available. The partial science keeps its credit. Cleanup
and handoff still finish before the replacement. If a later replan is made before the
cut has been flown, it keeps that pending retry. A proposed cut that is rejected
leaves the request history unchanged.

An accepted cut appears in the ``interruptions`` of the request and of the cut entry
only when it has been executed. Its timestamp is the science cutoff, which can
precede the end of cleanup. When a planner plans a future interruption, it uses a
separate request history, so the spacecraft's actual collected science and completion
state stay accurate.

Planning within the recorder
----------------------------

Every scheduler keeps the onboard recorder below its planning limit. The limit is
``recorder.planning_limit``, a fraction of ``capacity_gb`` (see
:doc:`data_management`). By default it is the recorder's ``red_threshold``, which is
where the red alert fires, so a plan stops just short of the red. Set it lower to
keep margin, or to ``None`` to plan without regard to the recorder.

The recorder fills between ground contacts and empties during them, so its volume is
highest just before each contact and at the end of the horizon. The reserved contacts
cut the horizon into segments, and the plan is checked at every segment's peak. The
peaks are worked out as follows:

* A segment's peak is the sum of three amounts: what was left after the contact
  before it, the housekeeping data generated in the segment, and the science data of
  the snapshots placed in it. A snapshot's science data is each instrument's rate for
  the snapshot's collection time plus the instrument's fixed per-observation amount.
  Where the target sets its own rates, those are used.
* Each contact removes what its rate and length allow, and no more than is on the
  recorder. What a contact cannot remove carries over to the segments after it.
* Housekeeping is counted at the nominal rate throughout, and at the ``PASS`` rate
  during spacecraft tracking. The nominal rate is used elsewhere because the plan does
  not know when the spacecraft will be in other modes. Housekeeping is counted on the
  simulation's step grid, as the simulation generates it: each step counts whole, at
  the rate of the mode it starts in.
* When a pass requires a spacecraft slew, ACS flies that slew in ``PASS`` mode too.
  The slew adds the difference between the two rates for the steps it is under way.
  How long the slew is depends on where the spacecraft is before it. The schedulers
  therefore add this amount with the contact, as they add science, and recompute it
  whenever a snapshot placed before the contact changes the slew. CP-SAT counts each
  contact's slew as it is in the plan CP-SAT starts from, and a solution that
  lengthens a slew is checked when it is decoded.
* A contact downlinks for the part of its link that is inside the horizon. The
  simulation stores a step's data after the downlink, so what the step in which the
  link ends generates after the link stays on the recorder.

A snapshot is placed only where every peak from its segment onwards stays within the
limit. It is never shortened to fit. A snapshot that does not fit before a contact is
placed after one, or not at all, and the log says so. The schedulers differ in how
they decide which science waits:

* The priority-first scheduler places higher-merit requests first. When data runs
  short, it is therefore the lowest-merit science that waits, whatever its data rate.
* Local search then trades snapshots in and out on merit within the same limit.
* CP-SAT has the recorder in its model, so the solver trades high-data science
  against low-data science directly (see `Optimizing plans with CP-SAT`_).

:class:`~conops.ditl.RollingHorizonDITL` starts each plan from the recorder's actual
volume, so a contact that downlinked less than planned is made up for at the next
replan. A plan takes effect after the commit lead time (see `Rolling-horizon
replanning`_). It therefore starts from what the committed activities leave on the
recorder by then: what they go on to generate, less what a committed contact
downlinks.

The arithmetic is not particular to the recorder. Anything a spacecraft
accumulates between the events that remove it, and must keep within a limit
meanwhile, has the same shape. :class:`~conops.schedulers.ConsumableLedger` holds
that arithmetic (segments between removals, accrual, removals that take no more
than is there, and the check at every later peak),
:class:`~conops.schedulers.RecorderLedger` maps the recorder onto it, and
:class:`~conops.schedulers.BatteryLedger` the battery (see `Planning within the
battery`_). Stored momentum is the one exception. It is a vector whose build-up
depends on the attitude flown at every moment, so it is forecast along the plan
and is not booked by segment (see `Planning within the wheels`_).

Contact booking and antenna coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Body-fixed tracking passes are booked by :class:`~conops.simulation.PassTimes`.
Omni and gimballed antennas use continuous network selection by default. Whole-pass
booking is available for them through ``contact_policy="scheduled"`` (see
:doc:`communications`).

A contact that requires no spacecraft tracking does not block science. Its usable
link spans downlink at the station's rate while housekeeping stays at the nominal
rate. A snapshot that does not fit before such a pass is tried after it within the
same timeline gap.

The schedulers and :class:`~conops.ditl.RollingHorizonDITL` take the coverage from
the configuration, or from the ``coverage=`` argument where the caller already has
it. A rolling run builds the coverage once and gives it to every plan.

The antenna can be re-pointed within contacts (see :doc:`communications`). Where it
is, a target whose category sets ``antenna_repoints: forbidden`` has its visibility
windows cut at each repoint. Every scheduler then places the target's snapshots
between repoints, as it places them between constraint violations. Dispatch starts
such an observation only when its minimum snapshot fits before the next repoint, and
ends the observation at that repoint.

The recorder in dispatch
^^^^^^^^^^^^^^^^^^^^^^^^

Dispatch (:class:`~conops.ditl.queue_ditl.QueueDITL`) keeps within the recorder
too. When it selects the next target, it works out how much the recorder can take
now. This room is the most data that keeps the volume under the planning limit at
every contact's start and at the run's end.

To find the room, dispatch carries the recorder through the rest of the run. Each
contact removes what its station's rate downlinks for its length, and never takes
the recorder below empty. A contact too short or too slow to empty the recorder
leaves the rest to the next contact, and the housekeeping generated between the two
has to fit as well.

Housekeeping is counted as the simulation generates it, for each whole step at the
rate of the mode the step starts in. Dispatch uses two rates:

* The waiting rate is the highest of the nominal, ``IDLE``, ``CHARGING`` and ``SAA``
  rates. The highest is used because the dispatcher can spend the time in any of
  those modes.
* The ``PASS`` rate is used for the slew into each contact, which ACS flies in
  ``PASS`` mode, where that rate is higher than the waiting rate. The slew is bounded
  as admission bounds it.

A candidate collects no more than the room holds. Two amounts are counted against the
room. The first is the candidate's science, at its own rates. The second is the
housekeeping that the candidate's own slew and its science mode generate beyond the
waiting rate, where the slew is the attitude-aware slew the candidate will fly. A
mode that generates less than the waiting rate is not credited with the saving, since
the spans are upper bounds.

A candidate that cannot collect its minimum snapshot within the room is passed over.
A snapshot shortened to fit stays shortened through admission, which asks for the
room again with the slew that is flown.

A Target of Opportunity can be admitted ahead of the observation under way, to start
once that observation's cleanup is over. It is admitted against the volume projected
to that time, which adds the science the observation still collects and the
housekeeping of the steps between.

Passes that require no spacecraft tracking have no ingress slew. Housekeeping stays
at the waiting rate, and each usable link span downlinks at its mean rate for its
length. Dispatch does not count on a contact that falls during the snapshot.

The log says how many candidates were passed over and how much room there was.

``config.targets.data_volume_weight`` adds a price on top of the limit, in merit
points per gigabit of science data. With the price, high-data science gives way to
low-data science before the recorder is full, and not only once it is full (see
:doc:`configuration`).

Planning within the wheels
--------------------------

Where the configuration tracks stored momentum (see the stored momentum under
:doc:`configuration`), every plan is forecast for it, and a planner whose spacecraft
has thrusters reserves momentum dumps where the forecast calls for them. The
priority-first, local-search and CP-SAT planners all plan within the wheels this way.

The momentum forecast
^^^^^^^^^^^^^^^^^^^^^

:class:`~conops.schedulers.momentum.MomentumForecast` integrates the configured
torque sources along the plan's timeline with the tracker the simulations use. It
covers the whole timeline:

* each slew, along its path;
* each observation, contact, charge and activity, at its attitude;
* the idle time between them, at the attitude held.

The forecast is sampled at the ephemeris step along a hold and at the configured
attitude step along a slew. Each thruster dump unloads the wheels as it ends.

A rolling-horizon run gives each replan the stored momentum it will have when its
horizon starts, with the committed entries and their dumps included, so the replan's
forecast starts where the wheels will be.

Where a dump is placed
^^^^^^^^^^^^^^^^^^^^^^

Where the forecast reaches the planning level of the capacity and the spacecraft has
thrusters, the planner reserves a dump. A dump is a fixed desaturation activity that
holds the attitude of whatever precedes it. The dump goes between observations, and
its place depends on where the level is reached:

* Where the level is reached between observations, in idle time, a slew or a
  contact, the dump goes at the crossing.
* Where the level is reached during an observation, the dump follows the
  observation, provided the wheels stay within the capacity through the dump's own
  length.
* Where the wheels would not stay within the capacity, the dump goes before the
  observation, which then starts from empty wheels. This requires that the
  observation alone fits within the capacity from empty.
* Where the observation does not fit from empty, the outcome depends on its kind. A
  visit is not cut, so a visit that does not fit is left out of the plan, with the
  reason logged. A plain observation that does not fit has the dump at the crossing,
  and the next pass places a shorter snapshot after it.

A dump can be configured to fire at a fixed attitude. Such a firing may wait until an
indivisible visit ends when its start has no fixed time or bounded slide window.
Fixed firing starts and the latest starts of bounded windows constrain whole-visit
admission, alongside calibration and maintenance deadlines.

Building the plan in passes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A dump changes the plan that follows it, so the planner builds the plan in passes.
Each pass forecasts the plan, reserves dumps and builds the plan again:

* A pass reserves a dump at the first crossing the forecast reaches. Where that dump
  overlaps nothing already on the timeline and has no fixed firing attitude, the pass
  goes on to the next crossing, found with the wheels unloaded at the dump before it,
  and reserves a dump there as well. A dump that overlaps an entry on the timeline, or
  that fires at a fixed attitude, is the last the pass reserves, because the work it
  displaces is placed again before later crossings are forecast. The plan is then
  built again once, from the first of these dumps.
* When the plan is built again, the planner keeps the science placed before the dump
  as it is and plans on from there with the dumps reserved so far. The next pass
  places from the first slew start that the pass before it let go, since nothing
  before that time has changed.
* The science after a dump moves a little when the plan is built with the dump. The
  next pass therefore checks the dumps against the plan as built, and replaces any
  dump that no longer answers its forecast.
* A dump found earlier than one already reserved replaces it and those after it.
* A plan reserves fifty dumps at most.

The passes share the spans of the timeline integrated so far. Each span is keyed on
its times, its attitude or slew, and the momentum it starts from. A pass therefore
integrates only from the dump it reserves onwards, and the plan before that dump is
read back.

Without thrusters the plan stands, and the log says where the forecast passes the
capacity. With thrusters, the log says where the forecast still reaches the capacity,
if it does.

Momentum carried during slews
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``slew_momentum_enabled`` counts the momentum the body carries while it slews. Where
it is set, the forecast checks what the wheels hold along each slew against the
capacity. A slew in which the wheels would reach the capacity is flown slower, at the
next of ``slew_rate_fractions``, and the plan is built again. At the slowest
fraction, a dump goes before the observation where its slew fits from empty wheels,
and the observation is left out where it does not (see :doc:`momentum`, Slews).

Observation sequences
---------------------

Some observations are a fixed series of steps instead of one pointing held for a
time. An example is a reference star, then the target, then the reference again,
each for its own duration and perhaps with its own instrument, with calibration steps
that take time but collect no science.

A :class:`~conops.targets.ObservationSequence` on a
:class:`~conops.targets.Pointing` lists the steps. Each step is a ``SequenceStep``
with:

* a name;
* a duration;
* an optional pointing and instrument;
* whether it collects science;
* whether it pays the setup budget;
* an optional ``roll_offset_deg`` on top of the target's own roll.

``visits`` says how many times the sequence is wanted. A sequence target's snapshot
is one visit's science, no more and no less, and its exposure is its visits' science
unless given.

Sequences in dispatch
^^^^^^^^^^^^^^^^^^^^^

Dispatch observes a visit step by step:

* The target is selected as any other target is. The time it needs counts the visit's
  calibration, the slews between steps that point elsewhere, and any step setup.
* The first step is the observation admitted, and it pays the payload's setup.
* When a step's time is up, the next step starts at once. A real slew is flown where
  the next step points elsewhere. The step pays no setup unless it asks for it.
  Cleanup and handoff come after the last step only.
* A step at the pointing of the step before keeps the power-optimal roll found there,
  as the planners do. Its roll offset is therefore a roll-only slew, and the same
  offset leaves the attitude where it is.
* Nothing is inserted inside a visit. A Target of Opportunity waits for the visit to
  end.

A step that cannot be observed cuts the visit short. This happens when the step's
pointing is not visible throughout, when its attitude or slew is constrained, or
when a ground pass falls in it. Any other termination of a step cuts the visit short
too. The cut is recorded on the target, which is revisited for the rest from its
first step.

Each step is its own plan entry. The entry has the target's obsid and its ``step``
name, and a calibration step's entry has ``calibration`` set. The target is credited
the science steps' collection and, separately, the calibration time. Calibration
steps generate housekeeping data but no science data, including no fixed science
readout amount.

Sequences in the planners
^^^^^^^^^^^^^^^^^^^^^^^^^

The planners place a visit as one block of steps.

The priority-first scheduler builds the chain of steps from a candidate start:

* The first step's collection begins after the slew in and the setup that follows
  it.
* Each later step follows a slew where it points elsewhere, and the setup budget
  where it asks for it.
* Cleanup and handoff come after the last step.
* Every step must be visible and allowed throughout, and every slew between steps
  must be allowed. Otherwise the start is tried later.

The visit fits before the next activity or is not placed at all. The slew to that
activity leaves from the last step's attitude. A visit that would run past the end of
the horizon is not placed, and nothing of it is checked beyond the horizon, which can
end with the ephemeris.

The plan lists the steps as entries. Each entry is in its step's instrument modes and
has the target's own timing where the target has one. The recorder ledger counts the
science steps' data, and the battery ledger counts the slews and holds of every step.

Local search moves a visit as one block. CP-SAT models a visit as one task. The
task's length includes the calibration, the slews between steps (pointing elsewhere
or rolling at the same pointing) and any step setup. The task arrives at the first
step's attitude and leaves from the last step's.

A rolling run credits each executed step to its target, as science or calibration,
and ``DITL`` commands the steps as it commands any science entries. Each planned step
carries the same ``visit_begin`` time, so rolling replans keep every remaining step
once the visit is committed. An urgent ToO waits for that visit to finish before it
can begin.

Timed requests in dispatch
--------------------------

Dispatch looks ahead to requests with a deadline, so that it does not start an
observation that makes one of them miss its deadline. A request with an
``earliest_start`` cannot be selected before it, and one with a ``deadline`` must
begin collecting by it.

Dispatch keeps time for each such request from its latest start. The latest start is
worked out as follows:

* It is the request's deadline less a dispatch step, a worst-case slew and the
  request's setup.
* It is no earlier than the request's opening, which is the first visibility from its
  earliest start.
* Where the request is not visible at its latest start, dispatch keeps time from the
  latest earlier time at which the request is visible for what it must collect, which
  is a whole visit or its minimum snapshot. Where there is no such time, dispatch
  keeps time from the request's first visibility after.

Dispatch then shortens or passes over candidates to keep that time:

* When a request ranked above a candidate must start before the candidate's snapshot
  would end, the snapshot ends there instead, so a long observation does not make the
  request miss its deadline. The request is then selected and slews from there.
* A request ranked below the candidate ends the snapshot the same way where the
  candidate still meets its own deadline after it. That is the case when the
  candidate has no deadline, or when the candidate's own latest start is at or after
  the request's start plus a slew, the request's overheads and what the request must
  collect. Two timed requests that both fit are therefore taken in order of their
  deadlines, whatever their merits, and a candidate that cannot wait keeps its place
  by merit.
* Once such a request's start has come, a candidate that can wait for the request is
  passed over for it while the request is selectable.
* A candidate whose minimum snapshot does not fit before the request's start is
  passed over, for a shorter candidate or, if none fits, for nothing.

These rules have the following consequences:

* A request with slack lets the observation before it run on until its slack is used.
* Dispatch keeps no time for a request with no deadline. Such a request competes by
  merit once it is open, and gives way to any timed request.
* An exposure need not be split into short snapshots to leave room for a timed
  request, and a timed request needs no slack beyond its slew and setup.

A request that must start at its opening, slack or not, says so with
``protect_opening`` (``queue.add(..., protect_opening=True)``). Dispatch then keeps
time for it from its opening, as it does for a request with no slack.

The ranking is by base merit, with any allocation bonus. A request's dynamic terms
are not known before it opens. Targets of opportunity join the queue when they are
submitted, so nothing is shortened ahead of them.

The log says when a snapshot was shortened and how many candidates were passed over.

Planning within the battery
---------------------------

Every scheduler keeps the battery above its planning level. The level is
``battery.planning_level``, a fraction of full (see :doc:`configuration`). By default
it is the depth-of-discharge floor, ``1 - max_depth_of_discharge``, so no plan takes
the battery past the discharge it is allowed. Set it higher to keep margin, or to
``None`` to plan without regard to the battery. ``DITL`` does not charge on its own
while executing a plan, so the plan carries its own charging.

The battery is the second consumable on :class:`~conops.schedulers.ConsumableLedger`.
The ledger holds it as the *deficit*: the energy below full, in watt-hours. Data on
the recorder only comes off during a contact, but energy comes back whenever the
panels are lit, whatever the spacecraft is doing.
:class:`~conops.schedulers.BatteryLedger` therefore orders a segment's additions in
time and walks them step by step, as the simulation drains and charges the battery.
Every span is counted at the attitude the spacecraft holds through it:

* A slew draws ``SLEWING`` power and generates along its path.
* A science snapshot draws ``SCIENCE`` power. It generates at its attitude and the
  power-optimal roll the scheduler gives it, which the simulation flies.
* A contact draws ``PASS`` power from the start of its slew, which ACS runs in
  ``PASS`` mode, and generates along its tracking profile. It then draws ``SLEWING``
  power until the spacecraft is at rest after it.
* A charge draws ``CHARGING`` power and generates at the charging attitude.
* Idle time draws the idle power and generates at the attitude the activity before it
  leaves the spacecraft in.
* A panel on a finite solar-array drive is carried along as the simulation carries
  it. It tracks the Sun in the modes its control lists and holds its angle in the
  rest. Each activity is therefore counted at the drive angles the plan reaches it
  with, and a plan's charges take the charging attitude for those angles. A rolling
  replan starts from where the drive is. The committed activities are flown as ACS
  flies them, slews and the idle time between and after them included, until the plan
  takes over.
* The deficit never falls below zero. A surplus before a draw is lost, as a full
  battery wastes what its panels make.
* Generation is not counted past ``planning_recharge_level``, so a plan never assumes
  the battery above it. A deficit already below that floor is left where it is until
  something draws.
* The segments start at each sunrise, and the plan is checked at every segment's
  peak.

The scheduler first evaluates the whole horizon without new charging manoeuvres.
If passive generation and the available battery keep it above the planning level,
it adds no charge, including during long sunlit idle gaps. A battery that is full
now can still require a charge for a later deficit, because the decision uses future
loads as well as the initial state of charge.

Where charging is needed, the scheduler builds geometry-feasible candidates in
sunlit idle gaps. For each it checks the actual ingress, the reconnected successor's
slew, IDLE generation before and after charging, saturation at the recharge level,
eclipse, and finite array-drive behaviour. It removes candidates whose omission
preserves battery feasibility and safe connections, then searches for shorter
holds on the simulation grid. Each accepted shorter hold is checked against the
whole horizon. This search is bounded and does not claim a globally optimal set of
manoeuvres. Explicit or committed charging entries remain reserved.

A snapshot is placed only where the resulting forecast stays within the battery
limit through the end of the horizon. Where it does not, the scheduler can try the
snapshot later in the gap, leaving time to charge first. The snapshot is never
shortened to fit. A schedule that remains infeasible is still rejected. The finished
plan carries the retained charges as ``CHARGE`` entries, which ``DITL`` executes and
:meth:`~conops.ditl.DITL.validate_plan_matches_execution` checks.

The ledger is the simulation's own arithmetic at the attitudes the simulation flies,
so a plan it accepts keeps the battery above the level when executed. What that means
for a plan depends on the spacecraft:

* A spacecraft whose panels generate more than observing draws, such as one with a
  side-mounted array rolled towards the Sun, needs no discretionary charge when
  passive generation suffices.
* A spacecraft whose panels only generate when pointing at the Sun reserves the
  necessary charges in idle sunlit time. It refuses the science the battery cannot
  carry through the eclipse after it. It is then the lowest-merit science that waits,
  as with the recorder.

The other schedulers use the same accounting. Local search decodes its sequences
through the same check. :class:`~conops.ditl.RollingHorizonDITL` starts each plan
from the battery the committed activities, charges included, leave by the time the
plan takes over. CP-SAT has the battery in its model (see `Optimizing plans with
CP-SAT`_).

Limitations
-----------

* Urgency is evaluated once, at the start of the horizon, when the plan is built.
  Deadlines are kept as constraints. Local search and CP-SAT also prefer earlier
  deadline science between equal-valued plans, and discount a late start only with an
  explicit positive ``earliness_weight``.
* Cadence is a minimum spacing. A visit before its cadence is due is never planned,
  and dispatch does not select one either. Spreading many visits through the day
  also leaves the scheduler more, smaller gaps to search, which lengthens planning.
* The priority-first scheduler is greedy. Apart from moving a snapshot later to make
  room before it, placements are never revisited. If a flexible, high-merit request
  is placed first, it can take the only slot a lower-merit request with a short
  window could use, and that request goes unplaced. Urgency avoids this when it
  raises the short-window request above the flexible one, so it is placed first.
  Local search can also recover it (see below).

Improving plans by local search
-------------------------------

Placing requests in priority order leaves gaps: a high-priority snapshot placed
early can strand time around it that nothing else fits into.
:class:`~conops.schedulers.LocalSearchScheduler` builds the priority-first plan and
then improves it.

The search works on the *order* of the plan's science snapshots. A decoder turns an
order into a timeline by placing each snapshot at the earliest time it fits after the
one before, with every check described above, so every plan it considers executes as
planned. Each snapshot starts out tied to the time and length the priority-first plan
gave it, so the starting point is that plan. The search then tries:

* **inserting** a snapshot of a request that still has exposure to place;
* **replacing** a snapshot with one of another request that still has exposure to
  place, which frees time that only a different target can use, such as the time the
  spacecraft would otherwise spend waiting on a hidden target until it rises;
* **releasing** a snapshot from its earlier time and length, letting it move earlier
  and run to its full ``ss_max``;
* **removing** a snapshot;
* **swapping** two nearby snapshots, or **moving** one a few positions.

A change is kept when it is no worse than the current plan, or no worse than the plan
of ``history_length`` evaluated changes ago. This rule is late-acceptance hill
climbing, and it lets the search cross plateaus. A proposal that changes nothing
consumes the original attempt budget without advancing or skipping a history slot.
The best plan found is returned, and it is never worse than the priority-first plan.

The objective is science time weighted by dynamic merit. It is scored as follows:

* Each snapshot's urgency uses its own visit deadline and visibility window at its
  planned slew start.
* Each snapshot's completion deficit uses the program shares delivered before it.
* The value a request has at the start of the horizon is an initial ordering
  estimate.
* Equal-valued plans prefer earlier science.
* An explicit positive ``earliness_weight`` applies an additional late-start
  discount.

For a windowed visit, the optional deadline discount uses the ``visit`` index stored
on that observation's plan entry, including the group's window limits. Rescoring an
earlier timeline therefore uses the same window as exact decoding, whichever visit
the request is currently trying to place. Sequence blocks and every exported step
retain that same visit index. Scoring does not change placement state.

Cadence visits keep their spacing, from each other and from locked visits. Each plan
entry's ``merit`` is the value its snapshot has where it is placed, as in the
priority-first scheduler.

.. code-block:: python

   from conops import LocalSearchScheduler

   scheduler = LocalSearchScheduler(config, targets, begin, end, time_limit=30.0)
   plan = scheduler.schedule()
   print(scheduler.initial_score, "->", scheduler.score, f"after {scheduler.iterations} changes")

``LocalSearchScheduler`` takes the same arguments as ``PriorityScheduler``, plus:

* ``start_plan``: a plan to start from where it scores higher than the priority-first
  plan, such as the plan a dispatch run of the same requests executed. Its science
  entries are decoded in time order, each at its own start, with a visit's steps
  decoded as one. An entry that cannot be placed at its start is left out. The search
  then improves on the better of the two starts, so the plan returned scores at least
  as well as either. CP-SAT takes ``start_plan`` the same way, with each chunk's hint
  coming from it where it is the better start.
* ``additional_start_plans``: other plans to decode as independent starting
  contenders before searching. Each is checked against the current state with
  separate activity, recorder and battery reservations. ``seed_score`` continues
  to describe the primary ``start_plan``, and ``start_score`` describes the selected
  decoded start.
* ``time_limit``: seconds of search after the priority-first plan is built
  (default 10).
* ``max_iterations``: changes to try at most. A run limited only by time depends on
  machine speed, so set ``max_iterations`` (and ``seed``) for reproducible plans.
* ``seed``: random seed, defaulting to the configuration's.
* ``neighborhood``: how many positions apart a swap or move can be (default 3).
* ``history_length``: length of the late-acceptance history (default 50).
* ``earliness_weight``: optional fraction of a deadline request's value lost if it
  starts at its deadline instead of at the start of the horizon, or at its earliest
  start if that is later (default 0). Valid science keeps its full value by default,
  and equal-valued plans prefer earlier deadline science, then earlier science
  overall. Set a positive value only when delay should reduce science value. It must
  be at most 1, and values below 1 keep a late snapshot worth more than none.

Ground passes and locked entries stay where the priority-first plan put them.

``scheduler.search_iterations`` and ``scheduler.search_limits`` record every
local-search pass in the latest ``schedule()`` call, including momentum replans.
An ending is ``"iteration budget"``, ``"wall clock"`` or ``"disabled"`` (zero
time or zero requested changes). The scalar ``iterations`` reports only the
last pass, so it cannot establish that all earlier passes completed their budgets.
These diagnostics also cover CP-SAT's optional local-search polishing.

Local search has two limitations. By default nothing is rewarded for starting early
beyond the preference between equal-valued plans, and the optional ``earliness_weight``
discount applies only to requests with a deadline. Each change is decoded from the
point it touches, so a search of a
few thousand changes takes tens of seconds for a day with a few hundred targets.

The notebook ``examples/Local_Search_Scheduler_Example.ipynb`` compares the two schedulers
on a one-day plan, as the number of changes grows, and in rolling-horizon runs.

Optimizing plans with CP-SAT
----------------------------

:class:`~conops.schedulers.CpSatScheduler` builds plans with the CP-SAT constraint solver
from Google's OR-Tools, which is an optional dependency (see :doc:`installation`). It
optimizes the same objective as local search, but chooses which snapshots to observe
and in what order by solving a constraint model rather than by trying changes one at
a time.

The scheduler first builds the priority-first plan. It then divides the horizon into
chunks and asks CP-SAT, one chunk after another, for a better choice and order of
snapshots than the plan has there. The solver's model is an approximation of the
spacecraft, so every answer is decoded with the scheduler's exact checks, as in local
search, and is kept only if the decoded plan scores at least as well as the plan the
scheduler already had. The subsections below describe each part in turn.

Chunks and candidates
^^^^^^^^^^^^^^^^^^^^^

The horizon advances in consecutive ``chunk`` lengths (three hours by default). The plan
the scheduler holds at any moment is called the incumbent here. It starts as the
priority-first plan, or as a decoded seed that scores higher (see `Starting from other
plans`_). While one chunk is
solved, the part of the incumbent that comes after the chunk, its continuation, is left
unchanged.

A chunk's model may also use up to one further chunk of empty time, where there is some
before the unchanged incumbent continuation. This look-ahead lets a flexible
observation move to after an observation whose visibility window is closing, so that
the plan fits both visits instead of keeping the flexible one first. The look-ahead
adds no solver budget, and the exact whole-horizon comparison described under
`Decoding and accepting a chunk`_ must still accept the proposal before it replaces
the incumbent. The model limits how late a snapshot may start, so a snapshot may finish
after the model's latest permitted start.

For each chunk the scheduler builds a list of candidates. A candidate is one snapshot
of a request in one of its visibility windows. A window gets as many candidates of a
request as the request's exposure needs and the window holds. Each candidate is
optional: the solver decides whether to take it, and for a candidate it takes, its
arrival time and a collection length between the request's ``ss_min`` and ``ss_max``.

What the model contains
^^^^^^^^^^^^^^^^^^^^^^^

CP-SAT chooses candidates and their order under these rules:

* A circuit through the chosen candidates fixes the order.
* Each slew starts on the first simulation step after the task before it ends, and
  once its target is visible. Each snapshot finishes within its window.
* Collection starts no earlier than the request's ``earliest_start`` and by its
  ``deadline``. The slew may start before the earliest start.
* Ground passes are fixed tasks, and snapshots must leave room to slew to them.
* A request's snapshots never add up to more than its remaining exposure.
* A cadence target's visits are at least its cadence apart, and the first of them
  comes at least its cadence after the target's last visit.
* The recorder stays below its planning limit (see `Planning within the recorder`_).
  The rules for this are given next.

The recorder is modelled in the segments between contacts that the recorder section
describes:

* Each candidate's data goes on the recorder in the segment where its collection
  starts. The data is the candidate's rate for the length of its collection, plus its
  fixed amount, at the observed target's rates. It is rounded up to whole kilobits so
  that the model never counts less than the ledger does.
* Each segment's volume is what entered the segment after its contact, plus
  housekeeping and the science already planned there, plus the chunk's candidates.
* What a contact cannot remove carries over to the next segment.
* Every segment from the chunk's own to the end of the horizon is kept within the
  limit. A candidate is therefore traded against the room it takes from everything
  planned after it.

The objective is dynamic-merit-weighted science. Two of the merit terms change with
time and are treated differently:

* Urgency follows the model's variable for the step at which the chosen candidate's
  slew starts, using the same closing-time function as dispatch.
* Completion deficit is estimated at the chunk's start, then evaluated again during
  exact decoding.

Configured earliness discounts and allocation bonuses (see `Long-range allocation`_)
also apply to the time-dependent urgency contribution. Calibration contributes no
science value.

CP-SAT's native model works in integers, so it rounds urgency fractions conservatively
at the steps it decides on, normally to one part in a million. On an exceptionally
long horizon the scheduler reduces that scale to keep the integer products in range.
The rounding does not reach the plan. Exact decoding evaluates the same urgency
function at full floating-point precision before plans are compared, and the exported
entry's ``merit`` records that unrounded value, including the allocation factor that
applies.

The battery in the model
^^^^^^^^^^^^^^^^^^^^^^^^

Where the battery limits the plan (see `Planning within the battery`_), the battery is
in the model too. The model divides time into runs: the horizon, from the run the chunk
starts in to the horizon's end, is cut into alternating sunlit and eclipse runs.

Each candidate belongs to the run its arrival falls in. Where the candidate's range of
possible arrival times lies within one run, it is placed in that run directly. Where the
range spans several runs, the model has a boolean for each run that says whether the
candidate is in it.

A candidate adds energy to its run as follows:

* For its collection and overhead it adds its net power less the run's idle power. Its
  net power is its mode's draw less what its panels generate at its power-optimal roll.
* It adds a typical slew in, at the slewing power.
* The part of the candidate that spills past the end of its run is counted in each run
  after it, for the seconds of the candidate that run holds.

A run's energy is then its idle power for its whole length, plus what the candidates
and the already planned activities change. Idle power charges the battery in sunlight,
at the charging attitude, and draws from it in eclipse.

The battery's deficit is carried from each run into the next. The deficit carried into
a run is bounded below by the previous run's deficit, by zero, and by the smaller of
the recharge floor and the deficit entering the run. Every run's end must stay within
the limit.

In an eclipse run everything draws, so the run's end is its peak. In a sunlit run the
deficit can rise and fall within the run, and the model, being linear, cannot follow
that. The decoder walks the plan exactly and so has the last word: a chunk whose
solution does not hold up keeps the incumbent's snapshots. The incumbent is hinted
into the run and carry-over variables, so the solver starts from a feasible
solution.

Decoding and accepting a chunk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The incumbent's snapshots in each chunk are the solver's starting hint. They
are adjusted where the model's approximate slews need it, so the solver starts from a
feasible solution. Each chunk finishes before the incumbent's next slew and
leaves to the incumbent the exposure it collects later, so the rest of the incumbent
can always follow whatever the solver chooses.

The model approximates slews in two ways. Each target's roll is chosen before solving.
With a thermal settling table, each slew settles for the change it makes to the Sun's
direction in the body frame, with the Sun where it is at the chunk's origin. Because of
these approximations, each chunk's order is decoded with the scheduler's exact checks,
as in local search.

The decoded order is kept if, followed by the rest of the incumbent, it scores higher
than the incumbent does from the same point. It is also kept where the two score the
same and its science is no later, comparing deadline science first and then all
science. Otherwise, or if the solver found no solution in its time, the incumbent's
snapshots for the chunk are kept. The next chunk is solved from where the kept
snapshots leave the spacecraft. Every plan therefore executes as planned and is never
worse than the priority-first plan. ``solver_chunks_used`` shows which chunks kept the
solver's snapshots.

The exact decoding also applies the rules the model leaves out:

* visits stay a cadence clear of locked visits of the same target;
* ``slew_before_visible`` lets a slew start before its target rises;
* each plan entry's ``merit`` is the value its snapshot has where it is placed.

Starting from other plans
^^^^^^^^^^^^^^^^^^^^^^^^^

``CpSatScheduler`` accepts the ``start_plan`` and ``additional_start_plans`` arguments
of ``LocalSearchScheduler`` (see `Improving plans by local search`_). Such a plan is
called a seed below.

The priority-first start and the supplied seeds are decoded independently, each with
its own reservations for sliding activities. A seed's activity times apply only while
that seed is evaluated, and they are kept only if the seed wins the starting-plan
comparison. A losing seed cannot move a mandatory block into the priority-first
science. Locked entries are copied when planning begins, so recomputing the slew into
one (its ingress) preserves the caller's plan and its collection windows.

When a dispatch seed decodes to a better starting plan, that decoded plan is the
incumbent for chunk hints and comparisons. While a chunk is solved, the recorder and
the battery must keep room for the observations the incumbent places later. Those
reservations come from the incumbent's continuation, decoded from the current prefix
(the plan accepted so far), rather than from the priority-first timeline, which is
unrelated to the seed.

* Recorder reservations are kept separately for each store. They include the actual
  shortened collection, sequence calibration data, and contact ingress changes. They
  subtract the old current-chunk prefix, so that the optional candidates that may
  replace it are not counted twice.
* Battery reservations use the continuation's actual blocks and drive history.
* A snapshot that exact decoding cannot place contributes no fixed science load.

An accepted chunk invalidates the old continuation, which is decoded again from the new
prefix before the next chunk.

Using the CP-SAT scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops import CpSatScheduler

   scheduler = CpSatScheduler(config, targets, begin, end, solver_time_limit=20.0)
   plan = scheduler.schedule()
   print(scheduler.solver_statuses, scheduler.initial_score, "->", scheduler.score)

``CpSatScheduler`` takes the arguments of ``LocalSearchScheduler``, plus:

* ``solver_time_limit``: seconds CP-SAT may search, shared across the chunks
  (default 20);
* ``solver_deterministic_time``: units of CP-SAT's deterministic time the search may
  take, shared across the chunks. With this limit the search ends at the same point
  every run, whatever the machine's speed. The wall-clock limit still applies. Set
  it, with ``workers`` 1 and ``seed``, for runs that repeat exactly;
* ``chunk``: length of each chunk (default three hours);
* ``workers``: CP-SAT search workers (default 8);
* ``max_candidates``: candidates per chunk at most (default 120), those of the
  incumbent first, then in priority order.

``time_limit`` defaults to 0 here; set it to spend that many seconds improving the
solver's plan by local search afterwards. The plan's metadata records
``"scheduler": "cp_sat"``, and ``solver_score`` and ``solver_seconds`` give the score
and time of the solver's plan before any local search.

CP-SAT solver diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^

After ``schedule()``, three lists describe the chunks, with one value per chunk:

* ``solver_statuses`` is CP-SAT's status for each chunk.
* ``solver_chunks_used`` says whether the solver's snapshots were kept over the
  incumbent's.
* ``solver_limits`` says what ended each chunk's search: ``"optimal"``,
  ``"deterministic budget"``, ``"wall clock"``, or ``"solver"`` where the solver ended
  the search itself. A chunk answered from a cached response is ``"reused"`` (see
  `Replan records`_).

Limit endings are inferred from the solver's counters. An unfinished search near
both limits is reported conservatively as ``"wall clock"``, so a possible
load-dependent stop is not labelled deterministic. A completed optimality or
infeasibility proof keeps its solver-proven ending even near the limits.

Where the configuration tracks stored momentum, a schedule is built in several passes
(see `Planning within the wheels`_), and these three lists describe the final momentum
pass. To check every actual solver call in the latest ``schedule()``, inspect
``solver_search_statuses`` and ``solver_search_limits``. They retain earlier passes
even if a later pass finishes optimally. Chunks that need no model contribute no
search record.

A chunk model that CP-SAT finds invalid is one the scheduler built wrong. It is a
different case from a search that ran out of time, and the scheduler does not fall
back to the incumbent for it: it raises a ``ValueError`` naming the model's
fault.

Limitations of the CP-SAT scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chunks are solved one after another. A chunk cannot borrow time or
exposure the incumbent uses later, so improvements that would need to
rearrange several chunks at once are left to local search. The model's slews are
approximate; the exact decoding corrects them but can drop or shorten a snapshot the
solver chose. With very short limits (well under a second per chunk) the solver may
find nothing, and the plan is the one the scheduler started from.

The notebook ``examples/CP_SAT_Scheduler_Example.ipynb`` compares CP-SAT with local
search on a one-day plan, chunk by chunk, as the planning time grows, and in
rolling-horizon runs.

The scheduling context
----------------------

:class:`~conops.schedulers.SchedulingContext` answers the questions a scheduler asks,
using the models ``DITL`` executes with:

* slew durations and paths from the real :class:`~conops.simulation.Slew`;
* scoped attitude-constraint checks as ``DITL`` records them;
* the attitude a fresh simulation idles at before its first command;
* the ground passes in the horizon.

Future schedulers are built on the same context so that their plans also execute as
planned.

Rolling-horizon replanning
--------------------------

:class:`~conops.ditl.rolling_ditl.RollingHorizonDITL` simulates a mission that is
planned on the ground and replanned periodically. It runs like
:class:`~conops.ditl.ditl.DITL`, but builds its own plan with ``PriorityScheduler`` and
rebuilds it as the simulation runs:

1. **Initial plan.** At the start, a plan is built for the next ``horizon``.
2. **Scheduled replans.** Every ``replan_interval``, the plan is rebuilt. Everything
   already commanded, or due to start within ``commit_lead_time`` (the time a new
   plan takes to reach the spacecraft), is kept, and so are the remaining steps
   of a sequence visit under way at that cutoff, since a visit is taken whole. The
   rest is planned again from where the committed activities leave the
   spacecraft, using each target's remaining exposure.
3. **Targets of Opportunity.** A ToO submitted with
   :meth:`~conops.ditl.rolling_ditl.RollingHorizonDITL.submit_too` joins the target
   pool when it becomes active and goes into the next scheduled plan. A ToO whose
   deadline cannot wait for that plan triggers a rapid replan, which may cut short
   the observation in progress (see `Targets of Opportunity`_ below).

.. code-block:: python

   from datetime import timedelta

   from conops import RollingHorizonDITL

   ditl = RollingHorizonDITL(
       config,
       targets,
       begin=begin,
       end=end,
       horizon=timedelta(days=1),
       replan_interval=timedelta(hours=12),
       commit_lead_time=timedelta(hours=1),
   )
   ditl.submit_too(
       obsid=1000001, ra=105.0, dec=10.0, merit=500, exptime=900, name="GRB",
       submit_time=begin + timedelta(hours=5),
       deadline=begin + timedelta(hours=6),
   )
   ditl.calc()

   for replan in ditl.replans:
       print(replan.reason.value, replan.kept, replan.dropped, replan.added)
   print(ditl.too_response_times())       # seconds to first science, by obsid
   assert ditl.validate_plan_matches_execution() == []

Pass ``scheduler=LocalSearchScheduler`` or ``scheduler=CpSatScheduler`` (and, for example,
``scheduler_options={"time_limit": 10.0}`` or ``{"solver_time_limit": 10.0}``) to build
each plan by local search or with CP-SAT.

Comparing a run with :class:`~conops.ditl.queue_ditl.QueueDITL` dispatch on the same
targets shows what planning ahead gains or costs in science time and ToO response.
A ``RollingHorizonDITL`` instance runs once; create a new one for each simulation.

Targets of Opportunity
^^^^^^^^^^^^^^^^^^^^^^

A ToO normally waits for the next scheduled plan. A **rapid replan** runs at once
instead if the ToO's ``deadline`` falls before that plan could start collecting it,
which is the plan's lead time plus a worst-case slew and the setup time.

A rapid replan may cut short the observation that is running at the commit cutoff,
whether or not that observation has started yet. The cut is made at the cutoff, and
only when all of these conditions hold:

* ``allow_interrupts`` is set;
* the ToO allows it (``interrupt``, an argument of ``submit_too``);
* the observation's category is ``interruptible``;
* the ToO's value, evaluated now, beats the value frozen onto that observation, which
  is the value recorded on its plan entry when it was planned;
* the ToO would not begin collecting by its deadline with the observation left whole;
* the value frozen onto the ToO's first snapshot in the replacement plan beats the
  value frozen onto that observation.

Both frozen values include any `long-range allocation`_'s bonus. Before it values the
ToO's snapshot, the replacement plan accounts for the horizon's share of each bin and
for the allocation already collected or committed.

Where the conditions do not hold, the ToO can start only after the observation ends.
A cut costs the observation's slew, settling and setup again, so it is made only where
the deadline needs it. To find out, the replan first plans the horizon with the
observation whole. If that plan collects the ToO by its deadline, the ToO waits for
the observation, as it does in dispatch.

These rules cover the remaining cases:

* A sequence visit is never cut. The ToO waits for it.
* Science an observation has already collected is kept, and the observation's cleanup
  follows the cut.
* The cut is kept only if the new plan can collect the ToO.
* If several ToOs need a rapid replan at the same time, they are tried highest-ranked
  first. The first that outranks the observation, needs the cut and can be placed
  interrupts it.
* A ToO that may not interrupt does not take the cut observation's place either,
  however it ranks. It is planned in after the interrupting ToO's whole visit,
  including all science and calibration steps and final cleanup. If its deadline falls
  before that visit ends, it waits for the next plan.

Results of a rolling run
^^^^^^^^^^^^^^^^^^^^^^^^

* ``ditl.replans`` lists each replan as a
  :class:`~conops.ditl.rolling_ditl.ReplanRecord` (see `Replan records`_ below).
* ``ditl.too_response_times()`` gives the seconds from each ToO's submission to its
  first science, or None if it was never observed.
* ``ditl.plan`` is the plan that was actually followed: the committed parts of
  every plan. ``validate_plan_matches_execution()`` checks the execution against it.
* Collected science is credited to the targets, so their remaining exposure
  (``exptime``) and ``collected_seconds`` reflect the run, as with a target queue.

Replan records
^^^^^^^^^^^^^^

A :class:`~conops.ditl.rolling_ditl.ReplanRecord` holds, for one replan:

* when and why it ran, and the commit cutoff;
* how many entries were kept, dropped and added, and how many targets did not fit;
* the planning time;
* any ToO that triggered it, and any observation it interrupted;
* how many entries of the run's start plan seeded it;
* for a CP-SAT plan, the solver's status for each chunk, whether each chunk's
  snapshots were kept and what ended each chunk's search (``solver_statuses``,
  ``solver_chunks_used`` and ``solver_limits``).

A replan can build several candidate plans and keep one, and each candidate can take
several search passes. The record therefore also carries cumulative lists that cover
all of that work. They are what to check when verifying that a run repeats exactly,
because a search that ended on the ``"wall clock"`` depends on the machine's speed and
load.

* ``search_iterations`` and ``search_limits`` include every local-search pass from all
  candidate plans built during that replan, including candidates later rejected. A
  later completed pass must not hide an earlier timeout, so check for ``"wall clock"``
  here. Both lists default to empty when reading older records or using a scheduler
  without local-search reporting.
* ``solver_search_statuses`` and ``solver_search_limits`` are the corresponding lists
  for CP-SAT. They retain every actual CP-SAT chunk search across all passes and
  candidate plans, including rejected candidates.
* ``solver_statuses``, ``solver_limits`` and ``solver_chunks_used`` keep their
  narrower scope: the final pass of the selected plan.

When auditing reproducibility, check for ``"wall clock"`` in both search-limit lists.
Older records default to empty cumulative lists, so they cannot retrospectively prove
that earlier attempts avoided wall-clock limits.

Within one schedule, CP-SAT can reuse the response to a chunk model it has already
solved instead of searching again. The record reports this as follows:

* ``solver_reused_chunks`` counts cached responses across all candidates, including
  rejected ones, and defaults to zero for older records.
* A cached response is labelled ``"reused"`` in ``solver_limits`` and is excluded from
  the actual-search histories.

Reuse happens only under these conditions:

* The scheduler runs one worker with a deterministic budget.
* The complete compiled model and the solver parameters are equal, including hints
  and all downstream resource reservations.
* The earlier search did not end on the wall clock. Wall-clock-ended searches are
  ineligible for reuse.

Both final and intermediate chunks can reuse responses. Their responses are held in
separate stores, which retain at most 16 models each per schedule. Because the stores
are separate, intermediate-model churn preserves the existing final-chunk retention.
Responses are decoded using the current requests and exact state. Both stores reset on
each schedule.

Starting attitude of a plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A planner that is not given the spacecraft's state starts from a default initial pose.
That pose uses the same origin-time roll as a fresh ACS. The planner does not replay
a command-free flight from the origin to each later candidate time, because that
hypothetical idle flight could fault even though the real spacecraft is following a
valid plan. Candidate generation leaves fault management untouched. The actual
ingress and hold still undergo their normal safety checks. A replan continues to use
its supplied execution state.

Long-range allocation
---------------------

:class:`~conops.schedulers.LongRangeAllocator` makes a long-range plan: it decides on
which day (or other period) each request should be observed, over a run of days to
months. The short-term scheduling, rolling replans or dispatch, then decides the order
within each day. This is the split between long-range and short-term planning that
observatories use.

It matters because the short-term scheduling looks only hours ahead. On its own, it
spends each day on the best targets available that day, so a target observable for
only part of the run, such as one the Sun will soon cover, or a request due beyond the
horizon, can miss its chance while time goes to targets that could have waited.

How the allocation is made
^^^^^^^^^^^^^^^^^^^^^^^^^^

The allocator splits the run into bins (a day by default) and decides how much of
each request's remaining exposure to collect in which bin. These rules set how much
a bin can hold and how much a request can be given:

* Each request can be allocated only the seconds its target is visible in a bin,
  between its earliest start and its deadline, from the same visibility windows the
  schedulers use.
* Each bin is expected to hold ``efficiency`` (default 0.75) of its length in science,
  the rest going to slews, setup and ground passes.
* ``reserve`` (default 0.1) of that capacity is kept free for work that isn't known
  yet, such as ToOs. Planned requests are never allocated it. ToOs the simulation has
  received are allocated first and use the reserve before anything else, so a ToO
  that fits in the reserve doesn't push planned requests to later days.
* A request gets either nothing in a bin or at least its ``ss_min``, and never more
  than its remaining exposure. A request with a cadence gets no more than an even
  share per bin.

The allocation is solved as a mixed-integer program with HiGHS, which comes with
OR-Tools (the optional ``cpsat`` extra; see :doc:`installation`), maximizing
merit-weighted time. Requests with a deadline are worth slightly more in earlier
bins.

The program is solved in two levels. ToOs come first, and their allocation is fixed
while the rest are solved, so they give up nothing for planned requests. Each level
starts from a greedy allocation of its requests into the room left. If the solver
finds no solution in time, that level and any after it keep the greedy allocation.

Two further preferences shape the result:

* The planned load is spread evenly over the bins as far as visibility allows, so
  every week keeps some slack. This is a tiebreak: it never costs a second of science
  for less than a thousand seconds of better balance.
* Re-allocating keeps the previous allocation unless changing it gains something:
  keeping a request's seconds where they were is worth an extra 1%. A ToO that fits
  in the reserve leaves every planned request where it was.

``solver="greedy"`` skips the mixed-integer program. It takes requests in order of
merit, each into the bins with the least competing demand, under the same bounds:
nothing in a bin or at least a snapshot, and no more than an even share of a cadence
request in any bin. It is quicker but not optimal, is the MILP's starting point, and
does not need OR-Tools.

The downlink budget
^^^^^^^^^^^^^^^^^^^

The allocator also makes the data the run generates fit what its ground contacts can
downlink:

* Each bin generates housekeeping, at the ``PASS`` rate during its contacts, plus the
  science data of the requests allocated to it. A request's science data is its
  target's rate for its allocated seconds and its fixed amount per observation, once
  per bin.
* The bin's contacts, from the same booked ground-station passes the schedulers use,
  take off what they can. What they cannot is carried on the recorder into the next
  bin, where it must stay within the recorder's planning limit (see `Planning within
  the recorder`_).
* The allocation starts from what is on the recorder, so a run that begins with a full
  recorder allocates its first day no more data than that day downlinks.

Where the contacts cannot keep up, the allocator therefore allocates low-data science
in place of high-data science. Without the budget, the high-data science would be
allocated and then refused day by day.

There is no budget without a planning limit, or when no request generates data, and
nothing is predicted for it then. When there is one,
:attr:`~conops.schedulers.Allocation.downlink`, ``housekeeping`` and ``science``
by bin and :meth:`~conops.schedulers.Allocation.recorder_volumes` report it.

A mission with named stores (see :doc:`data_management`) is budgeted store by store.
Each bin generates on each store what is routed to it, each store's contacts take off
only its reserved share of the link, and each store stays within its own planning
limit. A request is therefore allocated no more than every store it generates data on
can take. :attr:`~conops.schedulers.Allocation.recorders` reports each store's budget
by name, and the totals above sum them.

How scheduling follows the allocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The allocation says when each request should be observed, and short-term
scheduling follows it:

* A plan covering several bins plans each request's allocated seconds in the bin
  they are allocated to. Collection starting in the bin counts, up to the seconds
  allocated there. A bin that the plan's end cuts short counts pro rata.
* Allocated time is worth ``allocation_bonus`` (``bonus`` in the configuration,
  default 0.5) more of its request's value, shown as the ``allocation`` merit term.
  That is the only way the allocation steers scheduling. Allocated time competes
  with everything else on merit, so a larger bonus follows the allocation more
  closely. A bonus of ``0`` ignores the allocation altogether, so no request is held
  to its bin.
* The rest of a request's exposure, and allocated seconds that don't fit their bin,
  are planned like unallocated requests, by merit. A plan can therefore work ahead
  where that is worth more than the allocated time it displaces.
* Local search and CP-SAT score allocated time the same way, so they improve a plan
  without trading away what the allocation put in a bin.
* Every scheduler freezes the value with the bonus onto the entry. Rolling
  replanning compares the ToO's first planned snapshot with the observation it
  would interrupt, using those frozen values (see `Targets of Opportunity`_ above).
* Dispatch adds the same bonus to the requests allocated to the current bin, for
  the seconds allocated there that the bin has not yet collected. A snapshot
  only partly covered gets that part of the bonus, and once a request has
  collected its allocation the rest competes at its merit.

A request the allocator hasn't seen, such as a ToO submitted after the last
allocation, is treated as allocated throughout.

The allocation is refreshed as the run goes on, using the exposure still remaining:
rolling runs re-allocate at every replan, and dispatch re-allocates at the start of
every bin. A rolling run can instead follow a fixed allocation made elsewhere, such as
a weekly long-range plan (``allocation=``). Each replan then counts what has been
collected or committed in each bin against its seconds.

Using an allocator
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from datetime import timedelta
   from conops import LongRangeAllocator, RollingHorizonDITL

   allocator = LongRangeAllocator(config, begin, end, bin_length=timedelta(days=1))
   ditl = RollingHorizonDITL(config, targets, begin=begin, end=end, allocator=allocator)
   ditl.calc()
   print(ditl.allocation.seconds[10042])   # seconds allocated to each bin

Pass the same ``allocator`` to :class:`~conops.ditl.QueueDITL` for dispatch, or set
``scheduler.allocation`` in the configuration (see :doc:`configuration`). Schedulers
take the steering as ``allocated``, the seconds each request has in each span of
time, from :meth:`~conops.schedulers.Allocation.allocated`:

.. code-block:: python

   allocation = allocator.allocate(targets, begin.timestamp())
   scheduler = PriorityScheduler(
       config, targets, begin, end,
       allocated=allocation.allocated(
           [t.obsid for t in targets], begin.timestamp(), end.timestamp()
       ),
   )

Comparing scheduling modes
--------------------------

:mod:`conops.benchmark` runs the same scenario through each scheduling mode and
measures them the same way, from telemetry. A
:class:`~conops.benchmark.BenchmarkScenario` provides factories for a fresh
configuration and target pool, since simulations change both, and lists the ToOs
that arrive during the run. Each way of running the scenario is called a contender.
The contenders are:

* :func:`~conops.benchmark.dispatch`: ``QueueDITL`` picks each next target.
* :func:`~conops.benchmark.planned`: one plan built up front by a scheduler and
  executed by ``DITL``. A plan built in advance cannot react to ToOs.
* :func:`~conops.benchmark.rolling`: ``RollingHorizonDITL`` with a scheduler and
  replanning settings.
* :func:`~conops.benchmark.configured`: whatever the scenario configuration's
  ``scheduler`` section selects (see :doc:`configuration`), built with
  :func:`~conops.ditl.create_ditl`, so a configured run can be measured against
  the explicit modes.

A ``planned`` contender with ``seeded_by_dispatch`` starts from a dispatch run. The
scenario is first dispatched by the queue. The scheduler, which must be a local-search
or CP-SAT one, then takes the plan that run executed as its ``start_plan``, with the
scenario's ToOs as requests known in advance. The plan it returns therefore scores at
least as well as the dispatch run in the planner's own terms. The table then shows
both runs, with the dispatch run as the bound the plan must meet.

Dispatch and rolling contenders can be steered by a long-range allocator
(``allocation``, the bin length, and ``allocation_reserve``), which their names show
as ``+alloc``.

.. code-block:: python

   from conops.benchmark import (
       configured, dispatch, format_results, planned, rolling, run_benchmark,
   )
   from conops.schedulers import LocalSearchScheduler, PriorityScheduler

   results = run_benchmark(scenario, [
       dispatch(),
       planned(PriorityScheduler),
       planned(LocalSearchScheduler, time_limit=10.0),
       rolling(LocalSearchScheduler, scheduler_options={"time_limit": 10.0}),
       configured(),
   ])
   print(format_results(results))

Each :class:`~conops.benchmark.BenchmarkResult` reports, measured the same way for every
contender:

* science time and ``dynamic_science``. ``dynamic_science`` is the seconds actually
  collected, weighted by the dynamic ``merit`` the executed entry had when it was
  selected, including the urgency, cadence, completion-deficit and allocation terms
  that apply. Repeated visits use their own recorded values. Calibration earns no
  science value;
* ``weighted_science``, reported separately as a fixed-value diagnostic that uses the
  original input merits. The table labels these two columns ``dynamic value`` and
  ``fixed value``. Dynamic value is the primary planner comparison. Fixed value does
  not replace the planner's configured dynamic objective;
* calibration collection time separately as ``calibration_hours``;
* time slewing, idle and in contact, and the number of observations;
* each ToO's response time, and how many started by their deadline;
* how many other requests with a deadline got their whole exposure;
* each program's share of the science collected;
* for targets with a cadence, how far the mean gap between visits missed the requested
  interval, and how many targets were revisited at all;
* planning and run time, and plan/execution mismatches.

A contender that fails is reported with its error instead of stopping the benchmark:
the table shows the error's type, and the full message is listed below it.

Standard scenarios
^^^^^^^^^^^^^^^^^^

:mod:`conops.benchmark.scenarios` provides seven scenarios, each stressing a different
part of scheduling. All of them share the same setup:

* one low-Earth orbit from a TLE;
* Sun and Earth-limb avoidance;
* the default ground stations;
* a battery that never limits operations. The spacecraft has no solar panels, so the
  battery's recharge alerts are turned off rather than letting it drain into emergency
  charging;
* seeded random targets.

The scenarios are:

* ``baseline``: a day of 200 targets and one ToO with a one-hour deadline;
* ``too-heavy``: eight ToOs across the day, half urgent (merit 1000, deadlines of 30
  minutes to 2 hours) and half routine (merit 500, 4 to 12 hours);
* ``oversubscribed``: 600 targets requesting far more time than the day holds, a third
  of them with deadlines;
* ``cadence``: twelve monitoring targets wanting a visit every four hours, and two
  survey programs with equal time shares but different merit, with the cadence and
  completion-deficit merit terms switched on;
* ``multi-day``: three days, 400 targets and a ToO a day, for replanning over a long
  run;
* ``long-range``: a week of 300 targets, for long-range allocation. Half the targets
  have deadlines spread across the week. The targets form two programs by the Sun's
  avoidance zone: "Early" targets that the Sun covers as the week goes on, and "Late"
  ones it uncovers. Ten ToOs arrive through the week, alternately due within hours
  and within days;
* ``program``: two days with the structure of an observing campaign and no ground
  stations. It is the scenario on which the dispatch-seeded plans are measured against
  the dispatch queue. Its base merits are in bands wider than the dynamic merit terms.
  It contains:

  * deep fields observed in resumable snapshots, with per-category setup and cleanup
    budgets;
  * campaign targets visited as sequences of steps, in visit windows of their own, and
    not interruptible;
  * monitoring targets with a cadence;
  * a pool of fillers;
  * twice-daily calibration blocks that may slide;
  * a fixed maintenance window;
  * a ToO a day with a four-hour response.

``scripts/benchmark_schedulers.py`` runs the standard contenders on one scenario or all of
them, with rolling replans every six hours (or four times in a shorter run) and a
30-minute commit lead time. Runs of two days or more also compare dispatch and rolling
steered by an allocator with one-day bins::

   uv run python scripts/benchmark_schedulers.py --scenario all --time-limit 10


Cross-process dispatch verification
-----------------------------------

Setting a random seed does not by itself establish that a run is repeatable.
``scripts/check_program_repeatability.py`` tests repeatability directly: it captures
identical inputs and runs dispatch in fresh Python processes, called workers below.
Run it from the repository root::

   PYTHONPATH=. .venv/bin/python scripts/check_program_repeatability.py \
       --config mission.yaml --program program.json --ephemeris orbit.e \
       --output out/repeatability --hash-seeds 2029 2029 1

Before running it:

* The destination must be new, so that earlier results are preserved.
* Initialize the cached Earth-orientation provider before capture.
* Moving targets must already carry saved tracks. This command does not refresh them.
* The default uses analytic planetary models. Supply ``--planetary-kernel`` for an
  explicit SPICE kernel.

The first two workers load the same saved program. Subsequent workers save and reload
it before execution, which also tests the saved-program round trip.

Each worker records the loaded Earth-orientation (EOP) data, source hashes, dependency
versions and actual input hashes. The runtime fingerprint also covers the verification
driver. Each worker checks the runtime and inputs against the captured manifest before
execution, and checks them again after execution. A checkout, dependency or input
change fails verification even if all workers otherwise agree. Changes detected after
execution are retained in the failed run record.

``conops.benchmark.reproducibility.execution_record`` retains exact entries,
decisions, telemetry, ToOs and science reports. ``compare_runs`` reports the
first differing value and event, without rounding away timestamps or attitudes.
Equality passes only for complete, valid runs with identical provenance; two
identical failures do not pass. A run that raises an exception keeps its original
diagnostic state, which is not replaced by a validation of the unfinished plan.

This verification covers dispatch and saved-program round trips. It does not establish
planned-run science quality or solver repeatability.

Sequence-step visibility reuse
------------------------------

A sequence step is observed at a pointing of its own, with its own visibility windows
(see `Observation sequences`_). Within a planning pass, repeated placement attempts
share a sequence step's pointing and visibility windows, so that they are not built
again for each attempt. The sharing is limited as follows:

* Different step indices, visit geometry, roll requests and instruments remain
  separate.
* Each placed step still gets its own plan entry, and its actual slew, roll and hold
  are checked at the attempted time.
* The shared templates are discarded before the next request-building pass or
  schedule. They do not carry collected science or completion history between
  requests.

``Pointing.step_pointing`` still returns a fresh pointing for public callers.
Custom pointing subclasses keep their factory behavior without this reuse.

Exact scalar science queries
----------------------------

A scalar answer is a science constraint's result for one time, pointing and roll.
Repeated planning checks can reuse an explicit science constraint's exact scalar
answers across physically unchanged hardware bindings. This reuse comes from an
additional memo that sits beside the constraint's existing rounded point lookup. No
scheduler setting is needed.

The additional memo reuses an answer only when all of the following match: the
constraint and ephemeris identities, the serialized science geometry, the exact time,
pointing and roll, and the current body frame. Exact memo keys distinguish positive
and negative zero. These rules also apply:

* A pending hardware binding does not itself prevent reuse or force the hardware
  refresh to happen earlier.
* Unsupported geometry serialization disables this additional reuse.
* The extra store holds at most 65,536 answers, retains their native owners while
  needed, and is discarded when the configuration is copied deeply.

The existing lookup behaves as before. The rounded point lookup still takes
precedence within the current binding, and held batches keep their existing
evaluation and refresh order. The additional memo therefore preserves those lookup
semantics and does not change the precision of the original cache.

An explicit ``config.constraint.clear_cache()`` forces fresh evaluation by
clearing both stores, even when the geometry is unchanged. ``cache_stats()``
retains its original point-lookup accounting, so measure native backend calls
separately when assessing the additional memo.

Timed visits at optimization boundaries
---------------------------------------

This section describes two details of how :class:`~conops.schedulers.CpSatScheduler`
handles fixed tasks and timed visits (visits that must fall in a visit window) where
its chunk models begin and end (see `Optimizing plans with CP-SAT`_).

The first concerns the model's time grid. The CP-SAT departure grid applies to new
snapshots. A settled activity or retained task keeps its ingress time in seconds, so
an off-grid finish does not prevent the next fixed task from starting on time. Ingress
still waits for its predecessor and reserves the full modelled slew duration.

The second is a repair after solving. After its native chunk solves, CP-SAT may
recover one missing timed visit with an exact insertion that strictly improves the
decoded horizon objective. Resource limits, commitments, visit windows and the
subsequent momentum forecast still apply. The native solver score and search-ending
inventory continue to describe the solver itself, and the repair adds no solver work
budget.
