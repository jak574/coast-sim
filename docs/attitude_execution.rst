Physical attitude execution
===========================

COASTSim keeps the operating mode of the attitude control system (ACS) separate
from the physical motion of the spacecraft. Mode handlers and guidance code may
request new targets, but a single component, the attitude executor, advances
the spacecraft's orientation and its body-frame angular velocity. Assigning
``last_slew`` or changing modes does not move the spacecraft.

The executor flies trajectories. A trajectory is a bounded motion that has been
checked against the motion limits and then installed in the executor. On each
update the executor first advances the trajectory that was installed
previously, and only then processes new commands.

This page lists the rules the executor follows, describes what happens when a
requested motion cannot be flown, and states what the model leaves out.

Execution contract
------------------

Motion profiles
~~~~~~~~~~~~~~~

Every motion is a quaternion trajectory that obeys the configured rate and
acceleration limits. These limits can be scalar or directional. Ordinary slews
share the scheduler's analytic rest-to-rest motion profile. Tracking follows a
profile of timed attitudes, called knots, and preserves its rate through each
knot.

Snapshots of geometry
~~~~~~~~~~~~~~~~~~~~~

A trajectory takes a snapshot of its geometry and motion coefficients. Mutating
a command's slew object afterwards cannot rewrite motion that is already
installed. Changing the configuration during a run remains unsupported, so
start a new run after any configuration change.

Handoffs between trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A handoff is the point where a new trajectory takes over from the current
motion. The new trajectory must match both the orientation and the body
angular velocity at that point, including when the rate is nonzero.

When a slew interrupts motion in progress, or a command ends tracking, the
executor brakes along the current rate axis within the directional
acceleration envelope. It then holds until the next execution tick. The slew
to the next target starts from the attitude at which the spacecraft actually
stopped. It does not wait for the old target to be reached.

SAFE mode and solar guidance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Entering SAFE changes the mode immediately, but the repointing that follows
uses the same handoff policy as any other motion. Charging and SAFE solar
guidance also request bounded turns. A target that cannot be reached is
tracked with lag; the attitude is never corrected instantaneously. Queued
commands take priority over these discretionary guidance corrections.

Ground-contact tracking
~~~~~~~~~~~~~~~~~~~~~~~

Ground-contact attitudes are flown as normalized quaternion Hermite curves.
Adjacent curves share the rate at the knot between them, and that rate is
estimated from the neighboring shortest-arc secants. An interval of constant
spin uses an exact analytic rotation instead.

Both admission (the check made before a trajectory is accepted) and execution
verify rate and acceleration over each entire interval. They do this with
subdivided Bernstein bounds, so the result does not depend on where time
samples happen to fall. A curve that cannot be certified is rejected.

Acquiring a contact
~~~~~~~~~~~~~~~~~~~

When tracking begins, the spacecraft must already be within the selected
profile's pointing tolerance. The slews that currently bring the spacecraft to
a contact arrive at rest. The first tracking interval must therefore allow
acceleration from that boundary rate, but the spacecraft does not need to stop
at internal knots. Execution matches the actual initial rate. When the profile
ends, a final braking arc prevents an instantaneous stop.

An infeasible trajectory is rejected. The contact window is never stretched
silently to make the trajectory fit.

Initial attitude and reported values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After initialization, ``acs.ra``, ``acs.dec`` and ``acs.roll`` are read-only.
Set an initial attitude before the first update, and use ACS commands from
then on. If no initial roll is given explicitly, the solar-optimal roll is
selected once at the initial boundary. ``angular_velocity_body`` reports
degrees per second.

Prediction and sampling
~~~~~~~~~~~~~~~~~~~~~~~

The scheduler predicts from this installed physical state, which includes
tracking and any newly commanded braking arc. It does not extrapolate from the
last target's metadata.

Repeating an update at the same timestamp cannot advance motion. Coarse
telemetry samples do not make the underlying analytic trajectory
instantaneous.

Quaternion handoffs
-------------------

Execution compares and transfers quaternions directly. Slew waypoints retain
their native quaternions, and ACS supplies its exact initial ``AttitudeState``
when constructing a slew, a tracking profile, or a guidance turn. RA, Dec and
roll are coordinates for reporting and for target requests. They are not used
as the physical handoff state, because their round trip is ill-conditioned at
the celestial poles.

Use ``AttitudeTrajectory.turn_from_state(state, target, limits)`` to continue
from executed motion at rest. ``turn`` remains a convenience for independently
specified RA/Dec/roll endpoints. A rest-to-rest turn cannot replace nonzero
angular velocity; request a physical stop first.

Handoffs allow only numerical error: ``1e-8`` degrees in quaternion separation
and ``1e-9`` degrees/second in the norm of the body-rate difference. These are
fixed numerical tolerances, far below the operating motion limits. They are
not a budget for discontinuous maneuvers. A quaternion with its sign reversed
represents the same orientation and is accepted.

Execution faults
----------------

Trajectory builders and the executor raise ``AttitudeExecutionError`` when
their physical contract cannot be satisfied. ACS catches these failures during
command execution and dwell guidance. When it catches one, it does the
following:

* It records an ``operational_fault`` in the current run's fault report and an
  ERROR in the DITL log.
* It cancels pending commands and any science or contact activity.
* It does not record the rejected command as executed.

Recovery
~~~~~~~~

The recovery requests a bounded brake from the current quaternion and body
rate. Fault management then applies its existing ``safe_mode_on_red`` policy:

* With the default policy, ACS enters SAFE and queues its solar-pointing slew
  after braking.
* With automatic SAFE disabled, ACS brakes and holds without automatically
  commanding a new target.

Neither policy accepts the rejected trajectory.

When recovery itself fails
~~~~~~~~~~~~~~~~~~~~~~~~~~

If a brake cannot be constructed, the last installed trajectory is retained.
If SAFE motion itself is rejected, discretionary guidance is disabled for the
remainder of the run, so that it is not retried every tick. Both failures are
recorded. Being in the SAFE operating mode is therefore not a claim that solar
pointing was achieved. Existing constraint checks still run.

How a faulted run ends
~~~~~~~~~~~~~~~~~~~~~~

A run with an execution fault returns ``False`` while retaining telemetry and
fault diagnostics. Queue planning still runs its final audits and records any
resulting validation failures in the fault report.

Some problems are not handled by fault recovery and remain explicit errors:
initialization errors, nonmonotonic time, unrelated programming errors, and
validation failures without an operational attitude fault. Fault recovery does
not certify an invalid plan for export.

Faults with momentum capacity enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With momentum capacity enforcement enabled, a rejected command uses the same
fault recovery path. If initialization or advancing the installed trajectory
exceeds capacity, both simulators terminate with ``False`` and retain the
fault report and any completed telemetry. They cannot continue a physically
valid safehold by freezing attitude or disabling the momentum guard.

Model limits
------------

The executor is a kinematic execution boundary. It enforces angular rate and
acceleration, and it is not a full spacecraft dynamics engine.

Momentum checks
~~~~~~~~~~~~~~~

Setting the optional ``stored_momentum.capacity_n_m_s`` also enables sampled
rigid-body actuator-momentum checks on all installed and executed
trajectories. These checks work as follows:

* Admission and execution share integration knots anchored to the trajectory,
  including its acceleration boundaries.
* Partial scheduler ticks read the same linearly interpolated inertial-torque
  model without changing those knots.
* Interval bounds check interior demand for slews, braking, coasting and
  Hermite tracking. These bounds certify the sampled torque model. They do not
  certify the true continuous external torque.
* Changing capacity invalidates the cached certificate.

What is not modeled
~~~~~~~~~~~~~~~~~~~

The model does not include actuator torque, individual wheel/CMG states,
flexible modes, jerk limits or pointing-control error. Acceleration can change
discontinuously within the specified bound, but angular velocity cannot.

Contact tracking
~~~~~~~~~~~~~~~~

The contact model preserves sampled attitudes and continuous angular velocity,
but it is not an exact antenna-tracking solution between knots. Its terminal
brake can move beyond the last pointing sample.

Established constant-spin tracking is independent of knot spacing. Acquisition
from rest and curved profiles are not generally cadence-independent. Changing
guidance cadence can also change the requested trajectory. Sampling an already
installed trajectory does not change it.

Conservative searches
~~~~~~~~~~~~~~~~~~~~~

The phase search and interval certificates are conservative and are not
globally optimal. Failure to find or certify a feasible curve does not prove
that no such curve exists. Braking follows the current rate axis. It is not a
time-optimal controller for arbitrary multi-axis dynamics.

Keepouts and visibility
~~~~~~~~~~~~~~~~~~~~~~~

Keepout and visibility checks remain separate planning and validation
concerns. Kinematic feasibility does not prove continuous-time clearance, and
it does not guarantee that an emergency transition or its waiting attitude
satisfies every keepout. The independent recorded-attitude rate check remains
a secondary diagnostic.
