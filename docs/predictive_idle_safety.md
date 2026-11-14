# Predictive idle safety

Between slews the spacecraft holds a fixed inertial attitude. The geometry of
the attitude constraints changes as the orbit evolves, so an attitude that is
safe now need not remain safe. `QueueDITL` therefore forecasts the constraints
configured for IDLE mode at the ephemeris cadence, and it reserves time to
leave each science attitude before that attitude becomes unsafe.

This page describes how the forecast works, what the scheduler does when a
hold is about to become unsafe, and what happens when no safe recovery can be
found.

## The escape reserve

The reserve is the time the scheduler sets aside for the spacecraft to turn
away from an attitude. It is the duration of a conservative 180-degree,
rest-to-rest maneuver. The maneuver is evaluated with the smallest configured
acceleration and rate semiaxes, or with the scalar limits when no per-axis
limits are configured. The reserve also includes the settling time and two
ephemeris timesteps. The settling time is `settle_time`, or the longest settle
in the thermal settling table when one is configured. `QueueDITL` steps at the
ephemeris cadence, so an ephemeris timestep is also a scheduler timestep.

The reserve bounds the duration of a direct quaternion slew. It does not
guarantee that a route exists that avoids every obstacle on the way.

Science admission and collection deadlines use this same reserve.

## Sampling and the time of a crossing

The forecast is evaluated at discrete samples, so it cannot give the exact
time at which an attitude crosses into a constraint. The first violating
sample only shows that the crossing happened somewhere in the preceding
timestep. The scheduler treats that whole preceding interval as potentially
unsafe, which has three consequences:

- A violating sample at or just beyond the simulation end still requires an
  escape when its preceding interval overlaps the run.
- A crossing whose entire preceding interval falls after the run does not
  require an escape after the run.
- A recovery hold must end before the potentially unsafe interval begins.

## Science commands that start later

A science command that is delayed does not execute at an arbitrary time. It
executes on a scheduler tick, so its start is rounded up to the tick on which
it will actually execute. The path and the observation are validated against
that rounded time.

While the command waits, the spacecraft holds its current attitude. If a slew
is in progress, the attitude held is the one the slew ends at. That attitude
must remain safe through the execution tick. The departure cannot be later
than the sample that precedes the first violation.

## Recovery when no science target is available

A hold has a departure deadline, which is the latest time the spacecraft can
leave it with the full reserve in hand. Normally the next science target takes
the spacecraft away before then. If no science target is available when the
current hold reaches its departure deadline, the scheduler searches for a safe
attitude to move to. The search is deterministic and covers nearby attitudes
and alternative roll angles.

A candidate attitude must meet all of the following conditions:

- The slew to it is finite and obeys the configured directional motion limits.
- No sample along the slew trajectory violates a SLEWING-scope constraint.
- After arrival the hold is IDLE-safe for another escape reserve plus
  `spacecraft_bus.attitude_control.idle_min_hold_s` (default 300 seconds).
  This period is clipped to the simulation end.
- It leaves enough time for every available tracking profile of the next
  planned ground-station acquisition. This check uses the same admission
  deadlines as science, including the routed-slew bounds and the pass trigger
  buffer.

The selected maneuver is executed through the ordinary ACS command queue as a
slew with an `IDLE` destination. It contributes normal SLEWING telemetry and
power. Discretionary selection of science targets and TOOs does not interrupt
this protective motion. Subsequent holds retain the commanded attitude and
roll.

## The idle-safety fault

Three situations latch an `idle_safety` RED fault in
`ditl.fault_management.states`:

- the run is initialized at an attitude that is already unsafe;
- the finite search finds no feasible recovery;
- an IDLE hold that was executed turns out to be unsafe.

The first `operational_fault` event retains the timestamp, the cause, and the
body attitude (RA/Dec/roll in degrees). Repeated reports of the fault do not
restart recovery and do not flood the event log.

### Response to the fault

The response depends on the `safe_mode_on_red` policy.

With the default policy, `safe_mode_on_red: true`, the fault requests SAFE
through fault management and immediately executes the ordinary SAFE command
path. SAFE entry clears the pending commands and initiates a finite slew from
the current attitude. It does not substitute a new reported attitude. Science
and TOO selection stop, and any active observation is closed. The simulation
continues in safehold and records telemetry through its end.

Explicitly setting `safe_mode_on_red: false` keeps the fault visible but
disables automatic SAFE entry. This is consistent with the other
fault-management policies, and this page calls it monitor-only mode.

### Result of the run

`QueueDITL.calc()` returns `False` after an idle-safety fault. This is also
the case in monitor-only mode. The plan and telemetry are retained, but they
are diagnostic outputs and do not form a valid science plan.

Both end-of-run audits still run. If an audit finds an attitude-rate failure
or a plan-execution failure, that failure is recorded as a further operational
fault. It does not raise a Python exception, because the run has already been
diagnosed as failed. The public validation methods still expose the
violations. A nominal run, which has no such fault, keeps its strict behavior
and raises an exception for an execution-validation failure.

These decisions use only `ditl.fault_management`, which is the report for the
current run. Reports from earlier runs remain available in `ditl.fault_runs`.
The events in those earlier reports cannot fail another run, and they cannot
suppress the validation exceptions of another run. This holds even when both
runs cover identical timestamps. The run-scoped fault state and policy
snapshots that make this possible are provided by the fault-management
lifecycle.

## Limits of this policy

SAFE is an operational response. Entering SAFE does not prove that the
spacecraft escaped without crossing a keepout:

- The standard solar-pointing SAFE guidance does not gain a path-validity
  guarantee from this policy.
- An unsafe initial condition remains a failed study input.
- Any violations that occur during recovery remain visible.
- Continuous enforcement of rate and acceleration limits across all guidance
  modes is a separate execution layer. This predictive policy does not supply
  that model.

The policy is a conservative planning policy. It is not a flight controller,
and it does not prove continuous-time collision avoidance. Tests and
acceptance checks remain at the ephemeris cadence, so finer sampling must be
assessed for a specific study.

Keep the following in mind when using the policy in a study:

- The initial attitude, the reserve policy and the hold dwell affect science
  yield, energy and disturbance momentum. Record them with the study results.
- Changing constraints during a run is unsupported. Create a fresh run and
  planner after editing the configuration.
- An explicitly empty IDLE scope disables the safety forecast. Do not
  interpret that configuration as a safe operating policy.
