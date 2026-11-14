Stored Momentum
===============

A spacecraft that holds an attitude against the environment's torques stores
angular momentum in its reaction wheels, and the wheels have a capacity. COAST
tracks that momentum from the torques themselves, unloads it as magnetorquers and
thruster dumps would, and plans and dispatches science so the wheels stay within
their capacity. This page describes the model, how it is configured, what the
simulations and the planners do with it, and what it leaves out.

The model
---------

What is tracked
^^^^^^^^^^^^^^^

The wheels store the angular impulse of every environmental torque that the
spacecraft rejects while it holds or moves its attitude. COAST adds up the
configured torques and integrates the sum in inertial coordinates by the
trapezoidal rule. Because the integration is done in inertial coordinates, a
change of attitude on its own creates no momentum. The stored momentum is
reported as a vector in the current body frame, together with its norm.

The integrator is :class:`~conops.simulation.StoredMomentumTracker`. Its
configuration is :class:`~conops.config.momentum.StoredMomentumConfig`, found under
``spacecraft_bus.attitude_control.stored_momentum``. The tracker also needs the
spacecraft's inertia tensor, ``spacecraft_bus.inertia_tensor_body_kg_m2``.

Sources
^^^^^^^

Each torque source is evaluated in the body frame at every sample, from the
ephemeris and the attitude. The telemetry reports each source's torque by name
(``disturbance_torques_body_n_m``), so a run shows which source fills the wheels.
The sources are listed below, each with the setting that turns it on.

* *Gravity gradient* (``gravity_gradient_enabled``). This is the central body's
  torque on the inertia tensor, :math:`3\mu/r^3\,(\hat r \times I\hat r)`. During
  an inertial hold the radial direction sweeps round the body once per orbit, and
  the torque is secular. It is the dominant source for a spacecraft with a few
  hundred kg m² of inertia difference, and it fills a few N m s of wheel capacity
  in hours to a day.
* *Solar radiation pressure* (``solar_radiation_pressure_enabled``). This is the
  torque of sunlight on the spacecraft's surfaces. Each plate that the Sun shines
  on takes the force of the light it absorbs and the light it reflects, both
  specularly and diffusely:
  :math:`F = -P A \cos\theta\,[(1-\rho_s)\hat s + 2(\rho_s\cos\theta + \rho_d/3)\hat n]`.
  The pressure is scaled from its value at one astronomical unit by the Sun's
  distance. A plate that faces away from the Sun takes no force, and no plate
  takes any in eclipse. The torque is secular for an inertial hold and for a
  Sun-referenced hold.
* *Aerodynamic drag* (``aerodynamic_drag_enabled``). This is the torque of the
  atmosphere on the same surfaces. The flow is the spacecraft's velocity relative
  to the atmosphere, and the atmosphere rotates with the Earth:
  :math:`v_{rel} = v - \omega_\oplus \times r`. Each plate that faces the flow
  takes the force
  :math:`F = -\tfrac12 \rho |v_{rel}|^2 C_d A (\hat n \cdot \hat v)^+ \hat v`,
  with its own drag coefficient. The flow direction turns once per orbit, so
  during an inertial hold the torque largely averages out. The density is the
  uncertain input: it differs by a factor of twenty between quiet Sun and solar
  maximum.
* *Residual dipole* (``residual_dipole_body_a_m2``). The spacecraft has a magnetic
  dipole of its own, and the geomagnetic field exerts the torque
  :math:`m \times B` on it. For a dipole of a few A m² in low Earth orbit, this
  torque is of the order of solar radiation pressure.
* *Magnetorquer unloading* (``magnetorquer_max_dipole_a_m2``,
  ``magnetorquer_time_constant_s``). This is a negative source, because it removes
  momentum. The torquers command the dipole that would remove the stored momentum
  over the time constant, capped at their greatest dipole. At the cap, the
  unloading torque is the cap times the field. Magnetorquers turn the spacecraft
  only about axes across the field. They therefore take out the component of the
  stored momentum that lies across the field and leave the component along it.
  The component that is left turns as the field turns round the orbit.

Surfaces
^^^^^^^^

Sunlight and drag act on the plates in ``spacecraft_bus.surfaces``
(:class:`~conops.config.SurfaceModel`). Each plate is flat and has a centre, an
outward normal, an area, a specular and a diffuse reflectivity, and a drag
coefficient. The torques are taken about the model's centre of mass.

Three functions build plates from the rest of the configuration:

* :func:`~conops.config.box_surfaces` builds them from a box.
* :func:`~conops.config.panel_surfaces` builds them from the solar panels that
  have a geometry.
* :func:`~conops.config.telescope_surfaces` builds them from the telescopes'
  apertures.

When no surfaces are configured, :func:`~conops.config.guess_surfaces` builds a
placeholder. The placeholder is a 1.5 × 1.5 × 2 m box, the panels and the
apertures, about a centre of mass 5 cm behind the boresight. Its torques are of
the right order and nothing more, and a mission's mass properties should replace
it.

Plates do not shadow one another.

Atmosphere
^^^^^^^^^^

The density that drag uses comes from ``stored_momentum.atmosphere``
(:class:`~conops.config.AtmosphereConfig`). Its ``model`` field chooses between
two models:

* ``model: "table"`` (the default) is a piecewise exponential profile by geodetic
  height. It is Vallado's table for moderate solar activity and needs no data. It
  gives the density to within a factor of a few at any height a spacecraft flies.
* ``model: "msis"`` is NRLMSIS 2.1, through the optional ``pymsis`` package. The
  model is evaluated at the solar flux (``f107``, ``f107a``) and the geomagnetic
  activity (``ap``) that the run assumes. A run therefore states one level of
  activity and does not fetch the indices of its dates. Use this model where the
  density matters. Install it with the ``momentum`` extra (see
  :doc:`installation`).

``density_scale`` is a factor applied to the density of either model, for
sensitivity studies.

Geomagnetic field
^^^^^^^^^^^^^^^^^

The magnetorquers unload against the geomagnetic field, and the same field
torques the residual dipole. ``stored_momentum.geomagnetic_field`` chooses the
field model:

* ``"dipole"`` (the default) is a tilted dipole whose axis passes through the
  geomagnetic poles. Its strength is 31.2 µT at the magnetic equator on the
  surface and falls with the cube of the radius. It gives the strength and
  direction to ten or twenty per cent of the real field, which is enough for the
  torques' order of magnitude.
* ``"igrf"`` is the International Geomagnetic Reference Field, through the
  optional ``ppigrf`` package, evaluated at the sample's epoch. The package is in
  the ``momentum`` extra.

The telemetry reports the field at each sample in body coordinates
(``geomagnetic_field_body_t``). Where drag is tracked, it also reports the
atmosphere's density (``atmosphere_density_kg_m3``).

Sampling
^^^^^^^^

The torque is taken as linear between samples. How closely the samples are spaced
depends on whether the spacecraft is holding an attitude or slewing.

A held attitude is sampled at the simulation's step, which is the ephemeris
resolution. That is enough, since the torques on a held attitude vary only with
the orbit.

A slew is sampled more finely, wherever it falls. The simulations ask the ACS for
the attitude flown at intermediate times within the step. The intermediate times
are spaced every ``max_attitude_step_deg`` (5° by default) of motion at the
fastest configured slew rate, and the orbit is interpolated between the step's
ends. The planners' forecast samples the same way.

Tracking therefore works at whatever resolution the ephemeris has. The attitude
step is a convergence control. It does not guarantee a bound on the integration
error.

Configuration
-------------

The settings below are under ``spacecraft_bus.attitude_control.stored_momentum``.

* ``gravity_gradient_enabled``, ``solar_radiation_pressure_enabled`` and
  ``aerodynamic_drag_enabled`` (bool, default ``False``) select the sources that
  are tracked.
* ``atmosphere`` (:class:`~conops.config.AtmosphereConfig`) holds ``model``
  (``table`` or ``msis``), ``f107``, ``f107a`` and ``ap`` (150, 150 and 4 by
  default), and ``density_scale``.
* ``geomagnetic_field`` is ``dipole`` or ``igrf``. The default is ``dipole``.
* ``magnetorquer_max_dipole_a_m2`` (float or None) and
  ``magnetorquer_time_constant_s`` (default 1800) describe the torquers. Set the
  dipole to None for a spacecraft without torquers.
* ``residual_dipole_body_a_m2`` (three floats or None) is the spacecraft's own
  dipole.
* ``initial_momentum_body_n_m_s`` (three floats) is the wheels' momentum at the
  start.
* ``max_attitude_step_deg`` (default 5) is the attitude step along slews.
* ``max_sample_interval_s`` (default 10, at most 10) applies where
  ``enforce_capacity`` is on. It is the longest interval, in seconds, that the
  enforcement forecast integrates over. The interval is shortened further to
  ``max_attitude_step_deg`` of motion at the fastest configured slew rate. With a
  torque source enabled, the ephemeris must be sampled at least this finely, and
  an error is raised otherwise. The run's own tracker samples at the ephemeris
  resolution whatever this setting is.
* ``capacity_n_m_s`` (float or None) is the wheels' capacity, taken as the same
  about every axis. None sets no capacity.
* ``planning_level`` (default 0.8) is the fraction of the capacity at which a
  thruster dump is called for. The rest of the capacity is margin.
* ``thruster_dump`` (:class:`~conops.config.ThrusterDumpConfig` or None) describes
  the thruster dumps. ``duration_seconds`` (default 120) is how long the dump
  holds the spacecraft, and ``residual_fraction`` (default 0) is the fraction it
  leaves in the wheels. ``attitude``
  (:class:`~conops.config.activity_attitude.ActivityAttitude`) is the attitude the
  dump is fired at. Its ``kind`` is ``inherited`` (the default), which holds the
  attitude the spacecraft already has, or ``fixed``, which takes ``ra_deg``,
  ``dec_deg`` and ``roll_deg`` (see `Dumps`_). The ``power_positive`` kind is
  rejected for a dump. A dump needs a capacity. None means the spacecraft has no
  thrusters, and nothing empties the wheels. The telemetry shows the stored
  momentum as it grows, a planner logs where its forecast passes the capacity,
  and a planned simulation records a ``momentum_capacity`` event at the capacity
  (see `At the capacity`_). A queue simulation records no such event without
  thrusters.
* ``enforce_capacity`` (bool, default ``False``) enforces the capacity on the
  executed motion as well. Every slew and hold is certified against the capacity
  before it is flown, and an observation is admitted only where it fits. The run
  fails otherwise (see the capacity enforcement under :doc:`configuration`).
  Enforcement integrates into the same tracker as everything on this page, so it
  combines with the dumps: a dump empties the wheels, and enforcement forecasts on
  from what they hold.
* ``slew_momentum_enabled`` (bool, default ``False``) says whether the momentum
  that the body carries while it slews counts against the capacity (see
  `Slews`_). It needs a capacity and an inertia tensor. Turning it on enables
  tracking, even with no torque source enabled.
* ``slew_rate_fractions`` (floats, default ``[1, 0.75, 0.5, 0.25, 0.125]``) lists
  the fractions of the configured slew rates at which a slew may be flown where
  its momentum does not fit. The list descends from 1, and the first fraction
  that fits is taken.
* ``roll_step_deg`` (integer, default 30) applies where ``enforce_capacity`` is
  on. It is the step, in degrees, of the rolls that are tried when an observation
  is admitted. The observation's preferred roll is tried first, and then every
  multiple of the step from 0° to below 360°. An observation that commands its
  own roll is tried at that roll only. The step must divide 360.

The following rules apply to the configuration as a whole:

* The inertia tensor's principal moments must be positive and satisfy the
  triangle inequalities. A planar body's equality is allowed within rounding.
* Enabling any source with no inertia tensor is an error.
* The tracker reloads the configuration at the start of every run, so changes
  between runs take effect without rebuilding the simulation.

A minimal configuration that tracks gravity gradient and drag with thruster dumps:

.. code-block:: yaml

   spacecraft_bus:
     inertia_tensor_body_kg_m2: [[1000, 0, 0], [0, 1200, 0], [0, 0, 800]]
     attitude_control:
       stored_momentum:
         gravity_gradient_enabled: true
         aerodynamic_drag_enabled: true
         atmosphere: {model: msis, f107: 120, f107a: 120, ap: 8}
         capacity_n_m_s: 4.0
         planning_level: 0.8
         thruster_dump: {duration_seconds: 180, residual_fraction: 0.05}

Dumps
-----

Two mechanisms unload the wheels, and they are different in kind.

Magnetorquers unload continuously. They are a torque source in the integrator and
need no block of time in the schedule.

A thruster dump is a hold during which no science is observed. It is a
desaturation activity (``ScheduledActivity.desaturation``) that holds an attitude
for the dump's length, in the ``DESAT`` ACS mode, and unloads the wheels when it
ends. By default it holds the attitude the spacecraft already has. While the dump
holds:

* the telemetry reports ``DESAT``;
* an instrument that defines a mode named ``DESAT`` is in that mode;
* the attitude constraint scopes take an entry for ``DESAT``, which is hardware
  safety unless set;
* the bus power table takes an entry for ``DESAT``, which is the nominal power
  unless set.

Dumps are never timer-based. By default a dump does not change the attitude.

A firing attitude can be fixed instead, by setting ``thruster_dump.attitude`` to
the ``fixed`` kind with an RA, a Dec and a roll. The spacecraft then slews to that
attitude before every dump, and the dump's hold begins when the slew has ended.
The attitude is used as it is given. COAST does not work one out from the
thrusters' mounting or from the direction of the Sun.

Dispatch does not start the slew to a fixed firing attitude while a visit of an
observation sequence is under way. The dump waits for the visit to end. If the
slew or the hold at the fixed attitude would violate the attitude constraints,
the run stops with an activity scheduling error. The planners plan the slew from
the attitude of whatever precedes the dump, and the dump begins when that slew
ends.

The rest of this section describes when a thruster dump happens. Dispatch, which
selects observations as a queue simulation runs, calls for dumps from the momentum
the wheels hold. The planners reserve dumps in a plan from a forecast.

Dispatch
^^^^^^^^

Dispatch calls for a dump when the stored momentum reaches the planning level,
which is the ``planning_level`` fraction of the capacity. The dump goes between
observations. The observation under way runs on, and the dump begins once it has
ended and before another starts. The margin between the planning level and the
capacity is what this wait draws on.

Dispatch also looks ahead. Before it begins an observation, it forecasts the
momentum over that observation, starting from the wheels as they are. A visit of
an observation sequence (see :doc:`planning`) is forecast whole: every step at its
own pointing, with the slews between the steps and the cleanup after the last.
What dispatch does next depends on what the forecast reaches.

* Where the forecast reaches the planning level before the observation would
  collect, the wheels are emptied first: the dump goes before the observation.
* Where the forecast reaches the planning level during the observation but does
  not reach the capacity, the observation runs and the dump follows it.
* Where the forecast would reach the capacity during a plain observation, the
  observation is bounded to end where the planning level is reached, and the dump
  follows. If the observation's minimum snapshot is collected later than that and
  still fits, the observation ends there instead. If the minimum snapshot does
  not fit within the capacity from the wheels as they are, the wheels are emptied
  first and the observation is left for a later selection. If it does not fit
  even from empty wheels, the observation is held out of this selection and the
  reason is logged. It is not marked done, because the torque varies round the
  orbit and the minimum snapshot may fit later.
* Where the forecast would reach the capacity during a visit, the visit is not
  cut. The wheels are emptied first if the visit fits from empty wheels. If it
  does not fit even then, the visit is rejected and marked done, and the reason
  is logged.

At the capacity
^^^^^^^^^^^^^^^

The capacity is a red line. Where the stored momentum reaches it, fault
management records a ``momentum_capacity`` event against the ``stored_momentum``
state. The event names the observation under way, if there is one. No safe mode
is requested. A queue simulation records the event only where thruster dumps are
configured.

The dump is then an emergency and does not wait for the observation to end. It
begins once the observation's cleanup and handoff budget is paid, and the
observation's collection is stopped a budget before the dump begins. The same is
done before a target-of-opportunity preemption. When the wheels are unloaded, a
second event clears the state.

A planned simulation records the event the same way, against the plan entry under
way. It relies on the plan's dumps, and unloads the wheels at the end of each
entry marked ``desaturation``.

Planners
^^^^^^^^

Every plan built with momentum tracking on is forecast for it.
:class:`~conops.schedulers.momentum.MomentumForecast` integrates the configured
sources along the plan's timeline with the same tracker. It integrates each slew
along its path, each observation, contact, charge and activity at its attitude,
and the idle time between them at the attitude held. Each thruster dump unloads
the wheels as it ends.

Where the forecast reaches the planning level and the spacecraft has thrusters,
the planner reserves a dump. The dump is a desaturation activity that holds the
attitude of whatever precedes it, or the fixed firing attitude where one is
configured. The planner keeps the science placed before the dump as it is, and
plans on from there with the dumps reserved so far.

As in dispatch, the dump goes between observations. Its place depends on the
observation inside which the planning level is reached:

* The dump goes after that observation where the capacity is not reached before
  the dump would end.
* Where the capacity would be reached by then, and the observation fits from
  empty wheels, the dump goes before the observation.
* Where the capacity would be reached by then, and the observation does not fit
  from empty wheels, a visit is left out and a plain observation is cut at the
  planning level.

Where the planning level is reached outside any observation, the dump goes at
that time. If that time falls inside a scheduled activity or a tracked ground
station pass, the dump goes after it where the capacity is not reached before the
dump would end, and before it otherwise.

A dump found earlier than one already reserved replaces that dump and those after
it.

The planner works in passes. A pass reserves a dump for each crossing of the
planning level that the forecast reaches in turn, with the wheels unloaded at
each dump before it, and then builds the plan again once. A dump that overlaps
planned work, or that has a fixed firing attitude, is the last one its pass
reserves, because the plan after it has to be built again before later dumps can
be placed. The next pass checks those dumps against the plan as built. The passes of one plan share the spans integrated so
far. Each span is keyed on its times, its attitude or slew, and the momentum it
starts from, so a pass integrates only from its new dump onwards.

A rolling-horizon run gives each replan the momentum it will have when its horizon
starts.

Without thrusters the plan stands, and the log says where the forecast passes the
capacity. See :doc:`planning`, Planning within the wheels.

Slews
-----

The momentum tracked is the whole spacecraft's. While the body turns at an angular
velocity :math:`\omega`, the body itself carries :math:`I\omega` of that momentum
and the wheels hold the rest. What the wheels hold is the actuator momentum,
:math:`R H - I\omega` in body coordinates, where :math:`H` is the stored vector in
inertial coordinates. At rest the stored momentum and the actuator momentum are
the same.

Along a slew the body's share is large. A 1000 kg m² body turning at 0.5°/s
carries 8.7 N m s, an order of magnitude above a day's build-up from the
environment on the example orbit. The simulations and the planners' forecast
therefore give the tracker the body's rate at every sample. The simulations take
the rate from the ACS's executed motion, and the forecast takes it from the slew's
profile. The telemetry reports what the wheels hold beside the stored momentum.

What the capacity means
^^^^^^^^^^^^^^^^^^^^^^^

What ``capacity_n_m_s`` means is set by ``slew_momentum_enabled``.

* Off, which is the default, the capacity is the room left for stored momentum at
  the configured slew rates. Only the stored momentum is checked against it. The
  body's share is reported and not counted.
* On, the capacity is the wheels' whole capacity, and what the wheels hold along
  each slew is checked against it. A slew's momentum must fit the room that the
  stored momentum leaves.

The body's share is a transient and is not what a dump removes. The planning
level, and the dumps it calls for, are therefore on the stored momentum alone. The
capacity, in fault management and in the forecast's checks, is on what the wheels
hold.

Slower slews
^^^^^^^^^^^^

With ``slew_momentum_enabled`` on, a slew whose momentum does not fit is flown
slower.

Dispatch does this after it has chosen an observation. It forecasts what the
wheels hold over the slew into the observation, starting from the wheels as they
are, at each of ``slew_rate_fractions`` in turn, and flies the slew at the first
fraction that fits. Acceleration and settling are unchanged, so the slew takes
longer and the observation's collection starts later.

The planners do the same for a plan. Where the forecast finds a slew in which the
wheels reach the capacity, the observation's slew is capped at the next fraction
and the plan is built again. Every plan entry carries ``slew_rate_fraction``. A
planned simulation flies the slew at that fraction, and dispatch honours it as the
first fraction it tries.

Slews that fit at no rate
^^^^^^^^^^^^^^^^^^^^^^^^^

Slowing a slew is not always enough. There are two cases.

In the first, no fraction fits from the wheels as they are, but the slowest
fraction fits from empty wheels. The wheels are then emptied first. Dispatch calls
for a thruster dump and the observation follows it, and the planners reserve a
dump before the observation. Without thrusters, dispatch holds the observation
out, and a plan stands with the overload logged.

In the second, the slowest fraction does not fit even from empty wheels, so the
slew cannot be flown within the capacity at all. Dispatch holds the observation
out and marks it done, and the planners leave it out.

Slews inside a visit
^^^^^^^^^^^^^^^^^^^^

In the forecast that dispatch makes at selection, a visit's later steps are slewed
to at full rate, and the forecast holds the attitude through those slews. Only the
slew into the first step carries the body's momentum in that forecast. The
planners forecast every slew along a plan, step slews included.

Telemetry and reporting
-----------------------

Housekeeping telemetry carries the following at every step:

* each source's torque by name (``disturbance_torques_body_n_m``) and their total
  (``disturbance_torque_body_n_m``);
* the stored momentum in body coordinates and its norm
  (``stored_momentum_body_n_m_s``, ``stored_momentum_norm_n_m_s``);
* what the wheels hold, which is the stored momentum less the body's own while it
  turns (``actuator_momentum_body_n_m_s``, ``actuator_momentum_norm_n_m_s``);
* the geomagnetic field in body coordinates, where a source uses one
  (``geomagnetic_field_body_t``);
* the atmosphere's density, where drag is tracked (``atmosphere_density_kg_m3``).

The event log carries ``MOMENTUM`` events for each dump called for, each unload,
each slew flown slower and each selection decision the forecast makes. Fault
management carries the capacity events.

The timelines (:doc:`visualization`) draw dumps on the activities row. The
momentum example notebook plots the stored momentum with each source's share,
emptied at every dump.

What is left out
----------------

* The momentum of a slew between the samples. The body's rate is sampled every
  ``max_attitude_step_deg`` of motion, so a peak between samples is read a sample
  late. The attitude step is the convergence control.
* Self-shadowing between plates, for sunlight and drag alike.
* A thruster firing attitude worked out from the thrusters' mounting or from the
  Sun. A dump holds the attitude the spacecraft has, or the fixed attitude set in
  ``thruster_dump.attitude``.
* Propellant accounting for dumps.
* The accuracy of the environment. The built-in atmosphere table and the dipole
  field are order-of-magnitude models. The optional ``msis`` and ``igrf`` models
  are the ones to use where the numbers matter. The density's solar-activity
  inputs are what the run assumes and are not a forecast.
