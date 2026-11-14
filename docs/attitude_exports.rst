Executed attitude exports
=========================

A simulation retains the attitude trajectory it actually executed. After the
simulation has advanced, ``run.sample_attitude`` evaluates that retained
trajectory at an interval you choose, independently of the telemetry step.
Here ``run`` is the simulation object.

``sample_attitude`` recalls interrupted slews, braking arcs, tracking and holds
through ``AttitudeExecutor.flown``. It does not interpolate the coarse
housekeeping quaternions.

.. code-block:: python

   attitude = run.sample_attitude(
       begin="2029-03-01T01:00:00+00:00",
       end="2029-03-01T01:10:00+00:00",
       interval_seconds=1.0,
   )
   attitude.save_stk("attitude.a", reference_frame="J2000")
   attitude.save_euler_csv("attitude.csv", reference_frame="J2000")

The ``begin`` and ``end`` bounds accept UTC datetimes, ISO strings or Unix
seconds. Both must lie within ``run.acs._executor.flown_interval``, and an
export that asks for future or unflown times is rejected. If you omit ``end``,
it defaults to the last time to which the executor physically advanced. That
time may precede the requested simulation end by one step.

Both endpoints are included in the samples. By default the samples also include
the boundaries of each executed motion and the handoffs between motions, so
adjacent samples are at most the requested interval apart. Set
``include_transitions=False`` to retain only the regular grid and the final
endpoint.

Lifecycle and body frame
------------------------

Keeping and reloading a series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sampling needs the run, so keep the run until sampling is complete. After that,
either retain the returned series or attach it to a plan so that it is saved
with the plan:

.. code-block:: python

   run.plan.attitude_timeseries = attitude
   run.plan.save("plan.json")

   from conops.targets import Plan
   loaded = Plan.load("plan.json", load_timeseries=True)
   loaded.attitude_timeseries.save_stk("reloaded.a")

A series attached to a plan is saved in a separate sidecar file beside the plan.
``Plan.load`` loads only the plan entries by default, so request the sidecars
explicitly with ``load_timeseries=True``.

A saved fine series can be exported again. A saved coarse series does not
contain enough information to recover finer executed motion. The export
metadata records ``sample_source`` and the requested interval, so you can tell
which kind of series you have.

Body frame
~~~~~~~~~~

The quaternions already map GCRS coordinates into the configured physical body
frame. If the configuration declares
``SpacecraftFrame(boresight=(0, 0, 1), sun_side=(1, 0, 0))``, for example, the
notebook needs no additional quaternion rotation. The series records this frame
explicitly.

The series also has legacy RA/Dec/roll fields. These parameterize physical body
+X. They do not redefine the configured telescope boresight.

Origin offset
~~~~~~~~~~~~~

``origin_offset_m`` is optional metadata: a vector of three components in
metres. It does not change an orientation, and it does not translate the
exported orbit. A scalar distance from another spacecraft origin does not
determine this vector, so supply its components only when they are known.

Reference axes and quaternion convention
----------------------------------------

Reference frame
~~~~~~~~~~~~~~~

The ``reference_frame`` argument chooses the axes in which the attitude is
exported:

* ``reference_frame="ICRF"`` preserves the GCRS orientation axes. The origin
  remains geocentric.
* ``"J2000"`` explicitly converts to the FK5/IAU76/80 Earth J2000 realization
  described in the `STK frame documentation
  <https://help.agi.com/stk/Content/stk/referenceFramesCBdescriptions.htm>`_.

The J2000 conversion relates the IAU2006/2000A GCRS-to-Earth-fixed rotation to
the IAU76/80 J2000-to-Earth-fixed rotation, including the
equation-of-equinoxes correction. Polar motion is common to both rotations and
cancels.

UTC, TT and UT1 are distinct time scales, and newly sampled series retain
UT1-UTC for every sample. An older sidecar may lack this information. J2000
export rejects such a sidecar; it does not relabel the sidecar's GCRS
quaternions as J2000.

These conversions are rotations of the attitude axes. Stellar aberration and
origin translations are not applied.

STK attitude file
~~~~~~~~~~~~~~~~~

The STK file uses ``AttitudeTimeQuatScalarFirst``, with an explicit UTC
``ScenarioEpoch``, Earth central body, coordinate axes, count and relative
seconds.

STK's `orientation documentation
<https://help.agi.com/stk/Content/stk/sn-orientation.htm>`_ specifies Shuster
quaternions, whereas COASTSim uses Hamilton quaternions. The exporter therefore
conjugates the vector part as well as putting the scalar first, which preserves
the reference-to-body DCM. Quaternion signs remain continuous through 180
degrees.

The accompanying ``.metadata.json`` records representation, reference frame and
spacecraft axes. See also the `STK attitude file specification
<https://help.agi.com/stk/content/stk/importfiles-01.htm>`_.

Euler CSV for thermal analysis
------------------------------

The CSV contains the following columns: UTC timestamp, elapsed seconds, Euler
angles 1/2/3 in degrees, a gimbal-lock flag, and the angles of body +X and +Z
to the Sun.

Euler angle convention
~~~~~~~~~~~~~~~~~~~~~~

Rotations are intrinsic: a physical rotation about X, then Y', then Z''.
Precisely, ``R_reference_to_body = (Rx(a) Ry(b) Rz(c)).T``.

At gimbal lock the third angle is zero and the other two preserve the same
orientation. The decomposition uses ``atan2`` near the singularity to avoid
loss of precision from ``asin``. Euler angles themselves can wrap or change
branch at a singularity. The underlying quaternion remains continuous.

Sun angles
~~~~~~~~~~

The Sun direction is linearly interpolated from the orbit ephemeris and
expressed using the executed quaternion. Missing Sun vectors produce empty
angle columns.

Importing into a thermal tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Map the declared columns and rotation convention when importing into the
thermal-analysis tool. Validate that import in the intended STK/Thermal Desktop
installation before using the files as release evidence. A numerical round
trip does not show that a downstream application imports the files correctly.
