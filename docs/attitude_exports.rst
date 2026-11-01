Executed attitude exports
=========================

After a simulation has advanced, ``run.sample_attitude`` evaluates the retained
executed trajectory independently of the telemetry step. It recalls interrupted
slews, braking arcs, tracking and holds through ``AttitudeExecutor.flown``;
it does not interpolate the coarse housekeeping quaternions.

.. code-block:: python

   attitude = run.sample_attitude(
       begin="2029-03-01T01:00:00+00:00",
       end="2029-03-01T01:10:00+00:00",
       interval_seconds=1.0,
   )
   attitude.save_stk("attitude.a", reference_frame="J2000")
   attitude.save_euler_csv("attitude.csv", reference_frame="J2000")

Bounds accept UTC datetimes, ISO strings or Unix seconds and must lie within
``run.acs._executor.flown_interval``. The default endpoint is the last physically
advanced executor time, which may precede the requested simulation end by one
step. Exports reject future or unflown times. Both endpoints are included;
by default, executed motion boundaries and handoffs are included too, so adjacent
samples are at most the requested interval apart. Set ``include_transitions=False``
to retain only the regular grid and final endpoint.

Lifecycle and body frame
------------------------

Keep the run until sampling is complete. Then retain the returned series or
attach it to a plan for persistence:

.. code-block:: python

   run.plan.attitude_timeseries = attitude
   run.plan.save("plan.json")

   from conops.targets import Plan
   loaded = Plan.load("plan.json", load_timeseries=True)
   loaded.attitude_timeseries.save_stk("reloaded.a")

``Plan.load`` defaults to loading only plan entries; explicitly request sidecars
with ``load_timeseries=True``. A saved fine series can be exported again. A
saved coarse series does not contain enough information to recover finer executed
motion. Export metadata records ``sample_source`` and the requested interval.

Quaternions already map GCRS coordinates into the configured physical body frame.
Declaring ``SpacecraftFrame(boresight=(0, 0, 1), sun_side=(1, 0, 0))`` therefore
needs no additional quaternion rotation in the notebook. The series records this
frame explicitly. Its legacy RA/Dec/roll fields parameterize physical body +X;
they do not redefine the configured telescope boresight.

``origin_offset_m`` is optional three-component metadata in metres. It does not
change an orientation or translate the exported orbit. A scalar distance from
another spacecraft origin does not determine this vector; supply its components
only when known.

Reference axes and quaternion convention
----------------------------------------

``reference_frame="ICRF"`` preserves the GCRS orientation axes, with the origin
remaining geocentric. ``"J2000"`` explicitly converts to the FK5/IAU76/80 Earth
J2000 realization described in the `STK frame documentation
<https://help.agi.com/stk/Content/stk/referenceFramesCBdescriptions.htm>`_. It relates
the IAU2006/2000A GCRS-to-Earth-fixed rotation to the IAU76/80 J2000-to-Earth-fixed
rotation, including the equation-of-equinoxes correction. Common polar motion
cancels. UTC, TT and UT1 are distinct; newly sampled series retain UT1-UTC for
every sample. J2000 export rejects an older sidecar missing this information
rather than relabeling its GCRS quaternions. These are attitude-axis rotations;
stellar aberration and origin translations are not applied.

The STK file uses ``AttitudeTimeQuatScalarFirst``, with an explicit UTC
``ScenarioEpoch``, Earth central body, coordinate axes, count and relative seconds.
STK's `orientation documentation
<https://help.agi.com/stk/Content/stk/sn-orientation.htm>`_ specifies Shuster
quaternions; COASTSim uses Hamilton quaternions. The exporter conjugates the vector
part as well as putting the scalar first, preserving the reference-to-body DCM.
Quaternion signs remain continuous through 180 degrees. The accompanying
``.metadata.json`` records representation, reference frame and spacecraft axes.
See also the `STK attitude file specification
<https://help.agi.com/stk/content/stk/importfiles-01.htm>`_.

Euler CSV for thermal analysis
------------------------------

The CSV contains UTC timestamp, elapsed seconds, Euler angles 1/2/3 in degrees,
a gimbal-lock flag, and the angles of body +X and +Z to the Sun. Rotations are
intrinsic: physical rotation about X, then Y', then Z''. Precisely,
``R_reference_to_body = (Rx(a) Ry(b) Rz(c)).T``. At gimbal lock the third angle
is zero and the other two preserve the same orientation. The decomposition uses
``atan2`` near the singularity to avoid loss of precision from ``asin``.

The Sun direction is linearly interpolated from the orbit ephemeris and expressed
using the executed quaternion. Missing Sun vectors produce empty angle columns.
Euler angles themselves can wrap or change branch at a singularity; the underlying
quaternion remains continuous. Map the declared columns and rotation convention
when importing into the thermal-analysis tool. Validate that import in the
intended STK/Thermal Desktop installation before using the files as release
evidence; a downstream application import is separate from numerical round trips.
