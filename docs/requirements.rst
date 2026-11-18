Mission Requirements
====================

A requirement is a statement about a mission that a plan or a run either meets
or violates. Two examples are "a slew of up to 30 degrees completes, settled,
within 300 seconds" and "the battery stays at or above 40% charge". COASTSim
checks the requirements a mission configures and reports each violation.

Requirements are checks, not constraints. A scheduler does not change a plan to
meet a requirement, and a run does not stop when one is violated. To forbid
something, use a constraint (see :doc:`configuration`) or fault management (see
:doc:`fault_management`); use a requirement to measure whether the design, as
simulated, does what the mission asks of it.

Configuring requirements
------------------------

Requirements are a list on the mission configuration, ``MissionConfig.requirements``.
Each has a ``name``, which must be unique in the mission, and an optional
``description`` for the requirement's wording.

.. code-block:: python

   from conops.requirements import (
       SlewClass,
       SlewTimeRequirement,
       TelemetryLimitRequirement,
   )

   config.requirements = [
       SlewTimeRequirement(
           classes=[
               SlewClass(name="small", max_great_circle_deg=30.0, max_seconds=300.0),
               SlewClass(name="large", max_great_circle_deg=180.0, max_seconds=1500.0),
           ]
       ),
       TelemetryLimitRequirement(
           name="Battery depth of discharge",
           description="The battery shall not fall below 40% state of charge.",
           quantity="battery_level",
           minimum=0.4,
       ),
   ]

In a JSON or YAML mission file each requirement has a ``kind`` that selects its
type:

.. code-block:: yaml

   requirements:
     - kind: slew_time
       name: Slew time
       classes:
         - {name: small, max_great_circle_deg: 30.0, max_seconds: 300.0}
         - {name: large, max_great_circle_deg: 180.0, max_seconds: 1500.0}
     - kind: telemetry_limit
       name: Battery depth of discharge
       quantity: battery_level
       minimum: 0.4

Checking requirements
---------------------

After a run, ``check_requirements()`` on the simulation object returns a
:class:`~conops.requirements.base.RequirementReport`:

.. code-block:: python

   ditl.calc()
   report = ditl.check_requirements()

   print(report)              # one line per requirement
   report.met                 # True when every requirement was evaluated and met
   for violation in report.violations:
       print(violation.requirement, violation.utime, violation)

   battery = report.result("Battery depth of discharge")
   battery.status             # "met", "violated" or "not evaluated"
   battery.checked            # number of telemetry samples examined

Each :class:`~conops.requirements.base.RequirementViolation` has the name of the
requirement, the time, a message, and where they apply the observation ID, the
measured value and the limit.

``print_statistics()`` prints the same report in a "REQUIREMENTS" section.

A plan can be checked before it is executed with
:func:`~conops.requirements.evaluate.check_requirements`:

.. code-block:: python

   from conops.requirements import check_requirements

   plan = scheduler.schedule()
   report = check_requirements(config, list(plan))

Without telemetry, a requirement that needs it has the status
``"not evaluated"``, and ``report.met`` is False. The planners make this check
when they finish a plan and write each violation to the scheduler log.

Slew time
---------

:class:`~conops.requirements.slew.SlewTimeRequirement` (kind ``slew_time``)
states how long a slew may take, by class. Each
:class:`~conops.requirements.slew.SlewClass` has a ``name`` and two limits:

- ``max_great_circle_deg`` is the widest slew that belongs to the class,
  measured as the great-circle angle between the two pointings the slew joins.
- ``max_seconds`` is the longest time a slew of the class may take, including
  its settle.

A slew belongs to the narrowest class that is wide enough for it. A slew wider
than the widest class belongs to no class and is not checked.

Each plan entry records the great-circle angle of its slew
(``great_circle_deg``) beside the rotation actually flown (``slewdist``), which
is longer when the slew also changes the roll. The requirement uses the
great-circle angle, because that is how slew requirements are written.

``slew_class_summary()`` on the simulation object returns one
:class:`~conops.requirements.slew.SlewClassSummary` per class, with the number
of slews, the longest and the number over the allowance. ``print_statistics()``
prints it as a table.

Telemetry limits
----------------

:class:`~conops.requirements.telemetry.TelemetryLimitRequirement` (kind
``telemetry_limit``) states that a quantity in the housekeeping telemetry stays
within limits at every step of a run.

* ``quantity`` (str): The name of a housekeeping field, such as ``battery_level``,
  ``recorder_fill_fraction`` or ``sun_angle_deg``. See :doc:`telemetry` for the fields
* ``key`` (str, optional): For a field that holds a mapping, the entry to check
* ``minimum`` (float, optional): The least value allowed
* ``maximum`` (float, optional): The greatest value allowed
* ``modes`` (list[str], optional): The ACS modes, by name, in which the limits apply. Default: every mode

At least one of ``minimum`` and ``maximum`` is required. Samples where the
quantity is not recorded are not checked. Consecutive samples outside the limits
are reported as one violation, at the time of its first sample and with the
value furthest outside the limit.

.. code-block:: python

   # A radiator with a thermal mass stays between 250 K and 300 K
   TelemetryLimitRequirement(
       name="Bus radiator temperature",
       quantity="radiator_temperatures_k",
       key="BusRad-Y",
       minimum=250.0,
       maximum=300.0,
   )

   # The recorder is never more than 80% full while observing
   TelemetryLimitRequirement(
       name="Recorder margin",
       quantity="recorder_fill_fraction",
       maximum=0.8,
       modes=["SCIENCE"],
   )

Adding a kind of requirement
----------------------------

A new kind of requirement is a subclass of
:class:`~conops.requirements.base.Requirement`. It names its ``kind``, declares
its settings as fields, and implements ``evaluate``. No change to COASTSim is
needed.

.. code-block:: python

   from typing import ClassVar

   from conops.common import ACSMode
   from conops.requirements import (
       Requirement,
       RequirementEvidence,
       RequirementResult,
   )


   class MinimumScienceFraction(Requirement):
       """At least a fraction of the run is spent observing."""

       kind: ClassVar[str] = "minimum_science_fraction"

       minimum: float

       def evaluate(self, evidence: RequirementEvidence) -> RequirementResult:
           samples = evidence.housekeeping
           if samples is None:
               return self.not_evaluated("needs the telemetry of a run")
           science = sum(s.acs_mode == ACSMode.SCIENCE for s in samples)
           fraction = science / len(samples)
           violations = []
           if fraction < self.minimum:
               violations.append(
                   self.violation(
                       samples[0].timestamp.timestamp(),
                       f"observing for {fraction:.0%} of the run",
                       measured=fraction,
                       limit=self.minimum,
                   )
               )
           return self.result(len(samples), violations)


   config.requirements.append(
       MinimumScienceFraction(name="Observing efficiency", minimum=0.6)
   )

``evaluate`` receives a :class:`~conops.requirements.base.RequirementEvidence`
with three attributes:

* ``config``: the mission configuration
* ``entries``: the plan entries, each with its times, slew, pointing and observation ID
* ``housekeeping``: the telemetry samples of the run, or ``None`` for a plan that has not been executed

It returns ``self.result(checked, violations)``, where ``checked`` is the number
of items examined, or ``self.not_evaluated(reason)`` when the evidence it needs
is absent. ``self.violation(...)`` builds a violation with the requirement's
name filled in.

Defining the class registers its ``kind``. A mission file can then name it, as
long as the module that defines the class is imported before the file is
loaded:

.. code-block:: yaml

   requirements:
     - kind: minimum_science_fraction
       name: Observing efficiency
       minimum: 0.6

Two classes cannot use the same ``kind``. A subclass that sets no ``kind`` is
treated as a base class for other requirements and is not registered.

Mission files written before this page
--------------------------------------

Slew classes used to be configured on the ACS, as
``spacecraft_bus.attitude_control.slew_requirements``. A mission file that
still has that key loads: the classes become a ``slew_time`` requirement named
"Slew time", and the file is written back in the new form. In Python the
changes are:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Before
     - Now
   * - ``AttitudeControlSystem(slew_requirements=[SlewRequirement(...)])``
     - ``config.requirements = [SlewTimeRequirement(classes=[SlewClass(...)])]``
   * - ``conops.simulation.slew_requirements.check_plan(plan, acs)``
     - ``conops.requirements.check_requirements(config, list(plan)).violations``
   * - ``conops.simulation.slew_requirements.summarise_plan(plan, acs)``
     - ``requirement.summarise(plan)`` or ``ditl.slew_class_summary()``
   * - ``ditl.slew_requirement_mismatches()``
     - ``ditl.check_requirements().violations``
   * - ``acs.slew_class(angle)``
     - ``requirement.slew_class(angle)``
