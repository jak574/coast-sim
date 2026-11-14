Target of Opportunity (TOO)
===========================

Overview
--------

A Target of Opportunity (TOO) is a time-critical observation that arrives while the spacecraft is working through its normal queue and may interrupt the observation in progress. The TOO system lets a simulation model the response to transient astronomical events such as gamma-ray bursts (GRBs), gravitational wave counterparts, supernovae, or other phenomena that require immediate observation.

In COASTSim a TOO is an ordinary queue request with a short window. Each TOO has a ``submit_time``. When the simulation reaches that time, the TOO joins the target queue with its own merit and optional deadline, and the queue selects it in the same way as any other target. If the TOO is visible and outranks the observation in progress, that observation is preempted and the queue selects again.

Key Features
------------

* An active TOO is an ordinary queue target. Nothing boosts its merit automatically.
* Preemption is decided on merit. Every observation keeps the value it had when it was selected, which this page calls its frozen value. A TOO interrupts only when its own value, evaluated at the current time, beats the frozen value of the current observation.
* A TOO can have a deadline. The deadline drives the urgency term of the merit, and the TOO can no longer be selected once the deadline has passed.
* A TOO interrupts only when its target is observable at that moment.
* Two controls limit interruption. A TOO can be submitted so that it waits for the current observation to end, and the observations of a category can be protected from interruption.
* A TOO can be scheduled to become active at a future time.
* The simulation logs when each TOO is queued, each interrupt, and the merit breakdown of each selection.

TOORequest Model
----------------

The ``TOORequest`` class is a Pydantic model that represents a pending TOO. It holds the information needed to observe the target and to track whether the TOO has been observed.

.. code-block:: python

   from conops.ditl import TOORequest

   too = TOORequest(
       obsid=1000001,        # Unique observation ID
       ra=180.0,             # Right ascension (degrees)
       dec=45.0,             # Declination (degrees)
       merit=10000.0,        # Priority (higher = more urgent)
       exptime=3600,         # Exposure time (seconds)
       name="GRB 250101A",   # Human-readable name
       submit_time=0.0,      # When TOO becomes active (Unix timestamp)
       deadline=None,        # Latest start of science (Unix timestamp), or None
       interrupt=True,       # Whether it may cut short the current observation
       executed=False,       # Whether TOO has been dispatched
   )

Attributes
^^^^^^^^^^

* ``obsid`` (int): Unique observation identifier for this TOO
* ``ra`` (float): Right ascension in degrees
* ``dec`` (float): Declination in degrees
* ``merit`` (float): Base merit. The TOO interrupts only if its value exceeds the value of the current observation
* ``exptime`` (int): Requested exposure time in seconds
* ``name`` (str): Human-readable name for the TOO target (e.g., "GRB 250101A")
* ``submit_time`` (float): Unix timestamp when the TOO becomes active. Default is 0.0, meaning active from simulation start
* ``deadline`` (float or None): Latest Unix time at which science collection may begin. The deadline drives the urgency term of the merit (``config.targets.urgency_weight``). The TOO cannot be selected after the deadline
* ``interrupt`` (bool): Whether the TOO may cut short the observation in progress. Default is True. If False, the TOO waits in the queue for that observation to end
* ``executed`` (bool): Whether this TOO has been dispatched. Set to True once it becomes the current observation

Submitting TOOs
---------------

Use the ``submit_too()`` method on ``QueueDITL`` to register a TOO:

.. code-block:: python

   from conops import MissionConfig, QueueDITL
   from rust_ephem import TLEEphemeris
   from datetime import datetime, timedelta

   # Set up simulation
   cfg = MissionConfig.from_json_file("examples/example_config.json")
   begin = datetime(2025, 1, 1, 0, 0, 0)
   end = begin + timedelta(days=1)
   ephem = TLEEphemeris(tle="examples/example.tle", begin=begin, end=end)

   ditl = QueueDITL(config=cfg, ephem=ephem, begin=begin, end=end)

   # Submit a TOO that is active immediately
   ditl.submit_too(
       obsid=1000001,
       ra=180.0,
       dec=45.0,
       merit=10000.0,
       exptime=3600,
       name="GRB 250101A",
   )

   # Run the simulation
   ditl.calc()

Scheduled TOOs
^^^^^^^^^^^^^^

A TOO can be scheduled to become active at a future time by passing ``submit_time``, either as a Unix timestamp or as a ``datetime``. This is useful for simulating a TOO alert that arrives in the middle of an observation:

.. code-block:: python

   # TOO becomes active 2 hours into the simulation (using Unix timestamp)
   ditl.submit_too(
       obsid=1000002,
       ra=90.0,
       dec=-30.0,
       merit=10000.0,
       exptime=1800,
       name="GRB 250101B",
       submit_time=ditl.ustart + 7200,  # 2 hours after start
   )

   # Or use a datetime object
   from datetime import datetime
   ditl.submit_too(
       obsid=1000003,
       ra=270.0,
       dec=60.0,
       merit=10000.0,
       exptime=2400,
       name="GW Event",
       submit_time=datetime(2025, 1, 1, 6, 0, 0),
   )

How TOO Interrupts Work
-----------------------

The simulation checks for TOO interrupts at each step while the spacecraft is in science or idle mode. The check has two stages. The first stage decides whether any TOO is a candidate to interrupt. The second stage confirms that a replacement can be admitted before the current observation is stopped.

Checking for candidates
^^^^^^^^^^^^^^^^^^^^^^^

At each step the simulation does the following:

1. It adds every TOO whose ``submit_time`` has passed to the queue, with its own merit and deadline. Each TOO is added only once.
2. It marks a TOO as executed if that TOO is already the current observation or has collected science.
3. If an observation is in progress, a pending TOO becomes a candidate to interrupt it when all of the following hold:

   * the TOO may interrupt (``interrupt``) and the category of the observation is ``interruptible``;
   * the deadline of the TOO, if it has one, has not passed;
   * the target of the TOO is visible now; and
   * the value of the TOO, evaluated now, can beat the frozen value of the current observation.

The last condition needs one qualification when a long-range allocation is in use (see :doc:`planning`). An allocation adds a bonus to the value of a target, and the size of that bonus depends on the collection that admission settles on. The recorder or a deadline may cut that collection shorter than a snapshot, so the bonus is not known at this stage. The check therefore takes the value of the TOO with the most that the allocation can add. The value as admitted is compared in the next stage.

If no observation is in progress, nothing is interrupted. The TOO is already in the queue and is considered at the next selection.

Carrying out the interrupt
^^^^^^^^^^^^^^^^^^^^^^^^^^

When a TOO is a candidate, the simulation proceeds as follows:

1. The queue selects again, without commanding anything yet. The interrupted target is held out of this one selection so that it cannot simply resume. The queue chooses the highest-ranked target, which is normally the TOO. The value of that target as admitted, taken over the collection that was admitted, must beat the frozen value of the current observation. If it does not, the observation runs on and the TOO waits for the next selection.
2. The current observation is terminated. It is recorded as preempted and does not count as completed. The slew to the replacement is commanded.
3. The TOO is marked as executed if it was the target selected.
4. The events and the merit breakdown of the TOO are logged.

Preventing Interrupts
^^^^^^^^^^^^^^^^^^^^^

There are two ways to stop a TOO from interrupting. One is set on the TOO and the other on the observation category.

A TOO submitted with ``interrupt=False`` never cuts short the observation in progress. It is queued as usual and competes at the next selection, which happens when that observation ends. If a different TOO interrupts the observation, the TOO with ``interrupt=False`` does not take the place of the interrupted observation either. In a rolling-horizon replan it is planned in after the interrupting TOO (see :doc:`planning`).

.. code-block:: python

   ditl.submit_too(
       obsid=1000004,
       ra=120.0,
       dec=-20.0,
       merit=10000.0,
       exptime=1800,
       name="Follow-up",
       interrupt=False,  # wait for the current observation to finish
   )

To protect a kind of observation from every TOO, set ``interruptible=False`` on its observation category (see :doc:`configuration`). Observations in that category run to their end. A TOO that is waiting for one of them is selected when the observation finishes, provided the TOO still ranks highest.

.. code-block:: python

   from conops.config import ObservationCategory

   ObservationCategory(
       name="Calibration",
       obsid_min=90000,
       obsid_max=91000,
       interruptible=False,
   )

Both controls also apply to :class:`~conops.ditl.RollingHorizonDITL`. There, a TOO triggers a rapid replan. When either control prevents the TOO from interrupting, the replan starts the TOO once the observation in progress ends.

Making TOOs Win
---------------

A TOO receives no automatic merit boost. It outranks other work in the queue only when its value is higher, and there are two ways to arrange that:

* Give the TOO a base merit above that of normal targets. Normal targets typically use a merit of 1-1000.
* Give the TOO a ``deadline`` and set ``config.targets.urgency_weight``. The value of the TOO then rises as the deadline approaches.

The example below uses the second approach:

.. code-block:: python

   cfg.targets.urgency_weight = 50.0

   ditl.submit_too(
       obsid=1000001,
       ra=180.0,
       dec=45.0,
       merit=1000.0,
       exptime=3600,
       name="GRB 250101A",
       submit_time=ditl.ustart + 3600,
       deadline=ditl.ustart + 3600 + 4 * 3600,  # photons on target within 4 hours
   )

Accessing TOO Status
--------------------

Every submitted TOO is kept in a register, which is available as ``ditl.too_register``. After the simulation has run, the ``executed`` flag of each entry shows whether that TOO was dispatched:

.. code-block:: python

   # After running the simulation
   ditl.calc()

   # Check all TOOs
   for too in ditl.too_register:
       status = "Executed" if too.executed else "Pending"
       print(f"{too.name}: {status}")

   # Find executed TOOs
   executed_toos = [t for t in ditl.too_register if t.executed]

   # Find TOOs that were never dispatched (not visible, outranked, deadline passed, etc.)
   missed_toos = [t for t in ditl.too_register if not t.executed]

Event Logging
-------------

TOO events are logged to ``ditl.log`` with the event type ``"TOO"``. You can filter the log for these events:

.. code-block:: python

   # Get all TOO-related events
   too_events = [e for e in ditl.log.events if e.event_type == "TOO"]

   for event in too_events:
       print(f"{event.time_formatted}: {event.description}")

Example output::

   2025-01-01T02:30:00Z: Queued TOO GRB 250101A (obsid=1000001, merit=10000.0)
   2025-01-01T02:34:56Z: TOO interrupt: GRB 250101A (obsid=1000001, base=10000 value=10000.000 score=10000.000) preempting current observation (value=500)

Complete Example
----------------

The example below creates a queue of three normal targets, submits a TOO that arrives three hours into the simulation, and then inspects the result.

.. code-block:: python

   from conops import MissionConfig, QueueDITL
   from conops.targets import Queue
   from rust_ephem import TLEEphemeris
   from datetime import datetime, timedelta

   # Configuration
   cfg = MissionConfig.from_json_file("examples/example_config.json")
   begin = datetime(2025, 6, 1, 0, 0, 0)
   end = begin + timedelta(days=1)
   ephem = TLEEphemeris(tle="examples/example.tle", begin=begin, end=end)

   # Create queue with normal targets
   queue = Queue(config=cfg, ephem=ephem)
   queue.add(ra=0.0, dec=0.0, obsid=1, name="Target 1", merit=100.0, exptime=3600)
   queue.add(ra=45.0, dec=30.0, obsid=2, name="Target 2", merit=200.0, exptime=3600)
   queue.add(ra=90.0, dec=60.0, obsid=3, name="Target 3", merit=150.0, exptime=3600)

   # Create DITL
   ditl = QueueDITL(config=cfg, ephem=ephem, begin=begin, end=end, queue=queue)

   # Submit a TOO that arrives 3 hours into the simulation
   ditl.submit_too(
       obsid=9999,
       ra=120.0,
       dec=20.0,
       merit=10000.0,
       exptime=7200,
       name="GRB 250601A",
       submit_time=ditl.ustart + 10800,  # 3 hours
   )

   # Run simulation
   ditl.calc()

   # Analyze results
   print(f"TOO executed: {ditl.too_register[0].executed}")

   # Find the TOO observation in the plan
   too_obs = [p for p in ditl.plan if p.obsid == 9999]
   if too_obs:
       print(f"TOO observed for {too_obs[0].exposure_time} seconds")

   # Check which observation was preempted
   too_events = [e for e in ditl.log.events if e.event_type == "TOO"]
   for event in too_events:
       print(event.description)

API Reference
-------------

.. autoclass:: conops.ditl.TOORequest
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automethod:: conops.ditl.QueueDITL.submit_too
   :no-index:
