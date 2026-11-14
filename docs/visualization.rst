Visualization
=============

COASTSim includes plotting utilities for DITL simulation outputs and telemetry data.
They read the structured telemetry that a simulation records as it runs.

The plotting functions are implemented under `conops.visualization`. Each one accepts
an optional `config` parameter. When it is omitted, the function uses
`ditl.config.visualization` if that is available, and otherwise the defaults of the
`VisualizationConfig` model.

VisualizationConfig
-------------------

The `VisualizationConfig` model lives in `conops.config.visualization`. It holds the
configuration for fonts, colors, and timeline options.

Important fields:

- `font_family` (str): Font family used across plot titles, labels, and legends (default: `Helvetica`).
- `title_font_size` (int)
- `label_font_size` (int)
- `legend_font_size` (int)
- `tick_font_size` (int)
- `mode_colors` (dict[str, str]): Mode color mapping for ACS modes (e.g. `SCIENCE`, `SLEWING`, `SAA`).

Key plotting utilities
----------------------

The following plotting functions are available under `conops.visualization`:

- `plot_ditl_telemetry()` draws a basic multi-panel timeline of RA/Dec, ACS mode, battery,
  power, and observation IDs from telemetry data.
- `plot_data_management_telemetry()` plots recorder volume, fill fraction, generated and
  downlinked data, and alerts from telemetry data.
- `plot_acs_mode_distribution()` draws a pie chart of the time spent in each ACS mode,
  from telemetry data.
- `plot_ditl_timeline()` draws a timeline with orbit numbers, observations, slews, scheduled
  activities, SAA, and eclipses. Thruster momentum dumps are drawn separately from the
  other scheduled activities and labelled ``Momentum Dump``. The steps of a sequence visit
  are hatched over their observation. An antenna repoint during an observation is marked
  at the time it happens.
- `plot_sky_pointing()` draws an interactive Mollweide sky projection with the current
  pointing and constraints.
- `save_sky_pointing_movie()` exports the entire DITL sky pointing visualization as a movie
  (MP4, AVI, or GIF).
- `plot_spacecraft_3d()` draws a rotatable 3-D spacecraft model built directly from a
  ``MissionConfig``. It does not need a simulation.
  See :doc:`spacecraft_3d_visualization` for full details.

Reporting
---------

A finished simulation can also report on itself in numbers. ``print_statistics()`` prints
its statistics as a set of tables, and the methods below return the same statistics as
data. Each method corresponds to one of the printed tables.

``summary_by_category()``
~~~~~~~~~~~~~~~~~~~~~~~~~

This method returns one :class:`~conops.ditl.ditl_stats.CategorySummary` per observation
category. Each summary gives:

- the category's program;
- the number of targets observed;
- the number of visits taken, where a snapshot counts as one visit and each visit of a
  sequence counts as one;
- the science hours and the slew hours;
- the degrees slewed into the category's visits;
- the category's share of the run's science time;
- the fraction that the category's ``time_share`` allocates, where the category sets one;
- the category's time budget in hours and the fraction of it the run used, where the
  category sets a budget.

There is one row per configured category, in the order they are configured. A category
that collected nothing is included, with zeros. Science on a target whose observation ID
falls outside every configured category is counted under the default category, whose name
is the ``default_name`` of the observation categories (``Other`` in the default
configuration). That category gets a row after the configured ones, and only when it
collected science. The printed ``SCIENCE BY CATEGORY`` table
shows the share against the allocation.

``summary_by_group()``
~~~~~~~~~~~~~~~~~~~~~~

This method returns one :class:`~conops.ditl.ditl_stats.GroupSummary` per observation
group, in the configuration's order (see :doc:`configuration`, Observation groups). Each
summary gives:

- the number of targets in the group, and how many of those were observed;
- the group's visits and science hours;
- the group's time budget and the fraction of it the run used, where the group sets a
  budget;
- the group's window, where the group sets one.

The list is empty when no groups are configured. The printed ``SCIENCE BY GROUP`` table
follows the category table.

``slew_class_summary()``
~~~~~~~~~~~~~~~~~~~~~~~~

The ACS configuration can hold a table of slew requirement classes (see
:doc:`configuration`). This method returns one
:class:`~conops.simulation.slew_requirements.SlewClassSummary` per class, in the table's
order. Each summary gives:

- the number of slews that fall in the class, judged by great-circle angle;
- the duration of the longest of those slews, including its settling;
- how many of them took longer than the class allows.

A last row covers slews wider than the widest class. The list is empty when no table is
configured. The printed ``SLEWS BY REQUIREMENT CLASS`` table follows the group table.

``visit_table()``
~~~~~~~~~~~~~~~~~

This method returns one :class:`~conops.ditl.ditl_stats.VisitRecord` per visit. Each
record gives:

- the target and its category;
- the visit window, where the target has windows, and when the visit collected within it;
- the science collected;
- the slew into the visit and its roll;
- a status.

The status is ``complete``, ``cut`` or ``missed``. A visit is ``cut`` when it collected
less than planned, after allowing one step. A ``missed`` row stands for a window with no
visit.

Every attempt is a row. A visit that was cut short and taken again within its window
therefore appears as two rows. Pass ``per_window=True`` to get one row per window instead.
That row carries the verdict for the window and the number of attempts the visit took.

The windows come from the run's own requests. Pass ``targets`` to use another set. The
printed ``VISITS IN WINDOWS`` table lists the verdict per window for targets with windows.

``instrument_mode_hours()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method returns the hours that each instrument with modes spent in each mode. The
same figures are printed as ``INSTRUMENT MODE HOURS``.

Examples and advanced usage
---------------------------

As described above, the visualization functions take their configuration from the
optional `config` parameter, or from `ditl.config.visualization` if that is available.
The example below sets a few style options on the mission configuration and then draws
two plots, one using each route:

.. code-block:: python

   from conops import MissionConfig, QueueDITL
   from conops.visualization import plot_ditl_telemetry, plot_acs_mode_distribution
   from conops.config.visualization import VisualizationConfig
   from rust_ephem import TLEEphemeris
   from datetime import datetime, timedelta

   cfg = MissionConfig.from_json_file("examples/example_config.json")

   # Customize visual style
   cfg.visualization.font_family = "Helvetica"
   cfg.visualization.title_font_size = 14
   cfg.visualization.mode_colors["SAA"] = "#800080"

   begin = datetime.utcnow()
   end = begin + timedelta(days=1)
   ephem = TLEEphemeris(tle="examples/example.tle", begin=begin, end=end)

   ditl = QueueDITL(config=cfg, ephem=ephem)
   ditl.calc()

   fig, axes = plot_ditl_telemetry(ditl)
   fig2, ax2 = plot_acs_mode_distribution(ditl, config=cfg.visualization)

Working with Telemetry Data
----------------------------

All timeline and telemetry visualizations read COASTSim's structured telemetry.
The telemetry data is recorded automatically during a DITL simulation and stored in ``ditl.telemetry``.

.. code-block:: python

   from conops.ditl import QueueDITL
   from conops.visualization import plot_ditl_telemetry

   ditl = QueueDITL(config=config, ephem=ephemeris)
   ditl.calc()

   # Access structured telemetry data
   telemetry = ditl.telemetry
   housekeeping = telemetry.housekeeping  # All timestep data
   payload_data = telemetry.data          # Data generation events

   # Plot telemetry
   fig, axes = plot_ditl_telemetry(ditl)

For batch processing across many runs, persist logs with ``DITLLogStore`` and load events by ``run_id``:

.. code-block:: python

   from conops.ditl import DITLLogStore
   store = DITLLogStore("ditl_logs.sqlite")
   events = store.fetch_events("my-run-001")
   # feed events to your visualization pipeline as needed
   store.close()

Fonts and fallbacks
-------------------

If a requested font family is not installed on the system, Matplotlib falls back to
another font. To guarantee a specific font across platforms, provide a `FontProperties`
object with a path to the font file. Alternatively, ensure the font is installed on your system.

.. code-block:: python

   from matplotlib.font_manager import FontProperties
   fp = FontProperties(fname='/Library/Fonts/Helvetica.ttc')
   ax.set_title("Title", fontproperties=fp)

Example images
--------------

.. image:: /_static/visualization_acs_mode_distribution.png
   :alt: ACS mode distribution
   :align: center

.. image:: /_static/visualization_ditl_telemetry.png
   :alt: DITL telemetry example
   :align: center

Further references
------------------

See the API docs for the visualization functions under :mod:`conops.visualization`.
