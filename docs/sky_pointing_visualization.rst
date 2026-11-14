Sky Pointing Visualization
==========================

``plot_sky_pointing()`` is an interactive tool that shows where the spacecraft is
pointing on a mollweide projection of the whole sky. It needs a DITL simulation
that has already run.

Features
--------

The plot is a full-sky Mollweide projection with the following drawn on it:

* All planned targets, marked on the sky and color-coded by observation category.
* The constraint regions, as shaded areas for the Sun, Moon, Earth, Anti-Sun, and
  Solar Panel constraints.
* A highlighted marker at the direction in which the spacecraft is currently pointing.

Time controls let you move through the simulation. A slider jumps to any time,
Previous and Next buttons step through time, and a Play button animates through
the entire DITL.

Individual frames can be saved, for creating animations or presentations.

Quick Start
-----------

.. code-block:: python

   from datetime import datetime, timedelta

   from conops import MissionConfig, QueueDITL
   from conops.visualization import plot_sky_pointing
   import matplotlib.pyplot as plt
   from rust_ephem import TLEEphemeris

   # Load configuration and compute the orbit ephemeris
   config = MissionConfig.from_json_file('config.json')
   begin = datetime(2025, 11, 1)
   end = begin + timedelta(days=1)
   ephemeris = TLEEphemeris(tle='spacecraft.tle', begin=begin, end=end)

   # Create DITL and add a target to its queue
   ditl = QueueDITL(config=config, ephem=ephemeris)
   ditl.queue.add(ra=266, dec=-29, exptime=1000, obsid=10000)

   # Run simulation
   ditl.calc()

   # Create interactive visualization
   fig, ax, controller = plot_sky_pointing(
       ditl,
       figsize=(14, 8),
       show_controls=True,
   )

   plt.show()

Usage
-----

Basic Visualization
^^^^^^^^^^^^^^^^^^^

With ``show_controls=False`` the function draws a static plot of the first time step.

.. code-block:: python

   # Simple static plot at the first time step
   fig, ax, _ = plot_sky_pointing(
       ditl,
       show_controls=False,  # No interactive controls
   )

Interactive Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^

With ``show_controls=True`` the plot has the time controls described under
`Interactive Controls`_. In a Jupyter notebook, select the ``widget`` Matplotlib
backend first.

.. code-block:: python

   # Full interactive plot with controls
   # For Jupyter notebooks:
   %matplotlib widget

   fig, ax, controller = plot_sky_pointing(
       ditl,
       figsize=(14, 8),
       n_grid_points=100,  # Higher = more accurate constraints, slower
       show_controls=True,
       constraint_alpha=0.3,  # Transparency of constraint regions
   )

Customization
^^^^^^^^^^^^^

The figure size, the resolution of the constraint grid, the time step of the
controls, and the transparency of the constraint regions can all be set.

.. code-block:: python

   fig, ax, controller = plot_sky_pointing(
       ditl,
       figsize=(16, 10),           # Larger figure
       n_grid_points=50,            # Faster rendering
       time_step_seconds=120,       # Step by 2 minutes
       constraint_alpha=0.2,        # More transparent constraints
   )

Saving Frames
^^^^^^^^^^^^^

``save_sky_pointing_frames()`` writes frames to a directory and returns the list
of files it saved. The frames can then be assembled into an animation with a
library such as ``imageio``.

.. code-block:: python

   from conops.visualization import save_sky_pointing_frames

   # Save every 10th frame
   saved_files = save_sky_pointing_frames(
       ditl,
       output_dir='./sky_frames',
       n_grid_points=50,
       frame_interval=10,
   )

   # Create animation with imageio
   import imageio
   frames = [imageio.imread(f) for f in saved_files]
   imageio.mimsave('sky_pointing.gif', frames, fps=10)

Visualization Elements
----------------------

Observation Markers
^^^^^^^^^^^^^^^^^^^

Scheduled observations are shown as colored circles. The color and size of a
circle both depend on the target's observation ID (obsid), in different ways.

The color is the color of the target's observation category. The categories
come from the ``observation_categories`` argument, or from
``ditl.config.observation_categories`` when that argument is omitted. A target
whose obsid falls in no category takes the default color of the categories.
The target being observed at the displayed time is drawn in the full category
color, and all other targets are drawn in a lightened shade of it. The legend
lists the categories that appear in the plot. The frames and movies written by
``save_sky_pointing_frames()`` and ``save_sky_pointing_movie()`` do not use the
categories, and draw every target in blue.

The size is set by obsid thresholds:

.. list-table::
   :header-rows: 1

   * - Marker area (points squared)
     - Targets
   * - 100
     - obsid >= 1,000,000
   * - 60
     - obsid >= 20,000
   * - 40
     - All other targets

Current Pointing
^^^^^^^^^^^^^^^^

A star marks the current spacecraft pointing direction, with a circle of
5 degrees radius around it. Both take the color of the current ACS mode, from
``mode_colors`` in the visualization configuration. They are red for a mode
that has no entry there.

Constraint Regions
^^^^^^^^^^^^^^^^^^

Shaded areas show the regions where the spacecraft is forbidden to point. Each
constraint has its own color:

.. list-table::
   :header-rows: 1

   * - Color
     - Constraint
     - Reason
   * - Yellow
     - Sun constraint zone
     - Too close to the Sun
   * - Gray
     - Moon constraint zone
     - Too close to the Moon
   * - Blue
     - Earth limb constraint
     - Pointing too close to the Earth
   * - Orange
     - Anti-Sun constraint
     - Solar panel limitations
   * - Green
     - Solar panel constraint
     - Power generation limitations

Celestial Bodies
^^^^^^^^^^^^^^^^

Large circular markers show the current positions of the Sun (yellow with a
black edge), the Moon (gray with a black edge) and the Earth (blue with a black
edge).

Interactive Controls
--------------------

When ``show_controls=True``, the figure has four controls:

* The time slider can be dragged to jump to any point in the simulation.
* The "< Prev" button steps backward one time step.
* The "Next >" button steps forward one time step.
* The "Play" button advances through time automatically. Its label becomes "Pause"
  while the animation is playing.

Performance Tips
----------------

The constraint calculation is the slowest part of drawing a frame, and its cost
is set by ``n_grid_points``.

1. Lower ``n_grid_points`` (e.g., 30-50) for faster rendering during exploration.
2. Increase ``n_grid_points`` (e.g., 100-200) for publication-quality figures.
3. Use ``frame_interval > 1`` when saving many frames, to reduce the file count.
4. Consider caching results for animations, so that the constraint calculation
   is not repeated.

Example Workflows
-----------------

Interactive Exploration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use in Jupyter notebook for interactive exploration
   %matplotlib widget

   fig, ax, ctrl = plot_sky_pointing(ditl, n_grid_points=50)

   # Play through the simulation
   # Use the Play button or call programmatically:
   # ctrl.start_animation()

Publication Figures
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.visualization.mpl.sky_pointing import SkyPointingController

   # High-quality static image at specific time
   fig, ax, _ = plot_sky_pointing(
       ditl,
       show_controls=False,
       n_grid_points=150,
       figsize=(12, 8),
   )

   # plot_sky_pointing() returns no controller when show_controls=False,
   # so create one for the same figure
   ctrl = SkyPointingController(
       ditl=ditl,
       fig=fig,
       ax=ax,
       n_grid_points=150,
       observation_categories=ditl.config.observation_categories,
   )

   # Update to specific time
   specific_time = ditl.utime[500]  # e.g., 500th timestep
   ctrl.update_plot(specific_time)

   # Save figure
   fig.savefig('spacecraft_pointing.pdf', dpi=300, bbox_inches='tight')

Movie Export
^^^^^^^^^^^^

``save_sky_pointing_movie()`` exports the sky pointing visualization as an animated
movie, which shows how the spacecraft pointing and the constraints evolve throughout
the DITL simulation. The function supports multiple output formats:

.. code-block:: python

   from conops.visualization import save_sky_pointing_movie

   # Export as MP4 video (requires ffmpeg)
   save_sky_pointing_movie(
       ditl,
       "pointing.mp4",
       fps=15,  # frames per second
       frame_interval=5,  # use every 5th time step
       n_grid_points=30,  # constraint grid resolution
       dpi=100  # output resolution
   )

   # Export as animated GIF (requires pillow)
   save_sky_pointing_movie(
       ditl,
       "pointing.gif",
       fps=5,
       frame_interval=10
   )

The function takes these parameters:

* ``ditl`` is the DITL or QueueDITL object, which must have run ``calc()`` first.
* ``output_file`` is the name of the movie file. Its extension (``.mp4``,
  ``.avi`` or ``.gif``) selects the format.
* ``fps`` is the number of frames per second in the output movie (default: 10).
  It controls the playback speed.
* ``figsize`` is the figure size (width, height) in inches (default: (14, 8)).
* ``n_grid_points`` is the grid resolution for the constraint regions
  (default: 50). A lower value renders faster.
* ``frame_interval`` uses every Nth time step (default: 1). A value of 1 uses
  all frames.
* ``dpi`` is the output resolution (default: 100). A higher value gives better
  quality and a larger file.
* ``codec`` is the video codec for MP4/AVI (default: 'h264'; 'mpeg4' is another
  example).
* ``bitrate`` is the video bitrate in kbps (default: 1800). A higher value gives
  better quality and a larger file.
* ``config`` is an optional ``VisualizationConfig``. When it is omitted, the
  function uses ``ditl.config.visualization``.
* ``show_progress`` chooses whether to display a progress bar using tqdm
  (default: True).

The function returns the path of the saved movie file.

The output formats have these requirements:

* MP4 and AVI formats require ffmpeg to be installed on your system.
* GIF format requires the pillow library, which is usually bundled with matplotlib.
* The progress bar requires the tqdm library. This is optional, and the function
  falls back gracefully if tqdm is not available.

API Reference
-------------

plot_sky_pointing()
^^^^^^^^^^^^^^^^^^^

This is the main visualization function. It takes these parameters:

* ``ditl``: DITL or QueueDITL object (must have run ``calc()`` first)
* ``figsize``: tuple, figure size (width, height) in inches (default: (14, 8))
* ``n_grid_points``: int, resolution for constraint calculations (default: 100)
* ``show_controls``: bool, add interactive controls (default: True)
* ``time_step_seconds``: float, time step for controls (default: ditl.step_size)
* ``constraint_alpha``: float, transparency for constraint regions (default: 0.3)
* ``config``: VisualizationConfig, fonts and colors (default:
  ``ditl.config.visualization``)
* ``observation_categories``: ObservationCategories, marker colors (default:
  ``ditl.config.observation_categories``)

It returns three values:

* ``fig``: matplotlib Figure
* ``ax``: matplotlib Axes (mollweide projection)
* ``controller``: SkyPointingController, or None when ``show_controls=False``

save_sky_pointing_frames()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function saves individual frames to disk. It takes these parameters:

* ``ditl``: DITL or QueueDITL object
* ``output_dir``: str, directory to save frames
* ``figsize``: tuple, figure size (default: (14, 8))
* ``n_grid_points``: int, constraint resolution (default: 50)
* ``frame_interval``: int, save every Nth frame (default: 1)
* ``config``: VisualizationConfig, fonts and colors (default: None, which
  draws the frames with the built-in default fonts and mode colors)

It returns the list of saved file paths.

Implementation Notes
--------------------

* The visualization requires a completed DITL simulation with ephemeris data.
* The Mollweide projection shows the entire sky with minimal distortion.
* RA runs from 0° to 360° (or -180° to 180° in the projection).
* Dec runs from -90° (south) to +90° (north).
* Constraint regions are computed at each timestep, based on the orbital position.
* For very long simulations, consider using ``frame_interval > 1`` to reduce
  computation time.

Troubleshooting
---------------

ValueError: "DITL simulation has no pointings"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simulation has not been run. Run ``ditl.calc()`` before calling
``plot_sky_pointing()``.

ValueError: "DITL constraint has no ephemeris set"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure your MissionConfig has ephemeris data loaded.

Slow rendering
^^^^^^^^^^^^^^

Some slowness is expected, because the constraint calculation checks many sky
positions. Reduce ``n_grid_points`` to speed it up; try 30-50 for exploration.

Interactive controls not working in Jupyter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``%matplotlib widget`` or ``%matplotlib notebook``. You may need to install
``ipympl`` with ``pip install ipympl``.

See Also
--------

* ``plot_ditl_timeline()`` gives a timeline view of spacecraft operations.
* ``plot_ditl_telemetry()`` gives telemetry plots (power, battery, etc.).
* The example notebook ``examples/Quick_DITL_Example.ipynb`` includes a sky pointing plot.

API Documentation
-----------------

For detailed API documentation, see:

* :py:func:`conops.visualization.plot_sky_pointing`
* :py:func:`conops.visualization.save_sky_pointing_frames`
* :py:func:`conops.visualization.save_sky_pointing_movie`
* :py:class:`conops.visualization.mpl.sky_pointing.SkyPointingController`
