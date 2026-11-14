3-D Spacecraft Configuration Visualizer
========================================

``plot_spacecraft_3d`` renders an interactive, rotatable 3-D model of a
spacecraft directly from its :class:`~conops.config.MissionConfig`.  The
figure is produced with Plotly and can be rotated, zoomed, and panned in
any browser or Jupyter notebook without running a full DITL simulation.

Features
--------

The figure draws the spacecraft body, the components attached to it, and
arrows that show the body-frame axes and the direction each component faces.

Spacecraft body
^^^^^^^^^^^^^^^

How the body is drawn depends on whether ``spacecraft_bus.surfaces`` has
plates.

When it does, the visualizer renders the plates at their configured centers,
normals and areas in the declared body frame. This configured model replaces
the synthetic bus and telescope described below. A plate's optional
``geometry`` supplies its rectangular outline. A plate without ``geometry`` is
illustrated as a square of equal area, because area and normal alone cannot
recover a facet's aspect ratio or the outline of a scarfed baffle. Front and back
plates that coincide with a panel are represented once, by the panel geometry.
The plot bounds include all components.

When no surfaces are configured, the visualizer falls back to a synthetic
model:

* The spacecraft bus is a lit, solid-shaded box with configurable dimensions.
* The telescope assembly is a cylindrical baffle tube with an aperture ring,
  mid-tube baffle rings, and a secondary mirror housing.  It is sized
  automatically from ``optics.aperture_m`` and ``optics.tube_length_m``.

Components
^^^^^^^^^^

* Solar panels are drawn as deep-blue solar cell faces on a silver frame.  A
  panel uses its :class:`~conops.config.geometry.PanelGeometry` when one is
  configured.  Otherwise the panel is placed on the bus face whose outward
  normal best matches the panel normal vector, and its area is estimated from
  ``max_power``.
* Radiators are drawn as white/MLI panels.  Each is placed on the bus face
  matching ``orientation.normal`` and sized by ``width_m`` × ``height_m``.
* Star trackers are drawn as small prism bodies with a coloured aperture face,
  one per configured tracker.  Each tracker gets a distinct boresight-arrow
  colour.

Axes and arrows
^^^^^^^^^^^^^^^

* The body-frame axes are drawn as a +X / +Y / +Z triad with labelled arrows
  (red, green, blue).
* Solar panels and radiators have outward-normal arrows.  The telescope and
  each star tracker have boresight arrows.

Coordinate system
-----------------

All geometry is expressed directly in ``spacecraft_bus.frame``, the body frame
declared in the configuration.  The visualizer does not rotate vectors a
second time.  The boresight label follows the declared frame.  The following
axes describe the default frame:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Axis
     - Meaning
   * - +X
     - Boresight / telescope pointing direction
   * - +Y
     - Spacecraft "up"
   * - +Z
     - Completes the right-handed system

Quick start
-----------

.. code-block:: python

   from conops.config import MissionConfig
   from conops.visualization import plot_spacecraft_3d

   config = MissionConfig.from_json_file("examples/example_config.json")
   fig = plot_spacecraft_3d(config)
   fig.show()

No simulation run is needed, because the visualizer reads geometry directly
from the configuration.

Usage examples
--------------

Minimal default spacecraft
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.config import MissionConfig
   from conops.visualization import plot_spacecraft_3d

   # Default config gives a bus, one solar panel, one radiator, and one star tracker
   fig = plot_spacecraft_3d(MissionConfig())
   fig.show()

Custom Ritchey-Chrétien telescope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.config import MissionConfig
   from conops.config.instrument import Telescope, TelescopeConfig, TelescopeType, Payload
   from conops.config.solar_panel import SolarPanel, SolarPanelSet, create_solar_panel_vector
   from conops.visualization import plot_spacecraft_3d

   config = MissionConfig(
       payload=Payload(instruments=[
           Telescope(
               name="Primary",
               boresight=(1.0, 0.0, 0.0),
               optics=TelescopeConfig(
                   aperture_m=0.5,
                   focal_length_m=3.5,
                   tube_length_m=1.6,
                   telescope_type=TelescopeType.RITCHEY_CHRETIEN,
               ),
           )
       ]),
       solar_panel=SolarPanelSet(panels=[
           SolarPanel(name="Panel +Y", normal=create_solar_panel_vector("sidemount"),         max_power=1200.0),
           SolarPanel(name="Panel -Y", normal=create_solar_panel_vector("sidemount", cant_z=180.0), max_power=1200.0),
       ]),
   )

   fig = plot_spacecraft_3d(config, title="RC Telescope — Dual Wing Config")
   fig.show()

Overriding the bus box size
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # bus_half_dims = (half_X, half_Y, half_Z) in metres
   fig = plot_spacecraft_3d(config, bus_half_dims=(1.2, 0.65, 0.65))
   fig.show()

Hiding normal vectors or axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Clean view without arrows
   fig = plot_spacecraft_3d(config, show_normals=False, show_axes=False)
   fig.show()

Embedding in a Jupyter notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # The figure is a standard go.Figure — display inline with:
   fig = plot_spacecraft_3d(config)
   fig  # Jupyter evaluates this cell and renders the Plotly widget

Saving to HTML
^^^^^^^^^^^^^^

.. code-block:: python

   fig = plot_spacecraft_3d(config)
   fig.write_html("spacecraft_model.html")

Visual elements reference
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Element
     - Colour
     - Source in config
   * - Spacecraft bus
     - Silver-gray
     - ``bus_half_dims`` parameter (default 1.0 × 0.55 × 0.55 m)
   * - Telescope tube
     - Dark gunmetal
     - ``payload.instruments`` (``instrument_type = "Telescope"``)
   * - Aperture / baffle rings
     - Gold
     - ``optics.aperture_m``, ``optics.tube_length_m``
   * - Solar cell faces
     - Deep blue
     - ``solar_panel.panels[*].normal``, ``PanelGeometry``
   * - Solar panel frames
     - Silver
     - Derived from panel geometry
   * - Radiators
     - White / MLI
     - ``spacecraft_bus.radiators.radiators[*]``
   * - Star tracker body
     - Dark gray
     - ``spacecraft_bus.star_trackers.star_trackers[*]``
   * - Star tracker aperture
     - Per-tracker colour
     - One of orange / hotpink / cyan / limegreen / gold / violet
   * - +X axis
     - Red
     - Hard-coded body frame
   * - +Y axis
     - Limegreen
     - Hard-coded body frame
   * - +Z axis
     - Dodgerblue
     - Hard-coded body frame
   * - Panel / radiator normals
     - Yellow
     - ``normal`` / ``orientation.normal``
   * - Telescope boresight
     - Cyan
     - ``boresight``
   * - Star tracker boresights
     - Per-tracker colour
     - ``orientation.boresight``

Panel placement rules
---------------------

When a :class:`~conops.config.geometry.PanelGeometry` is attached to a solar
panel or radiator, its ``center_m``, ``u``, ``v``, ``width_m``, and
``height_m`` fields are used directly.

When no ``PanelGeometry`` is set, the visualizer falls back to automatic face
placement:

1. It takes the dot product of the component normal with each of the six bus
   face normals.
2. It selects the face with the largest dot product.
3. It places the component centre just outside that face and orients the
   panel spanning directions perpendicular to the face normal.  The gap
   between the face and the component is component-specific; for example, it
   is 20 mm for solar panels and 5 mm for radiators.

Solar panel area is estimated as ``max_power / (150 × conversion_efficiency)``
in square metres, where ``conversion_efficiency`` is the value set on the
``solar_panel`` set (0.95 by default).  The panel is drawn with a
width-to-height ratio of 1.6 and is at least 0.8 m wide and 0.5 m high.
Radiator size comes directly from ``width_m`` and ``height_m``.

API reference
-------------

.. autofunction:: conops.visualization.plot_spacecraft_3d

See also
--------

* :doc:`configuration` describes all geometric configuration fields in detail.
* :doc:`visualization` gives an overview of all COASTSim visualization
  utilities.
* :doc:`sky_pointing_visualization` covers the interactive sky-pointing globe.
* :py:func:`conops.visualization.plot_sky_pointing_globe` draws a
  celestial-sphere globe driven by a completed DITL simulation.
