"""Sky Pointing Visualization

Interactive visualization showing spacecraft pointing on a mollweide projection
of the sky with scheduled observations and constraint regions.
"""

import os
from datetime import timezone

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import Button, Slider
from rust_ephem import Constraint

from ..common import dtutcfromtimestamp
from ..config.visualization import VisualizationConfig


def _get_visualization_config(ditl, config=None):
    """Get visualization configuration, with fallback to defaults.

    Parameters
    ----------
    ditl : DITL or QueueDITL
        The DITL simulation object.
    config : VisualizationConfig, optional
        Explicit config to use. If None, tries to get from ditl.config.visualization.

    Returns
    -------
    VisualizationConfig
        The configuration object to use.
    """
    if config is None:
        if (
            hasattr(ditl, "config")
            and hasattr(ditl.config, "visualization")
            and isinstance(ditl.config.visualization, VisualizationConfig)
        ):
            config = ditl.config.visualization
        else:
            config = VisualizationConfig()
    return config


def plot_sky_pointing(
    ditl,
    figsize=(14, 8),
    n_grid_points=100,
    show_controls=True,
    time_step_seconds=None,
    constraint_alpha=1.0,
    config=None,
):
    """Plot spacecraft pointing on a mollweide sky map with constraints.

    Creates an interactive visualization showing:
    - All scheduled observations as markers
    - Current spacecraft pointing direction
    - Sun, Moon, and Earth constraint regions (shaded)
    - Time controls to step through or play the DITL

    Parameters
    ----------
    ditl : DITL or QueueDITL
        The DITL simulation object with completed simulation data.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (14, 8)).
    n_grid_points : int, optional
        Number of grid points per axis for constraint region calculation
        (default: 100). Higher values give smoother regions but slower rendering.
    show_controls : bool, optional
        Whether to show interactive time controls (default: True).
    time_step_seconds : float, optional
        Time step in seconds for controls. If None, uses ditl.step_size (default: None).
    constraint_alpha : float, optional
        Alpha transparency for constraint regions (default: 0.3).
    config : Config, optional
        Configuration object containing visualization settings. If None, uses ditl.config.visualization if available.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the sky map.
    controller : SkyPointingController or None
        The controller object if show_controls=True, otherwise None.

    Examples
    --------
    >>> from conops import DITL
    >>> ditl = DITL(config)
    >>> ditl.calc()
    >>> fig, ax, ctrl = plot_sky_pointing(ditl)
    >>> plt.show()

    Notes
    -----
    - RA coordinates are in degrees (0-360)
    - Dec coordinates are in degrees (-90 to 90)
    - Constraint regions are computed at each time step based on Sun, Moon,
      and Earth positions from the ephemeris
    """
    # Validate inputs
    if not hasattr(ditl, "plan") or len(ditl.plan) == 0:
        raise ValueError("DITL simulation has no pointings. Run calc() first.")
    if not hasattr(ditl, "utime") or len(ditl.utime) == 0:
        raise ValueError("DITL has no time data. Run calc() first.")
    if ditl.constraint.ephem is None:
        raise ValueError("DITL constraint has no ephemeris set.")

    # Set default time step
    if time_step_seconds is None:
        time_step_seconds = ditl.step_size

    # Get visualization config
    config = _get_visualization_config(ditl, config)

    # Create the visualization
    if show_controls:
        fig = plt.figure(figsize=figsize)
        # Main plot takes most of the space
        ax = fig.add_subplot(111, projection="mollweide")
        # Leave space at bottom for controls
        fig.subplots_adjust(bottom=0.25)
    else:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "mollweide"})

    # Create the controller
    controller = SkyPointingController(
        ditl=ditl,
        fig=fig,
        ax=ax,
        n_grid_points=n_grid_points,
        time_step_seconds=time_step_seconds,
        constraint_alpha=constraint_alpha,
        config=config,
    )

    # Initial plot
    controller.update_plot(ditl.utime[0])

    # Add interactive controls if requested
    if show_controls:
        controller.add_controls()

    return fig, ax, controller if show_controls else None


class SkyPointingController:
    """Controller for interactive sky pointing visualization."""

    def __init__(
        self,
        ditl,
        fig,
        ax,
        n_grid_points=100,
        time_step_seconds=60,
        constraint_alpha=1.0,
        config=None,
    ):
        """Initialize the controller.

        Parameters
        ----------
        ditl : DITL or QueueDITL
            The DITL simulation object.
        fig : matplotlib.figure.Figure
            The figure to draw on.
        ax : matplotlib.axes.Axes
            The axes with mollweide projection.
        n_grid_points : int
            Number of grid points for constraint calculation.
        time_step_seconds : float
            Time step for controls in seconds.
        constraint_alpha : float
            Alpha transparency for constraint regions.
        config : VisualizationConfig, optional
            Visualization configuration settings.
        """
        self.ditl = ditl
        self.fig = fig
        self.ax = ax
        self.n_grid_points = n_grid_points
        self.time_step_seconds = time_step_seconds
        self.constraint_alpha = constraint_alpha
        self.config = config

        # State
        self.current_time_idx = 0
        self.playing = False
        self.timer = None

        # Plot elements (will be created in update_plot)
        self.constraint_patches = {}
        self.current_pointing_marker = None
        self.scheduled_obs_scatter = None
        self.title_text = None

        # Control widgets
        self.slider = None
        self.play_button = None
        self.prev_button = None
        self.next_button = None

    def add_controls(self):
        """Add interactive control widgets to the figure."""
        # Create axes for controls
        ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
        ax_play = plt.axes([0.42, 0.05, 0.08, 0.04])
        ax_prev = plt.axes([0.32, 0.05, 0.08, 0.04])
        ax_next = plt.axes([0.52, 0.05, 0.08, 0.04])

        # Time slider
        self.slider = Slider(
            ax_slider,
            "Time",
            0,
            len(self.ditl.utime) - 1,
            valinit=0,
            valstep=1,
            valfmt="%d",
        )
        self.slider.on_changed(self.on_slider_change)

        # Buttons
        self.play_button = Button(ax_play, "Play")
        self.play_button.on_clicked(self.on_play_clicked)

        self.prev_button = Button(ax_prev, "< Prev")
        self.prev_button.on_clicked(self.on_prev_clicked)

        self.next_button = Button(ax_next, "Next >")
        self.next_button.on_clicked(self.on_next_clicked)

    def on_slider_change(self, val):
        """Handle slider value change."""
        idx = int(val)
        if idx != self.current_time_idx:
            self.current_time_idx = idx
            self.update_plot(self.ditl.utime[idx])

    def on_play_clicked(self, event):
        """Handle play button click."""
        if self.playing:
            self.stop_animation()
        else:
            self.start_animation()

    def on_prev_clicked(self, event):
        """Handle previous button click."""
        if self.current_time_idx > 0:
            self.current_time_idx -= 1
            self.slider.set_val(self.current_time_idx)
            self.update_plot(self.ditl.utime[self.current_time_idx])

    def on_next_clicked(self, event):
        """Handle next button click."""
        if self.current_time_idx < len(self.ditl.utime) - 1:
            self.current_time_idx += 1
            self.slider.set_val(self.current_time_idx)
            self.update_plot(self.ditl.utime[self.current_time_idx])

    def start_animation(self):
        """Start playing through time steps."""
        self.playing = True
        self.play_button.label.set_text("Pause")
        # Use matplotlib's animation timer
        self.timer = self.fig.canvas.new_timer(interval=100)  # 100ms between frames
        self.timer.add_callback(self.animation_step)
        self.timer.start()

    def stop_animation(self):
        """Stop playing animation."""
        self.playing = False
        if self.play_button is not None:
            self.play_button.label.set_text("Play")
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

    def animation_step(self):
        """Advance to next time step during animation."""
        if self.current_time_idx < len(self.ditl.utime) - 1:
            self.current_time_idx += 1
            self.slider.set_val(self.current_time_idx)
            self.update_plot(self.ditl.utime[self.current_time_idx])
        else:
            # Reached the end
            self.stop_animation()

    def update_plot(self, utime):
        """Update the plot for a given time.

        Parameters
        ----------
        utime : float
            Unix timestamp to display.
        """
        self.ax.clear()

        # Get current spacecraft pointing
        idx = self._find_time_index(utime)
        current_ra = self.ditl.ra[idx]
        current_dec = self.ditl.dec[idx]
        current_mode = self.ditl.mode[idx]

        # Plot scheduled observations
        self._plot_scheduled_observations()

        # Plot constraint regions
        self._plot_constraint_regions(utime)

        # Plot Earth physical disk
        self._plot_earth_disk(utime)

        # Plot current pointing
        self._plot_current_pointing(current_ra, current_dec, current_mode)

        # Set up the plot
        self._setup_plot_appearance(utime)

        # Redraw
        self.fig.canvas.draw_idle()

    def _find_time_index(self, utime):
        """Find the index in utime array closest to the given time."""
        idx = np.argmin(np.abs(np.array(self.ditl.utime) - utime))
        # Ensure index is within bounds for all arrays
        max_idx = (
            min(
                len(self.ditl.utime),
                len(self.ditl.ra),
                len(self.ditl.dec),
                len(self.ditl.mode),
            )
            - 1
        )
        return min(idx, max_idx)

    def _plot_scheduled_observations(self):
        """Plot all scheduled observations as markers."""
        if len(self.ditl.plan) == 0:
            return

        # Cache the observation data since it doesn't change between frames
        if not hasattr(self, "_cached_observations"):
            ras = []
            decs = []
            colors = []
            sizes = []

            for ppt in self.ditl.plan:
                ra = ppt.ra
                dec = ppt.dec

                # Convert RA from 0-360 to -180 to 180 for mollweide, with RA=0 on left
                ra_plot = self._convert_ra_for_plotting(np.array([ra]))[0]

                ras.append(np.deg2rad(ra_plot))
                decs.append(np.deg2rad(dec))

                # Color by observation type or ID
                if hasattr(ppt, "obsid"):
                    # Use obsid to determine color
                    if ppt.obsid >= 1000000:
                        colors.append("red")  # TOO/GRB
                        sizes.append(100)
                    elif ppt.obsid >= 20000:
                        colors.append("orange")  # High priority
                        sizes.append(60)
                    elif ppt.obsid >= 10000:
                        colors.append("yellow")  # Survey
                        sizes.append(40)
                    else:
                        colors.append("lightblue")  # Standard
                        sizes.append(40)
                else:
                    colors.append("lightblue")
                    sizes.append(40)

            self._cached_observations = {
                "ras": ras,
                "decs": decs,
                "colors": colors,
                "sizes": sizes,
            }

        # Scatter plot using cached data
        self.ax.scatter(
            self._cached_observations["ras"],
            self._cached_observations["decs"],
            s=self._cached_observations["sizes"],
            c=self._cached_observations["colors"],
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
            label="Targets",
            zorder=2,
            rasterized=True,  # Rasterize for faster rendering
        )

    def _precompute_constraints(self, time_indices=None):
        """Pre-compute constraint masks for all time steps using in_constraint_batch.

        This evaluates all constraints for the entire DITL in a single batch operation
        for much better performance during movie rendering.

        Parameters
        ----------
        time_indices : array-like, optional
            Indices of time steps to pre-compute. If None, uses all time steps.
        """
        if time_indices is None:
            time_indices = np.arange(len(self.ditl.utime))

        # Get sky grid points (same for all times)
        ra_flat, dec_flat = self._create_sky_grid(self.n_grid_points)
        n_points = len(ra_flat)

        # Convert times to datetime objects for rust_ephem
        times = [dtutcfromtimestamp(self.ditl.utime[idx]) for idx in time_indices]
        n_times = len(times)

        print(
            f"Pre-computing constraints for {n_times} time steps with {n_points} grid points..."
        )

        # Pre-compute all constraint types
        constraint_cache = {}

        constraint_types = [
            ("sun", self.ditl.config.constraint.sun_constraint),
            ("moon", self.ditl.config.constraint.moon_constraint),
            ("earth", self.ditl.config.constraint.earth_constraint),
            ("anti_sun", self.ditl.config.constraint.anti_sun_constraint),
            ("panel", self.ditl.config.constraint.panel_constraint),
        ]

        for name, constraint_func in constraint_types:
            # Batch evaluation with datetime array
            result = constraint_func.in_constraint_batch(
                ephemeris=self.ditl.ephem,
                target_ras=ra_flat,
                target_decs=dec_flat,
                times=times,  # Pass entire array of datetime objects
            )
            # Result shape is (n_points, n_times) from rust_ephem
            constraint_cache[name] = result

        # Store cache
        self._constraint_cache = {
            "ra_grid": ra_flat,
            "dec_grid": dec_flat,
            "time_indices": time_indices,
            "constraints": constraint_cache,
        }

        print(
            f"Constraint pre-computation complete. Cached {len(constraint_types)} constraint types."
        )

    def _plot_constraint_regions(self, utime):
        """Plot constraint regions for Sun, Moon, and Earth.

        Parameters
        ----------
        utime : float
            Unix timestamp for constraint calculation.
        """
        dt = dtutcfromtimestamp(utime)
        ephem = self.ditl.constraint.ephem
        idx = ephem.index(dt)

        # Get celestial body positions
        sun_ra = ephem.sun[idx].ra.deg
        sun_dec = ephem.sun[idx].dec.deg
        moon_ra = ephem.moon[idx].ra.deg
        moon_dec = ephem.moon[idx].dec.deg
        earth_ra = ephem.earth[idx].ra.deg
        earth_dec = ephem.earth[idx].dec.deg

        # Check each constraint type and plot regions
        constraint_types = [
            (
                "Sun",
                self.ditl.config.constraint.sun_constraint,
                "yellow",
                sun_ra,
                sun_dec,
            ),
            (
                "Moon",
                self.ditl.config.constraint.moon_constraint,
                "gray",
                moon_ra,
                moon_dec,
            ),
            (
                "Earth",
                self.ditl.config.constraint.earth_constraint,
                "blue",
                earth_ra,
                earth_dec,
            ),
            (
                "Anti-Sun",
                self.ditl.config.constraint.anti_sun_constraint,
                "orange",
                (sun_ra + 180) % 360,
                -sun_dec,
            ),
            (
                "Panel",
                self.ditl.config.constraint.panel_constraint,
                "green",
                None,
                None,
            ),
        ]

        for name, constraint_func, color, body_ra, body_dec in constraint_types:
            self._plot_single_constraint(
                name,
                constraint_func,
                color,
                utime,
                body_ra,
                body_dec,
            )

    def _plot_earth_disk(self, utime):
        """Plot the physical extent of Earth as seen from the spacecraft.

        Shows Earth with realistic surface coloring, coastlines, and day/night terminator.

        Parameters
        ----------
        utime : float
            Unix timestamp for Earth position calculation.
        """
        dt = dtutcfromtimestamp(utime)
        ephem = self.ditl.constraint.ephem
        idx = ephem.index(dt)

        # Get Earth position and angular radius
        earth_ra = ephem.earth[idx].ra.deg
        earth_dec = ephem.earth[idx].dec.deg
        earth_angular_radius = ephem.earth_radius_deg[idx]

        # Get Sun position for illumination calculation
        sun_ra = ephem.sun[idx].ra.deg
        sun_dec = ephem.sun[idx].dec.deg

        # Get sky grid points (use cached if available)
        if hasattr(self, "_constraint_cache"):
            ra_flat = self._constraint_cache["ra_grid"]
            dec_flat = self._constraint_cache["dec_grid"]
        else:
            ra_flat, dec_flat = self._create_sky_grid(self.n_grid_points)

        # Vectorized calculation of angular distances from Earth center
        delta_ra = np.radians(ra_flat - earth_ra)
        dec_rad = np.radians(dec_flat)
        earth_dec_rad = np.radians(earth_dec)

        cos_value = np.sin(earth_dec_rad) * np.sin(dec_rad) + np.cos(
            earth_dec_rad
        ) * np.cos(dec_rad) * np.cos(delta_ra)
        angular_dist = np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0)))

        # Find points inside the Earth disk
        inside_earth = angular_dist <= earth_angular_radius

        if not inside_earth.any():
            return

        # Get points on Earth disk
        ra_vals = ra_flat[inside_earth]
        dec_vals = dec_flat[inside_earth]

        # Convert sky coordinates to Earth surface coordinates
        earth_lon, earth_lat = self._sky_to_earth_surface(
            ra_vals, dec_vals, earth_ra, earth_dec, dt
        )

        # Get colors for each point based on Earth surface and illumination
        colors = self._get_earth_surface_colors(
            earth_lon, earth_lat, sun_ra, sun_dec, earth_ra, earth_dec, dt
        )

        # Plot Earth disk with surface colors
        ra_plot = self._convert_ra_for_plotting(ra_vals)
        self.ax.scatter(
            np.deg2rad(ra_plot),
            np.deg2rad(dec_vals),
            s=20,
            c=colors,
            alpha=0.9,
            marker="s",
            edgecolors="none",
            label="Earth",
            zorder=2.5,
            rasterized=True,
        )

    def _sky_to_earth_surface(self, ra_vals, dec_vals, earth_ra, earth_dec, dt):
        """Convert sky coordinates to Earth surface lat/lon.

        Uses orthographic projection from the spacecraft's viewpoint to map
        the visible hemisphere of Earth onto surface coordinates.

        Parameters
        ----------
        ra_vals : array
            RA coordinates of points on Earth disk (degrees).
        dec_vals : array
            Dec coordinates of points on Earth disk (degrees).
        earth_ra : float
            RA of Earth center as seen from spacecraft (degrees).
        earth_dec : float
            Dec of Earth center as seen from spacecraft (degrees).
        dt : datetime
            Time for Earth rotation calculation.

        Returns
        -------
        lon : array
            Longitude on Earth surface (-180 to 180 degrees).
        lat : array
            Latitude on Earth surface (-90 to 90 degrees).
        """
        # The subsatellite point (point on Earth directly under spacecraft)
        # corresponds to the Earth center in the sky (earth_ra, earth_dec)

        # Calculate Greenwich Mean Sidereal Time for Earth rotation
        from datetime import datetime

        j2000 = datetime(2000, 1, 1, 12, 0, 0)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days_since_j2000 = (
            dt.replace(tzinfo=timezone.utc) - j2000.replace(tzinfo=timezone.utc)
        ).total_seconds() / 86400

        # GMST calculation (simplified)
        gmst = (280.46061837 + 360.98564736629 * days_since_j2000) % 360

        # Subsatellite point on Earth surface
        subsatellite_lat = (
            earth_dec  # Dec of Earth center = latitude of subsatellite point
        )
        subsatellite_lon = (earth_ra - gmst) % 360
        if subsatellite_lon > 180:
            subsatellite_lon -= 360

        # Convert to radians for calculation
        ra_rad = np.radians(ra_vals)
        dec_rad = np.radians(dec_vals)
        earth_ra_rad = np.radians(earth_ra)
        earth_dec_rad = np.radians(earth_dec)

        # Calculate angular separation from Earth center (subsatellite point)
        # This is the angular distance in the sky
        delta_ra = ra_rad - earth_ra_rad
        cos_angular_dist = np.sin(earth_dec_rad) * np.sin(dec_rad) + np.cos(
            earth_dec_rad
        ) * np.cos(dec_rad) * np.cos(delta_ra)
        angular_dist = np.arccos(np.clip(cos_angular_dist, -1, 1))

        # Calculate position angle (azimuth from subsatellite point)
        # This tells us the direction from Earth center to the point in the sky
        sin_pa = np.sin(delta_ra) * np.cos(dec_rad)
        cos_pa = np.cos(earth_dec_rad) * np.sin(dec_rad) - np.sin(
            earth_dec_rad
        ) * np.cos(dec_rad) * np.cos(delta_ra)
        position_angle = np.arctan2(sin_pa, cos_pa)

        # For orthographic projection from spacecraft:
        # angular_dist in sky = angular_dist on Earth sphere
        # The position angle in sky = azimuth on Earth surface

        # Convert from subsatellite point coordinates to lat/lon
        subsatellite_lat_rad = np.radians(subsatellite_lat)

        # Latitude on Earth using spherical trigonometry
        lat_rad = np.arcsin(
            np.sin(subsatellite_lat_rad) * np.cos(angular_dist)
            + np.cos(subsatellite_lat_rad)
            * np.sin(angular_dist)
            * np.cos(position_angle)
        )
        lat = np.degrees(lat_rad)

        # Longitude offset from subsatellite point
        dlon = np.arctan2(
            np.sin(position_angle) * np.sin(angular_dist),
            np.cos(angular_dist) - np.sin(subsatellite_lat_rad) * np.sin(lat_rad),
        )

        lon = subsatellite_lon + np.degrees(dlon)

        # Normalize to -180 to 180
        lon = lon % 360
        lon = np.where(lon > 180, lon - 360, lon)

        return lon, lat

    def _get_earth_surface_colors(
        self, lon, lat, sun_ra, sun_dec, earth_ra, earth_dec, dt
    ):
        """Get colors for Earth surface points based on geography and illumination.

        Parameters
        ----------
        lon : array
            Longitude on Earth surface (-180 to 180 degrees).
        lat : array
            Latitude on Earth surface (-90 to 90 degrees).
        sun_ra : float
            RA of Sun (degrees).
        sun_dec : float
            Dec of Sun (degrees).
        earth_ra : float
            RA of Earth center (degrees).
        earth_dec : float
            Dec of Earth center (degrees).
        dt : datetime
            Time for GMST calculation.

        Returns
        -------
        colors : array
            RGB colors for each point.
        """
        # Detailed ocean colors with depth variation
        ocean_shallow = np.array([0.15, 0.35, 0.65])  # Lighter coastal blue
        ocean_deep = np.array([0.05, 0.15, 0.45])  # Deep ocean blue

        # Diverse land colors by biome
        # Tropical rainforest
        land_rainforest = np.array([0.15, 0.5, 0.2])  # Dark green
        # Savanna/grassland
        land_savanna = np.array([0.55, 0.55, 0.25])  # Yellow-green
        # Desert
        land_desert = np.array([0.75, 0.65, 0.45])  # Sandy tan
        # Temperate forest
        land_temperate = np.array([0.3, 0.6, 0.3])  # Green
        # Boreal forest/tundra
        land_boreal = np.array([0.25, 0.45, 0.3])  # Dark green-gray
        # Mountain/highland
        land_mountain = np.array([0.5, 0.4, 0.3])  # Brown
        # Ice/snow
        land_ice = np.array([0.95, 0.95, 0.98])  # Bright white

        # Determine land mask
        is_land = self._approximate_land_mask(lon, lat)

        # Create latitude-based biome classification
        abs_lat = np.abs(lat)

        # Calculate biome weights for land areas
        # Polar regions (>60°): ice
        ice_weight = np.maximum(0, (abs_lat - 60) / 30)
        # Boreal (45-60°): boreal forest/tundra
        boreal_weight = np.maximum(0, 1 - np.abs(abs_lat - 52.5) / 15)  # Peak at 52.5°
        # Temperate (30-50°): temperate forest
        temperate_weight = np.maximum(0, 1 - np.abs(abs_lat - 40) / 15)  # Peak at 40°
        # Subtropical/desert (15-35°): mix of desert and grassland
        desert_weight = np.maximum(0, 1 - np.abs(abs_lat - 25) / 15)  # Peak at 25°
        # Tropical (0-20°): rainforest and savanna
        tropical_weight = np.maximum(0, 1 - abs_lat / 20)

        # Add longitude-based variation for deserts (e.g., Sahara, Arabian, Australian)
        # Enhance desert in specific regions
        desert_regions = (
            # Sahara
            ((lon > -10) & (lon < 40) & (abs_lat > 15) & (abs_lat < 30))
            |
            # Arabian
            ((lon > 35) & (lon < 60) & (abs_lat > 15) & (abs_lat < 35))
            |
            # Australian interior
            ((lon > 115) & (lon < 145) & (lat > -35) & (lat < -20))
            |
            # Kalahari
            ((lon > 15) & (lon < 30) & (lat > -30) & (lat < -18))
            |
            # Atacama
            ((lon > -75) & (lon < -68) & (lat > -30) & (lat < -18))
            |
            # Great Basin/Mojave
            ((lon > -120) & (lon < -110) & (lat > 35) & (lat < 42))
        )
        desert_weight = np.where(desert_regions, desert_weight * 2, desert_weight)

        # Normalize weights
        total_weight = (
            ice_weight
            + boreal_weight
            + temperate_weight
            + desert_weight
            + tropical_weight
            + 1e-10
        )
        ice_weight /= total_weight
        boreal_weight /= total_weight
        temperate_weight /= total_weight
        desert_weight /= total_weight
        tropical_weight /= total_weight

        # Split tropical between rainforest and savanna
        # Rainforest: equatorial regions
        rainforest_weight = tropical_weight * np.maximum(0, 1 - abs_lat / 10)
        savanna_weight = tropical_weight - rainforest_weight

        # Calculate blended land color
        land_color = (
            ice_weight[:, np.newaxis] * land_ice
            + boreal_weight[:, np.newaxis] * land_boreal
            + temperate_weight[:, np.newaxis] * land_temperate
            + desert_weight[:, np.newaxis] * land_desert
            + rainforest_weight[:, np.newaxis] * land_rainforest
            + savanna_weight[:, np.newaxis] * land_savanna
        )

        # Add mountain coloring for high elevation regions (rough approximation)
        # Major mountain ranges
        is_mountain = (
            # Himalayas
            ((lon > 70) & (lon < 95) & (lat > 25) & (lat < 40))
            |
            # Andes
            ((lon > -78) & (lon < -65) & (lat > -50) & (lat < 10))
            |
            # Rockies
            ((lon > -120) & (lon < -105) & (lat > 35) & (lat < 55))
            |
            # Alps
            ((lon > 5) & (lon < 16) & (lat > 44) & (lat < 48))
            |
            # Ethiopian Highlands
            ((lon > 35) & (lon < 45) & (lat > 5) & (lat < 15))
        )
        land_color = np.where(is_mountain[:, np.newaxis], land_mountain, land_color)

        # Vary ocean color with latitude (simulating depth)
        # Deeper oceans near equator and poles
        ocean_depth_factor = 0.5 + 0.5 * np.cos(np.radians(lat * 2))
        ocean_color = ocean_deep * ocean_depth_factor[:, np.newaxis] + ocean_shallow * (
            1 - ocean_depth_factor[:, np.newaxis]
        )

        # Mix land and ocean
        base_colors = np.where(is_land[:, np.newaxis], land_color, ocean_color)

        # Calculate solar illumination
        # The subsolar point is where the Sun vector from Earth center points
        # Convert Sun position to Earth surface coordinates
        # Sun's declination directly gives latitude
        subsolar_lat = sun_dec

        # Calculate Sun's longitude using the same GMST calculation
        from datetime import datetime

        j2000 = datetime(2000, 1, 1, 12, 0, 0)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days_since_j2000 = (
            dt.replace(tzinfo=timezone.utc) - j2000.replace(tzinfo=timezone.utc)
        ).total_seconds() / 86400
        gmst = (280.46061837 + 360.98564736629 * days_since_j2000) % 360
        subsolar_lon = (sun_ra - gmst) % 360
        if subsolar_lon > 180:
            subsolar_lon -= 360

        # Calculate solar zenith angle for each point
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        subsolar_lat_rad = np.radians(subsolar_lat)
        subsolar_lon_rad = np.radians(subsolar_lon)

        # Cosine of solar zenith angle (dot product of surface normal with sun vector)
        cos_zenith = np.sin(lat_rad) * np.sin(subsolar_lat_rad) + np.cos(
            lat_rad
        ) * np.cos(subsolar_lat_rad) * np.cos(lon_rad - subsolar_lon_rad)

        # Illumination factor (0 = night, 1 = full daylight)
        # Add twilight zone (sun 6° below horizon)
        twilight_angle = np.radians(6)
        illumination = np.clip(
            (cos_zenith + np.sin(twilight_angle)) / np.sin(twilight_angle), 0, 1
        )

        # Apply illumination to colors
        # Night side is darker, day side is brighter
        night_factor = 0.1  # How dark the night side should be
        illuminated_colors = base_colors * (
            night_factor + (1 - night_factor) * illumination[:, np.newaxis]
        )

        return illuminated_colors

    def _approximate_land_mask(self, lon, lat):
        """Create detailed land mask based on accurate continental boundaries.

        Parameters
        ----------
        lon : array
            Longitude (-180 to 180 degrees).
        lat : array
            Latitude (-90 to 90 degrees).

        Returns
        -------
        is_land : array
            Boolean array, True for land, False for ocean.
        """
        is_land = np.zeros(len(lon), dtype=bool)

        # AFRICA - detailed regions
        # North Africa
        is_land |= (lon > -17) & (lon < 52) & (lat > 10) & (lat < 38)
        # West Africa (bulge)
        is_land |= (lon > -18) & (lon < 16) & (lat > 4) & (lat < 15)
        # Central/East Africa
        is_land |= (lon > 7) & (lon < 52) & (lat > -12) & (lat < 18)
        # Southern Africa
        is_land |= (lon > 10) & (lon < 41) & (lat > -35) & (lat < -8)
        # Horn of Africa
        is_land |= (lon > 33) & (lon < 51) & (lat > -5) & (lat < 18)
        # Madagascar
        is_land |= (lon > 43) & (lon < 51) & (lat > -26) & (lat < -12)

        # EURASIA - detailed regions
        # Europe - Western
        is_land |= (lon > -10) & (lon < 30) & (lat > 36) & (lat < 72)
        # Europe - Northern (Scandinavia)
        is_land |= (lon > 4) & (lon < 32) & (lat > 55) & (lat < 72)
        # Europe - Mediterranean
        is_land |= (lon > -10) & (lon < 45) & (lat > 35) & (lat < 47)
        # Eastern Europe/Western Russia
        is_land |= (lon > 20) & (lon < 65) & (lat > 45) & (lat < 72)
        # Central Asia
        is_land |= (lon > 45) & (lon < 95) & (lat > 35) & (lat < 55)
        # Siberia
        is_land |= (lon > 60) & (lon < 180) & (lat > 50) & (lat < 78)
        # Far East Russia
        is_land |= (lon > 100) & (lon < 180) & (lat > 40) & (lat < 70)
        # Middle East - Arabian Peninsula
        is_land |= (lon > 34) & (lon < 60) & (lat > 12) & (lat < 32)
        # Middle East - Anatolia/Levant
        is_land |= (lon > 26) & (lon < 48) & (lat > 31) & (lat < 42)
        # Indian Subcontinent
        is_land |= (lon > 68) & (lon < 89) & (lat > 8) & (lat < 36)
        # India - southern tip
        is_land |= (lon > 74) & (lon < 81) & (lat > 6) & (lat < 12)
        # Southeast Asia - mainland
        is_land |= (lon > 92) & (lon < 110) & (lat > 5) & (lat < 28)
        # Southeast Asia - Malay Peninsula
        is_land |= (lon > 99) & (lon < 105) & (lat > 1) & (lat < 8)
        # China
        is_land |= (lon > 73) & (lon < 135) & (lat > 18) & (lat < 54)
        # Korea
        is_land |= (lon > 124) & (lon < 131) & (lat > 33) & (lat < 43)
        # Japan
        is_land |= (lon > 129) & (lon < 146) & (lat > 30) & (lat < 46)
        # Indonesia - Sumatra
        is_land |= (lon > 95) & (lon < 106) & (lat > -6) & (lat < 6)
        # Indonesia - Java
        is_land |= (lon > 105) & (lon < 115) & (lat > -9) & (lat < -6)
        # Indonesia - Borneo
        is_land |= (lon > 109) & (lon < 119) & (lat > -4) & (lat < 7)
        # Indonesia - Sulawesi
        is_land |= (lon > 119) & (lon < 125) & (lat > -6) & (lat < 2)
        # Philippines
        is_land |= (lon > 117) & (lon < 127) & (lat > 5) & (lat < 20)
        # Taiwan
        is_land |= (lon > 120) & (lon < 122) & (lat > 22) & (lat < 25.5)

        # NORTH AMERICA - detailed regions
        # Canada - Western/Arctic
        is_land |= (lon > -141) & (lon < -95) & (lat > 50) & (lat < 75)
        # Canada - Eastern
        is_land |= (lon > -95) & (lon < -52) & (lat > 42) & (lat < 72)
        # Canada - Arctic islands
        is_land |= (lon > -120) & (lon < -60) & (lat > 60) & (lat < 84)
        # Alaska
        is_land |= (lon > -169) & (lon < -130) & (lat > 54) & (lat < 72)
        # USA - West Coast
        is_land |= (lon > -125) & (lon < -104) & (lat > 31) & (lat < 49)
        # USA - Central
        is_land |= (lon > -105) & (lon < -90) & (lat > 29) & (lat < 49)
        # USA - East
        is_land |= (lon > -90) & (lon < -67) & (lat > 25) & (lat < 48)
        # Florida
        is_land |= (lon > -87) & (lon < -80) & (lat > 25) & (lat < 31)
        # Mexico - mainland
        is_land |= (lon > -117) & (lon < -86) & (lat > 14) & (lat < 33)
        # Baja California
        is_land |= (lon > -115) & (lon < -109) & (lat > 22.5) & (lat < 32.5)
        # Central America
        is_land |= (lon > -93) & (lon < -77) & (lat > 7) & (lat < 18)
        # Caribbean - Cuba
        is_land |= (lon > -85) & (lon < -74) & (lat > 19.5) & (lat < 23.5)
        # Caribbean - Hispaniola
        is_land |= (lon > -75) & (lon < -68) & (lat > 17.5) & (lat < 20)
        # Caribbean - Jamaica
        is_land |= (lon > -78.5) & (lon < -76) & (lat > 17.5) & (lat < 18.5)
        # Caribbean - Puerto Rico
        is_land |= (lon > -67.5) & (lon < -65.5) & (lat > 17.8) & (lat < 18.6)

        # SOUTH AMERICA - detailed regions
        # Colombia/Venezuela
        is_land |= (lon > -79) & (lon < -59) & (lat > -4) & (lat < 13)
        # Brazil - Amazon
        is_land |= (lon > -75) & (lon < -48) & (lat > -16) & (lat < 5)
        # Brazil - Southeast
        is_land |= (lon > -55) & (lon < -34) & (lat > -34) & (lat < -3)
        # Peru/Ecuador
        is_land |= (lon > -82) & (lon < -68) & (lat > -18) & (lat < 2)
        # Bolivia/Paraguay
        is_land |= (lon > -70) & (lon < -54) & (lat > -28) & (lat < -10)
        # Chile - long strip
        is_land |= (lon > -76) & (lon < -66) & (lat > -56) & (lat < -17)
        # Argentina
        is_land |= (lon > -73) & (lon < -53) & (lat > -55) & (lat < -22)
        # Uruguay
        is_land |= (lon > -58.5) & (lon < -53) & (lat > -35) & (lat < -30)

        # AUSTRALIA - detailed
        # Western Australia
        is_land |= (lon > 112) & (lon < 129) & (lat > -35) & (lat < -13)
        # Northern Territory
        is_land |= (lon > 129) & (lon < 138) & (lat > -26) & (lat < -11)
        # Queensland/Eastern
        is_land |= (lon > 138) & (lon < 154) & (lat > -29) & (lat < -10)
        # South Australia
        is_land |= (lon > 129) & (lon < 141) & (lat > -38) & (lat < -26)
        # Tasmania
        is_land |= (lon > 144) & (lon < 149) & (lat > -44) & (lat < -40)

        # NEW ZEALAND
        # North Island
        is_land |= (lon > 172) & (lon < 179) & (lat > -42) & (lat < -34)
        # South Island
        is_land |= (lon > 166) & (lon < 175) & (lat > -47) & (lat < -40)

        # ANTARCTICA - detailed by region
        is_land |= lat < -60  # Base coverage

        # GREENLAND - more accurate
        is_land |= (lon > -73) & (lon < -18) & (lat > 59) & (lat < 84)

        # ICELAND
        is_land |= (lon > -25) & (lon < -13) & (lat > 63) & (lat < 67)

        # BRITISH ISLES
        # Great Britain
        is_land |= (lon > -6) & (lon < 2) & (lat > 49.5) & (lat < 59)
        # Ireland
        is_land |= (lon > -11) & (lon < -6) & (lat > 51) & (lat < 55.5)

        # PACIFIC ISLANDS
        # New Guinea
        is_land |= (lon > 140) & (lon < 151) & (lat > -11) & (lat < -1)
        # Sri Lanka
        is_land |= (lon > 79.5) & (lon < 82) & (lat > 5.5) & (lat < 10)

        return is_land

    def _create_sky_grid(self, n_points):
        """Create a grid of RA/Dec points optimized for Mollweide projection.

        Parameters
        ----------
        n_points : int
            Number of points per axis for the grid.

        Returns
        -------
        ra_flat : array
            Flattened RA coordinates (0-360 degrees).
        dec_flat : array
            Flattened Dec coordinates (-90-90 degrees).
        """
        # Linear declination sampling
        dec_samples = np.linspace(-90, 90, n_points)

        # For each declination, calculate how many RA samples we need
        # based on cos(dec) for even visual density in Mollweide projection
        cos_factors = np.cos(np.radians(dec_samples))
        n_ra_array = np.maximum(8, (n_points * 2 * cos_factors).astype(int))

        ra_flat = np.concatenate(
            [np.linspace(0, 360, n, endpoint=False) for n in n_ra_array]
        )
        dec_flat = np.concatenate(
            [np.full(n, dec) for n, dec in zip(n_ra_array, dec_samples)]
        )

        return ra_flat, dec_flat

    def _convert_ra_for_plotting(self, ra_vals):
        """Convert RA coordinates for Mollweide projection plotting.

        Parameters
        ----------
        ra_vals : array
            RA values in degrees (0-360).

        Returns
        -------
        array
            RA values converted for plotting (-180 to 180).
        """
        return ra_vals - 180

    def _plot_points_on_sky(
        self,
        ra_vals,
        dec_vals,
        color,
        alpha=0.3,
        size=20,
        marker="s",
        label=None,
        zorder=1,
    ):
        """Plot points on the sky map.

        Parameters
        ----------
        ra_vals : array
            RA coordinates in degrees (0-360).
        dec_vals : array
            Dec coordinates in degrees (-90-90).
        color : str
            Color for the points.
        alpha : float, optional
            Transparency (default: 0.3).
        size : int, optional
            Point size (default: 20).
        marker : str, optional
            Marker style (default: "s" for square).
        label : str, optional
            Legend label.
        zorder : float, optional
            Z-order for layering (default: 1).
        """
        ra_plot = self._convert_ra_for_plotting(ra_vals)

        self.ax.scatter(
            np.deg2rad(ra_plot),
            np.deg2rad(dec_vals),
            s=size,
            c=color,
            alpha=alpha,
            marker=marker,
            edgecolors="none",
            label=label,
            zorder=zorder,
            rasterized=True,  # Rasterize for faster rendering
        )

    def _plot_single_constraint(
        self,
        name: str,
        constraint_func: Constraint,
        color: str,
        utime: float,
        body_ra: float,
        body_dec: float,
    ) -> None:
        """Plot a single constraint region.

        Parameters
        ----------
        name : str
            Name of the constraint.
        constraint_func : Constraint
            Function to check if a point violates the constraint.
        color : str
            Color for the constraint region.
        utime : float
            Unix timestamp.
        ra_grid : array
            RA values to check.
        dec_grid : array
            Dec values to check.
        body_ra : float or None
            RA of celestial body (for marker).
        body_dec : float or None
            Dec of celestial body (for marker).
        """
        # Check if we have pre-computed constraints
        if (
            hasattr(self, "_constraint_cache")
            and name.lower() in self._constraint_cache["constraints"]
        ):
            cache = self._constraint_cache

            # Verify cache consistency
            current_ra_flat, current_dec_flat = self._create_sky_grid(
                self.n_grid_points
            )
            if len(cache["ra_grid"]) == len(current_ra_flat) and len(
                cache["dec_grid"]
            ) == len(current_dec_flat):
                # Use pre-computed constraints
                ra_flat = cache["ra_grid"]
                dec_flat = cache["dec_grid"]

                # Find time index in cache
                time_idx = self._find_time_index(utime)
                cache_time_idx = np.where(cache["time_indices"] == time_idx)[0]
                if (
                    len(cache_time_idx) > 0
                    and cache_time_idx[0] < cache["constraints"][name.lower()].shape[1]
                ):
                    cache_time_idx = cache_time_idx[0]
                    # Cache shape is (n_points, n_times), so index with [:, time_idx]
                    constrained_coords = cache["constraints"][name.lower()][
                        :, cache_time_idx
                    ]

                    # Ensure constrained_coords is a boolean array
                    constrained_coords = np.asarray(constrained_coords, dtype=bool)

                    # Plot constrained region
                    if constrained_coords.any():
                        points = np.column_stack(
                            (ra_flat[constrained_coords], dec_flat[constrained_coords])
                        )

                        ra_vals = points[:, 0]
                        dec_vals = points[:, 1]

                        self._plot_points_on_sky(
                            ra_vals,
                            dec_vals,
                            color,
                            alpha=self.constraint_alpha,
                            label=f"{name} Cons.",
                            zorder=1,
                        )

                    # Mark celestial body position
                    if body_ra is not None and body_dec is not None:
                        ra_plot = self._convert_ra_for_plotting(np.array([body_ra]))
                        self.ax.plot(
                            np.deg2rad(ra_plot),
                            np.deg2rad(body_dec),
                            marker="o",
                            markersize=8,
                            markeredgecolor="black",
                            markerfacecolor=color,
                            markeredgewidth=0,
                            zorder=3,
                        )
                    return

        # Fall back to real-time evaluation if no cache available
        # Get sky grid points
        ra_flat, dec_flat = self._create_sky_grid(self.n_grid_points)

        constrained_coords = constraint_func.in_constraint_batch(
            ephemeris=self.ditl.ephem,
            target_ras=ra_flat,
            target_decs=dec_flat,
            times=dtutcfromtimestamp(utime),
        ).T[0]

        # Plot constrained region
        if constrained_coords.any():
            points = np.column_stack(
                (ra_flat[constrained_coords], dec_flat[constrained_coords])
            )

            ra_vals = points[:, 0]
            dec_vals = points[:, 1]

            self._plot_points_on_sky(
                ra_vals,
                dec_vals,
                color,
                alpha=self.constraint_alpha,
                label=f"{name} Cons.",
                zorder=1,
            )

        # Mark celestial body position
        if body_ra is not None and body_dec is not None:
            ra_plot = self._convert_ra_for_plotting(np.array([body_ra]))
            self.ax.plot(
                np.deg2rad(ra_plot),
                np.deg2rad(body_dec),
                marker="o",
                markersize=12,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0,
                label=name,
                zorder=3,
            )

    def _plot_current_pointing(self, ra, dec, mode):
        """Plot the current spacecraft pointing direction.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        mode : ACSMode
            Current ACS mode.
        """
        # Convert RA for plotting (RA=0 on left)
        ra_plot = self._convert_ra_for_plotting(np.array([ra]))[0]

        # Color based on ACS mode
        mode_name = mode.name if hasattr(mode, "name") else str(mode)
        mode_colors = (
            self.config.mode_colors
            if self.config
            else {
                "SCIENCE": "green",
                "SLEWING": "orange",
                "SAA": "purple",
                "PASS": "cyan",
                "CHARGING": "yellow",
                "SAFE": "red",
            }
        )
        color = mode_colors.get(mode_name, "red")

        # Plot with distinctive marker
        self.ax.plot(
            np.deg2rad(ra_plot),
            np.deg2rad(dec),
            marker="*",
            markersize=25,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0,
            label="Pointing",
            zorder=5,
        )

        # Add a small circle around it for visibility
        circle = plt.Circle(
            (np.deg2rad(ra_plot), np.deg2rad(dec)),
            radius=np.deg2rad(5),
            fill=False,
            edgecolor=color,
            linewidth=2,
            zorder=4,
        )
        self.ax.add_patch(circle)

    def _setup_plot_appearance(self, utime):
        """Set up plot labels, grid, and appearance.

        Parameters
        ----------
        utime : float
            Unix timestamp for the title.
        """
        # Grid
        self.ax.grid(True, alpha=0.3)

        # Labels
        font_family = self.config.font_family if self.config else "Helvetica"
        label_font_size = self.config.label_font_size if self.config else 10
        title_font_size = self.config.title_font_size if self.config else 12
        title_prop = FontProperties(
            family=font_family, size=title_font_size, weight="bold"
        )
        tick_font_size = self.config.tick_font_size if self.config else 9

        self.ax.set_xlabel(
            "Right Ascension (deg)", fontsize=label_font_size, fontfamily=font_family
        )
        self.ax.set_ylabel(
            "Declination (deg)", fontsize=label_font_size, fontfamily=font_family
        )

        # Title with time
        dt = dtutcfromtimestamp(utime)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        self.ax.set_title(
            f"Spacecraft Pointing at {time_str}", fontproperties=title_prop, pad=20
        )

        # Legend (reduce clutter by only showing unique labels)
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # Add ACS mode color legend entries
        from matplotlib.lines import Line2D

        mode_colors = (
            self.config.mode_colors
            if self.config
            else {
                "SCIENCE": "green",
                "SLEWING": "orange",
                "SAA": "purple",
                "PASS": "cyan",
                "CHARGING": "yellow",
                "SAFE": "red",
            }
        )

        # Add separator and ACS mode entries
        mode_handles = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor=color,
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1,
                label=f"{mode.lower().capitalize()}",
            )
            for mode, color in mode_colors.items()
        ]

        all_handles = list(by_label.values()) + mode_handles
        all_labels = list(by_label.keys()) + [h.get_label() for h in mode_handles]

        self.ax.legend(
            all_handles,
            all_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=self.config.legend_font_size if self.config else 8,
            title="ACS Modes",
            prop={"family": font_family},
        )

        # Set RA tick labels (mollweide uses radians internally)
        # RA=0 is on the left at -180°
        ra_ticks = np.deg2rad(np.array([-180, -120, -60, 0, 60, 120, 180]))
        self.ax.set_xticks(ra_ticks)
        self.ax.set_xticklabels(
            ["0°", "60°", "120°", "180°", "240°", "300°", "360°"],
            fontsize=tick_font_size,
            fontfamily=font_family,
        )
        # Set y-axis tick labels
        self.ax.tick_params(axis="y", labelsize=tick_font_size)
        for label in self.ax.get_yticklabels():
            try:
                label.set_fontfamily(font_family)
            except Exception:
                # Some environments return MagicMock objects during tests; ignore
                pass


def save_sky_pointing_frames(
    ditl, output_dir, figsize=(14, 8), n_grid_points=50, frame_interval=1, config=None
):
    """Save individual frames of the sky pointing visualization.

    Useful for creating animations or reviewing specific time steps.

    Parameters
    ----------
    ditl : DITL or QueueDITL
        The DITL simulation object.
    output_dir : str
        Directory to save frames.
    figsize : tuple
        Figure size.
    n_grid_points : int
        Grid resolution for constraints.
    frame_interval : int
        Save every Nth time step (default: 1 = save all).
    config : Config, optional
        Configuration object containing visualization settings.

    Returns
    -------
    list
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create controller without controls
    fig, ax, _ = plot_sky_pointing(
        ditl,
        figsize=figsize,
        n_grid_points=n_grid_points,
        show_controls=False,
        config=config,
    )

    controller = SkyPointingController(
        ditl=ditl,
        fig=fig,
        ax=ax,
        n_grid_points=n_grid_points,
        time_step_seconds=ditl.step_size,
        constraint_alpha=1.0,
        config=config,
    )

    saved_files = []
    for idx in range(0, len(ditl.utime), frame_interval):
        utime = ditl.utime[idx]
        controller.update_plot(utime)

        # Save frame
        filename = os.path.join(output_dir, f"sky_pointing_frame_{idx:05d}.png")
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        saved_files.append(filename)

        if (idx + 1) % 100 == 0:
            print(f"Saved {idx + 1}/{len(ditl.utime)} frames")

    plt.close(fig)
    print(f"Saved {len(saved_files)} frames to {output_dir}")
    return saved_files


def save_sky_pointing_movie(
    ditl,
    output_file,
    fps=10,
    figsize=(14, 8),
    n_grid_points=50,
    frame_interval=1,
    dpi=100,
    codec="h264",
    bitrate=1800,
    config=None,
    show_progress=True,
):
    """Export the entire DITL sky pointing visualization as a movie.

    Creates an animated movie showing how spacecraft pointing and constraints
    evolve throughout the entire DITL simulation.

    Parameters
    ----------
    ditl : DITL or QueueDITL
        The DITL simulation object with completed simulation data.
    output_file : str
        Output filename for the movie. Extension determines format:
        - '.mp4' for MP4 video (requires ffmpeg)
        - '.gif' for animated GIF (requires pillow)
        - '.avi' for AVI video (requires ffmpeg)
    fps : float, optional
        Frames per second in the output movie (default: 10).
        Lower values create slower playback, higher values faster playback.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (14, 8)).
    n_grid_points : int, optional
        Number of grid points per axis for constraint region calculation
        (default: 50). Lower values render faster but with less detail.
    frame_interval : int, optional
        Use every Nth time step from the DITL (default: 1 = use all frames).
        Higher values create shorter movies with faster playback.
    dpi : int, optional
        Resolution in dots per inch (default: 100).
        Higher values create larger, higher quality files.
    codec : str, optional
        Video codec for MP4/AVI output (default: 'h264').
        Other options: 'mpeg4', 'libx264', etc.
    bitrate : int, optional
        Video bitrate in kbps (default: 1800).
        Higher values create better quality but larger files.
    config : Config, optional
        Configuration object containing visualization settings. If None,
        uses ditl.config.visualization if available.
    show_progress : bool, optional
        Whether to show a progress bar using tqdm (default: True).

    Returns
    -------
    str
        Path to the saved movie file.

    Raises
    ------
    ValueError
        If output format is not supported or required codecs are not available.
    RuntimeError
        If movie encoding fails.

    Examples
    --------
    >>> # Create MP4 movie at 15 fps
    >>> save_sky_pointing_movie(ditl, "pointing.mp4", fps=15)

    >>> # Create animated GIF (slower, larger file)
    >>> save_sky_pointing_movie(ditl, "pointing.gif", fps=5)

    >>> # Fast preview with reduced detail
    >>> save_sky_pointing_movie(
    ...     ditl, "preview.mp4",
    ...     fps=20, frame_interval=5, n_grid_points=30
    ... )

    >>> # Disable progress bar for automated scripts
    >>> save_sky_pointing_movie(ditl, "pointing.mp4", show_progress=False)

    Notes
    -----
    - MP4 and AVI formats require ffmpeg to be installed on your system
    - GIF format requires the pillow library
    - Frame rate (fps) controls playback speed, not simulation time
    - frame_interval controls which simulation time steps are included
    - Lower n_grid_points speeds up rendering but reduces visual quality
    - The movie shows the same view as plot_sky_pointing() but automated
    - Progress bar requires tqdm library
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    # Try to import tqdm, fall back to no progress bar if unavailable
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            show_progress = False
            print("Note: tqdm not available, progress bar disabled")

    # Validate inputs
    if not hasattr(ditl, "plan") or len(ditl.plan) == 0:
        raise ValueError("DITL simulation has no pointings. Run calc() first.")
    if not hasattr(ditl, "utime") or len(ditl.utime) == 0:
        raise ValueError("DITL has no time data. Run calc() first.")

    # Determine file format and writer
    file_ext = os.path.splitext(output_file)[1].lower()
    if file_ext == ".gif":
        writer_class = PillowWriter
        writer_kwargs = {"fps": fps}
    elif file_ext in [".mp4", ".avi"]:
        writer_class = FFMpegWriter
        writer_kwargs = {
            "fps": fps,
            "codec": codec,
            "bitrate": bitrate,
        }
    else:
        raise ValueError(
            f"Unsupported output format: {file_ext}. Use .mp4, .avi, or .gif"
        )

    # Get visualization config
    config = _get_visualization_config(ditl, config)

    # Create figure without controls
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "mollweide"})

    # Create controller for updates
    controller = SkyPointingController(
        ditl=ditl,
        fig=fig,
        ax=ax,
        n_grid_points=n_grid_points,
        time_step_seconds=ditl.step_size,
        constraint_alpha=1.0,
        config=config,
    )

    # Select time steps to animate
    time_indices = list(range(0, len(ditl.utime), frame_interval))
    total_frames = len(time_indices)

    # Pre-compute constraints for all time steps to be rendered
    controller._precompute_constraints(time_indices)

    print(f"Creating movie with {total_frames} frames at {fps} fps...")
    print(f"Movie duration: {total_frames / fps:.1f} seconds")
    print(f"Output: {output_file}")

    # Set up the writer
    writer = writer_class(**writer_kwargs)

    try:
        with writer.saving(fig, output_file, dpi=dpi):
            # Create iterator with optional progress bar
            if show_progress:
                iterator = tqdm(
                    enumerate(time_indices),
                    total=total_frames,
                    desc="Rendering frames",
                    unit="frame",
                    ncols=80,
                )
            else:
                iterator = enumerate(time_indices)

            for frame_num, idx in iterator:
                utime = ditl.utime[idx]
                controller.update_plot(utime)

                # Save this frame
                writer.grab_frame()

        plt.close(fig)
        print(f"\nSuccessfully saved movie to {output_file}")
        return output_file

    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Failed to create movie: {e}") from e
