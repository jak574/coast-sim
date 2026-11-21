from pydantic import Field, field_validator

from ..common.enums import ACSMode
from ._base import ConfigModel

DEFAULT_MODE_COLORS: dict[str, str] = {
    ACSMode.SCIENCE.name: "green",
    ACSMode.SLEWING.name: "orange",
    ACSMode.SAA.name: "purple",
    ACSMode.PASS.name: "cyan",
    ACSMode.CHARGING.name: "yellow",
    ACSMode.SAFE.name: "red",
    ACSMode.IDLE.name: "lightsteelblue",
    ACSMode.DESAT.name: "magenta",
    ACSMode.ACTIVITY.name: "saddlebrown",
}
"""A color for every ACS mode, each different from the others and from the
gray that plots use for a mode value they do not recognize."""


class VisualizationConfig(ConfigModel):
    """Configuration for visualization settings."""

    mode_colors: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_MODE_COLORS),
        description=(
            "Color of each ACS mode, by mode name. A mode left out of a "
            "supplied mapping takes its default color"
        ),
    )

    @field_validator("mode_colors")
    @classmethod
    def _color_every_mode(cls, colors: dict[str, str]) -> dict[str, str]:
        """Give each ACS mode missing from ``colors`` its default color, so a
        mapping written before a mode existed still colors that mode."""
        return {**DEFAULT_MODE_COLORS, **colors}

    # Plot appearance settings
    font_family: str = "Helvetica"
    title_font_size: int = 12
    label_font_size: int = 10
    legend_font_size: int = 8
    tick_font_size: int = 9

    # Sky pointing visualization settings
    figsize: tuple[int, int] = (14, 8)
    n_grid_points: int = 100
    constraint_alpha: float = 0.3
    time_step_seconds: float = 60.0
    optimal_roll_cache_sun_bucket_deg: float = 0.02
    optimal_roll_cache_max_entries: int = 64

    # DITL timeline settings
    timeline_figsize: tuple[int, int] = (12, 5)
    timeline_font_size: int = 11
    orbit_period_seconds: float = 5762.0

    # Data management telemetry settings
    data_telemetry_figsize: tuple[int, int] = (12, 10)
