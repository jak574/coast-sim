"""Every ACS mode has a color of its own in the mode plots."""

from types import SimpleNamespace

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from conops.common import ACSMode
from conops.config import VisualizationConfig
from conops.visualization import (
    plot_acs_mode_distribution,
    plot_acs_mode_distribution_plotly,
)

UNKNOWN = mcolors.to_rgba("gray")


def _ditl(config: VisualizationConfig | None = None) -> SimpleNamespace:
    """A stand-in run that spends one step in each ACS mode."""
    return SimpleNamespace(
        telemetry=SimpleNamespace(
            housekeeping=SimpleNamespace(acs_mode=[mode.value for mode in ACSMode])
        ),
        config=SimpleNamespace(
            name="test", visualization=config or VisualizationConfig()
        ),
    )


def test_default_colors_cover_every_mode_once() -> None:
    colors = VisualizationConfig().mode_colors
    assert set(colors) == {mode.name for mode in ACSMode}
    rgba = [mcolors.to_rgba(color) for color in colors.values()]
    assert len(set(rgba)) == len(ACSMode)
    assert UNKNOWN not in rgba


def test_a_partial_mapping_keeps_the_other_modes_colored() -> None:
    colors = VisualizationConfig(mode_colors={"SCIENCE": "navy"}).mode_colors
    assert colors["SCIENCE"] == "navy"
    assert set(colors) == {mode.name for mode in ACSMode}


def test_the_pie_chart_colors_every_mode_differently() -> None:
    fig, ax = plot_acs_mode_distribution(_ditl())
    wedges = [tuple(patch.get_facecolor()) for patch in ax.patches]
    plt.close(fig)
    assert len(wedges) == len(ACSMode)
    assert len(set(wedges)) == len(ACSMode)
    assert UNKNOWN not in wedges


def test_the_plotly_pie_chart_colors_every_mode_differently() -> None:
    partial = VisualizationConfig(mode_colors={"SCIENCE": "green"})
    fig = plot_acs_mode_distribution_plotly(_ditl(partial))
    colors = list(fig.data[0].marker.colors)
    assert len(colors) == len(ACSMode)
    assert len(set(colors)) == len(ACSMode)
    assert "gray" not in colors
