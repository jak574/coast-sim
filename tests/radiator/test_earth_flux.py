"""The Earth's load on a radiator: uniform by default, and with the
view-factor model scaled by distance, absorbed with the emissivity, and with
reflected sunlight."""

from datetime import timedelta

import numpy as np
import pytest

from conops.common.geodesy import MEAN_EARTH_RADIUS_KM
from conops.config import Radiator, RadiatorConfiguration, RadiatorOrientation
from tests.scheduler.planning_scenario import BEGIN, make_config


def _radiator(**fields: object) -> Radiator:
    defaults: dict[str, object] = {
        "efficiency": 1.0,
        "emissivity": 0.8,
        "absorptivity": 0.25,
        "sun_loading_factor": 1.0,
        "earth_loading_factor": 1.0,
        "earth_ir_flux_w_per_m2": 240.0,
        "solar_constant_w_per_m2": 1360.0,
    }
    return Radiator(**{**defaults, **fields})


def test_the_uniform_model_ignores_distance_and_uses_the_solar_absorptivity() -> None:
    radiator = _radiator()
    assert radiator.earth_loads(0.6, 0.1, 1.0) == (0.6, 0.0)
    assert radiator.absorbed_power_w(0.0, 0.6) == pytest.approx(0.25 * 240.0 * 0.6)


def test_the_view_factor_model_scales_with_distance_and_uses_emissivity() -> None:
    radiator = _radiator(earth_flux_model="view_factor")
    infrared, albedo = radiator.earth_loads(0.6, 0.5, 1.0)
    assert infrared == pytest.approx(0.6 * 0.25)
    # Directly below the Sun the whole visible Earth is sunlit.
    assert albedo == pytest.approx(infrared)
    # With no earth_albedo, reflected sunlight adds nothing.
    assert radiator.absorbed_power_w(0.0, 0.0, albedo) == 0.0
    assert radiator.absorbed_power_w(0.0, infrared) == pytest.approx(
        0.8 * 240.0 * 0.6 * 0.25
    )
    # Four times the distance, a sixteenth of the load.
    far, _ = radiator.earth_loads(0.6, 0.125, 1.0)
    assert far == pytest.approx(infrared / 16.0)


def test_albedo_follows_the_sunlit_fraction_below_the_spacecraft() -> None:
    radiator = _radiator(earth_flux_model="view_factor", earth_albedo=0.3)
    infrared, noon = radiator.earth_loads(1.0, 0.5, 1.0)
    assert noon == pytest.approx(0.25)
    assert radiator.absorbed_power_w(0.0, 0.0, noon) == pytest.approx(
        0.25 * 1360.0 * 0.3 * 0.25
    )
    assert radiator.earth_loads(1.0, 0.5, 0.5)[1] == pytest.approx(0.125)
    # Over the night side no sunlight is reflected.
    assert radiator.earth_loads(1.0, 0.5, -0.2)[1] == 0.0
    with pytest.raises(ValueError, match="view_factor"):
        _radiator(earth_albedo=0.3)


def test_the_metrics_use_the_orbit_distance_and_sun_direction() -> None:
    config = make_config(2)
    ephem = config.constraint.ephem
    normal = RadiatorOrientation(normal=(0.0, 0.0, 1.0))
    uniform = _radiator(name="uniform", orientation=normal)
    view = _radiator(
        name="view",
        orientation=normal,
        earth_flux_model="view_factor",
        earth_albedo=0.3,
    )
    radiators = RadiatorConfiguration(radiators=[uniform, view])
    seen = 0
    for minute in range(0, 120, 5):
        utime = (BEGIN + timedelta(minutes=minute)).timestamp()
        index = ephem.index(BEGIN + timedelta(minutes=minute))
        metrics = radiators.exposure_metrics(105.0, 10.0, utime, ephem)
        first, second = metrics["per_radiator"]
        assert first["earth_ir_exposure"] == first["earth_exposure"]
        assert first["albedo_exposure"] == 0.0
        if second["earth_exposure"] == 0.0:
            continue
        seen += 1
        position = np.asarray(ephem.gcrs_pv.position[index], dtype=float)
        ratio = MEAN_EARTH_RADIUS_KM / float(np.linalg.norm(position))
        assert second["earth_ir_exposure"] == pytest.approx(
            second["earth_exposure"] * ratio**2
        )
        sun = np.asarray(ephem.sun_pv.position[index], dtype=float)
        cosine = float(
            np.dot(position, sun) / np.linalg.norm(position) / np.linalg.norm(sun)
        )
        assert second["albedo_exposure"] == pytest.approx(
            second["earth_ir_exposure"] * max(0.0, cosine), abs=2e-4
        )
    assert seen > 0
