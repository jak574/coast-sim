"""A radiator with a thermal mass has a temperature that is integrated over a
run; one without keeps its fixed temperature and its earlier heat flow."""

import math
from datetime import timedelta

import pytest

from conops import DITL, QueueDITL
from conops.common import ACSMode
from conops.config import PowerDraw, Radiator, RadiatorConfiguration
from conops.config.radiator import STEFAN_BOLTZMANN_W_PER_M2_K4
from conops.schedulers import PriorityScheduler
from conops.simulation.radiator_thermal import RadiatorThermalModel
from tests.scheduler.planning_scenario import BEGIN, make_config

HOUR = 3600.0


def _radiator(**fields: object) -> Radiator:
    defaults: dict[str, object] = {
        "name": "Bus",
        "efficiency": 1.0,
        "emissivity": 0.8,
        "radiator_temperature_k": 280.0,
        "thermal_mass_j_per_k": 20_000.0,
        "heat_load_w": 150.0,
    }
    return Radiator(**{**defaults, **fields})


def test_without_a_thermal_mass_the_heat_flow_is_unchanged() -> None:
    radiator = Radiator()
    # 0.85 sigma 300^4 is 390 W/m2, capped at the 220 W/m2 coefficient.
    assert radiator.heat_dissipation_w(0.0, 0.0) == pytest.approx(0.9 * 220.0)
    assert radiator.temperature_after(300.0, HOUR, 500.0, 1.0, 1.0) == 300.0
    assert RadiatorThermalModel.for_radiators(RadiatorConfiguration()) is None


def test_equilibrium_balances_the_heat_received_and_emitted() -> None:
    radiator = _radiator()
    equilibrium = radiator.equilibrium_temperature_k(150.0, 0.5, 0.25)
    assert radiator.emitted_power_w(equilibrium) == pytest.approx(
        150.0 + radiator.absorbed_power_w(0.5, 0.25)
    )
    # 150 W from 1 m2 at emissivity 0.8 to a 3 K sink.
    expected = (3.0**4 + 150.0 / (0.8 * STEFAN_BOLTZMANN_W_PER_M2_K4)) ** 0.25
    assert radiator.equilibrium_temperature_k(150.0, 0.0, 0.0) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("start", [150.0, 280.0, 400.0])
def test_temperature_approaches_equilibrium_without_passing_it(start: float) -> None:
    radiator = _radiator()
    equilibrium = radiator.equilibrium_temperature_k(150.0, 0.0, 0.0)
    temperature = start
    for _ in range(48):
        after = radiator.temperature_after(temperature, HOUR, 150.0, 0.0, 0.0)
        assert abs(after - equilibrium) <= abs(temperature - equilibrium)
        assert (after - equilibrium) * (start - equilibrium) >= 0.0
        temperature = after
    assert temperature == pytest.approx(equilibrium, abs=0.01)


def test_one_long_step_agrees_with_many_short_steps() -> None:
    radiator = _radiator()
    long = radiator.temperature_after(280.0, 600.0, 150.0, 1.0, 0.0)
    short = 280.0
    for _ in range(6000):
        short = radiator.temperature_after(short, 0.1, 150.0, 1.0, 0.0)
    assert long == pytest.approx(short, abs=0.05)


def test_a_small_step_follows_the_energy_balance() -> None:
    radiator = _radiator(radiator_temperature_k=280.0)
    net_w = 150.0 - radiator.emitted_power_w(280.0)
    after = radiator.temperature_after(280.0, 1.0, 150.0, 0.0, 0.0)
    assert after - 280.0 == pytest.approx(net_w * 1.0 / 20_000.0, rel=1e-3)


def test_a_larger_thermal_mass_changes_temperature_more_slowly() -> None:
    light = _radiator(thermal_mass_j_per_k=5_000.0)
    heavy = _radiator(thermal_mass_j_per_k=500_000.0)
    light_rise = light.temperature_after(200.0, 600.0, 150.0, 0.0, 0.0) - 200.0
    heavy_rise = heavy.temperature_after(200.0, 600.0, 150.0, 0.0, 0.0) - 200.0
    assert 0.0 < heavy_rise < light_rise


def test_a_radiator_that_cannot_emit_stores_all_the_heat() -> None:
    radiator = _radiator(emissivity=0.0)
    assert radiator.temperature_after(280.0, 100.0, 150.0, 0.0, 0.0) == pytest.approx(
        280.0 + 150.0 * 100.0 / 20_000.0
    )
    assert math.isinf(radiator.equilibrium_temperature_k(150.0, 0.0, 0.0))


def test_heat_inputs_are_rejected_without_a_thermal_mass() -> None:
    with pytest.raises(ValueError, match="thermal_mass_j_per_k"):
        Radiator(heat_load_w=10.0)
    with pytest.raises(ValueError, match="subsystem"):
        _radiator(subsystem="optical bench", subsystem_power_fraction=0.5)
    with pytest.raises(ValueError, match="unique"):
        RadiatorConfiguration(radiators=[_radiator(), _radiator()])


def test_the_model_holds_each_steps_inputs_until_the_next() -> None:
    radiator = _radiator(heat_load_w=0.0, subsystem_power_fraction=0.5)
    fixed = Radiator(name="Fixed")
    model = RadiatorThermalModel.for_radiators(
        RadiatorConfiguration(radiators=[fixed, radiator])
    )
    assert model is not None
    dark = [(0.0, 0.0), (0.0, 0.0)]
    model.advance(0.0, dark, {"spacecraft_bus": 300.0})
    assert model.temperatures_k == {"Bus": 280.0}
    model.advance(600.0, dark, {"spacecraft_bus": 0.0})
    expected = radiator.temperature_after(280.0, 600.0, 150.0, 0.0, 0.0)
    assert model.temperatures_k == {"Bus": pytest.approx(expected)}
    assert model.heat_dissipation_w(dark) == pytest.approx(
        fixed.heat_dissipation_w(0.0, 0.0) + radiator.emitted_power_w(expected)
    )
    with pytest.raises(ValueError, match="exposures"):
        model.advance(1200.0, dark[:1], {})


@pytest.mark.parametrize("kind", ["dispatch", "plan"])
def test_a_run_reports_temperatures_that_follow_the_bus_power(kind: str) -> None:
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.power_draw = PowerDraw(nominal_power=400.0)
    radiator = _radiator(
        heat_load_w=0.0,
        subsystem_power_fraction=0.5,
        radiator_temperature_k=200.0,
        absorptivity=0.0,
    )
    config.spacecraft_bus.radiators = RadiatorConfiguration(radiators=[radiator])
    end = BEGIN + timedelta(hours=4)
    if kind == "dispatch":
        run: DITL | QueueDITL = QueueDITL(config=config, begin=BEGIN, end=end)
    else:
        plan = PriorityScheduler(
            config, [], BEGIN, end, include_passes=False
        ).schedule()
        run = DITL(config=config, plan=plan, begin=BEGIN, end=end)
    assert run.calc()

    samples = run.telemetry.housekeeping
    temperatures = [sample["Bus"] for sample in samples.radiator_temperatures_k]
    assert temperatures[0] == 200.0
    assert {sample.acs_mode for sample in samples} == {ACSMode.IDLE}
    # 200 W of the 400 W bus load reaches the radiator the whole run.
    expected = [200.0]
    for _ in temperatures[1:]:
        expected.append(radiator.temperature_after(expected[-1], 60.0, 200.0, 0.0, 0.0))
    assert temperatures == pytest.approx(expected)
    equilibrium = radiator.equilibrium_temperature_k(200.0, 0.0, 0.0)
    assert 200.0 < temperatures[-1] < equilibrium
    assert equilibrium - temperatures[-1] < 0.2 * (equilibrium - 200.0)
    assert samples.radiator_heat_dissipation_w[-1] == pytest.approx(
        radiator.emitted_power_w(temperatures[-1])
    )
    # The configuration still holds the starting temperature.
    assert radiator.radiator_temperature_k == 200.0
