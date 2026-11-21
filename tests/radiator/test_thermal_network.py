"""Interior thermal nodes and conductive couplings: heat is conserved between
coupled nodes, the steady state is the one the energy balance gives, and a run
reports the node temperatures."""

from datetime import timedelta

import pytest

from conops import DITL, QueueDITL
from conops.common import ACSMode
from conops.config import (
    Instrument,
    PowerDraw,
    Radiator,
    RadiatorConfiguration,
    SpacecraftBus,
    ThermalCoupling,
    ThermalNetwork,
    ThermalNode,
)
from conops.config.instrument import InstrumentMode
from conops.ditl import RollingHorizonDITL
from conops.schedulers import PriorityScheduler
from conops.simulation.thermal_network import ThermalNetworkModel
from tests.scheduler.planning_scenario import BEGIN, make_config

DARK = (0.0, 0.0, 0.0)


def _radiator(**fields: object) -> Radiator:
    defaults: dict[str, object] = {
        "name": "Panel",
        "efficiency": 1.0,
        "emissivity": 0.8,
        "radiator_temperature_k": 250.0,
        "thermal_mass_j_per_k": 10_000.0,
    }
    return Radiator(**{**defaults, **fields})


def _coupled(conductance: float = 5.0, **network: object) -> ThermalNetworkModel:
    """A 60 W bench conducting to a radiating panel."""
    return ThermalNetworkModel(
        [_radiator()],
        ThermalNetwork(
            nodes=[
                ThermalNode(
                    name="Bench",
                    thermal_mass_j_per_k=40_000.0,
                    temperature_k=290.0,
                    heat_load_w=60.0,
                )
            ],
            couplings=[
                ThermalCoupling(
                    between=("Bench", "Panel"), conductance_w_per_k=conductance
                )
            ],
            **network,
        ),
    )


def test_two_coupled_interior_nodes_conserve_heat_and_equalise() -> None:
    model = ThermalNetworkModel(
        [],
        ThermalNetwork(
            nodes=[
                ThermalNode(name="A", thermal_mass_j_per_k=1000.0, temperature_k=320.0),
                ThermalNode(name="B", thermal_mass_j_per_k=3000.0, temperature_k=280.0),
            ],
            couplings=[ThermalCoupling(between=("A", "B"), conductance_w_per_k=2.0)],
        ),
    )
    model.advance(0.0, [], {})
    heat = 1000.0 * 320.0 + 3000.0 * 280.0
    previous = 40.0
    for step in range(1, 200):
        model.advance(60.0 * step, [], {})
        t = model.node_temperatures_k
        assert 1000.0 * t["A"] + 3000.0 * t["B"] == pytest.approx(heat)
        assert 0.0 <= t["A"] - t["B"] <= previous
        previous = t["A"] - t["B"]
    assert t["A"] == pytest.approx(heat / 4000.0, abs=1e-3)
    assert t["B"] == pytest.approx(heat / 4000.0, abs=1e-3)


def test_an_uncoupled_interior_node_stores_its_heat_load() -> None:
    model = ThermalNetworkModel(
        [],
        ThermalNetwork(
            nodes=[
                ThermalNode(
                    name="A",
                    thermal_mass_j_per_k=1000.0,
                    temperature_k=300.0,
                    heat_load_w=5.0,
                )
            ]
        ),
    )
    model.advance(0.0, [], {})
    model.advance(600.0, [], {})
    assert model.node_temperatures_k == {"A": pytest.approx(303.0)}


def test_the_steady_state_passes_the_load_across_the_coupling() -> None:
    panel = _radiator()
    steady_panel = panel.equilibrium_temperature_k(60.0, 0.0, 0.0)

    settled = _coupled(start_at_equilibrium=True)
    settled.advance(0.0, [DARK], {})
    assert settled.radiator_temperatures_k["Panel"] == pytest.approx(steady_panel)
    # 60 W crosses 5 W/K, so the bench is 12 K above the panel.
    assert settled.node_temperatures_k["Bench"] == pytest.approx(steady_panel + 12.0)
    settled.advance(3600.0, [DARK], {})
    assert settled.node_temperatures_k["Bench"] == pytest.approx(steady_panel + 12.0)

    integrated = _coupled()
    integrated.advance(0.0, [DARK], {})
    assert integrated.node_temperatures_k == {"Bench": 290.0}
    for step in range(1, 2000):
        integrated.advance(600.0 * step, [DARK], {})
    assert integrated.radiator_temperatures_k["Panel"] == pytest.approx(
        steady_panel, abs=0.01
    )
    assert integrated.node_temperatures_k["Bench"] == pytest.approx(
        steady_panel + 12.0, abs=0.01
    )


def test_short_steps_and_long_steps_agree() -> None:
    long, short = _coupled(), _coupled()
    long.advance(0.0, [DARK], {})
    short.advance(0.0, [DARK], {})
    for step in range(1, 11):
        long.advance(600.0 * step, [DARK], {})
    for step in range(1, 6001):
        short.advance(1.0 * step, [DARK], {})
    assert long.node_temperatures_k["Bench"] == pytest.approx(
        short.node_temperatures_k["Bench"], abs=0.1
    )
    assert long.radiator_temperatures_k["Panel"] == pytest.approx(
        short.radiator_temperatures_k["Panel"], abs=0.1
    )


def test_a_very_large_conductance_stays_stable() -> None:
    model = _coupled(conductance=1e6)
    model.advance(0.0, [DARK], {})
    for step in range(1, 50):
        model.advance(600.0 * step, [DARK], {})
        panel = model.radiator_temperatures_k["Panel"]
        bench = model.node_temperatures_k["Bench"]
        assert 150.0 < panel < 290.0
        assert bench == pytest.approx(panel, abs=0.01)


def test_a_group_that_cannot_radiate_keeps_its_configured_temperatures() -> None:
    model = ThermalNetworkModel(
        [],
        ThermalNetwork(
            nodes=[
                ThermalNode(name="A", thermal_mass_j_per_k=1.0, temperature_k=310.0),
                ThermalNode(name="B", thermal_mass_j_per_k=1.0, temperature_k=290.0),
            ],
            couplings=[ThermalCoupling(between=("A", "B"), conductance_w_per_k=1.0)],
            start_at_equilibrium=True,
        ),
    )
    model.advance(0.0, [], {})
    assert model.node_temperatures_k == {"A": 310.0, "B": 290.0}


def test_names_must_resolve() -> None:
    node = ThermalNode(name="Bench", thermal_mass_j_per_k=1.0)
    with pytest.raises(ValueError, match="two different"):
        ThermalCoupling(between=("Bench", "Bench"), conductance_w_per_k=1.0)
    with pytest.raises(ValueError, match="subsystem"):
        ThermalNode(name="X", thermal_mass_j_per_k=1.0, subsystem_power_fraction=0.1)
    with pytest.raises(ValueError, match="neither"):
        SpacecraftBus(
            thermal=ThermalNetwork(
                nodes=[node],
                couplings=[
                    ThermalCoupling(
                        between=("Bench", "Default Radiator"), conductance_w_per_k=1.0
                    )
                ],
            )
        )
    with pytest.raises(ValueError, match="unique"):
        SpacecraftBus(
            radiators=RadiatorConfiguration(radiators=[_radiator(name="Bench")]),
            thermal=ThermalNetwork(nodes=[node]),
        )


@pytest.mark.parametrize("kind", ["dispatch", "plan", "rolling"])
def test_a_run_reports_nodes_heated_by_one_instrument(kind: str) -> None:
    config = make_config(3)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.power_draw = PowerDraw(nominal_power=400.0)
    config.payload.instruments = [
        Instrument(
            name="camera",
            modes=[
                InstrumentMode(name="IDLE", power_w=30.0),
                InstrumentMode(name="SCIENCE", power_w=90.0),
            ],
        ),
        Instrument(
            name="spectrograph",
            modes=[
                InstrumentMode(name="IDLE", power_w=7.0),
                InstrumentMode(name="SCIENCE", power_w=90.0),
            ],
        ),
    ]
    config.spacecraft_bus.radiators = RadiatorConfiguration(
        radiators=[_radiator(absorptivity=0.0)]
    )
    config.spacecraft_bus.thermal = ThermalNetwork(
        nodes=[
            ThermalNode(
                name="Detector",
                thermal_mass_j_per_k=2_000.0,
                temperature_k=250.0,
                subsystem="camera",
                subsystem_power_fraction=1.0,
            )
        ],
        couplings=[
            ThermalCoupling(between=("Detector", "Panel"), conductance_w_per_k=3.0)
        ],
    )
    end = BEGIN + timedelta(hours=3)
    run: DITL | QueueDITL
    if kind == "dispatch":
        run = QueueDITL(config=config, begin=BEGIN, end=end)
    elif kind == "rolling":
        run = RollingHorizonDITL(
            config,
            [],
            begin=BEGIN,
            end=end,
            scheduler=PriorityScheduler,
            include_passes=False,
            horizon=timedelta(hours=1),
            replan_interval=timedelta(minutes=30),
        )
    else:
        plan = PriorityScheduler(
            config, [], BEGIN, end, include_passes=False
        ).schedule()
        run = DITL(config=config, plan=plan, begin=BEGIN, end=end)
    assert run.calc()

    samples = run.telemetry.housekeeping
    assert {sample.acs_mode for sample in samples} == {ACSMode.IDLE}
    reference = ThermalNetworkModel(
        config.spacecraft_bus.radiators.radiators,
        config.spacecraft_bus.thermal,
        power_sources=["camera"],
    )
    for index, sample in enumerate(samples):
        reference.advance(60.0 * index, [DARK], {"camera": 30.0})
        assert sample.thermal_node_temperatures_k == pytest.approx(
            reference.node_temperatures_k
        )
        assert sample.radiator_temperatures_k == pytest.approx(
            reference.radiator_temperatures_k
        )
    # The detector is heated and the panel is not, so heat flows to the panel.
    last = samples[len(samples) - 1]
    assert (
        last.thermal_node_temperatures_k["Detector"]
        > last.radiator_temperatures_k["Panel"]
    )
