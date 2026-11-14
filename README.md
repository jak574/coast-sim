<p align="center">
  <img src="https://raw.githubusercontent.com/CosmicFrontierLabs/coast-sim/main/docs/_static/coast-sim-logo.png" alt="COASTSim Logo" width="400">
</p>

# COASTSim: ConOps Astronomical Space Telescope Simulator

A Python module for simulating Concept of Operations (ConOps) for space telescopes and astronomical spacecraft missions.

## Overview

COASTSim is a simulation framework that models and analyzes the operational behavior of space-based astronomical observatories. Mission planners and engineers can use it to simulate Day-In-The-Life (DITL) scenarios, evaluate spacecraft performance, optimize observation schedules, and validate operational constraints before launch.

The package is organized into submodules: `common` (utilities), `config` (configuration), `ditl` (simulation), `schedulers` (planning), `targets` (observations), and `simulation` (core components).

## Key Features

- The spacecraft bus simulation models power systems, attitude control, and thermal management.
- Orbit propagation computes a TLE-based ephemeris and tracks the orbit.
- Observation planning provides target queue management and scheduling algorithms.
- Instrument modeling supports multi-instrument configurations with power and pointing requirements.
- Constraint management handles Sun, Moon, Earth limb, and custom geometric constraints.
- Power budget analysis covers solar panel modeling, battery management, and emergency charging scenarios.
- Ground station pass modeling calculates communication windows and plans data downlink.
- The attitude control system model simulates slews, pointing accuracy, and settle time.
- South Atlantic Anomaly (SAA) avoidance handles the radiation belt constraint.
- DITL generation simulates a day-in-the-life timeline.
- Plan export writes versioned JSON plans, with optional TLE provenance and executed attitude and orbit-state sidecars.

## Installation

### From Source

```bash
git clone https://github.com/CosmicFrontierLabs/coast-sim.git
cd coast-sim
pip install -e .
```

### Requirements

COASTSim requires Python >= 3.10. The full dependency list is in `pyproject.toml`. The key dependencies are:

- `astropy` for astronomical calculations and coordinate systems
- `numpy` for numerical computations
- `matplotlib` for visualization
- `rust-ephem` for efficient ephemeris calculations
- `pydantic` for configuration validation
- `shapely` for geometric operations

## Quick Start

### Basic DITL Simulation

The example below loads a configuration, computes an orbit ephemeris from a TLE file, runs a one-day simulation, exports the executed plan and plots the results:

```python
from datetime import datetime, timedelta
from conops import Config, QueueDITL
from rust_ephem import TLEEphemeris

# Load configuration
config = Config.from_json_file("example_config.json")

# Set simulation period
begin = datetime(2025, 11, 1)
end = begin + timedelta(days=1)

# Compute orbit ephemeris
ephemeris = TLEEphemeris(tle="example.tle", begin=begin, end=end)

# Run DITL simulation
ditl = QueueDITL(config=config)
# Field-of-regard telemetry is optional and disabled by default.
# Enable it only when needed:
# ditl = QueueDITL(config=config, calculate_field_of_regard=True)
ditl.ephem = ephemeris
ditl.begin = begin
ditl.end = end
ditl.calc()

# Export the executed plan. When available, attitude and orbit-state sidecars
# are written beside the plan JSON and linked from it.
plan_path = ditl.plan.save("observation_plan.json")

# Analyze results
ditl.plot()
ditl.print_statistics()
```

### Configuration-Based Approach

Create a JSON configuration file defining your spacecraft parameters:

```json
{
    "name": "My Space Telescope",
    "spacecraft_bus": {
        "power_draw": {
            "nominal_power": 50.0,
            "peak_power": 300.0
        },
        "attitude_control": {
            "slew_acceleration": 0.01,
            "max_slew_rate": 1.0
        }
    },
    "solar_panel": {
        "panels": [...]
    },
    "instruments": {
        "instruments": [...]
    }
}
```

## Examples

The `examples/` directory has worked examples as Jupyter notebooks:

- `Example_Spacecraft_DITL.ipynb` runs a complete spacecraft DITL simulation with a custom spacecraft configuration, including power modeling, attitude control, and observation scheduling.
- `Example_DITL_from_JSON.ipynb` shows a simplified workflow that uses JSON configuration files for quick simulations.
- `Dynamic_Merit_Example.ipynb` shows how the queue scheduler ranks targets with urgency, cadence and program-balance merit terms, and how Targets of Opportunity interrupt.

To run the examples:

```bash
cd examples
jupyter notebook
```

## Visualization

COASTSim includes a set of plotting utilities to visualize DITL simulations and telemetry. Each visualization function accepts an optional `config` parameter. If it is omitted, the plot uses `ditl.config.visualization` if present, and otherwise the defaults defined by `VisualizationConfig`.

The key plotting utilities are in the `conops.visualization` module:

- `plot_ditl_telemetry()` shows RA, Dec, ACS mode, battery, power, and ObsIDs in a multi-panel timeline.
- `plot_data_management_telemetry()` shows recorder volume and fill fractions, cumulative data generated and downlinked, and alert timelines.
- `plot_acs_mode_distribution()` draws a pie chart of the time spent in each ACS mode. It supports custom mode colors.
- `plot_ditl_timeline()` draws a full timeline view with orbit numbers, observations, slews, SAA, and eclipses.
- `plot_sky_pointing()` draws an interactive mollweide sky projection showing current pointing, patterns, and constraints.

### How to customize fonts and colors

Visualization settings are held in the `VisualizationConfig` Pydantic model (`conops.config.visualization`). Important fields include:

- `font_family` (str): font used for titles, labels, and legends (default: `Helvetica`)
- `title_font_size`, `label_font_size`, `legend_font_size`, `tick_font_size` (int)
- `mode_colors` (dict[str, str]): color mapping for ACS modes used in plots such as `plot_acs_mode_distribution`

### Example — customizing fonts and colors

```python
from conops import Config, QueueDITL
from conops.visualization import plot_ditl_telemetry, plot_acs_mode_distribution
from datetime import datetime, timedelta
from rust_ephem import TLEEphemeris

cfg = Config.from_json_file("examples/example_config.json")
cfg.visualization.font_family = "Helvetica"
cfg.visualization.title_font_size = 14
cfg.visualization.mode_colors["SAA"] = "#800080"  # purple

begin = datetime.utcnow()
end = begin + timedelta(days=1)
ephem = TLEEphemeris(tle="examples/example.tle", begin=begin, end=end)
ditl = QueueDITL(config=cfg)
ditl.ephem = ephem
ditl.calc()

# Render using the resolved visualization config
fig, axes = plot_ditl_telemetry(ditl)
fig2, ax2 = plot_acs_mode_distribution(ditl, config=cfg.visualization)
```

Setting `cfg.visualization` keeps fonts and colors consistent across every plot. The requested font must be installed on your system. If the requested family isn't available, Matplotlib may fall back to another font (e.g., DejaVu Sans or Arial). To guarantee a specific font, use a `FontProperties` object with the font file path.

## Core Components

### Configuration (`conops.config`)

This module holds the Pydantic-based configuration for spacecraft parameters, instruments, and operational constraints.

### DITL Simulation (`conops.ditl`)

This module holds the Day-In-The-Life simulation classes, including DITL, DITLMixin, and QueueDITL, which simulate the timeline.

Instantaneous field-of-regard telemetry (`for_solid_angle_sr`) is optional. It is
disabled by default because computing it slows the simulation. To enable it, pass
`calculate_field_of_regard=True` when creating `DITL` or `QueueDITL`.

### Scheduling (`conops.schedulers`)

This module holds target observation queue management and the schedulers (PriorityScheduler, SequentialScheduler, QueueScheduler). The
former names `DumbScheduler` and `DumbQueueScheduler` remain as aliases of
`SequentialScheduler` and `QueueScheduler`.

### Targets (`conops.targets`)

This module holds the target management classes, including Pointing, Queue, TargetList, Plan, and PlanEntry. They are used for
observation planning and portable plan export.

### Benchmark (`conops.benchmark`)

The benchmark runs one scenario through each scheduling mode: queue dispatch, a plan built up front,
rolling-horizon replanning, or whatever the configuration selects. It measures every run
the same way, from telemetry. The measurements are:

- science and merit-weighted science
- slewing and idle time
- Target of Opportunity (ToO) response and deadlines
- program shares
- cadence error
- planning time
- plan/execution mismatches

Six standard scenarios come with the module, and
`scripts/benchmark_schedulers.py` compares the modes on them. See the planning
documentation for details.

### Simulation (`conops.simulation`)

This module holds the core simulation components, including the ACS (Attitude Control System), slew modeling, SAA handling, emergency charging, and other simulation utilities.

### Common Utilities (`conops.common`)

This module holds shared utilities, enums (ACSMode, ChargeState, ACSCommandType), vector mathematics, and common functions.

## Module Structure

```text
conops/
├── __init__.py              # Package initialization and exports
├── _version.py              # Version information
├── common/                  # Shared utilities and common functions
│   ├── __init__.py
│   ├── common.py
│   ├── enums.py
│   └── vector.py
├── config/                  # Configuration management
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── constraint.py
│   ├── battery.py
│   ├── solar_panel.py
│   ├── instrument.py
│   └── spacecraft_bus.py
├── ditl/                    # Day-In-The-Life simulation
│   ├── __init__.py
│   ├── ditl.py
│   ├── ditl_mixin.py
│   └── queue_ditl.py
├── benchmark/               # Comparing scheduling modes on a scenario
│   ├── __init__.py
│   ├── contenders.py
│   ├── harness.py
│   ├── metrics.py
│   └── scenarios.py
├── schedulers/              # Scheduling algorithms
│   ├── __init__.py
│   ├── context.py
│   ├── priority_scheduler.py
│   ├── sequential_scheduler.py
│   ├── queue_scheduler.py
│   ├── scheduler.py         # DumbScheduler alias
│   └── queue_scheduler.py   # DumbQueueScheduler alias
├── targets/                 # Target management and planning
│   ├── __init__.py
│   ├── pointing.py
│   ├── plan.py
│   ├── plan_entry.py
│   └── target_queue.py
└── simulation/              # Core simulation components
    ├── __init__.py
    ├── acs.py
    ├── acs_command.py
    ├── emergency_charging.py
    ├── passes.py
    ├── roll.py
    ├── saa.py
    ├── slew.py
    └── ephemeris.py
```

## Testing

Run the test suite with pytest:

```bash
pytest tests/
```

The test extra includes `pytest-xdist`, so tests run in parallel. The default run uses up to eight workers
and keeps every regression test enabled. For serial debugging, use
`pytest -n 0 tests/`. To set an explicit worker count, use for example `pytest -n 4 tests/`.

The test suite includes a COAST-owned default plan output regression check. Run
it directly when reviewing scheduler/output changes:

```bash
python scripts/check_default_plan_output.py
```

If a scheduler change intentionally changes the default output, update and review
the checked-in baseline:

```bash
python scripts/check_default_plan_output.py --update
```

To run the tests with coverage:

```bash
pytest --cov=conops tests/
```

To compare the scheduling modes on the standard benchmark scenarios (see
`conops.benchmark`), run one scenario or all of them:

```bash
python scripts/benchmark_schedulers.py --scenario baseline
python scripts/benchmark_schedulers.py --scenario all --time-limit 10
```

## Documentation

The full documentation, including the API reference, is in the `docs/` directory and is built with Sphinx.

### Building Documentation

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

Build the HTML documentation:

```bash
cd docs
make html
```

The built documentation is written to `docs/_build/html/index.html`.

For more information, see `docs/README.md`.

## Development

### Setup Development Environment

```bash
pip install -e ".[dev]"
pre-commit install
```

### Code Quality

This project uses:

- ruff, for linting and code formatting
- mypy, for static type checking
- pytest, the testing framework
- pre-commit, which runs Git hooks for code quality

## Use Cases

COASTSim can be used for the following:

- Mission planning: evaluating operational scenarios before launch.
- Performance analysis: assessing power budgets, observation efficiency, and data return.
- Schedule optimization: testing different scheduling algorithms and target prioritization.
- Constraint validation: verifying that observational constraints are satisfied.
- Trade studies: comparing different spacecraft configurations and operational strategies.
- Operations training: simulating realistic mission scenarios for operations teams.

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code quality checks succeed
5. Submit a pull request

## License

See LICENSE file for details.

## Author

**Jamie A. Kennea**
Email: <jak51@psu.edu>

## Acknowledgments

Developed for space telescope mission planning and operational analysis.

## Project Status

Development Status: Production/Stable

---

For questions, issues, or feature requests, please open an issue on GitHub.
