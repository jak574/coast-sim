# COASTSim Documentation

This directory contains the Sphinx documentation for COASTSim.

## Building the Documentation

### Install Dependencies

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or install from the requirements file:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.

### Other Build Formats

Sphinx can build other output formats as well:

```bash
make latexpdf  # Build PDF documentation
make epub      # Build EPUB documentation
make man       # Build man pages
make help      # See all available formats
```

### Clean Build

To clean the build directory:

```bash
make clean
```

### Auto-rebuild During Development

While you are writing documentation, `sphinx-autobuild` can rebuild it and reload the browser each time you save:

```bash
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

Then open http://127.0.0.1:8000 in your browser.

## Documentation Structure

- `conf.py` - Sphinx configuration file
- `requirements.txt` - Documentation build requirements
- `Makefile`, `make.bat` - Build scripts
- `index.rst` - Main documentation index
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `configuration.rst` - Mission configuration
- `telemetry.rst` - Telemetry system
- `examples.rst` - Examples and tutorials
- `visualization.rst` - Plotting utilities and run statistics
- `sky_pointing_visualization.rst` - Sky pointing plot
- `spacecraft_3d_visualization.rst` - 3-D spacecraft configuration visualizer
- `communications.rst` - Communications system configuration
- `data_management.rst` - Data management
- `fault_management.rst` - Fault management
- `predictive_idle_safety.md` - Predictive idle safety
- `attitude_execution.rst` - Physical attitude execution
- `attitude_exports.rst` - Executed attitude exports
- `momentum.rst` - Stored momentum
- `radiator_shadowing.rst` - Radiator thermal modelling and panel shadowing
- `target_of_opportunity.rst` - Targets of Opportunity
- `planning.rst` - Offline planning
- `plan_serialization.rst` - Plan serialisation
- `contributing.rst` - Contributing guidelines
- `_static/` - Images used by the pages
- `api/` - API reference documentation
  - `modules.rst` - API reference index
  - `conops.*.rst` - Individual module documentation

## Documentation Style

- Write in reStructuredText (.rst) format. Markdown (.md) pages are also built, through `myst-parser`
- Use Google or NumPy style docstrings in Python code
- Include code examples where helpful
- Keep documentation up to date with code changes
- Use appropriate Sphinx directives for notes, warnings, etc.

## Viewing Online

Once published, the documentation will be available at the project's documentation site.
