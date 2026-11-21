Installation
============

Requirements
------------

COASTSim requires Python >= 3.10. The full dependency list is in
``pyproject.toml``. The key dependencies are:

* ``rust-ephem`` - Efficient ephemeris calculations
* ``numpy`` - Numerical computations
* ``matplotlib`` - Visualization
* ``pydantic`` - Configuration validation
* ``shapely`` / ``pyproj`` - Geometric operations

From Source
-----------

To install COASTSim from source, clone the repository and install it with
``pip``:

.. code-block:: bash

   git clone https://github.com/CosmicFrontierLabs/coast-sim.git
   cd coast-sim
   pip install -e .

Optional: CP-SAT scheduler and long-range allocation
----------------------------------------------------

Two features need Google's OR-Tools package.
:class:`~conops.schedulers.CpSatScheduler` uses the OR-Tools CP-SAT solver, and
the default ``milp`` solver of :class:`~conops.schedulers.LongRangeAllocator` uses
the HiGHS solver that comes with OR-Tools. OR-Tools is an optional dependency,
installed with the ``cpsat`` extra:

.. code-block:: bash

   pip install -e ".[cpsat]"

The ``dev`` and ``test`` extras also include OR-Tools.

Optional: atmosphere and geomagnetic field models
-------------------------------------------------

Stored-momentum tracking (see :doc:`momentum`) needs an atmosphere model and a
geomagnetic field model. It has a built-in atmosphere table and a built-in
tilted-dipole geomagnetic field, so it works without extra packages. Two other
models are optional: the NRLMSIS 2.1 atmosphere, through ``pymsis``, and the IGRF
geomagnetic field, through ``ppigrf``. The ``momentum`` extra installs both
packages:

.. code-block:: bash

   pip install -e ".[momentum]"

The ``test`` extra also includes both packages.

Development Installation
------------------------

For development, install with the ``dev`` and ``test`` extras and set up the
pre-commit hooks:

.. code-block:: bash

   pip install -e ".[dev,test]"
   pre-commit install

The ``dev`` extra installs these tools:

* ruff, for linting and code formatting
* mypy, for static type checking, with the type stubs it needs
* pre-commit, which runs Git hooks for code quality

The ``test`` extra installs pytest, the testing framework, with ``pytest-cov``
and ``pytest-xdist``.

Verifying Installation
----------------------

To verify that COASTSim is installed correctly, import it and print its
version:

.. code-block:: python

   from importlib.metadata import version

   import conops
   print(version("coast-sim"))

You can also run the test suite, which needs the ``test`` extra:

.. code-block:: bash

   pytest tests/
