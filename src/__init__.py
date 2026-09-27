"""
ERTInv - synthetic ERT modelling and inversion with pyGIMLi.
============================================================================

A small, notebook-driven toolkit for experimenting with 2D Electrical
Resistivity Tomography (ERT) inversion. You describe a synthetic geological
model, a survey and an inversion method in a :class:`Config`, then let the
pipeline simulate data, invert it and assess the result.

Quick start
-----------
>>> from ertinv import default_config, run
>>> cfg = default_config()
>>> cfg.inversion.method = "occam"      # occam | marquardt | blocky | robust
>>> results = run(cfg)
>>> print(results.rms_table())

Modules
-------
config      configuration dataclasses (the "control panel")
model       build the synthetic geology and mesh
survey      electrode layouts / arrays
forward     simulate noisy synthetic data
inversion   the four inversion methods
metrics     residuals, model error, coverage, L-curve, DOI
plotting    figures
pipeline    high-level ``run`` / ``compare_methods`` / ``compare_arrays``
"""

from .config import (
    Config,
    Forward,
    Inversion,
    Layer,
    METHODS,
    Survey,
    Target,
    default_config,
)
from .inversion import (
    DEFAULT_LAMBDA,
    METHOD_DESCRIPTION,
    METHOD_PRESETS,
    InversionResult,
    run_inversion,
)
from .pipeline import RunResults, compare_arrays, compare_methods, run

__version__ = "0.2.0"

__all__ = [
    "Config", "Layer", "Target", "Survey", "Forward", "Inversion",
    "METHODS", "default_config",
    "run_inversion", "InversionResult",
    "METHOD_PRESETS", "METHOD_DESCRIPTION", "DEFAULT_LAMBDA",
    "run", "compare_methods", "compare_arrays", "RunResults",
    "__version__",
]
