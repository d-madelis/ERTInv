"""
Inversion methods.
============================================================================

Four inversion "flavours", each a genuine pyGIMLi regularisation rather than
just a different lambda:

===============  ==================  ============================================
method           pyGIMLi setting     character
===============  ==================  ============================================
``occam``        ``cType=1``         smooth, first-order roughness (default)
``marquardt``    ``cType=0``         Marquardt-Levenberg local damping
``blocky``       ``blockyModel``     L1 model norm -> sharp boundaries
``robust``       ``robustData``      L1 data norm -> resistant to outliers
===============  ==================  ============================================

Notes
-----
* ``occam`` is the standard smooth 2D inversion and the best general choice.
* ``marquardt`` (pure zeroth-order damping) pulls the model toward a
  homogeneous reference. It is the classic scheme for few-parameter / blocky
  problems and tends to *under-fit* dense 2D pixel models - keep this in mind
  when comparing RMS values.
* ``blocky`` and ``robust`` both use iteratively-reweighted least squares
  (IRLS); they need a few extra iterations to settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from pygimli.physics import ert

from .config import Inversion

#: Regularisation preset for each method (passed straight to ``invert``).
METHOD_PRESETS: Dict[str, dict] = {
    "occam": dict(cType=1, blockyModel=False, robustData=False),
    "marquardt": dict(cType=0, blockyModel=False, robustData=False),
    "blocky": dict(cType=1, blockyModel=True, robustData=False),
    "robust": dict(cType=1, blockyModel=False, robustData=True),
}

#: One-line description of each method (for reports and plot titles).
METHOD_DESCRIPTION: Dict[str, str] = {
    "occam": "Smooth (first-order) Occam-type inversion",
    "marquardt": "Marquardt-Levenberg local damping (zeroth-order)",
    "blocky": "Blocky L1 model inversion (sharp boundaries)",
    "robust": "Robust L1 data inversion (outlier resistant)",
}

#: A sensible default lambda per method when the user leaves ``lam=None``.
DEFAULT_LAMBDA: Dict[str, float] = {
    "occam": 20.0,
    "marquardt": 20.0,
    "blocky": 20.0,
    "robust": 20.0,
}


@dataclass
class InversionResult:
    """Everything produced by one inversion run."""

    method: str
    lam: float
    manager: object = None          #: the ERTManager (for plotting / coverage)
    model: np.ndarray = None        #: recovered resistivity on the inversion mesh
    response: np.ndarray = None     #: predicted apparent resistivities
    para_domain: object = None      #: the inversion (parameter) mesh
    chi2: float = float("nan")      #: data chi-squared
    rms: float = float("nan")       #: sqrt(chi2)
    n_iterations: int = 0
    success: bool = True
    error: str = ""
    extra: dict = field(default_factory=dict)


def run_inversion(data, inv_cfg: Inversion, verbose: bool = False) -> InversionResult:
    """Run a single inversion described by ``inv_cfg`` on ``data``.

    The measurement ``data`` must carry an error model (``data['err']``);
    datasets from :func:`ertinv.forward.simulate_datasets` already do.

    A failed inversion is caught and reported in the returned object rather
    than raised, so a batch run never dies on one bad case.
    """
    method = inv_cfg.method.lower()
    if method not in METHOD_PRESETS:
        raise ValueError(
            f"Unknown inversion method '{method}'. "
            f"Choose from {list(METHOD_PRESETS)}."
        )

    lam = inv_cfg.lam if inv_cfg.lam is not None else DEFAULT_LAMBDA[method]
    preset = METHOD_PRESETS[method]

    # Optional mesh controls (only pass them when explicitly set).
    mesh_kwargs = {}
    if inv_cfg.para_depth > 0:
        mesh_kwargs["paraDepth"] = inv_cfg.para_depth
    if inv_cfg.para_max_cell_size > 0:
        mesh_kwargs["paraMaxCellSize"] = inv_cfg.para_max_cell_size
    if inv_cfg.quality > 0:
        mesh_kwargs["quality"] = inv_cfg.quality

    result = InversionResult(method=method, lam=lam)
    try:
        mgr = ert.ERTManager(data)
        model = mgr.invert(
            lam=lam,
            zWeight=inv_cfg.z_weight,
            maxIter=inv_cfg.max_iter,
            lambdaOpt=(inv_cfg.lambda_opt and method == "occam"),
            verbose=verbose,
            **preset,
            **mesh_kwargs,
        )
        result.manager = mgr
        result.model = np.asarray(model)
        result.response = np.asarray(mgr.inv.response)
        result.para_domain = mgr.paraDomain
        result.chi2 = float(mgr.inv.chi2())
        result.rms = float(np.sqrt(result.chi2))
        try:
            result.n_iterations = int(mgr.inv.inv.iter())
        except Exception:
            result.n_iterations = 0
        result.lam = float(mgr.inv.inv.getLambda()) if inv_cfg.lambda_opt else lam

        # Sanity check: a model that never moved off its homogeneous start
        # means the inversion did not actually update (e.g. a broken
        # sensitivity kernel or a rejected first step). Flag it loudly so the
        # result is not mistaken for a real image.
        rel_spread = float(np.std(np.log10(result.model))) if result.model.size else 0.0
        result.extra["model_log_std"] = rel_spread
        if rel_spread < 1e-3:
            result.extra["stuck"] = True
            print(
                "   WARNING: recovered model is (near-)homogeneous - the "
                "inversion did not update. This usually means the pyGIMLi "
                "core could not build a sensitivity matrix. Check your "
                "pyGIMLi install (a conda-forge build is recommended)."
            )
    except Exception as exc:  # pragma: no cover - safety net
        result.success = False
        result.error = f"{type(exc).__name__}: {exc}"

    return result
