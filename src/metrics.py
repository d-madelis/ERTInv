"""
Quality assessment: residuals, model appraisal, L-curve and DOI.
============================================================================

These functions turn raw inversion output into the numbers that tell you how
much to trust it - both in *data space* (does the model reproduce the
measurements?) and in *model space* (since the true model is known here, how
close did we actually get, and where can we believe the image?).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygimli as pg
from pygimli.physics import ert

from .config import Config, Inversion
from .inversion import run_inversion
from .model import true_model_on


# ---------------------------------------------------------------------------
# Data-space residuals
# ---------------------------------------------------------------------------
@dataclass
class ResidualStats:
    residuals: np.ndarray            #: observed - predicted apparent resistivity
    normalized: np.ndarray           #: residual / data error (should be ~N(0,1))
    mean_abs: float
    std: float
    mean_abs_normalized: float
    outlier_fraction: float          #: fraction with |normalized| > 3


def residual_stats(data, response) -> ResidualStats:
    """Compute raw and error-normalised residuals for one inversion."""
    observed = np.asarray(data["rhoa"])
    predicted = np.asarray(response)
    residuals = observed - predicted

    errors = np.asarray(data["err"]) * observed  # absolute error per datum
    normalized = residuals / errors

    return ResidualStats(
        residuals=residuals,
        normalized=normalized,
        mean_abs=float(np.mean(np.abs(residuals))),
        std=float(np.std(residuals)),
        mean_abs_normalized=float(np.mean(np.abs(normalized))),
        outlier_fraction=float(np.mean(np.abs(normalized) > 3)),
    )


# ---------------------------------------------------------------------------
# Model-space error (only possible because this is synthetic)
# ---------------------------------------------------------------------------
@dataclass
class ModelError:
    true_on_para: np.ndarray         #: true resistivity on the inversion mesh
    log_rmse: float                  #: RMS of log10(rec) - log10(true)
    correlation: float               #: Pearson r of log resistivities
    rel_error_median: float          #: median |rec-true|/true


def model_error(result, forward_mesh, rhomap) -> ModelError:
    """Compare a recovered model with the (known) true model, cell by cell.

    The true model is interpolated from the forward mesh onto the inversion
    mesh so both live on the same support.
    """
    para = result.para_domain
    true_on_para = np.asarray(true_model_on(forward_mesh, rhomap, para.cellCenters()))
    rec = np.asarray(result.model)

    # Guard against non-positive interpolation artefacts before taking logs.
    good = (true_on_para > 0) & (rec > 0)
    lt = np.log10(true_on_para[good])
    lr = np.log10(rec[good])

    log_rmse = float(np.sqrt(np.mean((lr - lt) ** 2)))
    corr = float(np.corrcoef(lr, lt)[0, 1]) if good.sum() > 2 else float("nan")
    rel = float(np.median(np.abs(rec[good] - true_on_para[good]) / true_on_para[good]))

    return ModelError(
        true_on_para=true_on_para,
        log_rmse=log_rmse,
        correlation=corr,
        rel_error_median=rel,
    )


# ---------------------------------------------------------------------------
# Coverage (cumulative sensitivity)
# ---------------------------------------------------------------------------
def coverage(result) -> np.ndarray:
    """Return the logarithmic cumulative-sensitivity coverage per cell.

    Cells with low coverage are poorly constrained by the data and their
    resistivity should be interpreted with caution.
    """
    return np.asarray(result.manager.coverage())


# ---------------------------------------------------------------------------
# L-curve: trade-off between data misfit and model roughness
# ---------------------------------------------------------------------------
@dataclass
class LCurve:
    lambdas: np.ndarray
    phi_data: np.ndarray             #: data misfit  ||W(d - f(m))||^2
    phi_model: np.ndarray            #: model roughness ||L m||^2
    chi2: np.ndarray


def l_curve(data, base_inv: Inversion, lambdas: List[float], verbose: bool = False) -> LCurve:
    """Run smooth (Occam) inversions over a range of lambdas.

    The corner of the resulting L-curve (log phi_model vs log phi_data)
    marks a good compromise between fitting the data and keeping the model
    simple.
    """
    phid, phim, chi2, used = [], [], [], []
    for lam in lambdas:
        cfg = Inversion(
            method="occam",
            lam=lam,
            z_weight=base_inv.z_weight,
            max_iter=base_inv.max_iter,
            quality=base_inv.quality,
        )
        res = run_inversion(data, cfg, verbose=verbose)
        if not res.success:
            continue
        inv = res.manager.inv
        phid.append(float(inv.phiData()))
        phim.append(float(inv.phiModel()))
        chi2.append(res.chi2)
        used.append(lam)

    return LCurve(
        lambdas=np.array(used),
        phi_data=np.array(phid),
        phi_model=np.array(phim),
        chi2=np.array(chi2),
    )


# ---------------------------------------------------------------------------
# Depth of investigation (Oldenburg & Li style)
# ---------------------------------------------------------------------------
@dataclass
class DOIResult:
    doi_index: np.ndarray            #: per-cell DOI (0 = well resolved, 1 = not)
    para_domain: object


def depth_of_investigation(
    data, base_inv: Inversion, reference_factors: Tuple[float, float] = (0.1, 10.0),
    verbose: bool = False,
) -> DOIResult:
    """Estimate a depth-of-investigation index from two reference models.

    Two inversions are run with very different homogeneous starting/reference
    models. Where the data constrain the resistivity, both converge to the
    same answer (low index); at depth the result just follows the reference
    (index -> 1).
    """
    median_rhoa = float(np.median(np.asarray(data["rhoa"])))
    models = []
    para = None
    for fac in reference_factors:
        ref = fac * median_rhoa
        cfg = Inversion(
            method="occam",
            lam=base_inv.lam,
            z_weight=base_inv.z_weight,
            max_iter=base_inv.max_iter,
            quality=base_inv.quality,
        )
        mgr = ert.ERTManager(data)
        mgr.invert(
            lam=cfg.lam if cfg.lam is not None else 20.0,
            zWeight=cfg.z_weight,
            maxIter=cfg.max_iter,
            startModel=ref,
            verbose=verbose,
        )
        models.append(np.log10(np.asarray(mgr.model)))
        para = mgr.paraDomain

    m1, m2 = models
    r1, r2 = np.log10(reference_factors[0] * median_rhoa), np.log10(
        reference_factors[1] * median_rhoa
    )
    doi = np.abs(m1 - m2) / abs(r1 - r2)
    return DOIResult(doi_index=np.clip(doi, 0, 1), para_domain=para)
