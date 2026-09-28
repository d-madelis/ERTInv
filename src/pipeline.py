"""
High-level pipeline: glue everything into a few one-line calls.
============================================================================

Typical use from the notebook::

    from ertinv import default_config, run
    cfg = default_config()
    cfg.inversion.method = "occam"
    results = run(cfg)

or to compare inversion flavours on one dataset::

    from ertinv import compare_methods
    compare_methods(cfg, methods=["occam", "marquardt", "blocky", "robust"])
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import metrics, plotting
from .config import Config, Inversion
from .forward import simulate_datasets
from .inversion import METHOD_DESCRIPTION, run_inversion
from .model import build_true_model
from .survey import array_label, create_scheme


def _quiet_pygimli() -> None:
    """Silence pyGIMLi's INFO log lines, C++ verbosity and cosmetic warnings."""
    try:
        import pygimli as pg
        pg.setVerbose(False)
        pg.rc["hold"] = True   # never auto-plt.show() (keeps figures out of the notebook)
    except Exception:
        pass
    for name in ("pyGIMLi", "pygimli", "Core"):
        logging.getLogger(name).setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    warnings.filterwarnings("ignore", message=".*Adding colorbar.*")


@dataclass
class RunResults:
    """Container for a full single-method run over all noise levels."""

    config: Config
    forward_mesh: object
    rhomap: list
    datasets: list
    per_noise: Dict[float, dict] = field(default_factory=dict)
    rms_summary: Dict[str, Dict[float, float]] = field(default_factory=dict)

    def rms_table(self) -> str:
        """Pretty-print RMS (sqrt chi2) for every method and noise level."""
        noises = sorted({n for s in self.rms_summary.values() for n in s})
        head = "method".ljust(12) + "".join(f"{n*100:>8.0f}%" for n in noises)
        rows = [head, "-" * len(head)]
        for method, series in self.rms_summary.items():
            rows.append(method.ljust(12)
                        + "".join(f"{series.get(n, float('nan')):>9.2f}" for n in noises))
        return "\n".join(rows)


def run(cfg: Config, report_pdf: Optional[str] = None, save_png: bool = False,
        quiet: bool = True, verbose: bool = False) -> RunResults:
    """Run the full workflow for ``cfg.inversion.method`` over all noise levels.

    Parameters
    ----------
    report_pdf :
        Path of a single PDF collecting every figure, in order. If ``None``
        (default) it is written to ``<outdir>/ERTInv_report_<method>.pdf``.
    save_png :
        If ``True``, also save each figure as an individual PNG under
        ``<outdir>/`` (off by default - the PDF is usually all you need).
    quiet :
        If ``True`` (default) suppress pyGIMLi's INFO log spam and print only
        a short progress line per noise level plus a final summary.
    """
    if quiet:
        _quiet_pygimli()

    method = cfg.inversion.method
    os.makedirs(cfg.outdir, exist_ok=True)
    data_dir = cfg.datadir
    fig_root = os.path.join(cfg.outdir, f"inversion_{method}") if save_png else None
    if fig_root:
        os.makedirs(fig_root, exist_ok=True)

    if report_pdf is None:
        report_pdf = os.path.join(cfg.outdir, f"ERTInv_report_{method}.pdf")
    _dir = os.path.dirname(report_pdf)
    if _dir:
        os.makedirs(_dir, exist_ok=True)

    # On Windows a PDF still open in a viewer is locked and cannot be
    # overwritten. Detect that and fall back to a timestamped file name.
    try:
        open(report_pdf, "ab").close()
    except PermissionError:
        import time
        base, ext = os.path.splitext(report_pdf)
        report_pdf = f"{base}_{time.strftime('%H%M%S')}{ext}"
        print(f"  (the previous report is still open; "
              f"writing {os.path.basename(report_pdf)} instead)")
    pdf = PdfPages(report_pdf)

    def _page(fig, png_path: Optional[str] = None):
        """Add a figure as one PDF page (and optionally a PNG), then close it."""
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        if save_png and png_path:
            plotting._ensure_dir(png_path)
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _png(*parts):
        return os.path.join(fig_root, *parts) if fig_root else None

    print(f"ERTInv | method = {method} ({METHOD_DESCRIPTION.get(method, method)})")
    print(f"        noise levels = {[f'{n*100:.0f}%' for n in cfg.forward.noise_levels]}\n")

    # 1. survey + true model + synthetic data
    scheme = create_scheme(cfg.survey)
    forward_mesh, rhomap, _geom, _names = build_true_model(cfg, scheme)
    datasets = simulate_datasets(forward_mesh, scheme, rhomap, cfg,
                                 save_dir=data_dir, verbose=verbose)

    results = RunResults(config=cfg, forward_mesh=forward_mesh,
                         rhomap=rhomap, datasets=datasets)
    results.rms_summary[method] = {}

    # one inversion per noise level (compute; figures are assembled afterwards)
    for ds in datasets:
        res = run_inversion(ds.data, cfg.inversion, verbose=verbose)
        entry: dict = {"inversion": res}

        if not res.success:
            print(f"  {ds.noise_level*100:>3.0f}% noise : inversion FAILED ({res.error})")
            results.per_noise[ds.noise_level] = entry
            continue

        results.rms_summary[method][ds.noise_level] = res.rms
        entry["residuals"] = metrics.residual_stats(ds.data, res.response)

        merr_txt = ""
        if cfg.do_model_error:
            merr = metrics.model_error(res, forward_mesh, rhomap)
            entry["model_error"] = merr
            merr_txt = f", model corr = {merr.correlation:.3f}"

        print(f"  {ds.noise_level*100:>3.0f}% noise : chi2 = {res.chi2:.2f}"
              f" (RMS {res.rms:.2f}){merr_txt}")
        results.per_noise[ds.noise_level] = entry

    depth = getattr(cfg, "plot_depth", None)

    # Page 1: true model on its own
    _page(plotting.plot_true_model_page(forward_mesh, rhomap, depth=depth),
          _png("true_model.png"))

    # Page 2: all pseudosections
    _page(plotting.plot_pseudosections_page(datasets),
          _png("pseudosections.png"))

    # Page 3: all recovered models
    _page(plotting.plot_results_page(forward_mesh, rhomap, datasets,
                                     results.per_noise, method, depth=depth),
          _png(f"{method}_results.png"))

    # Page 4: all residual histograms
    _page(plotting.plot_residual_page(datasets, results.per_noise, method),
          _png(f"{method}_residuals.png"))

    pdf.close()
    print(f"\nDone. Report -> {os.path.abspath(report_pdf)}")
    return results


def compare_methods(cfg: Config, methods: List[str], noise_level: Optional[float] = None,
                    verbose: bool = False) -> Dict[str, object]:
    """Run several inversion methods on one noise level and compare them.

    Returns ``{method: InversionResult}`` and writes a side-by-side figure.
    """
    scheme = create_scheme(cfg.survey)
    forward_mesh, rhomap, _geom, _names = build_true_model(cfg, scheme)

    if noise_level is None:
        noise_level = cfg.forward.noise_levels[len(cfg.forward.noise_levels) // 2]

    # simulate just the one dataset we need
    single = Config(**{**cfg.__dict__})
    single.forward = type(cfg.forward)(**{**cfg.forward.__dict__,
                                          "noise_levels": [noise_level]})
    ds = simulate_datasets(forward_mesh, scheme, rhomap, single,
                           save_dir=None, verbose=verbose)[0]

    print(f"Comparing methods on {noise_level*100:.0f}% noise data:\n")
    results: Dict[str, object] = {}
    for method in methods:
        inv = Inversion(**{**cfg.inversion.__dict__, "method": method, "lam": None})
        res = run_inversion(ds.data, inv, verbose=verbose)
        results[method] = res
        status = f"chi2={res.chi2:.2f}" if res.success else f"FAILED ({res.error})"
        print(f"  {method:10s} {status}")

    ok = {m: r for m, r in results.items() if r.success}
    if ok:
        plotting.plot_method_comparison(
            ok, rhomap, outpath=os.path.join(cfg.outdir, "method_comparison.png"))
    return results


def compare_arrays(cfg: Config, arrays: List[str], noise_level: Optional[float] = None,
                   verbose: bool = False) -> Dict[str, object]:
    """Invert the same target with different electrode arrays and compare.

    Returns ``{array_code: InversionResult}``.
    """
    if noise_level is None:
        noise_level = cfg.forward.noise_levels[len(cfg.forward.noise_levels) // 2]

    results: Dict[str, object] = {}
    for code in arrays:
        survey = type(cfg.survey)(**{**cfg.survey.__dict__, "scheme": code})
        scheme = create_scheme(survey)
        forward_mesh, rhomap, _g, _n = build_true_model(cfg, scheme)
        single = Config(**{**cfg.__dict__})
        single.forward = type(cfg.forward)(**{**cfg.forward.__dict__,
                                              "noise_levels": [noise_level]})
        ds = simulate_datasets(forward_mesh, scheme, rhomap, single,
                               save_dir=None, verbose=verbose)[0]
        res = run_inversion(ds.data, cfg.inversion, verbose=verbose)
        results[code] = res
        status = f"chi2={res.chi2:.2f}, n={ds.data.size()}" if res.success else "FAILED"
        print(f"  {array_label(code):16s} {status}")
    return results
