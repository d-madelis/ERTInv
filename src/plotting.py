"""
Plotting helpers.
============================================================================

Every function creates one figure and (optionally) saves it. They all accept
an existing ``ax`` where it makes sense, so they compose into multi-panel
figures inside the notebook.

Colour conventions
------------------
* Resistivity sections use ``Spectral_r`` (blue = conductive, red = resistive).
* Pseudosections use ``rainbow`` to match the original ERTInv figures.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pygimli as pg
from pygimli.physics import ert
from scipy.stats import norm

RES_CMAP = "Spectral_r"
PSEUDO_CMAP = "rainbow"


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def resistivity_range(rhomap) -> tuple:
    """Return (cMin, cMax) spanning the true resistivities, rounded a little."""
    vals = [r for _, r in rhomap]
    return float(min(vals)), float(max(vals))


def show_pseudosection(data, ax, cmap: str = PSEUDO_CMAP,
                       title: Optional[str] = None):
    """Draw an apparent-resistivity pseudosection into ``ax``.

    A small, version-robust replacement for ``ert.show(data, ax=...)``.
    Some pyGIMLi versions (e.g. 1.5.x) draw the built-in pseudosection into a
    private figure and attach its colour bar to the wrong one, leaving the
    requested axis blank. Here we compute the pseudo-positions ourselves
    (dipole midpoint in x, a fraction of the dipole separation in depth) and
    plot with plain matplotlib, so it renders identically on every version.
    """
    sx = np.array(pg.x(data))
    idx = {k: np.array(data[k], dtype=int) for k in "abmn"}
    rhoa = np.array(data["rhoa"])

    xc = 0.25 * (sx[idx["a"]] + sx[idx["b"]] + sx[idx["m"]] + sx[idx["n"]])
    sep = np.abs(0.5 * (sx[idx["a"]] + sx[idx["b"]])
                 - 0.5 * (sx[idx["m"]] + sx[idx["n"]]))
    zc = -sep / 3.0 - 1.0

    ok = np.isfinite(rhoa) & (rhoa > 0)
    sc = ax.scatter(xc[ok], zc[ok], c=np.log10(rhoa[ok]), cmap=cmap,
                    s=26, edgecolors="none")
    cb = ax.figure.colorbar(sc, ax=ax)
    cb.set_label(r"log$_{10}\,\rho_a$ ($\Omega$m)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Pseudo-depth (m)")
    if title:
        ax.set_title(title, fontweight="bold")
    return sc


# ---------------------------------------------------------------------------
def plot_true_and_pseudosections(mesh, rhomap, datasets, outpath: Optional[str] = None):
    """True model (top-left) plus one pseudosection per noise level."""
    cmin, cmax = resistivity_range(rhomap)
    n = len(datasets) + 1
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    pg.show(mesh, data=rhomap, label=pg.unit("res"), ax=axes[0], showMesh=False,
            cMap=RES_CMAP, cMin=cmin, cMax=cmax)
    axes[0].set_title("True model", fontweight="bold")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Depth (m)")

    for i, ds in enumerate(datasets, 1):
        show_pseudosection(
            ds.data, axes[i],
            title=f"Pseudosection - {ds.noise_level * 100:.0f}% noise",
        )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_inversion_result(result, data, rhomap=None, forward_mesh=None,
                          outpath: Optional[str] = None, cmin=None, cmax=None):
    """Pseudosection, recovered model and (if available) the true model."""
    if rhomap is not None and (cmin is None or cmax is None):
        cmin, cmax = resistivity_range(rhomap)

    show_true = forward_mesh is not None and rhomap is not None
    ncols = 3 if show_true else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.5))

    show_pseudosection(data, axes[0], title="Apparent resistivity")

    # pg.show (rather than manager.showResult) avoids the automatic
    # coverage-alpha overlay, which can crash when some cells have zero
    # sensitivity (log10 -> -inf).
    pg.show(result.para_domain, result.model, ax=axes[1], cMap=RES_CMAP,
            cMin=cmin, cMax=cmax, logScale=True, label=pg.unit("res"))
    axes[1].set_title(
        f"{result.method.capitalize()} inversion\n"
        f"lam={result.lam:g}, chi2={result.chi2:.2f}",
        fontweight="bold",
    )

    if show_true:
        pg.show(forward_mesh, data=rhomap, ax=axes[2], cMap=RES_CMAP,
                cMin=cmin, cMax=cmax, showMesh=False)
        axes[2].set_title("True model", fontweight="bold")
        for ax in axes[1:]:
            ax.set_xlim(axes[2].get_xlim())

    for ax in axes:
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")

    fig.tight_layout()
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_residual_histogram(stats, title: str, outpath: Optional[str] = None):
    """Histogram of error-normalised residuals against a standard normal."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(stats.normalized, bins=30, density=True, alpha=0.7,
            color="skyblue", edgecolor="black")
    x = np.linspace(-4, 4, 200)
    ax.plot(x, norm.pdf(x), "r-", lw=2, label="Standard normal")
    ax.set_xlabel("Normalised residual")
    ax.set_ylabel("Probability density")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.98,
            f"mean = {np.mean(stats.normalized):.2f}\n"
            f"std  = {np.std(stats.normalized):.2f}\n"
            f"|r|>3: {stats.outlier_fraction * 100:.1f}%",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.tight_layout()
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_rms_vs_noise(summary: Dict[str, Dict[float, float]], outpath: Optional[str] = None):
    """RMS (sqrt chi2) versus noise level, one line per method.

    ``summary`` maps ``method -> {noise_level: rms}``.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, series in summary.items():
        noises = sorted(series)
        ax.plot([n * 100 for n in noises], [series[n] for n in noises],
                marker="o", label=method)
    ax.axhline(1.0, color="grey", ls="--", lw=1, label="chi2 = 1 (ideal)")
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("RMS  =  sqrt(chi2)")
    ax.set_title("Data fit vs noise level", fontweight="bold")
    ax.grid(alpha=0.4, ls="--")
    ax.legend()
    fig.tight_layout()
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_coverage(result, outpath: Optional[str] = None):
    """Logarithmic cumulative-sensitivity coverage of the inversion."""
    fig, ax = plt.subplots(figsize=(8, 4))
    cov = np.asarray(result.manager.coverage())
    # replace non-finite entries (zero-sensitivity cells) with the minimum
    # finite value so the colour scale and histogram stay well defined
    finite = cov[np.isfinite(cov)]
    fill = finite.min() if finite.size else 0.0
    cov = np.where(np.isfinite(cov), cov, fill)
    pg.show(result.para_domain, cov, ax=ax, cMap="magma",
            label="log10 cumulative sensitivity")
    ax.set_title(f"Coverage - {result.method.capitalize()}", fontweight="bold")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    fig.tight_layout()
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_lcurve(lcurve, outpath: Optional[str] = None):
    """Log-log L-curve of model roughness against data misfit."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(lcurve.phi_data, lcurve.phi_model, "o-", color="teal")
    for lam, xd, ym in zip(lcurve.lambdas, lcurve.phi_data, lcurve.phi_model):
        ax.annotate(f"{lam:g}", (xd, ym), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Data misfit  phi_d")
    ax.set_ylabel("Model roughness  phi_m")
    ax.set_title("L-curve (label = lambda)", fontweight="bold")
    ax.grid(alpha=0.4, which="both", ls="--")
    fig.tight_layout()
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_doi(doi, outpath: Optional[str] = None):
    """Depth-of-investigation index (0 = well resolved, 1 = unconstrained)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    pg.show(doi.para_domain, doi.doi_index, ax=ax, cMap="viridis_r",
            cMin=0, cMax=1, label="DOI index")
    ax.set_title("Depth of investigation", fontweight="bold")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    fig.tight_layout()
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_method_comparison(results: Dict[str, object], rhomap,
                           outpath: Optional[str] = None):
    """Recovered models from several methods, side by side, shared colour scale."""
    cmin, cmax = resistivity_range(rhomap)
    items = [(m, r) for m, r in results.items() if r.success]
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    axes = np.atleast_1d(axes)
    for ax, (method, res) in zip(axes, items):
        pg.show(res.para_domain, res.model, ax=ax, cMap=RES_CMAP,
                cMin=cmin, cMax=cmax, logScale=True, label=pg.unit("res"))
        ax.set_title(f"{method.capitalize()}\nchi2={res.chi2:.2f}", fontweight="bold")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
    fig.tight_layout()
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_method_page(forward_mesh, rhomap, datasets, per_noise, method,
                     outpath: Optional[str] = None):
    """One overview page for a single inversion method.

    Layout: the true model across the top, then one column per noise level
    with its pseudosection (middle row) and recovered model (bottom row).
    """
    import matplotlib.gridspec as gridspec

    cmin, cmax = resistivity_range(rhomap)
    n = len(datasets)
    fig = plt.figure(figsize=(5.4 * n, 12.5))
    gs = gridspec.GridSpec(3, n, height_ratios=[1.05, 1.0, 1.0],
                           hspace=0.42, wspace=0.28, figure=fig)

    # top: true model spanning all columns
    ax_true = fig.add_subplot(gs[0, :])
    pg.show(forward_mesh, data=rhomap, ax=ax_true, cMap=RES_CMAP,
            cMin=cmin, cMax=cmax, showMesh=False, label=pg.unit("res"))
    ax_true.set_title("True model", fontweight="bold")
    ax_true.set_xlabel("Distance (m)")
    ax_true.set_ylabel("Depth (m)")
    true_xlim = ax_true.get_xlim()

    for i, ds in enumerate(datasets):
        # middle: pseudosection
        ax_ps = fig.add_subplot(gs[1, i])
        show_pseudosection(ds.data, ax_ps,
                           title=f"Pseudosection - {ds.noise_level*100:.0f}% noise")

        # bottom: recovered model
        ax_inv = fig.add_subplot(gs[2, i])
        res = per_noise.get(ds.noise_level, {}).get("inversion")
        if res is not None and getattr(res, "success", False):
            pg.show(res.para_domain, res.model, ax=ax_inv, cMap=RES_CMAP,
                    cMin=cmin, cMax=cmax, logScale=True, label=pg.unit("res"))
            ax_inv.set_title(
                f"{method.capitalize()} - {ds.noise_level*100:.0f}% noise\n"
                f"chi2 = {res.chi2:.2f}", fontweight="bold")
            ax_inv.set_xlim(true_xlim)
        else:
            ax_inv.text(0.5, 0.5, "inversion failed", ha="center", va="center",
                        transform=ax_inv.transAxes)
        ax_inv.set_xlabel("Distance (m)")
        ax_inv.set_ylabel("Depth (m)")

    fig.suptitle(f"{method.capitalize()} inversion - overview",
                 fontsize=16, fontweight="bold")
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_residual_page(datasets, per_noise, method, outpath: Optional[str] = None):
    """All residual histograms for one method on a single page (one per noise)."""
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    axes = axes[0]
    x = np.linspace(-4, 4, 200)

    for i, ds in enumerate(datasets):
        ax = axes[i]
        stats = per_noise.get(ds.noise_level, {}).get("residuals")
        if stats is None:
            ax.axis("off")
            continue
        ax.hist(stats.normalized, bins=30, density=True, alpha=0.7,
                color="skyblue", edgecolor="black")
        ax.plot(x, norm.pdf(x), "r-", lw=2, label="Standard normal")
        ax.set_title(f"{ds.noise_level*100:.0f}% noise", fontweight="bold")
        ax.set_xlabel("Normalised residual")
        if i == 0:
            ax.set_ylabel("Probability density")
            ax.legend()
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.98,
                f"mean = {np.mean(stats.normalized):.2f}\n"
                f"std  = {np.std(stats.normalized):.2f}\n"
                f"|r|>3: {stats.outlier_fraction * 100:.1f}%",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    fig.suptitle(f"{method.capitalize()} - residual histograms",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_true_model_page(forward_mesh, rhomap, outpath: Optional[str] = None):
    """The true model alone, on its own page."""
    cmin, cmax = resistivity_range(rhomap)
    fig, ax = plt.subplots(figsize=(10, 6))
    pg.show(forward_mesh, data=rhomap, ax=ax, cMap=RES_CMAP,
            cMin=cmin, cMax=cmax, showMesh=False, label=pg.unit("res"))
    ax.set_title("True model", fontweight="bold")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_pseudosections_page(datasets, outpath: Optional[str] = None):
    """All apparent-resistivity pseudosections on one page (one per noise level)."""
    n = len(datasets)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows),
                             squeeze=False)
    axes = axes.flatten()
    for i, ds in enumerate(datasets):
        show_pseudosection(ds.data, axes[i],
                           title=f"Pseudosection - {ds.noise_level*100:.0f}% noise")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Apparent-resistivity pseudosections", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
def plot_results_page(forward_mesh, rhomap, datasets, per_noise, method,
                      outpath: Optional[str] = None):
    """All recovered models for one method on one page (one per noise level)."""
    cmin, cmax = resistivity_range(rhomap)
    n = len(datasets)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.4 * nrows),
                             squeeze=False)
    axes = axes.flatten()
    for i, ds in enumerate(datasets):
        ax = axes[i]
        res = per_noise.get(ds.noise_level, {}).get("inversion")
        if res is not None and getattr(res, "success", False):
            pg.show(res.para_domain, res.model, ax=ax, cMap=RES_CMAP,
                    cMin=cmin, cMax=cmax, logScale=True, label=pg.unit("res"))
            ax.set_title(f"{method.capitalize()} - {ds.noise_level*100:.0f}% noise\n"
                         f"chi2 = {res.chi2:.2f}", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "inversion failed", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{method.capitalize()} inversion - recovered models",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if outpath:
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig
