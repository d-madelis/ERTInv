"""
Configuration objects for the ERTInv pipeline.
============================================================================

Everything the pipeline needs is described by plain ``dataclass`` objects.
You build a :class:`Config`, edit its fields (usually inside the notebook),
and hand it to :func:`ertinv.pipeline.run`.

The philosophy: *declare what you want, then run it*. No values are hidden
inside the processing functions - the geological model, the survey layout,
the noise levels and the inversion settings all live here, in one place.

A ready-made model that reproduces the original ERTInv example is available
through :func:`default_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Building blocks of the synthetic geology
# ---------------------------------------------------------------------------
@dataclass
class Layer:
    """One horizontal geological layer.

    Parameters
    ----------
    bottom :
        Depth of the *bottom* of the layer, in metres (negative, downward).
        The top of the first layer is the surface (z = 0); the top of every
        other layer is the bottom of the one above it. The deepest layer
        extends down to the bottom of the modelling world, so its ``bottom``
        value is only used to know it is the last one - set it to ``None``.
    resistivity :
        Resistivity of the layer in Ohm-m.
    name :
        Optional human-readable label (only used in printouts / plots).
    """

    bottom: float | None
    resistivity: float
    name: str = ""


@dataclass
class Target:
    """A buried body (anomaly), either a circle or an arbitrary polygon.

    Give **either** ``center`` + ``radius`` (a circular body) **or**
    ``points`` (a closed polygon). Circles are the easy way to reproduce the
    classic "two anomalies in a homogeneous background" test.

    Parameters
    ----------
    name :
        Label used in printouts and plots.
    resistivity :
        Resistivity of the body in Ohm-m.
    center :
        ``(x, z)`` centre of a circular body, in metres (z negative, down).
    radius :
        Radius of the circular body, in metres.
    points :
        List of ``[x, z]`` vertices for a polygonal body (closed
        automatically). Ignored if ``center`` and ``radius`` are given.
    area :
        Maximum triangle area inside the body (smaller = finer mesh there).
    """

    name: str
    resistivity: float
    center: Sequence[float] | None = None
    radius: float | None = None
    points: Sequence[Sequence[float]] | None = None
    area: float = 0.5

    @property
    def is_circle(self) -> bool:
        return self.center is not None and self.radius is not None


# ---------------------------------------------------------------------------
# Survey geometry
# ---------------------------------------------------------------------------
@dataclass
class Survey:
    """Electrode layout and measurement array.

    ``scheme`` is any array name understood by pyGIMLi's ``ert.createData``:

    ============  ====================================
    code          array
    ============  ====================================
    ``"dd"``      dipole-dipole
    ``"wa"``      Wenner-alpha
    ``"wb"``      Wenner-beta
    ``"slm"``     Schlumberger
    ``"pd"``      pole-dipole
    ``"pp"``      pole-pole
    ``"gr"``      gradient
    ============  ====================================
    """

    scheme: str = "dd"
    n_electrodes: int = 31
    x_start: float = -20.0
    x_end: float = 20.0


# ---------------------------------------------------------------------------
# Forward (synthetic-data) settings
# ---------------------------------------------------------------------------
@dataclass
class Forward:
    """Settings for the synthetic forward simulation.

    Parameters
    ----------
    noise_levels :
        Relative Gaussian noise added to the apparent resistivities, as
        fractions (``0.05`` = 5 %). One dataset is produced per level.
    noise_abs :
        Absolute voltage error floor (V), added in quadrature to the
        relative error.
    seed :
        Base random seed (each noise level uses ``seed + index`` so the
        realisations are reproducible yet different).
    mesh_quality :
        Minimum-angle quality of the *forward* (true-model) mesh. This mesh
        is deliberately kept separate from the inversion mesh to avoid the
        "inverse crime" of fitting data with the very same discretisation
        that generated it.
    world :
        ``((x_min, z_top), (x_max, z_bottom))`` extent of the modelling
        world (metres).
    """

    noise_levels: List[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.20]
    )
    noise_abs: float = 1e-6
    seed: int = 1337
    mesh_quality: float = 34.0
    world: Tuple[Tuple[float, float], Tuple[float, float]] = ((-40.0, 0.0), (40.0, -50.0))


# ---------------------------------------------------------------------------
# Inversion settings
# ---------------------------------------------------------------------------
#: The four inversion "flavours". Each maps to a *real* pyGIMLi
#: regularisation - not just a different lambda. See ``inversion.py``.
METHODS = ("occam", "marquardt", "blocky", "robust")


@dataclass
class Inversion:
    """Which inversion to run and how.

    Parameters
    ----------
    method :
        One of :data:`METHODS`:

        * ``"occam"``     - smooth, first-order roughness (the classic choice)
        * ``"marquardt"`` - Marquardt-Levenberg local damping (zeroth-order)
        * ``"blocky"``    - L1 model constraint, recovers sharp boundaries
        * ``"robust"``    - L1 data constraint, resistant to outliers
    lam :
        Regularisation strength (lambda). If ``None`` a sensible per-method
        default is used (see :data:`ertinv.inversion.DEFAULT_LAMBDA`).
    z_weight :
        Vertical-to-horizontal smoothness ratio (< 1 favours layering).
    max_iter :
        Maximum Gauss-Newton iterations.
    lambda_opt :
        If ``True``, pyGIMLi tries to find lambda so that chi^2 ~ 1
        (only meaningful for the smooth ``occam`` method).
    para_depth :
        Depth of the inversion (parameter) domain in metres. ``0`` = auto.
    para_max_cell_size :
        Largest cell area in the inversion domain. ``0`` = auto.
    quality :
        Minimum-angle quality of the inversion mesh.
    """

    method: str = "occam"
    lam: float | None = None
    z_weight: float = 1.0
    max_iter: int = 20
    lambda_opt: bool = False
    para_depth: float = 0.0
    para_max_cell_size: float = 0.0
    quality: float = 33.6


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """The complete description of one ERTInv experiment."""

    layers: List[Layer]
    targets: List[Target]
    survey: Survey = field(default_factory=Survey)
    forward: Forward = field(default_factory=Forward)
    inversion: Inversion = field(default_factory=Inversion)

    #: Where results (the PDF report, figures) are written.
    outdir: str = "results"
    #: Where the simulated ``.dat`` datasets are written (kept out of ``outdir``).
    datadir: str = "data"
    #: Maximum depth (m) shown in the true-model / recovered-model figures.
    plot_depth: float = 15.0

    # ---- optional analyses (all cheap except where noted) ----
    do_coverage: bool = True        #: cumulative-sensitivity coverage map
    do_model_error: bool = True     #: quantitative true-vs-recovered error
    do_lcurve: bool = False         #: L-curve lambda study (extra inversions)
    lcurve_lambdas: List[float] = field(
        default_factory=lambda: [1, 3, 10, 30, 100, 300]
    )
    do_doi: bool = False            #: depth-of-investigation (2 extra inversions)
    doi_reference_factors: Tuple[float, float] = (0.1, 10.0)

    def describe(self) -> str:
        """Return a short human-readable summary of the configuration."""
        lines = ["ERTInv configuration", "-" * 20]
        lines.append(f"  layers      : {len(self.layers)}")
        lines.append(f"  targets     : {len(self.targets)}")
        lines.append(
            f"  survey      : {self.survey.scheme}, "
            f"{self.survey.n_electrodes} electrodes "
            f"({self.survey.x_start} to {self.survey.x_end} m)"
        )
        lines.append(
            f"  noise levels: {[f'{n*100:.0f}%' for n in self.forward.noise_levels]}"
        )
        lines.append(
            f"  inversion   : {self.inversion.method} "
            f"(lam={self.inversion.lam}, z_weight={self.inversion.z_weight})"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# A ready-made model (reproduces the original ERTInv example)
# ---------------------------------------------------------------------------
def default_config() -> Config:
    """Return the reference model: three layers and three irregular targets.

    This reproduces the geology of the original ``ert_inv.py`` so that the
    refactored pipeline gives comparable results out of the box.
    """
    layers = [
        Layer(bottom=-2.0, resistivity=100.0, name="surface"),
        Layer(bottom=-8.0, resistivity=80.0, name="shallow"),
        Layer(bottom=None, resistivity=200.0, name="deep"),
    ]
    targets = [
        Target(
            name="resistive_block",
            points=[[-17, -4], [-12, -3], [-8, -6], [-13, -8], [-16, -7]],
            resistivity=400.0,
            area=0.5,
        ),
        Target(
            name="conductive_block",
            points=[[3, -5], [7, -4], [9, -7], [6, -9], [4, -8]],
            resistivity=10.0,
            area=0.5,
        ),
        Target(
            name="deep_resistor",
            points=[[-20, -8], [-5, -10], [10, -8], [20, -14],
                    [15, -20], [-10, -18], [-20, -14]],
            resistivity=500.0,
            area=0.5,
        ),
    ]
    return Config(layers=layers, targets=targets)
