# ERTInv

**Synthetic Electrical Resistivity Tomography (ERT) modelling and inversion with [pyGIMLi](https://www.pygimli.org/).**

ERTInv builds a synthetic geological model, simulates the ERT data it would
produce at several noise levels, inverts that data with a choice of
regularisation schemes, and assesses the result — both against the data
(residuals, misfit) and against the *known* true model. Because the answer is
known in advance, it is an honest laboratory for understanding how an inversion
behaves, how the electrode array and the regularisation shape the image, and how
everything degrades as noise grows.

The project is **notebook-driven**: you declare *what* you want in
[`notebooks/run_inversion.ipynb`](notebooks/run_inversion.ipynb) — the geology,
the survey, the noise levels and the inversion method — and the `ertinv/`
package does the rest, writing a single multi-page PDF report.

---

## Part 1 — Background

### What is ERT?

Electrical Resistivity Tomography images the electrical resistivity of the
ground. A known electric current `I` is injected between two **current
electrodes** (conventionally labelled A and B), and the resulting voltage `ΔV`
is measured between two **potential electrodes** (M and N). For a homogeneous
half-space these quantities give the true resistivity through a purely geometric
constant `k` that depends only on the four electrode positions:

```
rho_apparent = k · ΔV / I
```

Over real, heterogeneous ground the same formula no longer returns the true
resistivity but an **apparent resistivity** — a volume-averaged blend of the
resistivities the current sampled on its way through the subsurface. Collecting
many such readings, with electrodes at many spacings and positions, builds up a
data set that is sensitive to how resistivity varies both laterally and with
depth.

Resistivity is a useful proxy because it responds to porosity, water and clay
content, salinity, compaction and lithology. That is why ERT is used across
hydrogeology, engineering-site and levee investigations, mineral exploration,
archaeology, and the detection of buried objects (utilities, cavities, ordnance).

### The pseudosection

Before any inversion, the raw apparent resistivities are usually shown as a
**pseudosection**: each reading is plotted at a horizontal position midway
between its electrodes and at a "pseudo-depth" that grows with the electrode
separation. It is *not* a depth section — it is a convenient picture of the raw
data that reveals its coverage and its overall structure. In this project the
pseudosections have the characteristic triangular footprint of a surface survey:
wide spacings (deep, few readings) taper to a point beneath the line's centre.

### Electrode arrays

An **array** is the rule for which electrodes act as A, B, M and N in each
reading. Different arrays trade off signal strength, depth of investigation, and
lateral vs. vertical resolution. ERTInv exposes the standard pyGIMLi arrays via
short codes:

| Code  | Array          | Character                                                         |
|-------|----------------|------------------------------------------------------------------|
| `dd`  | dipole–dipole  | strong lateral resolution; good for steep/blocky targets; weaker signal at depth |
| `wa`  | Wenner-alpha   | strong signal, robust to noise; good vertical resolution; poorer lateral detail |
| `wb`  | Wenner-beta    | a dipole–dipole variant of the Wenner family                     |
| `slm` | Schlumberger   | a good all-round compromise between depth and lateral resolution |
| `pd`  | pole–dipole    | greater depth of investigation; needs a remote electrode         |
| `pp`  | pole–pole      | largest depth range but the poorest resolution                   |
| `gr`  | gradient       | efficient for multi-channel acquisition; broad coverage          |

There is no universally "best" array — the right choice depends on the target.
Comparing them on the *same* synthetic model (see `compare_arrays`) is one of the
things this project makes easy.

### Forward modelling

The **forward problem** answers: *given a resistivity model, what would the
survey measure?* Physically the electric potential obeys a Poisson-type equation,

```
∇ · ( (1/rho) ∇V ) = − I · δ(source)
```

which pyGIMLi solves numerically on a finite-element mesh for each current
injection, then samples at the potential electrodes to predict every apparent
resistivity. ERTInv adds Gaussian noise to these predictions to mimic a real
acquisition. Crucially, the data are simulated on a **fine, geometry-conforming
mesh** that is *different* from the mesh the inversion later uses — see
"inverse crime" below.

### The inverse problem and regularisation

**Inversion** goes the other way: *given the measured data, what resistivity
model produced them?* This is **ill-posed** — many different models fit the same
data equally well, and small data errors can produce wild swings in the model.
The cure is **regularisation**: an extra constraint that expresses what we
consider a "reasonable" model, added to the data-misfit that we minimise:

```
minimise    ‖ (d_obs − f(m)) / σ ‖²   +   λ · R(m)
            └──────── data misfit ────────┘     └ regularisation ┘
```

`f(m)` is the forward operator, `σ` the data errors, `λ` the **regularisation
strength** (larger λ → simpler, smoother model but poorer data fit), and `R(m)`
the regularisation term. The four methods in ERTInv are genuinely different
choices of `R`, not just different `λ`:

- **`occam`** — smooth, first-order (Occam) regularisation. `R` penalises
  *roughness* (differences between neighbouring cells), so the inversion returns
  the smoothest model consistent with the data. The safe, standard choice; it
  blurs sharp boundaries into gradients.
- **`marquardt`** — zeroth-order (Marquardt–Levenberg) damping. `R` penalises
  the deviation from a homogeneous reference model, damping each parameter
  toward it. Classic for few-parameter problems; on dense 2-D pixel models it
  tends to under-fit, so expect a higher χ².
- **`blocky`** — an L1 (rather than L2) norm on the model roughness. L1 tolerates
  a few large jumps instead of spreading change smoothly, so it recovers
  **block-like bodies with sharp edges** — useful for walls, ore bodies, or
  compact buried objects.
- **`robust`** — an L1 norm on the *data* misfit. This down-weights **outliers**,
  so a handful of bad readings cannot drag the whole model. Use it when the data
  are spiky or contaminated.

The inversion itself is iterative (Gauss–Newton): starting from a homogeneous
model, it repeatedly computes the sensitivity (Jacobian) of the data to the
model, solves a linearised, regularised least-squares step, and updates the
model until the data fit stops improving.

### Reading the result: χ², RMS and coverage

The primary quality number is the **noise-weighted data misfit**:

```
χ² = mean( ( (d_obs − d_pred) / σ )² )
```

In this project `RMS = sqrt(χ²)`. The target is **χ² ≈ 1**, which means the model
explains the data *to within the assigned noise* — the inversion has fit the
signal and stopped at the noise floor (the *discrepancy principle*):

- **χ² ≫ 1** — under-fitting: the model is too smooth or simply wrong; the data
  are not explained.
- **χ² ≪ 1** — over-fitting: the model is chasing the noise and inventing
  structure that isn't real.
- **χ² ≈ 1** — the goal.

A healthy run sits close to 1 at every noise level and rises gently as noise
grows. The **residuals** (per-datum, error-normalised misfits) should look like a
standard normal distribution centred on zero; systematic skew or fat tails point
to a problem with the model or the noise assumptions.

Because ERTInv knows the true model, it also reports a **model-space
correlation** — how much the recovered section actually resembles the truth. This
is a luxury only synthetic tests have, and it naturally falls as noise increases.

### The "inverse crime"

If you simulate data and then invert it *on the very same mesh*, the inversion
can reproduce the data almost perfectly for the wrong reasons — an artificially
flattering result that would never occur with field data. ERTInv avoids this
**inverse crime** by generating the data on one fine mesh and inverting on a
separate, independently built parameter mesh.

---

## Part 2 — Running it

### Installation

pyGIMLi ships a compiled C++ core (`pgcore`); the most reliable way to get a
complete, working build — **especially on Windows** — is conda/mamba using the
`gimli` **and** `conda-forge` channels together:

```bash
conda create -n pg -c gimli -c conda-forge python=3.11 pygimli numpy matplotlib scipy jupyter ipykernel
conda activate pg
python -c "import pygimli; print(pygimli.__version__)"
```

> **Windows note.** If a package fails to extract with an `InvalidArchiveError`
> mentioning a very long path, enable long paths (run once, as Administrator, then
> reboot):
> ```powershell
> New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
> ```

> **Do not** `pip install pygimli` into the same environment — the pip build does
> not bring the compiled core, and mixing the two is the usual cause of a
> `No module named 'pgcore'` error.

Register the environment as a Jupyter kernel so the notebook can find it:

```bash
python -m ipykernel install --user --name pg --display-name "Python (pg)"
```

### How to run

1. Open [`notebooks/run_inversion.ipynb`](notebooks/run_inversion.ipynb).
2. Select the **`Python (pg)`** kernel (top-right in VS Code / *Kernel → Change
   Kernel* in Jupyter).
3. **Run All**.

The pipeline runs quietly and writes one PDF,
`results/ERTInv_report_<method>.pdf`, with four pages: the **true model**, the
**pseudosections**, the **recovered models**, and the **residual histograms** —
one column per noise level. The last cell opens the PDF automatically. Nothing is
drawn inline in the notebook.

### What to change if you want to experiment

Everything you control lives in **section 2** of the notebook (a `Config`
object). Nothing is hidden inside the processing code.

- **Inversion method** — `cfg.inversion.method = "occam"` — swap for
  `"marquardt"`, `"blocky"` or `"robust"`. Each run makes its own PDF, so you can
  compare reports side by side.
- **Noise levels** — `cfg.forward.noise_levels = [0.02, 0.05, 0.10]` — the
  relative Gaussian noise added to each simulated dataset (0.05 = 5 %). Add or
  remove values to change how many columns the report has.
- **The geology** — edit the `layers` and `targets` of the `Config`:
  - a **`Layer`** has a `bottom` depth (metres, negative down; the deepest layer
    uses `bottom=None`) and a `resistivity` in Ω·m;
  - a **`Target`** is a buried body given by its polygon `points`
    (`[x, z]` corners, z negative down), its `resistivity`, and a mesh `area`
    (smaller → finer local mesh).
- **Target size / shape / depth** — change a target's `points`. For example,
  widen a block by moving its left/right corners, or bury it deeper by making the
  `z` values more negative.
- **The survey** — `Survey(scheme="dd", n_electrodes=31, x_start=-20, x_end=20)`:
  change `scheme` to any array code (`dd`, `wa`, `wb`, `slm`, `pd`, `pp`, `gr`),
  add electrodes for finer coverage, or widen the line.
- **Where results go** — `cfg.outdir`.

Two convenience routines round out the notebook:

- `compare_methods(cfg, methods=["occam","marquardt","blocky","robust"], noise_level=0.05)`
  runs all four methods on one dataset and writes a side-by-side figure — the
  quickest way to *see* how each regularisation reshapes the same data.
- `compare_arrays(cfg, arrays=["dd","wa","slm"], noise_level=0.05)`
  inverts the same target with different electrode arrays.

The final notebook cell shows how to build a **completely custom `Config`** from
scratch.

### Building your own model

```python
from ertinv import Config, Layer, Target, Survey, Forward, Inversion, run

cfg = Config(
    layers=[
        Layer(bottom=-3.0,  resistivity=100.0, name="topsoil"),
        Layer(bottom=-10.0, resistivity=50.0,  name="clay"),
        Layer(bottom=None,  resistivity=300.0, name="bedrock"),
    ],
    targets=[
        Target(name="buried block", resistivity=15.0, area=0.4,
               points=[[-4, -2], [4, -2], [4, -6], [-4, -6]]),
    ],
    survey=Survey(scheme="dd", n_electrodes=31, x_start=-20, x_end=20),
    forward=Forward(noise_levels=[0.03, 0.07]),
    inversion=Inversion(method="blocky"),
    outdir="results_custom",
)
run(cfg)
```

---

## Project layout

```
ERTInv/
├─ ertinv/                  the package
│  ├─ config.py             configuration dataclasses (the control panel)
│  ├─ model.py              build the synthetic geology and forward mesh
│  ├─ survey.py             electrode arrays / measurement schemes
│  ├─ forward.py            simulate noisy synthetic data
│  ├─ inversion.py          the four inversion methods
│  ├─ metrics.py            residuals, model error (and coverage / L-curve / DOI helpers)
│  ├─ plotting.py           the report figures
│  └─ pipeline.py           high-level run / compare_methods / compare_arrays
├─ notebooks/
│  └─ run_inversion.ipynb   ← start here
├─ results/                 the PDF report and any PNGs are written here
├─ environment.yml          conda environment
├─ requirements.txt         pip dependencies (see the install note above)
└─ CITATION.cff
```

## Requirements

- Python 3.11
- [pyGIMLi](https://www.pygimli.org/) ≥ 1.4 (with its compiled `pgcore`)
- NumPy, SciPy, Matplotlib
- Jupyter (plus `ipykernel` to register the environment)

The pinned conda environment is in [`environment.yml`](environment.yml); note
that pyGIMLi itself is best installed from the `gimli` + `conda-forge` channels
as shown above rather than from pip.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on **[pyGIMLi](https://www.pygimli.org/)** — Rücker, C., Günther, T. &
Wagner, F.M. (2017), *pyGIMLi: An open-source library for modelling and inversion
in geophysics*, **Computers & Geosciences** 109, 106–123,
[doi:10.1016/j.cageo.2017.07.011](https://doi.org/10.1016/j.cageo.2017.07.011).

If you use ERTInv in academic work, please cite pyGIMLi and this repository (see
[CITATION.cff](CITATION.cff)).
