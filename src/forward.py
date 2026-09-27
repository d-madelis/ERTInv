"""
Forward modelling: turning a true model into synthetic ERT data.
============================================================================

Simulates apparent-resistivity data for the configured survey and adds
Gaussian noise. One dataset is produced per noise level. Datasets are
optionally written to disk as pyGIMLi ``.dat`` files so they can be reused.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
from pygimli.physics import ert

from .config import Config


@dataclass
class SyntheticDataset:
    """A single simulated dataset and its metadata."""

    noise_level: float          #: relative noise fraction (0.05 = 5 %)
    data: object                #: pyGIMLi DataContainerERT
    n_negative: int             #: number of (unphysical) negative rhoa values


def simulate_datasets(
    mesh, scheme, rhomap, cfg: Config, save_dir: str | None = None, verbose: bool = True
) -> List[SyntheticDataset]:
    """Simulate one noisy dataset per configured noise level.

    Parameters
    ----------
    mesh, scheme, rhomap :
        Forward mesh, measurement scheme and resistivity map.
    cfg :
        Experiment configuration (uses ``cfg.forward``).
    save_dir :
        If given, each dataset is written there as
        ``synthetic_data_XXpercent_noise.dat``.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    datasets: List[SyntheticDataset] = []
    for i, noise in enumerate(cfg.forward.noise_levels):
        if verbose:
            print(f"  simulating {noise * 100:.0f}% noise ...")

        data = ert.simulate(
            mesh,
            scheme=scheme,
            res=rhomap,
            noiseLevel=noise,
            noiseAbs=cfg.forward.noise_abs,
            seed=cfg.forward.seed + i,
            verbose=verbose,
        )

        n_neg = int(np.sum(np.asarray(data["rhoa"]) <= 0))
        if n_neg:
            # Negative apparent resistivities are unphysical (from strong
            # noise on small readings). Drop them so the inversion is stable.
            data.remove(data["rhoa"] <= 0)

        if save_dir:
            fname = os.path.join(
                save_dir, f"synthetic_data_{noise * 100:.0f}percent_noise.dat"
            )
            data.save(fname)

        datasets.append(SyntheticDataset(noise, data, n_neg))

    return datasets
