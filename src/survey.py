"""
Survey / measurement-scheme construction.
============================================================================

Thin wrapper around ``pygimli.physics.ert.createData`` that builds an
electrode layout and measurement array from a :class:`~ertinv.config.Survey`.
"""

from __future__ import annotations

import numpy as np
from pygimli.physics import ert

from .config import Survey

#: Human-readable names for the array codes, for reporting / plots.
ARRAY_NAMES = {
    "dd": "dipole-dipole",
    "wa": "Wenner-alpha",
    "wb": "Wenner-beta",
    "slm": "Schlumberger",
    "pd": "pole-dipole",
    "pp": "pole-pole",
    "gr": "gradient",
}


def create_scheme(survey: Survey):
    """Create an ERT measurement scheme from a :class:`Survey`.

    Returns
    -------
    scheme :
        A pyGIMLi ``DataContainerERT`` with electrode positions and the
        chosen four-point configuration, ready to be simulated.
    """
    electrodes = np.linspace(survey.x_start, survey.x_end, survey.n_electrodes)
    scheme = ert.createData(elecs=electrodes, schemeName=survey.scheme)
    return scheme


def array_label(scheme_code: str) -> str:
    """Return a readable label for an array code (falls back to the code)."""
    return ARRAY_NAMES.get(scheme_code, scheme_code)
