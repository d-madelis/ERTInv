"""
Synthetic geological model construction.
============================================================================

Turns a :class:`~ertinv.config.Config` into a pyGIMLi geometry, a meshed
"true model", and the resistivity map that ties region markers to
resistivity values.

Marker convention
-----------------
* Horizontal layers get markers ``1, 2, ..., N`` from top to bottom
  (this is what ``pygimli.meshtools.createWorld`` assigns).
* Targets get markers ``N+1, N+2, ...`` in the order listed in the config.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pygimli as pg
import pygimli.meshtools as mt

from .config import Config


def build_geometry(cfg: Config):
    """Build the PLC geometry (world + layers + targets) from the config.

    Returns
    -------
    geom :
        pyGIMLi geometry (piecewise-linear complex).
    rhomap :
        List of ``[marker, resistivity]`` pairs.
    marker_names :
        ``{marker: label}`` for nicer reporting.
    """
    (x0, z_top), (x1, z_bot) = cfg.forward.world

    # Bottoms of every layer except the last define the internal interfaces.
    interfaces = [ly.bottom for ly in cfg.layers[:-1] if ly.bottom is not None]

    world = mt.createWorld(
        start=[x0, z_top], end=[x1, z_bot], layers=interfaces, worldMarker=True
    )

    rhomap: List[List[float]] = []
    marker_names: Dict[int, str] = {}

    # Layers -> markers 1..N
    for i, layer in enumerate(cfg.layers, start=1):
        rhomap.append([i, float(layer.resistivity)])
        marker_names[i] = layer.name or f"layer_{i}"

    # Targets -> markers N+1..
    next_marker = len(cfg.layers) + 1
    geom = world
    for j, tgt in enumerate(cfg.targets):
        marker = next_marker + j
        if tgt.is_circle:
            body = mt.createCircle(
                pos=list(tgt.center),
                radius=float(tgt.radius),
                marker=marker,
                area=tgt.area,
            )
        else:
            body = mt.createPolygon(
                [list(p) for p in tgt.points],
                isClosed=True,
                marker=marker,
                area=tgt.area,
            )
        geom = geom + body
        rhomap.append([marker, float(tgt.resistivity)])
        marker_names[marker] = tgt.name or f"target_{marker}"

    return geom, rhomap, marker_names


def build_true_model(cfg: Config, scheme=None):
    """Build the geometry, add electrode nodes, and mesh the true model.

    Parameters
    ----------
    cfg :
        The experiment configuration.
    scheme :
        Optional measurement scheme whose electrode positions are inserted
        as mesh nodes (needed for an accurate forward simulation).

    Returns
    -------
    mesh :
        The forward (true-model) mesh.
    rhomap :
        List of ``[marker, resistivity]`` pairs.
    geom :
        The underlying geometry.
    marker_names :
        ``{marker: label}``.
    """
    geom, rhomap, marker_names = build_geometry(cfg)

    if scheme is not None:
        for p in scheme.sensors():
            geom.createNode(p)
            # a node slightly below improves numerical accuracy at electrodes
            geom.createNode(p - [0, 0.1])

    mesh = mt.createMesh(geom, quality=cfg.forward.mesh_quality)

    # Merging a target polygon across a layer interface can leave a few
    # sliver cells with the default marker 0. Relabel them by depth so every
    # cell belongs to its correct layer and the true model stays physical.
    _relabel_background_cells(mesh, cfg, rhomap)

    return mesh, rhomap, geom, marker_names


def _relabel_background_cells(mesh, cfg: Config, rhomap) -> None:
    """Assign layer markers to any leftover marker-0 cells, by centroid depth."""
    known = {int(m) for m, _ in rhomap}
    # Depth intervals for each layer: (marker, z_top, z_bottom)
    (_x0, z_top), (_x1, z_bot) = cfg.forward.world
    intervals = []
    top = z_top
    for i, layer in enumerate(cfg.layers, start=1):
        bottom = layer.bottom if layer.bottom is not None else z_bot
        intervals.append((i, top, bottom))
        top = bottom

    for cell in mesh.cells():
        if int(cell.marker()) in known:
            continue
        z = cell.center().y()
        for marker, ztop, zbot in intervals:
            if zbot <= z <= ztop:
                cell.setMarker(marker)
                break
        else:
            cell.setMarker(intervals[-1][0])  # fall back to deepest layer


def resistivity_per_cell(mesh, rhomap) -> np.ndarray:
    """Return the true resistivity of every cell in ``mesh`` from ``rhomap``."""
    lookup = {int(m): float(r) for m, r in rhomap}
    return np.array([lookup.get(int(m), np.nan) for m in mesh.cellMarkers()])


def true_model_on(mesh, rhomap, target_positions) -> np.ndarray:
    """Interpolate the true resistivity onto arbitrary positions.

    Used to compare the recovered model (defined on the inversion mesh) with
    the true model (defined on the forward mesh) cell-by-cell.
    """
    cell_res = resistivity_per_cell(mesh, rhomap)
    return pg.interpolate(mesh, cell_res, target_positions)
