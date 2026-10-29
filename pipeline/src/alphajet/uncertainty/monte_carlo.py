"""Seeded Monte Carlo over geomodel and relative-permeability parameters."""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc, spearmanr

from alphajet.simulation.waterflood import RunParams

PARAM_NAMES = ("median_darcy", "sigma_ln", "corr_length_x_cells", "corey_water", "corey_oil", "swc", "sor")


def sample_params(mc: dict, base_seed: int) -> list[RunParams]:
    """Latin-hypercube sample of `mc["ranges"]`; each run also gets its own field seed."""
    n, rng_cfg = mc["n_runs"], mc["ranges"]
    u = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=mc["seed"]).random(n)
    cols = {}
    for j, name in enumerate(PARAM_NAMES):
        r = rng_cfg[name]
        lo, hi = (np.log(r["low"]), np.log(r["high"])) if r["scale"] == "log" else (r["low"], r["high"])
        v = lo + u[:, j] * (hi - lo)
        cols[name] = np.exp(v) if r["scale"] == "log" else v
    aniso = mc["corr_anisotropy_y_over_x"]
    return [
        RunParams(
            seed=base_seed + i,
            median_darcy=float(cols["median_darcy"][i]),
            sigma_ln=float(cols["sigma_ln"][i]),
            corr_length_cells=(float(cols["corr_length_x_cells"][i]), float(cols["corr_length_x_cells"][i]) * aniso),
            corey_exponents=(float(cols["corey_water"][i]), float(cols["corey_oil"][i])),
            swc=float(cols["swc"][i]),
            sor=float(cols["sor"][i]),
        )
        for i in range(n)
    ]


def percentiles(curves: np.ndarray) -> dict[str, list[float]]:
    """P10/P50/P90 per time step, rows = runs. Reserves convention: P90 is the low case
    (90 % of runs exceed it), P10 the high case. Applied to every quantity, incl. water cut."""
    return {
        "P10": np.percentile(curves, 90, axis=0).tolist(),
        "P50": np.percentile(curves, 50, axis=0).tolist(),
        "P90": np.percentile(curves, 10, axis=0).tolist(),
    }


def sensitivities(params: list[RunParams], final_rf: np.ndarray) -> dict[str, float]:
    """Spearman rank correlation of each input with final oil recovery factor."""
    x = {
        "median_darcy": [p.median_darcy for p in params],
        "sigma_ln": [p.sigma_ln for p in params],
        "corr_length_x_cells": [p.corr_length_cells[0] for p in params],
        "corey_water": [p.corey_exponents[0] for p in params],
        "corey_oil": [p.corey_exponents[1] for p in params],
        "swc": [p.swc for p in params],
        "sor": [p.sor for p in params],
    }
    return {k: float(spearmanr(v, final_rf).statistic) for k, v in x.items()}
