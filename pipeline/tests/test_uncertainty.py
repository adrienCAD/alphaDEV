import json
from pathlib import Path

import numpy as np

from alphajet.uncertainty import percentiles, sample_params

MC = json.loads((Path(__file__).resolve().parents[1] / "configs" / "monte_carlo.json").read_text())


def test_sampling_reproducible_and_in_range():
    a, b = sample_params(MC, 42), sample_params(MC, 42)
    assert a == b and len(a) == MC["n_runs"]
    r = MC["ranges"]["median_darcy"]
    assert all(r["low"] <= p.median_darcy <= r["high"] for p in a)
    assert len({p.seed for p in a}) == len(a)


def test_percentile_convention():
    curves = np.arange(101, dtype=float)[:, None]
    q = percentiles(curves)
    assert q["P10"][0] == 90.0 and q["P50"][0] == 50.0 and q["P90"][0] == 10.0
