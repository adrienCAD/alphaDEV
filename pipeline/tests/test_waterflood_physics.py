"""Physics tests on a real (small) JutulDarcy run. Slow the first time (Julia compilation)."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jutuldarcy")
from alphajet.simulation.waterflood import RunParams, run_waterflood  # noqa: E402

CFG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "waterflood_spike.json").read_text())
SWC, SOR = CFG["fluids"]["swc"], CFG["fluids"]["sor"]


@pytest.fixture(scope="module")
def homogeneous():
    # sigma_ln = 0 -> constant permeability
    return run_waterflood(CFG, RunParams(1, 0.2, 0.0, (4.0, 4.0), (2.0, 2.0), SWC, SOR))


def test_no_water_before_breakthrough(homogeneous):
    r = homogeneous
    early = [wc for t, wc in zip(r["years"], r["water_cut"]) if t <= 2.0]
    assert early and max(early) < 1e-6  # connate water is immobile, nothing arrives at the producer
    assert r["breakthrough_years"] is None or r["breakthrough_years"] > 2.0


def test_sw_near_injector_bounded_by_residual_oil(homogeneous):
    sw = homogeneous["final_sw"]
    near = sw.reshape(CFG["grid"]["ny"], CFG["grid"]["nx"])[:3, :3]  # injector corner
    assert near.max() <= 1 - SOR + CFG["checks"]["saturation_endpoint_tol"]
    assert near.max() > SWC + 0.1  # and the flood has actually swept it


def test_checks_pass_with_tolerance_unchanged(homogeneous):
    assert CFG["checks"]["material_balance_rel_tol"] == 1e-3
    assert all(homogeneous["checks"].values()), homogeneous["checks"]
