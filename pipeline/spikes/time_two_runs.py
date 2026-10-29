"""Time two consecutive run_waterflood calls in one process (first includes Julia compilation).

Run from alphaDEV/pipeline:  python spikes/time_two_runs.py
"""
import json
from pathlib import Path

from alphajet.simulation.waterflood import RunParams, run_waterflood

cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "waterflood_spike.json").read_text())
pc = cfg["permeability"]
p = RunParams(cfg["seed"], pc["median_darcy"], pc["sigma_ln"], tuple(pc["corr_length_cells"]),
              tuple(cfg["fluids"]["corey_exponents"]), cfg["fluids"]["swc"], cfg["fluids"]["sor"])
for i in (1, 2):
    r = run_waterflood(cfg, p)
    print(f"call {i}: {r['seconds']:.2f} s | final RF {r['oil_recovery_factor'][-1]:.4f} | "
          f"final WCT {r['water_cut'][-1]:.4f} | checks ok {all(r['checks'].values())}")
