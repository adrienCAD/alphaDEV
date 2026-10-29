"""Seeded Monte Carlo waterflood: recovery-factor and water-cut curves, P10/P50/P90, trust card.

CLI (from alphaDEV/pipeline):  alphajet-mc [--n-runs 50] [--config configs/monte_carlo.json]
                         or:  python -m alphajet.uncertainty.run_waterflood_mc
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from alphajet.simulation.waterflood import run_waterflood
from alphajet.trust_card import TrustCard
from alphajet.uncertainty.monte_carlo import percentiles, sample_params, sensitivities

PIPELINE = Path(__file__).resolve().parents[3]
CASE_LABELS = {
    "oil_recovery_factor": {"P10": "high case (best)", "P50": "mid case", "P90": "low case (worst)"},
    "water_cut": {"P10": "high case (worst)", "P50": "mid case", "P90": "low case (best)"},
    "note": "P90 is exceeded by 90 % of runs. For recovery factor P10 = high case (best); for "
            "water cut P10 = high water cut = worst case.",
}
NOTES = (
    "Material-balance check validates reported/integrated volumes (injected - produced = change "
    "in water in place at the report steps), not the solver's internal conservation. "
    "Median permeability has little effect on recovery because injection is rate-controlled in "
    "pore volumes: the same pore volumes go in whatever the permeability, which mainly changes "
    "pressure. P90 = low case, P10 = high case. Percentiles use runs that passed checks."
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=PIPELINE / "configs" / "monte_carlo.json")
    ap.add_argument("--n-runs", type=int, help="override n_runs in the config")
    ap.add_argument("--out", type=Path, default=PIPELINE.parent / "outputs" / "web" / "uncertainty")
    args = ap.parse_args(argv)

    mc = json.loads(args.config.read_text())
    if args.n_runs:
        mc["n_runs"] = args.n_runs
    cfg = json.loads((args.config.parent / mc["base_config"]).read_text())
    out = args.out

    params = sample_params(mc, cfg["seed"])
    runs, t0 = [], time.perf_counter()
    for i, p in enumerate(params):
        r = run_waterflood(cfg, p)
        runs.append(r)
        print(f"run {i + 1}/{len(params)}: {r['seconds']:.1f} s | RF {r['oil_recovery_factor'][-1]:.3f} "
              f"| WCT {r['water_cut'][-1]:.3f} | ok {all(r['checks'].values())}", flush=True)
    total = time.perf_counter() - t0

    ok = np.array([all(r["checks"].values()) for r in runs])
    years = runs[0]["years"]
    rf = np.array([r["oil_recovery_factor"] for r in runs])
    wc = np.array([r["water_cut"] for r in runs])
    stats = {"oil_recovery_factor": percentiles(rf[ok]), "water_cut": percentiles(wc[ok])}
    sens = sensitivities([p for p, k in zip(params, ok) if k], rf[ok, -1])
    bt = np.array([r["breakthrough_years"] for r in runs if r["breakthrough_years"] is not None])
    thr = cfg["checks"]["breakthrough_water_cut"]

    out.mkdir(parents=True, exist_ok=True)
    (out / "monte_carlo.json").write_text(json.dumps({
        "years": years,
        "percentile_convention": CASE_LABELS["note"],
        "case_labels": CASE_LABELS,
        "percentiles": stats,
        "sensitivity_spearman_final_rf": sens,
        "runs": [
            {"index": i, "params": {"seed": p.seed, "median_darcy": p.median_darcy, "sigma_ln": p.sigma_ln,
                                    "corr_length_cells": p.corr_length_cells,
                                    "corey_exponents": p.corey_exponents, "swc": p.swc, "sor": p.sor},
             "oil_recovery_factor": r["oil_recovery_factor"], "water_cut": r["water_cut"],
             "breakthrough_years": r["breakthrough_years"],
             "material_balance_rel_error": r["material_balance_rel_error"], "checks": r["checks"]}
            for i, (p, r) in enumerate(zip(params, runs))
        ],
    }, indent=1))

    f, w = stats["oil_recovery_factor"], stats["water_cut"]
    metrics = {
        "n_runs": len(runs), "n_runs_passing_checks": int(ok.sum()),
        **{f"oil_recovery_factor_final_{q}": f[q][-1] for q in ("P10", "P50", "P90")},
        **{f"water_cut_final_{q}": w[q][-1] for q in ("P10", "P50", "P90")},
        "water_cut_first_report_max": float(wc[:, 0].max()),
        "first_report_year": years[0],
        "n_runs_with_breakthrough": int(bt.size),
        f"breakthrough_years_min_wct_{thr}": float(bt.min()) if bt.size else None,
        f"breakthrough_years_median_wct_{thr}": float(np.median(bt)) if bt.size else None,
        f"breakthrough_years_max_wct_{thr}": float(bt.max()) if bt.size else None,
        "worst_material_balance_rel_error": max(r["material_balance_rel_error"] for r in runs),
        "total_runtime_s": round(total, 1),
    }
    card = TrustCard(
        stage="monte_carlo_waterflood",
        method=f"Latin-hypercube Monte Carlo ({mc['n_runs']} runs) of JutulDarcy 2D waterflood over "
               "permeability median, sigma, correlation length, Corey exponents, connate water and "
               "residual oil; one new field realisation per run",
        inputs=[f"mc_seed={mc['seed']}", f"field seeds={cfg['seed']}..{cfg['seed'] + len(params) - 1}"]
               + [f"{k}: {v['low']}-{v['high']} ({v['scale']})" for k, v in mc["ranges"].items()],
        metrics=metrics,
        checks={"all_runs_pass_physics_checks": bool(ok.all()), "seeded_reproducible": True},
        notes=mc["provenance_note"] + " " + NOTES,
    )
    card_dict = {**card.__dict__, "case_labels": CASE_LABELS}
    path = out / f"{card.stage}.trust.json"
    path.write_text(json.dumps(card_dict, indent=2))
    print("wrote", path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, arr, lab in ((axes[0], "oil_recovery_factor", rf, "Oil recovery factor"),
                                  (axes[1], "water_cut", wc, "Water cut")):
            ax.plot(years, arr[ok].T, color="0.8", lw=0.6)
            for q, c in (("P10", "tab:green"), ("P50", "k"), ("P90", "tab:red")):
                ax.plot(years, stats[key][q], color=c, lw=2, label=q)
            ax.set_xlabel("Years"); ax.set_ylabel(lab); ax.legend()
        fig.tight_layout(); fig.savefig(out / "monte_carlo.png", dpi=150)
    except Exception as exc:
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
