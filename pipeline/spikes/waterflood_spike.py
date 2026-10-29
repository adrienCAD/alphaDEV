"""Spike: 2D heterogeneous waterflood with PyJutulDarcy (JutulDarcy.jl).

Goal: prove the install, run a small water-oil case in seconds, and discover the
result keys we will use for the surrogate and the trust card.
Built from the PyJutulDarcy README example. All parameters live in configs/waterflood_spike.json.
Writes a trust card (outputs/web/spike/waterflood_spike.trust.json) with a water material-balance check.

Run from alphaDEV/pipeline:  python spikes/waterflood_spike.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import jutuldarcy as jd

from alphajet.geomodel import lognormal_perm
from alphajet.simulation import water_material_balance
from alphajet.trust_card import TrustCard

CFG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "waterflood_spike.json").read_text())
OUT = Path(__file__).resolve().parents[2] / "outputs" / "web" / "spike"
OUT.mkdir(parents=True, exist_ok=True)

SEED = CFG["seed"]
NX, NY = CFG["grid"]["nx"], CFG["grid"]["ny"]
N_YEARS = CFG["years"]
POROSITY = CFG["porosity"]
SW0 = CFG["fluids"]["swc"]  # initial Sw = connate water
SOR = CFG["fluids"]["sor"]
P0_BAR = CFG["initial"]["pressure_bar"]

day = jd.si_unit("day")
darcy = jd.si_unit("darcy")
meter = jd.si_unit("meter")
bar = jd.si_unit("bar")
kg = jd.si_unit("kilogram")

# --- grid and rock: spatially correlated lognormal permeability (seeded), 1 layer
gc = CFG["grid"]
g = jd.CartesianMesh(
    (NX, NY, 1), (NX * gc["dx_m"] * meter, NY * gc["dy_m"] * meter, gc["thickness_m"] * meter)
)
pc = CFG["permeability"]
rng = np.random.default_rng(SEED)
perm_darcy = lognormal_perm(
    (NY, NX), pc["median_darcy"], pc["sigma_ln"], tuple(pc["corr_length_cells"]), rng
)
perm = perm_darcy.ravel() * darcy
domain = jd.reservoir_domain(g, permeability=perm, porosity=POROSITY)

# --- wells: injector in one corner, producer in the opposite corner ---------
inj = jd.setup_vertical_well(domain, 1, 1, name="Injector")
prod = jd.setup_vertical_well(domain, NX, NY, name="Producer")

# --- fluids: water (aqueous) + oil (liquid), immiscible ---------------------
rho_w, rho_o = CFG["fluids"]["rho_w"] * kg / meter**3, CFG["fluids"]["rho_o"] * kg / meter**3
system = jd.ImmiscibleSystem(
    (jd.AqueousPhase(), jd.LiquidPhase()), reference_densities=[rho_w, rho_o]
)
model = jd.setup_reservoir_model(domain, system, wells=[inj, prod])

density = jd.ConstantCompressibilityDensities(
    p_ref=P0_BAR * bar,
    density_ref=[rho_w, rho_o],
    compressibility=[np.float64(4e-5) / bar, np.float64(1e-4) / bar],
)
kr = jd.BrooksCoreyRelativePermeabilities(system, CFG["fluids"]["corey_exponents"], [SW0, SOR])
jd.replace_variables(model, PhaseMassDensities=density, RelativePermeabilities=kr)
parameters = jd.setup_parameters(model)

# Initial state from config (p0 = reference pressure, so surface = reservoir water volume)
state0 = jd.setup_reservoir_state(
    model, Pressure=P0_BAR * bar, Saturations=np.array([SW0, 1.0 - SW0])
)

# --- schedule and controls: inject 0.5 PV over the run, producer at fixed BHP
spy = CFG.get("report_steps_per_year", 1)
dt = np.repeat(365.0 * day / spy, N_YEARS * spy)
pv = float(np.sum(jd.pore_volume(model, parameters)))
inj_rate = CFG["controls"]["injected_pore_volumes"] * pv / float(np.sum(dt))
controls = dict(
    Injector=jd.setup_injector_control(inj_rate, "rate", np.array([1.0, 0.0]), density=rho_w),
    Producer=jd.setup_producer_control(CFG["controls"]["producer_bhp_bar"] * bar, "bhp"),
)
forces = jd.setup_reservoir_forces(model, control=controls)

# --- simulate ---------------------------------------------------------------
t0 = time.perf_counter()
result = jd.simulate_reservoir(
    state0, model, dt, parameters=parameters, forces=forces, convert=True, units="metric"
)
elapsed = time.perf_counter() - t0
print(f"simulate_reservoir: {elapsed:.1f} s (first run includes Julia compilation; run twice)")

# --- discover what came back (we will wire real keys after this) ------------
print("top-level keys:", list(result.keys()))
print("wells:", list(result["WELLS"].keys()))
for w, d in result["WELLS"].items():
    print(f"  {w} keys:", list(d.keys()))
print("field keys:", list(result["FIELD"].keys()))
print("n states:", len(result["STATES"]), "| state keys:", list(result["STATES"][-1].keys()))

# --- basic checks on the final state ---------------------------------------
final = result["STATES"][-1]
sat = np.asarray(final["Saturations"])  # phases x cells
tol = CFG["checks"]["saturation_endpoint_tol"]
checks = {
    "saturations_sum_to_one": bool(np.allclose(sat.sum(axis=0), 1.0, atol=1e-6)),
    "saturations_in_bounds": bool((sat >= -1e-9).all() and (sat <= 1 + 1e-9).all()),
    "water_saturation_increased": bool(sat[0].mean() > SW0),
    "sw_between_swc_and_1_minus_sor": bool(sat[0].min() >= SW0 - tol and sat[0].max() <= 1 - SOR + tol),
}

# --- water material balance: d(FWIP) = FWIT - FWPT at every report step -----
F = result["FIELD"]
# With units="metric", cumulative field totals (FWIT, FWPT, FOPT) come back multiplied by 86400
# (rate in m3/day integrated in seconds). In-place volumes (FWIP) are plain m3. Confirm the
# factor against the injected volume we prescribed instead of trusting it blindly.
CUM = float(day)
fwit, fwpt, fopt = (np.asarray(F[k]) / CUM for k in ("FWIT", "FWPT", "FOPT"))
inj_target = CFG["controls"]["injected_pore_volumes"] * pv
assert abs(fwit[-1] - inj_target) / inj_target < 1e-3, "cumulative unit factor changed; recheck"
fwip0 = pv * SW0  # m3; p0 = p_ref so reservoir and surface volumes coincide
mb_err = water_material_balance(fwit, fwpt, np.asarray(F["FWIP"]), fwip0)
checks["water_material_balance"] = bool(mb_err < CFG["checks"]["material_balance_rel_tol"])
print(f"water material balance: worst rel. error {mb_err:.2e}")
print("checks:", checks)

# --- save JSON (full summary) and a quick picture ---------------------------
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k != "STATES"}
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o

(OUT / "waterflood_spike.json").write_text(
    json.dumps({"seconds": elapsed, "checks": checks, **_clean(result)}, indent=1)
)

card = TrustCard(
    stage="waterflood_spike",
    method="JutulDarcy 2D immiscible water-oil, correlated lognormal permeability, rate-controlled injector",
    inputs=[
        f"seed={SEED}",
        f"grid={NX}x{NY}",
        f"permeability: median {pc['median_darcy']} D, sigma_ln {pc['sigma_ln']}, "
        f"corr length cells {pc['corr_length_cells']} ({pc['provenance']})",
    ],
    metrics={
        "runtime_s": round(elapsed, 2),
        "material_balance_rel_error": mb_err,
        "final_mean_sw": float(sat[0].mean()),
        "oil_recovery_factor": float(fopt[-1] / (pv * (1 - SW0))),
    },
    checks=checks,
    notes=pc["provenance_note"] + " Runtime of the first run includes Julia compilation.",
)
print("wrote", card.write(OUT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(perm_darcy, origin="lower")
    axes[0].scatter([0, NX - 1], [0, NY - 1], c=["w", "r"], edgecolors="k", s=40)
    axes[0].set_title("Permeability (D)")
    im = axes[1].imshow(sat[0].reshape(NY, NX), origin="lower", vmin=0, vmax=1)
    axes[1].scatter([0, NX - 1], [0, NY - 1], c=["w", "r"], edgecolors="k", s=40)
    axes[1].set_title(f"Water saturation after {N_YEARS} y")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(OUT / "waterflood_spike.png", dpi=150)
    print("wrote", OUT / "waterflood_spike.png")
except Exception as exc:  # plotting is optional for the spike
    print("plot skipped:", exc)
