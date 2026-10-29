"""One 2D waterflood run on JutulDarcy, parameterised for the Monte Carlo."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from alphajet.geomodel import lognormal_perm
from alphajet.simulation.checks import water_material_balance


@dataclass(frozen=True)
class RunParams:
    seed: int
    median_darcy: float
    sigma_ln: float
    corr_length_cells: tuple[float, float]
    corey_exponents: tuple[float, float]  # (water, oil)
    swc: float  # connate water saturation; initial Sw = swc
    sor: float  # residual oil saturation


def run_waterflood(cfg: dict, p: RunParams) -> dict:
    """Simulate once; return curves on the annual report grid plus checks and timing."""
    import jutuldarcy as jd  # imported lazily: starts Julia

    day, darcy, meter = (jd.si_unit(u) for u in ("day", "darcy", "meter"))
    bar, kg = jd.si_unit("bar"), jd.si_unit("kilogram")
    gc, fc, cc = cfg["grid"], cfg["fluids"], cfg["controls"]
    nx, ny, sw0, p0 = gc["nx"], gc["ny"], p.swc, cfg["initial"]["pressure_bar"]

    perm_darcy = lognormal_perm(
        (ny, nx), p.median_darcy, p.sigma_ln, p.corr_length_cells, np.random.default_rng(p.seed)
    )
    mesh = jd.CartesianMesh(
        (nx, ny, 1), (nx * gc["dx_m"] * meter, ny * gc["dy_m"] * meter, gc["thickness_m"] * meter)
    )
    domain = jd.reservoir_domain(mesh, permeability=perm_darcy.ravel() * darcy, porosity=cfg["porosity"])
    wells = [
        jd.setup_vertical_well(domain, 1, 1, name="Injector"),
        jd.setup_vertical_well(domain, nx, ny, name="Producer"),
    ]
    rho_w, rho_o = fc["rho_w"] * kg / meter**3, fc["rho_o"] * kg / meter**3
    system = jd.ImmiscibleSystem((jd.AqueousPhase(), jd.LiquidPhase()), reference_densities=[rho_w, rho_o])
    model = jd.setup_reservoir_model(domain, system, wells=wells)
    density = jd.ConstantCompressibilityDensities(
        p_ref=p0 * bar,
        density_ref=[rho_w, rho_o],
        compressibility=[np.float64(4e-5) / bar, np.float64(1e-4) / bar],
    )
    # positional args (JutulDarcy src/variables/relperm/simple.jl): exponents, residuals, endpoints.
    # residuals are per phase [water, oil] = [swc, sor]: water is immobile at swc, oil at sor.
    kr = jd.BrooksCoreyRelativePermeabilities(system, list(p.corey_exponents), [p.swc, p.sor])
    jd.replace_variables(model, PhaseMassDensities=density, RelativePermeabilities=kr)
    parameters = jd.setup_parameters(model)
    state0 = jd.setup_reservoir_state(model, Pressure=p0 * bar, Saturations=np.array([sw0, 1.0 - sw0]))

    spy = cfg.get("report_steps_per_year", 1)
    dt = np.repeat(365.0 * day / spy, cfg["years"] * spy)
    pv = float(np.sum(jd.pore_volume(model, parameters)))
    inj_target = cc["injected_pore_volumes"] * pv
    controls = dict(
        Injector=jd.setup_injector_control(inj_target / float(np.sum(dt)), "rate", np.array([1.0, 0.0]), density=rho_w),
        Producer=jd.setup_producer_control(cc["producer_bhp_bar"] * bar, "bhp"),
    )
    forces = jd.setup_reservoir_forces(model, control=controls)

    t0 = time.perf_counter()
    res = jd.simulate_reservoir(state0, model, dt, parameters=parameters, forces=forces, convert=True, units="metric")
    seconds = time.perf_counter() - t0

    F = res["FIELD"]
    # Cumulative totals come back x86400 with units="metric"; verify against the prescribed injection.
    fwit, fwpt, fopt = (np.asarray(F[k]) / float(day) for k in ("FWIT", "FWPT", "FOPT"))
    unit_ok = bool(abs(fwit[-1] - inj_target) / inj_target < 1e-3)
    mb_err = water_material_balance(fwit, fwpt, np.asarray(F["FWIP"]), pv * sw0)
    sat = np.asarray(res["STATES"][-1]["Saturations"])
    tol = cfg["checks"]["saturation_endpoint_tol"]
    checks = {
        "saturations_sum_to_one": bool(np.allclose(sat.sum(axis=0), 1.0, atol=1e-6)),
        "sw_between_swc_and_1_minus_sor": bool(
            # slack: fluid compressibility moves Sw off the endpoints when pressure changes by tens
            # of bar (seen up to ~2e-3 in low-permeability runs); not a mass-balance tolerance
            sat[0].min() >= p.swc - tol and sat[0].max() <= 1 - p.sor + tol
        ),
        "saturations_in_bounds": bool((sat >= -1e-9).all() and (sat <= 1 + 1e-9).all()),
        "cumulative_unit_factor": unit_ok,
        "water_material_balance": bool(mb_err < cfg["checks"]["material_balance_rel_tol"]),
    }
    t_years = np.asarray(res["DAYS"]) / 365.0
    wcut = np.asarray(F["FWCT"])
    hit = np.nonzero(wcut > cfg["checks"]["breakthrough_water_cut"])[0]
    return {
        "seconds": seconds,
        "breakthrough_years": float(t_years[hit[0]]) if hit.size else None,
        "years": t_years.tolist(),
        "oil_recovery_factor": (fopt / (pv * (1 - sw0))).tolist(),  # of initial oil in place
        "water_cut": wcut.tolist(),
        "material_balance_rel_error": mb_err,
        "checks": checks,
        "final_sw": sat[0],
        "perm_darcy": perm_darcy,
    }
