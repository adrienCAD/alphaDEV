import numpy as np

from alphajet.geomodel import fit_lognormal, lognormal_perm
from alphajet.simulation import water_material_balance


def test_field_reproducible_and_correlated():
    mk = lambda s: lognormal_perm((40, 40), 0.2, 0.6, (8.0, 4.0), np.random.default_rng(s))
    a, b = mk(1), mk(1)
    assert np.array_equal(a, b) and not np.array_equal(a, mk(2))
    ln = np.log(a)
    # nearest-neighbour correlation must be clearly positive (white noise would be ~0)
    assert np.corrcoef(ln[:, :-1].ravel(), ln[:, 1:].ravel())[0, 1] > 0.5


def test_fit_roundtrip():
    p = lognormal_perm((200, 200), 0.3, 0.8, (1.0, 1.0), np.random.default_rng(0))
    med, sig = fit_lognormal(p)
    assert abs(med - 0.3) / 0.3 < 0.02 and abs(sig - 0.8) < 0.02


def test_material_balance():
    fwit = np.array([10.0, 20.0, 30.0])
    fwpt = np.array([0.0, 4.0, 12.0])
    fwip0 = 100.0
    assert water_material_balance(fwit, fwpt, fwip0 + fwit - fwpt, fwip0) < 1e-12
    assert water_material_balance(fwit, fwpt, fwip0 + fwit - fwpt + 1.0, fwip0) > 0.03
