"""Physics-consistency checks on simulator output."""
from __future__ import annotations

import numpy as np


def water_material_balance(
    fwit: np.ndarray, fwpt: np.ndarray, fwip: np.ndarray, fwip0: float
) -> float:
    """Worst relative error of  d(water in place) = injected - produced, over all report steps.

    All arguments are cumulative / in-place surface volumes in one unit. Error is relative to
    the cumulative water injected at the end of the run.
    """
    fwit, fwpt, fwip = (np.asarray(a, dtype=float) for a in (fwit, fwpt, fwip))
    resid = np.abs((fwip - fwip0) - (fwit - fwpt))
    return float(resid.max() / max(fwit[-1], 1e-30))
