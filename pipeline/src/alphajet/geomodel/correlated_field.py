"""Spatially correlated lognormal permeability fields (Gaussian random field, FFT method)."""
from __future__ import annotations

import numpy as np


def fit_lognormal(perm: np.ndarray) -> tuple[float, float]:
    """Return (median, sigma_ln) of positive permeability samples (any consistent unit)."""
    ln = np.log(np.asarray(perm, dtype=float)[np.asarray(perm) > 0])
    if ln.size < 2:
        raise ValueError("need at least two positive permeability samples")
    return float(np.exp(ln.mean())), float(ln.std(ddof=1))


def gaussian_field(
    shape: tuple[int, int], corr_length_cells: tuple[float, float], rng: np.random.Generator
) -> np.ndarray:
    """Zero-mean, unit-variance field with a Gaussian covariance, shape (ny, nx).

    Generated on a 2x padded grid and cropped so the result is not periodic. The sample is
    standardized, so its mean and std match the target exactly (the spatial structure does not).
    """
    ny, nx = shape
    ax, ay = corr_length_cells
    py, px = 2 * ny, 2 * nx
    kx = 2 * np.pi * np.fft.fftfreq(px)[None, :]
    ky = 2 * np.pi * np.fft.fftfreq(py)[:, None]
    amp = np.exp(-((kx * ax) ** 2 + (ky * ay) ** 2) / 4.0)  # sqrt of Gaussian spectral density
    noise = rng.standard_normal((py, px))
    field = np.fft.ifft2(np.fft.fft2(noise) * amp).real[:ny, :nx]
    return (field - field.mean()) / field.std()


def lognormal_perm(
    shape: tuple[int, int],
    median: float,
    sigma_ln: float,
    corr_length_cells: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Correlated lognormal permeability, (ny, nx), in the unit of `median`."""
    return median * np.exp(sigma_ln * gaussian_field(shape, corr_length_cells, rng))
