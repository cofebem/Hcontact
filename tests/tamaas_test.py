"""Minimal regression tests for Tamaas rough-surface normal contact.

This is intentionally much smaller than the production script:
- no BlackDynamite output;
- no pressure histograms;
- no fluid solver;
- no files written to disk.

Run with:
    pytest -q test_tamaas_contact.py
"""

from __future__ import annotations

import numpy as np
import pytest


tm = pytest.importorskip("tamaas")
stats = tm.Statistics2D


def make_self_affine_surface(
    n: int = 64,
    *,
    hurst: float = 0.8,
    k0: int = 1,
    k1: int = 4,
    k2: int = 32,
    seed: int = 12345,
    target_rms_slope: float = 1.0,
) -> np.ndarray:
    """Generate and normalize one rough surface as in the original script."""
    spectrum = tm.Isopowerlaw2D()
    spectrum.hurst = hurst
    spectrum.q0 = k0
    spectrum.q1 = k1
    spectrum.q2 = k2

    generator = tm.SurfaceGeneratorFilter2D([n, n])
    generator.spectrum = spectrum
    generator.random_seed = seed

    surface = np.asarray(generator.buildSurface(), dtype=float)
    surface -= surface.mean()

    rms_height = stats.computeRMSHeights(surface)
    assert rms_height > 0.0
    surface /= rms_height

    rms_slope = stats.computeSpectralRMSSlope(surface)
    assert rms_slope > 0.0
    surface *= target_rms_slope / rms_slope

    # Same convention as in the original run: highest asperity at zero,
    # all other points below it.
    surface -= surface.max()
    return np.ascontiguousarray(surface)


def make_nonperiodic_model(n: int, e_star: float = 1.0):
    """Create the same non-periodic 2D Tamaas model as the reference script."""
    model = tm.ModelFactory.createModel(tm.model_type.basic_2d, [1.0, 1.0], [n, n])
    tm.ModelFactory.registerNonPeriodic(model, "dcfft")
    model.E = e_star
    model.nu = 0.0
    return model


def solve_rough_contact(
    surface: np.ndarray,
    *,
    e_star: float = 1.0,
    nominal_pressure: float = 0.05,
    epsilon: float = 1.0e-8,
):
    """Solve one frictionless normal-contact problem."""
    n = surface.shape[0]
    assert surface.shape == (n, n)

    model = make_nonperiodic_model(n, e_star=e_star)
    solver = tm.PolonskyKeerRey(model, surface, epsilon)
    solver.max_iter = 5000
    objective = solver.solve(nominal_pressure)
    return model, objective


def test_nonperiodic_rough_contact_sanity() -> None:
    """A small non-periodic rough-contact solve should give a valid pressure field."""
    surface = make_self_affine_surface(n=64)
    nominal_pressure = 0.05

    model, objective = solve_rough_contact(surface, nominal_pressure=nominal_pressure)
    traction = np.asarray(model.traction, dtype=float)
    displacement = np.asarray(model.displacement, dtype=float)

    assert traction.shape == surface.shape
    assert displacement.shape == surface.shape
    assert np.isfinite(traction).all()
    assert np.isfinite(displacement).all()

    # Unilateral contact: no tensile pressure, up to a small numerical tolerance.
    assert traction.min() >= -1.0e-8

    positive = traction > 1.0e-10
    contact_fraction = positive.mean()
    assert 0.0 < contact_fraction < 1.0

    # For a unit-area model, the mean pressure should match the imposed load.
    assert np.isclose(traction.mean(), nominal_pressure, rtol=5.0e-3, atol=1.0e-8)

    if np.isscalar(objective):
        assert np.isfinite(objective)


def test_contact_area_increases_with_nominal_pressure() -> None:
    """For the same rough surface, contact area should increase with pressure."""
    surface = make_self_affine_surface(n=64)
    model = make_nonperiodic_model(n=64)
    solver = tm.PolonskyKeerRey(model, surface, 1.0e-8)
    solver.max_iter = 5000

    pressures = [0.025, 0.05, 0.10]
    contact_fractions = []
    mean_pressures = []

    for p_ext in pressures:
        solver.solve(p_ext)
        traction = np.asarray(model.traction, dtype=float)
        contact_fractions.append((traction > 1.0e-10).mean())
        mean_pressures.append(traction.mean())

    assert np.all(np.diff(contact_fractions) >= -1.0e-12)
    assert np.allclose(mean_pressures, pressures, rtol=5.0e-3, atol=1.0e-8)