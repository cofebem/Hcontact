"""Tamaas reference run for the H-matrix solver benchmark.

Runs inside the `fluidpaper` conda env (the only one with Tamaas):

    conda run -n fluidpaper python tamaas_reference.py --out data

Generates the self-affine surface of test/tamaas_test.py (n=64, hurst=0.8,
k0=1, k1=4, k2=32, seed=12345, unit RMS slope), solves non-periodic normal
contact at p_nominal = 0.05 with PolonskyKeerRey, and saves:

    surface.npy            rough surface heights (max at 0, rest negative)
    tamaas_pressure.npy    converged contact pressure field (n, n)
    tamaas_meta.json       parameters, contact stats, wall times
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tamaas as tm

stats = tm.Statistics2D


def make_self_affine_surface(n=64, *, hurst=0.8, k0=1, k1=4, k2=32,
                             seed=12345, target_rms_slope=1.0) -> np.ndarray:
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
    surface /= stats.computeRMSHeights(surface)
    surface *= target_rms_slope / stats.computeSpectralRMSSlope(surface)
    surface -= surface.max()  # highest asperity at zero, rest below
    return np.ascontiguousarray(surface)


def make_nonperiodic_model(n: int, e_star: float = 1.0):
    model = tm.ModelFactory.createModel(tm.model_type.basic_2d, [1.0, 1.0], [n, n])
    tm.ModelFactory.registerNonPeriodic(model, "dcfft")
    model.E = e_star
    model.nu = 0.0
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--pressure", type=float, default=0.05)
    ap.add_argument("--epsilon", type=float, default=1e-8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    surface = make_self_affine_surface(n=args.n)
    t_surface = time.perf_counter() - t0

    # Tamaas 2.8.1 quirks (both verified against Hertz theory):
    #  1. PKR must be told to use the dcfft operator via setIntegralOperator,
    #     otherwise it silently solves the *periodic* problem.
    #  2. The dcfft operator's effective modulus is 2 E^2, so E = 1/sqrt(2)
    #     reproduces the half-space with E* = 1 (Hertz: a and p_max match
    #     theory to 0.1% then; with E = 1 they are off by 2^(±1/3, 2/3)).
    model = make_nonperiodic_model(args.n, e_star=1.0 / np.sqrt(2.0))
    solver = tm.PolonskyKeerRey(model, surface, args.epsilon)
    solver.setIntegralOperator("dcfft")
    solver.max_iter = 5000

    t0 = time.perf_counter()
    objective = solver.solve(args.pressure)
    t_solve = time.perf_counter() - t0

    traction = np.asarray(model.traction, dtype=float)
    meta = {
        "n": args.n,
        "p_nominal": args.pressure,
        "epsilon": args.epsilon,
        "tamaas_version": tm.__version__,
        "objective": float(objective) if np.isscalar(objective) else None,
        "contact_fraction": float((traction > 1e-10).mean()),
        "mean_pressure": float(traction.mean()),
        "time_surface_s": t_surface,
        "time_solve_s": t_solve,
    }

    np.save(args.out / "surface.npy", surface)
    np.save(args.out / "tamaas_pressure.npy", traction)
    (args.out / "tamaas_meta.json").write_text(json.dumps(meta, indent=2))
    print("Tamaas reference saved to", args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
