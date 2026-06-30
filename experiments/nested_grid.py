"""Prototype: nested-grid (cascadic / full-multigrid) continuation for contact.

Solve the contact problem coarse->fine. The surface is generated on the finest
grid and restricted (2x2 block average) to each coarser level, so every level
is the SAME physical surface. The coarse pressure is prolonged (injected) to
the next finer grid as a warm start, seeding both the pressure and the active
set; the fine solve then only refines. Each level uses the |q| spectral
preconditioner (precond_pcg).

Compares, at each target grid: cold preconditioned solve vs nested continuation
(fine-grid iterations and cost-weighted total work).

Run (fenicsx-env):  python experiments/nested_grid.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import hmatrix_contact as hc

from precond_pcg import pk_pcg, make_precond, symbol_absq, bandlimited_surface


def restrict(field2d, factor=2):
    """Block-average restriction (fine -> coarse)."""
    Ns = field2d.shape[0]
    c = Ns // factor
    return field2d.reshape(c, factor, c, factor).mean(axis=(1, 3))


def prolong(p_coarse2d, factor=2, mode="inject"):
    """Prolongation (coarse -> fine). 'inject' replicates each cell into its
    factor x factor block (blocky, load-preserving); 'bilinear' smooths."""
    if mode == "inject":
        return np.repeat(np.repeat(p_coarse2d, factor, axis=0), factor, axis=1)
    from scipy.ndimage import zoom
    return np.maximum(zoom(p_coarse2d, factor, order=1, mode="nearest"), 0.0)


def solver_for(Ns):
    s = hc.ContactSolver(grid_size=Ns, backend="h2", q=6)
    return s, (lambda v: np.asarray(s.matvec(v)).ravel())


def nested_solve(Ns_target, Ns_coarsest, p_bar, tol=1e-8, cascadic=True,
                 prolong_mode="inject"):
    levels = [Ns_coarsest]
    while levels[-1] < Ns_target:
        levels.append(levels[-1] * 2)

    # finest surface, restricted down to each level
    fine = bandlimited_surface(Ns_target)
    surf = {Ns_target: fine}
    for Ns in reversed(levels[:-1]):
        surf[Ns] = restrict(surf[Ns * 2])

    p_init = None
    per_level = []
    last = None
    for Ns in levels:
        _, mv = solver_for(Ns)
        g0 = (-surf[Ns]).ravel()
        pc = make_precond(symbol_absq(Ns), Ns)
        lvl_tol = 1e-4 if (cascadic and Ns < Ns_target) else tol
        res = pk_pcg(mv, g0, p_bar, precond=pc, tol=lvl_tol, p_init=p_init)
        per_level.append((Ns, res["iterations"]))
        last = res
        if Ns < Ns_target:
            p2d = res["pressure"].reshape(Ns, Ns)
            p_init = prolong(p2d, mode=prolong_mode).ravel()
    return levels, per_level, last


def main():
    p_bar = 0.05
    Ns_coarsest = 64
    print(f"{'target':>7} {'cold_it':>8} {'fine(inj)':>10} {'fine(bilin)':>12} "
          f"{'work(inj)':>10} {'Ac/A':>6}")
    for Ns_target in [256, 512, 1024]:
        # cold: preconditioned solve from scratch on the same finest surface
        fine = bandlimited_surface(Ns_target)
        _, mv = solver_for(Ns_target)
        g0 = (-fine).ravel()
        cold = pk_pcg(mv, g0, p_bar, precond=make_precond(symbol_absq(Ns_target),
                                                          Ns_target), tol=1e-8)

        _, per_inj, last = nested_solve(Ns_target, Ns_coarsest, p_bar,
                                        prolong_mode="inject")
        _, per_bil, _ = nested_solve(Ns_target, Ns_coarsest, p_bar,
                                     prolong_mode="bilinear")
        fine_inj = per_inj[-1][1]
        fine_bil = per_bil[-1][1]

        cold_work = cold["iterations"] * Ns_target ** 2
        nested_work = sum(it * Ns ** 2 for Ns, it in per_inj)
        ratio = nested_work / cold_work
        print(f"{Ns_target:>7} {cold['iterations']:>8d} {fine_inj:>10d} "
              f"{fine_bil:>12d} {ratio:>9.2f}x {last['contact_fraction']:>6.3f}")


if __name__ == "__main__":
    main()
