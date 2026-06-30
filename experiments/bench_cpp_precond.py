"""Improvement of the C++ solver: spectral preconditioner (+ nested warm-start)
vs the previous (unpreconditioned) implementation.

For each grid size on the same fixed-band rough surface:
  - none    : solver.solve(precond="none")    -- the previous implementation
  - fourier : solver.solve(precond="fourier") -- |q| spectral preconditioner
  - nested  : coarse->fine, each level fourier-preconditioned and warm-started
              with the prolonged coarse pressure (p_init)

Reports iterations and wall time for each, plus solution agreement (contact
fraction and pressure rel-L2 of fourier vs none) to confirm identical results.

Run (fenicsx-env):  python experiments/bench_cpp_precond.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import hmatrix_contact as hc

from precond_pcg import bandlimited_surface

P_BAR = 0.05


def timed_solve(solver, g0, **kw):
    t0 = time.perf_counter()
    res = solver.solve(g0, P_BAR, tol=1e-8, max_iter=20000, **kw)
    return res, time.perf_counter() - t0


def main():
    print(f"{'Ns':>5} | {'none':>13} | {'fourier':>13} | {'nested':>13} | "
          f"{'speedup_it':>10} {'dArea':>7} {'relL2':>8}")
    print(f"{'':>5} | {'it    t[s]':>13} | {'it    t[s]':>13} | {'it    t[s]':>13} | "
          f"{'(none/nest)':>10}")
    for Ns in [128, 256, 512, 1024]:
        surf = bandlimited_surface(Ns)
        g0 = (-surf).ravel()
        solver = hc.ContactSolver(grid_size=Ns, backend="h2", q=6)

        rn, tn = timed_solve(solver, g0, precond="none")
        rf, tf = timed_solve(solver, g0, precond="fourier")
        # single-entry nested-grid solve (coarse->fine handled in C++)
        t0 = time.perf_counter()
        rnest = hc.solve_nested(grid_size=Ns, gap=g0, p_nominal=P_BAR,
                                coarsest=64, q=6, tol=1e-8)
        nit, tnest = rnest.iterations, time.perf_counter() - t0

        d_area = abs(rf.contact_area - rn.contact_area)
        relL2 = (np.linalg.norm(np.asarray(rf.pressure) - np.asarray(rn.pressure))
                 / np.linalg.norm(np.asarray(rn.pressure)))
        print(f"{Ns:>5} | {rn.iterations:>4d} {tn:>7.2f} | {rf.iterations:>4d} {tf:>7.2f} "
              f"| {nit:>4d} {tnest:>7.2f} | {rn.iterations/max(1,nit):>9.2f}x "
              f"{d_area:>7.4f} {relL2:>8.1e}")


if __name__ == "__main__":
    main()
