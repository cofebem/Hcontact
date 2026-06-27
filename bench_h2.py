"""Scaling benchmark: matrix-free H2 (bbFMM) vs classical ACA H-matrix.

Reports build time, matvec time, and stored memory across grid sizes, and
the H2 matvec accuracy vs the H-matrix.  Demonstrates O(N) H2 memory.

Run (fenicsx-env):  python bench_h2.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
import numpy as np
import hmatrix_contact as hc

Q = 6
NS = [64, 128, 256, 512]


def time_matvec(solver, x, reps=5):
    solver.matvec(x)  # warm up
    t0 = time.perf_counter()
    for _ in range(reps):
        y = solver.matvec(x)
    return (time.perf_counter() - t0) / reps, np.asarray(y).ravel()


def bench():
    print(f"{'Ns':>5} {'N':>9} | "
          f"{'H2 build':>9} {'H2 mv':>9} {'H2 MiB':>9} | "
          f"{'Hm build':>9} {'Hm mv':>9} {'Hm MiB':>9} | {'relerr':>9}")
    for Ns in NS:
        N = Ns * Ns
        rng = np.random.default_rng(0)
        x = rng.standard_normal(N)

        t0 = time.perf_counter()
        h2 = hc.ContactSolver(grid_size=Ns, backend="h2", q=Q)
        tb2 = time.perf_counter() - t0
        info2 = h2.hmatrix_info()
        tm2, y2 = time_matvec(h2, x)

        t0 = time.perf_counter()
        hm = hc.ContactSolver(grid_size=Ns, leaf_size=64)
        tbh = time.perf_counter() - t0
        infoh = hm.hmatrix_info()
        tmh, yh = time_matvec(hm, x)

        rel = np.linalg.norm(y2 - yh) / np.linalg.norm(yh)
        mib2 = info2["bytes"] / 1048576.0
        mibh = infoh["bytes"] / 1048576.0
        print(f"{Ns:>5} {N:>9} | "
              f"{tb2:>9.3f} {tm2*1e3:>8.2f}m {mib2:>9.1f} | "
              f"{tbh:>9.3f} {tmh*1e3:>8.2f}m {mibh:>9.1f} | {rel:>9.2e}")


if __name__ == "__main__":
    bench()
