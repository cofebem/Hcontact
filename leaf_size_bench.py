"""
leaf_size_bench.py — Sweep leaf sizes to find the optimal H-matrix leaf.

Metrics reported per (Ns, leaf_size) combination:
  - n_dense, n_lowrank     block counts
  - avg_rank, max_rank
  - compression ratio
  - memory (MiB)
  - assembly time (ms)
  - matvec time (ms)
  - Hertz solve iterations and error

Run:
    conda activate fenicsx-env
    cd Hcontact && python leaf_size_bench.py
"""

import sys, time
import numpy as np

sys.path.insert(0, "/home/vyastrebov/WORK/PROJECTS/FENICS/Hcontact/python")
import hmatrix_contact as hc

R, L, E_STAR, P_BAR = 1.0, 1.0, 1.0, 0.05

def hertz_gap(Ns):
    h = L / Ns
    x, y = np.meshgrid((np.arange(Ns) + 0.5) * h - 0.5 * L,
                        (np.arange(Ns) + 0.5) * h - 0.5 * L, indexing="ij")
    return ((x**2 + y**2) / (2 * R)).ravel()

def bench(Ns, leaf_size):
    g0 = hertz_gap(Ns)
    t0 = time.perf_counter()
    solver = hc.ContactSolver(
        grid_size=Ns, domain_size=L, E_star=E_STAR,
        eta=2.0, aca_tol=1e-6, leaf_size=leaf_size,
        use_hmatrix=True,
    )
    t_build = (time.perf_counter() - t0) * 1e3

    info = solver.hmatrix_info()

    # Matvec timing (3 repetitions)
    p_test = np.ones(Ns * Ns) * P_BAR
    for _ in range(2): solver.matvec(p_test)       # warm-up
    t0 = time.perf_counter()
    for _ in range(5): solver.matvec(p_test)
    t_mv = (time.perf_counter() - t0) / 5 * 1e3

    # Solve
    res = solver.solve(g0, p_nominal=P_BAR, tol=1e-8, max_iter=5000)

    return {
        "n_dense":   info["n_dense_blocks"],
        "n_lr":      info["n_lowrank_blocks"],
        "avg_rank":  info["avg_rank"],
        "max_rank":  info["max_rank"],
        "compr":     info["compression"],
        "mem_mib":   info["bytes"] / 1024**2,
        "build_ms":  t_build,
        "mv_ms":     t_mv,
        "iters":     res.iterations,
        "err":       res.error,
        "conv":      res.converged,
    }

LEAF_SIZES = [8, 16, 32, 64, 128]

for Ns in [64, 128]:
    print(f"\n{'='*80}")
    print(f"Ns={Ns}  (N={Ns*Ns})")
    print(f"{'='*80}")
    hdr = (f"{'leaf':>6} {'n_d':>5} {'n_lr':>5} {'avg_k':>7} {'max_k':>6} "
           f"{'compr':>7} {'mem_MiB':>9} {'build_ms':>9} {'mv_ms':>7} "
           f"{'iters':>6} {'err':>9} {'ok':>4}")
    print(hdr)
    print("-" * len(hdr))
    for ls in LEAF_SIZES:
        try:
            r = bench(Ns, ls)
            print(f"{ls:>6} {r['n_dense']:>5} {r['n_lr']:>5} "
                  f"{r['avg_rank']:>7.2f} {r['max_rank']:>6} "
                  f"{r['compr']:>7.4f} {r['mem_mib']:>9.2f} "
                  f"{r['build_ms']:>9.1f} {r['mv_ms']:>7.2f} "
                  f"{r['iters']:>6} {r['err']:>9.2e} {'Y' if r['conv'] else 'N':>4}")
        except Exception as e:
            print(f"{ls:>6}  ERROR: {e}")
