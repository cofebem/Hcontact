"""
bench_pr.py — compare Fletcher-Reeves vs Polak-Ribière+ in the P-K solver,
and benchmark H-matrix assembly / solve time for large grids.

Uses Hertz contact gap  g0(x,y) = (x²+y²) / (2R),  R=1, L=1, E*=1.

Run:
    conda activate fenicsx-env
    cd Hcontact && python bench_pr.py
"""

import sys, time
import numpy as np

sys.path.insert(0, "/home/vyastrebov/WORK/PROJECTS/FENICS/Hcontact/python")
import hmatrix_contact as hc

R = 1.0
L = 1.0
E_STAR = 1.0
P_BAR = 0.05
TOL = 1e-8
MAX_ITER = 5000


def hertz_gap(Ns, L=L, R=R):
    h = L / Ns
    coords = (np.arange(Ns) + 0.5) * h - 0.5 * L
    x, y = np.meshgrid(coords, coords, indexing="ij")
    return ((x**2 + y**2) / (2 * R)).ravel()


def run(Ns, use_pr, label, *, eta=2.0, aca_tol=1e-6, leaf_size=32):
    g0 = hertz_gap(Ns)

    t0 = time.perf_counter()
    solver = hc.ContactSolver(
        grid_size=Ns,
        domain_size=L,
        E_star=E_STAR,
        eta=eta,
        aca_tol=aca_tol,
        leaf_size=leaf_size,
        use_hmatrix=True,
    )
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = solver.solve(g0, p_nominal=P_BAR, tol=TOL, max_iter=MAX_ITER, use_pr=use_pr)
    t_solve = time.perf_counter() - t0

    status = "OK" if res.converged else "FAIL"
    print(
        f"  {label:<4}  iters={res.iterations:4d}  err={res.error:.2e}  "
        f"Ac={res.contact_area:.4f}  build={t_build*1e3:7.1f}ms  "
        f"solve={t_solve*1e3:7.1f}ms  [{status}]"
    )
    return res


print("=" * 72)
print("FR vs PR+ comparison (Hertz contact, p_bar=0.05, tol=1e-8)")
print("=" * 72)

for Ns in [64, 128, 256]:
    print(f"\nNs={Ns}  (N={Ns*Ns})")
    run(Ns, use_pr=False, label="FR")
    run(Ns, use_pr=True,  label="PR+")

print("\n" + "=" * 72)
print("Large-grid timing — FR vs PR+ (tol=1e-6; Ns=512 may need ~6 GiB)")
print("=" * 72)

for Ns in [512]:
    print(f"\nNs={Ns}  (N={Ns*Ns})")
    g0 = hertz_gap(Ns)

    t0 = time.perf_counter()
    solver = hc.ContactSolver(
        grid_size=Ns,
        domain_size=L,
        E_star=E_STAR,
        eta=2.0,
        aca_tol=1e-6,
        leaf_size=32,
        use_hmatrix=True,
    )
    t_build = time.perf_counter() - t0

    info = solver.hmatrix_info()
    print(f"  assembly={t_build*1e3:.1f}ms  compression={info['compression']:.3f}x  "
          f"memory={info['bytes']/1024**2:.1f}MiB")

    for use_pr, label in [(False, "FR"), (True, "PR+")]:
        t0 = time.perf_counter()
        res = solver.solve(g0, p_nominal=P_BAR, tol=1e-6, max_iter=MAX_ITER,
                           use_pr=use_pr)
        t_solve = time.perf_counter() - t0
        status = "OK" if res.converged else "FAIL"
        print(f"  {label:<4} iters={res.iterations:4d}  err={res.error:.2e}  "
              f"Ac={res.contact_area:.4f}  solve={t_solve*1e3:.1f}ms  [{status}]")
