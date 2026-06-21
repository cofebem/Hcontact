"""
bench_hmatrix.py — Comprehensive benchmark: Hertz + Rough contact on Ns=128..1024.

Tests three configurations per grid size:
  A) ACA classical, leaf_size=64        (new default)
  B) ACA + SVD recompression tol=0.01   (memory-efficient path)
  C) ACA-GP, leaf_size=64               (geometric pivot)

Self-affine rough surface generated via FFT (pure numpy, no Tamaas):
  Hurst H=0.8, band k1=1..k2=Ns/2, unit RMS slope, seed=42.

Run:
    conda activate fenicsx-env
    cd Hcontact && python bench_hmatrix.py [--max-ns 512]
"""

import sys, time, argparse
import numpy as np

sys.path.insert(0, "/home/vyastrebov/WORK/PROJECTS/FENICS/Hcontact/python")
import hmatrix_contact as hc

# ──────────────────────────────────────────────────────────────
# Surface generators
# ──────────────────────────────────────────────────────────────
def hertz_gap(Ns, L=1.0, R=1.0):
    h = L / Ns
    x, y = np.meshgrid((np.arange(Ns) + 0.5) * h - 0.5 * L,
                        (np.arange(Ns) + 0.5) * h - 0.5 * L, indexing="ij")
    return ((x**2 + y**2) / (2 * R)).ravel()


def self_affine_surface(Ns, hurst=0.8, k1=1, k2=None, seed=42):
    """
    Spectral synthesis of a 2D self-affine surface on an Ns×Ns grid.
    PSD: C(q) ∝ |q|^{-2(H+1)} for |q| in [k1, k2].
    Returns gap field (flattened Ns²): max=0, negative elsewhere.
    """
    if k2 is None:
        k2 = Ns // 2
    rng = np.random.default_rng(seed)

    qx = np.fft.fftfreq(Ns) * Ns   # cycles per grid
    qy = np.fft.fftfreq(Ns) * Ns
    Qx, Qy = np.meshgrid(qx, qy, indexing="ij")
    Q = np.sqrt(Qx**2 + Qy**2)
    Q[0, 0] = 1.0  # avoid div-by-zero at DC

    amplitude = np.where((Q >= k1) & (Q <= k2), Q ** (-(hurst + 1)), 0.0)
    amplitude[0, 0] = 0.0  # zero mean

    # Random complex coefficients with unit Gaussian amplitude
    noise = rng.standard_normal((Ns, Ns)) + 1j * rng.standard_normal((Ns, Ns))
    h_fft = amplitude * noise

    # Enforce Hermitian symmetry so IFFT is real
    surface = np.fft.ifft2(h_fft).real
    surface -= surface.mean()

    # Normalise to unit RMS slope
    gx = np.gradient(surface, axis=0)
    gy = np.gradient(surface, axis=1)
    rms_slope = np.sqrt(np.mean(gx**2 + gy**2))
    if rms_slope > 0:
        surface /= rms_slope

    # Gap convention: gap=0 at the highest asperity (contacts first),
    # gap>0 at valleys (same convention as Hertz: gap >= 0 everywhere).
    return (surface.max() - surface).ravel()


# ──────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────
def run_one(Ns, gap0, p_bar, label,
            use_acagp=False, svd_tol=None, inline_svd_tol=0.0,
            aca_tol=1e-6, tol_solve=1e-8, max_iter=5000):
    t0 = time.perf_counter()
    solver = hc.ContactSolver(
        grid_size=Ns, domain_size=1.0, E_star=1.0,
        eta=2.0, aca_tol=aca_tol, leaf_size=64,
        use_hmatrix=True, use_acagp=use_acagp,
        inline_svd_tol=inline_svd_tol,
    )
    t_build = (time.perf_counter() - t0) * 1e3
    info_raw = solver.hmatrix_info()

    if svd_tol is not None:
        t_svd0 = time.perf_counter()
        solver.recompress(svd_tol)
        t_svd = (time.perf_counter() - t_svd0) * 1e3
    else:
        t_svd = 0.0
    info = solver.hmatrix_info()

    t_s0 = time.perf_counter()
    res = solver.solve(gap0, p_nominal=p_bar, tol=tol_solve, max_iter=max_iter)
    t_solve = (time.perf_counter() - t_s0) * 1e3

    return {
        "label":      label,
        "Ns":         Ns,
        "avg_k_raw":  info_raw["avg_rank"],
        "avg_k":      info["avg_rank"],
        "compr":      info["compression"],
        "mem_mib":    info["bytes"] / 1024**2,
        "build_ms":   t_build,
        "svd_ms":     t_svd,
        "solve_ms":   t_solve,
        "iters":      res.iterations,
        "err":        res.error,
        "conv":       res.converged,
        "contact":    res.contact_area,
        "mean_p":     res.mean_pressure,
    }


def print_table(rows, title, show_contact=False):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    if show_contact:
        hdr = (f"{'Ns':>5} {'config':<26} {'k_raw':>6} {'k':>5} {'compr':>7} "
               f"{'MiB':>7} {'build':>8} {'SVD':>6} {'solve':>8} "
               f"{'iters':>6} {'err':>9} {'Ac/A':>7} {'ok':>3}")
    else:
        hdr = (f"{'Ns':>5} {'config':<26} {'k_raw':>6} {'k':>5} {'compr':>7} "
               f"{'MiB':>7} {'build':>8} {'SVD':>6} {'solve':>8} "
               f"{'iters':>6} {'err':>9} {'ok':>3}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if show_contact:
            print(f"{r['Ns']:>5} {r['label']:<26} {r['avg_k_raw']:>6.2f} {r['avg_k']:>5.2f} "
                  f"{r['compr']:>7.4f} {r['mem_mib']:>7.1f} "
                  f"{r['build_ms']:>8.0f} {r['svd_ms']:>6.0f} {r['solve_ms']:>8.0f} "
                  f"{r['iters']:>6} {r['err']:>9.2e} {r['contact']:>7.4f} "
                  f"{'Y' if r['conv'] else 'N':>3}")
        else:
            print(f"{r['Ns']:>5} {r['label']:<26} {r['avg_k_raw']:>6.2f} {r['avg_k']:>5.2f} "
                  f"{r['compr']:>7.4f} {r['mem_mib']:>7.1f} "
                  f"{r['build_ms']:>8.0f} {r['svd_ms']:>6.0f} {r['solve_ms']:>8.0f} "
                  f"{r['iters']:>6} {r['err']:>9.2e} "
                  f"{'Y' if r['conv'] else 'N':>3}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ns", type=int, default=512,
                    help="Largest Ns to benchmark (default 512; try 1024 if RAM permits)")
    ap.add_argument("--hertz-tol", type=float, default=1e-8)
    ap.add_argument("--rough-tol", type=float, default=1e-8)
    args = ap.parse_args()

    sizes = [ns for ns in [128, 256, 512, 1024] if ns <= args.max_ns]
    P_BAR = 0.05

    # (label, use_acagp, post_svd_tol, inline_svd_tol)
    configs = [
        ("ACA  leaf=64",              False, None,  0.0),
        ("ACA  leaf=64 +SVD0.01",     False, 0.01,  0.0),
        ("ACA  leaf=64 +SVD0.05",     False, 0.05,  0.0),
        ("ACA-GP leaf=64",            True,  None,  0.0),
        ("ACA  inline_SVD0.01",       False, None,  0.01),
    ]

    # ── Hertz benchmark ──────────────────────────────────────
    hertz_rows = []
    print("\nRunning Hertz benchmark ...")
    for Ns in sizes:
        g0 = hertz_gap(Ns)
        tol = args.hertz_tol if Ns <= 256 else 1e-6  # relax for large Ns
        for label, use_gp, svd_tol, isvd in configs:
            try:
                r = run_one(Ns, g0, P_BAR, label,
                            use_acagp=use_gp, svd_tol=svd_tol,
                            inline_svd_tol=isvd, tol_solve=tol)
                hertz_rows.append(r)
                print(f"  Ns={Ns:4d} {label}  iters={r['iters']}  err={r['err']:.2e}  "
                      f"MiB={r['mem_mib']:.1f}  {'OK' if r['conv'] else 'FAIL'}")
            except MemoryError:
                print(f"  Ns={Ns:4d} {label}  OOM")
            except Exception as e:
                print(f"  Ns={Ns:4d} {label}  ERROR: {e}")

    print_table(hertz_rows, "HERTZ CONTACT  (gap = r²/2R, p_bar=0.05)")

    # ── Rough surface benchmark ───────────────────────────────
    rough_rows = []
    print("\n\nRunning rough surface benchmark ...")
    for Ns in sizes:
        gap_rough = self_affine_surface(Ns, hurst=0.8, seed=42)
        tol = args.rough_tol if Ns <= 256 else 1e-6
        for label, use_gp, svd_tol, isvd in configs:
            try:
                r = run_one(Ns, gap_rough, P_BAR, label,
                            use_acagp=use_gp, svd_tol=svd_tol,
                            inline_svd_tol=isvd, tol_solve=tol)
                rough_rows.append(r)
                print(f"  Ns={Ns:4d} {label}  iters={r['iters']}  Ac/A={r['contact']:.4f}  "
                      f"MiB={r['mem_mib']:.1f}  {'OK' if r['conv'] else 'FAIL'}")
            except MemoryError:
                print(f"  Ns={Ns:4d} {label}  OOM")
            except Exception as e:
                print(f"  Ns={Ns:4d} {label}  ERROR: {e}")

    print_table(rough_rows, "ROUGH CONTACT  (self-affine H=0.8, p_bar=0.05)", show_contact=True)
    print()


if __name__ == "__main__":
    main()
