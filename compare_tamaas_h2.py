"""Compare the matrix-free H2 backend against Tamaas FFT (dcfft) references.

For each grid size, a non-periodic Tamaas reference is generated in the
`fluidpaper` env (via `conda run`, cached under data/tamaas_n<N>/), then the
same surface is solved with the H2 backend and the pressure fields compared.

    conda activate fenicsx-env
    python compare_tamaas_h2.py [--regen] [--max-n 512]

Reports per grid size the relative L2 pressure difference and a timing
comparison: per-matvec operator apply (Tamaas dcfft FFT vs H2) and the
end-to-end solve. For a fair algorithmic comparison run single-threaded:

    OMP_NUM_THREADS=1 python compare_tamaas_h2.py --regen --max-n 256

The residual ~3% L2 difference is dominated by Tamaas 2.8.1's dcfft near-field
error (+/-2-8% at 1-3 cells; see README), not by H2: the H2 operator
reproduces the exact Love operator to ~3e-6 at q=6 (tests/test_h2.cpp).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python"))
import hmatrix_contact as hmc  # noqa: E402

MiB = 1024.0 * 1024.0


def ensure_reference(n: int, regen: bool) -> Path:
    out = HERE / "data" / f"tamaas_n{n}"
    files = ["surface.npy", "tamaas_pressure.npy", "tamaas_meta.json"]
    if not regen and all((out / f).exists() for f in files):
        return out
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda is None:
        sys.exit("conda not found: cannot run the Tamaas reference")
    cmd = [conda, "run", "-n", "fluidpaper", "python",
           str(HERE / "tamaas_reference.py"), "--grid", str(n), "--out", str(out)]
    print(f"  generating Tamaas reference (n={n}) ...", flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--max-n", type=int, default=256)
    args = ap.parse_args()

    ns_all = [n for n in (64, 128, 256, 512) if n <= args.max_n]

    nthreads = os.environ.get("OMP_NUM_THREADS", "default")
    print(f"H2 (q={args.q}) vs Tamaas dcfft FFT   [OMP_NUM_THREADS={nthreads}]\n")
    print("                       per-matvec [ms]        end-to-end solve [s]")
    print(f"{'Ns':>5} {'N':>9} {'relL2':>7} | {'Tamaas':>8} {'H2':>8} {'H2/T':>6} | "
          f"{'Tamaas':>8} {'H2':>8} {'H2/T':>6} {'H2 it':>6}")
    worst_l2 = 0.0
    for n in ns_all:
        data = ensure_reference(n, args.regen)
        surface = np.load(data / "surface.npy")
        p_tamaas = np.load(data / "tamaas_pressure.npy")
        meta = json.loads((data / "tamaas_meta.json").read_text())
        p_nominal = meta["p_nominal"]

        solver = hmc.ContactSolver(grid_size=n, domain_size=1.0, E_star=1.0,
                                   backend="h2", q=args.q)

        # per-matvec timing (same operation u = S p both sides)
        x = np.full(n * n, p_nominal)
        solver.matvec(x)  # warm up
        reps = 50
        t0 = time.perf_counter()
        for _ in range(reps):
            solver.matvec(x)
        t_mv_h2 = (time.perf_counter() - t0) / reps
        t_mv_t = meta.get("time_matvec_s", float("nan"))

        # end-to-end solve timing
        t0 = time.perf_counter()
        res = solver.solve(gap=-surface, p_nominal=p_nominal,
                           tol=meta["epsilon"], max_iter=5000)
        t_solve_h2 = time.perf_counter() - t0
        t_solve_t = meta["time_solve_s"]

        p_h2 = np.asarray(res.pressure)
        frac_t = meta["contact_fraction"]
        frac_h2 = float((p_h2 > 1e-10).mean())
        l2 = float(np.linalg.norm(p_h2 - p_tamaas) / np.linalg.norm(p_tamaas))
        worst_l2 = max(worst_l2, l2)

        print(f"{n:>5} {n*n:>9} {l2:>7.4f} | "
              f"{t_mv_t*1e3:>8.3f} {t_mv_h2*1e3:>8.3f} {t_mv_h2/t_mv_t:>6.2f} | "
              f"{t_solve_t:>8.3f} {t_solve_h2:>8.3f} {t_solve_h2/t_solve_t:>6.2f} "
              f"{res.iterations:>6d}")

        assert res.converged, f"H2 did not converge at n={n}"
        assert abs(frac_h2 - frac_t) < 0.01, \
            f"contact fractions differ by {abs(frac_h2-frac_t):.4f} at n={n}"

    print(f"\nworst-case pressure rel L2 vs Tamaas: {worst_l2:.4%}")
    print("per-matvec: Tamaas dcfft is O(N log N) FFT; H2 is O(N). End-to-end "
          "solve also reflects differing PCG iteration counts.")
    assert worst_l2 < 0.05, f"L2 {worst_l2:.4%} exceeds 5%"
    print("PASS: H2 backend agrees with Tamaas FFT within tolerance")


if __name__ == "__main__":
    main()
