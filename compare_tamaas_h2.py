"""Compare the matrix-free H2 backend against Tamaas FFT (dcfft) references.

For each grid size, a non-periodic Tamaas reference is generated in the
`fluidpaper` env (via `conda run`, cached under data/tamaas_n<N>/), then the
same surface is solved with the H2 backend and the pressure fields compared.

    conda activate fenicsx-env
    python compare_tamaas_h2.py [--regen] [--max-n 512]

Reports per grid size: contact fractions, mean-pressure error, solve time,
relative L2 pressure difference, and H2 operator memory vs the dense size.

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

    print(f"H2 (q={args.q}) vs Tamaas dcfft FFT\n")
    print(f"{'Ns':>5} {'frac_T':>8} {'frac_H2':>8} {'meanp_err':>10} "
          f"{'t_solve[s]':>11} {'relL2':>9} {'H2 MiB':>8} {'dense MiB':>12}")
    worst_l2 = 0.0
    for n in ns_all:
        data = ensure_reference(n, args.regen)
        surface = np.load(data / "surface.npy")
        p_tamaas = np.load(data / "tamaas_pressure.npy")
        meta = json.loads((data / "tamaas_meta.json").read_text())
        p_nominal = meta["p_nominal"]

        solver = hmc.ContactSolver(grid_size=n, domain_size=1.0, E_star=1.0,
                                   backend="h2", q=args.q)
        t0 = time.perf_counter()
        res = solver.solve(gap=-surface, p_nominal=p_nominal,
                           tol=meta["epsilon"], max_iter=5000)
        t_solve = time.perf_counter() - t0

        p_h2 = np.asarray(res.pressure)
        frac_t = meta["contact_fraction"]
        frac_h2 = float((p_h2 > 1e-10).mean())
        l2 = float(np.linalg.norm(p_h2 - p_tamaas) / np.linalg.norm(p_tamaas))
        worst_l2 = max(worst_l2, l2)
        info = solver.hmatrix_info()
        dense_mib = 8.0 * (n * n) ** 2 / MiB

        print(f"{n:>5} {frac_t:>8.4f} {frac_h2:>8.4f} "
              f"{abs(res.mean_pressure/p_nominal-1):>10.2e} {t_solve:>11.3f} "
              f"{l2:>9.4f} {info['bytes']/MiB:>8.2f} {dense_mib:>12.1f}")

        assert res.converged, f"H2 did not converge at n={n}"
        assert abs(frac_h2 - frac_t) < 0.01, \
            f"contact fractions differ by {abs(frac_h2-frac_t):.4f} at n={n}"

    print(f"\nworst-case pressure rel L2 vs Tamaas: {worst_l2:.4%}")
    assert worst_l2 < 0.05, f"L2 {worst_l2:.4%} exceeds 5%"
    print("PASS: H2 backend agrees with Tamaas FFT within tolerance")


if __name__ == "__main__":
    main()
