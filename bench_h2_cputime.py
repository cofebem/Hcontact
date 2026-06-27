"""CPU-time scaling study of the H2 backend for the example_rough_contact
problem, from Ns=128 up to Ns=16384 (N up to 2.7e8).

For each grid size it reports operator build time, per-matvec wall time, and
peak RSS (the per-iteration cost and memory that decide feasibility). For
sizes up to --solve-max it also runs the full rough-surface contact solve and
reports its wall time and iteration count; a full solve to tolerance is
impractical at the largest sizes because the projected-CG iteration count
grows with the problem.

Sweep mode spawns one subprocess per size (clean peak RSS):

    python bench_h2_cputime.py [--solve-max 1024] [--max-n 16384]

Single-size worker (used internally, prints one JSON line):

    python bench_h2_cputime.py --grid 1024 [--solve]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NS_ALL = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def self_affine_surface(Ns, H=0.8, rms=0.02, seed=12345, L=1.0):
    import numpy as np
    rng = np.random.default_rng(seed)
    q1d = 2.0 * np.pi * np.fft.fftfreq(Ns, d=L / Ns)
    QX, QY = np.meshgrid(q1d, q1d)
    q = np.hypot(QX, QY)
    q[0, 0] = 1.0
    amp = q ** (-(H + 1.0))
    amp[0, 0] = 0.0
    spec = rng.standard_normal((Ns, Ns)) + 1j * rng.standard_normal((Ns, Ns))
    h = np.fft.ifft2(spec * amp).real
    h -= h.mean()
    h *= rms / h.std()
    return h


def worker(Ns, q, do_solve):
    import resource
    import numpy as np
    sys.path.insert(0, os.path.join(HERE, "python"))
    import hmatrix_contact as hc

    N = Ns * Ns
    t0 = time.perf_counter()
    solver = hc.ContactSolver(grid_size=Ns, domain_size=1.0, E_star=1.0,
                              backend="h2", q=q)
    t_build = time.perf_counter() - t0
    info = solver.hmatrix_info()

    x = np.ones(N)
    solver.matvec(x)  # warm up
    reps = max(2, min(20, 40_000_000 // N))
    t0 = time.perf_counter()
    for _ in range(reps):
        solver.matvec(x)
    t_mv = (time.perf_counter() - t0) / reps

    out = {"Ns": Ns, "N": N, "build_s": t_build, "matvec_s": t_mv,
           "op_bytes": int(info["bytes"]), "solve_s": None, "iters": None}

    if do_solve:
        surface = self_affine_surface(Ns)
        t0 = time.perf_counter()
        res = solver.solve(gap=-surface.ravel(), p_nominal=0.05,
                           tol=1e-8, max_iter=20000)
        out["solve_s"] = time.perf_counter() - t0
        out["iters"] = int(res.iterations)
        out["converged"] = bool(res.converged)

    out["rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("JSON " + json.dumps(out))


def sweep(q, solve_max, max_n):
    rows = []
    for Ns in [n for n in NS_ALL if n <= max_n]:
        cmd = [sys.executable, os.path.abspath(__file__), "--grid", str(Ns),
               "--q", str(q)]
        if Ns <= solve_max:
            cmd.append("--solve")
        print(f"  running Ns={Ns} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("JSON ")),
                    None)
        if line is None:
            print(r.stdout[-500:]); print(r.stderr[-500:]); continue
        rows.append(json.loads(line[5:]))

    print(f"\nH2 CPU-time study (q={q}, threads={os.environ.get('OMP_NUM_THREADS','all')})\n")
    print(f"{'Ns':>6} {'N':>12} {'build[s]':>9} {'matvec[s]':>10} "
          f"{'RSS[GiB]':>9} {'B/DOF':>6} {'solve[s]':>9} {'iters':>6}")
    for d in rows:
        solve = f"{d['solve_s']:9.2f}" if d["solve_s"] is not None else f"{'-':>9}"
        iters = f"{d['iters']:6d}" if d["iters"] is not None else f"{'-':>6}"
        print(f"{d['Ns']:>6} {d['N']:>12} {d['build_s']:>9.3f} {d['matvec_s']:>10.4f} "
              f"{d['rss_kb']/1048576:>9.3f} {d['op_bytes']/d['N']:>6.1f} {solve} {iters}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=0, help="single-size worker")
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--solve-max", type=int, default=1024)
    ap.add_argument("--max-n", type=int, default=16384)
    args = ap.parse_args()
    if args.grid:
        worker(args.grid, args.q, args.solve)
    else:
        sweep(args.q, args.solve_max, args.max_n)
