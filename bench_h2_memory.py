"""Memory consumption of the matrix-free H2 operator vs grid size.

Builds the H2 operator (no solve) across grid sizes and reports its intrinsic
storage broken into the three caches (coupling / near / work buffers), the
per-DOF cost, and the equivalent dense matrix size N^2*8 (which is never
formed).  Demonstrates O(N) memory: bytes/DOF stays bounded as N grows.

Run (fenicsx-env):  python bench_h2_memory.py [--plot]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
import numpy as np
import hmatrix_contact as hc

NS = [64, 128, 256, 512, 1024, 2048]
Q = 6
MiB = 1024.0 * 1024.0


def run():
    rows = []
    print(f"q={Q}, leaf_side=8, near_radius=1\n")
    print(f"{'Ns':>6} {'N':>11} {'coupling':>9} {'near':>8} {'buffers':>9} "
          f"{'total':>9} {'B/DOF':>7} {'dense':>12}")
    print(f"{'':>6} {'':>11} {'[MiB]':>9} {'[MiB]':>8} {'[MiB]':>9} "
          f"{'[MiB]':>9} {'':>7} {'[MiB]':>12}")
    for Ns in NS:
        N = Ns * Ns
        s = hc.ContactSolver(grid_size=Ns, backend="h2", q=Q)
        info = s.hmatrix_info()
        tot = info["bytes"]
        rows.append((N, tot))
        dense_mib = 8.0 * N * N / MiB
        print(f"{Ns:>6} {N:>11} "
              f"{info['bytes_coupling']/MiB:>9.2f} {info['bytes_near']/MiB:>8.2f} "
              f"{info['bytes_buffers']/MiB:>9.2f} {tot/MiB:>9.2f} "
              f"{tot/N:>7.1f} {dense_mib:>12.1f}")

    # O(N) check: bytes/DOF bounded (asymptotically flat)
    bpd = [tot / N for N, tot in rows]
    print(f"\nbytes/DOF range over Ns: {min(bpd):.1f} .. {max(bpd):.1f} "
          f"(bounded => O(N) memory)")
    return rows


def plot(rows):
    import matplotlib.pyplot as plt
    N = np.array([r[0] for r in rows], float)
    tot = np.array([r[1] for r in rows], float) / MiB
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(N, tot, "o-", label="H2 operator (measured)")
    ax.loglog(N, 8.0 * N * N / MiB, "--", color="gray", label="dense $N^2$ (never formed)")
    ax.loglog(N, tot[0] * N / N[0], ":", color="C1", label="$O(N)$ reference")
    ax.set_xlabel("N = Ns$^2$ (DOFs)")
    ax.set_ylabel("memory [MiB]")
    ax.legend(frameon=False)
    ax.set_title("H2 operator memory scaling")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "bench_h2_memory.png")
    fig.savefig(out, dpi=130)
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    rows = run()
    if args.plot:
        plot(rows)
