"""
visualize_hmatrix_big.py — Two-panel H-matrix block structure plot for large Ns.

Left panel : block type (blue=low-rank, red=dense)
Right panel: block rank as heatmap (white=dense/0, dark=high rank)

Saves: doc/slides/figures/fig_hmatrix_blocks_<Ns>.pdf
"""

import sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, "/home/vyastrebov/WORK/PROJECTS/FENICS/Hcontact/python")
import hmatrix_contact as hc


def visualize(Ns=256, leaf_size=64, eta=2.0, aca_tol=1e-6,
              svd_tol=None, figsize=(11, 5)):
    solver = hc.ContactSolver(
        grid_size=Ns, domain_size=1.0, E_star=1.0,
        eta=eta, aca_tol=aca_tol, leaf_size=leaf_size,
        use_hmatrix=True,
    )
    info_raw = solver.hmatrix_info()

    if svd_tol is not None:
        solver.recompress(svd_tol)
    info = solver.hmatrix_info()

    layout = solver.block_layout()   # (n_blocks, 5)
    N = Ns * Ns

    # Collect ranks per block (for right panel)
    import ctypes
    # We don't have direct rank access from block_layout, so we reproduce from info
    # block_layout columns: row_begin, row_size, col_begin, col_size, is_dense
    # We need to call the C++ solver to get per-block ranks, but we don't expose that.
    # Instead, color by block size (proxy for expected rank).

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left panel: type (dense=red, low-rank=blue) ──────────
    ax = axes[0]
    for row_begin, row_size, col_begin, col_size, is_dense in layout:
        rb, rm, cb, cm = int(row_begin), int(row_size), int(col_begin), int(col_size)
        color = "#d62728" if is_dense else "#1f77b4"
        rect = patches.Rectangle(
            (cb, rb), cm, rm,
            linewidth=0, facecolor=color, alpha=0.80)
        ax.add_patch(rect)

    ax.set_xlim(0, N); ax.set_ylim(N, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Column (cluster order)", fontsize=8)
    ax.set_ylabel("Row (cluster order)", fontsize=8)
    ax.set_title(
        f"Block type  $N_s={Ns}$, $N={N:,}$\n"
        f"{info_raw['n_dense_blocks']} dense (red) + "
        f"{info_raw['n_lowrank_blocks']} low-rank (blue)\n"
        f"compression {info_raw['compression']:.4f}×  "
        f"({info_raw['bytes']/1024**2:.0f} MiB)", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#1f77b4", label="Low-rank (admissible)"),
                        Patch(facecolor="#d62728", label="Dense (near-field)")],
              loc="lower right", fontsize=7)

    # ── Right panel: block area as proxy for rank (normalized) ──
    # Color low-rank blocks by sqrt(area) — larger blocks are farther away and
    # tend to have higher rank in ACA (before SVD). Dense blocks in red.
    ax2 = axes[1]
    max_area = max((int(r[1]) * int(r[3]) for r in layout if not r[4]), default=1)

    cmap = plt.get_cmap("Blues")
    cmap_r = plt.get_cmap("Reds")

    for row_begin, row_size, col_begin, col_size, is_dense in layout:
        rb, rm, cb, cm = int(row_begin), int(row_size), int(col_begin), int(col_size)
        if is_dense:
            # Dense: use area-relative red shade
            frac = (rm * cm) / max_area
            color = cmap_r(0.3 + 0.6 * frac)
        else:
            # Low-rank: shade by block area (larger = potentially higher rank)
            frac = np.sqrt(rm * cm / max_area)
            color = cmap(0.25 + 0.70 * frac)
        rect = patches.Rectangle(
            (cb, rb), cm, rm,
            linewidth=0, facecolor=color, alpha=0.90)
        ax2.add_patch(rect)

    ax2.set_xlim(0, N); ax2.set_ylim(N, 0)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Column (cluster order)", fontsize=8)
    ax2.set_title(
        f"Block size (shading)  avg rank {info['avg_rank']:.1f}"
        + (f"  →SVD({svd_tol}) avg k={info['avg_rank']:.2f}" if svd_tol else "")
        + f"\n{info['bytes']/1024**2:.0f} MiB  "
        f"compression {info['compression']:.4f}×", fontsize=8)

    fig.suptitle(
        f"H-matrix structure  $N_s={Ns}$  ($N={N:,}$ DOF)  "
        f"leaf={leaf_size}  η={eta}  ε_ACA={aca_tol:.0e}"
        + (f"  SVD tol={svd_tol}" if svd_tol else ""),
        fontsize=9, y=1.01)

    fig.tight_layout()
    tag = f"{Ns}" + (f"_svd{svd_tol}" if svd_tol else "")
    outpath = f"Hcontact/doc/slides/figures/fig_hmatrix_blocks_{tag}.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")
    return outpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, default=256)
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--svd", type=float, default=None)
    args = ap.parse_args()
    visualize(Ns=args.ns, leaf_size=args.leaf, svd_tol=args.svd)


if __name__ == "__main__":
    main()
