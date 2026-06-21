"""
visualize_hmatrix_big.py — Two-panel H-matrix block structure plot for large Ns.

Left panel : block type (blue=low-rank, red=dense), outlines, rank numbers.
Right panel: same outlines, blocks shaded by rank value (white=rank-1, dark=high rank).

Saves: doc/slides/figures/fig_hmatrix_blocks_<Ns>[_svd<tol>].pdf
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
              svd_tol=None, figsize=(13, 6)):
    # Use inline SVD during assembly to keep peak RAM at O(compressed) not O(full-rank).
    # For Ns=1024 this is the difference between ~5 GiB and ~28 GiB peak.
    solver = hc.ContactSolver(
        grid_size=Ns, domain_size=1.0, E_star=1.0,
        eta=eta, aca_tol=aca_tol, leaf_size=leaf_size,
        use_hmatrix=True,
        inline_svd_tol=svd_tol if svd_tol is not None else 0.0,
    )
    info = solver.hmatrix_info()

    layout = solver.block_layout()   # (n_blocks, 6): rb, rm, cb, cm, is_dense, rank
    N = Ns * Ns

    ranks_lr = layout[layout[:, 4] == 0, 5]
    max_rank = int(ranks_lr.max()) if len(ranks_lr) else 1
    min_rank = int(ranks_lr.min()) if len(ranks_lr) else 1

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Minimum block side in data units to show a rank label (avoid overplotting)
    min_side_for_label = N / Ns * 1.5   # ~1.5 leaf widths

    for ax_idx, ax in enumerate(axes):
        cmap_lr = plt.get_cmap("Blues")
        cmap_dense = plt.get_cmap("Reds")

        for row in layout:
            rb, rm, cb, cm, is_dense, rank = (
                int(row[0]), int(row[1]), int(row[2]), int(row[3]),
                bool(row[4]), int(row[5])
            )

            if is_dense:
                facecolor = "#d62728"
                edgecolor = "#8b0000"
            else:
                if ax_idx == 0:
                    facecolor = "#1f77b4"
                else:
                    # shade by rank: rank 1 → light, max_rank → dark
                    frac = (rank - min_rank) / max(max_rank - min_rank, 1)
                    facecolor = cmap_lr(0.20 + 0.75 * frac)
                edgecolor = "#0c4a8a"

            rect = patches.Rectangle(
                (cb, rb), cm, rm,
                linewidth=0.25, facecolor=facecolor,
                edgecolor=edgecolor, alpha=0.85)
            ax.add_patch(rect)

            # Rank label for low-rank blocks large enough to hold text
            if (not is_dense and ax_idx == 0
                    and rm >= min_side_for_label and cm >= min_side_for_label):
                ax.text(cb + cm / 2, rb + rm / 2, str(rank),
                        ha="center", va="center",
                        fontsize=max(3, min(7, int(min(rm, cm) / (N / Ns) * 0.7))),
                        color="white", fontweight="bold")

        ax.set_xlim(0, N); ax.set_ylim(N, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("Column (cluster order)", fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel("Row (cluster order)", fontsize=8)

    # Titles
    svd_str = f"  SVD tol={svd_tol}" if svd_tol else ""
    axes[0].set_title(
        f"Block type  $N_s={Ns}$,  $N={N:,}$\n"
        f"{info['n_dense_blocks']} dense (red) + "
        f"{info['n_lowrank_blocks']} low-rank (blue)\n"
        f"avg rank {info['avg_rank']:.2f}  "
        f"compression {info['compression']:.4f}×  "
        f"({info['bytes']/1024**2:.0f} MiB)", fontsize=8)
    axes[1].set_title(
        f"Rank value (shading: light=low, dark=high)\n"
        f"rank range [{min_rank}, {max_rank}]{svd_str}\n"
        f"{info['bytes']/1024**2:.0f} MiB  "
        f"compression {info['compression']:.4f}×", fontsize=8)

    from matplotlib.patches import Patch
    axes[0].legend(
        handles=[Patch(facecolor="#1f77b4", label="Low-rank (admissible)"),
                 Patch(facecolor="#d62728", label="Dense (near-field)")],
        loc="lower right", fontsize=7)

    # Colorbar for right panel
    sm = ScalarMappable(cmap=cmap_lr,
                        norm=Normalize(vmin=min_rank, vmax=max_rank))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("ACA rank k", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"H-matrix structure  $N_s={Ns}$  ($N={N:,}$ DOF)  "
        f"leaf={leaf_size}  η={eta}  ε_ACA={aca_tol:.0e}" + svd_str,
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
