"""
visualize_hmatrix.py — Save a colored plot of the H-matrix block structure.

Blue = low-rank (admissible) blocks
Red  = dense (near-field) blocks

Run:
    conda activate fenicsx-env
    cd Hcontact && python visualize_hmatrix.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, "/home/vyastrebov/WORK/PROJECTS/FENICS/Hcontact/python")
import hmatrix_contact as hc

def visualize(Ns=64, leaf_size=32, eta=2.0, aca_tol=1e-6,
              out="doc/slides/figures/fig_hmatrix_blocks.pdf",
              figsize=(6, 6)):
    solver = hc.ContactSolver(
        grid_size=Ns, domain_size=1.0, E_star=1.0,
        eta=eta, aca_tol=aca_tol, leaf_size=leaf_size,
        use_hmatrix=True,
    )
    layout = solver.block_layout()   # (n_blocks, 5)
    info   = solver.hmatrix_info()
    N = Ns * Ns

    fig, ax = plt.subplots(figsize=figsize)

    n_lr = n_d = 0
    for row_begin, row_size, col_begin, col_size, is_dense in layout:
        rb, rm, cb, cm = int(row_begin), int(row_size), int(col_begin), int(col_size)
        color = "#d62728" if is_dense else "#1f77b4"  # red / blue
        alpha = 0.85
        rect = patches.Rectangle(
            (cb, rb), cm, rm,
            linewidth=0.0, edgecolor="none",
            facecolor=color, alpha=alpha,
        )
        ax.add_patch(rect)
        if is_dense:
            n_d += 1
        else:
            n_lr += 1

    ax.set_xlim(0, N)
    ax.set_ylim(N, 0)          # row 0 at top
    ax.set_aspect("equal")
    ax.set_xlabel("Column index (cluster order)")
    ax.set_ylabel("Row index (cluster order)")
    ax.set_title(
        f"H-matrix block structure  $N_s={Ns}$, $N={N}$\n"
        f"{n_d} dense (red), {n_lr} low-rank (blue), "
        f"compression {info['compression']:.3f}×",
        fontsize=9,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Low-rank (admissible)"),
        Patch(facecolor="#d62728", label="Dense (near-field)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  ({n_d} dense + {n_lr} low-rank blocks)")


if __name__ == "__main__":
    import os
    os.makedirs("doc/slides/figures", exist_ok=True)

    # Standard Ns=64 visualization
    visualize(Ns=64, leaf_size=32, out="doc/slides/figures/fig_hmatrix_blocks.pdf")

    # Also produce a larger-grid version for comparison
    visualize(Ns=128, leaf_size=32, out="doc/slides/figures/fig_hmatrix_blocks_128.pdf")
