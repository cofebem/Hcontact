# Beamer Slides Design: H-Matrix BEM for Rough-Surface Normal Contact

## Goal

17-slide LaTeX Beamer conference talk. Section order: H-Matrix solver first,
then Hertz validation, then rough contact, then performance. Clean white
scientific style (Nature/JMPS aesthetic, viridis/RdBu colourmaps).

## File layout

```
Hcontact/doc/slides/
├── slides.tex          # master Beamer file, \input each section
├── sections/
│   ├── hmatrix.tex     # §1  H-Matrix solver (5 slides)
│   ├── hertz.tex       # §2  Hertz Problem (4 slides)
│   ├── rough.tex       # §3  Rough Surface Contact (4 slides)
│   └── performance.tex # §4  Performance (3 + conclusion slides)
├── generate_figures.py # produces all PDF figures into figures/
└── figures/            # output directory (gitignored except *.pdf)
```

## Beamer setup

- Theme: `metropolis` (clean white, no navigation bar)
- Font: `lmodern`, math in `amsmath`/`amssymb`
- Colours: accent `#2c7bb6` (mid blue), alert `#d73027` (red)
- Packages: `tikz` (with `calc`, `arrows.meta`, `shapes`), `pgfplots`,
  `booktabs`, `siunitx`, `mhchem` (not needed but keep for generality)
- Figures: `\includegraphics` of PDF outputs from `generate_figures.py`

## Section §1 — H-Matrix Solver (5 slides)

**Slide 1: The N² wall**
- Cost table: dense assembly O(N²), matvec O(N²), memory 8N² bytes
- Concrete numbers: N=256² → 34 GB just to store S; infeasible
- Motivation for H-matrix

**Slide 2: Cluster tree (quad-tree)**
- TikZ: 4×4 toy grid subdivided recursively into quadrants; leaf boxes
  coloured by depth; show `perm[]` reordering arrows
- Text: depth ≤ log₄N, each leaf ≤ `leaf_size` elements

**Slide 3: Admissibility and block partition**
- TikZ: N×N matrix painted as coloured blocks (blue = low-rank, grey = dense)
- Admissibility criterion: min(diam(t), diam(s)) ≤ η · dist(t, s)
- ACA: rank-k factorisation S_{IJ} ≈ UV, Frobenius stopping ‖u_k‖‖v_k‖ ≤ ε‖A_k‖_F

**Slide 4: H-matrix matvec**
- TikZ: block list with arrows u(row) += D·x (dense) or U·(V·x) (low-rank)
- OpenMP parallel loop over blocks
- Complexity: O(N log²N) assembly, O(N log N) matvec

**Slide 5: Boussinesq kernel — the Love formula**
- Integral operator u_z(x) = ∫ G(x−x') p(x') dA'
- Green's function G(r) = 1/(π E* |r|)
- Love (1929) exact element integral L(Δx, Δy, a, b) — formula on slide
- Self-term: S_ii = 4h ln(1+√2) / (π E*)
- Matplotlib: S_ij vs distance r/h showing 1/r asymptote

## Section §2 — Hertz Problem (4 slides)

**Slide 6: Geometry & physics**
- TikZ: cross-section of sphere (radius R) pressing on elastic half-space;
  contact zone of radius a; Hertz parabolic gap g(r) = r²/(2R)
- Labels: E*, ν, F (total load), a (contact radius), p(r) pressure

**Slide 7: Analytical solution**
- Contact radius: a = (3FR / 4E*)^{1/3}
- Pressure: p(r) = p₀ √(1 − (r/a)²), p₀ = 3F/(2πa²)
- Indentation: δ = a²/R
- Brief derivation note (Boussinesq + boundary conditions)

**Slide 8: BEM discretisation**
- Grid Ns×Ns, element size h = L/Ns
- Matrix entry S_ij = L(x_i − x_j, h/2, h/2) / (π E*)
- Polonsky–Keer projected CG: pseudocode block (7 lines)

**Slide 9: Hertz validation**
- Matplotlib 2-panel:
  - Left: computed pressure p(r) vs analytic Hertz parabola; error bar
  - Right: contact area circle overlay on pressure map (2D colour plot)
- Numbers from test_contact.cpp: a_num/a = 1.016, p_max/p₀ = 0.998

## Section §3 — Rough Surface Contact (4 slides)

**Slide 10: Real surfaces are rough**
- TikZ: multi-scale roughness sketch (sine waves at 3 scales)
- Power spectral density C(q) ~ q^{−2(1+H)}, H = Hurst exponent
- Matplotlib: 3D surface plot of the benchmark surface (surface.npy)

**Slide 11: The contact problem as a QP**
- Objective: W(p) = ½ pᵀSp + pᵀg₀
- Constraints: p_i ≥ 0, mean(p) = p̄
- KKT: (Sp + g₀)_i ≥ 0, p_i ≥ 0, p_i(Sp + g₀)_i = 0
- Physical meaning: non-negative gap, non-negative pressure, complementarity

**Slide 12: Polonsky–Keer algorithm (full)**
- Full pseudocode as `algorithm` block (or Beamer `block` environment)
- Active-set projection, overlap correction, normalisation step
- Convergence: complementarity error Σ p|g| / (N p̄ g_scale) < ε

**Slide 13: Rough contact result**
- Matplotlib 3-panel from real data (surface.npy + our solver):
  - Surface height field (viridis)
  - Pressure field (RdBu, zero = white)
  - Gap field (non-contact = grey, contact = coloured)
- Stats: contact fraction 0.126, mean pressure 0.05, converged in 26 iters

## Section §4 — Performance (4 slides)

**Slide 14: Memory scaling**
- Matplotlib: storage (MiB) vs N on log-log axes
  - Dense: 8N² (line)
  - H-matrix: measured at N = 1024, 4096, 16384, 65536 (points + fitted line)
- Annotate: compression ratio 0.31 at N=4096 (40 MiB vs 128 MiB dense)

**Slide 15: Timing scaling**
- Matplotlib: assembly time and matvec time vs N on log-log axes
  - Fitted slopes labelled (target O(N log²N))
  - Measured: assembly 0.007s (N=4096), matvec 0.62ms

**Slide 16: Comparison with Tamaas periodic**
- Bar chart: solve time (our BEM vs Tamaas PKR, same surface, p̄=0.05)
- Table: contact fraction (0.126 vs 0.127), pressure L2 diff (3.3%),
  mean pressure error (< 1e-14), iterations (26 vs 33)
- Note: Tamaas uses periodic Westergaard operator (FFT); our BEM is
  non-periodic with exact Love coefficients

**Slide 17: Conclusions**
- Bullet summary: Love kernel + H-matrix + PCG → O(N log²N) non-periodic contact
- Compression 0.31, 26 iterations, Hertz validated to 0.2%
- Outlook: larger grids (N > 10⁶ via MPI), plasticity, adhesion (JKR)
- Acknowledgements

## Figure generation (generate_figures.py)

All figures output as PDF to `figures/`. Script reads real data from
`../data/surface.npy` and `../data/tamaas_pressure.npy`; imports
`../python/hmatrix_contact` to generate timing/memory sweeps.

Figures to produce:
- `fig_kernel_decay.pdf`      — S_ij vs r/h (§1)
- `fig_hmatrix_blocks.pdf`    — block structure visualisation (§1, matplotlib)
- `fig_hertz_validation.pdf`  — pressure profile + contact map (§2)
- `fig_surface_3d.pdf`        — 3D surface height (§3)
- `fig_rough_result.pdf`      — 3-panel pressure/gap (§3)
- `fig_memory_scaling.pdf`    — memory vs N (§4)
- `fig_timing_scaling.pdf`    — timing vs N (§4)
- `fig_tamaas_comparison.pdf` — bar + table comparison (§4)

TikZ diagrams are drawn inline in the `.tex` files (no external files).

## Build

```bash
cd Hcontact/doc/slides
conda activate fenicsx-env
python generate_figures.py          # writes figures/*.pdf
pdflatex slides.tex && pdflatex slides.tex   # twice for references
```

Requires: `texlive-full` (or at minimum `beamer`, `metropolis`, `tikz`,
`pgfplots`, `booktabs`, `siunitx`). Check: `kpsewhich beamerthememetropolis.sty`.
