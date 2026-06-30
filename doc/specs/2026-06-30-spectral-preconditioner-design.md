# Design: Fourier-preconditioned projected CG for faster contact convergence

Date: 2026-06-30
Status: approved (design), Python prototype phase
Related: `src/contact_solver.cpp` (Polonsky–Keer PCG), `doc/theory/pcg.tex`,
`doc/theory/h2_fmm.tex`

## Problem

The Polonsky–Keer projected CG iteration count grows steeply with grid size
on rough surfaces (measured: 90 → 199 → 241 → 906 for Ns = 128 → 1024). Two
causes:

1. **Operator conditioning.** The Boussinesq operator `S` has Fourier symbol
   `Ŝ(q) ∝ 1/|q|`, so `κ(S) ≈ q_max/q_min ∼ N_s` and unpreconditioned CG needs
   `∼ √N_s` iterations — intrinsic to the kernel. (Hertz, a smooth single
   contact, grows mildly 24→44 over Ns 64→256, confirming this term.)
2. **Active-set discovery**, worsened in the example because
   `self_affine_surface` adds higher-frequency content as `N_s` grows (a
   rougher problem at each size).

## Goal

A **spectral (Fourier) preconditioner** for the projected CG that collapses
`κ` from `∼N_s` to `≈O(1)`, giving near grid-independent iteration counts,
validated in a **Python prototype** before any C++ port.

## The preconditioner

`S` is (nearly) diagonalized by the DFT; its eigenvalues are the FFT of its
influence coefficients. The ideal preconditioner `M⁻¹` has symbol `1/Ŝ(q)`:

- **Circulant (parameter-free, preferred):** `w(q) = 1/Ŝ_c(q)` where `Ŝ_c` is
  the FFT of the kernel's influence function, obtained with one
  `solver.matvec(δ)` on a unit impulse (or directly from the kernel table).
  Exact for the periodic part, near-optimal for the non-periodic operator.
- **Asymptotic equivalent:** `w(q) ∝ |q|` (continuous symbol), the fallback.
- `q = 0` mode is **zeroed**: total load / mean is fixed by the constraint and
  the rigid-body shift, not by CG. The overall scale of `M⁻¹` is irrelevant
  (cancels in CG); only the shape matters.

Apply per iteration via FFT: `z = IFFT( w(q) · FFT(r_c) )`, where `r_c` is the
residual masked to the contact set; keep `z` on contact and subtract its mean
over the contact set.

## Change to the Polonsky–Keer PCG (minimal)

Only the direction build changes; the exact line search is untouched. Per
iteration (cf. `contact_solver.cpp`):

- residual/gradient `g = S p + g₀ − α` (rigid-body shift α over contact) — as now
- `z = M⁻¹ g` (FFT preconditioner, contact-masked, mean-zeroed) — **new**
- `β = max(0, ⟨z, g − g_prev⟩) / ⟨z_prev, g_prev⟩` (PR⁺ in the M-inner product) — **modified**
- direction `t = z + β t` on contact, `0` off contact — **modified** (was `g + β t`)
- step `τ = ⟨g, t⟩ / ⟨(S t − mean), t⟩` — **unchanged** (exact line min along `t`)
- project `p ≥ 0`, overlap correction `p_i −= τ g_i` for released penetrating
  nodes, load balance `p ·= P_total/Σp` — **unchanged**

Setting `precond = None` reproduces the current C++ solver exactly (baseline).

## Components (Python prototype, new `experiments/` dir)

1. `precond_symbol(matvec, Ns, L, E_star) -> w` — build `w(q)` (circulant via
   one impulse matvec; `q=0 → 0`). Also a `|q|` analytic variant for reference.
2. `apply_precond(r, contact_mask, w) -> z` — FFT apply (numpy), mask to
   contact, zero the contact-set mean.
3. `pk_pcg(matvec, g0, p_bar, precond=None, tol=1e-8, max_iter) -> result` —
   Python Polonsky–Keer mirroring `contact_solver.cpp`, with the optional
   preconditioner. Returns pressure, gap, iterations, converged, contact
   fraction, error history.
4. `bandlimited_surface(Ns, L, hurst, q0, q1, q2, seed) -> heights` —
   self-affine surface with a **fixed physical cutoff** (q1,q2 in physical
   wavenumbers, independent of Ns) so coarse/fine grids see the same problem.
5. `bench_precond.py` — iterations vs `N_s` (e.g. 64…1024), preconditioned vs
   unpreconditioned; prints a table and (optional) convergence-history plot.

## Testing / success criteria

- **Correctness:** `precond=None` reproduces the C++ iteration counts (±a few)
  on the same surface; the preconditioned solution matches the unpreconditioned
  one (contact area within ~1e-3, pressure rel-L2 ≲ 1e-6 at the same tol).
- **Goal:** preconditioned iteration count is roughly **flat (grid-independent)**
  vs the steep unpreconditioned growth (target O(10–30) across Ns on the rough
  case).

## Scope / non-goals

- **In:** the Python prototype, the benchmark, and a short results note.
- **Done since (in C++):** the `|q|` preconditioner (`precond="fourier"`,
  `fourier_precond.{hpp,cpp}`), the warm-start path (`p_init`), and the
  single-entry nested-grid solve `hc.solve_nested(...)` (`nested_solve.{hpp,cpp}`).

## Status (achieved)

The spectral preconditioner + nested-grid continuation give ~4× fewer
iterations at Ns=1024 (180→45) with the full coarse→fine solve costing less
than one cold solve; iteration counts scale well in practice up to 8192×8192.
Solutions are identical to the unpreconditioned solver (ΔArea 0, rel-L2 ~5e-7).
Remaining cost at the largest grids is wall-clock, not iteration count: a
8192×8192 solve takes ≈17 min, dominated by iterations × O(N) matvec.

## Future direction: monotone multigrid (grid-independent convergence)

The preconditioner conditions the *unconstrained* operator and the nested
continuation supplies a good initial guess, but neither solves the inequality
constraint `p ≥ 0` hierarchically — hence convergence improves but is not
grid-independent. **Monotone multigrid (MMG; Kornhuber 1994)** is a multigrid
built for the variational inequality itself and gives globally convergent,
asymptotically grid-independent rates for the *constrained* problem. Three
mechanisms:

1. **Projected (truncated) Gauss–Seidel smoother** — constrained local updates
   `p_i ← max(0, p_i − r_i/S_ii)` that decrease the energy and resolve the
   contact boundary locally (this is what discovers the active set).
2. **Truncated coarse-grid correction** — coarse basis functions are zeroed
   where the fine constraint is active, so coarse corrections cannot disturb the
   resolved inactive (out-of-contact) set.
3. **Monotone obstacle restriction** — the bound is carried to coarse levels so
   that any admissible coarse correction stays feasible after prolongation;
   combined with the energy-decreasing smoother this gives unconditional global
   convergence.

**Why it is non-trivial here.** MMG is mature for *sparse* FEM contact (local
stiffness matrices, cheap local Gauss–Seidel). Our operator `S` is *dense /
nonlocal* (BEM): a Gauss–Seidel sweep is `O(N²)` and inherently sequential, the
opposite of the cheap, parallel smoother MMG relies on. A dense-BEM contact MMG
is therefore research-grade.

**Pragmatic first cut (if pursued).** A truncated-monotone V-cycle that reuses
the existing nested hierarchy and H2 operators, with a *projected block / Jacobi
smoother restricted to the contact boundary*, residuals applied via the H2
matvec, and the mean-load equality handled as today. This bolts the
constraint-aware coarse correction onto the machinery already built.

**References.** Kornhuber, *Monotone multigrid methods for elliptic variational
inequalities I*, Numer. Math. 69 (1994); Gräser & Kornhuber, *Multigrid methods
for obstacle problems*, J. Comput. Math. 27 (2009).

**Recommendation.** Not planned. The current preconditioner + nested solve
capture most of the available speedup at low cost; pursue MMG only if flat
iteration counts at very large Ns become a hard requirement.
