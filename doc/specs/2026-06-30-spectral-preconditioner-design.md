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
- **Deferred (separate spec/plan):** porting the validated preconditioner into
  C++ `solve_contact` (header-only FFT, e.g. pocketfft, + optional
  preconditioner argument/binding); nested-grid (cascadic/FMG) continuation.
