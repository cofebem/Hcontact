# Design: Matrix-free H2/FMM half-space operator

Date: 2026-06-27
Status: approved (design), implementation pending
Related: `hmatrix_halfspace_revisited_codex.md` (source spec), `doc/theory/pcg.tex`

## Goal

Add a **matrix-free** hierarchical operator for the translation-invariant
Boussinesq half-space flexibility kernel on a uniform `Ns×Ns` grid, exposed
through a whole-operator API (no user-visible blocks), and wired into the
existing Polonsky–Keer PCG `ContactSolver` as an **additive backend**
alongside `dense` and `hmatrix`. The classical ACA H-matrix is retained as a
fallback. Target memory **O(N)** and matvec **O(N)**, enabling `Ns ≥ 1024`.

This replaces the "large list of independently ACA-compressed blocks" design
(which stores redundant low-rank bases per block) with an H2/FMM operator that
**shares bases per cluster and couplings per interaction**, all cached by
`(level, relative offset)` thanks to translation invariance.

## Approach: black-box FMM (Chebyshev interpolation, Fong & Darve 2009)

The source spec's interpolation scheme (§9–14) is black-box FMM: kernel-agnostic,
O(N), and — for a translation-invariant kernel on a uniform tree — every transfer
and coupling operator depends only on `(level, relative offset)` and is cached.

### Kernel: one function everywhere (refines spec §6–7)

Use the existing Love (1929) cell integral `love_uz` (already validated; builds
the current table) as the single kernel function:
`g(x, y) = love_uz(x − y, a=h/2, b=h/2) / (π E*)`.

- **Near field** (touching leaves, `near_radius = 1`): exact per-pair Love values,
  cached as dense stencils keyed by relative leaf offset (spec §12). Reuses the
  current `Ns×Ns` table.
- **Far field**: Chebyshev-interpolate `g` at continuous node offsets. Using `g`
  (not raw `1/r`) keeps far/near consistent — the only far-field error is the
  Chebyshev interpolation error, not a midpoint-vs-exact-integral mismatch.

## Black-box FMM math (reference for implementation)

1D Chebyshev nodes of order `q` on `[-1,1]`:
`ξ_m = cos((2m+1)π/(2q))`, `m = 0..q-1`.
Chebyshev interpolation weight (maps a point `x∈[-1,1]` onto the nodes):
`S_q(x, ξ_m) = 1/q + (2/q) Σ_{n=1}^{q-1} T_n(ξ_m) T_n(x)`, `T_n` = Chebyshev poly.
2D operators are tensor products of the 1D weights; `r = q²` nodes per box.

Passes (each box `b` has multipole coeffs `M_b` and local coeffs `L_b`, size `r`):
- **P2M** (leaf anterpolation): `M_s[a] = Σ_{j∈s} S2D(ŷ_j, ξ_a) x_j` (`ŷ_j` = source
  DOF mapped to box-normalized `[-1,1]²`). Applied as two 1D passes (spec §22).
- **M2M** (up): `M_parent += T_c · M_child`, `T_c` = S2D evaluated at child nodes in
  parent-normalized coords. Depends only on child quadrant `c∈{0..3}` and `q` →
  **4 cached matrices**, reused at all levels (scale-invariant on `[-1,1]`).
- **M2L** (interaction): `L_t[a] += Σ_b K_ts[a,b] M_s[b]`,
  `K_ts[a,b] = g(ξ_a^t − ξ_b^s)` at physical node coords. Depends on
  `(level, dx_box, dy_box, q)` → cached (spec §13).
- **L2L** (down): `L_child += T_c^T · L_parent`.
- **L2P** (leaf eval): `y_i += Σ_a S2D(x̂_i, ξ_a) L_t[a]`.
- **Near**: `y_t += A_near(t,s) x_s` for the ≤9 leaves in the `near_radius=1`
  neighborhood, via cached dense stencils.

## Components (focused, independently testable)

1. `UniformQuadTree` — `Box{level, parent, child[4], ix0,ix1,iy0,iy1, cx,cy, half, leaf}`.
   **Index ranges, not index lists** (spec §4, §21). Built top-down to a
   power-of-two leaf side.
2. `ChebBasis(q)` — 1D nodes + interpolation matrices; tensor-product 2D apply via
   two 1D passes; never forms the Kronecker product (spec §22).
3. Caches (translation invariance): 4 M2M + 4 L2L; M2L by `FarCouplingKey{level,dx,dy,q}`;
   near stencils by `NearStencilKey{leaf_nx,leaf_ny,dx,dy}` (spec §13–14).
4. `H2Operator` — owns tree + caches + interaction lists; `build()` and `matvec(x)`
   running `P2M → M2M↑ → M2L → L2L↓ → L2P → near`. OpenMP over boxes/leaves.
5. Interaction lists (spec §11): children of parent's neighbors minus near neighbors;
   bounded constant size; `near_radius = 1`.
6. `print_statistics()` — full profiling block (spec §29): box/level counts, leaf size,
   `q`, `r`, near/far interaction counts, unique coupling/stencil counts, per-cache
   memory, build + matvec time, per-pass timings.
7. Bindings + `ContactSolver` `backend` enum `{DENSE, HMATRIX, H2}`. PCG already takes a
   `MatVec` functor, so `H2Operator::matvec` plugs in with **no change to existing
   dense/hmatrix paths or tests** (additive).

## Public API (target)

```cpp
H2OperatorParams p;            // leaf_side=8, q=4, near_radius=1, caches on, symmetry on
H2Operator A(kernel, grid, p);
A.build();
A.print_statistics();
Eigen::VectorXd y = A.matvec(x);
```

No `build_block/assemble_block/compress_block`; no user-side block loop;
no full dense assembly.

## Parameters

Defaults: `leaf_side = 8` (64 dof/leaf), `q = 4` (r=16), `near_radius = 1`,
symmetry + caches on. Sweeps: `q∈{4,5,6,8}`, `leaf∈{8,16}`, `near_radius∈{1,2}`.

## Testing (TDD, validated against the existing dense path)

- Unit: Chebyshev interpolation reproduces degree `<q` polynomials exactly;
  tensor-product apply equals explicit Kronecker on a small case; M2L cache equals
  direct node evaluation; M2M/L2L round-trip on a smooth field.
- Operator accuracy: `‖A·x − S_dense·x‖ / ‖S_dense·x‖ < tol`, decreasing as `q↑`
  (≈1e-3 at q=4 → ≈1e-5 at q=6).
- Integration: Hertz via H2 reproduces `a_num/a ≈ 1.016`, `p_max/p0 ≈ 0.998`;
  rough-surface `Ac/A` matches the H-matrix backend within tolerance.
- Scaling: build/matvec time + memory vs `Ns` (64…1024) vs H-matrix →
  demonstrate **O(N) memory** and near-O(N) matvec.

## Non-goals (YAGNI)

- No FFT backend (separate, deferred).
- No active-domain/masking sparsity yet — the operator runs on the full grid and
  the PCG projection handles the constraint as today. The H2 structure *enables*
  masking later; we do not build it now.
- Square grids only in v1 (`leaf_nx = leaf_ny`); design leaves room for `nx ≠ ny`.
- No H-LU / H-Cholesky.

## Integration & rollout

Additive: existing `dense` and `hmatrix` backends and all current ctests stay green.
New `H2` backend selected via `ContactSolver(backend=...)`. New ctest `test_h2`
covers unit + operator-vs-dense accuracy; a Python scaling bench compares H2 vs
H-matrix memory/time.
