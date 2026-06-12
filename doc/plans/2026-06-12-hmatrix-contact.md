# H-Matrix BEM Rough-Contact Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** C++17 library + pybind11 module `hmatrix_contact` implementing a Boussinesq BEM normal-contact solver (Polonsky–Keer PCG) backed by an H-matrix (quad-tree + ACA), validated against Tamaas.

**Architecture:** A translation-invariant Love-formula kernel feeds both a dense reference matrix and an H-matrix (quad-tree cluster tree in 2D index space, Chebyshev admissibility, partially pivoted ACA low-rank blocks, dense near-field blocks, OpenMP block-parallel matvec). A Polonsky–Keer projected CG solver runs on top of either operator. pybind11 exposes `ContactSolver`/`ContactResult`.

**Tech Stack:** C++17, Eigen 3.4 (from `fenicsx-env` conda env), OpenMP, pybind11 (cmake dir from `dolfinx-010` env, header-only, target interpreter = `fenicsx-env` Python 3.12), CTest, Tamaas 2.8.1 (in `fluidpaper` env) for reference.

**Spec:** `../../HMATRIX_CONTACT_PROMPT.md` (all formulas referenced below live there).

**Environment facts (probed 2026-06-12):**
- conda base: `/home/users02/vyastrebov/DISTR/miniconda3`
- Eigen3 cmake config: `$ENVS/fenicsx-env/share/eigen3/cmake`
- pybind11 cmake dir: `$ENVS/dolfinx-010/lib/python3.12/site-packages/pybind11/share/cmake/pybind11`
- `fenicsx-env` python: 3.12.3, numpy 1.26.4, scipy 1.12.0; `fluidpaper` has tamaas 2.8.1
- g++ 11.4, cmake 3.29, 20 cores

---

## File Structure (all under `Hcontact/`)

```
Hcontact/
├── CMakeLists.txt              # static lib + pybind module + ctest
├── include/
│   ├── boussinesq_kernel.hpp   # Love formula, O(1) table-lookup entries
│   ├── cluster_tree.hpp        # quad-tree over Ns×Ns index grid + permutation
│   ├── hmatrix.hpp             # Block (dense | UV), ACA fill, OpenMP matvec
│   └── contact_solver.hpp      # PolonskyKeer PCG, ContactResult
├── src/{boussinesq_kernel,cluster_tree,hmatrix,contact_solver}.cpp
├── python/bindings.cpp         # module hmatrix_contact; .so output -> python/
├── tests/
│   ├── test_kernel.cpp         # self-term, symmetry, far-field asymptotics
│   ├── test_hmatrix.cpp        # H-matvec vs dense matvec, compression < 1
│   └── test_contact.cpp        # Hertz sphere vs analytic a, p0
├── compare_tamaas.py           # benchmark vs Tamaas (subprocess into fluidpaper)
├── tamaas_reference.py         # runs inside fluidpaper, dumps .npy + timings
├── doc/plans/                  # this plan
└── README.md
```

Design decisions locked in:
- **Kernel table:** `S_ij` depends only on `(|ix-jx|, |iy-jy|)`; precompute an `Ns×Ns` lookup table once → ACA/dense entry evaluation is an O(1) load. Love formula used for ALL entries (spec §2.2).
- **Cluster ordering:** the quad-tree induces a permutation `perm[cluster_pos] = flat_index`; every block addresses contiguous ranges in permuted space; matvec permutes in/out.
- **ACA:** partially pivoted with incremental Frobenius-norm stopping `‖u_k‖‖v_k‖ ≤ ε‖A_k‖_F`, row-restart on zero pivot, rank cap `min(m,n)`.
- **PCG:** full Polonsky–Keer (1999) including the overlap correction step (`p=0 & g<0` points get `p -= τ g`) — the spec pseudocode is a simplification of this; the paper version is what Tamaas implements and converges robustly.
- **C++ tests:** plain `assert`-style executables registered with CTest (no gtest dep).

---

### Task 1: Scaffold, CMake, Boussinesq kernel + kernel unit test

**Files:** `CMakeLists.txt`, `include/boussinesq_kernel.hpp`, `src/boussinesq_kernel.cpp`, `tests/test_kernel.cpp`

- [x] Write `boussinesq_kernel.{hpp,cpp}`: free function
  `double love_uz(double x, double y, double a, double b)` implementing spec eq. L(x,y,a,b)
  (4 log terms with R±± = sqrt((x±a)²+(y±b)²)); class `BoussinesqKernel(Ns, L, E_star)`
  precomputing `table[dy*Ns+dx] = love_uz(dx*h, dy*h, h/2, h/2)/(π E*)`, with
  `entry(i,j)` = table lookup and `assemble_dense()` helper.
- [x] Write `tests/test_kernel.cpp`:
  - self-term: `entry(i,i) == 4h·ln(1+√2)/(π E*)` to 1e-14 rel
  - symmetry: `entry(i,j) == entry(j,i)` for random pairs
  - far field: `|entry/point_source − 1| < 0.002` at r ≥ 5h; ≈3.7% at r=h (sanity of spec claim)
  - positivity and monotone decay along a row
- [x] Write `CMakeLists.txt`: C++17, `-O3 -march=native -Wall -Wextra`, appends
  `$ENV{CONDA_PREFIX}` to `CMAKE_PREFIX_PATH`, `find_package(Eigen3 REQUIRED)`,
  `find_package(OpenMP)`, `find_package(pybind11)` (QUIET at this stage), static lib
  `hmatrix_contact_core`, `enable_testing()`, test exe + `add_test`.
- [x] Configure + build + `ctest` green:
  `cmake -S Hcontact -B Hcontact/build -DCMAKE_BUILD_TYPE=Release` (inside fenicsx-env)
- [x] Commit `feat(hcontact): Boussinesq Love kernel + tests`

### Task 2: Quad-tree cluster tree

**Files:** `include/cluster_tree.hpp`, `src/cluster_tree.cpp`, asserts appended to `tests/test_hmatrix.cpp` (tree-only part first)

- [x] `ClusterTree(Ns, leaf_size)`: nodes `{begin,end, box{ix0,ix1,iy0,iy1}, children[4]}` over
  index space; recursive midpoint split (skip empty quadrants); permutation arrays
  `perm` / `iperm`; `diam()` and `dist()` in Chebyshev index-metric scaled by h.
- [x] Test: perm is a permutation of 0..N-1; leaves ≤ leaf_size; leaf boxes tile the grid;
  child ranges partition parent range.
- [x] Build + run, commit `feat(hcontact): quad-tree cluster tree`

### Task 3: H-matrix (admissibility + ACA + matvec) vs dense

**Files:** `include/hmatrix.hpp`, `src/hmatrix.cpp`, `tests/test_hmatrix.cpp`

- [x] `HMatrix(kernel, tree, eta, aca_tol)`: recursive block partition; admissible
  (`min(diam) ≤ η·dist`, dist>0) → ACA block (U,V); both-leaves → dense block; else recurse.
  ACA as locked in above. Store blocks in flat `std::vector<Block>`.
- [x] `matvec(p)`: permute → per-block GEMV (`U*(V*x)` for low-rank) → inverse permute.
  Serial first; OpenMP comes in Task 6.
- [x] `info()`: counts, ranks, compressed bytes vs `8N²` dense bytes → compression ratio.
- [x] Test (Ns=64, N=4096): rel. L2 error `‖Hp − Sp‖/‖Sp‖ < 1e-5` for 5 random vectors at
  `aca_tol=1e-6`; compression ratio < 0.35; also Ns=32 exactness when eta makes all dense.
- [x] Build + run, commit `feat(hcontact): ACA H-matrix with verified matvec`

### Task 4: Polonsky–Keer PCG contact solver + Hertz test

**Files:** `include/contact_solver.hpp`, `src/contact_solver.cpp`, `tests/test_contact.cpp`

- [x] `LinearOperator` interface (dense | hmatrix). `solve_contact(op, g0, p_bar, tol, max_iter)`
  implementing P&K: contact set `p>0`, gap shift by contact mean, conjugation factor `δ·G/G_old`
  (δ=0 when overlap correction fired), line search `τ = Σ g t / Σ r t`, projection `p≥0`,
  overlap correction, normalization `p *= p̄N/Σp`, error `Σ p|g| / (N p̄ · g_scale)`.
- [x] `ContactResult{pressure, displacement, objective W = ½pᵀu + pᵀg0, iterations, error,
  contact_fraction, mean_pressure}`.
- [x] Hertz test (Ns=64, L=1, E*=1, R=2, p̄ chosen so contact diameter ≈ L/3):
  `a_num` from contact area vs `a = (3FR/(4E*))^(1/3)` within 5%; `p_max` vs
  `p0 = 3F/(2πa²)` within 5%; mean pressure == p̄ to 1e-10; complementarity residual small.
  Run with both dense and H-matrix operators, fields agree < 1e-4 rel L2.
- [x] Build + run, commit `feat(hcontact): Polonsky-Keer PCG solver, Hertz validated`

### Task 5: pybind11 bindings

**Files:** `python/bindings.cpp`, CMake additions

- [x] Module `hmatrix_contact`: `ContactSolver(grid_size, domain_size, E_star, eta, aca_tol,
  leaf_size)` (+kwarg `use_hmatrix=True`), `.matvec(np 1D/2D)`, `.solve(gap, p_nominal, tol,
  max_iter) -> ContactResult` with `(Ns,Ns)` numpy `pressure`/`displacement`, scalars
  `objective`, `contact_area`, `mean_pressure`, `iterations`, `error`; `.hmatrix_info()`.
- [x] CMake: `pybind11 REQUIRED` with documented `-Dpybind11_DIR`, module output dir
  `${CMAKE_SOURCE_DIR}/python`.
- [x] Smoke test from fenicsx-env python: matvec vs kernel dense via numpy; tiny solve runs.
- [x] Commit `feat(hcontact): pybind11 module hmatrix_contact`

### Task 6: OpenMP matvec + diagnostics polish

- [x] Parallelize block loop: `#pragma omp parallel` with thread-local accumulator, reduce.
  Re-run test_hmatrix + timing print (matvec µs dense vs H).
- [x] Commit `perf(hcontact): OpenMP H-matvec`

### Task 7: compare_tamaas.py

**Files:** `tamaas_reference.py`, `compare_tamaas.py`

- [x] `tamaas_reference.py` (runs under fluidpaper): builds the spec's self-affine surface
  (n=64, hurst=0.8, k0=1,k1=4,k2=32, seed=12345, rms slope 1), non-periodic dcfft model,
  PolonskyKeerRey at p̄=0.05, saves `surface.npy`, `tamaas_pressure.npy`,
  `tamaas_meta.json` (timings, contact fraction).
- [x] `compare_tamaas.py` (runs under fenicsx-env): invokes
  `conda run -n fluidpaper python tamaas_reference.py` (skips if outputs fresh), runs
  hmatrix_contact on same surface/pressure, reports contact fractions, mean-pressure errors,
  rel-L2 pressure diff, assembly/matvec/solve wall times, compression ratio.
  Asserts: rel-L2 < 2%, |Δ contact fraction| < 0.01.
- [x] Run it; iterate until asserts pass. Commit `feat(hcontact): Tamaas comparison benchmark`

### Task 8: README + final verification

- [x] `README.md`: 2–3 build commands, example usage (spec §Python Interface), test & benchmark
  instructions, known env facts (pybind11 via dolfinx-010).
- [x] Full clean rebuild, `ctest`, python smoke, compare_tamaas — all green; verify spec
  Deliverables Checklist; commit `docs(hcontact): README + final validation`.

---

## Self-Review Notes

- Spec coverage: kernel §2.2→T1, tree §4.1→T2, admissibility/ACA/matvec §4.2–4.5→T3+T6,
  solver §3→T4, python §6→T5, comparison §7→T7, deliverables §10→T8. Covered.
- Spec's `compare_tamaas.py` asks for surface via cross-env subprocess — handled by
  `tamaas_reference.py` + `conda run`.
- Deviation from spec recorded: full P&K with overlap correction instead of the abbreviated
  pseudocode (rationale in "Design decisions"); module dir is `Hcontact/` not
  `hmatrix_contact/` per user instruction.
