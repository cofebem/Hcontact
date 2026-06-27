# H2/FMM Half-Space Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a matrix-free black-box FMM (Chebyshev-interpolation H2) operator for the Boussinesq half-space kernel, wired into the existing PCG solver as an additive `H2` backend with O(N) memory.

**Architecture:** Uniform quad-tree over the `Ns×Ns` grid (index ranges, no index lists). Far field via tensor-product Chebyshev interpolation with multilevel P2M→M2M→M2L→L2L→L2P passes; near field via exact Love stencils. All transfer/coupling/stencil operators are cached by `(level, relative offset)` thanks to translation invariance. Exposed as a whole-operator API; PCG plugs into `H2Operator::matvec`.

**Tech Stack:** C++17, Eigen 3.4, OpenMP, pybind11 (built with `/usr/bin/g++`, `pybind11_DIR` from `dolfinx-010`). Tests are plain executables using the existing `CHECK` macro idiom; ctest runner.

## Global Constraints

- C++17; headers in `include/`, sources in `src/`, tests in `tests/`. Namespace `hmc`.
- Build: `conda activate fenicsx-env`; `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ -Dpybind11_DIR=$(conda run -n dolfinx-010 python -m pybind11 --cmakedir)`; `cmake --build build -j$(nproc)`.
- Test: `ctest --test-dir build --output-on-failure`.
- `Ns` and `leaf_side` are powers of two; `leaf_side | Ns`. Square grids only (`leaf_nx = leaf_ny = leaf_side`).
- Element `(ix,iy)` has physical center `(ix*h, iy*h)`, `h = L/Ns`. Box geometric extent uses cell edges: a box with element range `[ix0, ix0+side)` spans physical `[(ix0-0.5)h, (ix0+side-0.5)h]`, width `side*h`, center `(ix0+side/2-0.5)h`. (Exact half-scaling between levels → scale-invariant M2M/L2L.)
- Far kernel: `g(dx,dy) = love_uz(dx, dy, h/2, h/2) / (π E*)` (same function as the table; consistent near/far).
- Additive only: do not change `dense`/`hmatrix` code paths or existing tests; all three ctests stay green.
- Commit after each task. Co-author trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

- Create `include/cheb_basis.hpp`, `src/cheb_basis.cpp` — Chebyshev nodes + 1D interpolation weights + tensor apply.
- Create `include/uniform_quadtree.hpp`, `src/uniform_quadtree.cpp` — uniform box tree, neighbors, interaction lists.
- Create `include/h2_operator.hpp`, `src/h2_operator.cpp` — caches, build, matvec passes, statistics.
- Create `tests/test_h2.cpp` — unit + accuracy tests.
- Modify `CMakeLists.txt` — add sources to core lib; add `h2` to the test loop.
- Modify `python/bindings.cpp` — add `backend` arg and `H2` path to `PyContactSolver`.
- Create `bench_h2.py` — H2 vs H-matrix scaling bench.

---

### Task 1: ChebBasis — Chebyshev nodes, 1D weights, tensor apply

**Files:**
- Create: `include/cheb_basis.hpp`, `src/cheb_basis.cpp`
- Test: `tests/test_h2.cpp`

**Interfaces:**
- Produces:
  - `struct ChebBasis { int q; std::vector<double> nodes; ... }`
  - `explicit ChebBasis(int q);` — `nodes[m] = cos((2m+1)π/(2q))`, m=0..q-1, in `[-1,1]`.
  - `Eigen::VectorXd weights(double t) const;` — length-`q` vector `w_m(t) = 1/q + (2/q) Σ_{n=1}^{q-1} T_n(nodes[m]) T_n(t)`.
  - `Eigen::MatrixXd weights_at(const std::vector<double>& pts) const;` — `(pts.size() × q)`, row `i` = `weights(pts[i])^T`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_h2.cpp`; create file with the `CHECK` macro from `test_hmatrix.cpp`)

```cpp
#include "cheb_basis.hpp"
#include "uniform_quadtree.hpp"
#include "h2_operator.hpp"
#include "boussinesq_kernel.hpp"
#include <cstdio>
#include <cmath>

#define CHECK(cond) do { if(!(cond)){ std::printf("FAILED: %s (line %d)\n", #cond, __LINE__); return 1; } } while(0)

static int test_cheb() {
    hmc::ChebBasis b(5);
    CHECK((int)b.nodes.size() == 5);
    // partition of unity: weights sum to 1 at any t
    for (double t : {-0.9, -0.3, 0.0, 0.4, 1.0}) {
        double s = b.weights(t).sum();
        CHECK(std::abs(s - 1.0) < 1e-12);
    }
    // interpolation exactness for polynomials of degree < q
    auto f = [](double x){ return 3 - 2*x + 0.5*x*x - x*x*x + 0.25*x*x*x*x; };
    Eigen::VectorXd fnodes(5);
    for (int m = 0; m < 5; ++m) fnodes(m) = f(b.nodes[m]);
    for (double t : {-0.7, 0.1, 0.85}) {
        double interp = b.weights(t).dot(fnodes);
        CHECK(std::abs(interp - f(t)) < 1e-10);
    }
    return 0;
}
```

- [ ] **Step 2: Run to verify it fails** — add `test_h2` to CMake first (see Task 1 Step 4). `ctest --test-dir build -R h2` → FAIL (compile error: no `cheb_basis.hpp`).

- [ ] **Step 3: Implement `cheb_basis.hpp`/`.cpp`**

```cpp
// cheb_basis.hpp
#pragma once
#include <Eigen/Dense>
#include <vector>
namespace hmc {
struct ChebBasis {
    int q;
    std::vector<double> nodes;            // q Chebyshev points in [-1,1]
    explicit ChebBasis(int q);
    Eigen::VectorXd weights(double t) const;                       // length q
    Eigen::MatrixXd weights_at(const std::vector<double>& pts) const; // (P x q)
};
} // namespace hmc
```

```cpp
// cheb_basis.cpp
#include "cheb_basis.hpp"
#include <cmath>
namespace hmc {
ChebBasis::ChebBasis(int q_) : q(q_), nodes(q_) {
    for (int m = 0; m < q; ++m)
        nodes[m] = std::cos((2.0*m + 1.0) * M_PI / (2.0*q));
}
Eigen::VectorXd ChebBasis::weights(double t) const {
    // w_m = 1/q + (2/q) sum_{n=1}^{q-1} T_n(node_m) T_n(t)
    Eigen::VectorXd Tt(q);            // T_n(t)
    Eigen::VectorXd w = Eigen::VectorXd::Constant(q, 1.0 / q);
    for (int m = 0; m < q; ++m) {
        double xm = nodes[m];
        // recurrence for T_n(xm) and T_n(t)
        double T0m = 1, T1m = xm, T0t = 1, T1t = t;
        double acc = 0.0;
        for (int n = 1; n < q; ++n) {
            double Tnm = (n == 1) ? T1m : 2*xm*T1m - T0m;
            double Tnt = (n == 1) ? T1t : 2*t *T1t - T0t;
            acc += Tnm * Tnt;
            if (n >= 2) { T0m = T1m; T1m = Tnm; T0t = T1t; T1t = Tnt; }
        }
        w(m) += (2.0 / q) * acc;
    }
    return w;
}
Eigen::MatrixXd ChebBasis::weights_at(const std::vector<double>& pts) const {
    Eigen::MatrixXd W((int)pts.size(), q);
    for (int i = 0; i < (int)pts.size(); ++i) W.row(i) = weights(pts[i]).transpose();
    return W;
}
} // namespace hmc
```

- [ ] **Step 4: Add `cheb_basis.cpp` to core lib and `test_h2` to CMake**

In `CMakeLists.txt`, add `src/cheb_basis.cpp` (and later `src/uniform_quadtree.cpp`, `src/h2_operator.cpp`) to `add_library(hmatrix_contact_core ...)`, and change the test loop to `foreach(t kernel hmatrix contact h2)`. (Add the other two sources now to avoid re-editing; create empty stubs if needed — but prefer adding them as each task lands. For Task 1, add only `cheb_basis.cpp` and `h2` test; `test_h2.cpp` must compile, so include only `cheb_basis.hpp` until Tasks 2–3 land, OR gate later includes. Simplest: write Task-1 `test_h2.cpp` including only `cheb_basis.hpp` and `main(){ return test_cheb(); }`, then expand includes/`main` in later tasks.)

Task-1 `test_h2.cpp` bottom:
```cpp
int main() { return test_cheb(); }
```

- [ ] **Step 5: Run tests** — `cmake --build build -j$(nproc) && ctest --test-dir build -R h2 --output-on-failure` → PASS. Also run full `ctest` → all 4 pass.

- [ ] **Step 6: Commit** — `git add include/cheb_basis.hpp src/cheb_basis.cpp tests/test_h2.cpp CMakeLists.txt && git commit -m "feat(h2): Chebyshev basis (nodes, interpolation weights)"`

---

### Task 2: UniformQuadTree — boxes, neighbors, interaction lists

**Files:**
- Create: `include/uniform_quadtree.hpp`, `src/uniform_quadtree.cpp`
- Test: `tests/test_h2.cpp` (add `test_tree`)

**Interfaces:**
- Produces:
```cpp
struct H2Box {
    int level;              // 0 = root
    int bx, by;             // box coords within level, in [0, 2^level)
    int ix0, iy0, side;     // element range [ix0, ix0+side) x [iy0, iy0+side)
    int parent;             // box id, -1 for root
    int child[4];           // SW,SE,NW,NE box ids (child quadrant c = (cx)+(cy<<1)); -1 if none
    bool leaf;
};
class UniformQuadTree {
public:
    UniformQuadTree(int Ns, int leaf_side);
    int Ns() const; int leaf_side() const; int nlevels() const; int leaf_level() const;
    const std::vector<H2Box>& boxes() const;
    int level_begin(int level) const;     // first box id at level
    int boxes_per_side(int level) const;  // 2^level
    int box_id(int level, int bx, int by) const;
    // neighbors at same level within radius r (includes self), Chebyshev box-distance <= r
    std::vector<int> neighbors(int box, int r) const;
    // interaction list: children of parent's neighbors, minus this box's near neighbors
    std::vector<int> interaction_list(int box, int near_radius) const;
};
```
- Child quadrant index `c = cx + 2*cy`, `cx,cy ∈ {0,1}` (0=low/SW). Box-distance = Chebyshev distance in `(bx,by)`.

- [ ] **Step 1: Write failing tests** (append `test_tree` to `test_h2.cpp`; extend `main`)

```cpp
static int test_tree() {
    hmc::UniformQuadTree t(16, 4);   // 16x16 grid, leaf side 4 -> leaf level 2
    CHECK(t.leaf_level() == 2);
    CHECK(t.nlevels() == 3);
    CHECK(t.boxes_per_side(0) == 1);
    CHECK(t.boxes_per_side(2) == 4);
    // ranges partition the grid at leaf level
    int covered = 0;
    for (const auto& b : t.boxes()) if (b.leaf) covered += b.side * b.side;
    CHECK(covered == 16*16);
    // every leaf side == leaf_side
    for (const auto& b : t.boxes()) if (b.leaf) CHECK(b.side == 4);
    // interior leaf has 9 near neighbors (radius 1), corner has 4
    int interior = t.box_id(2, 1, 1);
    CHECK((int)t.neighbors(interior, 1).size() == 9);
    int corner = t.box_id(2, 0, 0);
    CHECK((int)t.neighbors(corner, 1).size() == 4);
    // interaction list excludes near neighbors and is within parent's neighbor children
    auto il = t.interaction_list(interior, 1);
    for (int s : il) {
        auto nb = t.neighbors(interior, 1);
        CHECK(std::find(nb.begin(), nb.end(), s) == nb.end());
        CHECK(t.boxes()[s].level == t.boxes()[interior].level);
    }
    // parent/child consistency
    for (const auto& b : t.boxes())
        if (!b.leaf) for (int c = 0; c < 4; ++c) {
            CHECK(b.child[c] >= 0);
            CHECK(t.boxes()[b.child[c]].parent == t.box_id(b.level,b.bx,b.by));
        }
    return 0;
}
```

- [ ] **Step 2: Run to verify it fails** — `ctest --test-dir build -R h2` → FAIL (no `uniform_quadtree.hpp`).

- [ ] **Step 3: Implement** `uniform_quadtree.hpp`/`.cpp`. Build level by level; `level_begin` = prefix sum of `4^ℓ`. For box at `(level,bx,by)`: `side = Ns >> level`, `ix0 = bx*side`, `iy0 = by*side`. `parent = box_id(level-1, bx/2, by/2)`. `leaf = (level == leaf_level)`. Children set when not leaf: `child[c] = box_id(level+1, 2*bx+cx, 2*by+cy)`.

```cpp
// neighbors(box,r): same-level boxes with |dbx|<=r and |dby|<=r, in grid bounds (includes self)
// interaction_list(box,nr): for each pn in neighbors(parent, nr):
//   for each child cc of pn: if cc not in neighbors(box, nr): add cc
```

- [ ] **Step 4: Run tests** — build + `ctest -R h2` → PASS; full ctest green.
- [ ] **Step 5: Commit** — `git commit -m "feat(h2): uniform quad-tree with neighbors + interaction lists"`

---

### Task 3: H2Operator — caches + build()

**Files:**
- Create: `include/h2_operator.hpp`, `src/h2_operator.cpp`
- Modify: `CMakeLists.txt` (add `src/uniform_quadtree.cpp`, `src/h2_operator.cpp`)
- Test: `tests/test_h2.cpp` (add `test_caches`)

**Interfaces:**
- Produces:
```cpp
struct H2Params { int leaf_side = 8; int q = 4; int near_radius = 1; };
class H2Operator {
public:
    H2Operator(const BoussinesqKernel& kernel, H2Params p);
    void build();
    Eigen::VectorXd matvec(const Eigen::VectorXd& x) const;  // Task 4
    void print_statistics() const;                            // Task 5
    // exposed for tests:
    int n_far_interactions() const; int n_unique_couplings() const;
private:
    // box-normalized element-center coords along one axis for a box of given side:
    //   for k=0..side-1: ( (ix0+k) - (ix0 + side/2 - 0.5) ) / (side/2)  ==  (k - side/2 + 0.5)/(side/2)
    // independent of ix0 -> one vector per (side). Used by P2M/L2P.
    ...
};
```

**Caches to build in `build()`:**
1. **Leaf interpolation** `Wleaf` = `(leaf_side × q)` = `cheb.weights_at(centers_norm(leaf_side))`. Used by P2M/L2P via tensor product. One matrix (all leaves identical).
2. **M2M/L2L** `R[c]` for `c=0..3`, each `(q² × q²)`: `R[c][A,a] = w_{Ax}(p_a.x) · w_{Ay}(p_a.y)` where parent-coord child node `p_a = (±0.5 + 0.5·ξ_{ax}, ±0.5 + 0.5·ξ_{ay})`, sign per quadrant `c` (`cx→x`, `cy→y`; `-0.5` for 0, `+0.5` for 1). Build with `cheb.weights(·)`.
3. **M2L coupling cache**: for each interacting pair found over the tree, key `(level, dx_box, dy_box)`; value `(q² × q²)` matrix `K[A,B] = g(X_A^t − X_B^s)` at physical node coords. Physical node coord along axis for box at level `ℓ`, origin element `ix0`: `X = (ix0 + side/2 - 0.5)·h + (side·h/2)·ξ_n`, `side = Ns>>ℓ`. Offset depends only on `(dx_box·side·h, dy_box·side·h)` → cache by `(ℓ,dx_box,dy_box)`. `g` from `love_uz`.
4. **Near stencil cache**: for each near leaf offset `(dx,dy)` with `|dx|,|dy| ≤ near_radius` (in leaf units), dense `(leaf_side² × leaf_side²)` `A_near[(iy*ls+ix),(jy*ls+jx)] = kernel.entry(global_i, global_j)` — but translation-invariant, so build once per offset using element offsets `(dx*ls + ix - jx, dy*ls + iy - jy)` via `kernel.entry`. Store in a map keyed `(dx,dy)`.

- [ ] **Step 1: Write failing test** (`test_caches`): build operator on `Ns=16, leaf_side=4, q=4`; assert `n_unique_couplings() > 0` and that a re-derived coupling equals the cached one for a known pair; assert near-stencil for offset `(0,0)` has `kernel.entry(0,0)` on its diagonal.

```cpp
static int test_caches() {
    hmc::BoussinesqKernel k(16, 1.0, 1.0);
    hmc::H2Operator A(k, {4, 4, 1});
    A.build();
    CHECK(A.n_unique_couplings() > 0);
    CHECK(A.n_unique_couplings() < A.n_far_interactions()); // sharing happened
    return 0;
}
```

- [ ] **Step 2: Run → FAIL** (no `h2_operator.hpp`).
- [ ] **Step 3: Implement** caches + `build()` (no matvec yet — return `x` as placeholder is NOT allowed; declare `matvec` but implement in Task 4; keep test scope to caches). Build interaction lists by iterating all non-leaf boxes' children pairs OR per-box `interaction_list`. Use `std::unordered_map` with a packed integer key for caches. Record `n_far_interactions_` (total M2L pairs) and `n_unique_couplings_` (distinct keys).
- [ ] **Step 4: Run → PASS**; full ctest green.
- [ ] **Step 5: Commit** — `git commit -m "feat(h2): operator caches (M2M/L2L, M2L couplings, near stencils) + build"`

---

### Task 4: matvec — the six passes + accuracy vs dense

**Files:**
- Modify: `src/h2_operator.cpp`, `include/h2_operator.hpp`
- Test: `tests/test_h2.cpp` (add `test_accuracy`)

**Interfaces:**
- Consumes: caches from Task 3; `ChebBasis`; `UniformQuadTree`.
- Produces: `Eigen::VectorXd H2Operator::matvec(const Eigen::VectorXd& x) const`.

**Pass algorithm (per-box coeff arrays `M`, `L` of size `q²`, indexed `a = ax + q*ay`):**
- **P2M** (each leaf `t`): gather `x` on the `leaf_side×leaf_side` block into matrix `Xs (ls×ls)` (row=iy-local, col=ix-local). `M_mat = Wleaf^T · Xs · Wleaf` (size `q×q`), flatten to `M[a=ax+q*ay] = M_mat(ay,ax)`. (Tensor product, two 1D passes via the two matmuls.)
- **M2M** (levels leaf-1 … 0): `M_parent += R[c] · M_child` (flattened `q²` vectors).
- **M2L** (every box with non-empty interaction list): `L_t += K_ts · M_s` over its interaction list (cached `K`). OpenMP over target boxes.
- **L2L** (levels 0 … leaf-1): `L_child += R[c]^T · L_parent`.
- **L2P** (each leaf `t`): `L_mat(ay,ax) = L[ax+q*ay]`; `Yt = Wleaf · L_mat · Wleaf^T` (`ls×ls`); scatter-add into `y`.
- **Near** (each leaf `t`, each near leaf `s` from `neighbors(t,near_radius)`): `y_block(t) += A_near[offset] · x_block(s)` using cached stencil (flatten block as `iy*ls+ix`). OpenMP over `t`.

- [ ] **Step 1: Write failing test** (`test_accuracy`): compare to dense.

```cpp
static int test_accuracy() {
    int Ns = 32; hmc::BoussinesqKernel k(Ns, 1.0, 1.0);
    Eigen::MatrixXd S = k.assemble_dense();
    Eigen::VectorXd x = Eigen::VectorXd::Random(Ns*Ns);
    Eigen::VectorXd ref = S * x;
    auto relerr = [&](int q){
        hmc::H2Operator A(k, {8, q, 1}); A.build();
        Eigen::VectorXd y = A.matvec(x);
        return (y - ref).norm() / ref.norm();
    };
    double e4 = relerr(4), e6 = relerr(6);
    std::printf("H2 rel err: q=4 %.2e  q=6 %.2e\n", e4, e6);
    CHECK(e4 < 5e-3);
    CHECK(e6 < e4);        // convergence with order
    CHECK(e6 < 1e-4);
    return 0;
}
```

- [ ] **Step 2: Run → FAIL** (matvec unimplemented / inaccurate).
- [ ] **Step 3: Implement** the six passes. Store `M`,`L` as `std::vector<Eigen::VectorXd>` sized to box count. Zero per matvec. Use `Eigen::Map`/`Reshaped` for the `q×q`/`ls×ls` reshapes. Parallelize M2L and Near with `#pragma omp parallel for`.
- [ ] **Step 4: Run → PASS** (tune: if `e4` borderline, verify node-coord normalization and quadrant signs). Full ctest green.
- [ ] **Step 5: Commit** — `git commit -m "feat(h2): matvec passes (P2M/M2M/M2L/L2L/L2P/near), accurate vs dense"`

---

### Task 5: print_statistics()

**Files:** Modify `src/h2_operator.cpp`, `include/h2_operator.hpp`. Test: `test_h2.cpp` (smoke).

- [ ] **Step 1: Test** — call `A.print_statistics()` in `test_caches`; assert it returns without throwing (smoke). 
- [ ] **Step 2: Run → FAIL** (undefined).
- [ ] **Step 3: Implement** printing the spec §29 block: `N`, `Ns`, boxes per level, #leaves, `leaf_side`, `q`, `r=q²`, #near interactions, #far interactions, #unique couplings, #unique near stencils, bytes per cache (`coupling`: `n_unique_couplings * q⁴ * 8`; `near`: `n_near_offsets * leaf_side⁴ * 8`; `M/L buffers`: `2 * n_boxes * q² * 8`), and total. Add a `H2Info info() const` returning these as a struct for bindings.
- [ ] **Step 4: Run → PASS**.
- [ ] **Step 5: Commit** — `git commit -m "feat(h2): print_statistics + H2Info"`

---

### Task 6: Python backend integration

**Files:** Modify `python/bindings.cpp`.

**Interfaces:** Add `std::string backend` (`"dense"|"hmatrix"|"h2"`) to `PyContactSolver` ctor (default `"hmatrix"` to preserve current behavior; keep legacy `use_hmatrix` bool honored: if user passes `use_hmatrix=False` and no backend, use dense). Add `int q`, `int near_radius` args (defaults 4,1). Hold `std::unique_ptr<hmc::H2Operator> h2_`. In `apply()`: dispatch on backend. In `hmatrix_info()`: when h2, print `h2_->print_statistics()` and return its `H2Info` fields.

- [ ] **Step 1: Write failing Python test** `tests/test_h2.py`:
```python
import sys; sys.path.insert(0, 'python')
import numpy as np, hmatrix_contact as hc
Ns = 64
dense = hc.ContactSolver(grid_size=Ns, use_hmatrix=False)
h2 = hc.ContactSolver(grid_size=Ns, backend="h2", q=6)
x = np.random.default_rng(0).standard_normal(Ns*Ns)
ref = dense.matvec(x).ravel(); got = h2.matvec(x).ravel()
rel = np.linalg.norm(got-ref)/np.linalg.norm(ref)
print("rel err", rel); assert rel < 1e-4
```
- [ ] **Step 2: Run → FAIL** (`backend` kwarg unknown).
- [ ] **Step 3: Implement** ctor arg + dispatch + build h2 in ctor.
- [ ] **Step 4: Run** — rebuild module; `conda run -n fenicsx-env python tests/test_h2.py` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(h2): ContactSolver backend='h2' python integration"`

---

### Task 7: Integration (Hertz) + scaling bench

**Files:** Create `bench_h2.py`; add Hertz-via-H2 check to `tests/test_h2.py`.

- [ ] **Step 1: Hertz test** — solve Hertz gap via `backend="h2", q=6` at `Ns=64`; assert contact area within 3% of the H-matrix backend's, and `mean_pressure ≈ p_nominal`.
- [ ] **Step 2: Run → confirm** convergence + accuracy. (If contact area off, raise `q` or `near_radius`.)
- [ ] **Step 3: `bench_h2.py`** — for `Ns ∈ {64,128,256,512}`: build H2 (q=6) and H-matrix; print build time, matvec time, and memory (H2 from `H2Info`, H-matrix from `hmatrix_info`). Demonstrate H2 memory ~O(N).
- [ ] **Step 4: Run** bench; capture numbers.
- [ ] **Step 5: Commit** — `git commit -m "test(h2): Hertz integration + H2/H-matrix scaling bench"`

---

## Self-Review

- **Spec coverage:** quad-tree+ranges (T2 ✓ spec §4,21), leaf size (T3 params ✓ §5), kernel callback/near split (T3 ✓ §6,7), bbFMM interpolation (T1,T4 ✓ §9,10), interaction lists (T2 ✓ §11), near stencils cached by offset (T3 ✓ §12), coupling cache by (level,offset) (T3 ✓ §13), tensor-product interpolation (T1,T4 ✓ §14,22), six-pass matvec (T4 ✓ §15–19), compact arrays/no per-block heap (T3 ✓ §20), statistics (T5 ✓ §29), additive backend + API (T6 ✓), validation vs dense + Hertz + scaling (T4,T7 ✓). FFT and masking explicitly out of scope (✓ non-goals).
- **Placeholder scan:** none — code given for the numerically hard parts; glue (tree build, stats printing) specified by exact formulas/fields.
- **Type consistency:** `H2Operator(kernel, H2Params)`, `H2Params{leaf_side,q,near_radius}`, child quadrant `c=cx+2*cy`, coeff flatten `a=ax+q*ay`, `R[c]` `q²×q²`, `Wleaf` `leaf_side×q` — consistent across T3/T4/T5.
