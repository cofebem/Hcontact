# Codex Task: Rework half-space flexibility operator as a matrix-free hierarchical operator

## Executive summary

The previous design was too close to a classical blockwise H-matrix with ACA-compressed blocks.
For a regular Cartesian grid and an elastic half-space kernel, this is probably not the best first target.

The better target is:

```text
matrix-free global operator + quadtree hierarchy + cached kernel interactions + optional H2/FMM-style interpolation
```

The user must not assemble the matrix block by block.
The public API should expose only a whole-operator construction and matvec interface:

```cpp
HalfSpaceOperator A(grid, kernel, params);
A.build();
y = A.matvec(x);
```

Internally the code may use trees, boxes, near-field dense stencils, interaction lists, interpolation matrices, and cached translation operators, but these are implementation details.

---

## Important interpretation check

If the matrix dimension is truly only

```text
N = 1024
```

then the dense matrix has only

```text
1024 * 1024 * 8 bytes ≈ 8 MB
```

In that case a classical H-matrix may be slower and larger than the dense matrix because of overhead from:

```text
many small blocks
many std::vector allocations
block-tree metadata
low-rank factor metadata
ACA temporary buffers
non-contiguous memory access
```

For `N = 1024`, use one of these instead:

```text
1. dense matrix, if repeated solves/matvecs are needed;
2. matrix-free direct kernel matvec, if only a few matvecs are needed;
3. FFT convolution, if the grid is regular and the full rectangular domain is active.
```

Use H/H2 only when `N` is significantly larger, for example:

```text
N >= 10^4 to 10^5
```

or when the active domain is irregular and dense/FFT approaches are not appropriate.

If instead the grid is `1024 x 1024`, then

```text
N = 1,048,576
```

and a dense matrix is impossible. In that case, use the matrix-free H2/FMM-like design below.

---

# Recommended design

## 1. Do not build a classical blockwise H-matrix first

Avoid a design whose central object is a large list of independently compressed blocks:

```cpp
std::vector<DenseBlock> dense_blocks;
std::vector<LowRankBlock> low_rank_blocks;
```

This design can work, but it is often memory-heavy because each admissible block stores its own low-rank factors:

```text
A_ts ≈ U_ts V_ts^T
```

For a regular half-space kernel, many bases are redundant. The better design is to share bases between interactions of the same cluster.

Use an H2/FMM-like structure:

```text
A_ts ≈ V_t S_ts V_s^T
```

where:

```text
V_t = basis/interpolation operator for target cluster t
V_s = basis/interpolation operator for source cluster s
S_ts = small coupling matrix between interpolation nodes of t and s
```

The important difference is:

```text
classical H-matrix: stores factors per block
H2/FMM-like operator: stores bases per cluster and couplings per interaction
```

Expected memory changes from roughly

```text
O(r N log N)
```

toward

```text
O(r N)
```

up to interaction-list constants.

---

## 2. Use the quadtree only as an internal acceleration structure

The public interface should be simple:

```cpp
Grid grid(Lx, Ly, nx, ny);
HalfSpaceKernel kernel(E_star, h);

HalfSpaceOperatorParams params;
params.leaf_size = 32;
params.interpolation_order = 4;
params.near_radius = 1;
params.use_symmetry = true;
params.use_h2 = true;
params.use_cache = true;

HalfSpaceOperator A(grid, kernel, params);
A.build();

std::vector<double> y = A.matvec(x);
```

The user must not call anything like:

```cpp
build_block(i, j);
assemble_block(i, j);
compress_block(i, j);
```

Those operations should be internal.

---

# Geometry and hierarchy

## 3. Grid

Assume a regular grid over a square or rectangular domain.

Each degree of freedom has coordinates:

```cpp
struct Point {
    double x;
    double y;
};
```

The number of unknowns is:

```text
N = nx * ny
```

For a contact half-space flexibility operator, the kernel is translation-invariant away from boundaries and singularities:

```text
G(x, y) = G(x - y)
```

For normal displacement due to normal pressure:

```text
G(r) ~ 1 / r
```

with a special self-term.

---

## 4. Quadtree boxes

Use a balanced quadtree.

Each box stores:

```cpp
struct Box {
    int id;
    int level;
    int parent;
    std::array<int, 4> child;
    bool is_leaf;

    int ix0, ix1;
    int iy0, iy1;

    double xmin, xmax;
    double ymin, ymax;
    double cx, cy;
    double diameter;

    int n_dofs;
};
```

For a regular grid, avoid storing an explicit `std::vector<int> indices` for every box if possible.
Store index ranges instead:

```text
ix0 <= ix < ix1
iy0 <= iy < iy1
```

Convert local `(ix, iy)` to global index only when needed:

```cpp
int global_id = iy * nx + ix;
```

This avoids huge memory overhead in the tree.

---

## 5. Leaf size

Use:

```text
leaf_size = 32 to 128 unknowns per leaf
```

For 2D square leaves, prefer powers of two in each direction.
For example:

```text
8 x 8 = 64 unknowns per leaf
```

or

```text
16 x 16 = 256 unknowns per leaf
```

If using near-field dense blocks, too small leaves create too many tiny allocations.
Start with:

```text
leaf_nx = leaf_ny = 8
```

then test:

```text
leaf_nx = leaf_ny = 16
```

---

# Kernel

## 6. Kernel callback

Use a callback:

```cpp
double kernel(Point xi, Point xj, double h);
```

For normal half-space contact:

```cpp
double kernel(Point xi, Point xj, double h) {
    double dx = xi.x - xj.x;
    double dy = xi.y - xj.y;
    double r = std::sqrt(dx*dx + dy*dy);

    if (r > 0.0) {
        return C / r;
    }
    return self_term(h);
}
```

The constant `C` depends on the convention already used in the code, for example involving `E_star`.

Do not use `1 / 0` on the diagonal.

---

## 7. Self and near terms

For the diagonal and immediate neighbors, pointwise `1/r` is not ideal.
The near field should use either:

```text
1. analytical cell-integrated kernel;
2. accurate quadrature over source cell;
3. existing validated regularization from the code.
```

For a first computational test, a regularized self-term is acceptable, but it must be isolated in one function:

```cpp
double self_term(double h);
```

so that it can later be replaced by the correct cell-integrated value.

---

# Preferred algorithm: H2/FMM-like matvec

## 8. Main idea

Do not store low-rank factors for every admissible block.
Instead, perform matvecs through three stages:

```text
1. upward pass: compress source values into multipole/interpolation coefficients;
2. interaction pass: translate source coefficients to target local coefficients;
3. downward/evaluation pass: evaluate target coefficients at physical DOFs;
4. near-field correction: apply direct interactions for neighboring leaves.
```

This is much closer to an FMM/H2 operator than to a classical H-matrix.

---

## 9. Interpolation-based approximation

For well-separated source box `s` and target box `t`, approximate:

```text
G(x, y) ≈ sum_a sum_b L_a^t(x) G(xi_a^t, xi_b^s) L_b^s(y)
```

where:

```text
xi_a^t = interpolation nodes in target box
xi_b^s = interpolation nodes in source box
L_a^t = Lagrange basis in target box
L_b^s = Lagrange basis in source box
```

Then:

```text
A_ts ≈ P_t K_ts P_s^T
```

where:

```text
P_s^T maps source DOFs to source interpolation coefficients
K_ts is a small kernel matrix between interpolation nodes
P_t maps target interpolation values to target DOFs
```

For a tensor-product interpolation order `q` in 2D:

```text
r = q^2
```

Start with:

```text
q = 4
```

Then test:

```text
q = 5, 6, 8
```

---

## 10. Chebyshev nodes

Use tensor-product Chebyshev nodes on each box.

For one dimension on interval `[a, b]`:

```text
xi_k = 0.5*(a+b) + 0.5*(b-a)*cos((2k+1)*pi/(2q)),  k = 0,...,q-1
```

In 2D, use all pairs:

```text
(xi_a, eta_b),  a,b = 0,...,q-1
```

The number of interpolation nodes per box is:

```text
r = q*q
```

For example:

```text
q = 4 -> r = 16
q = 6 -> r = 36
q = 8 -> r = 64
```

---

## 11. Interaction lists instead of all admissible blocks

For each box, do not interact with all far boxes individually.
Use standard FMM-style interaction lists.

For a box `t`, define:

```text
near(t) = boxes at the same leaf level that touch t or are within near_radius boxes
far/interactions(t) = children of neighbors of parent(t) excluding near(t)
```

For a quadtree in 2D, the interaction list size is bounded by a small constant.
This is essential.

Do not build an all-pairs admissible block tree if the final target is H2/FMM-like matvec.

---

## 12. Near-field direct interactions

For leaf boxes, apply direct interactions only with neighboring leaf boxes.

For example, with `near_radius = 1`, each leaf interacts directly with at most:

```text
3 x 3 = 9 neighboring leaves
```

including itself.

For each near pair, compute direct kernel contributions on the fly or store a reusable stencil.

Because the grid is regular and the kernel is translation-invariant, near interactions can often be cached by relative offset.

Do not allocate a separate dense matrix for every near block if many are identical.

Use a cache key:

```cpp
struct NearStencilKey {
    int leaf_nx;
    int leaf_ny;
    int dx_leaf;
    int dy_leaf;
};
```

Then store one dense near stencil per relative offset.

---

# Translation-invariance caching

## 13. Cache small coupling matrices

For a regular grid and uniform tree, the far-field coupling matrix depends only on:

```text
level
relative box offset
interpolation order
box size
```

For same-level interactions:

```cpp
struct FarCouplingKey {
    int level;
    int dx_box;
    int dy_box;
    int q;
};
```

The coupling matrix is:

```text
K_ts[a,b] = G(xi_a^t - xi_b^s)
```

with size:

```text
(q*q) x (q*q)
```

Do not recompute this matrix for every box pair.
Reuse it through the cache.

---

## 14. Cache interpolation matrices

For a uniform grid, the interpolation matrix from leaf grid points to Chebyshev nodes is the same for all leaves at the same level.

Cache:

```text
P_leaf
P_leaf^T
```

or more precisely the 1D interpolation matrices and exploit tensor products.

Do not store a full 2D interpolation matrix per leaf if it can be represented as tensor products.

For a leaf with `m x m` grid points and interpolation order `q`, use:

```text
P_2D = P_y ⊗ P_x
```

Apply it using two 1D passes rather than forming the Kronecker product explicitly.

---

# Matvec structure

## 15. Public matvec

The matvec should look like:

```cpp
std::vector<double> HalfSpaceOperator::matvec(const std::vector<double>& x) const {
    zero internal buffers;
    upward_pass(x);
    interaction_pass();
    downward_pass();
    near_field_pass(x);
    return y;
}
```

The user should never see internal boxes or blocks.

---

## 16. Upward pass

For each leaf box `s`, compress the source values into interpolation coefficients:

```text
m_s = P_s^T x_s
```

Then recursively aggregate child coefficients into parent coefficients:

```text
m_parent += transfer_child_to_parent * m_child
```

The transfer matrices are small and can be cached by child position.

---

## 17. Interaction pass

For each box `t` and each source box `s` in its interaction list:

```text
l_t += K_ts m_s
```

where:

```text
m_s = source coefficients of box s
l_t = local target coefficients of box t
K_ts = cached coupling matrix
```

This is the main far-field computation.

---

## 18. Downward pass

Propagate local target coefficients from parents to children:

```text
l_child += transfer_parent_to_child * l_parent
```

At each leaf, evaluate at physical DOFs:

```text
y_leaf += P_leaf l_leaf
```

Again, use cached transfer/interpolation matrices.

---

## 19. Near-field pass

For each leaf `t`, for each neighboring leaf `s`:

```text
y_t += A_near(ts) x_s
```

where `A_near(ts)` is either:

```text
1. computed on the fly by direct kernel evaluations;
2. retrieved from a near-stencil cache;
3. stored as a small dense block if the geometry is not reusable.
```

For a regular grid, prefer a near-stencil cache.

---

# Avoiding memory blow-up

## 20. Avoid per-block allocation

Do not allocate thousands or millions of small objects with their own heap storage.

Avoid:

```cpp
std::vector<std::unique_ptr<Block>> blocks;
```

if each block owns separate vectors.

Prefer compact arrays:

```cpp
std::vector<Box> boxes;
std::vector<Interaction> interactions;
std::vector<double> coupling_cache_values;
std::vector<double> near_cache_values;
```

An interaction should store only small integer indices:

```cpp
struct Interaction {
    int target_box;
    int source_box;
    int coupling_cache_id;
};
```

---

## 21. Store ranges, not lists of indices

For a regular Cartesian grid, never store all global indices for every box.

Avoid:

```cpp
std::vector<int> indices;
```

inside each box.

Use:

```cpp
int ix0, ix1;
int iy0, iy1;
```

This may dramatically reduce memory.

---

## 22. Use tensor-product interpolation

Avoid storing 2D interpolation matrices explicitly.

For source compression:

```text
M = P_y^T X P_x
```

or an equivalent two-pass operation.

For target evaluation:

```text
Y = P_y L P_x^T
```

This reduces storage and improves cache behavior.

---

# Optional fallback: classical H-matrix mode

## 23. When to keep classical H-matrix mode

Keep classical H-matrix/ACA only if:

```text
1. the active domain is irregular;
2. the grid is locally refined;
3. the kernel is not translation-invariant;
4. you need explicit approximate matrix entries;
5. you need H-LU or H-Cholesky later.
```

Otherwise, for a uniform half-space grid, H2/FMM or FFT is usually more natural.

---

## 24. If classical H-matrix mode is retained

If keeping the classical H-matrix, still apply these corrections:

```text
1. never assemble the full dense matrix;
2. store only one triangular part for symmetric operators;
3. avoid explicit index vectors in every cluster;
4. use admissibility with leaf sizes not too small;
5. use ACA through entry callbacks;
6. impose hard rank cap;
7. recompress by QR + small SVD;
8. cache same-level same-offset blocks;
9. avoid per-block heap allocations where possible.
```

Suggested parameters:

```text
leaf_size = 64 or 128
eta = 1.0
aca_eps = 1e-5 to 1e-6
r_max = 24 to 48
```

But this should be treated as a secondary path, not the primary design.

---

# FFT alternative

## 25. Use FFT if the whole rectangular grid is active

For a full regular rectangular grid, the half-space operator is convolution-like:

```text
u = G * p
```

In that case, FFT matvec is usually the simplest and fastest approach.

Use FFT if:

```text
1. the grid is uniform;
2. the active domain is the full rectangle;
3. only matvecs are needed;
4. boundary conditions/periodicity/zero-padding are acceptable.
```

Then the operator is applied by:

```text
1. pad pressure field;
2. FFT pressure;
3. multiply by FFT kernel;
4. inverse FFT;
5. crop physical domain.
```

Expected complexity:

```text
O(N log N)
```

with very small memory compared with dense storage.

---

# Recommended implementation path

## 26. First decide the actual problem size

Add this diagnostic at the start:

```cpp
std::cout << "nx = " << nx << "\n";
std::cout << "ny = " << ny << "\n";
std::cout << "N  = " << nx * ny << "\n";
std::cout << "Dense memory estimate = "
          << double(nx) * double(ny) * double(nx) * double(ny) * 8.0 / 1e9
          << " GB\n";
```

Be careful:

```text
1024 x 1024 matrix -> N = 1024 -> dense memory ≈ 8 MB
1024 x 1024 grid   -> N = 1,048,576 -> dense memory ≈ 8 TB
```

Actually, for `N = 1,048,576`, the dense matrix has `N^2` entries:

```text
N^2 * 8 bytes ≈ 8.8 TB
```

so dense assembly is impossible.

---

## 27. Best first version

Implement this first:

```text
regular-grid FFT matvec
```

if the domain is a full rectangle.

Then implement:

```text
matrix-free H2/FMM-like quadtree matvec
```

if you need:

```text
1. masked active domains;
2. non-rectangular contact zones;
3. local refinement;
4. non-periodic/localized treatment;
5. compatibility with future contact active-set updates.
```

Only then implement a classical H-matrix if explicit block-compressed matrix storage is really required.

---

# Minimal H2/FMM-like parameter set

Start with:

```text
leaf_nx = 8
leaf_ny = 8
q = 4
near_radius = 1
use_tensor_product_interpolation = true
use_coupling_cache = true
use_near_stencil_cache = true
```

Then test:

```text
leaf_nx = 16
leaf_ny = 16
q = 5 or 6
near_radius = 1
```

If accuracy is insufficient near the singularity, increase near field first:

```text
near_radius = 2
```

before increasing interpolation order too much.

---

# Profiling output

After `build()`, print:

```text
N
nx, ny
number of boxes by level
number of leaves
leaf_nx, leaf_ny
interpolation order q
rank r = q*q
number of near interactions
number of far interactions
number of unique far coupling matrices
number of unique near stencils
memory used by tree
memory used by interaction list
memory used by coupling cache
memory used by near cache
memory used by work buffers
build time
matvec time
```

During `matvec()`, optionally time:

```text
upward pass
interaction pass
downward pass
near-field pass
```

This is essential to determine whether the bottleneck is:

```text
tree construction
coupling construction
near field
far field
memory allocation
cache misses
```

---

# Final expected public usage

```cpp
Grid grid(L, nx, ny);
HalfSpaceKernel kernel(E_star, h);

HalfSpaceOperatorParams params;
params.backend = HalfSpaceBackend::H2;
params.leaf_nx = 8;
params.leaf_ny = 8;
params.interpolation_order = 4;
params.near_radius = 1;
params.use_tensor_product_interpolation = true;
params.use_coupling_cache = true;
params.use_near_stencil_cache = true;

HalfSpaceOperator A(grid, kernel, params);
A.build();
A.print_statistics();

std::vector<double> y = A.matvec(x);
```

No full dense matrix should be assembled.
No user-side block loop should exist.
No per-block ACA should be required in the primary path.

---

# Main conclusion

For a regular elastic half-space flexibility operator, the best redesign is not a more carefully assembled blockwise H-matrix.

The best redesign is:

```text
FFT for full rectangular uniform domains;
H2/FMM-like matrix-free quadtree operator for large irregular or active domains;
classical ACA H-matrix only as a fallback.
```

The implementation should prioritize:

```text
1. matrix-free global interface;
2. no full dense assembly;
3. no user-visible block assembly;
4. compact quadtree boxes with index ranges;
5. tensor-product interpolation;
6. cached same-level coupling matrices;
7. cached near-field stencils;
8. profiling before further optimization.
```
