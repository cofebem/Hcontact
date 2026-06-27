#include "cheb_basis.hpp"
#include "uniform_quadtree.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static int test_cheb() {
    hmc::ChebBasis b(5);
    CHECK(static_cast<int>(b.nodes.size()) == 5);

    // partition of unity: weights sum to 1 at any t
    for (double t : {-0.9, -0.3, 0.0, 0.4, 1.0}) {
        const double s = b.weights(t).sum();
        CHECK(std::abs(s - 1.0) < 1e-12);
    }

    // interpolation exactness for polynomials of degree < q
    auto f = [](double x) {
        return 3 - 2 * x + 0.5 * x * x - x * x * x + 0.25 * x * x * x * x;
    };
    Eigen::VectorXd fnodes(5);
    for (int m = 0; m < 5; ++m) fnodes(m) = f(b.nodes[m]);
    for (double t : {-0.7, 0.1, 0.85}) {
        const double interp = b.weights(t).dot(fnodes);
        CHECK(std::abs(interp - f(t)) < 1e-10);
    }
    return 0;
}

static int test_tree() {
    hmc::UniformQuadTree t(16, 4); // 16x16 grid, leaf side 4 -> leaf level 2
    CHECK(t.leaf_level() == 2);
    CHECK(t.nlevels() == 3);
    CHECK(t.boxes_per_side(0) == 1);
    CHECK(t.boxes_per_side(2) == 4);

    // leaf ranges partition the grid
    int covered = 0;
    for (const auto& b : t.boxes())
        if (b.leaf) covered += b.side * b.side;
    CHECK(covered == 16 * 16);
    for (const auto& b : t.boxes())
        if (b.leaf) CHECK(b.side == 4);

    // near-neighbor counts (radius 1)
    const int interior = t.box_id(2, 1, 1);
    CHECK(static_cast<int>(t.neighbors(interior, 1).size()) == 9);
    const int corner = t.box_id(2, 0, 0);
    CHECK(static_cast<int>(t.neighbors(corner, 1).size()) == 4);

    // interaction list excludes near neighbors and stays at same level
    const auto il = t.interaction_list(interior, 1);
    const auto nb = t.neighbors(interior, 1);
    for (int s : il) {
        CHECK(std::find(nb.begin(), nb.end(), s) == nb.end());
        CHECK(t.boxes()[s].level == t.boxes()[interior].level);
    }

    // parent/child consistency
    for (const auto& b : t.boxes())
        if (!b.leaf)
            for (int c = 0; c < 4; ++c) {
                CHECK(b.child[c] >= 0);
                CHECK(t.boxes()[b.child[c]].parent == t.box_id(b.level, b.bx, b.by));
            }
    return 0;
}

int main() {
    if (int rc = test_cheb()) return rc;
    if (int rc = test_tree()) return rc;
    return 0;
}
