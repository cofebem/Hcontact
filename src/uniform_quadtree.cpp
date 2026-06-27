#include "uniform_quadtree.hpp"

#include <algorithm>
#include <stdexcept>

namespace hmc {

static int ilog2(int v) {
    int k = 0;
    while ((1 << k) < v) ++k;
    return k;
}

UniformQuadTree::UniformQuadTree(int Ns, int leaf_side)
    : Ns_(Ns), leaf_side_(leaf_side) {
    if (Ns < 1 || leaf_side < 1 || (Ns % leaf_side) != 0)
        throw std::invalid_argument("UniformQuadTree: need leaf_side | Ns");
    if ((Ns & (Ns - 1)) != 0 || (leaf_side & (leaf_side - 1)) != 0)
        throw std::invalid_argument("UniformQuadTree: Ns, leaf_side must be powers of two");

    nlevels_ = ilog2(Ns / leaf_side) + 1; // levels 0 .. leaf_level

    // level_begin_ via prefix sum of 4^level
    level_begin_.assign(nlevels_, 0);
    int total = 0;
    for (int l = 0; l < nlevels_; ++l) {
        level_begin_[l] = total;
        total += (1 << l) * (1 << l);
    }
    boxes_.resize(total);

    const int leaf_lvl = leaf_level();
    for (int l = 0; l < nlevels_; ++l) {
        const int nbps = 1 << l;
        const int side = Ns_ >> l;
        for (int by = 0; by < nbps; ++by)
            for (int bx = 0; bx < nbps; ++bx) {
                H2Box& b = boxes_[box_id(l, bx, by)];
                b.level = l;
                b.bx = bx;
                b.by = by;
                b.ix0 = bx * side;
                b.iy0 = by * side;
                b.side = side;
                b.parent = (l == 0) ? -1 : box_id(l - 1, bx / 2, by / 2);
                b.leaf = (l == leaf_lvl);
                for (int c = 0; c < 4; ++c) b.child[c] = -1;
                if (!b.leaf) {
                    for (int cy = 0; cy < 2; ++cy)
                        for (int cx = 0; cx < 2; ++cx)
                            b.child[cx + 2 * cy] =
                                box_id(l + 1, 2 * bx + cx, 2 * by + cy);
                }
            }
    }
}

std::vector<int> UniformQuadTree::neighbors(int box, int r) const {
    const H2Box& b = boxes_[box];
    const int nbps = boxes_per_side(b.level);
    std::vector<int> out;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            const int nx = b.bx + dx, ny = b.by + dy;
            if (nx < 0 || nx >= nbps || ny < 0 || ny >= nbps) continue;
            out.push_back(box_id(b.level, nx, ny));
        }
    return out;
}

std::vector<int> UniformQuadTree::interaction_list(int box, int near_radius) const {
    const H2Box& b = boxes_[box];
    if (b.parent < 0) return {}; // root has no interactions
    const std::vector<int> near = neighbors(box, near_radius);
    std::vector<int> out;
    for (int pn : neighbors(b.parent, near_radius)) {
        const H2Box& p = boxes_[pn];
        for (int c = 0; c < 4; ++c) {
            const int cc = p.child[c];
            if (cc < 0) continue;
            if (std::find(near.begin(), near.end(), cc) == near.end())
                out.push_back(cc);
        }
    }
    return out;
}

} // namespace hmc
