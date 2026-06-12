#pragma once

#include <algorithm>
#include <vector>

namespace hmc {

// Half-open element-index ranges [ix0, ix1) x [iy0, iy1).
struct Box {
    int ix0, ix1, iy0, iy1;
};

struct ClusterNode {
    int begin, end; // half-open range into ClusterTree::perm()
    Box box;
    int child[4]; // indices into ClusterTree::nodes(), -1 when absent
    bool leaf;
};

// Quad-tree over the Ns x Ns element grid. The tree induces an ordering of
// the flat indices such that every cluster owns a contiguous range of perm().
class ClusterTree {
public:
    ClusterTree(int Ns, int leaf_size);

    int grid_size() const { return Ns_; }
    int root() const { return 0; }
    const std::vector<ClusterNode>& nodes() const { return nodes_; }
    const std::vector<int>& perm() const { return perm_; }   // cluster pos -> flat
    const std::vector<int>& iperm() const { return iperm_; } // flat -> cluster pos

    // Chebyshev geometry in element-index units; the grid spacing h cancels
    // in the admissibility inequality so it is never needed here.
    static int diam(const Box& b) {
        return std::max(b.ix1 - b.ix0, b.iy1 - b.iy0);
    }
    static int dist(const Box& a, const Box& b) {
        const int dx = std::max({0, b.ix0 - a.ix1, a.ix0 - b.ix1});
        const int dy = std::max({0, b.iy0 - a.iy1, a.iy0 - b.iy1});
        return std::max(dx, dy);
    }

private:
    int build(int begin, int end, Box box, int leaf_size);

    int Ns_;
    std::vector<ClusterNode> nodes_;
    std::vector<int> perm_, iperm_;
};

} // namespace hmc
