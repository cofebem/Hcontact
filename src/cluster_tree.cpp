#include "cluster_tree.hpp"

#include <array>
#include <stdexcept>

namespace hmc {

ClusterTree::ClusterTree(int Ns, int leaf_size) : Ns_(Ns) {
    if (Ns < 1 || leaf_size < 1)
        throw std::invalid_argument("ClusterTree: Ns and leaf_size must be >= 1");
    const int N = Ns * Ns;
    perm_.resize(N);
    for (int k = 0; k < N; ++k) perm_[k] = k;
    build(0, N, Box{0, Ns, 0, Ns}, leaf_size);
    iperm_.resize(N);
    for (int pos = 0; pos < N; ++pos) iperm_[perm_[pos]] = pos;
}

int ClusterTree::build(int begin, int end, Box box, int leaf_size) {
    const int id = static_cast<int>(nodes_.size());
    nodes_.push_back({begin, end, box, {-1, -1, -1, -1}, true});
    if (end - begin <= leaf_size) return id;

    const int mx = (box.ix0 + box.ix1) / 2;
    const int my = (box.iy0 + box.iy1) / 2;
    std::array<std::vector<int>, 4> bucket;
    for (int p = begin; p < end; ++p) {
        const int k = perm_[p];
        const int ix = k % Ns_, iy = k / Ns_;
        bucket[(ix >= mx ? 1 : 0) + (iy >= my ? 2 : 0)].push_back(k);
    }
    const Box qbox[4] = {{box.ix0, mx, box.iy0, my},
                         {mx, box.ix1, box.iy0, my},
                         {box.ix0, mx, my, box.iy1},
                         {mx, box.ix1, my, box.iy1}};
    int b = begin;
    for (int q = 0; q < 4; ++q) {
        if (bucket[q].empty()) continue;
        std::copy(bucket[q].begin(), bucket[q].end(), perm_.begin() + b);
        const int e = b + static_cast<int>(bucket[q].size());
        const int cid = build(b, e, qbox[q], leaf_size);
        nodes_[id].child[q] = cid;
        nodes_[id].leaf = false;
        b = e;
    }
    return id;
}

} // namespace hmc
