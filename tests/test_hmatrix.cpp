#include "cluster_tree.hpp"
#include "hmatrix.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static int check_tree(int Ns, int leaf_size) {
    hmc::ClusterTree tree(Ns, leaf_size);
    const int N = Ns * Ns;

    // perm is a permutation of 0..N-1
    std::vector<int> sorted = tree.perm();
    std::sort(sorted.begin(), sorted.end());
    for (int k = 0; k < N; ++k) CHECK(sorted[k] == k);
    for (int k = 0; k < N; ++k) CHECK(tree.perm()[tree.iperm()[k]] == k);

    int covered = 0;
    for (const auto& n : tree.nodes()) {
        // every node owns exactly the elements of its box
        CHECK(n.end - n.begin == (n.box.ix1 - n.box.ix0) * (n.box.iy1 - n.box.iy0));
        for (int p = n.begin; p < n.end; ++p) {
            const int k = tree.perm()[p];
            const int ix = k % Ns, iy = k / Ns;
            CHECK(ix >= n.box.ix0 && ix < n.box.ix1);
            CHECK(iy >= n.box.iy0 && iy < n.box.iy1);
        }
        if (n.leaf) {
            CHECK(n.end - n.begin <= leaf_size);
            covered += n.end - n.begin;
        } else {
            // children partition the parent's range
            int b = n.begin;
            for (int q = 0; q < 4; ++q) {
                if (n.child[q] < 0) continue;
                const auto& c = tree.nodes()[n.child[q]];
                CHECK(c.begin == b);
                b = c.end;
            }
            CHECK(b == n.end);
        }
    }
    CHECK(covered == N); // leaves tile the grid
    return 0;
}

int main() {
    for (int Ns : {1, 7, 32, 64})
        for (int ls : {1, 16, 32, 64})
            if (check_tree(Ns, ls) != 0) return 1;
    std::printf("test_hmatrix: cluster tree checks passed\n");

    // eta = 0 -> no block is admissible -> H-matrix must equal dense exactly
    {
        const int Ns = 16;
        hmc::BoussinesqKernel K(Ns, 1.0, 1.0);
        hmc::ClusterTree tree(Ns, 16);
        hmc::HMatrix H(K, tree, 0.0, 1e-6);
        const Eigen::MatrixXd S = K.assemble_dense();
        const Eigen::VectorXd p = Eigen::VectorXd::Random(Ns * Ns);
        const double err = (H.matvec(p) - S * p).norm() / (S * p).norm();
        CHECK(err < 1e-14);
        CHECK(H.info().n_lowrank == 0);
    }

    // eta = 2, aca_tol = 1e-6: matvec error well below 1e-5, real compression
    {
        const int Ns = 64;
        hmc::BoussinesqKernel K(Ns, 1.0, 1.0);
        hmc::ClusterTree tree(Ns, 32);
        hmc::HMatrix H(K, tree, 2.0, 1e-6);
        const Eigen::MatrixXd S = K.assemble_dense();

        std::mt19937 rng(42);
        std::normal_distribution<double> gauss(0.0, 1.0);
        for (int trial = 0; trial < 5; ++trial) {
            Eigen::VectorXd p(Ns * Ns);
            for (int i = 0; i < p.size(); ++i)
                p(i) = (trial == 0) ? 1.0 : gauss(rng); // uniform + random
            const Eigen::VectorXd ref = S * p;
            const double err = (H.matvec(p) - ref).norm() / ref.norm();
            CHECK(err < 1e-5);
        }

        const auto inf = H.info();
        std::printf("%s\n", inf.to_string().c_str());
        CHECK(inf.n_lowrank > 0);
        CHECK(inf.compression < 0.5);
    }

    std::printf("test_hmatrix: all checks passed\n");
    return 0;
}
