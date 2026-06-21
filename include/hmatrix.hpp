#pragma once

#include "boussinesq_kernel.hpp"
#include "cluster_tree.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

namespace hmc {

struct HBlock {
    int row_begin, row_size, col_begin, col_size; // permuted (cluster) indexing
    bool dense;
    Eigen::MatrixXd D;    // dense storage when dense == true
    Eigen::MatrixXd U, V; // low-rank factors, block ~= U * V
};

struct HMatrixInfo {
    int n = 0;
    int n_dense = 0, n_lowrank = 0;
    std::int64_t bytes = 0;
    int max_rank = 0;
    double avg_rank = 0.0;
    double compression = 1.0; // stored bytes / (8 n^2)
    std::string to_string() const;
};

// Hierarchical approximation of the Boussinesq influence matrix: quad-tree
// block partition, Chebyshev admissibility min(diam) <= eta * dist, ACA
// (partial pivoting or ACA-GP) for admissible blocks, dense leaves otherwise.
class HMatrix {
public:
    // use_acagp=true: ACA with geometric pivot selection (Yastrebov 2025).
    // central_fraction: relative radius of central subset (fraction of cluster diameter).
    // inline_svd_tol > 0: apply truncated SVD recompression block-by-block right after
    //   ACA fill, keeping peak memory at (one block uncompressed + all previous compressed).
    //   Enables Ns=1024 without first accumulating the full uncompressed matrix.
    HMatrix(const BoussinesqKernel& kernel, const ClusterTree& tree,
            double eta, double aca_tol,
            bool use_acagp = false, double central_fraction = 0.3,
            double inline_svd_tol = 0.0);

    Eigen::VectorXd matvec(const Eigen::VectorXd& p) const;
    HMatrixInfo info() const;
    const std::vector<HBlock>& blocks() const { return blocks_; }

    // Post-ACA recompression via truncated SVD: drops singular values below
    // svd_tol * sigma_max for each low-rank block.
    void recompress(double svd_tol);

private:
    void build(int t, int s);
    void fill_dense(HBlock& blk) const;
    void fill_aca(HBlock& blk) const;
    void fill_aca_gp(HBlock& blk) const;
    static void recompress_block(HBlock& blk, double svd_tol);
    double entry_perm(int pi, int pj) const {
        return kernel_->entry(tree_->perm()[pi], tree_->perm()[pj]);
    }

    const BoussinesqKernel* kernel_;
    const ClusterTree* tree_;
    double eta_, tol_;
    bool use_acagp_;
    double central_fraction_;
    double inline_svd_tol_;
    std::vector<HBlock> blocks_;
};

} // namespace hmc
