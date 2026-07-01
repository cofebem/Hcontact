#pragma once

#include "boussinesq_kernel.hpp"
#include "cheb_basis.hpp"
#include "uniform_quadtree.hpp"

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <vector>

namespace hmc {

struct H2Params {
    int leaf_side = 8;   // square leaf side in elements (power of two)
    int q = 4;           // Chebyshev interpolation order (rank r = q*q)
    int near_radius = 1; // direct near field within this many leaf boxes
};

struct H2Info {
    int N = 0, Ns = 0, nlevels = 0, leaf_side = 0, q = 0, r = 0;
    int n_boxes = 0, n_leaves = 0;
    std::int64_t n_near_interactions = 0, n_far_interactions = 0;
    int n_unique_couplings = 0, n_near_stencils = 0;
    std::int64_t bytes_coupling = 0, bytes_near = 0, bytes_buffers = 0, bytes_total = 0;
};

// Matrix-free black-box FMM (Fong & Darve 2009) operator for the translation-
// invariant Boussinesq half-space kernel on a uniform Ns x Ns grid. Far field
// via tensor-product Chebyshev interpolation with cached, translation-invariant
// transfer (M2M/L2L) and coupling (M2L) operators; near field via exact Love
// stencils cached by relative leaf offset. O(N) memory and matvec.
class H2Operator {
public:
    H2Operator(const BoussinesqKernel& kernel, H2Params params);

    void build();

    // u = S x, with x and u in natural flat order (global = iy*Ns + ix).
    Eigen::VectorXd matvec(const Eigen::VectorXd& x) const;

    // Single-precision matvec: builds float copies of the caches on first use
    // (idempotent) and runs the passes in float, halving the O(N) working set.
    Eigen::VectorXf matvec_single(const Eigen::VectorXf& x) const;
    void build_single_caches() const;

    H2Info info() const { return info_; }
    void print_statistics() const;

    int n_far_interactions() const { return static_cast<int>(info_.n_far_interactions); }
    int n_unique_couplings() const { return info_.n_unique_couplings; }

private:
    struct FarInter { int source_box; int coupling_id; };
    struct NearInter { int source_box; int stencil_id; };

    // 1D element-center coordinates of a box, normalized to [-1, 1] (independent
    // of the box origin; depends only on the side length).
    std::vector<double> centers_norm(int side) const;

    const BoussinesqKernel* kernel_;
    H2Params p_;
    int Ns_, q_, q2_, ls_, ls2_;
    double h_, scale_; // scale_ = 1 / (pi E*)

    ChebBasis cheb_;
    UniformQuadTree tree_;
    Eigen::MatrixXd Wleaf_;                 // (ls x q) leaf interpolation
    std::array<Eigen::MatrixXd, 4> R_;      // (q2 x q2) M2M per child quadrant

    std::vector<int> leaves_;
    std::vector<std::vector<FarInter>> far_by_target_;   // per box id
    std::vector<std::vector<NearInter>> near_by_leaf_;   // per leaf index

    std::vector<Eigen::MatrixXd> couplings_;             // q2 x q2, by id
    std::vector<Eigen::MatrixXd> near_stencils_;         // ls2 x ls2, by id

    // float copies of the numeric caches (built lazily by build_single_caches)
    mutable bool have_single_ = false;
    mutable Eigen::MatrixXf Wleaf_f_;
    mutable std::array<Eigen::MatrixXf, 4> R_f_;
    mutable std::vector<Eigen::MatrixXf> couplings_f_, near_stencils_f_;

    H2Info info_;

    double far_kernel(double dx, double dy) const; // g(dx,dy)

    // Scalar-templated matvec: the passes are identical for double/float, only
    // the numeric caches differ. The tree/leaf/interaction indices are shared.
    template <class S>
    Eigen::Matrix<S, Eigen::Dynamic, 1> matvec_impl(
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
        const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& Wleaf,
        const std::array<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>, 4>& R,
        const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& couplings,
        const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& near_st) const;
};

// ── matvec_impl (header-defined member template) ──────────────────────────────
template <class S>
Eigen::Matrix<S, Eigen::Dynamic, 1> H2Operator::matvec_impl(
    const Eigen::Matrix<S, Eigen::Dynamic, 1>& x,
    const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& Wleaf,
    const std::array<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>, 4>& R,
    const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& couplings,
    const std::vector<Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>>& near_st) const {
    using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using Mat = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    const auto& boxes = tree_.boxes();
    const int nbox = static_cast<int>(boxes.size());
    std::vector<Vec> M(nbox), L(nbox);
    for (int b = 0; b < nbox; ++b) {
        M[b] = Vec::Zero(q2_);
        L[b] = Vec::Zero(q2_);
    }
    Vec y = Vec::Zero(x.size());

    // P2M
#pragma omp parallel for schedule(dynamic, 16)
    for (int li = 0; li < static_cast<int>(leaves_.size()); ++li) {
        const int t = leaves_[li];
        const int ix0 = boxes[t].ix0, iy0 = boxes[t].iy0;
        Mat Xs(ls_, ls_);
        for (int ly = 0; ly < ls_; ++ly)
            for (int lx = 0; lx < ls_; ++lx)
                Xs(ly, lx) = x((iy0 + ly) * Ns_ + (ix0 + lx));
        const Mat Mmat = Wleaf.transpose() * Xs * Wleaf;
        Vec& Mt = M[t];
        for (int ay = 0; ay < q_; ++ay)
            for (int ax = 0; ax < q_; ++ax)
                Mt(ax + q_ * ay) = Mmat(ay, ax);
    }

    // M2M
    for (int l = tree_.leaf_level() - 1; l >= 0; --l) {
        const int b0 = tree_.level_begin(l);
        const int nb = (1 << l) * (1 << l);
#pragma omp parallel for schedule(dynamic, 16)
        for (int k = 0; k < nb; ++k) {
            const int b = b0 + k;
            Vec acc = Vec::Zero(q2_);
            for (int c = 0; c < 4; ++c) {
                const int cc = boxes[b].child[c];
                if (cc >= 0) acc.noalias() += R[c] * M[cc];
            }
            M[b] = acc;
        }
    }

    // M2L
#pragma omp parallel for schedule(dynamic, 16)
    for (int t = 0; t < nbox; ++t) {
        Vec acc = Vec::Zero(q2_);
        for (const FarInter& fi : far_by_target_[t])
            acc.noalias() += couplings[fi.coupling_id] * M[fi.source_box];
        L[t] = acc;
    }

    // L2L
    for (int l = 1; l <= tree_.leaf_level(); ++l) {
        const int b0 = tree_.level_begin(l);
        const int nb = (1 << l) * (1 << l);
#pragma omp parallel for schedule(dynamic, 16)
        for (int k = 0; k < nb; ++k) {
            const int b = b0 + k;
            const int c = (boxes[b].bx & 1) + 2 * (boxes[b].by & 1);
            L[b].noalias() += R[c].transpose() * L[boxes[b].parent];
        }
    }

    // L2P + near
#pragma omp parallel for schedule(dynamic, 16)
    for (int li = 0; li < static_cast<int>(leaves_.size()); ++li) {
        const int t = leaves_[li];
        const int ix0 = boxes[t].ix0, iy0 = boxes[t].iy0;
        Mat Lmat(q_, q_);
        for (int ay = 0; ay < q_; ++ay)
            for (int ax = 0; ax < q_; ++ax)
                Lmat(ay, ax) = L[t](ax + q_ * ay);
        Mat Yt = Wleaf * Lmat * Wleaf.transpose();

        Vec yloc = Vec::Zero(ls2_);
        for (const NearInter& ni : near_by_leaf_[li]) {
            const int s = ni.source_box;
            const int six0 = boxes[s].ix0, siy0 = boxes[s].iy0;
            Vec xs(ls2_);
            for (int ly = 0; ly < ls_; ++ly)
                for (int lx = 0; lx < ls_; ++lx)
                    xs(lx + ls_ * ly) = x((siy0 + ly) * Ns_ + (six0 + lx));
            yloc.noalias() += near_st[ni.stencil_id] * xs;
        }
        for (int ly = 0; ly < ls_; ++ly)
            for (int lx = 0; lx < ls_; ++lx)
                y((iy0 + ly) * Ns_ + (ix0 + lx)) = Yt(ly, lx) + yloc(lx + ls_ * ly);
    }
    return y;
}

} // namespace hmc
