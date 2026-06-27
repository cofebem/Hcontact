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

    H2Info info_;

    double far_kernel(double dx, double dy) const; // g(dx,dy)
};

} // namespace hmc
