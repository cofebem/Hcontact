#pragma once

#include <Eigen/Dense>
#include <vector>

namespace hmc {

// Tensor-product Chebyshev interpolation basis (black-box FMM, Fong & Darve 2009).
// q Chebyshev points of the first kind on [-1, 1]; weights(t) gives the
// interpolation weights w_m(t) such that sum_m w_m(t) f(nodes[m]) interpolates
// f at t and is exact for polynomials of degree < q.
struct ChebBasis {
    int q;
    std::vector<double> nodes; // q Chebyshev points in [-1,1]

    explicit ChebBasis(int q);

    Eigen::VectorXd weights(double t) const;                          // length q
    Eigen::MatrixXd weights_at(const std::vector<double>& pts) const; // (pts x q)
};

} // namespace hmc
