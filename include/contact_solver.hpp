#pragma once

#include <Eigen/Dense>
#include <functional>

namespace hmc {

struct ContactResult {
    Eigen::VectorXd pressure;
    Eigen::VectorXd displacement; // u = S p
    Eigen::VectorXd gap;          // u + g0 - approach (>= 0, = 0 in contact)
    double approach = 0.0;        // rigid-body shift (mean gap over contact)
    double objective = 0.0;       // W = 1/2 p.u + p.g0
    double error = 0.0;
    int iterations = 0;
    bool converged = false;
    double contact_fraction = 0.0;
    double mean_pressure = 0.0;
};

using MatVec = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

// Polonsky & Keer (Wear 231, 1999) projected CG for the constrained problem
//   min 1/2 p^T S p + p^T g0   s.t.  p >= 0,  mean(p) = p_bar.
// Two operator applications per iteration (gradient + line search).
ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol = 1e-8,
                            int max_iter = 5000);

} // namespace hmc
