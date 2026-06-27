#include "cheb_basis.hpp"

#include <cmath>

namespace hmc {

ChebBasis::ChebBasis(int q_) : q(q_), nodes(q_) {
    for (int m = 0; m < q; ++m)
        nodes[m] = std::cos((2.0 * m + 1.0) * M_PI / (2.0 * q));
}

// w_m(t) = 1/q + (2/q) * sum_{n=1}^{q-1} T_n(nodes[m]) T_n(t)
Eigen::VectorXd ChebBasis::weights(double t) const {
    Eigen::VectorXd w = Eigen::VectorXd::Constant(q, 1.0 / q);
    for (int m = 0; m < q; ++m) {
        const double xm = nodes[m];
        double T0m = 1.0, T1m = xm; // T_0, T_1 at node m
        double T0t = 1.0, T1t = t;  // T_0, T_1 at t
        double acc = 0.0;
        for (int n = 1; n < q; ++n) {
            double Tnm, Tnt;
            if (n == 1) {
                Tnm = T1m;
                Tnt = T1t;
            } else {
                Tnm = 2.0 * xm * T1m - T0m;
                Tnt = 2.0 * t * T1t - T0t;
                T0m = T1m; T1m = Tnm;
                T0t = T1t; T1t = Tnt;
            }
            acc += Tnm * Tnt;
        }
        w(m) += (2.0 / q) * acc;
    }
    return w;
}

Eigen::MatrixXd ChebBasis::weights_at(const std::vector<double>& pts) const {
    Eigen::MatrixXd W(static_cast<int>(pts.size()), q);
    for (int i = 0; i < static_cast<int>(pts.size()); ++i)
        W.row(i) = weights(pts[i]).transpose();
    return W;
}

} // namespace hmc
