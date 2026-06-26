#include "contact_solver.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace hmc {

ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol, int max_iter, bool use_pr) {
    const int N = static_cast<int>(g0.size());
    if (N == 0 || p_bar <= 0.0)
        throw std::invalid_argument("solve_contact: empty gap or p_bar <= 0");

    const double P_total = p_bar * N;
    double g_scale = g0.maxCoeff() - g0.minCoeff();
    if (g_scale <= 0.0) g_scale = 1.0;

    Eigen::VectorXd p = Eigen::VectorXd::Constant(N, p_bar);
    Eigen::VectorXd t = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd u(N), g(N), g_prev(N);
    g_prev.setZero();

    ContactResult res;
    double G_old = 1.0;
    double delta = 0.0; // conjugation switch: 0 after the active set changed
    double alpha = 0.0;

    int it = 0;
    for (it = 0; it < max_iter; ++it) {
        u = S(p);
        g = u + g0;

        // rigid-body shift: zero mean gap over the current contact set
        double gsum = 0.0;
        int nc = 0;
        for (int i = 0; i < N; ++i)
            if (p(i) > 0.0) { gsum += g(i); ++nc; }
        alpha = nc ? gsum / nc : 0.0;
        g.array() -= alpha;

        // complementarity error: sum p |g| (g = 0 where p > 0 at the solution)
        double e = 0.0;
        for (int i = 0; i < N; ++i) e += p(i) * std::abs(g(i));
        res.error = e / (P_total * g_scale);
        if (res.error < tol) {
            res.converged = true;
            break;
        }

        // conjugate direction restricted to the contact set
        double G = 0.0, G_pr = 0.0;
        for (int i = 0; i < N; ++i) {
            if (p(i) > 0.0) {
                G    += g(i) * g(i);
                G_pr += g(i) * (g(i) - g_prev(i));
            }
        }
        double beta_val = use_pr ? std::max(0.0, G_pr / G_old) : G / G_old;
        const double beta = delta * beta_val;
        for (int i = 0; i < N; ++i)
            t(i) = (p(i) > 0.0) ? g(i) + beta * t(i) : 0.0;
        g_prev = g;
        G_old = G;

        // line search tau = <g, t> / <S t, t> on the contact set
        Eigen::VectorXd r = S(t);
        double rsum = 0.0;
        for (int i = 0; i < N; ++i)
            if (p(i) > 0.0) rsum += r(i);
        const double rmean = nc ? rsum / nc : 0.0;
        double num = 0.0, den = 0.0;
        for (int i = 0; i < N; ++i)
            if (p(i) > 0.0) {
                num += g(i) * t(i);
                den += (r(i) - rmean) * t(i);
            }
        if (den <= 0.0) { // direction lost positive curvature: steepest descent
            delta = 0.0;
            continue;
        }
        const double tau = num / den;

        // update, project onto p >= 0
        for (int i = 0; i < N; ++i)
            if (p(i) > 0.0) p(i) -= tau * t(i);
        p = p.cwiseMax(0.0);

        // overlap correction: points released to p = 0 but penetrating
        bool overlap = false;
        for (int i = 0; i < N; ++i)
            if (p(i) == 0.0 && g(i) < 0.0) {
                p(i) -= tau * g(i);
                overlap = true;
            }
        delta = overlap ? 0.0 : 1.0;

        // enforce the load balance
        const double total = p.sum();
        if (total > 0.0)
            p *= P_total / total;
        else
            p.setConstant(p_bar);
    }

    u = S(p);
    g = u + g0;
    double gsum = 0.0;
    int nc = 0;
    for (int i = 0; i < N; ++i)
        if (p(i) > 0.0) { gsum += g(i); ++nc; }
    alpha = nc ? gsum / nc : 0.0;

    res.pressure = p;
    res.displacement = u;
    res.gap = g.array() - alpha;
    res.approach = alpha;
    res.objective = 0.5 * p.dot(u) + p.dot(g0);
    res.iterations = it;
    res.contact_fraction = double(nc) / N;
    res.mean_pressure = p.mean();
    return res;
}

} // namespace hmc
