#include "contact_solver.hpp"

#include <cmath>
#include <stdexcept>

namespace hmc {

template <class Real>
ContactResult solve_contact_impl(const MatVecT<Real>& S, const VecT<Real>& g0,
                                 Real p_bar, Real tol, int max_iter, bool use_pr,
                                 const PrecondT<Real>& precond,
                                 const VecT<Real>* p_init, bool light) {
    using Vec = VecT<Real>;
    const int N = static_cast<int>(g0.size());
    if (N == 0 || p_bar <= Real(0))
        throw std::invalid_argument("solve_contact: empty gap or p_bar <= 0");

    const Real P_total = p_bar * N;
    Real g_scale = g0.maxCoeff() - g0.minCoeff();
    if (g_scale <= Real(0)) g_scale = Real(1);

    Vec p;
    if (p_init) {
        if (static_cast<int>(p_init->size()) != N)
            throw std::invalid_argument("solve_contact: p_init size mismatch");
        p = p_init->cwiseMax(Real(0));
        const Real s = p.sum();
        p = (s > Real(0)) ? (p * (P_total / s)).eval() : Vec::Constant(N, p_bar);
    } else {
        p = Vec::Constant(N, p_bar);
    }
    Vec t = Vec::Zero(N);
    Vec u(N), g(N), g_prev(N), z(N);
    std::vector<std::uint8_t> contact(N, 0);
    g_prev.setZero();

    ContactResult res;
    Real G_old = 1.0;
    Real delta = 0.0; // conjugation switch: 0 after the active set changed
    Real alpha = 0.0;

    int it = 0;
    for (it = 0; it < max_iter; ++it) {
        u = S(p);
        g = u + g0;

        // rigid-body shift: zero mean gap over the current contact set
        Real gsum = 0.0;
        int nc = 0;
        for (int i = 0; i < N; ++i)
            if (p(i) > Real(0)) { gsum += g(i); ++nc; }
        alpha = nc ? gsum / nc : Real(0);
        g.array() -= alpha;

        // complementarity error: sum p |g| (g = 0 where p > 0 at the solution)
        Real e = 0.0;
        for (int i = 0; i < N; ++i) e += p(i) * std::abs(g(i));
        res.error = static_cast<double>(e / (P_total * g_scale));
        if (res.error < static_cast<double>(tol)) {
            res.converged = true;
            break;
        }

        // preconditioned residual z (z = g on contact when no preconditioner)
        for (int i = 0; i < N; ++i) contact[i] = (p(i) > Real(0)) ? 1 : 0;
        if (precond) {
            z = precond(g, contact);
        } else {
            for (int i = 0; i < N; ++i) z(i) = contact[i] ? g(i) : Real(0);
        }

        // conjugate direction restricted to the contact set (M-inner product)
        Real G = 0.0, G_pr = 0.0;
        for (int i = 0; i < N; ++i)
            if (contact[i]) {
                G    += z(i) * g(i);
                G_pr += z(i) * (g(i) - g_prev(i));
            }
        Real beta_val = use_pr ? std::max(Real(0), G_pr / G_old) : G / G_old;
        const Real beta = delta * beta_val;
        for (int i = 0; i < N; ++i)
            t(i) = contact[i] ? z(i) + beta * t(i) : Real(0);
        g_prev = g;
        G_old = G;

        // line search tau = <g, t> / <S t, t> on the contact set
        Vec r = S(t);
        Real rsum = 0.0;
        for (int i = 0; i < N; ++i)
            if (p(i) > Real(0)) rsum += r(i);
        const Real rmean = nc ? rsum / nc : Real(0);
        Real num = 0.0, den = 0.0;
        for (int i = 0; i < N; ++i)
            if (p(i) > Real(0)) {
                num += g(i) * t(i);
                den += (r(i) - rmean) * t(i);
            }
        if (den <= Real(0)) { // direction lost positive curvature: steepest descent
            delta = 0.0;
            continue;
        }
        const Real tau = num / den;

        // update, project onto p >= 0
        for (int i = 0; i < N; ++i)
            if (p(i) > Real(0)) p(i) -= tau * t(i);
        p = p.cwiseMax(Real(0));

        // overlap correction: points released to p = 0 but penetrating
        bool overlap = false;
        for (int i = 0; i < N; ++i)
            if (p(i) == Real(0) && g(i) < Real(0)) {
                p(i) -= tau * g(i);
                overlap = true;
            }
        delta = overlap ? Real(0) : Real(1);

        // enforce the load balance
        const Real total = p.sum();
        if (total > Real(0))
            p *= P_total / total;
        else
            p.setConstant(p_bar);
    }

    u = S(p);
    g = u + g0;
    Real gsum = 0.0;
    int nc = 0;
    for (int i = 0; i < N; ++i)
        if (p(i) > Real(0)) { gsum += g(i); ++nc; }
    alpha = nc ? gsum / nc : Real(0);

    res.pressure = p.template cast<double>();
    if (!light) {
        res.displacement = u.template cast<double>();
        res.gap = (g.array() - alpha).template cast<double>();
    }
    res.approach = static_cast<double>(alpha);
    res.objective = static_cast<double>(Real(0.5) * p.dot(u) + p.dot(g0));
    res.iterations = it;
    res.contact_fraction = double(nc) / N;
    res.mean_pressure = static_cast<double>(p.mean());
    return res;
}

// explicit instantiations
template ContactResult solve_contact_impl<double>(
    const MatVecT<double>&, const VecT<double>&, double, double, int, bool,
    const PrecondT<double>&, const VecT<double>*, bool);
template ContactResult solve_contact_impl<float>(
    const MatVecT<float>&, const VecT<float>&, float, float, int, bool,
    const PrecondT<float>&, const VecT<float>*, bool);

ContactResult solve_contact(const MatVec& S, const Eigen::VectorXd& g0,
                            double p_bar, double tol, int max_iter, bool use_pr,
                            const Precond& precond, const Eigen::VectorXd* p_init,
                            bool light) {
    return solve_contact_impl<double>(S, g0, p_bar, tol, max_iter, use_pr,
                                      precond, p_init, light);
}

} // namespace hmc
