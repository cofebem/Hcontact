#include "boussinesq_kernel.hpp"

namespace hmc {

double love_uz(double x, double y, double a, double b) {
    const double xpa = x + a, xma = x - a;
    const double ypb = y + b, ymb = y - b;
    const double Rpp = std::hypot(xpa, ypb);
    const double Rpm = std::hypot(xpa, ymb);
    const double Rmp = std::hypot(xma, ypb);
    const double Rmm = std::hypot(xma, ymb);
    return xpa * std::log((ypb + Rpp) / (ymb + Rpm))
         + ypb * std::log((xpa + Rpp) / (xma + Rmp))
         + xma * std::log((ymb + Rmm) / (ypb + Rmp))
         + ymb * std::log((xma + Rmm) / (xpa + Rpm));
}

BoussinesqKernel::BoussinesqKernel(int Ns, double L, double E_star)
    : Ns_(Ns), L_(L), h_(L / Ns), E_star_(E_star),
      table_(static_cast<std::size_t>(Ns) * Ns) {
    const double a = 0.5 * h_;
    const double scale = 1.0 / (M_PI * E_star_);
    for (int dy = 0; dy < Ns_; ++dy)
        for (int dx = 0; dx < Ns_; ++dx)
            table_[static_cast<std::size_t>(dy) * Ns_ + dx] =
                scale * love_uz(dx * h_, dy * h_, a, a);
}

Eigen::MatrixXd BoussinesqKernel::assemble_dense() const {
    const int N = size();
    Eigen::MatrixXd S(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            S(i, j) = entry(i, j);
    return S;
}

} // namespace hmc
