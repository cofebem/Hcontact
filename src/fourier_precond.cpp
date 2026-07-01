#include "fourier_precond.hpp"

#include <unsupported/Eigen/FFT>

#include <cmath>

namespace hmc {

FourierPreconditioner::FourierPreconditioner(int Ns) : Ns_(Ns), w_(Ns, Ns) {
    // integer wavenumbers k = i (i < Ns/2) else i - Ns; symbol |k|, DC zeroed.
    // Absolute scale is irrelevant (cancels in CG), so 2π/L is dropped.
    auto kof = [Ns](int i) { return (i < Ns / 2) ? i : i - Ns; };
    for (int ky = 0; ky < Ns; ++ky)
        for (int kx = 0; kx < Ns; ++kx)
            w_(ky, kx) = std::hypot(static_cast<double>(kof(kx)),
                                    static_cast<double>(kof(ky)));
    w_(0, 0) = 0.0;
}

// 2D FFT by 1D transforms over rows then columns.
static void fft2_fwd(Eigen::FFT<double>& fft, const Eigen::MatrixXd& in,
                     Eigen::MatrixXcd& out) {
    const int n = static_cast<int>(in.rows());
    Eigen::MatrixXcd tmp(n, n);
    std::vector<double> rin(n);
    std::vector<std::complex<double>> rout(n), cin(n), cout(n);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) rin[c] = in(r, c);
        fft.fwd(rout, rin);
        for (int c = 0; c < n; ++c) tmp(r, c) = rout[c];
    }
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < n; ++r) cin[r] = tmp(r, c);
        fft.fwd(cout, cin);
        for (int r = 0; r < n; ++r) out(r, c) = cout[r];
    }
}

static void fft2_inv(Eigen::FFT<double>& fft, const Eigen::MatrixXcd& in,
                     Eigen::MatrixXd& out) {
    const int n = static_cast<int>(in.rows());
    Eigen::MatrixXcd tmp(n, n);
    std::vector<std::complex<double>> cin(n), cout(n), rin(n);
    std::vector<double> rout(n);
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < n; ++r) cin[r] = in(r, c);
        fft.inv(cout, cin);
        for (int r = 0; r < n; ++r) tmp(r, c) = cout[r];
    }
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) rin[c] = tmp(r, c);
        fft.inv(rout, rin);          // Eigen scales each inverse by 1/n
        for (int c = 0; c < n; ++c) out(r, c) = rout[c];
    }
}

Eigen::VectorXd
FourierPreconditioner::apply(const Eigen::VectorXd& g,
                             const std::vector<std::uint8_t>& contact) const {
    const int N = Ns_ * Ns_;
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(Ns_, Ns_);
    for (int i = 0; i < N; ++i)
        if (contact[i]) R(i / Ns_, i % Ns_) = g(i); // i = iy*Ns + ix

    Eigen::FFT<double> fft;
    Eigen::MatrixXcd C(Ns_, Ns_);
    fft2_fwd(fft, R, C);
    C.array() *= w_.array(); // real symbol
    Eigen::MatrixXd Z(Ns_, Ns_);
    fft2_inv(fft, C, Z);

    Eigen::VectorXd z = Eigen::VectorXd::Zero(N);
    double zsum = 0.0;
    int nc = 0;
    for (int i = 0; i < N; ++i)
        if (contact[i]) { z(i) = Z(i / Ns_, i % Ns_); zsum += z(i); ++nc; }
    if (nc) {
        const double zmean = zsum / nc;
        for (int i = 0; i < N; ++i)
            if (contact[i]) z(i) -= zmean;
    }
    return z;
}

Eigen::VectorXf
FourierPreconditioner::apply_single(const Eigen::VectorXf& g,
                                    const std::vector<std::uint8_t>& contact) const {
    // The float residual is scattered into a double FFT (the transform is the
    // memory-cheap part relative to the O(N) solver vectors); result cast back.
    const int N = Ns_ * Ns_;
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(Ns_, Ns_);
    for (int i = 0; i < N; ++i)
        if (contact[i]) R(i / Ns_, i % Ns_) = static_cast<double>(g(i));

    Eigen::FFT<double> fft;
    Eigen::MatrixXcd C(Ns_, Ns_);
    fft2_fwd(fft, R, C);
    C.array() *= w_.array();
    Eigen::MatrixXd Z(Ns_, Ns_);
    fft2_inv(fft, C, Z);

    Eigen::VectorXf z = Eigen::VectorXf::Zero(N);
    double zsum = 0.0;
    int nc = 0;
    for (int i = 0; i < N; ++i)
        if (contact[i]) { z(i) = static_cast<float>(Z(i / Ns_, i % Ns_));
                          zsum += z(i); ++nc; }
    if (nc) {
        const float zmean = static_cast<float>(zsum / nc);
        for (int i = 0; i < N; ++i)
            if (contact[i]) z(i) -= zmean;
    }
    return z;
}

} // namespace hmc
