#include "fourier_precond.hpp"

#include <unsupported/Eigen/FFT>

#include <cmath>

namespace hmc {

FourierPreconditioner::FourierPreconditioner(int Ns) : Ns_(Ns), w_(Ns, Ns) {
    // integer wavenumbers k = i (i < Ns/2) else i - Ns; symbol |k|, DC zeroed.
    // Absolute scale is irrelevant (cancels in CG), so 2π/L is dropped. The
    // symbol is stored in float: integer |k| < Ns is exact in float for the
    // grids used, and the float path multiplies with it directly.
    auto kof = [Ns](int i) { return (i < Ns / 2) ? i : i - Ns; };
    for (int ky = 0; ky < Ns; ++ky)
        for (int kx = 0; kx < Ns; ++kx)
            w_(ky, kx) = std::hypot(static_cast<float>(kof(kx)),
                                    static_cast<float>(kof(ky)));
    w_(0, 0) = 0.0f;
}

// 2D FFT by 1D transforms over rows then columns (scalar-templated: S is the
// real type, so the working buffers are double or float accordingly).
template <class S>
static void fft2_fwd(Eigen::FFT<S>& fft,
                     const Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& in,
                     Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>& out) {
    const int n = static_cast<int>(in.rows());
    Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic> tmp(n, n);
    std::vector<S> rin(n);
    std::vector<std::complex<S>> rout(n), cin(n), cout(n);
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

template <class S>
static void fft2_inv(Eigen::FFT<S>& fft,
                     const Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>& in,
                     Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>& out) {
    const int n = static_cast<int>(in.rows());
    Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic> tmp(n, n);
    std::vector<std::complex<S>> cin(n), cout(n), rin(n);
    std::vector<S> rout(n);
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

// Scalar-templated preconditioner apply. w_ (float symbol) is used directly in
// float and cast up in double.
template <class S>
static Eigen::Matrix<S, Eigen::Dynamic, 1>
apply_t(int Ns, const Eigen::MatrixXf& wf,
        const Eigen::Matrix<S, Eigen::Dynamic, 1>& g,
        const std::vector<std::uint8_t>& contact) {
    using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using Mat = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    using CMat = Eigen::Matrix<std::complex<S>, Eigen::Dynamic, Eigen::Dynamic>;
    const int N = Ns * Ns;
    Mat R = Mat::Zero(Ns, Ns);
    for (int i = 0; i < N; ++i)
        if (contact[i]) R(i / Ns, i % Ns) = g(i); // i = iy*Ns + ix

    Eigen::FFT<S> fft;
    CMat C(Ns, Ns);
    fft2_fwd<S>(fft, R, C);
    C.array() *= wf.template cast<S>().array(); // real symbol
    Mat Z(Ns, Ns);
    fft2_inv<S>(fft, C, Z);

    Vec z = Vec::Zero(N);
    double zsum = 0.0;
    int nc = 0;
    for (int i = 0; i < N; ++i)
        if (contact[i]) { z(i) = Z(i / Ns, i % Ns); zsum += z(i); ++nc; }
    if (nc) {
        const S zmean = static_cast<S>(zsum / nc);
        for (int i = 0; i < N; ++i)
            if (contact[i]) z(i) -= zmean;
    }
    return z;
}

Eigen::VectorXd
FourierPreconditioner::apply(const Eigen::VectorXd& g,
                             const std::vector<std::uint8_t>& contact) const {
    return apply_t<double>(Ns_, w_, g, contact);
}

Eigen::VectorXf
FourierPreconditioner::apply_single(const Eigen::VectorXf& g,
                                    const std::vector<std::uint8_t>& contact) const {
    return apply_t<float>(Ns_, w_, g, contact);
}

} // namespace hmc
