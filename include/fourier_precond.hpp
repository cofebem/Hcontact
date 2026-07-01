#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace hmc {

// Spectral preconditioner for the projected CG. The Boussinesq operator S has
// Fourier symbol Ŝ(q) ∝ 1/|q|, so M⁻¹ with symbol ∝ |q| collapses κ(S) ∼ Ns to
// ≈ O(1). Applied per iteration by FFT to the residual masked to the contact
// set; the result is kept on the contact set and its contact-mean removed (the
// total load / mean is fixed by the constraint, not by CG). The overall scale
// of M⁻¹ is irrelevant (it cancels in CG), so the bare wavenumber |k| is used
// and the q=0 mode is zeroed.
class FourierPreconditioner {
public:
    explicit FourierPreconditioner(int Ns);

    // z = M⁻¹ g, masked to {i : contact[i] != 0} and mean-zeroed over it.
    Eigen::VectorXd apply(const Eigen::VectorXd& g,
                          const std::vector<std::uint8_t>& contact) const;

    // Single-precision variant (FFT done in float; symbol applied as float).
    Eigen::VectorXf apply_single(const Eigen::VectorXf& g,
                                 const std::vector<std::uint8_t>& contact) const;

private:
    int Ns_;
    Eigen::MatrixXd w_; // Ns×Ns wavenumber symbol |k|, w_(0,0)=0
};

} // namespace hmc
