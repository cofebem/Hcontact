#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace hmc {

// Love (1929) closed-form surface displacement at (x, y) due to uniform unit
// pressure on the rectangle [-a, a] x [-b, b], scaled by pi * E*
// (Johnson, Contact Mechanics, eq. 3.25).
double love_uz(double x, double y, double a, double b);

// Boussinesq influence matrix S for an Ns x Ns grid of square elements of
// side h = L / Ns carrying uniform pressure. The kernel is translation
// invariant: S_ij depends only on (|ix - jx|, |iy - jy|), so all N^2 entries
// are served from an Ns x Ns lookup table built once at construction.
class BoussinesqKernel {
public:
    BoussinesqKernel(int Ns, double L, double E_star);

    int    grid_size() const { return Ns_; }
    int    size() const { return Ns_ * Ns_; }
    double element_size() const { return h_; }
    double domain_size() const { return L_; }
    double E_star() const { return E_star_; }

    // S_ij for flat row-major indices k = iy * Ns + ix.
    double entry(int i, int j) const {
        const int dx = std::abs(i % Ns_ - j % Ns_);
        const int dy = std::abs(i / Ns_ - j / Ns_);
        return table_[static_cast<std::size_t>(dy) * Ns_ + dx];
    }

    Eigen::MatrixXd assemble_dense() const;

private:
    int Ns_;
    double L_, h_, E_star_;
    std::vector<double> table_;
};

} // namespace hmc
