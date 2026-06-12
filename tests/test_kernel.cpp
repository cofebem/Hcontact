#include "boussinesq_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <random>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main() {
    const int Ns = 64;
    const double L = 1.0, E = 1.7; // deliberately non-unit E*
    const double h = L / Ns;
    hmc::BoussinesqKernel K(Ns, L, E);

    // Exact self term: S_ii = 4 h ln(1 + sqrt(2)) / (pi E*)
    const double exact = 4.0 * h * std::log(1.0 + std::sqrt(2.0)) / (M_PI * E);
    CHECK(std::abs(K.entry(0, 0) / exact - 1.0) < 1e-14);
    CHECK(std::abs(K.entry(Ns * Ns / 2, Ns * Ns / 2) / exact - 1.0) < 1e-14);

    // Symmetry on random index pairs
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> d(0, Ns * Ns - 1);
    for (int k = 0; k < 200; ++k) {
        const int i = d(rng), j = d(rng);
        CHECK(std::abs(K.entry(i, j) - K.entry(j, i)) <= 1e-15);
    }

    // Deviation from the point-source approximation h^2 / (pi E* r):
    // ~3.7% at r = h, below 0.2% for r >= 5h (spec section 2.2).
    auto rel = [&](int dx, int dy) {
        const double r = h * std::hypot(double(dx), double(dy));
        const double ps = h * h / (M_PI * E * r);
        return std::abs(K.entry(0, dy * Ns + dx) / ps - 1.0);
    };
    CHECK(rel(1, 0) > 0.03 && rel(1, 0) < 0.05);
    CHECK(rel(1, 1) > 0.015 && rel(1, 1) < 0.035);
    CHECK(rel(5, 0) < 0.002);
    CHECK(rel(4, 3) < 0.002);
    CHECK(rel(20, 11) < 0.0005);

    // Positivity and monotone decay away from the source
    double prev = K.entry(0, 0);
    for (int dx = 1; dx < Ns; ++dx) {
        const double v = K.entry(0, dx);
        CHECK(v > 0.0 && v < prev);
        prev = v;
    }

    std::printf("test_kernel: all checks passed\n");
    return 0;
}
