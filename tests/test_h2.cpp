#include "cheb_basis.hpp"

#include <cmath>
#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAILED: %s (line %d)\n", #cond, __LINE__);        \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static int test_cheb() {
    hmc::ChebBasis b(5);
    CHECK(static_cast<int>(b.nodes.size()) == 5);

    // partition of unity: weights sum to 1 at any t
    for (double t : {-0.9, -0.3, 0.0, 0.4, 1.0}) {
        const double s = b.weights(t).sum();
        CHECK(std::abs(s - 1.0) < 1e-12);
    }

    // interpolation exactness for polynomials of degree < q
    auto f = [](double x) {
        return 3 - 2 * x + 0.5 * x * x - x * x * x + 0.25 * x * x * x * x;
    };
    Eigen::VectorXd fnodes(5);
    for (int m = 0; m < 5; ++m) fnodes(m) = f(b.nodes[m]);
    for (double t : {-0.7, 0.1, 0.85}) {
        const double interp = b.weights(t).dot(fnodes);
        CHECK(std::abs(interp - f(t)) < 1e-10);
    }
    return 0;
}

int main() { return test_cheb(); }
