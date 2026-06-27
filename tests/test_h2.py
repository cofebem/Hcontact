"""H2 backend: matvec accuracy vs dense, and Hertz integration vs H-matrix."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import hmatrix_contact as hc


def test_matvec_accuracy():
    Ns = 64
    dense = hc.ContactSolver(grid_size=Ns, use_hmatrix=False)
    h2 = hc.ContactSolver(grid_size=Ns, backend="h2", q=6)
    x = np.random.default_rng(0).standard_normal(Ns * Ns)
    ref = dense.matvec(x).ravel()
    got = h2.matvec(x).ravel()
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"matvec rel err (q=6): {rel:.3e}")
    assert rel < 1e-4, rel


def hertz_gap(Ns, R=1.0, delta=0.01, L=1.0):
    xs = (np.arange(Ns) + 0.5) * L / Ns - L / 2
    X, Y = np.meshgrid(xs, xs)
    return (X**2 + Y**2) / (2 * R) - delta


def test_hertz_integration():
    Ns = 64
    g0 = hertz_gap(Ns).ravel()
    pbar = 0.02
    hmat = hc.ContactSolver(grid_size=Ns, leaf_size=64)
    h2 = hc.ContactSolver(grid_size=Ns, backend="h2", q=6)
    r_h = hmat.solve(g0, pbar, tol=1e-8)
    r_2 = h2.solve(g0, pbar, tol=1e-8)
    print(f"H-matrix: area={r_h.contact_area:.4f} iters={r_h.iterations} "
          f"mean_p={r_h.mean_pressure:.4f}")
    print(f"H2:       area={r_2.contact_area:.4f} iters={r_2.iterations} "
          f"mean_p={r_2.mean_pressure:.4f}")
    assert r_2.converged
    assert abs(r_2.mean_pressure - pbar) < 1e-6
    rel_area = abs(r_2.contact_area - r_h.contact_area) / r_h.contact_area
    assert rel_area < 0.03, rel_area


if __name__ == "__main__":
    test_matvec_accuracy()
    test_hertz_integration()
    print("OK")
