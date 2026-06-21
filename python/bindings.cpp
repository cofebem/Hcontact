// pybind11 (and Python.h) must precede any standard header: Python.h sets
// feature-test macros that break <ctime> with the conda gcc toolchain otherwise.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "boussinesq_kernel.hpp"
#include "cluster_tree.hpp"
#include "contact_solver.hpp"
#include "hmatrix.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace {

Eigen::VectorXd to_vector(const py::array_t<double, py::array::c_style |
                                                        py::array::forcecast>& a,
                          int expected) {
    if (a.size() != expected)
        throw std::invalid_argument("array has " + std::to_string(a.size()) +
                                    " entries, expected " +
                                    std::to_string(expected));
    Eigen::VectorXd v(expected);
    std::memcpy(v.data(), a.data(), sizeof(double) * expected);
    return v;
}

py::array_t<double> as_grid(const Eigen::VectorXd& v, int Ns) {
    py::array_t<double> a({Ns, Ns});
    std::memcpy(a.mutable_data(), v.data(), sizeof(double) * v.size());
    return a;
}

struct PyResult {
    hmc::ContactResult r;
    int Ns = 0;
};

class PyContactSolver {
public:
    PyContactSolver(int grid_size, double domain_size, double E_star, double eta,
                    double aca_tol, int leaf_size, bool use_hmatrix,
                    bool use_acagp, double central_fraction, double inline_svd_tol)
        : kernel_(grid_size, domain_size, E_star), use_h_(use_hmatrix) {
        if (use_h_) {
            tree_ = std::make_unique<hmc::ClusterTree>(grid_size, leaf_size);
            hmat_ = std::make_unique<hmc::HMatrix>(
                kernel_, *tree_, eta, aca_tol, use_acagp, central_fraction,
                inline_svd_tol);
        } else {
            dense_ = kernel_.assemble_dense();
        }
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& p) const {
        return use_h_ ? hmat_->matvec(p) : dense_ * p;
    }

    py::array_t<double>
    matvec(const py::array_t<double, py::array::c_style | py::array::forcecast>& p)
        const {
        Eigen::VectorXd v = to_vector(p, kernel_.size());
        Eigen::VectorXd u;
        {
            py::gil_scoped_release release;
            u = apply(v);
        }
        py::array_t<double> out(kernel_.size());
        std::memcpy(out.mutable_data(), u.data(), sizeof(double) * u.size());
        return out;
    }

    PyResult solve(const py::array_t<double, py::array::c_style |
                                                 py::array::forcecast>& gap,
                   double p_nominal, double tol, int max_iter,
                   bool use_pr) const {
        Eigen::VectorXd g0 = to_vector(gap, kernel_.size());
        PyResult out;
        out.Ns = kernel_.grid_size();
        {
            py::gil_scoped_release release;
            auto op = [this](const Eigen::VectorXd& v) { return apply(v); };
            out.r = hmc::solve_contact(op, g0, p_nominal, tol, max_iter, use_pr);
        }
        return out;
    }

    // Returns (N_blocks, 6) array:
    //   [row_begin, row_size, col_begin, col_size, is_dense, rank]
    // rank = U.cols() for low-rank blocks, 0 for dense blocks.
    // All indices are in permuted (cluster) index space.
    py::array_t<double> block_layout() const {
        if (!use_h_) return py::array_t<double>(std::vector<py::ssize_t>{0, 6});
        const auto& blks = hmat_->blocks();
        const int nb = static_cast<int>(blks.size());
        py::array_t<double> out({nb, 6});
        auto r = out.mutable_unchecked<2>();
        for (int i = 0; i < nb; ++i) {
            r(i, 0) = blks[i].row_begin;
            r(i, 1) = blks[i].row_size;
            r(i, 2) = blks[i].col_begin;
            r(i, 3) = blks[i].col_size;
            r(i, 4) = blks[i].dense ? 1.0 : 0.0;
            r(i, 5) = blks[i].dense ? 0.0 : static_cast<double>(blks[i].U.cols());
        }
        return out;
    }

    void recompress(double svd_tol) {
        if (use_h_) hmat_->recompress(svd_tol);
    }

    py::dict hmatrix_info() const {
        py::dict d;
        if (!use_h_) {
            d["dense"] = true;
            d["bytes"] = 8LL * kernel_.size() * kernel_.size();
            py::print("dense influence matrix,", kernel_.size(), "x",
                      kernel_.size());
            return d;
        }
        const auto s = hmat_->info();
        d["n"] = s.n;
        d["n_dense_blocks"] = s.n_dense;
        d["n_lowrank_blocks"] = s.n_lowrank;
        d["max_rank"] = s.max_rank;
        d["avg_rank"] = s.avg_rank;
        d["bytes"] = s.bytes;
        d["compression"] = s.compression;
        py::print(s.to_string());
        return d;
    }

private:
    hmc::BoussinesqKernel kernel_;
    bool use_h_;
    std::unique_ptr<hmc::ClusterTree> tree_;
    std::unique_ptr<hmc::HMatrix> hmat_;
    Eigen::MatrixXd dense_;
};

} // namespace

PYBIND11_MODULE(hmatrix_contact, m) {
    m.doc() = "H-matrix BEM normal-contact solver for an elastic half-space "
              "(Boussinesq kernel, Love element integration, Polonsky-Keer CG)";

    py::class_<PyResult>(m, "ContactResult")
        .def_property_readonly(
            "pressure", [](const PyResult& s) { return as_grid(s.r.pressure, s.Ns); })
        .def_property_readonly(
            "displacement",
            [](const PyResult& s) { return as_grid(s.r.displacement, s.Ns); })
        .def_property_readonly(
            "gap", [](const PyResult& s) { return as_grid(s.r.gap, s.Ns); })
        .def_property_readonly("objective",
                               [](const PyResult& s) { return s.r.objective; })
        .def_property_readonly(
            "contact_area", [](const PyResult& s) { return s.r.contact_fraction; })
        .def_property_readonly(
            "mean_pressure", [](const PyResult& s) { return s.r.mean_pressure; })
        .def_property_readonly("approach",
                               [](const PyResult& s) { return s.r.approach; })
        .def_property_readonly("iterations",
                               [](const PyResult& s) { return s.r.iterations; })
        .def_property_readonly("error", [](const PyResult& s) { return s.r.error; })
        .def_property_readonly("converged",
                               [](const PyResult& s) { return s.r.converged; })
        .def("__repr__", [](const PyResult& s) {
            std::ostringstream os;
            os << "<ContactResult: " << (s.r.converged ? "converged" : "NOT converged")
               << " in " << s.r.iterations << " iters, error " << s.r.error
               << ", contact area " << s.r.contact_fraction << ", mean p "
               << s.r.mean_pressure << ">";
            return os.str();
        });

    py::class_<PyContactSolver>(m, "ContactSolver")
        .def(py::init<int, double, double, double, double, int, bool, bool, double, double>(),
             py::arg("grid_size"), py::arg("domain_size") = 1.0,
             py::arg("E_star") = 1.0, py::arg("eta") = 2.0,
             py::arg("aca_tol") = 1e-6, py::arg("leaf_size") = 64,
             py::arg("use_hmatrix") = true,
             py::arg("use_acagp") = false,
             py::arg("central_fraction") = 0.3,
             py::arg("inline_svd_tol") = 0.0)
        .def("matvec", &PyContactSolver::matvec, py::arg("p"),
             "Influence-matrix product u = S p; accepts shape (N,) or (Ns, Ns)")
        .def("solve", &PyContactSolver::solve, py::arg("gap"),
             py::arg("p_nominal"), py::arg("tol") = 1e-8,
             py::arg("max_iter") = 5000, py::arg("use_pr") = true,
             "Solve the normal contact problem; use_pr=True (default) uses "
             "Polak-Ribiere+ beta, use_pr=False uses Fletcher-Reeves")
        .def("block_layout", &PyContactSolver::block_layout,
             "Return (N_blocks, 5) array [row_begin, row_size, col_begin, col_size, is_dense]")
        .def("recompress", &PyContactSolver::recompress, py::arg("svd_tol"),
             "Truncated SVD recompression: drop singular values below svd_tol * sigma_max")
        .def("hmatrix_info", &PyContactSolver::hmatrix_info,
             "Print and return H-matrix block/compression statistics");
}
