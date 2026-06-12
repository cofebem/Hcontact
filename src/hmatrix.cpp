#include "hmatrix.hpp"

#include <cmath>
#include <sstream>

namespace hmc {

HMatrix::HMatrix(const BoussinesqKernel& kernel, const ClusterTree& tree,
                 double eta, double aca_tol)
    : kernel_(&kernel), tree_(&tree), eta_(eta), tol_(aca_tol) {
    build(tree.root(), tree.root());
}

void HMatrix::build(int t, int s) {
    const ClusterNode& nt = tree_->nodes()[t];
    const ClusterNode& ns = tree_->nodes()[s];
    const int d = ClusterTree::dist(nt.box, ns.box);
    const int dmin = std::min(ClusterTree::diam(nt.box), ClusterTree::diam(ns.box));

    if (d > 0 && dmin <= eta_ * d) {
        HBlock blk{nt.begin, nt.end - nt.begin, ns.begin, ns.end - ns.begin,
                   false, {}, {}, {}};
        fill_aca(blk);
        blocks_.push_back(std::move(blk));
        return;
    }
    if (nt.leaf && ns.leaf) {
        HBlock blk{nt.begin, nt.end - nt.begin, ns.begin, ns.end - ns.begin,
                   true, {}, {}, {}};
        fill_dense(blk);
        blocks_.push_back(std::move(blk));
        return;
    }
    if (nt.leaf) {
        for (int q = 0; q < 4; ++q)
            if (ns.child[q] >= 0) build(t, ns.child[q]);
    } else if (ns.leaf) {
        for (int q = 0; q < 4; ++q)
            if (nt.child[q] >= 0) build(nt.child[q], s);
    } else {
        for (int qt = 0; qt < 4; ++qt)
            for (int qs = 0; qs < 4; ++qs)
                if (nt.child[qt] >= 0 && ns.child[qs] >= 0)
                    build(nt.child[qt], ns.child[qs]);
    }
}

void HMatrix::fill_dense(HBlock& blk) const {
    blk.D.resize(blk.row_size, blk.col_size);
    for (int i = 0; i < blk.row_size; ++i)
        for (int j = 0; j < blk.col_size; ++j)
            blk.D(i, j) = entry_perm(blk.row_begin + i, blk.col_begin + j);
}

// ACA with partial pivoting; never materialises the full block. Stops when
// ||u_k|| ||v_k|| <= tol * ||A_k||_F with the Frobenius norm of the
// accumulated approximant updated incrementally.
void HMatrix::fill_aca(HBlock& blk) const {
    const int m = blk.row_size, n = blk.col_size;
    const int kmax = std::min(m, n);
    Eigen::MatrixXd U(m, kmax), V(kmax, n);
    std::vector<char> row_used(m, 0);
    double frob2 = 0.0;
    int k = 0, i_star = 0;

    while (k < kmax) {
        Eigen::RowVectorXd r(n);
        for (int j = 0; j < n; ++j)
            r(j) = entry_perm(blk.row_begin + i_star, blk.col_begin + j);
        if (k > 0) r.noalias() -= U.row(i_star).head(k) * V.topRows(k);
        row_used[i_star] = 1;

        int j_star = 0;
        const double rmax = r.cwiseAbs().maxCoeff(&j_star);
        const bool negligible =
            k > 0 && rmax * rmax * m * n <= tol_ * tol_ * frob2;
        if (rmax == 0.0 || negligible) {
            i_star = -1;
            for (int i = 0; i < m; ++i)
                if (!row_used[i]) { i_star = i; break; }
            if (i_star < 0) break; // all rows visited: block resolved
            continue;
        }

        Eigen::VectorXd c(m);
        for (int i = 0; i < m; ++i)
            c(i) = entry_perm(blk.row_begin + i, blk.col_begin + j_star);
        if (k > 0) c.noalias() -= U.leftCols(k) * V.col(j_star).head(k);

        const Eigen::VectorXd u = c / r(j_star);
        const double u2 = u.squaredNorm(), v2 = r.squaredNorm();
        double cross = 0.0;
        for (int l = 0; l < k; ++l)
            cross += U.col(l).dot(u) * V.row(l).dot(r);
        frob2 += 2.0 * cross + u2 * v2;
        U.col(k) = u;
        V.row(k) = r;
        ++k;

        if (u2 * v2 <= tol_ * tol_ * frob2) break;

        double best = -1.0;
        i_star = -1;
        for (int i = 0; i < m; ++i)
            if (!row_used[i] && std::abs(u(i)) > best) {
                best = std::abs(u(i));
                i_star = i;
            }
        if (i_star < 0) break;
    }
    blk.U = U.leftCols(k);
    blk.V = V.topRows(k);
}

Eigen::VectorXd HMatrix::matvec(const Eigen::VectorXd& p) const {
    const int N = static_cast<int>(tree_->perm().size());
    Eigen::VectorXd pt(N), ut = Eigen::VectorXd::Zero(N);
    for (int pos = 0; pos < N; ++pos) pt(pos) = p(tree_->perm()[pos]);

    for (const auto& b : blocks_) {
        const auto x = pt.segment(b.col_begin, b.col_size);
        auto y = ut.segment(b.row_begin, b.row_size);
        if (b.dense)
            y.noalias() += b.D * x;
        else
            y.noalias() += b.U * (b.V * x);
    }

    Eigen::VectorXd u(N);
    for (int pos = 0; pos < N; ++pos) u(tree_->perm()[pos]) = ut(pos);
    return u;
}

HMatrixInfo HMatrix::info() const {
    HMatrixInfo s;
    s.n = static_cast<int>(tree_->perm().size());
    std::int64_t rank_sum = 0;
    for (const auto& b : blocks_) {
        if (b.dense) {
            ++s.n_dense;
            s.bytes += 8LL * b.row_size * b.col_size;
        } else {
            ++s.n_lowrank;
            const int k = static_cast<int>(b.U.cols());
            rank_sum += k;
            s.max_rank = std::max(s.max_rank, k);
            s.bytes += 8LL * k * (b.row_size + b.col_size);
        }
    }
    s.avg_rank = s.n_lowrank ? double(rank_sum) / s.n_lowrank : 0.0;
    s.compression = double(s.bytes) / (8.0 * double(s.n) * double(s.n));
    return s;
}

std::string HMatrixInfo::to_string() const {
    std::ostringstream os;
    os << "H-matrix " << n << " x " << n << ": " << n_dense << " dense + "
       << n_lowrank << " low-rank blocks, max rank " << max_rank
       << ", avg rank " << avg_rank << ", storage "
       << double(bytes) / (1 << 20) << " MiB, compression "
       << compression << " of dense";
    return os.str();
}

} // namespace hmc
