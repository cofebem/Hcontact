"""Prototype: Fourier-preconditioned projected CG for normal contact.

Validates a spectral preconditioner for the Polonsky-Keer projected CG. The
operator S (Boussinesq) has Fourier symbol ~1/|q|, so a preconditioner with
symbol ~|q| (or 1/Shat from the kernel's circulant eigenvalues) collapses the
condition number from ~Ns to ~O(1), giving near grid-independent iterations.

`pk_pcg(..., precond=None)` reproduces src/contact_solver.cpp exactly (baseline);
with a preconditioner only the CG direction/beta change, the exact line search
is untouched.

Run (fenicsx-env):  python experiments/precond_pcg.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import hmatrix_contact as hc


# ── kernel / surface helpers ──────────────────────────────────────────────────
def love_uz(x, y, a, b):
    """Love (1929) cell integral (same formula as src/boussinesq_kernel.cpp)."""
    xpa, xma = x + a, x - a
    ypb, ymb = y + b, y - b
    Rpp = np.hypot(xpa, ypb); Rpm = np.hypot(xpa, ymb)
    Rmp = np.hypot(xma, ypb); Rmm = np.hypot(xma, ymb)
    return (xpa * np.log((ypb + Rpp) / (ymb + Rpm))
            + ypb * np.log((xpa + Rpp) / (xma + Rmp))
            + xma * np.log((ymb + Rmm) / (ypb + Rmp))
            + ymb * np.log((xma + Rmm) / (xpa + Rpm)))


def bandlimited_surface(Ns, L=1.0, hurst=0.8, k1=4, k2=32, seed=12345, rms=0.02):
    """Self-affine surface with a FIXED physical wavenumber band [k1, k2]
    (independent of Ns), so coarse and fine grids resolve the same surface."""
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(Ns, d=1.0 / Ns)          # integer cycles over the domain
    KX, KY = np.meshgrid(kx, kx)
    k = np.hypot(KX, KY)
    amp = np.zeros_like(k)
    band = (k >= k1) & (k <= k2)
    amp[band] = k[band] ** (-(hurst + 1.0))      # self-affine roll-off in-band
    spec = (rng.standard_normal((Ns, Ns)) + 1j * rng.standard_normal((Ns, Ns)))
    h = np.fft.ifft2(spec * amp).real
    h -= h.mean()
    s = h.std()
    if s > 0:
        h *= rms / s
    return h


# ── preconditioner symbols ────────────────────────────────────────────────────
def symbol_absq(Ns, L=1.0, E_star=1.0):
    """w(q) = |q| (asymptotic inverse of S's 1/|q| symbol); DC zeroed."""
    q1d = 2.0 * np.pi * np.fft.fftfreq(Ns, d=L / Ns)
    QX, QY = np.meshgrid(q1d, q1d)
    w = np.hypot(QX, QY)
    w[0, 0] = 0.0
    return w


def symbol_circulant(Ns, L=1.0, E_star=1.0):
    """w(q) = 1/Shat_c(q): inverse of the circulant eigenvalues of the discrete
    Love operator (exact for the periodic part). DC zeroed; non-positive modes
    floored for robustness."""
    h = L / Ns
    idx = np.arange(Ns)
    off = np.where(idx < Ns // 2, idx, idx - Ns).astype(float)   # signed offset
    DX, DY = np.meshgrid(off * h, off * h)
    K = (1.0 / (np.pi * E_star)) * love_uz(DX, DY, 0.5 * h, 0.5 * h)
    Shat = np.fft.fft2(K).real
    floor = 1e-6 * Shat.flat[np.argmax(np.abs(Shat))]
    Shat = np.where(Shat > floor, Shat, np.inf)                   # 1/inf -> 0
    w = 1.0 / Shat
    w[0, 0] = 0.0
    return w


def make_precond(symbol, Ns):
    """Return apply(g, contact) -> z = M^-1 g, masked to contact, mean-zeroed."""
    w = symbol

    def apply(g, contact):
        r = np.where(contact, g, 0.0).reshape(Ns, Ns)
        z = np.fft.ifft2(w * np.fft.fft2(r)).real.ravel()
        z = np.where(contact, z, 0.0)
        nc = contact.sum()
        if nc:
            z = z - np.where(contact, z[contact].mean(), 0.0)
            z = np.where(contact, z, 0.0)
        return z

    return apply


# ── projected (optionally preconditioned) CG, mirroring contact_solver.cpp ─────
def pk_pcg(matvec, g0, p_bar, precond=None, tol=1e-8, max_iter=20000, use_pr=True):
    N = g0.size
    P_total = p_bar * N
    g_scale = g0.max() - g0.min()
    if g_scale <= 0:
        g_scale = 1.0

    p = np.full(N, p_bar)
    t = np.zeros(N)
    g_prev = np.zeros(N)
    G_old = 1.0
    delta = 0.0
    converged = False
    hist = []

    it = 0
    for it in range(max_iter):
        u = matvec(p)
        g = u + g0
        contact = p > 0.0
        nc = int(contact.sum())
        alpha = g[contact].sum() / nc if nc else 0.0
        g = g - alpha

        e = float(np.sum(p * np.abs(g)) / (P_total * g_scale))
        hist.append(e)
        if e < tol:
            converged = True
            break

        z = np.where(contact, g, 0.0) if precond is None else precond(g, contact)

        G = float(np.sum((z * g)[contact]))
        G_pr = float(np.sum((z * (g - g_prev))[contact]))
        beta_val = max(0.0, G_pr / G_old) if use_pr else G / G_old
        beta = delta * beta_val
        t = np.where(contact, z + beta * t, 0.0)
        g_prev = g.copy()
        G_old = G

        r = matvec(t)
        rmean = r[contact].sum() / nc if nc else 0.0
        num = float(np.sum((g * t)[contact]))
        den = float(np.sum(((r - rmean) * t)[contact]))
        if den <= 0.0:
            delta = 0.0
            continue
        tau = num / den

        p = np.where(contact, p - tau * t, p)
        p = np.maximum(p, 0.0)
        overlap = (p == 0.0) & (g < 0.0)
        if overlap.any():
            p = p - np.where(overlap, tau * g, 0.0)
        delta = 0.0 if overlap.any() else 1.0

        total = p.sum()
        if total > 0.0:
            p *= P_total / total
        else:
            p[:] = p_bar

    frac = float((p > 0).mean())
    return dict(pressure=p, iterations=it, converged=converged,
                contact_fraction=frac, error=hist[-1] if hist else 0.0,
                history=hist)


# ── benchmark ─────────────────────────────────────────────────────────────────
def main():
    p_bar = 0.05
    print(f"{'Ns':>5} {'N':>9} {'cpp':>5} {'none':>5} {'|q|':>5} {'circ':>5} "
          f"{'speedup':>8} {'Ac/A':>6} {'relp':>8}")
    for Ns in [64, 128, 256, 512, 1024]:
        surf = bandlimited_surface(Ns)
        g0 = (-surf).ravel()
        solver = hc.ContactSolver(grid_size=Ns, backend="h2", q=6)
        mv = lambda v: np.asarray(solver.matvec(v)).ravel()

        # C++ baseline (faithfulness check) on the same surface
        cpp = solver.solve(g0, p_bar, tol=1e-8, max_iter=20000)

        base = pk_pcg(mv, g0, p_bar, precond=None)
        pq = pk_pcg(mv, g0, p_bar, precond=make_precond(symbol_absq(Ns), Ns))
        pc = pk_pcg(mv, g0, p_bar, precond=make_precond(symbol_circulant(Ns), Ns))

        d_absq = np.linalg.norm(pq["pressure"] - base["pressure"]) / np.linalg.norm(base["pressure"])
        speedup = base["iterations"] / max(1, pc["iterations"])
        print(f"{Ns:>5} {Ns*Ns:>9} {cpp.iterations:>5d} {base['iterations']:>5d} "
              f"{pq['iterations']:>5d} {pc['iterations']:>5d} {speedup:>7.2f}x "
              f"{pc['contact_fraction']:>6.3f} {d_absq:>8.1e}")


if __name__ == "__main__":
    main()
