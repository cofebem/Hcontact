"""Minimal rough-surface contact example (matrix-free H2 backend).

Flow:
  1. import what's needed
  2. define the size Ns
  3. define the rough surface
  4. apply pressure (solve the normal contact problem)
  5. get the contact area and plot it

Run (fenicsx-env):  python example_rough_contact.py
"""
# ── 1. imports ────────────────────────────────────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
import numpy as np
import matplotlib.pyplot as plt
import hmatrix_contact as hc

# ── 2. size ───────────────────────────────────────────────────────────────────
Ns = 512          # grid is Ns x Ns; must be a power of two for backend="h2"
L = 1.0           # domain side length
p_bar = 0.05      # applied (nominal) mean pressure, in units of E*


# ── 3. rough surface (self-affine, Hurst exponent H) ──────────────────────────
def self_affine_surface(Ns, H=0.8, rms=0.02, seed=12345, L=1.0):
    """Isotropic self-affine height field via spectral filtering."""
    rng = np.random.default_rng(seed)
    q1d = 2.0 * np.pi * np.fft.fftfreq(Ns, d=L / Ns)
    QX, QY = np.meshgrid(q1d, q1d)
    q = np.hypot(QX, QY)
    q[0, 0] = 1.0                      # avoid divide-by-zero at the mean
    amp = q ** (-(H + 1.0))            # 2D self-affine amplitude ~ q^-(H+1)
    amp[0, 0] = 0.0                    # zero-mean surface
    spec = (rng.standard_normal((Ns, Ns)) + 1j * rng.standard_normal((Ns, Ns)))
    h = np.fft.ifft2(spec * amp).real
    h -= h.mean()
    h *= rms / h.std()                 # normalize to target RMS roughness
    return h


surface = self_affine_surface(Ns, H=0.8, rms=0.02, L=L)
x = np.linspace(0,1,Ns)
y = np.linspace(0,1,Ns)
X,Y = np.meshgrid(x,y)
surface = -((X-0.5)**2 + (Y-0.5)**2)
p_bar = 0.01

# ── 4. apply pressure: solve the contact problem ──────────────────────────────
# A rigid flat is pressed onto the rough surface, so the initial gap is -height.
solver = hc.ContactSolver(grid_size=Ns, domain_size=L, E_star=1.0,
                          backend="h2", q=6)
res = solver.solve(gap=-surface, p_nominal=p_bar, tol=1e-8, max_iter=5000)

# ── 5. contact area + plot ────────────────────────────────────────────────────
pressure = np.asarray(res.pressure)          # (Ns, Ns)
contact = pressure > 0.0                      # in-contact mask
area_fraction = res.contact_area              # = Ac / A (also contact.mean())

print(f"Ns={Ns}  applied mean pressure={p_bar}")
print(f"converged={res.converged} in {res.iterations} iters, "
      f"mean_p={res.mean_pressure:.4f}")
print(f"contact area fraction Ac/A = {area_fraction:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
im0 = ax[0].imshow(surface, origin="lower", cmap="terrain", extent=[0, L, 0, L])
ax[0].set_title("rough surface (height)")
fig.colorbar(im0, ax=ax[0], fraction=0.046)

im1 = ax[1].imshow(pressure, origin="lower", cmap="inferno", extent=[0, L, 0, L])
ax[1].set_title("contact pressure")
fig.colorbar(im1, ax=ax[1], fraction=0.046)

ax[2].imshow(contact, origin="lower", cmap="Greys", extent=[0, L, 0, L])
ax[2].set_title(f"contact area  (Ac/A = {area_fraction:.3f})")

for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "example_rough_contact.png")
fig.savefig(out, dpi=130)
print(f"saved figure -> {out}")
