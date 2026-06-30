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
import rfgen as rf

# ── 2. size ───────────────────────────────────────────────────────────────────
Ns = 1024          # grid is Ns x Ns; must be a power of two for backend="h2"
L = 1.0           # domain side length
p_bar = 0.05      # applied (nominal) mean pressure, in units of E*



x = np.linspace(0,1,Ns)
y = np.linspace(0,1,Ns)
X,Y = np.meshgrid(x,y)

surface = -((X-0.5)**2 + (Y-0.5)**2)
roughness = rf.selfaffine_field(
    dim=2,           # Dimension (1, 2, or 3)
    N=Ns,           # Grid size per dimension
    Hurst=0.8,       # Hurst exponent ∈ [0, 1]
    k_low=12/Ns,      # Lower wavenumber cutoff
    k_high=128/Ns,      # Upper wavenumber cutoff (≤ 0.5 Nyquist)
    plateau=False,   # Flat spectrum for k < k_low
    noise=True,      # True: filtered noise, False: ideal spectrum
    rng=None,        # numpy.random.Generator for reproducibility
    verbose=False    # Print parameters
)
rms = 0.01
roughness *= rms / np.std(roughness)

surface += roughness
surface -= np.max(surface)
p_bar = 0.02
N_load_steps = 3
plot_every = 1

pressures = np.linspace(0,p_bar,N_load_steps,endpoint=True)
# ── 4. apply pressure: solve the contact problem ──────────────────────────────
# A rigid flat is pressed onto the rough surface, so the initial gap is -height.
solver = hc.ContactSolver(grid_size=Ns, domain_size=L, E_star=1.0,
                          backend="h2", q=6)
for inc,p in enumerate(pressures[1:]):
    res = solver.solve(gap=-surface, p_nominal=p, tol=1e-8, max_iter=5000)

    # ── 5. contact area + plot ────────────────────────────────────────────────────
    if inc % plot_every == 0 or inc == pressures.shape[0]-2:
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
        out = os.path.join(os.path.dirname(__file__), f"Rough_contact_{inc}.png")
        fig.savefig(out, dpi=600)
        print(f"saved figure -> {out}")
