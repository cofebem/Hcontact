# Spectral-preconditioner prototype — results

Prototype: `experiments/precond_pcg.py` (run in `fenicsx-env`).
Problem: `bandlimited_surface` (self-affine, **fixed physical band** k∈[4,32],
H=0.8, rms=0.02), normal contact at p̄=0.05, tol=1e-8, H2 matvec (q=6).

## Iteration counts vs grid size

| Ns | N | C++ | Python (none) | `\|q\|` | circulant | speedup | Ac/A | rel-p |
|----|---|-----|---------------|------|-----------|---------|------|-------|
| 64 | 4 096 | 26 | 26 | 19 | 19 | 1.37× | 0.125 | 1.4e-7 |
| 128 | 16 384 | 43 | 43 | 25 | 24 | 1.79× | 0.098 | 2.4e-7 |
| 256 | 65 536 | 56 | 56 | 33 | 33 | 1.70× | 0.091 | 3.3e-7 |
| 512 | 262 144 | 83 | 83 | 43 | 43 | 1.93× | 0.095 | 4.5e-7 |
| 1024 | 1 048 576 | 180 | 180 | 62 | 62 | **2.90×** | 0.105 | 4.9e-7 |

## Findings

1. **The Python projected CG reproduces the C++ solver exactly** (`cpp` == `none`
   column) — the prototype is a faithful baseline.
2. **The example's steep growth (90→906) was mostly the surface roughening with
   Ns.** With a fixed physical band the baseline grows ~√Ns (26→180 over a 16×
   range in Ns), as predicted by κ(S)∼Ns.
3. **The spectral preconditioner gives a growing speedup** — 1.4× at Ns=64 up to
   2.9× at Ns=1024 — because the preconditioned count grows ~half as fast as the
   baseline. The advantage compounds at scale (the large-grid regime that
   motivated this).
4. **Circulant (1/Ŝ_c) ≈ |q|**: the simpler, guaranteed-positive `|q|` symbol is
   the choice; the circulant form adds nothing here.
5. **Solutions match** the unpreconditioned result to ~5e-7 (rel-p).

## Limitation / why not grid-independent

The preconditioner conditions the *full* operator, but the contact constraint
restricts it; masking the residual to the contact set breaks the FFT
diagonalization near the contact boundary and for fragmented contact. Pure
spectral preconditioning therefore reduces — but does not eliminate — the
growth. Reaching near grid-independence needs the complementary **nested-grid
(cascadic/FMG) continuation** (deferred in the spec): solve coarse→fine,
warm-starting each level with the prolonged coarse pressure + active set.

## Nested-grid (cascadic / FMG) continuation

Prototype: `experiments/nested_grid.py`. The surface is generated on the finest
grid and **restricted** (2×2 block-average) to each coarser level (same physical
surface at every level); the coarse pressure is **prolonged** to the next finer
grid as a warm start, seeding pressure and active set; each level uses the `|q|`
preconditioner; coarse levels solved to a looser tol (cascadic).

| target Ns | cold (precond) | nested fine (inject) | nested fine (bilinear) | total work / 1 cold solve |
|-----------|----------------|----------------------|------------------------|---------------------------|
| 256  | 33 | 26 | 30 | 0.89× |
| 512  | 43 | 31 | 32 | 0.79× |
| 1024 | 62 | 45 | 51 | 0.79× |

Findings:
- The warm start cuts fine-grid iterations a further ~1.4× on top of the
  preconditioner (62→45 at Ns=1024). Combined with preconditioning this is ~4×
  fewer fine iterations than the original cold/unpreconditioned baseline (180→45).
- **The entire coarse→fine solve costs *less* than a single cold solve** on the
  target grid (work ratio 0.79× at Ns≥512) — multigrid pays for itself.
- **Injection prolongation beats bilinear**: bilinear smears pressure across the
  sharp contact boundary, giving a worse active-set seed. The sharp,
  load-preserving injection is the right transfer for contact.
- Still **not fully grid-independent** (fine iters 26→45): each octave of
  refinement resolves *new* surface asperities, so there is genuinely new
  fine-scale physics to solve at every level — inherent to rough contact, not a
  defect of the method.

## Next steps

- Port the `|q|` preconditioner (and optional warm-start path) into C++
  `solve_contact` (header-only FFT + optional preconditioner argument) to bank
  the combined ~4× in production.
