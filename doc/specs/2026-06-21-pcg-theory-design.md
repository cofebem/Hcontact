---
name: pcg-theory-doc
description: Design spec for the Hcontact theoretical document on projected CG with line search
metadata:
  type: project
---

# Design: Projected CG Theory Document

**Output file:** `Hcontact/doc/theory/pcg.tex`  
**Format:** LaTeX article, standalone, compiles with `pdflatex`  
**Scope:** General reference on CG, β formulas, line search, and bound-constrained extensions;
anchored in the Boussinesq/Polonsky-Keer contact problem.

## Section Structure

| Layer | Sections | Content |
|---|---|---|
| I — Unconstrained CG | §1 | CG derivation, step size, A-orthogonality, finite termination |
| I — β formulas | §2 | FR, PR/PR+, HS, DY; comparison table; convergence remarks |
| II — Line search | §3 | Exact (quadratic), Armijo, Wolfe; polynomial interpolation; cost table |
| III — Constrained CG | §4 | Projected gradient, active/free sets, KKT, optimality measure |
| III — P-K algorithm | §5 | Contact QP, rigid-body shift, mean-corrected line search, overlap correction, full algorithm |
| III — GPCG | §6 | Moré-Toraldo 1991 two-phase algorithm; comparison table with P-K |
| IV — Numerics | §7 | FR vs PR+ benchmark table, H-matrix scaling table |

## LaTeX Setup

- `article`, 11pt, A4, `geometry` margins 2.5cm
- `amsmath`, `amsthm`, `amssymb`, `booktabs`, `algorithm2e`, `hyperref`
- Theorem environments: `theorem`, `proposition`, `remark`, `definition`
- Notation macros: `\bA`, `\bS`, `\p`, `\bg`, `\br`, `\bd`, `\bt`, `\Proj`, `\calC`, `\calF`

## Code Changes

- `include/contact_solver.hpp`: flip `use_pr` default to `true`
- `python/bindings.cpp`: flip `use_pr` default to `true`
- `CLAUDE.md`: update Hertz iteration counts (PR+: 24, FR: 28), add Ns=512 numbers,
  add reference to `doc/theory/pcg.tex`

## References

Hestenes & Stiefel (1952); Fletcher & Reeves (1964); Polak & Ribière (1969);
Moré & Toraldo (1991); Polonsky & Keer (1999); Dai & Yuan (1999);
Gilbert & Nocedal (1992); Al-Baali (1985).
