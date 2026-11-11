# Numerical checks for the YM4 note and blueprint

These programs back Remarks `rem:ol-numerical` (Section `sec:ol-chart`), `rem:ol-f-numerical` and `rem:ol-integrand-numerical` (Section `sec:ol-feynman`). They are pure Python 3 and C; no packages are needed.

## 1. Theorem ol-chart on small 2d lattices

`chart.py L NC seed amplitude nV` runs the test for d=2, SU(2) and k=1, with block side `L` on an `NC`×`NC` coarse torus. It draws `nV` random coarse fields V, each the average of a random fine configuration of the given amplitude. For each V it computes:

- a critical point U0 of S on the fibre {U^(1) = V}, by a damped Newton (SQP) method that accepts only descent steps, in the tree gauge;
- **LHS**: Laplace's method in the tree gauge, −½ log det(Hessian of the Lagrangian on ker dc) − ½ log det(dc dcᵀ);
- **RHS**: the formula of Theorem ol-chart, using the gauge-adapted averaging 𝒬 relative to U0.

The printed value `LHS - RHS` must not depend on V. In fact it equals −(d/2)·dim𝔤·|T|·log L exactly (−12 log 2, −12 log 3, −27 log 2 in the table below): the program imposes the coarse condition in unnormalized Lie coordinates, in which the block Θ₀ = Q'₁ of the Faddeev–Popov operator (Proposition ol-FP) carries a factor L^{-d/2}. The program also prints diagnostics: the Euler–Lagrange residual, q₂ D 𝒩(Q') and 𝐐 D 𝒩(Q') (both should vanish), and the RHS at two values of a. Lines starting `PARTS` give the three terms of the RHS separately, and the RHS without the multiplier term, so that wrong variants can be tested.

`output/` holds the runs reported in the note. Each was made with `python3 chart.py L NC seed amplitude nV`:

| File | L | NC | seed | amplitude | nV | LHS − RHS |
|---|---|---|---|---|---|---|
| `chart1.txt` | 2 | 2 | 1 | 0.3 | 1 | −8.3177661667 |
| `chartA.txt` | 2 | 2 | 11 | 0.3 | 2 | −8.3177661667 |
| `chartB.txt` | 2 | 2 | 12 | 0.5 | 2 | −8.3177661667 |
| `chartP.txt` | 2 | 2 | 41 | 0.3 | 2 | −8.3177661667 |
| `chartL3a.txt` | 3 | 2 | 21 | 0.25 | 1 | −13.1833474640 |
| `chartL3b.txt` | 3 | 2 | 22 | 0.25 | 1 | −13.1833474641 |
| `chartN3.txt` | 2 | 3 | 31 | 0.25 | 2 | −18.7149738747, −18.7149738751 |

`chartA.txt` and `chartB.txt` come from an earlier version of `chart.py` that did not print the `PARTS` lines. The computation is otherwise the same.

`ghost.py` checks the Jacobi identity of Corollary ol-ghost on a random 2d background.

## 2. The lattice Feynman-gauge coefficient f(m)

1. `python3 tensors.py tensors.txt` computes the Taylor coefficients of the SU(2) plaquette in the chart e^{ih}e^{iA}, up to order h²A², by Cauchy's formula. Output: `tensors.txt`.
2. `cc -O3 -o vacpol vacpol.c -lm` builds the integrator for the one-loop vacuum polarization in background Feynman gauge with mass m. `./vacpol tensors.txt test` runs its self-checks.
3. `python3 run.py m N p1 epsdir sym` integrates on N⁴ points, split over 8 processes, and prints the second difference in p₁. Use `epsdir=1` for the transverse case and `epsdir=0` for the longitudinal (Ward identity) case.
4. `scan.py` runs the mass scan into `scan.txt`, with step 10⁻³ in p₁. The same scan with step 4·10⁻³ goes into `scan4.txt`, which removes the O(step²) bias of the second difference by Richardson extrapolation. `output/ward_m1.txt` holds f(1) and the longitudinal (Ward identity) run at m = 1, p₁ = 0.5. `fit.py`, run in the folder holding `scan.txt` and `scan4.txt` (e.g. `cd output && python3 ../fit.py`), applies the Richardson correction, fits f(m) = b log(1/m) + c + O(m² log m), and compares with b = 11/(6π²) and the Dashen–Gross constant.

## 3. Bałaban's symmetric averaging (blueprint, Section Y2)

These programs back Section `subsec:Y2-numerics` of `YM4-blueprint.tex`. They transcribe Bałaban's four-dimensional scheme (CMP 109, (0.5)–(0.12)) to d=2, SU(2), L odd:
- blocks centred at the coarse points;
- families G(y,x) of coordinate-ordered contours, one per ordering of the nonzero directions;
- the group average M solving Σ_j log(U_j U⁻¹) = 0;
- averaged contour variables U(y,x) = M({U(Γ)});
- the averaging (0.12);
- the averaged axial gauge and the gauge-adapted averaging of Definition adapted, built from the averaged contour variables.

`su2.py` holds the SU(2) primitives and M. `symavg.py` and `symavg_k2d4.py` contain the checks.

- `python3 symavg.py checks L NC seed` runs the algebraic checks. Output: `output/symavg-checks.txt` (L=3, NC=2, seed 1; seconds).
  - **A1:** the axioms of M for 2–24 elements, and its derivative weights.
  - **A2:** the transformation law under Haar-random gauge transformations, with log-mean contour variables as the control.
  - **A3:** D₄ and reversal covariance of (0.12), with one contour ordering as the control; D₄ covariance of the gauge-adapted average.
  - **A4:** the averaged axial gauge, and Theorem correspondence (e), (f) nonlinearly.
  - **A5:** linearization at U=1 (tube average, γ terms, the coefficient L^{1−d} of b₀(c), compositions).
  - **A6:** Proposition linearized-averages (a), (b) (E=0), (f).
  - **A7:** enumeration for Claim Y2-tree.
  - **A8:** A3 and A6(b) repeated with the iterative solver of (0.10).
  - Lines marked `[construction]` hold algebraically once A2 holds.
- `python3 symavg.py chart L NC seed amp nV` tests Theorem ol-chart on the averaged axial slice. For each coarse field V it computes:
  - the critical point U₀ on the fibre;
  - **LHS_tree:** Laplace's method in the tree gauge;
  - **LHS_exp:** Laplace's method over all bonds with Bałaban's exponential gauge-fixing weight relative to U₀, predicted to equal LHS_tree + (n_λ/2) log 2 with n_λ = 3(#fine − #coarse sites);
  - **RHS:** the formula of Theorem ol-chart, predicted LHS_tree − RHS = −(d/2)·dim𝔤·|T|·log L;
  - the same RHS with the tree contour in the gauge (chart independence: the same value);
  - with log-mean contour variables (E ≠ 0; must miss);
  - the one-loop term at rV for three elements r of D₄ (must agree).

  The minimization aborts if it does not converge, and the Laplace step aborts if the multiplier residual exceeds 1e-10.
- `python3 symavg_k2d4.py seed` runs further checks. Output: `output/symavg-k2d4.txt` (seed 1; about a minute).
  - **Part C (d=2):**
    - a second implementation of the geometry, compared with `symavg.py`;
    - E=0, Theorem correspondence(e) and the linearization at k=2 on an 18×18 torus;
    - the smallest eigenvalue at U=1 of |dB|² + |𝔊B|² + |qB|² on tori up to 15×15, for L=3 and L=5.
  - **Part D (d=4, L=3, one coarse bond):**
    - the transformation law, covariance of (0.12) under all 384 elements of B₄ (with the one-ordering control), and reversal;
    - the coefficient 1/27 of b₀(c) and the linearization at U=1;
    - E=0 and Theorem correspondence(e) at a random U₀, with complex steps through the iterative solver for families of up to 24 contours.

Runs of `symavg.py chart 3 2 seed amp 1`. Each side varies by 0.64 between the four fields; the prediction for LHS_tree − RHS is −12 log 3 = −13.1833474640.

| File | seed | amplitude | LHS_tree − RHS − prediction | LHS_exp − LHS_tree − 48 log 2 | log-mean control | D₄ invariance |
|---|---|---|---|---|---|---|
| `symavg-chart21.txt` | 21 | 0.25 | −2.5e-11 | 6.5e-11 | 2.1e-4 | ≤ 4.4e-11 |
| `symavg-chart22.txt` | 22 | 0.25 | 1.4e-11 | 1.3e-10 | 3.2e-4 | ≤ 2.5e-11 |
| `symavg-chart23.txt` | 23 | 0.35 | −4.0e-11 | 1.6e-10 | 4.9e-4 | ≤ 2.2e-10 |
| `symavg-chart24.txt` | 24 | 0.4 | 4.5e-11 | 2.5e-10 | 6.6e-4 | ≤ 6.0e-11 |

## 4. Checks for the four-dimensional flow (blueprint, Section Y3)

The programs are in `y3/`. They import `su2.py`, `symavg.py` and `symavg_k2d4.py` from this folder. Their outputs are in `y3/output/`. The full report, with every number and its consequence for the plan, is `y3/REPORT.md`. All checks pass.

| Program | What it checks | Result |
|---|---|---|
| `y3/b4forms.py` (N0) | dimension of the Ad-invariant quadratic forms on Λ²ℝ⁴ invariant under B₄ and under B₄⁺ | 1 (the action density) and 2 (with the Pfaffian); reflections are needed |
| `y3/n1ab.py` (N1a, N1b) | d=4, L=3: the linearized averaging maps a linear field A_F to the coarse linear field; the nonlinear averaging maps a U(1)⊂SU(2) field of constant curvature to one of constant curvature L²F | exact to ≤ 7e-15; no gradient term for the symmetric averaging |
| `y3/n1c.py` (N1c) | d=2, L=3, coarse tori 2×2 to 4×4, flux 1–2: the constant-flux configuration is critical on the constrained fibre, with positive constrained Hessian; random starts return to it | yes, to 2e-15; it is a saddle of S without the constraint |
| `y3/n2.py` (N2) | 4d constant-flux tori: constant plaquettes, zero current, Pfaffian sum ∝ Pf(f), reflections act as on the flux | to ≤ 2e-14; the Pfaffian flips by det R |
| `y3/n3.py`, `y3/n3b.py` (N3) | d=4 abelian: period density versus Fourier p²-tensor differ by a Pfaffian multiple; a family with B₄-invariant sum can carry a Pfaffian period density, a B₄-covariant family cannot; normalization of T̄_S | confirmed; β relative to T̄_S equals Part XI's coefficient with factor 1 |
| `y3/nf_check.py` | the gauge construction of Lemma later-normal-form in SL(2,ℂ) | to round-off; the coefficient ½ of the curvature term |

## 5. Checks for Section `sec:ol-feynman`: the integrand and the tensor (Y5c, N-a and N-b)

The programs are in `y5c/`, their outputs in `y5c/output/`. They back Remark `rem:ol-integrand-numerical`. Both pass.

| Check | Program | Run time | Result |
|---|---|---|---|
| N-b: spherical average of the continuum tensor, exact | `y5c/nb_sphere.py` | < 1 s | **pass**: (11/3)κ_ρ(δ\|p\|² − pp) exactly, transverse part by part |
| N-a: scaling of the lattice integrand, SU(2) | `y5c/na_scaling.c` | < 1 s | **pass**: \|ρ⁴𝓘(ρω) − 𝓘₀(ω)\| ≤ 4.4ρ² for ρ = 2⁻ʲ, j ≤ 12 |

### N-b. The spherical average (Lemma `ol-classical`)

`cd y5c && python3 nb_sphere.py > output/nb-out.txt` (pure Python 3, exact arithmetic with `fractions`).

- **Input:** the continuum vertices of Lemma `ol-naive-limit`, colour-stripped, with κ_ρ = C_A/T_ρ factored out:
  - scalar: (2q+p)·c;
  - vector: ((2q+p)·c) δ_ab + 2(p_a c_b − p_b c_a).
- **Computation:**
  - the bubble numerators of −½ log det(vector) + log det(scalar), with the routing of the lattice integral (propagators at q and q+p) and the vertex at (q+p; −p) for the conjugate polarization;
  - the expansion of G(q+p) at |q| = 1, m = 0 to second order in p;
  - the average over S³ with the exact moments ⟨∏ ω_i^{2b_i}⟩ = ∏(2b_i − 1)!! / (4·6···(2n+2)), n = Σ b_i (from Gaussian integrals).
- **Checks:**
  - the moment formula, against Σ_i ⟨ω_i² ω^a⟩ = ⟨ω^a⟩ for all monomials of degree ≤ 6;
  - all 256 components of the four-index tensor have the O(4) form a δ_{μν}δ_{κλ} + b (δ_{μκ}δ_{νλ} + δ_{μλ}δ_{νκ}), part by part;
  - transversality a + 2b = 0;
  - the orbital × spin cross terms vanish identically.
- **Result** (per unit κ_ρ, before the factor (2π)⁻⁴|S³| = 1/(8π²)):

| Part | a | b | isotropic coefficient |
|---|---|---|---|
| scalar determinant (orbital) | 1/3 | −1/6 | 1/3 |
| vector determinant, orbital | −2/3 | 1/3 | −2/3 |
| vector determinant, spin | 4 | −2 | 4 |
| total | 11/3 | −11/6 | 11/3 |

So 𝖻 = 11κ_ρ/(24π²); for SU(2), κ_ρ = 4 and 𝖻 = 0.185756.

### N-a. The scaling of the lattice integrand (Lemmas `ol-naive-limit` and `ol-continuum-integrand`)

`cd y5c && cc -O2 -o na_scaling na_scaling.c -lm && ./na_scaling ../tensors.txt 200 12 7 > output/na-out.txt`

- **Reuse:** `na_scaling.c` includes `../vacpol.c` unchanged (its `main` is renamed by the preprocessor) and calls its integrand. So the lattice vertices are those of Section 2: SU(2), background along T³, initial-point convention, propagators at q and q+p, Wilson vertices from the Taylor tensors `../tensors.txt`. They do not use the proof of Lemma `ol-naive-limit`.
- **Computation:** for 200 random directions ω = (ω_q, ω_m) on the closed half-sphere |ω_q|² + ω_m² = 1, ω_m ≥ 0, a random real unit polarization c, a random unit direction u of p, and ρ = 2⁻ʲ, 0 ≤ j ≤ 12:
  - R(ρ) = ρ⁴ 𝓘[c,u](ρω), where 𝓘[c,u] = Σ c_μc_νu_κu_λ 𝓘_{μνκλ} is d²/dt² of the integrand at p = tu (in the component pairing of `vacpol.c` this equals the trace-pairing tensor of the note);
  - the second derivative is a second difference with steps 10⁻³ρ and 4·10⁻³ρ and Richardson extrapolation;
  - the comparison value is 𝓘₀[c,u](ω) = κ_ρD⁻²[4 − 3s² − 4g²/D + 16a²g²/D² − 8ags/D], with κ_ρ = 4, D = |q|² + m², a = q·u, g = q·c, s = u·c. This is ½ d²/dt² of (eq:ol-continuum-P).
  - Direction 0 has m = 0 and q = e₁; direction 1 is Π₂₂ with p ∥ e₁; direction 2 is longitudinal (c = u). The output lists the rows of the first 6 directions and the maxima over all 200.
- **Result** (maximum over the 200 directions):

| ρ | 1 | 2⁻² | 2⁻⁴ | 2⁻⁶ | 2⁻⁸ | 2⁻¹⁰ | 2⁻¹² |
|---|---|---|---|---|---|---|---|
| max \|R − 𝓘₀\| | 4.33 | 2.50e-1 | 1.55e-2 | 9.71e-4 | 6.07e-5 | 3.79e-6 | 2.44e-7 |
| max \|R − 𝓘₀\|/ρ² | 4.33 | 4.00 | 3.98 | 3.98 | 3.98 | 3.98 | 4.09 |

- **Consequence:** the difference decreases like 4ρ², faster than the bound Cρ of Lemma `ol-continuum-integrand`. The limit includes the factor 2 of the spin vertex and the weight of the ghost: a spin factor 1 instead of 2, or a different ghost weight, would change 𝓘₀ by O(1).
- **Accuracy:** below ρ = 2⁻¹² the accuracy 10⁻¹⁰ of `tensors.txt` dominates. In a run to ρ = 2⁻¹⁶, max |R − 𝓘₀|/ρ stays between 4·10⁻⁴ and 1.1·10⁻³ from ρ = 2⁻¹³ on, the round-off floor of order 10⁻¹⁰/ρ.
