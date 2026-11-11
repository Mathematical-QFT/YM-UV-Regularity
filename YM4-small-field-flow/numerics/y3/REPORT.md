# Y3 numerical checks N1–N3 (2 November 2026)

These checks were run for the Y3 plan, `scratchpad/Y3-plan.md`, Section 5. N4 was not run. No file of the note, of `numerics/` or of the blueprint was changed.

- **Scripts and outputs:** `scratchpad/y3num/`. Everything is pure Python 3.9, with no numpy.
- **Code reused by import:** `numerics/su2.py`, `numerics/symavg.py` (d=2) and `numerics/symavg_k2d4.py` (Part D, d=4).
- **Conventions:**
  - SU(2), U = exp(i x·σ/2), so x are the components in the basis T^a = σ^a/2, with tr(T^aT^b) = ½δ^{ab} = T_ρδ^{ab}.
  - S = Σ_p(2 − Re tr U(∂p)), with the matrix trace.
  - The linear field is A_F(x)_μ = ½Σ_ν x_ν F_{νμ}. Its plaquette angle is F_{μν}.

| Check | Script | Run time | Result |
|---|---|---|---|
| N1(a) linearized averaging of A_F, d=4 | `n1ab.py` | 3 s | **pass** (≤ 7e-15) |
| N1(b) nonlinear averaging of constant curvature, d=4 | `n1ab.py` | 20 s | **pass** (≤ 3e-15), F → L²F per step in lattice units |
| N1(c) constrained minimizer = U_F, d=2 | `n1c.py` | 1 s – 4 min per case | **pass**: the second-order test passes in 5 of 6 tori; in the sixth, which is far outside small field, the Hessian is positive *semi*definite with 2 zero modes. All 4 SQP runs return to U_F |
| N2 constant-flux 4d tori | `n2.py` | 18 s | **pass** (≤ 2e-14) |
| N3 period density vs p²-tensor; null family; T̄_S | `n3.py`, `n3b.py` | 17 s | **pass**: conversion factor 1 (in Part XI's trace pairing) |

---

## N1(a). The linearized averaging of a linear field (d=4, L=3)

**What was computed.**
- The linearization at U=1 of Bałaban's averaging (0.12), by complex steps through the iterative solver of (0.10), with the 24-contour families of `symavg_k2d4.py`.
- The input was A_F with an independent random antisymmetric F (N(0,1) entries) in each of the three colours.
- 13 coarse bonds of both orientations and all directions, at three coarse base points.
- The result was compared with three things:
  - the coarse linear field B_F(c) = L·A_F(c₋)_μ, which in coarse units is the linear field of curvature L²F; for A_F, the value at c₋ equals the value at the coarse midpoint, because A_μ does not depend on x_μ;
  - the centred tube average;
  - γ_sym(y) = L^{-d}Σ_x (mean over G(y,x) of A(Γ)).

**Numbers** (output `n1ab-out.txt`):

| Quantity | Value |
|---|---|
| plain (0.12)(A_F) − B_F | 1.8e-15 (with \|B_F\| up to 16) |
| tube average(A_F) − B_F | 7.1e-15 |
| γ_sym(A_F) | 8.1e-16, i.e. **zero** |
| A_F + Dφ, φ random: plain − (B_F + φ(c₊) − φ(c₋)) | 2.7e-15 |
| A_F + Dφ: plain − (tube − D′γ) | 5.3e-15 |
| A_F + Dφ: tube − (B_F + D′φ) | 3.70 (control: the tube average alone is not covariant) |

**Pass.**

**Meaning.**
- The centred tube average maps A_F *exactly* to B_F at the coarse bond midpoints, and γ_sym(A_F) = 0 identically. The reason is one line: for an ordered contour from y to x, A_F(Γ) = ½yᵀF(x−y) + ½Σ_{i<k}v_iᵀFv_k. The second term is antisymmetric in the order, so it averages to zero over the family, and Σ_{x∈B(y)}(x−y) = 0 for centred blocks.
- So the "γ-gradient" in plan 2.6, step 1, vanishes for the symmetric-gauge A_F. For A_F + Dφ, the plain averaging gives B_F + D′(φ|_coarse), a pure coarse gauge term.
- **For Lemma ol-pullback (s51):**
  - The sentence "shifted by a constant that depends on the position of the coarse point in its block" describes corner blocks. With the centred blocks of Part IV there is no shift: 𝐐A_F = B_F exactly.
  - The second "to be verified" item of s51 (tube averages of linear fields for the blocks and contours of Part IV) is settled by this two-line argument, now confirmed numerically through Bałaban's nonlinear code.

## N1(b). Nonlinear averaging of a constant-curvature U(1) ⊂ SU(2) configuration (d=4, L=3)

**What was computed.**
- The configuration: U(b) = exp(i θ(b) n·σ/2), with a random unit vector n and θ the curvature-F field.
- The flux matrices F had random entries of size 0.03, 0.08 and 0.15, with F₁₂ and F₃₄ emphasized. The largest coarse plaquette angle was 1.47.
- Three gauges:
  - symmetric, θ = A_F;
  - axial, θ_μ = Σ_{ν<μ}x_νF_{νμ};
  - symmetric followed by a Haar-random SU(2) gauge transformation.
- Bałaban's averaging was evaluated on all bonds of 12 coarse plaquettes (6 planes at 2 base points) and of coarse 2×1 rectangles.

**Numbers** (output `n1ab-out.txt`):

| Comparison | Error |
|---|---|
| coarse plaquettes vs exp(i L²F_{μν} n·σ/2) (conjugated by g(y) in the Haar case) | ≤ 5.4e-16 |
| traces | ≤ 2.7e-15 |
| 2×1 rectangles vs exp(2i L²F) | ≤ 2.4e-15 |
| symmetric gauge: the coarse bond variables themselves vs exp(i L A_F(c₋) n·σ/2) | ≤ 2.2e-16 |

In the symmetric gauge, therefore, a(U_F) is literally U_{L²F} in coarse units.

**Pass.**

**Units.**
- In each lattice's own units, the curvature is multiplied by **L² per step** (by L^{2(k−j)} from scale j to scale k).
- In the note's units (Lie fields measured on the unit lattice, X = e^{iδB}), F is unchanged: the fine plaquette angle is δ²F and the coarse one is (Lδ)²F.
- Period densities per site scale by L⁴ and F² by L⁻⁴. So β is scale free in d=4, consistent with the factor N′^{d−4} = 1 in Lemma ol-pullback.

**Meaning.**
- In Lemma constant-flux, the averages U_F^{(j)} of a constant-flux configuration are constant-flux configurations of the same physical flux, *exactly* and nonlinearly, not only to first order.
- For the coarse-to-fine step of the β agreement, the averaging half (𝐐A_F = B_F) is exact.

## N1(c). The constrained minimizer is the constant-flux configuration (d=2, L=3)

**What was done.**
- Setup, with `symavg.py`'s geometry and averaging imported unchanged:
  - coarse NC×NC torus, fine (3NC)² torus;
  - U_F = U(1) ⊂ SU(2) configuration with n flux quanta, fine plaquette angle θ = 4πn/(3NC)²;
  - V = a(U_F).
- The minimizer machinery (dense complex-step Hessians of the full Lagrangian) is too slow beyond NC=2. I therefore did the following, which together give a faithful version of the claim:
  1. Checked that V has constant curvature L²θ.
  2. Checked that U_F is a critical point of S on the fibre {a(U)=V}: grad S(U_F) = 0 on all bonds, so the multiplier is 0.
  3. Second-order test. I took the Hessian of S on the tree-gauge slice through U_F (free links only; the slice fixes 𝔊₀), restricted it to ker(d cons), and computed:
     - its inertia, by symmetric elimination (Sylvester's law);
     - its smallest eigenvalue, by inverse iteration.

     For comparison, I computed the same at U=1 and without the constraint.
  4. Global test. I ran SQP minimizations on the fibre from random starts: U_F times SU(2) noise, then projected onto the fibre.
     - The algorithm is `symavg.minimize_on_fibre`, except that the Hessian of the Lagrangian is assembled from plaquette-local and coarse-bond-local pieces.
     - These pieces agree with `symavg.hess_cs` to 1.7e-11 at a random point of the NC=2 fibre, and are 14 times faster.
     - The test passes if the minimizer has S(U_min) = S(U_F) and the same tree-gauge representative, i.e. U_min = U_F up to 𝔊₀.

**Numbers** (outputs `n1c-NC*-n*.txt`):

| NC | n | fine / coarse plaquette angle | \|grad S(U_F)\| | coarse plaquettes vs exp(iL²θ) | inertia without constraint (neg, 0, pos) | on ker d cons: inertia | λ_min (U_F) | λ_min (U=1) |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 0.349 / π | 7.7e-16 | 3.7e-16 | (4, 13, 103) | (0, 0, 96) | 0.13194 | 0.13397 |
| 2 | 2 | 0.698 / 2π | 1.6e-15 | 1.1e-15 | (8, 13, 99) | **(0, 2, 94)** | — | 0.13397 |
| 3 | 1 | 0.155 / 1.396 | 5.0e-16 | 5.5e-16 | (4, 28, 238) | (0, 0, 216) | 0.13264 | 0.15301 |
| 3 | 2 | 0.310 / 2.793 | 7.9e-16 | 1.0e-15 | (9, 27, 234) | (0, 0, 216) | 0.12216 | 0.15301 |
| 4 | 1 | 0.087 / 0.785 | 8.2e-16 | 3.7e-16 | (4, 49, 427) | (0, 0, 384) | 0.13385 | 0.13397 |
| 4 | 2 | 0.175 / 1.571 | 1.4e-15 | 1.3e-15 | (8, 49, 423) | (0, 0, 384) | 0.13347 | 0.13397 |

In every case S(U_F) = (3NC)²(2 − 2cos(θ/2)) to 1e-12.

Global SQP runs (`n1c-NC2-n1-global*.txt`, `n1c-NC3-n1-global.txt`):

| NC, n | seed, noise amplitude | S after the 1st SQP step | iterations | S(U_min) − S(U_F) | tree-gauge \|U_min − U_F\| | constraint residual |
|---|---|---|---|---|---|---|
| 2, 1 | 2, 0.3 (start 0) | 8.61 | 19 | −2.3e-14 | 1.7e-15 | 1.8e-16 |
| 2, 1 | 2, 0.3 (start 1) | 10.45 | 24 | 9.3e-15 | 1.3e-15 | 2.7e-16 |
| 2, 1 | 4, 0.8 | 58.64 | 109 | 5.2e-14 | 2.0e-15 | 2.2e-16 |
| 3, 1 | 5, 0.12 | 2.18 | 7 | 1.6e-14 | 7.8e-16 | 3.3e-16 |

Notes on these runs:
- The amplitude-0.8 start did not converge within the default cap of 60 iterations, but S was still decreasing steadily (58.6 → 19.3; `n1c-NC2-n1-global-amp08-60it.txt`). With 400 allowed iterations it converged after 109 (`N1C_ITERS=400`).
- A first NC=3 run with noise 0.3 started at S ≈ 19.7 and was descending at about 20 s per iteration. I stopped it to stay within the time budget (`n1c-NC3-n1-global-aborted.txt`).

**Pass**, with one exception: NC=2, n=2. There the coarse plaquette is exp(2πi n·σ/2) = −1, a central element at the maximum of the coarse plaquette action. The constrained Hessian is positive *semi*definite with 2 zero modes. That case is as far from small field as possible and does not test the lemma.

**Meaning.**
- **Criticality is algebraic.** On the fibre, U_F is critical with multiplier 0 for every flux. Zero current needs no small field.
- **The averaging constraint is essential.** Without it, U_F is a *saddle* of S on the slice. It has 4n negative modes (8 + 1 for NC=3, n=2): these are the Nielsen–Olesen modes of the charged components, one lowest Landau level of degeneracy 2n per complex component. It also has 3|T| (+1) zero modes. These are the coarse gauge directions g = E₁(h), i.e. h ∈ G^{T} transported along the tree contours, which leave the tree links fixed; the +1 is one further flat direction (an abelian modulus), whose count depends on the zero tolerance. The constraint removes all of them, because a(U^{E₁(h)}) = V^h ≠ V. The constrained Hessian is coercive, with λ_min within a few per cent of its value at U=1 (0.122–0.134 against 0.134–0.153).
  - **For Lemma constant-flux:** the proof must identify U_F through the constrained problem: critical on the fibre plus uniqueness of the critical orbit (Theorem J) in the small-field region, as the plan says. It must not use any minimality of U_F for S alone; that minimality is false.
- **The global minimizer.** From disordered starts on the fibre (S = 8.6, 10.5 and 58.6 after the first SQP step at NC=2, against S(U_F) = 1.09), the SQP iteration returns to U_F, to 2e-15 in the tree gauge, even though the coarse flux per plaquette is π. No other local minimum was met. So in d=2 the identity U_F = U(a(U_F)) (background field of the average) holds well beyond small field. This is the finite-torus substitute for the infinite-volume 𝐇B_F = A_F in risk 3 of the plan, at the nonlinear level.

## N2. Constant-flux configurations on 4d tori

**What was computed.**
- Tori of side N = 4, 5, 6.
- Integer flux matrices:
  - n₁₂ = 1, n₃₄ = 2;
  - n₁₂ = 1, n₃₄ = −1;
  - all six entries nonzero (Pf(n) = 4).
- Plaquette angle θ_{μν} = 4πn_{μν}/N² in exp(iθ n·σ/2). This is the plan's F = 2πfH/ñ² with H = n·σ, so that e^{2πiH} = 1.
- Construction: θ_μ(x) = Σ_{ν<μ}θ_{νμ}x_ν, with the transition term −NΣ_{ν>μ}θ_{μν}x_ν on the bonds that cross x_μ = N−1.
- Checks:
  - all plaquettes;
  - the current (the gradient of S on every bond and colour, by complex steps);
  - the naive Pfaffian sum Σ_x ε_{μνκλ} tr(F_{μν}(x)F_{κλ}(x)), with F from the plaquette at x;
  - B₄ about the origin, through gauge-invariant data. For rU_F and U_{RnRᵀ} I compared all plaquette traces, tr(P₁P₂) for all pairs of plaquettes with a common base point (these detect relative signs), all Polyakov-loop traces, and tr(Polyakov loop × plaquette). Ten elements r per torus.

**Numbers** (output `n2-out.txt`):

| Check | Result |
|---|---|
| plaquettes constant | ≤ 7.4e-15 |
| current | ≤ 1.5e-14 (control, one bond perturbed by 0.1: 0.29) |
| Pfaffian sum vs 4N⁴Pf(θ) (= 8·½·N⁴Pf(θ)) | agrees to 10 digits |
| (1/32π²)·(Pfaffian sum) | 4.0000000000, −2.0000000000, 8.0000000000 = **2Pf(n)** (in general Pf(f)·tr H²) |
| invariants of rU_F vs U_{RnRᵀ} | ≤ 2.3e-14 |
| invariants of rU_F vs U_F (control, whenever RnRᵀ is not ±n) | differ by 0.12–4.0 |
| Pf sum(rU_F)/Pf sum(U_F) | **det R** to 12 digits, for all 30 elements tried |
| invariants and Pf sum of U_F^g for Haar-random g | unchanged to 5e-15 |

**Pass.**

**Meaning.**
- Every algebraic statement of Lemma constant-flux item 3 holds as stated:
  - constant curvature;
  - zero current for every flux (not only in small field);
  - rU_F ≅ U_{rF} for all of B₄ (including Polyakov loops, so with no flat-modulus mismatch for this construction);
  - Pf(rF) = det R · Pf(F).
- For SU(2), also U_F ≅ U_{−F} (Weyl reflection). This is harmless, since Pf is even.
- Configurations with Pf(f) ≠ 0 exist on every torus side. With f₁₂ = f₃₄ = 1 the plaquette angle is 4π/ñ², small field once ñ² ≳ 4π/h, so the Pfaffian sector is available as option (i) needs.

## N3. Period density against the Fourier p²-tensor (d=4, abelian linearized form)

**What was computed.**
- Pieces are gauge-invariant local quadratic forms q(A) = Σ_k c_k (dA)_{p_k}(dA)_{p′_k}. Families are periodic under the grid Mℤ⁴, M = 2.
- (a) The period density T̄(F) = M^{-4}Σ_{pieces of the cell} q(A_F), as a 6×6 form. It was computed through A_F and, independently, through the plaquette planes; the two agree to 1e-13.
- (b) The p²-tensor of the Bloch-diagonal Hessian per site, in the bond-midpoint convention. It was obtained from the exact second moments, checked against Re Π(hu)/h² at h = 1e-3, and recomputed in the b₋ convention.
- (c) Φ(T̄) against T. Here Φ(K)(p)_{μν} = 2K(p∧e_μ, p∧e_ν), which encodes T̄(p∧a) = ½aᵀT(p)a. Numerically, ker Φ = ℝPf and Φ(Iso) = 2(δ|p|² − pp).
  - The decomposition used: T̄ = α·Iso + τ·Pf + aniso, and T = β(δ|p|² − pp) + rest.
  - The preimage of T orthogonal to Pf was formed, giving (a) − (b) = τ′Pf + residual.
- Families:
  - **R:** 2 random pieces, no symmetry.
  - **S:** the R pieces under all of B₄ about z = (½,½,½,½), 768 pieces per cell. The sum is B₄-invariant; single pieces are not.
  - **S⁺:** the same with B₄⁺ only.
  - **N:** a null family, 0.37·Σε F_{μν}(x)F_{κλ}(x+μ̂+ν̂) at every site (Lüscher's abelian cup product, an exact coboundary).
  - **S+N.**
  - The naive Pf density 0.37·Σε F(x)F(x), which is not null.
- Totals on the periodic torus (ℤ/4)⁴ with random periodic A, under r ∈ B₄ about z.
- `n3b.py`: the same families on the constant-flux torus N=4, n₁₂ = n₃₄ = 1.

**Numbers** (outputs `n3-out.txt`, `n3b-out.txt`):

| Family | α (iso of T̄) | τ (Pf of T̄) | aniso of T̄ | β (Fourier) | \|Φ(T̄) − T\| | (a) − (b) | 2α − β |
|---|---|---|---|---|---|---|---|
| R | 0.01823 | 0.01331 | 0.153 | 0.03646 (non-iso part 0.31) | 1.7e-16 | 0.01331 Pf + 1e-17 | 1e-17 |
| S | 7.00021 | 0 (1e-14) | 1.2e-14 | 14.00042 (non-iso 8e-14) | 1.1e-13 | 0 Pf + 1e-14 | 1e-14 |
| S⁺ | 3.50010 | **2.55553** | 4e-15 | 7.00021 (non-iso 4e-14) | 4.6e-14 | 2.55553 Pf | 4e-15 |
| N | 0 | **2.96 = 8·0.37** | 0 | **0** (also Π(p) ≡ 0 at finite p) | 0 | 2.96 Pf | 0 |
| S+N | 7.00021 | **2.96** | 1.2e-14 | 14.00042 | 1.1e-13 | 2.96 Pf | 1e-14 |
| naive Pf | 0 | 2.96 | 0 | 0 (but Π ≠ 0 at order p⁴) | 0 | 2.96 Pf | 0 |

Additional facts:
- The orders p⁰ and p¹ of Π vanish (≤ 3e-17) for every family.
- The b₋ and midpoint conventions give the same p²-tensor (≤ 4e-14).

On the periodic torus, under r ∈ B₄ about z:

| Total | Q(rA) − Q(A) |
|---|---|
| S | ≤ 1.4e-11 (\|Q\| = 797) |
| S+N | ≤ 1.9e-11 |
| one piece of S (control) | 52–78 |
| S⁺, reflections | 33 |
| N | ≡ 0 (≤ 7e-13; Σ\|terms\| = 221) |

On the constant-flux torus (`n3b-out.txt`):

| Family | Behaviour under reflections |
|---|---|
| S | invariant |
| N | 467.42 → −467.42, = 8τ′Pf(θ)N⁴ |
| S+N | 2678.28 → 1743.43 |
| S⁺ | changes by 807.1 = 2·2.5555·Pf(θ)N⁴ |

All are invariant under rotations.

**Pass**, for (a)–(d).

**The Wilson action and Part XI's normalization (SU(2), nonabelian, real functions).**
- The period density was computed as ½∂_t² of the density on exp(itA_F n·σ/2).
- The Fourier tensor came from the full SU(2) Hessian at U=1, colour by colour.

| Function | α (x-coords) | τ | β (x-coords) | 2α/β | colour off-diagonal |
|---|---|---|---|---|---|
| S (plaquettes) | **0.25000000** | 0 | **0.50000000 = T_ρ** | 1.00000000 | 0 |
| 2×1 rectangles | 2.00000000 | 0 | 4.00000000 | 1.00000000 | 0 |
| figure-eight tr(P₀₁P₂₃) | 0.08333333 | **0.16666667** | 0.16666667 | 1.00000000 | 0 |

- \|Φ(a) − (b)\| ≤ 7e-9, which is the finite-difference error.
- Rectangles relative to S: 7.99999999 from the period density and 8.00000000 from the Fourier tensor (the Symanzik value is 8).

**The conversion factor of D2, stated explicitly.**
1. **The period density of S:** T̄_S(F) = ½Σ_{μ<ν} tr F_{μν}² = (T_ρ/2)Σ_{μ<ν}Σ_a(F^a_{μν})² per site. In σ/2 components this is ¼Σ_{μ<ν}f_{μν}², and α = 0.25 numerically.
2. **The marginal tensor of S:** in the trace pairing ⟨A,B⟩ = Σ tr(AB) of Section action, which s54:100 says Part XI uses, it is D₁ᵀD₁, so β_S^{XI} = 1. In Lie components (tr T^aT^b = T_ρδ) every kernel carries a factor T_ρ, so β_S^{comp} = T_ρ (0.5 numerically).
3. **The general identity:** for every local gauge-invariant family, T̄_E(F) = (β_E^{XI}/2)·Σ_{μ<ν} tr F_{μν}² + τ_E·Pf_tr(F) + (anisotropic part, absent when T_E is isotropic), with Pf_tr(F) = tr(F₀₁F₂₃ − F₀₂F₁₃ + F₀₃F₁₂). Hence, defining β_E by T̄_E = β_E T̄_S + τ_E Pf_tr:
   - **β_E = β_E^{XI} / β_S^{XI} = β_E^{XI}. The conversion factor is 1.**
   - The ½ of c^X is matched by the ½ of T̄_S.
   - T_ρ cancels in the ratio, and is present in both numerator and denominator if Lie components are used.
   - The midpoint convention is irrelevant at order p² (checked), and A_F has the same values at b₋ and at the midpoint.
4. **If c^X uses e^{iη_j tA_F}** with a lattice spacing η_j ≠ 1, both T̄_E and T̄_S per site carry η_j⁴. The ratio β_E is unaffected. Comparison with Part XI's β_Φ, defined with e^{itB} on the unit lattice, needs η_j = 1 in the units of scale j (or division by η_j⁴).
5. **Recommendation:** Def marginal-tensor (s51:177) should name the pairing explicitly ("kernel with respect to ⟨A,B⟩ = Σ tr(AB); Lie indices suppressed relative to it"). Otherwise "proportional to the identity on 𝔤" leaves a T_ρ ambiguity. At present the pairing is stated only in s54:100.

**Meaning for Lemma period-vacpol.**
- The algebraic core holds **piece by piece, with no symmetry**: Φ(T̄) = T to 1e-16 even for the random family R, and ker Φ = ℝPf.
- So "(a) − (b) is a multiple of Pf, and 2·iso(T̄) = β" is the identity T̄(p∧a) = ½aᵀT(p)a, which follows from gauge invariance and the second-order Taylor expansion, together with the Plücker fact.
- The proof in the plan (null Lagrangians vanish on decomposable bivectors) is right. It needs only absolutely summable second moments of the pieces, and it can be written as this identity.
- The SU(2) checks show the same for nonabelian functions: Hessians at U=1 are colour diagonal, and Φ(T̄) = T.

**Meaning for Prop period-iso and decision D1.**
- **The gap of plan 2.3 is real.**
  - S+N has a total quadratic form that is *exactly* B₄-invariant on the periodic torus, and the same Fourier tensor as S, yet its period density carries τ = 2.96.
  - B₄⁺-symmetrization (S⁺) gives an isotropic Fourier tensor and τ ≠ 0.
  - Nonabelian Wilson-loop functions carry Pf densities naturally (figure-eight: τ = 1/6) that are invisible in Π.
  - Hence B₄-invariance of the total on periodic configurations, or isotropy of Π, does **not** give τ = 0.
- **What does give τ = 0:**
  - Covariance of the family as a family. For S, τ = 0 exactly, although no single piece is invariant.
  - Evaluation on constant-flux configurations with Pf(f) ≠ 0. On the constant-flux torus the null family's total is 8τ′Pf(θ)N⁴ ≠ 0 and changes sign under reflections, while S stays invariant (`n3b`).
- This is exactly the mechanism of option (i). Its step 4 (invariance of the total on the Pf sector ⇒ τ = 0) is confirmed in the toy model. The plan's recommendation (i) stands.

## Reproduction

All commands are run in `scratchpad/y3num/`:
- `python3 n1ab.py 1`
- `python3 n2.py 1`
- `python3 n3.py 1`
- `python3 n3b.py 1`
- `python3 n1c.py NC n seed nstart amp`:
  - the second-order cases use nstart = 0;
  - the global runs used `2 1 2 2 0.3`, `N1C_ITERS=400 … 2 1 4 1 0.8` and `3 1 5 1 0.12`.

`timing.py` only measured the cost of `symavg` evaluations.

---

## Consequences for the plan

1. **Lemma constant-flux.**
   - Items 3 and 4 hold as stated (N2), and the averaging statement holds exactly and nonlinearly (N1(b)): a(U_F) is constant flux with the same physical F (L²F in coarse lattice units), with no shift.
   - The identification U_F = U_j(U_F^{(j)}) must go through the constrained problem: criticality on the fibre with multiplier 0, plus Theorem J's uniqueness. U_F is a saddle of S itself (4n Nielsen–Olesen modes in d=2), and only the averaging constraint makes it a minimizer (N1(c)).
   - In d=2 the identification holds far outside small field: from random starts, and with coarse flux up to 2.8 per plaquette.
   - The topological charge of U_F is Pf(f)·tr H² (2Pf(f) for SU(2) with H = n·σ).
2. **Lemma period-vacpol.** Confirmed, including the normalization: T̄(p∧a) = ½aᵀT(p)a, iso(T̄) = β/2, and (a) − (b) = τPf. The identity holds piece by piece, so the lemma needs no B₄ symmetry. Exponential decay of the pieces suffices, as the plan says.
3. **Decision D2.** Define β_E by T̄_E = β_E T̄_S, with T̄_S(F) = ½Σ_{μ<ν} tr F_{μν}². Then **β_E = β_E^{XI} exactly** (factor 1), provided Part XI's kernels are taken in the trace pairing (β_S^{XI} = 1). In σ/2 or T^a components both sides carry T_ρ. The midpoint convention does not matter. The pairing should be stated in Def marginal-tensor.
4. **Coarse-to-fine step of the β agreement.**
   - The averaging half is exact: 𝐐A_F = B_F with γ_sym(A_F) = 0 (N1(a)), so the "γ-gradient" vanishes for symmetric-gauge A_F, and the second "to be verified" item of s51 can be closed with the two-line argument above.
   - Lemma ol-pullback's sentence about a "shift by a constant depending on the position of the coarse point in its block" should be dropped (it describes corner blocks).
   - The minimizer half (𝐇B_F = A_F) was not tested in infinite volume. Its nonlinear finite-torus analogue holds in d=2 (N1(c)), which supports the plan's mitigation of risk 3 via constant-flux tori.
5. **Prop period-iso / D1.** The numerics confirm that a B₄-invariant total does not exclude a Pfaffian period density, and that the constant-flux (Pf ≠ 0) sector detects it with a sign flip under reflections. So option (i), or covariance of each generation as a family, is necessary. Option (i)'s mechanism works in the toy model.

No check failed. The one non-positive-definite case (N1(c), NC=2, n=2) is a degenerate extreme, with the coarse plaquette equal to the central element −1. It is positive semidefinite, has no negative modes, and is irrelevant to small field.
