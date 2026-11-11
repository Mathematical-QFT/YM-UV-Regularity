# N-b: exact rational check of Lemma ol-classical (Section sec:ol-feynman).
#
# Computes, in exact rational arithmetic, the spherical average over S^3 of the
# full O(4) tensor of the continuum integrand I_0(omega, 0) of
# Lemma ol-continuum-integrand, from the continuum vertices of Lemma ol-naive-limit:
#   scalar vertex   v_S(q; p, c)          = ((2q+p).c) ad X
#   vector vertex   v_V(q; p, c)_{ab}     = ((2q+p).c) delta_ab ad X + 2 (p_a c_b - p_b c_a) ad X
# with the routing of the lattice integral (propagators at q and q+p).  Colour traces
# give kappa = C_A/T_rho; everything below is per unit kappa.
# The integrand of the p^2-tensor is (1/2) d^2/dp^2 at p = 0 of
#   P_0(q,p)_{mu nu} = G(q) G(q+p) N_{mu nu}(q,p) + (p-independent tadpoles),
#   N_{mu nu} = (1/2) tr_4( vbar_V^{(mu)} v_V^{(nu)} ) - vbar_S^{(mu)} v_S^{(nu)},
# where vbar = vertex at (q+p; -p) with polarization e_mu, v = vertex at (q; p) with e_nu,
# and the signs are those of -1/2 log det(vector) + log det(scalar) and of the bubble.
# At m = 0 and |q| = 1: G(q) = 1 and G(q+p) = 1 - (2q.p + p^2) + (2q.p)^2 + O(p^3).
#
# Spherical moments on S^3 in R^4 (d = 4):
#   < prod_i w_i^{2 b_i} > = prod_i (2 b_i - 1)!! / ( d (d+2) ... (d + 2n - 2) ),  n = sum b_i,
# and 0 if some exponent is odd.  (Gaussian integrals: E[prod x_i^{2b_i}] = prod (2b_i-1)!!,
# E[|x|^{2n}] = d(d+2)...(d+2n-2), and |x| is independent of x/|x|.)
#
# Usage: python3 nb_sphere.py        (pure Python 3, no packages; runs in about a second)
from fractions import Fraction as Fr
from itertools import product

NV = 8                                  # variables q1..q4 (0..3), p1..p4 (4..7)
def var(i):
    e = [0]*NV; e[i] = 1
    return {tuple(e): Fr(1)}
def const(c):
    return {tuple([0]*NV): Fr(c)} if c != 0 else {}
def add(*ps):
    r = {}
    for p in ps:
        for k, v in p.items():
            r[k] = r.get(k, 0) + v
    return {k: v for k, v in r.items() if v != 0}
def scal(c, p):
    return {k: c*v for k, v in p.items() if c*v != 0}
def mul(a, b):
    r = {}
    for k1, v1 in a.items():
        for k2, v2 in b.items():
            k = tuple(x+y for x, y in zip(k1, k2))
            r[k] = r.get(k, 0) + v1*v2
    return {k: v for k, v in r.items() if v != 0}
def pdeg(k):
    return sum(k[4:])
def part(p, d):
    """part of total degree d in the p-variables"""
    return {k: v for k, v in p.items() if pdeg(k) == d}

q = [var(i) for i in range(4)]
pv = [var(4+i) for i in range(4)]
def delta(a, b): return 1 if a == b else 0

# ---- vertices (colour-stripped), as polynomials in q, p ----
def vS(qq, pp, nu):                      # ((2q+p).e_nu)
    return add(scal(2, qq[nu]), pp[nu])
def vV(qq, pp, nu):                      # 4x4 matrix
    M = [[None]*4 for _ in range(4)]
    for a in range(4):
        for b in range(4):
            t = scal(delta(a, b), vS(qq, pp, nu))
            spin = add(scal(2*delta(b, nu), pp[a]), scal(-2*delta(a, nu), pp[b]))
            M[a][b] = add(t, spin)
    return M
def shift_back(qq, pp):                  # (q+p, -p): arguments of vbar
    return [add(qq[i], pp[i]) for i in range(4)], [scal(-1, pp[i]) for i in range(4)]

qb, pb = shift_back(q, pv)

def tr4(A, B):
    return add(*[mul(A[a][b], B[b][a]) for a in range(4) for b in range(4)])

# expansion of G(q+p) at |q| = 1, m = 0, to second order in p
qdotp = add(*[mul(q[i], pv[i]) for i in range(4)])
psq = add(*[mul(pv[i], pv[i]) for i in range(4)])
Gp = add(const(1), scal(-1, add(scal(2, qdotp), psq)), mul(scal(2, qdotp), scal(2, qdotp)))

def p2_tensor(N):
    """degree-2 part in p of N * G(q+p) (at |q|=1, m=0): this is (1/2) d^2_p at p=0, as a quadratic form in p"""
    return part(mul(N, Gp), 2)

# ---- spherical moments ----
def dfact(n):
    r = 1
    while n > 1:
        r *= n; n -= 2
    return r
def moment(exps, d=4):
    if any(e % 2 for e in exps): return Fr(0)
    n = sum(exps)//2
    den = 1
    for i in range(n): den *= d + 2*i
    num = 1
    for e in exps: num *= dfact(e-1)
    return Fr(num, den)
def sphere_avg(poly):
    """average over |q|=1 of a polynomial in q (coefficients may be polynomials in p): returns a polynomial in p"""
    r = {}
    for k, v in poly.items():
        m = moment(k[:4])
        if m != 0:
            kk = tuple([0]*4) + k[4:]
            r[kk] = r.get(kk, 0) + m*v
    return {k: v for k, v in r.items() if v != 0}

# check of the moment formula: sum_i <w_i^2 w^a> = <w^a> for all monomials of degree <= 6
bad = 0
for a in product(range(7), repeat=4):
    if sum(a) > 6: continue
    lhs = sum(moment(tuple(a[j] + 2*(j == i) for j in range(4))) for i in range(4))
    if lhs != moment(a): bad += 1
print('moment formula: <1> = %s, <w1^2> = %s, <w1^4> = %s, <w1^2 w2^2> = %s; |w|^2 = 1 consistency failures: %d'
      % (moment((0,0,0,0)), moment((2,0,0,0)), moment((4,0,0,0)), moment((2,2,0,0)), bad))

# ---- the tensors ----
def coeff_pp(poly, k, l):
    """coefficient T_{kl} (symmetric) of the quadratic form sum T_kl p_k p_l"""
    e = [0]*NV; e[4+k] += 1; e[4+l] += 1
    c = poly.get(tuple(e), Fr(0))
    return c if k == l else c/2

parts = {}
for name in ('orbital-vector', 'spin', 'scalar', 'total'):
    parts[name] = [[None]*4 for _ in range(4)]
for mu in range(4):
    for nu in range(4):
        VVb, VV = vV(qb, pb, mu), vV(q, pv, nu)
        tr = tr4(VVb, VV)
        orb_only = scal(4, mul(vS(qb, pb, mu), vS(q, pv, nu)))       # 4 = tr_4 of the identity part
        Nvec_orb = scal(Fr(1, 2), orb_only)                         # -1/2 * (-bubble)
        Nvec_spin = scal(Fr(1, 2), add(tr, scal(-1, orb_only)))     # cross terms vanish (tr N = 0), checked below
        Nsc = scal(-1, mul(vS(qb, pb, mu), vS(q, pv, nu)))          # +1 * (-bubble)
        for name, N in (('orbital-vector', Nvec_orb), ('spin', Nvec_spin), ('scalar', Nsc),
                        ('total', add(scal(Fr(1, 2), tr), Nsc))):
            parts[name][mu][nu] = sphere_avg(p2_tensor(N))

def report(name):
    T = parts[name]
    # full tensor T_{mu nu k l}
    full = {(mu, nu, k, l): coeff_pp(T[mu][nu], k, l) for mu in range(4) for nu in range(4) for k in range(4) for l in range(4)}
    # fit a delta_{mu nu} delta_{kl} + b (delta_{mu k} delta_{nu l} + delta_{mu l} delta_{nu k})
    a = full[(0, 0, 1, 1)]
    b = full[(0, 1, 0, 1)]
    resid = max(abs(full[(mu, nu, k, l)] - (a*delta(mu, nu)*delta(k, l) + b*(delta(mu, k)*delta(nu, l) + delta(mu, l)*delta(nu, k))))
                for (mu, nu, k, l) in full)
    t1, t2 = a, 2*b                       # T(p) = t1 delta |p|^2 + t2 p p
    iso = (2*t1 - t2)/3                   # isotropic coefficient (projection onto delta|p|^2 - pp)
    print('%-15s a = %-8s b = %-8s  O(4) residual = %s   a + 2b = %s   isotropic coefficient = %s'
          % (name, a, b, resid, a + 2*b, iso))
    return iso, resid, a + 2*b

print('\nSpherical average of the p^2-tensor of I_0(omega,0), per unit kappa = C_A/T_rho;')
print('form a delta_{mu nu} delta_{kl} + b (delta_{mu k} delta_{nu l} + delta_{mu l} delta_{nu k}); transversality <=> a + 2b = 0')
res = {name: report(name) for name in ('scalar', 'orbital-vector', 'spin', 'total')}

# vector bubble: cross terms between orbital and spin parts have zero trace
cross_ok = True
for mu in range(4):
    for nu in range(4):
        VVb, VV = vV(qb, pb, mu), vV(q, pv, nu)
        Sb = [[add(VVb[a][b], scal(-delta(a, b), vS(qb, pb, mu))) for b in range(4)] for a in range(4)]
        S = [[add(VV[a][b], scal(-delta(a, b), vS(q, pv, nu))) for b in range(4)] for a in range(4)]
        c1 = add(*[mul(vS(qb, pb, mu), S[a][a]) for a in range(4)])
        c2 = add(*[mul(Sb[a][a], vS(q, pv, nu)) for a in range(4)])
        if c1 or c2: cross_ok = False
print('\norbital x spin cross terms vanish identically: %s' % cross_ok)

iso = res['total'][0]
print('\nTotal: isotropic coefficient %s * kappa of the spherical average; exact value 11/3: %s' % (iso, iso == Fr(11, 3)))
print('Orbital (vector + scalar) = %s, spin = %s  (expected -1/3 and 4)' % (res['orbital-vector'][0] + res['scalar'][0], res['spin'][0]))
print('Per log det(-D^2+m^2) (scalar, coefficient +1, bubble sign -1): %s  (expected +1/3)' % (res['scalar'][0]))
print('b = (2 pi)^{-4} |S^3| * %s * kappa = %s kappa / (8 pi^2) = 11 kappa / (24 pi^2): %s'
      % (iso, iso, iso/8 == Fr(11, 24)))
import math
print('SU(2), fundamental (kappa = 4): b = %.6f ;  11/(6 pi^2) = %.6f' % (float(iso)*4/(8*math.pi**2), 11/(6*math.pi**2)))
ok = (bad == 0 and cross_ok and iso == Fr(11, 3) and all(r[1] == 0 and r[2] == 0 for r in res.values()))
print('\nN-b: %s' % ('PASS' if ok else 'FAIL'))
