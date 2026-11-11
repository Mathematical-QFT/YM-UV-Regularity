# N3 of the Y3 plan: period density versus Fourier p^2-tensor, d=4, abelian linearized form (one colour), plus the
# normalization of the Wilson action (SU(2), matrix trace) relative to Part XI.
#
# A piece is a gauge-invariant local quadratic form q(A) = sum_k c_k (dA)_{p_k} (dA)_{p'_k} in the plaquette variables.
# A family is grid periodic: the pieces of one period cell Q = {0,..,M-1}^4 and their translates by M Z^4.
#   (a) period density  Tbar(F) = M^{-4} sum_{pieces of Q} q(A_F),  A_F(x)_mu = 1/2 sum_nu x_nu F_{nu mu}   (a 6x6 form in F)
#   (b) Fourier p^2-tensor of the total: Bloch-diagonal Hessian kernel per site, bond-midpoint convention,
#       Pi(p)_{mu nu} = sum T_{mu nu ka la} p_ka p_la + O(p^3)
#   (c) Phi(Tbar) = T, where Phi(K)(p)_{mu nu} = 2 K(p^e_mu, p^e_nu);  ker Phi = R Pf;  so (a) - (b) = tau Pf, iso parts equal
#   (d) a null family (cup product F u F, Luescher's abelian density) changes (a) by 8 tau' Pf and (b) by 0
#   python3 n3.py [seed]
import sys, math, random, itertools, time
import os
NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, NUM)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
random.seed(seed)
d = 4
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]; PI = {p: i for i, p in enumerate(PAIRS)}
EPS = {}
for p in itertools.permutations(range(d)):
    s = 1
    for i in range(d):
        for j in range(i + 1, d):
            if p[i] > p[j]: s = -s
    EPS[p] = s
def e(mu, n=1): v = [0]*d; v[mu] = n; return tuple(v)
def add(x, y): return tuple(a + b for a, b in zip(x, y))
B4 = [(perm, sgn) for perm in itertools.permutations(range(d)) for sgn in itertools.product((1, -1), repeat=d)]
def detr(r):
    perm, sgn = r; s = 1
    for i in range(d):
        for j in range(i + 1, d):
            if perm[i] > perm[j]: s = -s
    for v in sgn: s *= v
    return s
B4p = [r for r in B4 if detr(r) == 1]

# ------------------------------------------------------------------ plaquettes, pieces
def plaq_bonds(p):
    """(dA)_p for p = (x, mu, nu), mu < nu: list of ((site, dir), sign)."""
    x, mu, nu = p
    return [((x, mu), 1), (((add(x, e(mu))), nu), 1), ((add(x, e(nu)), mu), -1), ((x, nu), -1)]
def r_point(r, x, z):     # r(x) = R(x - z) + z with (Rv)_{perm[i]} = sgn[i] v_i; z half-integer, coordinates as floats then int
    perm, sgn = r; out = [0.0]*d
    for i in range(d): out[perm[i]] = sgn[i]*(x[i] - z[i]) + z[perm[i]]
    return tuple(int(round(v)) for v in out)
def r_plaq(r, p, z):
    """Image of the plaquette p under r: returns (standard plaquette, orientation sign)."""
    x, mu, nu = p; perm, sgn = r
    rx = r_point(r, x, z); a, sa = perm[mu], sgn[mu]; b, sb = perm[nu], sgn[nu]
    base = list(rx)
    if sa < 0: base[a] -= 1
    if sb < 0: base[b] -= 1
    if a < b: return (tuple(base), a, b), sa*sb
    return (tuple(base), b, a), -sa*sb
def r_piece(r, q, z):
    """(r.q)(A) = q(r^{-1} A): (d(r^{-1}A))_p = sigma (dA)_{rp}."""
    out = []
    for (c, p1, p2) in q:
        P1, s1 = r_plaq(r, p1, z); P2, s2 = r_plaq(r, p2, z)
        out.append((c*s1*s2, P1, P2))
    return out
def rand_piece(nterms, anchor, ext=2):
    q = []
    for _ in range(nterms):
        p1 = (add(anchor, tuple(random.randint(0, ext) for _ in range(d))),) + random.choice(PAIRS)
        p2 = (add(anchor, tuple(random.randint(0, ext) for _ in range(d))),) + random.choice(PAIRS)
        q.append((random.gauss(0, 1), p1, p2))
    return q
def null_piece(x, c):
    """c * sum eps F_{mu nu}(x) F_{ka la}(x + mu + nu) (cup product), in standard plaquettes."""
    q = []
    for (mu, nu, ka, la), s in EPS.items():
        p1 = (x, min(mu, nu), max(mu, nu)); s1 = 1 if mu < nu else -1
        y = add(add(x, e(mu)), e(nu))
        p2 = (y, min(ka, la), max(ka, la)); s2 = 1 if ka < la else -1
        q.append((c*s*s1*s2, p1, p2))
    return q
def naive_pf_piece(x, c):
    q = []
    for (mu, nu, ka, la), s in EPS.items():
        s1 = 1 if mu < nu else -1; s2 = 1 if ka < la else -1
        q.append((c*s*s1*s2, (x, min(mu, nu), max(mu, nu)), (x, min(ka, la), max(ka, la))))
    return q

# ------------------------------------------------------------------ (a) period density
def AF(F, b):
    (x, mu) = b; return 0.5*sum(x[n]*F[n][mu] for n in range(d))
def eval_piece(q, A):
    s = 0.0
    for (c, p1, p2) in q:
        s += c*sum(sg*A(b) for b, sg in plaq_bonds(p1))*sum(sg*A(b) for b, sg in plaq_bonds(p2))
    return s
def Fmat(v):
    F = [[0.0]*d for _ in range(d)]
    for I, (a, b) in enumerate(PAIRS): F[a][b] = v[I]; F[b][a] = -v[I]
    return F
def period_density(pieces, M):
    def T(v):
        F = Fmat(v); return sum(eval_piece(q, lambda b: AF(F, b)) for q in pieces)/M**d
    K = [[0.0]*6 for _ in range(6)]
    for I in range(6):
        v = [0.0]*6; v[I] = 1.0; K[I][I] = T(v)
    for I in range(6):
        for J in range(I + 1, 6):
            v = [0.0]*6; v[I] = 1.0; v[J] = 1.0; K[I][J] = K[J][I] = (T(v) - K[I][I] - K[J][J])/2
    return K
def period_density_direct(pieces, M):     # (dA_F)_p = F_{mu nu} on every plaquette: the same form from the plaquette planes
    K = [[0.0]*6 for _ in range(6)]
    for q in pieces:
        for (c, p1, p2) in q:
            I, J = PI[(p1[1], p1[2])], PI[(p2[1], p2[2])]
            K[I][J] += c/2/M**d; K[J][I] += c/2/M**d
    return K

# ------------------------------------------------------------------ (b) Fourier p^2 tensor (Bloch diagonal, per site)
def fourier_tensor(pieces, M, conv='mid'):
    T = [[[[0.0]*d for _ in range(d)] for _ in range(d)] for _ in range(d)]
    o0 = [[0.0]*d for _ in range(d)]; o1 = [[[0.0]*d for _ in range(d)] for _ in range(d)]
    def m(b):
        x, mu = b
        return tuple(x[i] + (0.5 if (i == mu and conv == 'mid') else 0.0) for i in range(d))
    for q in pieces:
        for (c, p1, p2) in q:
            for b1, s1 in plaq_bonds(p1):
                for b2, s2 in plaq_bonds(p2):
                    w = c*s1*s2; D = [u - v for u, v in zip(m(b1), m(b2))]; mu1, mu2 = b1[1], b2[1]
                    # c s s' [delta_{mu(b1) mu} delta_{mu(b2) nu} e^{-ip.D} + delta_{mu(b2) mu} delta_{mu(b1) nu} e^{+ip.D}]
                    o0[mu1][mu2] += w; o0[mu2][mu1] += w
                    for k in range(d):
                        o1[mu1][mu2][k] += -w*D[k]; o1[mu2][mu1][k] += w*D[k]
                        for l in range(d):
                            T[mu1][mu2][k][l] += -0.5*w*D[k]*D[l]; T[mu2][mu1][k][l] += -0.5*w*D[k]*D[l]
    n = M**d
    return [[[[T[a][b][k][l]/n for l in range(d)] for k in range(d)] for b in range(d)] for a in range(d)], \
           max(abs(v)/n for r in o0 for v in r), max(abs(v)/n for r in o1 for s in r for v in s)
def Pi_exact(pieces, M, p):   # Bloch-diagonal kernel at finite p (complex), midpoint convention
    P = [[0j]*d for _ in range(d)]
    def m(b):
        x, mu = b; return tuple(x[i] + (0.5 if i == mu else 0.0) for i in range(d))
    for q in pieces:
        for (c, p1, p2) in q:
            for b1, s1 in plaq_bonds(p1):
                for b2, s2 in plaq_bonds(p2):
                    w = c*s1*s2; ph = sum(pk*(u - v) for pk, u, v in zip(p, m(b1), m(b2)))
                    P[b1[1]][b2[1]] += w*complex(math.cos(ph), -math.sin(ph)); P[b2[1]][b1[1]] += w*complex(math.cos(ph), math.sin(ph))
    return [[v/M**d for v in r] for r in P]

# ------------------------------------------------------------------ (c) the map Phi, projections
def C(mu, I, k):
    a, b = PAIRS[I]; return (1 if (k == a and b == mu) else 0) - (1 if (k == b and a == mu) else 0)
def Phi(K):
    return [[[[2*sum(K[I][J]*0.5*(C(mu, I, k)*C(nu, J, l) + C(mu, I, l)*C(nu, J, k)) for I in range(6) for J in range(6))
               for l in range(d)] for k in range(d)] for nu in range(d)] for mu in range(d)]
ISO = [[1.0 if I == J else 0.0 for J in range(6)] for I in range(6)]
PF = [[0.0]*6 for _ in range(6)]
for (I, J, s) in ((0, 5, 0.5), (1, 4, -0.5), (2, 3, 0.5)): PF[I][J] = PF[J][I] = s
T0 = [[[[(1.0 if (m == n and k == l) else 0.0) - 0.5*((1.0 if (m == k and n == l) else 0.0) + (1.0 if (m == l and n == k) else 0.0))
         for l in range(d)] for k in range(d)] for n in range(d)] for m in range(d)]
def fip(A, B): return sum(a*b for r, s in zip(A, B) for a, b in zip(r, s))
def tip(A, B): return sum(A[m][n][k][l]*B[m][n][k][l] for m in range(d) for n in range(d) for k in range(d) for l in range(d))
def tdiff(A, B): return max(abs(A[m][n][k][l] - B[m][n][k][l]) for m in range(d) for n in range(d) for k in range(d) for l in range(d))
def tmax(A): return max(abs(A[m][n][k][l]) for m in range(d) for n in range(d) for k in range(d) for l in range(d))
def decompose(K):
    a = fip(K, ISO)/fip(ISO, ISO); t = fip(K, PF)/fip(PF, PF)
    an = [[K[I][J] - a*ISO[I][J] - t*PF[I][J] for J in range(6)] for I in range(6)]
    return a, t, max(abs(v) for r in an for v in r)
# basis of symmetric 6x6 forms orthogonal to Pf, and the least-squares preimage under Phi
SYMB = []
for I in range(6):
    for J in range(I, 6):
        B = [[0.0]*6 for _ in range(6)]; B[I][J] = B[J][I] = 1.0; SYMB.append(B)
def preimage(T):
    """K orthogonal to Pf with Phi(K) = T (least squares over the 20-dim complement of Pf); returns K and |Phi(K) - T|."""
    basis = []
    for B in SYMB:
        if B[0][5] != 0: continue          # drop B_05: the other 20 project onto a basis of the complement of Pf
        c = fip(B, PF)/fip(PF, PF); basis.append([[B[I][J] - c*PF[I][J] for J in range(6)] for I in range(6)])
    imgs = [Phi(B) for B in basis]
    n = len(basis); G = [[tip(imgs[i], imgs[j]) for j in range(n)] for i in range(n)]; rhs = [tip(imgs[i], T) for i in range(n)]
    A = [G[i][:] + [rhs[i]] for i in range(n)]; pmin = float('inf')
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(A[r][i])); A[i], A[piv] = A[piv], A[i]; pmin = min(pmin, abs(A[i][i]))
        for r in range(n):
            if r != i and A[r][i]:
                f = A[r][i]/A[i][i]; A[r] = [u - f*v for u, v in zip(A[r], A[i])]
    x = [A[i][n]/A[i][i] for i in range(n)]
    K = [[sum(x[k]*basis[k][I][J] for k in range(n)) for J in range(6)] for I in range(6)]
    return K, tdiff(Phi(K), T)

def report(name, pieces, M, extra=''):
    t0 = time.time()
    Ka = period_density(pieces, M); Kd = period_density_direct(pieces, M)
    Tb, o0, o1 = fourier_tensor(pieces, M, 'mid'); Tm, _, _ = fourier_tensor(pieces, M, 'minus')
    al, ta, an = decompose(Ka)
    beta = tip(Tb, T0)/tip(T0, T0); Tani = tdiff(Tb, [[[[beta*T0[m][n][k][l] for l in range(d)] for k in range(d)] for n in range(d)] for m in range(d)])
    e_c = tdiff(Phi(Ka), Tb)
    Kb, res = preimage(Tb)
    D = [[Ka[I][J] - Kb[I][J] for J in range(6)] for I in range(6)]; aD, tD, anD = decompose(D)
    # finite-p check of the Taylor tensor: Pi(h u)/h^2 -> T(u)
    u = [random.gauss(0, 1) for _ in range(d)]; h = 1e-3
    Pe = Pi_exact(pieces, M, [h*v for v in u])
    Tu = [[sum(Tb[m][n][k][l]*u[k]*u[l] for k in range(d) for l in range(d)) for n in range(d)] for m in range(d)]
    e_fin = max(abs(Pe[m][n].real/h**2 - Tu[m][n]) for m in range(d) for n in range(d))
    print('%s%s' % (name, extra))
    print('   (a) period density: iso coefficient alpha = %+.10f, Pf coefficient tau = %+.10f, anisotropic remainder %.2e'
          '   [via A_F vs plaquette planes: %.1e]' % (al, ta, an, max(abs(a - b) for r, s in zip(Ka, Kd) for a, b in zip(r, s))))
    print('   (b) Fourier p^2 tensor: beta = %+.10f, non-isotropic remainder %.2e;  orders p^0, p^1: %.1e, %.1e;  b- convention vs midpoint: %.1e;'
          '  Re Pi(hu)/h^2 - T(u) at h=1e-3: %.1e' % (beta, Tani, o0, o1, tdiff(Tb, Tm), e_fin))
    print('   (c) |Phi(a) - (b)| = %.1e (scale %.2f);  preimage of (b) orthogonal to Pf (residual %.1e): (a) - (b) = %+.10f Pf + %.1e;  2 alpha - beta = %.1e'
          % (e_c, tmax(Tb), res, tD, max(abs(aD), anD), 2*al - beta))
    print('       [%.1f s]' % (time.time() - t0), flush=True)
    return al, ta, an, beta

print('N3, d=4, abelian; seed %d' % seed)
print('Checks of the map Phi: |Phi(Pf)| = %.1e;  Phi(Iso) - 2 T0 = %.1e;  rank of Phi on symmetric forms = 20 is tested by the preimage residuals below'
      % (tmax(Phi(PF)), tdiff(Phi(ISO), [[[[2*T0[m][n][k][l] for l in range(d)] for k in range(d)] for n in range(d)] for m in range(d)])))

M = 2; z = (0.5, 0.5, 0.5, 0.5)
cell = list(itertools.product(range(M), repeat=d))
base = [rand_piece(6, random.choice(cell)) for _ in range(2)]
famR = base
famS = [r_piece(r, q, z) for q in base for r in B4]
famSp = [r_piece(r, q, z) for q in base for r in B4p]
tau_n = 0.37
famN = [null_piece(x, tau_n) for x in cell]
famNaive = [naive_pf_piece(x, tau_n) for x in cell]
print('\nGrid M=%d (cell {0,1}^4), B_4 about z = %s; base: %d random pieces of 6 plaquette products each, plaquettes within {0,1,2}^4 of an anchor'
      % (M, z, len(base)))
report('R  random family (no symmetry)', famR, M)
aS, tS, anS, bS = report('S  B_4-symmetrized family (sum B_4-invariant about z; pieces not)', famS, M, ' : %d pieces per cell' % len(famS))
report('S+ B_4^+-symmetrized family (rotations only)', famSp, M, ' : %d pieces per cell' % len(famSp))
report('N  null family: %.2f * sum eps F_{mu nu}(x) F_{ka la}(x+mu+nu) at every site' % tau_n, famN, M, ' (predicted (a) = 8*%.2f Pf = %.2f Pf, (b) = 0)' % (tau_n, 8*tau_n))
report('S+N', famS + famN, M, ' (sum B_4-invariant: S invariant, N sums to 0)')
report('naive Pf density %.2f * sum eps F_{mu nu}(x) F_{ka la}(x) (not null: its total is O(p^3))' % tau_n, famNaive, M)

# ------------------------------------------------------------------ the totals on a periodic torus
NT = 4
def torus_total(pieces, A):
    """Sum over all Mz^4 translates on the torus (Z/NT)^4 of the pieces; A: dict over (site, dir) of the torus."""
    s = 0.0
    def Ab(b):
        (x, mu) = b; return A[(tuple(v % NT for v in x), mu)]
    for a in itertools.product(range(0, NT, M), repeat=d):
        for q in pieces:
            for (c, p1, p2) in q:
                s += c*sum(sg*Ab((add(b[0], a), b[1])) for b, sg in plaq_bonds(p1))*sum(sg*Ab((add(b[0], a), b[1])) for b, sg in plaq_bonds(p2))
    return s
def r_field(r, A):
    """(rA)(r b) = A(b) on the torus, r about z."""
    out = {}
    for (x, mu), v in A.items():
        perm, sgn = r; rx = r_point(r, x, z); nu = perm[mu]
        if sgn[mu] == 1: out[(tuple(t % NT for t in rx), nu)] = v
        else:
            y = list(rx); y[nu] -= 1; out[(tuple(t % NT for t in y), nu)] = -v
    return out
print('\nTotals on the periodic torus (Z/%d)^4, random periodic A (N(0,1) on every bond), r in B_4 about z:' % NT)
t0 = time.time()
sites = list(itertools.product(range(NT), repeat=d))
A = {(x, mu): random.gauss(0, 1) for x in sites for mu in range(d)}
rs = [((0, 1, 2, 3), (-1, 1, 1, 1)), ((1, 0, 2, 3), (1, 1, 1, 1)), ((2, 0, 3, 1), (1, -1, 1, -1)), ((0, 1, 2, 3), (-1, -1, -1, -1))]
QN = torus_total(famN, A)
print('  null family N: total = %.1e (scale: sum of |terms| = %.1f)' % (QN, abs(torus_total([[(abs(c), p1, p2) for (c, p1, p2) in q] for q in famN], A))))
QS = torus_total(famS, A); QSp = torus_total(famSp, A)
for r in rs:
    rA = r_field(r, A)
    print('  r = perm %s signs %s (det %+d): S total %.1e (|Q| = %.1f);  S+N total %.1e;  S+ total %.1e;  one piece of S: %.2f;  N alone: %.1e'
          % (r[0], r[1], detr(r), abs(torus_total(famS, rA) - QS), abs(QS), abs(torus_total(famS + famN, rA) - QS - QN),
             abs(torus_total(famSp, rA) - QSp), abs(torus_total([famS[0]], rA) - torus_total([famS[0]], A)), abs(torus_total(famN, rA))), flush=True)
print('  (differences Q(rA) - Q(A));  [%.0f s]' % (time.time() - t0))

# ------------------------------------------------------------------ the Wilson action and Part XI's normalization (SU(2))
from su2 import mul, inv, ONE, E, ell, retr
HC = 1e-20
def s_plaq(U, x, mu, nu):
    return 2 - retr(mul(mul(mul(U(x, mu), U(add(x, e(mu)), nu)), inv(U(add(x, e(nu)), mu))), inv(U(x, nu))))
def s_rect(U, x, mu, nu):     # 2 x 1 rectangle: two steps in mu, one in nu
    y1 = add(x, e(mu)); y2 = add(x, e(mu, 2)); y3 = add(y1, e(nu)); y4 = add(x, e(nu))
    return 2 - retr(mul(mul(mul(mul(mul(U(x, mu), U(y1, mu)), U(y2, nu)), inv(U(y3, mu))), inv(U(y4, mu))), inv(U(x, nu))))
def s_eight(U, x, mu, nu, ka, la):   # product of the (mu,nu) and (ka,la) plaquettes based at x
    P1 = mul(mul(mul(U(x, mu), U(add(x, e(mu)), nu)), inv(U(add(x, e(nu)), mu))), inv(U(x, nu)))
    P2 = mul(mul(mul(U(x, ka), U(add(x, e(ka)), la)), inv(U(add(x, e(la)), ka))), inv(U(x, la)))
    return 2 - retr(mul(P1, P2))
# densities: the function's terms based at one site (translation invariant functions)
def dens_S(U, x): return sum(s_plaq(U, x, m, n) for (m, n) in PAIRS)
def dens_rect(U, x): return sum(s_rect(U, x, m, n) + s_rect(U, x, n, m) for (m, n) in PAIRS)
def dens_eight(U, x): return s_eight(U, x, 0, 1, 2, 3)
def su2_period_density(dens, nh):
    """(1/2) d^2/dt^2 dens(exp(i t A_F n.sigma/2)) at t = 0, as a 6x6 form in F (per site, x-coordinates of the Lie algebra)."""
    def T(v):
        F = Fmat(v); h = 1e-4
        U = lambda x, mu: E(tuple((h + 1j*HC)*AF(F, (x, mu))*c for c in nh))
        return 0.5*(complex(dens(U, (0, 0, 0, 0))).imag/HC)/h     # f even in t: f''(0) = f'(h)/h + O(h^2)
    K = [[0.0]*6 for _ in range(6)]
    for I in range(6):
        v = [0.0]*6; v[I] = 1.0; K[I][I] = T(v)
    for I in range(6):
        for J in range(I + 1, 6):
            v = [0.0]*6; v[I] = 1.0; v[J] = 1.0; K[I][J] = K[J][I] = (T(v) - K[I][I] - K[J][J])/2
    return K
def su2_fourier(dens, reach=None):
    """Hessian at U=1 of F(U) = sum_x dens(U, x) in the x-coordinates (U(b) = E(x_b)), per site in Fourier space:
    Pi(p)_{mu nu} = sum over the pairs of bonds of dens(., 0) of H_0(b, b') e^{-ip(m_b - m_b')}; p^2 tensor, midpoint
    convention; colour 0-0, and the largest colour 0-1 entry."""
    rec = set()
    def Urec(x, m): rec.add((x, m)); return ONE
    dens(Urec, (0, 0, 0, 0)); bonds = sorted(rec); h = 1e-5
    T = [[[[0.0]*d for _ in range(d)] for _ in range(d)] for _ in range(d)]; offd = 0.0; o0 = [[0.0]*d for _ in range(d)]
    def mid(b):
        x, m = b; return [x[i] + (0.5 if i == m else 0.0) for i in range(d)]
    for i, b in enumerate(bonds):
        for j, b2 in enumerate(bonds):
            if j < i: continue
            for a2 in (0, 1):
                def U(x, m, sg):
                    v = [0j, 0j, 0j]
                    if (x, m) == b: v[0] += 1j*HC
                    if (x, m) == b2: v[a2] += sg*h
                    return E(tuple(v))
                val = (complex(dens(lambda x, m: U(x, m, 1.0), (0, 0, 0, 0))).imag - complex(dens(lambda x, m: U(x, m, -1.0), (0, 0, 0, 0))).imag)/(2*HC*h)
                if a2 == 1: offd = max(offd, abs(val)); continue
                pairs = [(b, b2)] if i == j else [(b, b2), (b2, b)]
                for (u, w) in pairs:
                    D = [p - q for p, q in zip(mid(u), mid(w))]; o0[u[1]][w[1]] += val
                    for k in range(d):
                        for l in range(d): T[u[1]][w[1]][k][l] += -0.5*val*D[k]*D[l]
    return T, offd, max(abs(v) for r in o0 for v in r)
print('\nWilson action S = sum_p (2 - Re tr U(dp)) for SU(2) (matrix trace, T_rho = 1/2), x-coordinates U = exp(i x.sigma/2):')
nh = (0.0, 0.0, 1.0)
t0 = time.time()
for name, dens, reach in (('S (plaquettes)', dens_S, 1), ('rectangles 2x1', dens_rect, 2), ('figure-eight P_01 P_23', dens_eight, 1)):
    K = su2_period_density(dens, nh); al, ta, an = decompose(K)
    Tf, offd, o0 = su2_fourier(dens)
    beta = tip(Tf, T0)/tip(T0, T0)
    e_c = tdiff(Phi(K), Tf)
    print('  %-24s period density (x-coords): alpha = %.8f, tau = %+.8f, aniso %.1e;  Fourier (x-coords): beta = %.8f, colour off-diagonal %.1e, order p^0 %.1e;'
          '  |Phi(a) - (b)| = %.1e;  2 alpha / beta = %.8f   [%.0f s]' % (name, al, ta, an, beta, offd, o0, e_c, 2*al/beta if beta else float('nan'), time.time() - t0), flush=True)
    if name.startswith('S '): alS, bS_ = al, beta
    if name.startswith('rect'): print('      relative to S: period density %.8f, Fourier %.8f (Symanzik: 8)' % (al/alS, beta/bS_))
