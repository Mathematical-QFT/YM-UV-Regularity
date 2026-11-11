# Numerical checks of Balaban's symmetric averaging (CMP 109, (0.5)-(0.12)); YM4 blueprint, Section Y2, "First step".
# d=2, SU(2), L odd, blocks B(y) of side L centred at the coarse points, coarse torus NC x NC, fine torus (L*NC)^2.
# G(y,x): the coordinate-ordered contours from y to x, one per ordering of the directions in which x-y is nonzero.
# M: the group average, the solution U of sum_j ell(U_j U^{-1}) = 0 (0.10). U(y,x) = M({U(Gamma)}) (0.11).
#
#   python3 symavg.py checks L NC seed            Part A: algebraic checks (seconds)
#   python3 symavg.py chart  L NC seed amp nV     Part B: Theorem ol-chart on the averaged axial slice (long)
# k=2 and d=4 checks: symavg_k2d4.py. SU(2) primitives and the group average M: su2.py.
#
# Conventions as in chart.py: h = E(x) = exp(i x.sigma/2), ell = E^{-1}, R(h) = Ad(h) on components; fine and coarse
# Lie fields in unnormalized coordinates (U = E(A) U0 bondwise). Derivatives by complex steps on holomorphic extensions.
import itertools, math, random, sys, time

MODE = sys.argv[1] if len(sys.argv) > 1 else 'checks'
L = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NC = int(sys.argv[3]) if len(sys.argv) > 3 else 2
assert L % 2 == 1 and L >= 3
NF = L * NC
d = 2
HW = (L - 1)//2
HC = 1e-20      # complex step
DLT = 1e-5      # real step for second derivatives

import su2
from su2 import mul, inv, ONE, E, ell, adj, retr, dist, rand_su2, real, gavg, gavg_iter

# ---------------- lattice geometry ----------------
def site(x): return (x[0] % NF, x[1] % NF)
def add(x, e, n=1): return site((x[0] + n*e[0], x[1] + n*e[1]))
EV = [(1, 0), (0, 1)]
def step(x, mu, sgn=1): return add(x, EV[mu], sgn)
sites = [(i, j) for i in range(NF) for j in range(NF)]
SI = {s: n for n, s in enumerate(sites)}
links = [(s, mu) for s in sites for mu in range(d)]
LI = {l: n for n, l in enumerate(links)}
csites = [(L*i, L*j) for i in range(NC) for j in range(NC)]
CSI = {y: n for n, y in enumerate(csites)}
clinks = [(y, mu) for y in csites for mu in range(d)]
CLI = {c: n for n, c in enumerate(clinks)}
OFFS = [(a, b) for a in range(-HW, HW + 1) for b in range(-HW, HW + 1)]
def block(y): return [add(y, (a, b)) for (a, b) in OFFS]
def compile_path(p):   # p: list of (site, mu, sgn) steps -> list of (link index, sgn)
    return [(LI[(s, mu)], 1) if sg == 1 else (LI[(step(s, mu, -1), mu)], -1) for (s, mu, sg) in p]
def axis_path(x, mu, n):
    p = []; cur = x; sg = 1 if n > 0 else -1
    for _ in range(abs(n)): p.append((cur, mu, sg)); cur = step(cur, mu, sg)
    return p, cur
def contour(y, off, order):
    p = []; cur = y
    for mu in order:
        q, cur = axis_path(cur, mu, off[mu]); p += q
    assert cur == add(y, off)
    return compile_path(p)
def tr(U, cp):
    m = ONE
    for (k, sg) in cp: m = mul(m, U[k] if sg == 1 else inv(U[k]))
    return m
# families G(y,x) (symmetric: all orderings) and the single ordering d,...,1 (the tree contours)
FAMS, FAM1 = {}, {}
for y in csites:
    for off in OFFS:
        nz = [mu for mu in range(d) if off[mu] != 0]
        FAMS[(y, off)] = [contour(y, off, o) for o in itertools.permutations(nz)]
        FAM1[(y, off)] = [contour(y, off, tuple(mu for mu in (1, 0) if off[mu] != 0))]
tree_links = set(k for v in FAM1.values() for (k, sg) in v[0])
free_links = [n for n in range(len(links)) if n not in tree_links]
assert len(tree_links) == len(sites) - len(csites)
LINE = {}            # straight lines [x, x + L e_mu] and [x, x - L e_mu]
for x in sites:
    for mu in range(d):
        for sg in (1, -1): LINE[(x, mu, sg)] = compile_path(axis_path(x, mu, sg*L)[0])

def acv_table(U, kind='M'):
    """Averaged contour variables U(y,x), x = y + off. kind: 'M' (Balaban, (0.11)), 'logmean' (E of the mean of the
    logarithms; not equivariant), 'tree' (the single ordered contour)."""
    out = {}
    for key in FAMS:
        if kind == 'tree': out[key] = tr(U, FAM1[key][0]); continue
        Ts = [tr(U, p) for p in FAMS[key]]
        if kind == 'M': out[key] = gavg(Ts)
        else:
            m = [sum(ell(t)[a] for t in Ts)/len(Ts) for a in range(3)]; out[key] = E(tuple(m))
    return out
def avg_one(U, A, y, mu, sg):
    """(0.12) for the coarse bond from y in direction sg*e_mu; A = acv_table(U)."""
    yp = add(y, EV[mu], sg*L); Yc = tr(U, LINE[(y, mu, sg)]); Yci = inv(Yc); acc = [0j, 0j, 0j]
    for off in OFFS:
        x = add(y, off)
        v = ell(mul(mul(mul(A[(y, off)], tr(U, LINE[(x, mu, sg)])), inv(A[(yp, off)])), Yci))
        for a in range(3): acc[a] += v[a]
    return mul(E(tuple(v/L**d for v in acc)), Yc)
def average(U, kind='M', A=None):
    if A is None: A = acv_table(U, kind)
    return [avg_one(U, A, y, mu, 1) for (y, mu) in clinks]
def action(U):
    s = 0.0
    for x in sites:
        P = mul(mul(mul(U[LI[(x, 0)]], U[LI[(step(x, 0), 1)]]), inv(U[LI[(step(x, 1), 0)]])), inv(U[LI[(x, 1)]]))
        s += 2 - retr(P)
    return s
def gauge(U, g):        # U^g(b) = g(b_-) U(b) g(b_+)^{-1}, g indexed by SI
    return [mul(mul(g[SI[s]], U[k]), inv(g[SI[step(s, mu)]])) for k, (s, mu) in enumerate(links)]
def cgauge(V, h):       # coarse, h indexed by CSI
    return [mul(mul(h[CSI[y]], V[k]), inv(h[CSI[add(y, EV[mu], L)]])) for k, (y, mu) in enumerate(clinks)]

# ---------------- the point group D_4 about the coarse point 0 ----------------
D4 = [((1, 0), (0, 1)), ((0, -1), (1, 0)), ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
      ((-1, 0), (0, 1)), ((1, 0), (0, -1)), ((0, 1), (1, 0)), ((0, -1), (-1, 0))]
D4NAMES = ['id', 'rot90', 'rot180', 'rot270', 'refl1', 'refl2', 'transp', 'antitransp']
def rmap(r, x): return (r[0][0]*x[0] + r[0][1]*x[1], r[1][0]*x[0] + r[1][1]*x[1])
def act(r, U, linklist, index, stp):
    """(rU)(r b) = U(b) for oriented bonds; a bond mapped onto a reversed bond carries U(b)^{-1}."""
    out = [None]*len(linklist)
    for k, (s, mu) in enumerate(linklist):
        re = rmap(r, EV[mu]); nu = 0 if re[0] != 0 else 1; sg = re[nu]; rs = site(rmap(r, s))
        if sg == 1: out[index[(rs, nu)]] = U[k]
        else: out[index[(add(rs, EV[nu], -stp), nu)]] = inv(U[k])
    return out

# ---------------- the gauge-adapted averaging relative to U0 (Definition adapted, averaged version) ----------------
class Ref:
    """Reference U0 with V = W_1 = average(U0). taukind: contour variables used in the gauge (tau_X, g_X, Q')."""
    def __init__(self, U0, taukind='M', avgkind='M'):
        self.U0 = U0; self.tk = taukind; self.ak = avgkind
        self.W0 = acv_table(U0, taukind)                  # W_0(y,x)
        self.V = average(U0, avgkind)
        self.Vinv = [inv(v) for v in self.V]
    def config(self, A):
        return [mul(E((A[3*k], A[3*k+1], A[3*k+2])), self.U0[k]) for k in range(len(links))]
    def tau(self, U, At=None):
        if At is None: At = acv_table(U, self.tk)
        t = {}
        for y in csites:
            acc = [0j, 0j, 0j]
            for off in OFFS:
                v = ell(mul(At[(y, off)], inv(self.W0[(y, off)])))
                for a in range(3): acc[a] += v[a]
            t[y] = E(tuple(v/L**d for v in acc))
        return t
    def calQ_group(self, U):      # tilde a(X)(c) = tau(c_-)^{-1} Xbar(c) R(V(c)) tau(c_+), Xbar = average(U) V^{-1}
        At = acv_table(U, self.tk); t = self.tau(U, At)
        Ub = average(U, self.ak, A=At if self.tk == self.ak else None); out = []
        for ci, (y, mu) in enumerate(clinks):
            yp = add(y, EV[mu], L)
            out.append(mul(mul(mul(inv(t[y]), Ub[ci]), t[yp]), self.Vinv[ci]))
        return out
    def calQ(self, A):
        out = []
        for q in self.calQ_group(self.config(A)): out.extend(ell(q))
        return out
    def axial_g(self, U, h):
        """Lemma axial(a): g(x) = R(W(y,x))^{-1}(h(y) X_W(y,x)), X_W(y,x) = U(y,x) W(y,x)^{-1}."""
        At = acv_table(U, self.tk); g = [None]*len(sites)
        for y in csites:
            for off in OFFS:
                w = self.W0[(y, off)]
                g[SI[add(y, off)]] = mul(mul(inv(w), mul(h[CSI[y]], mul(At[(y, off)], inv(w)))), w)
        return g
    def m(self, g):      # Balaban's average of gauge transformations: m_1(g)(y) = g(y) E(L^{-d} sum_x ell(g(y)^{-1} R(W(y,x)) g(x)))
        out = []
        for y in csites:
            acc = [0j, 0j, 0j]
            for off in OFFS:
                w = self.W0[(y, off)]
                v = ell(mul(mul(mul(inv(g[SI[y]]), w), g[SI[add(y, off)]]), inv(w)))
                for a in range(3): acc[a] += v[a]
            out.append(mul(g[SI[y]], E(tuple(v/L**d for v in acc))))
        return out
    def E1(self, h):     # extension by transport: E_1(h)(x) = R(W(y,x))^{-1} h(y)
        g = [None]*len(sites)
        for y in csites:
            for off in OFFS:
                w = self.W0[(y, off)]; g[SI[add(y, off)]] = mul(mul(inv(w), h[CSI[y]]), w)
        return g
    def Qprime(self, normalized=True):   # (Q'lambda)(y) = L^{-d} sum_x Ad(W(y,x)) lambda(x); rows orthonormal if normalized
        f = L**(-d/2) if normalized else L**(-d)
        Q = [[0.0]*(3*len(sites)) for _ in range(3*len(csites))]
        for yi, y in enumerate(csites):
            for off in OFFS:
                Rt = adj(self.W0[(y, off)]); x = add(y, off)
                for a in range(3):
                    for b in range(3): Q[3*yi+a][3*SI[x]+b] += complex(Rt[a][b]).real*f
        return Q
    def Dfine(self):     # (D lambda)(b) = Ad(U0(b)) lambda(b_+) - lambda(b_-)
        D = [[0.0]*(3*len(sites)) for _ in range(3*len(links))]
        for k, (s, mu) in enumerate(links):
            Rb = adj(self.U0[k]); t = SI[step(s, mu)]
            for a in range(3):
                D[3*k+a][3*SI[s]+a] -= 1.0
                for b in range(3): D[3*k+a][3*t+b] += complex(Rb[a][b]).real
        return D
    def Dcoarse(self):
        D = [[0.0]*(3*len(csites)) for _ in range(3*len(clinks))]
        for k, (y, mu) in enumerate(clinks):
            Rb = adj(self.V[k]); t = CSI[add(y, EV[mu], L)]
            for a in range(3):
                D[3*k+a][3*CSI[y]+a] -= 1.0
                for b in range(3): D[3*k+a][3*t+b] += complex(Rb[a][b]).real
        return D

# ---------------- generic derivatives and dense linear algebra (as in chart.py) ----------------
def grad_cs(f, n):
    g = []
    for i in range(n):
        x = [0j]*n; x[i] = 1j*HC; g.append(f(x).imag/HC)
    return g
def jac_cs(F, n):
    cols = []
    for i in range(n):
        x = [0j]*n; x[i] = 1j*HC; cols.append([v.imag/HC for v in F(x)])
    return [list(r) for r in zip(*cols)]
def hess_cs(f, n):
    H = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            xp = [0j]*n; xm = [0j]*n
            xp[i] += 1j*HC; xm[i] += 1j*HC; xp[j] += DLT; xm[j] -= DLT
            H[i][j] = H[j][i] = (f(xp).imag - f(xm).imag)/(2*HC*DLT)
    return H
def T(M): return [list(r) for r in zip(*M)]
def mm(A, B):
    Bt = T(B); return [[sum(a*b for a, b in zip(r, c)) for c in Bt] for r in A]
def mv(A, v): return [sum(a*b for a, b in zip(r, v)) for r in A]
def mdiff(A, B): return max(abs(a - b) for r, s in zip(A, B) for a, b in zip(r, s))
def mmax(A): return max(abs(a) for r in A for a in r)
def logdet(M):
    A = [r[:] for r in M]; N = len(A); s = 0.0; sign = 1
    for i in range(N):
        p = max(range(i, N), key=lambda r: abs(A[r][i]))
        if p != i: A[i], A[p] = A[p], A[i]; sign = -sign
        if A[i][i] < 0: sign = -sign
        s += math.log(abs(A[i][i]))
        for r in range(i+1, N):
            f = A[r][i]/A[i][i]
            if f: A[r] = [u - f*v for u, v in zip(A[r], A[i])]
    return s, sign
def solve(M, B):
    vec = not isinstance(B[0], list)
    Bm = [[b] for b in B] if vec else B
    N = len(M); A = [M[i][:] + Bm[i][:] for i in range(N)]
    for i in range(N):
        p = max(range(i, N), key=lambda r: abs(A[r][i])); A[i], A[p] = A[p], A[i]
        piv = A[i][i]; A[i] = [v/piv for v in A[i]]
        for r in range(N):
            if r != i and A[r][i]:
                f = A[r][i]; A[r] = [u - f*v for u, v in zip(A[r], A[i])]
    X = [r[N:] for r in A]
    return [x[0] for x in X] if vec else X
def onb(vectors, tol=1e-9):
    basis = []
    for v in vectors:
        w = v[:]
        for _ in range(2):
            for b in basis:
                c = sum(s*t for s, t in zip(w, b)); w = [s - c*t for s, t in zip(w, b)]
        nv = math.sqrt(sum(t*t for t in w))
        if nv > tol: basis.append([t/nv for t in w])
    return basis
def nullspace(J, n):
    rows = onb(J); comp = []
    for i in range(n):
        e = [0.0]*n; e[i] = 1.0
        for r in rows:
            c = r[i]; e = [s - c*t for s, t in zip(e, r)]
        comp.append(e)
    return onb(comp)

# ======================================================================================================
# Part A: algebraic checks
# ======================================================================================================
def rand_config(amp): return [rand_su2(amp) for _ in links]
def cfg_dist(U1, U2): return max(dist(a, b) for a, b in zip(U1, U2))

def checks(seed):
    random.seed(seed)
    print('d=2, SU(2), L=%d, centred blocks, coarse torus %dx%d, fine %dx%d; seed %d' % (L, NC, NC, NF, NF, seed))
    print('family sizes: %s' % sorted(set(len(v) for v in FAMS.values())))
    print('Informative checks: A1, A3 (point group), A4 (e)-(f), A5, A6(b), A7, A8. The lines marked [construction] hold\n'
          'algebraically once the transformation law of A2 holds; they test the implementation only.')

    print('\nA1. The group average M (0.5)-(0.10), families of n elements')
    for n in (2, 3, 5, 24):
        base = rand_su2(); Us = [mul(rand_su2(0.3), base) for _ in range(n)]
        M = gavg_iter(Us); res = max(abs(z) for z in [sum(ell(mul(u, inv(M)))[a] for u in Us) for a in range(3)])
        u, v = rand_su2(), rand_su2()
        e_inv = dist(gavg_iter([inv(x) for x in Us]), inv(M))
        e_eq = dist(gavg_iter([mul(mul(u, x), v) for x in Us]), mul(mul(u, M), v))
        Up = Us[:]; random.shuffle(Up); e_perm = dist(gavg_iter(Up), M)
        e_mid = dist(gavg(Us), M) if n == 2 else float('nan')
        As = [[random.gauss(0, 1) for _ in range(3)] for _ in range(n)]       # derivative at a coincident family
        def Mt(t, base=base, As=As): return ell(mul(gavg_iter([mul(E(tuple(t*a for a in A)), base) for A in As]), inv(base)))
        der = [z.imag/HC for z in Mt(1j*HC)]; mean = [sum(A[a] for A in As)/n for a in range(3)]
        e_lin = max(abs(p - q) for p, q in zip(der, mean))
        print('  n=%2d: equation %.1e | inverse %.1e | two-sided equivariance (Haar u,v) %.1e | permutation %.1e | '
              'midpoint %.1e | derivative at coincident family = mean %.1e' % (n, res, e_inv, e_eq, e_perm, e_mid, e_lin))
    print('  weights of the derivative at a family scaled by t, max_i |W_i - I/n| (sum_i W_i = I):')
    n = 5; base = rand_su2(); As = [[random.gauss(0, 0.5) for _ in range(3)] for _ in range(n)]
    for t in (0.8, 0.4, 0.2, 0.1, 0.05, 0.025):
        Us = [mul(E(tuple(t*a for a in A)), base) for A in As]; M0 = gavg_iter(Us); Ws = []
        for i in range(n):
            cols = []
            for b in range(3):
                e = [0j, 0j, 0j]; e[b] = 1j*HC
                Ui = Us[:]; Ui[i] = mul(E(tuple(e)), Us[i])
                cols.append([z.imag/HC for z in ell(mul(gavg_iter(Ui), inv(M0)))])
            Ws.append(T(cols))
        S = [[sum(W[a][b] for W in Ws) for b in range(3)] for a in range(3)]
        dev = max(abs(W[a][b] - (1.0/n if a == b else 0.0)) for W in Ws for a in range(3) for b in range(3))
        sz = max(dist(p, q) for p in Us for q in Us)
        print('    t=%.3f, diameter %.4f: max |W_i - I/n| = %.2e (ratio to diameter: %.4f), |sum W_i - I| = %.1e'
              % (t, sz, dev, dev/sz, max(abs(S[a][b] - (a == b)) for a in range(3) for b in range(3))))

    U = rand_config(0.3); g = [rand_su2() for _ in sites]; Ug = gauge(U, g); h = [g[SI[y]] for y in csites]
    print('\nA2. Gauge covariance under an arbitrary (Haar-random) gauge transformation g')
    for kind in ('M', 'logmean', 'tree'):
        A, Ag = acv_table(U, kind), acv_table(Ug, kind)
        e = max(dist(Ag[(y, off)], mul(mul(g[SI[y]], A[(y, off)]), inv(g[SI[add(y, off)]]))) for (y, off) in FAMS)
        print('  contour variables %-8s: max |U^g(y,x) - g(y) U(y,x) g(x)^{-1}| = %.1e' % (kind, e))
    for kind in ('M', 'tree'):
        print('  averaging (0.12) with %-5s contour variables: max |a(U^g) - a(U)^{g|T}| = %.1e'
              % (kind, cfg_dist(average(Ug, kind), cgauge(average(U, kind), h))))
    print('  ("logmean" is the shortcut that fails; "tree" is covariant but not symmetric)')

    print('\nA3. Point-group and reversal covariance of (0.12); U random, amplitude 0.3')
    A = acv_table(U, 'M')
    e_rev = max(dist(avg_one(U, A, add(y, EV[mu], L), mu, -1), inv(avg_one(U, A, y, mu, 1))) for (y, mu) in clinks)
    print('  [construction] reversal: max |a(U)(-c) - a(U)(c)^{-1}| = %.1e' % e_rev)
    for kind in ('M', 'tree'):
        aU = average(U, kind); row = []
        for r in D4:
            rU = act(r, U, links, LI, 1)
            row.append(cfg_dist(average(rU, kind), act(r, aU, clinks, CLI, L)))
        print('  %-4s families: %s' % ('all' if kind == 'M' else 'one', '  '.join('%s %.1e' % (nm, e) for nm, e in zip(D4NAMES, row))))
    U0 = rand_config(0.3); X = rand_config(0.3); Ux = [mul(x, u) for x, u in zip(X, U0)]
    ref = Ref(U0); QX = [mul(q, v) for q, v in zip(ref.calQ_group(Ux), ref.V)]
    row = []
    for r in D4:
        rref = Ref(act(r, U0, links, LI, 1))
        rQX = [mul(q, v) for q, v in zip(rref.calQ_group(act(r, Ux, links, LI, 1)), rref.V)]
        row.append(cfg_dist(rQX, act(r, QX, clinks, CLI, L)))
    print('  gauge-adapted average Q relative to U0: max over D4 of |Q_{rU0}(rX) - r Q_{U0}(X)| = %.1e' % max(row))

    print('\nA4. The averaged axial gauge and Theorem correspondence (k=1); U0, X random, amplitude 0.3')
    hh = [rand_su2() for _ in csites]
    gx = ref.axial_g(Ux, hh); Uax = gauge(Ux, gx); Aax = acv_table(Uax)
    e_ax = max(dist(Aax[key], ref.W0[key]) for key in FAMS)
    e_h = max(dist(gx[SI[y]], hh[CSI[y]]) for y in csites)
    print('  [construction] g of Lemma axial(a), arbitrary h: max_(y,x) |X^(g)_W(y,x) - 1| = %.1e,  |g|_T - h| = %.1e' % (e_ax, e_h))
    u = [ONE if s in CSI else rand_su2() for s in sites]; Uu = gauge(Ux, u)
    one = [ONE]*len(csites)
    rep1 = gauge(Ux, ref.axial_g(Ux, one)); rep2 = gauge(Uu, ref.axial_g(Uu, one))
    print('  [construction] U and U^u, u in G_0 Haar-random: distance of their axial representatives (h=1) = %.1e' % cfg_dist(rep1, rep2))
    t = ref.tau(Ux); gX = ref.axial_g(Ux, [inv(t[y]) for y in csites])
    lhs_ = [mul(a, b) for a, b in zip(average(gauge(Ux, gX)), ref.Vinv)]
    print('  [construction] Definition adapted: |avg(X^(g_X)) - tau_X(c-)^{-1} Xbar R(V)(tau_X(c+))| = %.1e' % cfg_dist(lhs_, ref.calQ_group(Ux)))
    tax = ref.tau(rep1); Xbar = [mul(a, b) for a, b in zip(average(rep1), ref.Vinv)]
    print('  [construction] Theorem correspondence(d): on the axial slice tau = 1 (%.1e) and Q = Xbar (%.1e)'
          % (max(dist(v, ONE) for v in tax.values()), cfg_dist(ref.calQ_group(rep1), Xbar)))
    print('  Theorem correspondence (e), (f), nonlinear; g random near 1 (amplitude 0.3), h Haar-random:')
    for tk in ('M', 'tree', 'logmean'):
        rf = Ref(U0, taukind=tk)
        g = [rand_su2(0.3) for _ in sites]; Ug = gauge(U0, g); mg = rf.m(g)
        e1 = cfg_dist(rf.calQ_group(Ug), [mul(a, b) for a, b in zip(cgauge(rf.V, mg), rf.Vinv)])
        tg = rf.tau(Ug); wg = rf.axial_g(Ug, [inv(tg[y]) for y in csites]); Em = rf.E1(mg)
        e2 = max(dist(wg[i], mul(Em[i], inv(g[i]))) for i in range(len(sites)))
        h2 = [rand_su2() for _ in csites]
        e3 = max(dist(a, mul(b, c)) for a, b, c in zip(rf.m([mul(a, b) for a, b in zip(rf.E1(h2), g)]), h2, mg))
        tx = rf.tau(Ux); e4 = max(dist(v, ONE) for v in rf.m(rf.axial_g(Ux, [inv(tx[y]) for y in csites])))
        print('    %-8s contour variables: (e) |Q(1^(g)) - 1^(m(g))| = %.1e, |w_1(1^(g)) - E_1(m(g)) g^{-1}| = %.1e;'
              '  (f) |m(E_1(h)g) - h m(g)| = %.1e, |m(w_1(X))| = %.1e' % (tk, e1, e2, e3, e4))

    print('\nA5. Linearization at U = 1 (scalar coefficient, colour-diagonal)')
    ref1 = Ref([ONE]*len(links)); n = 3*len(links)
    Jplain = jac_cs(lambda A: [z for v in average(ref1.config(A)) for z in ell(v)], n)
    JQ = jac_cs(ref1.calQ, n)
    JQt = jac_cs(Ref([ONE]*len(links), taukind='tree').calQ, n)
    Tm = [[0.0]*n for _ in range(3*len(clinks))]       # centred tube average
    Gs = [[0.0]*n for _ in range(3*len(csites))]       # gamma_sym(y) = L^{-d} sum_x (mean over G(y,x) of B(Gamma))
    Gt = [[0.0]*n for _ in range(3*len(csites))]       # gamma_tree
    for ci, (y, mu) in enumerate(clinks):
        for off in OFFS:
            for (k, sg) in LINE[(add(y, off), mu, 1)]:
                for a in range(3): Tm[3*ci+a][3*k+a] += sg/L**d
    for yi, y in enumerate(csites):
        for off in OFFS:
            for fam, G in ((FAMS[(y, off)], Gs), (FAM1[(y, off)], Gt)):
                for p in fam:
                    for (k, sg) in p:
                        for a in range(3): G[3*yi+a][3*k+a] += sg/(L**d*len(fam))
    def cgrad(G):      # (D'f)(c) = f(c_+) - f(c_-), the note's D_W at W = 1
        out = []
        for (y, mu) in clinks:
            yp = add(y, EV[mu], L)
            for a in range(3): out.append([p - q for p, q in zip(G[3*CSI[yp]+a], G[3*CSI[y]+a])])
        return out
    gs, gt = cgrad(Gs), cgrad(Gt)
    print('  gauge-adapted Q (averaged contour variables) = centred tube average: %.1e' % mdiff(JQ, Tm))
    print('  plain (0.12) = tube average - D\'gamma_sym: %.1e   (|D\'gamma_sym| = %.3f)'
          % (mdiff(Jplain, [[p - q for p, q in zip(r, s)] for r, s in zip(Tm, gs)]), mmax(gs)))
    dts = [[p - q for p, q in zip(r, s)] for r, s in zip(gt, gs)]
    print('  gauge-adapted Q with the tree contour in the gauge = tube average + D\'(gamma_tree - gamma_sym): %.1e;'
          '  distance from the tube average %.3f' % (mdiff(JQt, [[p + q for p, q in zip(r, s)] for r, s in zip(Tm, dts)]), mdiff(JQt, Tm)))
    b0 = [LI[(add(y, EV[mu], HW), mu)] for (y, mu) in clinks]      # middle bond b_0(c) of [c]
    coef = [Jplain[3*ci][3*k] for ci, k in enumerate(b0)]
    others = max(abs(Jplain[3*cj][3*k]) for ci, k in enumerate(b0) for cj in range(len(clinks)) if cj != ci)
    print('  coefficient of Z(b_0(c)) in the plain constraint: %s (L^{1-d} = %.6f); in other coarse bonds: %.1e'
          % (sorted(set(round(c, 12) for c in coef)), L**(1 - d), others))
    composition_check()

    print('\nA6. Linearized averages at a random real U0 (amplitude 0.3): Proposition linearized-averages')
    for tk, ak, label in (('M', 'M', "Balaban's scheme"), ('tree', 'M', 'tree contour in the gauge'), ('logmean', 'logmean', 'log-mean contour variables')):
        rf = Ref(U0, taukind=tk, avgkind=ak)
        Qn = rf.Qprime(); J = jac_cs(rf.calQ, 3*len(links))
        D, DV, Qu = rf.Dfine(), rf.Dcoarse(), rf.Qprime(False)
        e_orth = max(abs(v - (i == j)) for i, r in enumerate(mm(Qn, T(Qn))) for j, v in enumerate(r))
        e_cov = mdiff(mm(J, D), mm(DV, Qu))
        print('  %-27s: [construction] (a) |Q\'Q\'^T - I| = %.1e   (b) |Q D - D_V Q\'| = %.1e   (scale |Q D| = %.2f)' % (label, e_orth, e_cov, mmax(mm(J, D))))
    u = [rand_su2() for _ in sites]; ru = Ref(gauge(U0, u)); r0 = Ref(U0)
    def Rfine(u):
        R = [[0.0]*(3*len(links)) for _ in range(3*len(links))]
        for k, (s, mu) in enumerate(links):
            A_ = adj(u[SI[s]])
            for a in range(3):
                for b in range(3): R[3*k+a][3*k+b] = complex(A_[a][b]).real
        return R
    def Rsite(v, pts, idx):
        R = [[0.0]*(3*len(pts)) for _ in range(3*len(pts))]
        for p in pts:
            A_ = adj(v[idx(p)])
            for a in range(3):
                for b in range(3): R[3*idx(p)+a][3*idx(p)+b] = complex(A_[a][b]).real
        return R
    def Rcoarse(h):
        R = [[0.0]*(3*len(clinks)) for _ in range(3*len(clinks))]
        for k, (y, mu) in enumerate(clinks):
            A_ = adj(h[CSI[y]])
            for a in range(3):
                for b in range(3): R[3*k+a][3*k+b] = complex(A_[a][b]).real
        return R
    hu = [u[SI[y]] for y in csites]
    e_wk = cfg_dist(ru.V, cgauge(r0.V, hu))
    e_qp = mdiff(mm(ru.Qprime(), Rsite(u, sites, lambda p: SI[p])), mm(Rsite(hu, csites, lambda p: CSI[p]), r0.Qprime()))
    e_q = mdiff(mm(jac_cs(ru.calQ, 3*len(links)), Rfine(u)), mm(Rcoarse(hu), jac_cs(r0.calQ, 3*len(links))))
    print('  (f) Haar-random u: |W_1(U0^u) - W_1(U0)^u| = %.1e, |Q\'(U0^u)R(u) - R(u|T)Q\'(U0)| = %.1e, |Q(U0^u)R(u) - R(u|T)Q(U0)| = %.1e'
          % (e_wk, e_qp, e_q))

    if L == 3: tree_check()

    print('\nA8. The pipeline with the iterative solver of (0.10) also for two-element families')
    aU = average(U); su2.FORCE_ITER = True
    e_sol = cfg_dist(average(U), aU)
    e_d4 = max(cfg_dist(average(act(r, U, links, LI, 1)), act(r, average(U), clinks, CLI, L)) for r in D4)
    rf = Ref(U0); J = jac_cs(rf.calQ, 3*len(links)); e_cov = mdiff(mm(J, rf.Dfine()), mm(rf.Dcoarse(), rf.Qprime(False)))
    su2.FORCE_ITER = False
    print('  |a(U) iterative - a(U) midpoint| = %.1e;  D4 covariance %.1e;  E=0 by complex steps through the solver %.1e' % (e_sol, e_d4, e_cov))

def tree_check():
    """Claim Y2-tree for L=3 by enumeration: the 8-bond subsets of the 12 bonds of a 3x3 block; spanning trees,
    transposition-invariant ones, and those on which the linearized constraint G of the averaged contour variables
    (at U=1) is invertible."""
    from fractions import Fraction
    y = csites[0]; pts = block(y)
    bonds = [LI[(s, mu)] for s in pts for mu in range(d) if step(s, mu) in pts]
    rows = []
    for off in OFFS:
        if off == (0, 0): continue
        fam = FAMS[(y, off)]; row = {b: Fraction(0) for b in bonds}
        for p in fam:
            for (k, sg) in p: row[k] += Fraction(sg, len(fam))
        rows.append(row)
    ends = {LI[(s, mu)]: (s, step(s, mu)) for s in pts for mu in range(d)}
    tmap = {}
    for b in bonds:
        s, t = ends[b]; s2, t2 = site((s[1], s[0])), site((t[1], t[0]))   # transposition about y = 0
        tmap[b] = LI[(s2, 0 if t2 == step(s2, 0) else 1)]
    def rank(M):
        M = [r[:] for r in M]; rk = 0; ncol = len(M[0])
        for c in range(ncol):
            p = next((i for i in range(rk, len(M)) if M[i][c] != 0), None)
            if p is None: continue
            M[rk], M[p] = M[p], M[rk]
            for i in range(len(M)):
                if i != rk and M[i][c] != 0:
                    f = M[i][c]/M[rk][c]; M[i] = [a - f*b for a, b in zip(M[i], M[rk])]
            rk += 1
        return rk
    def is_tree(S):
        par = {p: p for p in pts}
        def f(p):
            while par[p] != p: p = par[p]
            return p
        for b in S:
            a, c = f(ends[b][0]), f(ends[b][1])
            if a == c: return False
            par[a] = c
        return True
    n_tree = n_sym = n_inv = n_inv_tree = n_sym_tree = 0
    for S in itertools.combinations(bonds, len(pts) - 1):
        tree = is_tree(S); sym = set(tmap[b] for b in S) == set(S)
        inv_ = rank([[r[b] for b in S] for r in rows]) == len(S)
        n_tree += tree; n_sym += sym; n_sym_tree += tree and sym; n_inv += inv_; n_inv_tree += inv_ and tree
    print('\nA7. Claim Y2-tree (3x3 block, %d bonds, %d-bond subsets):' % (len(bonds), len(pts) - 1))
    print('  spanning trees %d, of which transposition-invariant %d; transposition-invariant subsets %d' % (n_tree, n_sym_tree, n_sym))
    print('  subsets on which the linearized constraint G is invertible: %d, of which spanning trees %d' % (n_inv, n_inv_tree))

def tube(N, Lb):
    """Scalar centred tube average from the N x N lattice to the coarse lattice of spacing Lb (U = 1)."""
    hw = (Lb - 1)//2; nc = N//Lb
    fl = {((i, j), mu): k for k, ((i, j), mu) in enumerate(((i, j), mu) for i in range(N) for j in range(N) for mu in range(2))}
    cl = [((i, j), mu) for i in range(nc) for j in range(nc) for mu in range(2)]
    M = [[0.0]*len(fl) for _ in cl]
    for ci, ((i, j), mu) in enumerate(cl):
        for a in range(-hw, hw + 1):
            for b in range(-hw, hw + 1):
                x = [Lb*i + a, Lb*j + b]
                for _ in range(Lb):
                    M[ci][fl[((x[0] % N, x[1] % N), mu)]] += Lb**(-2); x[mu] += 1
    return M
def blockavg(N, Lb):
    hw = (Lb - 1)//2; nc = N//Lb
    M = [[0.0]*(N*N) for _ in range(nc*nc)]
    for i in range(nc):
        for j in range(nc):
            for a in range(-hw, hw + 1):
                for b in range(-hw, hw + 1): M[i*nc + j][((Lb*i + a) % N)*N + (Lb*j + b) % N] += Lb**(-2)
    return M
def composition_check():
    N = L*L*NC
    t2 = mm(tube(L*NC, L), tube(N, L)); t22 = tube(N, L*L)
    b2 = mm(blockavg(L*NC, L), blockavg(N, L)); b22 = blockavg(N, L*L)
    print('  composition of two centred tube averages = centred tube average of side L^2: %.1e;'
          ' block averages: %.1e   (fine torus %dx%d)' % (mdiff(t2, t22), mdiff(b2, b22), N, N))

# ======================================================================================================
# Part B: Theorem ol-chart on the averaged axial slice
# ======================================================================================================
def tree_gauge(U):
    g = [ONE]*len(sites)
    for key, fam in FAM1.items(): g[SI[add(key[0], key[1])]] = tr(U, fam[0])
    return gauge(U, g)
def lhs_functions(Ucur, V, varlinks):
    Vinv = [inv(v) for v in V]
    def config(X):
        U = list(Ucur)
        for k, li in enumerate(varlinks): U[li] = mul(E((X[3*k], X[3*k+1], X[3*k+2])), Ucur[li])
        return U
    def cons(X):
        Ub = average(config(X)); out = []
        for c in range(len(clinks)): out.extend(ell(mul(Ub[c], Vinv[c])))
        return out
    def S(X): return action(config(X))
    return config, cons, S, 3*len(varlinks)
def restore(Ucur, V, tol=1e-14):
    for _ in range(20):
        config, cons, S, n = lhs_functions(Ucur, V, free_links)
        c0 = [v.real for v in cons([0j]*n)]
        if max(abs(v) for v in c0) < tol: return Ucur, True
        J = jac_cs(cons, n); y = solve(mm(J, T(J)), c0); dY = [-v for v in mv(T(J), y)]
        Ucur = [real(u) for u in config([complex(v) for v in dY])]
    return Ucur, False
def minimize_on_fibre(U, V, iters=60, verbose=True):
    Ucur, ok = restore(tree_gauge(U), V); omega = None; mu = 0.0
    for it in range(iters):
        config, cons, S, n = lhs_functions(Ucur, V, free_links)
        J = jac_cs(cons, n); g = grad_cs(S, n); m = len(J)
        if omega is None: omega = solve(mm(J, T(J)), mv(J, g))
        Lag = lambda X: S(X) - sum(w*v for w, v in zip(omega, cons(X)))
        H = hess_cs(Lag, n)
        while True:
            K = [[H[i][j] + (mu if i == j else 0.0) for j in range(n)] + [J[r][i] for r in range(m)] for i in range(n)] + [J[r] + [0.0]*m for r in range(m)]
            sol = solve(K, [-gi for gi in g] + [0.0]*m)
            dX = sol[:n]; om_new = [-v for v in sol[n:]]
            big = max(abs(v) for v in dX)
            if big > 0.3: dX = [v*0.3/big for v in dX]
            Utr, ok = restore([real(u) for u in config([complex(v) for v in dX])], V)
            S0 = S([0j]*n).real; S1 = action(Utr)
            if ok and (S1 <= S0 + 1e-13 or big < 1e-10): break
            mu = max(4*mu, 0.05)
            if mu > 1e6: raise RuntimeError('no descent')
        Ucur, omega = Utr, om_new
        if verbose: print('   it %2d: S = %.14f  |dX| = %.2e  mu = %.2g' % (it, S1, big, mu), flush=True)
        if big < 1e-11 and mu == 0.0: return Ucur
        mu = mu/8 if mu > 1e-3 else 0.0
    raise RuntimeError('minimization on the fibre did not converge')
def chol_min(M):
    """Cholesky factorization; returns the smallest pivot (positive iff M is positive definite)."""
    N = len(M); Lc = [[0.0]*N for _ in range(N)]; pmin = float('inf')
    for i in range(N):
        for j in range(i + 1):
            s = M[i][j] - sum(Lc[i][k]*Lc[j][k] for k in range(j))
            if i == j:
                if s <= 0: return s
                pmin = min(pmin, s); Lc[i][i] = math.sqrt(s)
            else: Lc[i][j] = s/Lc[j][j]
    return pmin
def laplace(U0, V, varlinks, extra=None):
    """-1/2 log det(Hess of the Lagrangian on ker J) - 1/2 log det(J J^T) over the variables on varlinks."""
    config, cons, S, n = lhs_functions(U0, V, varlinks)
    J = jac_cs(cons, n); g = grad_cs(S, n)
    JJ = mm(J, T(J)); omega = solve(JJ, mv(J, g))
    resid = max(abs(a - b) for a, b in zip(g, mv(T(J), omega)))
    if resid > 1e-10: raise RuntimeError('not a critical point: multiplier residual %.1e' % resid)
    xt = extra(config) if extra else (lambda X: 0.0)
    Lag = lambda X: S(X) - sum(w*v for w, v in zip(omega, cons(X))) + xt(X)
    H = hess_cs(Lag, n)
    B = nullspace(J, n); HB = mm(B, mm(H, T(B)))
    return -0.5*logdet(HB)[0] - 0.5*logdet(JJ)[0], resid, chol_min(HB)
def lhs_tree(U0, V): return laplace(U0, V, free_links)
def lhs_exp(U0, V):
    """Laplace over all bonds with Balaban's exponential gauge fixing relative to the base W = U0 (Claim Y2-exp):
    weight exp(-g^{-2} sum_{y, x != y} [2 - tr U(y,x) W(y,x)^{-1}]) at the coupling of the action; no Faddeev-Popov
    determinant."""
    W0 = acv_table(U0)
    def extra(config):
        def gf(X):
            A = acv_table(config(X)); return sum(2 - retr(mul(A[key], inv(W0[key]))) for key in FAMS if key[1] != (0, 0))
        return gf
    return laplace(U0, V, list(range(len(links))), extra)
def rhs(U0, taukind='M', avals=(0.5, 2.0), HPhi=None):
    rf = Ref(U0, taukind=taukind); n = 3*len(links)
    def Phi(A): return action(rf.config(A))
    Q = jac_cs(rf.calQ, n); g = grad_cs(Phi, n); m = len(Q)
    QQ = mm(Q, T(Q)); omega = solve(QQ, mv(Q, g))
    resid = max(abs(a - b) for a, b in zip(g, mv(T(Q), omega)))
    if HPhi is None: HPhi = hess_cs(Phi, n)
    Hom = hess_cs(lambda A: sum(w*v for w, v in zip(omega, rf.calQ(A))), n)
    q2 = [[HPhi[i][j] - Hom[i][j] for j in range(n)] for i in range(n)]
    D = rf.Dfine(); Qp = rf.Qprime(); ns = 3*len(sites)
    X = mm(T(D), D); N = nullspace(Qp, ns); XN = mm(X, T(N)); W = onb(T(XN))
    Rproj = mm(T(W), W); ghost = 0.5*logdet(mm(T(XN), XN))[0]; DRDt = mm(mm(D, Rproj), T(D))
    zm = mmax(mm(q2, mm(D, T(N)))); cov = mmax(mm(Q, mm(D, T(N))))
    out = []; parts = {}
    for a in avals:
        for tag, qq in (('', q2), ('nomult', HPhi)):
            Ls = [[qq[i][j] + DRDt[i][j] + a*sum(Q[r][i]*Q[r][j] for r in range(m)) for j in range(n)] for i in range(n)]
            ldL, sL = logdet(Ls); ldM, sM = logdet(mm(Q, solve(Ls, T(Q))))
            if tag == '': out.append((a, -0.5*ldL - 0.5*ldM + ghost, sL, sM))
            parts[(tag, a)] = (-0.5*ldL, -0.5*ldM, ghost)
    return out, resid, zm, cov, parts, HPhi

def chart(seed, amp, nV):
    random.seed(seed)
    nlam = 3*(len(sites) - len(csites))
    pred = -0.5*d*3*len(csites)*math.log(L)       # LHS - RHS = -(d/2) dim g |T| log L (normalization of the coarse delta)
    print('d=2, SU(2), L=%d, centred blocks, coarse torus %dx%d, fine %dx%d; free links %d of %d; seed %d, amplitude %g'
          % (L, NC, NC, NF, NF, len(free_links), len(links), seed, amp), flush=True)
    print('predictions: LHS_tree - RHS = -(d/2) dim(g) |T| log L = %.10f;  LHS_exp - LHS_tree = (n_lambda/2) log 2 = %.10f, n_lambda = %d'
          % (pred, 0.5*nlam*math.log(2), nlam), flush=True)
    for iv in range(nV):
        t0 = time.time()
        Uin = rand_config(amp); V = average(Uin)
        U0 = minimize_on_fibre(Uin, V)
        print('  V%d: S(U0) = %.10f, constraint residual %.1e' % (iv, action(U0).real, cfg_dist(average(U0), V)), flush=True)
        l, res_l, pm = lhs_tree(U0, V)
        print('  V%d: LHS_tree = %.10f   (multiplier residual %.1e, smallest Cholesky pivot %.3e)' % (iv, l, res_l, pm), flush=True)
        le, res_e, pme = lhs_exp(U0, V)
        print('  V%d: LHS_exp  = %.10f   (multiplier residual %.1e, smallest Cholesky pivot %.3e);  LHS_exp - LHS_tree - (n_lambda/2) log 2 = %.1e'
              % (iv, le, res_e, pme, le - l - 0.5*nlam*math.log(2)), flush=True)
        r, res_r, zm, cov, parts, HPhi = rhs(U0)
        pL, pM, pg = parts[('', 0.5)]; nL, nM, _ = parts[('nomult', 0.5)]
        print('  V%d: PARTS %.10f %.10f %.10f %.10f %.10f %.10f' % (iv, l, pL, pM, pg, nL, nM), flush=True)
        print('  V%d: RHS = %s   (EL residual %.1e, |q2 D N(Q\')| = %.1e, |Q D N(Q\')| = %.1e)' %
              (iv, '  '.join('a=%.1f: %.10f (signs %d,%d)' % x for x in r), res_r, zm, cov), flush=True)
        print('  V%d: LHS_tree - RHS = %.10f;  LHS_tree - RHS - prediction = %.1e' % (iv, l - r[0][1], l - r[0][1] - pred), flush=True)
        rt, res_t, zmt, covt, _, _ = rhs(U0, taukind='tree', avals=(0.5,), HPhi=HPhi)
        print('  V%d: chart independence, tree contour in the gauge: RHS = %.10f (|Q D N| = %.1e);  LHS_tree - RHS - prediction = %.1e'
              % (iv, rt[0][1], covt, l - rt[0][1] - pred), flush=True)
        rl, res_m, zml, covl, _, _ = rhs(U0, taukind='logmean', avals=(0.5,), HPhi=HPhi)
        print('  V%d: negative control, log-mean contour variables in the gauge (E != 0): |Q D N| = %.1e;  LHS_tree - RHS - prediction = %.1e'
              % (iv, covl, l - rl[0][1] - pred), flush=True)
        for ri in (1, 4, 6):
            r_ = D4[ri]; rU = tree_gauge(act(r_, U0, links, LI, 1)); rV = act(r_, V, clinks, CLI, L)
            lr, _, _ = lhs_tree(rU, rV); rr = rhs(rU, avals=(0.5,))[0][0][1]
            print('  V%d: point group, %-6s: constraint residual %.1e;  LHS_tree(rV) - LHS_tree(V) = %.1e,  RHS(rU0) - RHS(U0) = %.1e'
                  % (iv, D4NAMES[ri], cfg_dist(average(rU), rV), lr - l, rr - r[0][1]), flush=True)
        print('  V%d: [%.0f s]' % (iv, time.time() - t0), flush=True)

if __name__ == '__main__':
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    if MODE == 'checks': checks(seed)
    else: chart(seed, float(sys.argv[5]) if len(sys.argv) > 5 else 0.25, int(sys.argv[6]) if len(sys.argv) > 6 else 1)
