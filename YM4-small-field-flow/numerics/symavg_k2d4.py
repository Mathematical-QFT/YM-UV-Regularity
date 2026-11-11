# Further checks of Balaban's symmetric averaging (YM4 blueprint, Section Y2), complementing symavg.py.
#   python3 symavg_k2d4.py [seed]
# Part C (d=2, SU(2), L=3): an independent implementation on tori of any size, compared with symavg.py at k=1;
#   exact covariance E=0 and Theorem correspondence(e) at k=2 (fine torus 18x18); the linearization at U=1 at k=2;
#   the Poincare form |dB|^2 + |G B|^2 + |q B|^2 of the exponential gauge at U=1 (Y2.1, "Delta^exp") on growing tori.
# Part D (d=4, SU(2), L=3): one coarse bond on the infinite lattice (link variables drawn on demand); families of up to
#   24 contours, so the group average is computed by the iterative solver of (0.10), also inside complex-step derivatives.
# Conventions as in symavg.py.
import itertools, math, random, sys, time
from su2 import mul, inv, ONE, E, ell, adv, dist, rand_su2, gavg

HC = 1e-20
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1

def vsub(a, b): return tuple(x - y for x, y in zip(a, b))
def vmax(vs): return max(abs(z) for v in vs for z in v)
def cs(v): return tuple(z.imag/HC for z in v)

# ======================================================================================================
# Part C: d = 2 on an N x N torus with centred blocks of side L
# ======================================================================================================
class Scale:
    def __init__(self, N, L):
        self.N, self.L, self.hw, self.nc = N, L, (L - 1)//2, N//L
        self.sites = [(i, j) for i in range(N) for j in range(N)]; self.SI = {s: n for n, s in enumerate(self.sites)}
        self.links = [(s, mu) for s in self.sites for mu in range(2)]; self.LI = {l: n for n, l in enumerate(self.links)}
        self.csites = [(L*i, L*j) for i in range(self.nc) for j in range(self.nc)]; self.CSI = {y: n for n, y in enumerate(self.csites)}
        self.clinks = [(y, mu) for y in self.csites for mu in range(2)]     # same order as the links of Scale(N//L, L)
        self.offs = [(a, b) for a in range(-self.hw, self.hw + 1) for b in range(-self.hw, self.hw + 1)]
        self.fams = {}
        for y in self.csites:
            for off in self.offs:
                nz = [mu for mu in range(2) if off[mu] != 0]
                self.fams[(y, off)] = [self.path(y, [(mu, off[mu]) for mu in o]) for o in itertools.permutations(nz)]
        self.line = {(x, mu): self.path(x, [(mu, L)]) for x in self.sites for mu in range(2)}
    def site(self, x): return (x[0] % self.N, x[1] % self.N)
    def add(self, x, mu, n): e = [0, 0]; e[mu] = n; return self.site((x[0] + e[0], x[1] + e[1]))
    def path(self, x, moves):   # list of (link index, sgn)
        p = []; cur = x
        for mu, n in moves:
            sg = 1 if n > 0 else -1
            for _ in range(abs(n)):
                if sg == 1: p.append((self.LI[(cur, mu)], 1)); cur = self.add(cur, mu, 1)
                else: cur = self.add(cur, mu, -1); p.append((self.LI[(cur, mu)], -1))
        return p
    def tr(self, U, p):
        m = ONE
        for k, sg in p: m = mul(m, U[k] if sg == 1 else inv(U[k]))
        return m
    def acv(self, U): return {key: gavg([self.tr(U, p) for p in fam]) for key, fam in self.fams.items()}
    def avg(self, U, A=None):
        if A is None: A = self.acv(U)
        out = []
        for (y, mu) in self.clinks:
            yp = self.add(y, mu, self.L); Yc = self.tr(U, self.line[(y, mu)]); Yci = inv(Yc); acc = [0j, 0j, 0j]
            for off in self.offs:
                x = self.site((y[0] + off[0], y[1] + off[1]))
                v = ell(mul(mul(mul(A[(y, off)], self.tr(U, self.line[(x, mu)])), inv(A[(yp, off)])), Yci))
                for a in range(3): acc[a] += v[a]
            out.append(mul(E(tuple(v/self.L**2 for v in acc)), Yc))
        return out
    def tau(self, A, AW):
        t = {}
        for y in self.csites:
            acc = [0j, 0j, 0j]
            for off in self.offs:
                v = ell(mul(A[(y, off)], inv(AW[(y, off)])))
                for a in range(3): acc[a] += v[a]
            t[y] = E(tuple(v/self.L**2 for v in acc))
        return t
    def atilde(self, U, AW):    # the modified average, as a full configuration: tau(c_-)^{-1} a(U)(c) tau(c_+)
        A = self.acv(U); t = self.tau(A, AW); Ub = self.avg(U, A)
        return [mul(mul(inv(t[y]), Ub[k]), t[self.add(y, mu, self.L)]) for k, (y, mu) in enumerate(self.clinks)]
    def m(self, g, AW):         # Balaban's average of gauge transformations, g indexed by SI; result indexed by CSI
        out = []
        for y in self.csites:
            gy = g[self.SI[y]]; acc = [0j, 0j, 0j]
            for off in self.offs:
                w = AW[(y, off)]; x = self.site((y[0] + off[0], y[1] + off[1]))
                v = ell(mul(mul(mul(inv(gy), w), g[self.SI[x]]), inv(w)))
                for a in range(3): acc[a] += v[a]
            out.append(mul(gy, E(tuple(v/self.L**2 for v in acc))))
        return out
    def Qp(self, lam, AW):      # (Q'lambda)(y) = L^{-d} sum_x Ad(W(y,x)) lambda(x)
        out = []
        for y in self.csites:
            acc = [0.0, 0.0, 0.0]
            for off in self.offs:
                v = adv(AW[(y, off)], lam[self.SI[self.site((y[0] + off[0], y[1] + off[1]))]])
                acc = [a + b for a, b in zip(acc, v)]
            out.append(tuple(a/self.L**2 for a in acc))
        return out
    def D(self, U, lam):        # (D lambda)(b) = Ad(U(b)) lambda(b_+) - lambda(b_-)
        return [vsub(adv(U[k], lam[self.SI[self.add(s, mu, 1)]]), lam[self.SI[s]]) for k, (s, mu) in enumerate(self.links)]
    def Dc(self, V, h):         # the same on the coarse lattice, h indexed by CSI
        return [vsub(adv(V[k], h[self.CSI[self.add(y, mu, self.L)]]), h[self.CSI[y]]) for k, (y, mu) in enumerate(self.clinks)]
    def gauge(self, U, g): return [mul(mul(g[self.SI[s]], U[k]), inv(g[self.SI[self.add(s, mu, 1)]])) for k, (s, mu) in enumerate(self.links)]
    def cgauge(self, V, h): return [mul(mul(h[self.CSI[y]], V[k]), inv(h[self.CSI[self.add(y, mu, self.L)]])) for k, (y, mu) in enumerate(self.clinks)]

def part_c():
    print('Part C: d=2, SU(2), L=3')
    # C1. the independent implementation against symavg.py (k=1, 6x6)
    argv = sys.argv; sys.argv = [argv[0]]
    import symavg as sa
    sys.argv = argv
    S = Scale(6, 3); U0 = [rand_su2(0.3) for _ in S.links]; X = [rand_su2(0.3) for _ in S.links]; U = [mul(x, u) for x, u in zip(X, U0)]
    AW = S.acv(U0); V = S.avg(U0, AW)
    e1 = max(dist(a, b) for a, b in zip(S.avg(U), sa.average(U)))
    e2 = max(dist(mul(a, inv(v)), b) for a, b, v in zip(S.atilde(U, AW), sa.Ref(U0).calQ_group(U), V))
    print('  C1. independent implementation vs symavg.py (6x6, k=1): averaging %.1e, gauge-adapted average %.1e' % (e1, e2))

    # C2. k = 2 on the 18x18 torus: E = 0 and Theorem correspondence(e)
    S1, S2 = Scale(18, 3), Scale(6, 3)
    U0 = [rand_su2(0.3) for _ in S1.links]
    AW0 = S1.acv(U0); W1 = S1.avg(U0, AW0); AW1 = S2.acv(W1); W2 = S2.avg(W1, AW1)
    def Q2(U): return S2.atilde(S1.atilde(U, AW0), AW1)      # full configuration; Q_2 = Q2 W_2^{-1}
    errs, scale = [], []
    for _ in range(3):
        lam = [tuple(random.gauss(0, 1) for _ in range(3)) for _ in S1.sites]
        Dl = S1.D(U0, lam)
        Up = [mul(E(tuple(-1j*HC*z for z in v)), u) for v, u in zip(Dl, U0)]
        lhs = [cs(ell(mul(q, inv(w)))) for q, w in zip(Q2(Up), W2)]                       # Q_2(-D lambda)
        rhs = [tuple(-z for z in v) for v in S2.Dc(W2, S2.Qp(S1.Qp(lam, AW0), AW1))]      # -D_{W_2} Q'_2 lambda
        errs.append(vmax([vsub(a, b) for a, b in zip(lhs, rhs)])); scale.append(vmax(lhs))
    print('  C2. k=2 (18x18 -> 6x6 -> 2x2), U0 random (amplitude 0.3): E=0, |Q_2 D lambda - D_W2 Q\'_2 lambda| = %.1e (scale %.2f), three random lambda'
          % (max(errs), min(scale)))
    g = [rand_su2(0.2) for _ in S1.sites]
    Mg = S2.m(S1.m(g, AW0), AW1)
    e = max(dist(a, b) for a, b in zip(Q2(S1.gauge(U0, g)), S2.cgauge(W2, Mg)))
    print('      Theorem correspondence(e) at k=2: |Q_2(1^(g)) - 1^(M^(2)(g))| = %.1e  (g random, amplitude 0.2)' % e)
    # linearization at U = 1 at k = 2: the centred tube average of side L^2
    one = [ONE]*len(S1.links); AW0 = S1.acv(one); AW1 = S2.acv(S1.avg(one))
    B = [tuple(random.gauss(0, 1) for _ in range(3)) for _ in S1.links]
    lin = [cs(ell(q)) for q in S2.atilde(S1.atilde([E(tuple(1j*HC*z for z in b)) for b in B], AW0), AW1)]
    Lb = 9; hw = 4; tub = []
    for (y, mu) in S2.clinks:
        y1 = (3*y[0], 3*y[1]); acc = [0.0, 0.0, 0.0]
        for a in range(-hw, hw + 1):
            for b in range(-hw, hw + 1):
                x = S1.site((y1[0] + a, y1[1] + b))
                for (k, sg) in S1.path(x, [(mu, Lb)]): acc = [p + q for p, q in zip(acc, B[k])]
        tub.append(tuple(v/Lb**2 for v in acc))
    print('      linearization of Q_2 at U=1 = centred tube average of side 9: %.1e' % vmax([vsub(a, b) for a, b in zip(lin, tub)]))

    # C3. the Poincare form of Delta^exp at U = 1 (abelian, one colour)
    print('  C3. smallest eigenvalue at U=1 of |dB|^2 + |G B|^2 + |q B|^2 (q: the plain constraint (0.12)), and with one term dropped')
    for (N, L) in ((3, 3), (6, 3), (9, 3), (12, 3), (15, 3), (5, 5), (10, 5), (15, 5)):
        S = Scale(N, L); n = len(S.links); rows_d, rows_G, rows_q = [], [], []
        for x in S.sites:      # plaquettes
            r = [0.0]*n
            for (k, sg) in S.path(x, [(0, 1), (1, 1), (0, -1), (1, -1)]): r[k] += sg
            rows_d.append(r)
        for (y, off), fam in S.fams.items():
            if off == (0, 0): continue
            r = [0.0]*n
            for p in fam:
                for (k, sg) in p: r[k] += sg/len(fam)
            rows_G.append(r)
        for (y, mu) in S.clinks:     # linearization of (0.12) at U = 1: tube average minus the coarse gradient of gamma_sym
            r = [0.0]*n; yp = S.add(y, mu, L)
            for off in S.offs:
                x = S.site((y[0] + off[0], y[1] + off[1]))
                for (k, sg) in S.line[(x, mu)]: r[k] += sg/L**2
                for yy, s_ in ((y, 1), (yp, -1)):
                    fam = S.fams[(yy, off)]
                    for p in fam:
                        for (k, sg) in p: r[k] += s_*sg/(L**2*len(fam))
            rows_q.append(r)
        res = []
        for parts in ((rows_d, rows_G, rows_q), (rows_d, rows_q), (rows_d, rows_G), (rows_G, rows_q)):
            M = [[0.0]*n for _ in range(n)]
            for R in parts:
                for r in R:
                    nz = [(i, v) for i, v in enumerate(r) if v]
                    for i, a in nz:
                        for j, b in nz: M[i][j] += a*b
            res.append(min_eig(M))
        print('    torus %2dx%-2d L=%d: %.6f   (without G: %.1e, without q: %.1e, without d: %.1e)' % (N, N, L, *res))

def min_eig(M):
    """Smallest eigenvalue of a symmetric positive semidefinite matrix: Cholesky of M + s I, inverse iteration."""
    n = len(M); s = 1e-10
    Mi = [[M[i][j] + (s if i == j else 0.0) for j in range(n)] for i in range(n)]
    Lc = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            t = Mi[i][j] - sum(Lc[i][k]*Lc[j][k] for k in range(j))
            if i == j: Lc[i][i] = math.sqrt(max(t, 1e-300))
            else: Lc[i][j] = t/Lc[j][j]
    def solve(b):
        y = [0.0]*n
        for i in range(n): y[i] = (b[i] - sum(Lc[i][k]*y[k] for k in range(i)))/Lc[i][i]
        x = [0.0]*n
        for i in reversed(range(n)): x[i] = (y[i] - sum(Lc[k][i]*x[k] for k in range(i + 1, n)))/Lc[i][i]
        return x
    v = [random.gauss(0, 1) for _ in range(n)]; lam = None
    for it in range(400):
        nv = math.sqrt(sum(t*t for t in v)); v = [t/nv for t in v]
        w = solve(v); ray = sum(a*b for a, b in zip(v, w))
        if lam is not None and abs(1/ray - lam) < 1e-12*max(1.0, abs(lam)): break
        lam = 1/ray; v = w
    Mv = [sum(M[i][j]*v[j] for j in range(n)) for i in range(n)]
    return sum(a*b for a, b in zip(v, Mv))/sum(t*t for t in v) if lam > 1e-8 else 0.0

# ======================================================================================================
# Part D: d = 4, one coarse bond on the infinite lattice
# ======================================================================================================
D4D = 4; L4 = 3; HW4 = 1
OFFS4 = list(itertools.product(range(-HW4, HW4 + 1), repeat=D4D))
def e4(mu, n=1): v = [0]*D4D; v[mu] = n; return tuple(v)
def vadd(x, y): return tuple(a + b for a, b in zip(x, y))
def steps(moves):      # moves: list of (mu, n) -> list of (offset of the site where the step starts, mu, sgn)
    p = []; cur = (0,)*D4D
    for mu, n in moves:
        sg = 1 if n > 0 else -1
        for _ in range(abs(n)): p.append((cur, mu, sg)); cur = vadd(cur, e4(mu, sg))
    return p
FAM4 = {off: [steps([(mu, off[mu]) for mu in o]) for o in itertools.permutations([mu for mu in range(D4D) if off[mu] != 0])] for off in OFFS4}
FAM4one = {off: [steps([(mu, off[mu]) for mu in reversed(range(D4D)) if off[mu] != 0])] for off in OFFS4}
class Field:     # link variables on Z^4, drawn on demand
    def __init__(self, gen): self.v = {}; self.gen = gen
    def __call__(self, s, mu):
        k = (s, mu)
        if k not in self.v: self.v[k] = self.gen()
        return self.v[k]
def tr4(U, y, p):
    m = ONE
    for (o, mu, sg) in p:
        s = vadd(y, o)
        m = mul(m, U(s, mu) if sg == 1 else inv(U(vadd(s, e4(mu, -1)), mu)))
    return m
def acv4(U, y, fams=FAM4): return {off: gavg([tr4(U, y, p) for p in fams[off]]) for off in OFFS4}
def avg4(U, y, mu, sg, fams=FAM4, A=None, Ap=None):
    """(0.12) for the coarse bond from y in direction sg*e_mu."""
    yp = vadd(y, e4(mu, sg*L4)); line = steps([(mu, sg*L4)])
    if A is None: A = acv4(U, y, fams)
    if Ap is None: Ap = acv4(U, yp, fams)
    Yc = tr4(U, y, line); Yci = inv(Yc); acc = [0j, 0j, 0j]
    for off in OFFS4:
        v = ell(mul(mul(mul(A[off], tr4(U, vadd(y, off), line)), inv(Ap[off])), Yci))
        for a in range(3): acc[a] += v[a]
    return mul(E(tuple(v/L4**D4D for v in acc)), Yc)
def gauged(U, g): return lambda s, mu: mul(mul(g(s), U(s, mu)), inv(g(vadd(s, e4(mu)))))
def act4(r, U):
    """(rU)(r b) = U(b) for oriented bonds; r a signed permutation (perm, signs): (r x)_{perm[i]} = signs[i] x_i."""
    perm, sgn = r; ip = [0]*D4D
    for i, p in enumerate(perm): ip[p] = i
    def rinv(x): return tuple(sgn[i]*x[perm[i]] for i in range(D4D))
    def f(s, nu):
        mu = ip[nu]; t = rinv(s)
        return U(t, mu) if sgn[mu] == 1 else inv(U(vadd(t, e4(mu, -1)), mu))
    return f
def rvec(r, x):
    perm, sgn = r; out = [0]*D4D
    for i in range(D4D): out[perm[i]] = sgn[i]*x[i]
    return tuple(out)
B4 = [(perm, sgn) for perm in itertools.permutations(range(D4D)) for sgn in itertools.product((1, -1), repeat=D4D)]

def part_d():
    print('\nPart D: d=4, SU(2), L=3, coarse bond c = (0, e_1); families of sizes %s' % sorted(set(len(f) for f in FAM4.values())))
    U = Field(lambda: rand_su2(0.3)); g = {}
    gf = lambda s: g.setdefault(s, rand_su2())
    A = acv4(U, (0,)*D4D); Ug = gauged(U, gf); Ag = acv4(Ug, (0,)*D4D)
    e_law = max(dist(Ag[off], mul(mul(gf((0,)*D4D), A[off]), inv(gf(off)))) for off in OFFS4)
    c0 = avg4(U, (0,)*D4D, 0, 1)
    e_cov = dist(avg4(Ug, (0,)*D4D, 0, 1), mul(mul(gf((0,)*D4D), c0), inv(gf(e4(0, L4)))))
    e_rev = dist(avg4(U, e4(0, L4), 0, -1), inv(c0))
    print('  D1. transformation law of the contour variables (Haar-random g): %.1e;  gauge covariance of (0.12): %.1e;  reversal: %.1e' % (e_law, e_cov, e_rev))
    t0 = time.time(); worst = 0.0; worst_one = []; c1 = avg4(U, (0,)*D4D, 0, 1, FAM4one)
    for r in B4:
        rU = act4(r, U); im = rvec(r, e4(0)); nu = next(i for i in range(D4D) if im[i] != 0)
        worst = max(worst, dist(avg4(rU, (0,)*D4D, nu, im[nu]), c0))
        worst_one.append(dist(avg4(rU, (0,)*D4D, nu, im[nu], FAM4one), c1))
    bad = sum(1 for w in worst_one if w > 1e-10)
    print('  D2. hyperoctahedral covariance of (0.12), all %d elements of B_4: %.1e   [%.0f s]' % (len(B4), worst, time.time() - t0))
    print('      control with one ordering per contour family: %d of %d elements fail (max %.2f)' % (bad, len(B4), max(worst_one)))
    # linearization at U = 1
    one = lambda s, mu: ONE
    b0 = (e4(0, HW4), 0)
    def pert(k):
        def f(s, mu):
            if (s, mu) != b0: return ONE
            e = [0j, 0j, 0j]; e[k] = 1j*HC; return E(tuple(e))
        return f
    coef = [cs(ell(avg4(pert(k), (0,)*D4D, 0, 1)))[k] for k in range(3)]
    print('  D3. coefficient of Z(b_0(c)) in the plain constraint at U=1: %s (L^{1-d} = %.10f)' % (' '.join('%.10f' % c for c in coef), L4**(1 - D4D)))
    Bf = Field(lambda: tuple(random.gauss(0, 1) for _ in range(3)))
    Ucs = lambda s, mu: E(tuple(1j*HC*z for z in Bf(s, mu)))
    def Bline(y, p):
        acc = [0.0, 0.0, 0.0]
        for (o, mu, sg) in p:
            s = vadd(y, o); b = Bf(s, mu) if sg == 1 else Bf(vadd(s, e4(mu, -1)), mu)
            acc = [a + sg*z for a, z in zip(acc, b)]
        return acc
    yp = e4(0, L4); line = steps([(0, L4)])
    tube = [sum(Bline(off, line)[a] for off in OFFS4)/L4**D4D for a in range(3)]
    def gam(y): return [sum(sum(Bline(y, p)[a] for p in FAM4[off])/len(FAM4[off]) for off in OFFS4)/L4**D4D for a in range(3)]
    gm, gp = gam((0,)*D4D), gam(yp)
    Acs, Apcs = acv4(Ucs, (0,)*D4D), acv4(Ucs, yp)
    plain = cs(ell(avg4(Ucs, (0,)*D4D, 0, 1, A=Acs, Ap=Apcs)))
    def tau(Aa): return E(tuple(sum(ell(Aa[off])[a] for off in OFFS4)/L4**D4D for a in range(3)))
    adapted = cs(ell(mul(mul(inv(tau(Acs)), avg4(Ucs, (0,)*D4D, 0, 1, A=Acs, Ap=Apcs)), tau(Apcs))))
    print('      plain (0.12) = tube average - (gamma_sym(c+) - gamma_sym(c-)): %.1e;  gauge-adapted average = tube average: %.1e  (random direction B)'
          % (max(abs(p - (t - (b - a))) for p, t, a, b in zip(plain, tube, gm, gp)), max(abs(p - t) for p, t in zip(adapted, tube))))
    # E = 0 and Theorem correspondence(e) at one coarse bond, U0 random
    U0 = Field(lambda: rand_su2(0.3)); W0m, W0p = acv4(U0, (0,)*D4D), acv4(U0, yp); V = avg4(U0, (0,)*D4D, 0, 1, A=W0m, Ap=W0p)
    lamf = {}; lam = lambda s: lamf.setdefault(s, tuple(random.gauss(0, 1) for _ in range(3)))
    def Dl(s, mu): return vsub(adv(U0(s, mu), lam(vadd(s, e4(mu)))), lam(s))
    Up = lambda s, mu: mul(E(tuple(-1j*HC*z for z in Dl(s, mu))), U0(s, mu))
    def calQ(Ux):
        Am, Ap = acv4(Ux, (0,)*D4D), acv4(Ux, yp)
        tm = E(tuple(sum(ell(mul(Am[o], inv(W0m[o])))[a] for o in OFFS4)/L4**D4D for a in range(3)))
        tp = E(tuple(sum(ell(mul(Ap[o], inv(W0p[o])))[a] for o in OFFS4)/L4**D4D for a in range(3)))
        return mul(mul(inv(tm), avg4(Ux, (0,)*D4D, 0, 1, A=Am, Ap=Ap)), tp)
    def Qp(y, W):
        acc = [0.0, 0.0, 0.0]
        for o in OFFS4: acc = [a + b for a, b in zip(acc, adv(W[o], lam(vadd(y, o))))]
        return tuple(a/L4**D4D for a in acc)
    lhs = cs(ell(mul(calQ(Up), inv(V))))
    rhs = tuple(-z for z in vsub(adv(V, Qp(yp, W0p)), Qp((0,)*D4D, W0m)))
    print('  D4. E=0 at U0 random (amplitude 0.3), complex steps through the solver: |Q D lambda - D_V Q\' lambda| = %.1e (scale %.2f)'
          % (max(abs(a - b) for a, b in zip(lhs, rhs)), max(abs(a) for a in lhs)))
    gg = {}; gfun = lambda s: gg.setdefault(s, rand_su2(0.3))
    def m(y, W):
        acc = [0j, 0j, 0j]
        for o in OFFS4:
            v = ell(mul(mul(mul(inv(gfun(y)), W[o]), gfun(vadd(y, o))), inv(W[o])))
            for a in range(3): acc[a] += v[a]
        return mul(gfun(y), E(tuple(v/L4**D4D for v in acc)))
    e = dist(calQ(gauged(U0, gfun)), mul(mul(m((0,)*D4D, W0m), V), inv(m(yp, W0p))))
    print('      Theorem correspondence(e): |Q(1^(g)) - 1^(m(g))| = %.1e  (g random, amplitude 0.3)' % e)

if __name__ == '__main__':
    random.seed(SEED)
    t0 = time.time(); part_c(); part_d(); print('\n[%.0f s]' % (time.time() - t0))
