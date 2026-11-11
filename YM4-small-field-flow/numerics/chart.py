# Numerical test of Theorem ol-chart (Remark ol-numerical): d=2, SU(2), k=1, block side L, coarse torus nc x nc.
#
# LHS: Gamma^(1)(V) by Laplace's method for  int prod_b dU(b) delta(U^(1), V) e^{-S/g^2}  in the tree (axial) gauge
#      U = 1 on the tree bonds (Lemma FP, no Jacobian):  -1/2 log det(H_L|ker dc) - 1/2 log det(dc dc^T).
# RHS: -1/2 log det L_* - 1/2 log det(Q G_* Q^T) + 1/2 log det(P X^2 P|_N(Q'))     (Theorem ol-chart)
#      L_* = q_2 + D R D^T + a Q^T Q,  q_2 = Hess of Phi(A) - <omega, l(calQ(e^{iA}))> at A = 0,  calQ the gauge-adapted average.
# Derivatives: complex step (first), complex step x central difference (second); all maps are holomorphic in the Lie coordinates.
import cmath, math, random, sys, time

L = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 2
NF = L * NC
d = 2
HC = 1e-20      # complex step
DLT = 1e-5      # real step for second derivatives

# ---------------- SU(2) holomorphic primitives, quaternion form h = h0 + i hv.sigma ----------------
# Real arithmetic for real input (needed for the complex-step derivatives).
def mul(p, q):
    return (p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
            p[0]*q[1] + q[0]*p[1] - (p[2]*q[3] - p[3]*q[2]),
            p[0]*q[2] + q[0]*p[2] - (p[3]*q[1] - p[1]*q[3]),
            p[0]*q[3] + q[0]*p[3] - (p[1]*q[2] - p[2]*q[1]))
def inv(p): return (p[0], -p[1], -p[2], -p[3])
ONE = (1.0, 0.0, 0.0, 0.0)
def E(x):    # exp(i x.sigma/2)
    th2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
    if abs(th2) < 1e-12:
        c = 1 - th2/8 + th2*th2/384; s = 0.5 - th2/48 + th2*th2/3840
    else:
        th = cmath.sqrt(th2); c = cmath.cos(th/2); s = cmath.sin(th/2)/th
    return (c, s*x[0], s*x[1], s*x[2])
def ell(h):  # x with h = E(x) (principal branch, h near 1)
    u = 1 - h[0]
    if abs(u) < 1e-5: f = 1 + u/3 + 2*u*u/15 + 2*u**3/35
    else: f = cmath.acos(h[0])/cmath.sqrt(1 - h[0]*h[0])
    return (2*f*h[1], 2*f*h[2], 2*f*h[3])
def adj(h):  # R(h) = Ad(h) on components: h (i X.sigma) h^{-1} = i (R X).sigma
    hi = inv(h); R = [[0.0]*3 for _ in range(3)]
    for b in range(3):
        e = [0.0, 0.0, 0.0, 0.0]; e[b+1] = 1.0
        m = mul(mul(h, tuple(e)), hi)
        for a in range(3): R[a][b] = m[a+1]
    return R
def retr(h): return 2*h[0]

# ---------------- lattice geometry ----------------
def site(x): return ((x[0] % NF), (x[1] % NF))
sites = [(i, j) for i in range(NF) for j in range(NF)]
SI = {s: n for n, s in enumerate(sites)}
links = [(s, mu) for s in sites for mu in range(d)]
LI = {l: n for n, l in enumerate(links)}
csites = [(L*i, L*j) for i in range(NC) for j in range(NC)]
clinks = [(y, mu) for y in csites for mu in range(d)]
def step(x, mu, sgn=1):
    e = [0, 0]; e[mu] = sgn; return site((x[0] + e[0], x[1] + e[1]))
def block(y): return [site((y[0] + a, y[1] + b)) for a in range(L) for b in range(L)]
def tree_path(y, x):   # coordinate order d,...,1 : direction index 1 first, then 0. Returns list of (site, mu, +1) steps.
    p = []; cur = y
    for mu in (1, 0):
        n = (x[mu] - y[mu]) % NF
        for _ in range(n): p.append((cur, mu, 1)); cur = step(cur, mu)
    assert cur == x
    return p
def straight(x, mu, n):
    p = []; cur = x
    for _ in range(n): p.append((cur, mu, 1)); cur = step(cur, mu)
    return p, cur
def reverse(p):
    out = []
    for (s, mu, sg) in reversed(p):
        if sg == 1: out.append((step(s, mu), mu, -1))
        else: out.append((step(s, mu, -1), mu, 1))
    return out
def transport(U, p):
    m = ONE
    for (s, mu, sg) in p:
        m = mul(m, U[LI[(s, mu)]] if sg == 1 else inv(U[LI[(step(s, mu, -1), mu)]]))
    return m
# averaging contours for each coarse link c=(y,mu): for x in B(y): Gamma_{c,x} and [c]
AVG = {}
for (y, mu) in clinks:
    cp, cplus = straight(y, mu, L)
    lst = []
    for x in block(y):
        s1 = tree_path(y, x); s2, xc = straight(x, mu, L); s3 = reverse(tree_path(cplus, xc))
        lst.append(s1 + s2 + s3)
    AVG[(y, mu)] = (lst, cp, cplus)
TREE = {}
for y in csites:
    for x in block(y): TREE[(y, x)] = tree_path(y, x)
tree_links = set()
for p in TREE.values():
    for (s, mu, sg) in p: tree_links.add(LI[(s, mu)])
free_links = [n for n in range(len(links)) if n not in tree_links]

def average(U):   # Balaban's one-step averaging a_1(U)(c), c in clinks
    out = []
    for c in clinks:
        lst, cp, cplus = AVG[c]
        Yc = transport(U, cp); Yci = inv(Yc); acc = [0j, 0j, 0j]
        for p in lst:
            v = ell(mul(transport(U, p), Yci))
            for a in range(3): acc[a] += v[a]
        out.append(mul(E(tuple(v/L**d for v in acc)), Yc))
    return out
def action(U):    # sum_p (2 - tr U(dp))
    s = 0.0
    for x in sites:
        P = mul(mul(mul(U[LI[(x, 0)]], U[LI[(step(x, 0), 1)]]), inv(U[LI[(step(x, 1), 0)]])), inv(U[LI[(x, 1)]]))
        s += 2 - retr(P)
    return s

# ---------------- generic derivatives ----------------
def grad_cs(f, n):            # f: complex vector of length n -> complex scalar (real for real input)
    g = []
    for i in range(n):
        x = [0j]*n; x[i] = 1j*HC; g.append(f(x).imag/HC)
    return g
def jac_cs(F, n):             # F -> list of complex (real for real input); returns m x n
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

# ---------------- dense linear algebra ----------------
def T(M): return [list(r) for r in zip(*M)]
def mm(A, B):
    Bt = T(B); return [[sum(a*b for a, b in zip(r, c)) for c in Bt] for r in A]
def mv(A, v): return [sum(a*b for a, b in zip(r, v)) for r in A]
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
def solve(M, B):   # M x = B (B matrix or vector)
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
def nullspace(J, n):  # ONB of ker J (J: m x n)
    rows = onb(J)
    comp = []
    for i in range(n):
        e = [0.0]*n; e[i] = 1.0
        for r in rows:
            c = r[i]; e = [s - c*t for s, t in zip(e, r)]
        comp.append(e)
    return onb(comp)

# ---------------- LHS: Laplace in the tree gauge ----------------
def tree_gauge(U):
    g = [ONE]*len(sites)
    for (y, x), p in TREE.items(): g[SI[x]] = transport(U, p)
    return [mul(mul(g[SI[s]], U[LI[(s, mu)]]), inv(g[SI[step(s, mu)]])) for (s, mu) in links]
def lhs_functions(Ucur, V):
    Vinv = [inv(v) for v in V]; nf = len(free_links)
    def config(X):
        U = list(Ucur)
        for k, li in enumerate(free_links): U[li] = mul(E((X[3*k], X[3*k+1], X[3*k+2])), Ucur[li])
        return U
    def cons(X):
        Ub = average(config(X)); out = []
        for c in range(len(clinks)): out.extend(ell(mul(Ub[c], Vinv[c])))
        return out
    def S(X): return action(config(X))
    return config, cons, S, 3*nf
def restore(Ucur, V, tol=1e-14):
    for _ in range(20):
        config, cons, S, n = lhs_functions(Ucur, V)
        c0 = [v.real for v in cons([0j]*n)]
        if max(abs(v) for v in c0) < tol: return Ucur, True
        J = jac_cs(cons, n); y = solve(mm(J, T(J)), c0); dY = [-v for v in mv(T(J), y)]
        Ucur = [tuple(complex(z).real for z in u) for u in config([complex(v) for v in dY])]
    return Ucur, False
def minimize_on_fibre(U, V, iters=60, verbose=True):
    Ucur, ok = restore(tree_gauge(U), V); omega = None; mu = 0.0
    for it in range(iters):
        config, cons, S, n = lhs_functions(Ucur, V)
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
            Utr, ok = restore([tuple(complex(z).real for z in u) for u in config([complex(v) for v in dX])], V)
            S0 = S([0j]*n).real; S1 = action(Utr)
            if ok and (S1 <= S0 + 1e-13 or big < 1e-10): break
            mu = max(4*mu, 0.05)
            if mu > 1e6: raise RuntimeError('no descent')
        Ucur, omega = Utr, om_new
        if verbose: print('   it %2d: S = %.14f  |dX| = %.2e  mu = %.2g' % (it, S1, big, mu), flush=True)
        if big < 1e-11 and mu == 0.0: break
        mu = mu/8 if mu > 1e-3 else 0.0
    return Ucur
def lhs(U0, V):
    config, cons, S, n = lhs_functions(U0, V)
    J = jac_cs(cons, n); g = grad_cs(S, n)
    JJ = mm(J, T(J)); omega = solve(JJ, mv(J, g))
    resid = max(abs(a - b) for a, b in zip(g, mv(T(J), omega)))
    Lag = lambda X: S(X) - sum(w*v for w, v in zip(omega, cons(X)))
    H = hess_cs(Lag, n)
    B = nullspace(J, n); HB = mm(B, mm(H, T(B)))
    ld1, s1 = logdet(HB); ld2, s2 = logdet(JJ)
    return -0.5*ld1 - 0.5*ld2, resid, s1

# ---------------- RHS: the Landau-chart formula ----------------
def rhs(U0, V, avals=(1.0,)):
    nl = len(links); n = 3*nl; Vinv = [inv(v) for v in V]
    def config(A): return [mul(E((A[3*k], A[3*k+1], A[3*k+2])), U0[k]) for k in range(nl)]
    U0tree = {key: transport(U0, p) for key, p in TREE.items()}
    def calQ(A):
        U = config(A)
        tau = {}
        for y in csites:
            acc = [0j, 0j, 0j]
            for x in block(y):
                p = TREE[(y, x)]
                v = ell(mul(transport(U, p), inv(U0tree[(y, x)])))
                for a in range(3): acc[a] += v[a]
            tau[y] = E(tuple(v/L**d for v in acc))
        Ub = average(U); out = []
        for ci, (y, mu) in enumerate(clinks):
            yp = step(y, mu) if L == 1 else site((y[0] + L*(mu == 0), y[1] + L*(mu == 1)))
            Xbar = mul(Ub[ci], Vinv[ci])
            Q = mul(mul(mul(inv(tau[y]), Xbar), V[ci]), mul(tau[yp], Vinv[ci]))
            out.extend(ell(Q))
        return out
    def Phi(A): return action(config(A))
    Q = jac_cs(calQ, n); g = grad_cs(Phi, n)
    m = len(Q)
    QQ = mm(Q, T(Q)); omega = solve(QQ, mv(Q, g))
    resid = max(abs(a - b) for a, b in zip(g, mv(T(Q), omega)))
    HPhi = hess_cs(Phi, n)
    Hom = hess_cs(lambda A: sum(w*v for w, v in zip(omega, calQ(A))), n)
    q2 = [[HPhi[i][j] - Hom[i][j] for j in range(n)] for i in range(n)]
    # covariant derivative D: scalars (3 per site) -> 1-forms (3 per link)
    D = [[0.0]*(3*len(sites)) for _ in range(n)]
    for k, (s, mu) in enumerate(links):
        Rb = adj(U0[k]); t = SI[step(s, mu)]
        for a in range(3):
            D[3*k+a][3*SI[s]+a] -= 1.0
            for b in range(3): D[3*k+a][3*t+b] += complex(Rb[a][b]).real
    # Q': transported block average, rows orthonormal
    ns = 3*len(sites); Qp = [[0.0]*ns for _ in range(3*len(csites))]
    for yi, y in enumerate(csites):
        for x in block(y):
            Rt = adj(U0tree[(y, x)])
            for a in range(3):
                for b in range(3): Qp[3*yi+a][3*SI[x]+b] += complex(Rt[a][b]).real/L**(d/2)
    X = mm(T(D), D)
    N = nullspace(Qp, ns)                       # ONB of N(Q')
    XN = mm(X, T(N))                            # columns span W
    W = onb(T(XN))                              # ONB of W
    Rproj = mm(T(W), W)
    ghost = 0.5*logdet(mm(T(XN), XN))[0]
    DRDt = mm(mm(D, Rproj), T(D))
    # diagnostics: gauge zero modes q2 D lambda = 0 for lambda in N(Q'); exact covariance Q D = D_V Q' (Q D N = 0)
    q2DN = mm(q2, mm(D, T(N))); zm = max(abs(v) for r in q2DN for v in r)
    QDN = mm(Q, mm(D, T(N))); cov = max(abs(v) for r in QDN for v in r)
    out = []; parts = {}
    for a in avals:
        for tag, qq in (('', q2), ('nomult', HPhi)):
            Ls = [[qq[i][j] + DRDt[i][j] + a*sum(Q[r][i]*Q[r][j] for r in range(m)) for j in range(n)] for i in range(n)]
            ldL, sL = logdet(Ls)
            G_Qt = solve(Ls, T(Q)); M = mm(Q, G_Qt)
            ldM, sM = logdet(M)
            if tag == '': out.append((a, -0.5*ldL - 0.5*ldM + ghost, sL, sM))
            parts[(tag, a)] = (-0.5*ldL, -0.5*ldM, ghost)
    return out, resid, zm, cov, ghost, parts

if __name__ == '__main__':
    random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    amp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    nV = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    print('d=2, SU(2), L=%d, coarse torus %dx%d, fine %dx%d; free links %d of %d' % (L, NC, NC, NF, NF, len(free_links), len(links)), flush=True)
    for iv in range(nV):
        t0 = time.time()
        Uin = [E(tuple(random.gauss(0, amp) for _ in range(3))) for _ in links]
        V = average(Uin)
        U0 = minimize_on_fibre(Uin, V)
        cV = [ell(mul(a, inv(b))) for a, b in zip(average(U0), V)]
        print('  V%d: S(U0) = %.10f, constraint residual %.1e' % (iv, action(U0).real, max(abs(z) for v in cV for z in v)), flush=True)
        l, res_l, sgn = lhs(U0, V)
        print('  V%d: LHS = %.10f   (multiplier residual %.1e, Hessian sign %d)' % (iv, l, res_l, sgn), flush=True)
        r, res_r, zm, cov, gh, parts = rhs(U0, V, avals=(0.5, 2.0))
        pL, pM, pg = parts[('', 0.5)]; nL, nM, _ = parts[('nomult', 0.5)]
        print('  V%d: PARTS %.10f %.10f %.10f %.10f %.10f %.10f' % (iv, l, pL, pM, pg, nL, nM), flush=True)
        print('  V%d: RHS = %s   (EL residual %.1e, |q2 D N(Q\')| = %.1e, |Q D N(Q\')| = %.1e)' %
              (iv, '  '.join('a=%.1f: %.10f (signs %d,%d)' % x for x in r), res_r, zm, cov), flush=True)
        print('  V%d: LHS - RHS = %.10f   [%.0f s]' % (iv, l - r[0][1], time.time() - t0), flush=True)
