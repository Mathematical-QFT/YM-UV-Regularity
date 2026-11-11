# N1(c) of the Y3 plan: d=2, SU(2), L=3, Balaban's averaging (0.12) as implemented in numerics/symavg.py (imported, not
# modified). On the coarse NC x NC torus (fine torus (3 NC)^2) take the U(1) subset SU(2) configuration U_F of constant
# curvature with n flux quanta (fine plaquette angle theta = 4 pi n / (3 NC)^2 in exp(i theta n.sigma/2)), and V = a(U_F).
#  1. V has constant curvature L^2 theta (coarse plaquettes).
#  2. U_F is critical for S on the fibre {a(U) = V}: grad S(U_F) = 0 (multiplier 0).
#  3. Second order: the Hessian of S at U_F on the tree-gauge slice (free links), restricted to ker(d cons), is positive
#     definite (inertia by symmetric elimination; smallest eigenvalue by inverse iteration). Also without the constraint.
#  4. Global (optional, nstart > 0): SQP minimizations on the fibre from random starts converge to U_F up to G_0
#     (tree-gauge representatives coincide). Same method as symavg.minimize_on_fibre, with a local (fast) Hessian.
#   python3 n1c.py NC n_flux seed nstart amp
import sys, math, random, time
import os
NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, NUM)
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NFLUX = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
NSTART = int(sys.argv[4]) if len(sys.argv) > 4 else 0
AMP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
sys.argv = ['symavg.py', 'checks', '3', str(NC)]
import symavg as sa
from su2 import mul, inv, ONE, E, ell, dist, rand_su2, retr, real
random.seed(SEED)
L, NF, HC, DLT = sa.L, sa.NF, sa.HC, sa.DLT

# ------------------------------------------------------------------ the configuration
v = [random.gauss(0, 1) for _ in range(3)]; nv = math.sqrt(sum(t*t for t in v)); NH = tuple(t/nv for t in v)
theta = 4*math.pi*NFLUX/NF**2
def ang(s, mu):
    if mu == 1: return theta*s[0]
    return -theta*NF*s[1] if s[0] == NF - 1 else 0.0
UF = [E(tuple(ang(s, mu)*c for c in NH)) for (s, mu) in sa.links]
UF = [real(u) for u in UF]
def plaq(U, x): return mul(mul(mul(U[sa.LI[(x, 0)]], U[sa.LI[(sa.step(x, 0), 1)]]), inv(U[sa.LI[(sa.step(x, 1), 0)]])), inv(U[sa.LI[(x, 1)]]))
def cplaq(V, y):
    return mul(mul(mul(V[sa.CLI[(y, 0)]], V[sa.CLI[(sa.add(y, (1, 0), L), 1)]]), inv(V[sa.CLI[(sa.add(y, (0, 1), L), 0)]])), inv(V[sa.CLI[(y, 1)]]))
print('N1(c): d=2, SU(2), L=%d, coarse %dx%d, fine %dx%d, flux %d quanta: fine plaquette angle %.4f, coarse %.4f; U(1) direction %s; seed %d'
      % (L, NC, NC, NF, NF, NFLUX, theta, L*L*theta, ' '.join('%.3f' % c for c in NH), SEED), flush=True)
print('  fine plaquettes = exp(i theta n.sigma/2): %.1e' % max(dist(plaq(UF, x), E(tuple(theta*c for c in NH))) for x in sa.sites))
V = sa.average(UF)
print('  1. coarse plaquettes of V = a(U_F) vs exp(i L^2 theta n.sigma/2): %.1e' % max(dist(cplaq(V, y), E(tuple(L*L*theta*c for c in NH))) for y in sa.csites))
S0 = sa.action(UF)
print('     S(U_F) = %.12f = |T_fine| (2 - 2 cos(theta/2)) = %.12f' % (complex(S0).real, NF**2*(2 - 2*math.cos(theta/2))))

# ------------------------------------------------------------------ local derivatives
FREE = sa.free_links; FIDX = {li: k for k, li in enumerate(FREE)}; n = 3*len(FREE)
PLAQ_LINKS = {x: [sa.LI[(x, 0)], sa.LI[(sa.step(x, 0), 1)], sa.LI[(sa.step(x, 1), 0)], sa.LI[(x, 1)]] for x in sa.sites}
def apply(U, pert):
    """pert: {variable: complex}; variable 3k+a acts on free link FREE[k] by left multiplication E(.)"""
    Up = list(U); byk = {}
    for var, val in pert.items(): byk.setdefault(var//3, [0j, 0j, 0j])[var % 3] += val
    for k, x in byk.items(): Up[FREE[k]] = mul(E(tuple(x)), Up[FREE[k]])
    return Up
def hess_local(f, U, vars_, H, scale=1.0):
    """H += scale * Hessian of f(apply(U, pert)) over the variables vars_ (mixed complex step / real central difference)."""
    for i, vi in enumerate(vars_):
        for vj in vars_[i:]:
            p1 = {vi: 1j*HC}; p1[vj] = p1.get(vj, 0) + DLT
            p2 = {vi: 1j*HC}; p2[vj] = p2.get(vj, 0) - DLT
            h = scale*(complex(f(apply(U, p1))).imag - complex(f(apply(U, p2))).imag)/(2*HC*DLT)
            H[vi][vj] += h
            if vj != vi: H[vj][vi] += h
def hess_S(U):
    H = [[0.0]*n for _ in range(n)]
    for x in sa.sites:
        vars_ = [3*FIDX[li] + a for li in PLAQ_LINKS[x] if li in FIDX for a in range(3)]
        hess_local(lambda Up, x=x: 2 - retr(plaq(Up, x)), U, vars_, H)
    return H
# local constraint of one coarse link, and its variables
def links_of(paths): return set(k for p in paths for (k, sg) in p)
CVARS = {}
for ci, (y, mu) in enumerate(sa.clinks):
    yp = sa.add(y, sa.EV[mu], L)
    ls = links_of([p for off in sa.OFFS for yy in (y, yp) for p in sa.FAMS[(yy, off)]])
    ls |= links_of([sa.LINE[(sa.add(y, off), mu, 1)] for off in sa.OFFS])
    CVARS[ci] = [3*FIDX[li] + a for li in sorted(ls) if li in FIDX for a in range(3)]
def cons_local(U, ci, Vinv):
    y, mu = sa.clinks[ci]; yp = sa.add(y, sa.EV[mu], L)
    A = {}
    for yy in (y, yp):
        for off in sa.OFFS: A[(yy, off)] = sa.gavg([sa.tr(U, p) for p in sa.FAMS[(yy, off)]])
    return ell(mul(sa.avg_one(U, A, y, mu, 1), Vinv[ci]))
def hess_lag(U, omega, Vinv):
    H = hess_S(U)
    for ci in range(len(sa.clinks)):
        w = omega[3*ci:3*ci + 3]
        hess_local(lambda Up, ci=ci, w=w: sum(a*b for a, b in zip(w, cons_local(Up, ci, Vinv))), U, CVARS[ci], H, scale=-1.0)
    return H

# ------------------------------------------------------------------ linear algebra helpers
def inertia(M, tol=1e-10):
    """Numbers of negative, zero (|pivot| < tol * scale) and positive eigenvalues of a symmetric matrix, by symmetric
    elimination with diagonal pivoting (Sylvester); also the smallest |pivot|."""
    A = [r[:] for r in M]; N = len(A); neg = zero = pos = 0; idx = list(range(N)); pmin = float('inf')
    scale = max(abs(A[i][i]) for i in range(N)) or 1.0
    active = list(range(N))
    while active:
        p = max(active, key=lambda i: abs(A[i][i]))
        piv = A[p][p]; active.remove(p)
        if abs(piv) < tol*scale:
            # remaining block (numerically) zero?
            if max([abs(A[i][j]) for i in active for j in active] + [0.0]) < tol*scale:
                zero += 1 + len(active); break
            zero += 1; continue
        pmin = min(pmin, abs(piv))
        if piv > 0: pos += 1
        else: neg += 1
        row = A[p]
        for i in active:
            f = A[i][p]/piv
            if f:
                Ai = A[i]
                for j in active: Ai[j] -= f*row[j]
    return neg, zero, pos, pmin
def min_eig_pd(M):
    """Smallest eigenvalue of a positive definite matrix (Cholesky, inverse iteration)."""
    N = len(M); Lc = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1):
            t = M[i][j] - sum(Lc[i][k]*Lc[j][k] for k in range(j))
            if i == j:
                if t <= 0: return None
                Lc[i][i] = math.sqrt(t)
            else: Lc[i][j] = t/Lc[j][j]
    def solve(b):
        y = [0.0]*N
        for i in range(N): y[i] = (b[i] - sum(Lc[i][k]*y[k] for k in range(i)))/Lc[i][i]
        x = [0.0]*N
        for i in reversed(range(N)): x[i] = (y[i] - sum(Lc[k][i]*x[k] for k in range(i + 1, N)))/Lc[i][i]
        return x
    v = [random.gauss(0, 1) for _ in range(N)]; lam = None
    for it in range(500):
        nv = math.sqrt(sum(t*t for t in v)); v = [t/nv for t in v]
        w = solve(v); ray = sum(a*b for a, b in zip(v, w))
        if lam is not None and abs(1/ray - lam) < 1e-13*max(1.0, abs(lam)): lam = 1/ray; break
        lam = 1/ray; v = w
    return lam

# ------------------------------------------------------------------ 2. criticality, 3. second order
t0 = time.time()
config, cons, S, nn = sa.lhs_functions(UF, V, FREE); assert nn == n
g_all = 0.0
for k in range(len(sa.links)):
    for a in range(3):
        e = [0j, 0j, 0j]; e[a] = 1j*HC; Up = list(UF); Up[k] = mul(E(tuple(e)), UF[k])
        g_all = max(g_all, abs(complex(sa.action(Up)).imag/HC))
J = sa.jac_cs(cons, n)
print('  2. |grad S(U_F)| over all %d bonds and colours = %.1e: U_F is critical on its fibre, with multiplier 0; constraint residual %.1e'
      % (len(sa.links), g_all, max(abs(complex(c).real) for c in cons([0j]*n))), flush=True)
H = hess_S(UF)
print('     Hessian of S on the %d free-link variables: %.0f s' % (n, time.time() - t0), flush=True)
neg, zer, pos, pm = inertia(H)
print('  3. without the constraint (tree-gauge slice only): inertia (neg, zero, pos) = (%d, %d, %d)' % (neg, zer, pos), flush=True)
B = sa.nullspace(J, n)
HB = sa.mm(B, sa.mm(H, sa.T(B)))
neg, zer, pos, pm = inertia(HB)
lm = min_eig_pd(HB) if neg == 0 and zer == 0 else None
print('     on ker(d cons) (dim %d = %d - %d): inertia (neg, zero, pos) = (%d, %d, %d), smallest |pivot| %.3e, smallest eigenvalue %s   [%.0f s]'
      % (len(B), n, n - len(B), neg, zer, pos, pm, ('%.6f' % lm) if lm is not None else 'n/a', time.time() - t0), flush=True)
# the same at U = 1 (flux 0), for comparison of the scale of the smallest eigenvalue
one = [ONE]*len(sa.links); V1 = sa.average(one)
c1, cs1, S1, _ = sa.lhs_functions(one, V1, FREE)
H1 = hess_S(one); B1 = sa.nullspace(sa.jac_cs(cs1, n), n); HB1 = sa.mm(B1, sa.mm(H1, sa.T(B1)))
n1_, z1_, p1_, _ = inertia(HB1); l1 = min_eig_pd(HB1) if n1_ == 0 and z1_ == 0 else None
print('     comparison at U = 1: inertia (%d, %d, %d), smallest eigenvalue %s   [%.0f s]' % (n1_, z1_, p1_, ('%.6f' % l1) if l1 else 'n/a', time.time() - t0), flush=True)

# ------------------------------------------------------------------ 4. global: SQP from random starts
def minimize(Uin, V, iters=int(__import__("os").environ.get("N1C_ITERS", "60"))):
    Vinv = [inv(v) for v in V]
    Ucur, ok = sa.restore(sa.tree_gauge(Uin), V); omega = None; mu = 0.0
    for it in range(iters):
        config, cons, S, nn = sa.lhs_functions(Ucur, V, FREE)
        J = sa.jac_cs(cons, n); g = sa.grad_cs(S, n); m = len(J)
        if omega is None: omega = sa.solve(sa.mm(J, sa.T(J)), sa.mv(J, g))
        Hl = hess_lag(Ucur, omega, Vinv)
        while True:
            K = [[Hl[i][j] + (mu if i == j else 0.0) for j in range(n)] + [J[r][i] for r in range(m)] for i in range(n)] + [J[r] + [0.0]*m for r in range(m)]
            sol = sa.solve(K, [-gi for gi in g] + [0.0]*m)
            dX = sol[:n]; om_new = [-v for v in sol[n:]]
            big = max(abs(v) for v in dX)
            if big > 0.3: dX = [v*0.3/big for v in dX]
            Utr, ok = sa.restore([real(u) for u in config([complex(v) for v in dX])], V)
            S0_ = complex(S([0j]*n)).real; S1_ = complex(sa.action(Utr)).real
            if ok and (S1_ <= S0_ + 1e-13 or big < 1e-10): break
            mu = max(4*mu, 0.05)
            if mu > 1e6: raise RuntimeError('no descent')
        Ucur, omega = Utr, om_new
        print('     it %2d: S = %.14f  |dX| = %.2e  mu = %.2g' % (it, S1_, big, mu), flush=True)
        if big < 1e-11 and mu == 0.0: return Ucur
        mu = mu/8 if mu > 1e-3 else 0.0
    raise RuntimeError('minimization did not converge')
if NSTART:
    # first: the local Hessian of the Lagrangian against symavg's hess_cs at a random point (small torus only)
    if NC == 2:
        Ut = sa.tree_gauge([mul(rand_su2(0.3), u) for u in UF]); Ut, _ = sa.restore(Ut, V)
        cf, cn, Sf, _ = sa.lhs_functions(Ut, V, FREE); om = [random.gauss(0, 1) for _ in range(3*len(sa.clinks))]
        t1 = time.time(); Href = sa.hess_cs(lambda X: Sf(X) - sum(w*c for w, c in zip(om, cn(X))), n); t2 = time.time()
        Hloc = hess_lag(Ut, om, [inv(v) for v in V]); t3 = time.time()
        print('  4. check: local Hessian of the Lagrangian vs symavg.hess_cs at a random point of the fibre: %.1e (scale %.2f); %.0f s vs %.0f s'
              % (sa.mdiff(Href, Hloc), sa.mmax(Href), t3 - t2, t2 - t1), flush=True)
    UFt = sa.tree_gauge(UF)
    for st in range(NSTART):
        t1 = time.time()
        Uin = [mul(rand_su2(AMP), u) for u in UF]
        print('  4. start %d: U_F times random SU(2) noise of amplitude %.2f, projected onto the fibre' % (st, AMP), flush=True)
        Umin = minimize(Uin, V)
        Smin = complex(sa.action(Umin)).real
        print('     minimum S = %.12f, S(U_F) = %.12f, difference %.1e;  constraint residual %.1e;  tree-gauge representatives: |U_min - U_F| = %.1e;'
              '  fine plaquette traces vs 2cos(theta/2): %.1e   [%.0f s]'
              % (Smin, complex(S0).real, Smin - complex(S0).real, sa.cfg_dist(sa.average(Umin), V), sa.cfg_dist(sa.tree_gauge(Umin), UFt),
                 max(abs(complex(retr(plaq(Umin, x))).real - 2*math.cos(theta/2)) for x in sa.sites), time.time() - t1), flush=True)
print('[%.0f s]' % (time.time() - t0))
