# N1(a), N1(b) of the Y3 plan: d=4, SU(2), L=3, Balaban's averaging (0.12) on the infinite lattice (fields on demand),
# with the geometry of numerics/symavg_k2d4.py Part D (imported, not modified).
#   (a) linearization at U=1 (complex steps through the iterative solver of (0.10)) applied to a linear field
#       A_F(x)_mu = 1/2 sum_nu x_nu F_nu,mu (one antisymmetric F per colour): coarse field = B_F (coarse linear field,
#       curvature L^2 F in coarse units) + coarse gradient; the gradient vanishes in the symmetric gauge.
#   (b) nonlinear: U(1) subset SU(2) configuration of constant curvature -> coarse plaquettes = exp(i L^2 F_{mu nu} n.sigma/2).
#   python3 n1ab.py [seed]
import sys, os, math, random, itertools, time
import os
NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, NUM)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
sys.argv = [sys.argv[0]]
import su2
from su2 import mul, inv, ONE, E, ell, dist, rand_su2, retr
import symavg_k2d4 as P
from symavg_k2d4 import (D4D, L4, OFFS4, FAM4, e4, vadd, steps, tr4, acv4, avg4, gauged, cs, HC)

random.seed(seed)
L = L4; d = D4D
def rand_F(scale):
    F = [[0.0]*d for _ in range(d)]
    for m in range(d):
        for n in range(m + 1, d):
            F[m][n] = random.gauss(0, scale); F[n][m] = -F[m][n]
    return F
def AF(F, x, mu, x0=(0, 0, 0, 0)):          # A_F(b)_mu for the bond (x, mu): 1/2 sum_nu (x - x0)_nu F_nu,mu (= value at midpoint)
    return 0.5*sum((x[n] - x0[n])*F[n][mu] for n in range(d))

print('N1 (d=4, SU(2), L=3), seed %d' % seed)
# ---------------------------------------------------------------------------------------------------------
# (a) linearization at U = 1
# ---------------------------------------------------------------------------------------------------------
Fc = [rand_F(1.0) for _ in range(3)]       # one F per colour
def Blin(s, mu): return tuple(AF(Fc[a], s, mu) for a in range(3))
phi_tab = {}
def phi(s):                                 # a random scalar (3 colours) gauge function, drawn on demand
    if s not in phi_tab: phi_tab[s] = tuple(random.gauss(0, 1) for _ in range(3))
    return phi_tab[s]
def Bgrad(s, mu):                           # A_F + D phi (abelian gradient, colourwise)
    b = Blin(s, mu); p1 = phi(vadd(s, e4(mu))); p0 = phi(s)
    return tuple(b[a] + p1[a] - p0[a] for a in range(3))
def lin_avg(Bf, y, mu, sg):
    Ucs = lambda s, nu: E(tuple(1j*HC*z for z in Bf(s, nu)))
    return cs(ell(avg4(Ucs, y, mu, sg)))
def Bline(Bf, y, p):
    acc = [0.0, 0.0, 0.0]
    for (o, mu, sg) in p:
        s = vadd(y, o); b = Bf(s, mu) if sg == 1 else Bf(vadd(s, e4(mu, -1)), mu)
        acc = [a + sg*z for a, z in zip(acc, b)]
    return acc
def tube(Bf, y, mu, sg):
    line = steps([(mu, sg*L)])
    return [sum(Bline(Bf, vadd(y, off), line)[a] for off in OFFS4)/L**d for a in range(3)]
def gam(Bf, y):
    return [sum(sum(Bline(Bf, y, p)[a] for p in FAM4[off])/len(FAM4[off]) for off in OFFS4)/L**d for a in range(3)]
def BF_coarse(y, mu, sg):                  # coarse linear field of curvature L^2 F in coarse units, at the coarse bond from y
    yc = tuple(v//L for v in y)            # coarse coordinates
    if sg == 1: return [L*L*0.5*sum(yc[n]*Fc[a][n][mu] for n in range(d)) for a in range(3)]
    ycm = list(yc); ycm[mu] -= 1
    return [-L*L*0.5*sum(ycm[n]*Fc[a][n][mu] for n in range(d)) for a in range(3)]
bonds = [((0, 0, 0, 0), mu, 1) for mu in range(d)] + [((3, -3, 6, 0), mu, sg) for mu in range(d) for sg in (1, -1)] + [((-6, 3, 3, -3), 2, 1)]
t0 = time.time()
e_plain = e_tube = e_gam = 0.0; e_pg = e_pg2 = 0.0; sc = 0.0; scg = 0.0
for (y, mu, sg) in bonds:
    yp = vadd(y, e4(mu, sg*L))
    pl = lin_avg(Blin, y, mu, sg); tb = tube(Blin, y, mu, sg); bf = BF_coarse(y, mu, sg)
    g0, g1 = gam(Blin, y), gam(Blin, yp)
    e_plain = max(e_plain, max(abs(a - b) for a, b in zip(pl, bf)))
    e_tube = max(e_tube, max(abs(a - b) for a, b in zip(tb, bf)))
    e_gam = max(e_gam, max(abs(v) for v in g0 + g1)); sc = max(sc, max(abs(v) for v in bf))
    # with a gauge term: plain(A_F + D phi) = B_F + phi(c+) - phi(c-);  plain = tube - (gamma(c+) - gamma(c-))
    plg = lin_avg(Bgrad, y, mu, sg); tbg = tube(Bgrad, y, mu, sg); h0, h1 = gam(Bgrad, y), gam(Bgrad, yp)
    pred = [b + p1 - p0 for b, p1, p0 in zip(bf, phi(yp), phi(y))]
    e_pg = max(e_pg, max(abs(a - b) for a, b in zip(plg, pred)))
    e_pg2 = max(e_pg2, max(abs(a - (t - (q1 - q0))) for a, t, q0, q1 in zip(plg, tbg, h0, h1)))
    scg = max(scg, max(abs(a - b) for a, b in zip(tbg, pred)))
print('\n(a) linearization at U=1 of (0.12), complex steps through the solver; %d coarse bonds (both orientations), 3 colours, F random (N(0,1) entries)' % len(bonds))
print('    plain (0.12)(A_F) - B_F                = %.1e   (scale |B_F| up to %.1f)' % (e_plain, sc))
print('    centred tube average(A_F) - B_F        = %.1e' % e_tube)
print('    gamma_sym(A_F) at c-, c+ (should be 0) = %.1e' % e_gam)
print('    with a gauge term, A_F + D phi (phi random): plain - (B_F + phi(c+) - phi(c-)) = %.1e;  plain - (tube - D\'gamma) = %.1e;'
      '  tube - (B_F + D\'phi) = %.2f (nonzero: the tube average alone is not covariant)' % (e_pg, e_pg2, scg))
# midpoint convention: B_F evaluated at the coarse bond midpoint equals the value at c- (A_mu does not depend on x_mu)
print('    [%.0f s]' % (time.time() - t0))

# ---------------------------------------------------------------------------------------------------------
# (b) nonlinear: U(1) subset SU(2), constant curvature
# ---------------------------------------------------------------------------------------------------------
def nhat_rand():
    v = [random.gauss(0, 1) for _ in range(3)]; n = math.sqrt(sum(t*t for t in v)); return tuple(t/n for t in v)
def coarse_plaq(U, y, mu, nu, cache):
    def V(z, m, s):
        k = (z, m, s)
        if k not in cache: cache[k] = avg4(U, z, m, s)
        return cache[k]
    a = V(y, mu, 1); b = V(vadd(y, e4(mu, L)), nu, 1); c = V(vadd(y, e4(nu, L)), mu, 1); dd = V(y, nu, 1)
    return mul(mul(mul(a, b), inv(c)), inv(dd))
def coarse_rect(U, y, mu, nu, cache):      # coarse 2 x 1 rectangle (2 coarse steps in mu, 1 in nu)
    def V(z, m, s):
        k = (z, m, s)
        if k not in cache: cache[k] = avg4(U, z, m, s)
        return cache[k]
    y1 = vadd(y, e4(mu, L)); y2 = vadd(y, e4(mu, 2*L)); y3 = vadd(y1, e4(nu, L)); y4 = vadd(y, e4(nu, L))
    return mul(mul(mul(mul(mul(V(y, mu, 1), V(y1, mu, 1)), V(y2, nu, 1)), inv(V(y3, mu, 1))), inv(V(y4, mu, 1))), inv(V(y, nu, 1)))
planes = [(m, n) for m in range(d) for n in range(m + 1, d)]
print('\n(b) nonlinear: U(b) = exp(i theta(b) n.sigma/2), theta = A_F (fine plaquette angle F_{mu nu}), coarse plaquette predicted exp(i L^2 F_{mu nu} n.sigma/2)')
for scale in (0.03, 0.08, 0.15):
    F = rand_F(scale); F[0][1] = 0.9*scale + F[0][1]*0.2; F[1][0] = -F[0][1]; F[2][3] = -0.7*scale + 0.2*F[2][3]; F[3][2] = -F[2][3]
    nh = nhat_rand()
    for gauge_kind in ('symmetric', 'axial', 'symmetric+Haar g'):
        t0 = time.time()
        if gauge_kind == 'axial':          # theta_mu(x) = sum_{nu < mu} x_nu F_nu,mu : same curvature, different gauge
            th = lambda s, mu: sum(s[n]*F[n][mu] for n in range(mu))
        else:
            th = lambda s, mu: AF(F, s, mu)
        Ub = lambda s, mu: E(tuple(th(s, mu)*v for v in nh))
        if gauge_kind == 'symmetric+Haar g':
            gt = {}; gf = lambda s: gt.setdefault(s, rand_su2()); U = gauged(Ub, gf)
        else: U = Ub
        cache = {}; e_pl = 0.0; e_tr = 0.0; e_rect = 0.0; e_V = 0.0; maxang = 0.0
        for y in ((0, 0, 0, 0), (3, -3, 6, 0)):
            for (m, n) in planes:
                Pc = coarse_plaq(U, y, m, n, cache); pred = E(tuple(L*L*F[m][n]*v for v in nh))
                maxang = max(maxang, abs(L*L*F[m][n]))
                e_tr = max(e_tr, abs(retr(Pc) - retr(pred)))
                if gauge_kind != 'symmetric+Haar g':
                    e_pl = max(e_pl, dist(Pc, pred))
                else:
                    g0 = gt[y] if y in gt else gf(y)
                    e_pl = max(e_pl, dist(Pc, mul(mul(g0, pred), inv(g0))))
            for (m, n) in ((0, 1), (2, 3), (1, 3)):
                R = coarse_rect(U, y, m, n, cache); pr = E(tuple(2*L*L*F[m][n]*v for v in nh))
                e_rect = max(e_rect, abs(retr(R) - retr(pr)))
            if gauge_kind == 'symmetric':   # the coarse configuration itself = exp(i B_F n.sigma/2), B_F = L A_F(c-)
                for mu in range(d):
                    V = cache[(y, mu, 1)]; e_V = max(e_V, dist(V, E(tuple(L*AF(F, y, mu)*v for v in nh))))
        extra = ('; coarse bond variables = exp(i L A_F(c-) n.sigma/2): %.1e' % e_V) if gauge_kind == 'symmetric' else ''
        print('  |F| scale %.2f (max coarse plaquette angle %.2f), %-16s: coarse plaquettes (12, 6 planes x 2 base points) %.1e,'
              ' traces %.1e; coarse 2x1 rectangles (trace vs exp(2i L^2 F)) %.1e%s  [%.0f s]'
              % (scale, maxang, gauge_kind, e_pl, e_tr, e_rect, extra, time.time() - t0), flush=True)
