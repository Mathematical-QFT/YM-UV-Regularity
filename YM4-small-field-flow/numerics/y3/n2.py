# N2 of the Y3 plan: U(1) subset SU(2) constant-flux configurations on 4d tori (side N), flux matrix n (integer, antisymmetric):
# plaquette angle theta_{mu nu} = 4 pi n_{mu nu} / N^2 in U = exp(i theta n.sigma/2) (= 2 pi n H / N^2 with H = n.sigma, e^{2 pi i H} = 1).
#   - plaquettes constant; current (gradient of S) zero; naive lattice Pfaffian sum;
#   - B_4 (about the origin): r U_F versus U_{R n R^T}, by gauge-invariant loops (plaquette traces, products of two plaquettes
#     at a common base point, Polyakov loops, Polyakov loop x plaquette); sign of the Pfaffian sum = det R.
#   python3 n2.py [seed]
import sys, math, random, itertools, time
import os
NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, NUM)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
from su2 import mul, inv, ONE, E, ell, dist, rand_su2, adv
from su2 import retr as _retr
def retr(h): return complex(_retr(h)).real
random.seed(seed)
d = 4; HC = 1e-20
EPS = [[[[0]*d for _ in range(d)] for _ in range(d)] for _ in range(d)]
for p in itertools.permutations(range(d)):
    s = 1
    for i in range(d):
        for j in range(i + 1, d):
            if p[i] > p[j]: s = -s
    EPS[p[0]][p[1]][p[2]][p[3]] = s
def pf(n): return n[0][1]*n[2][3] - n[0][2]*n[1][3] + n[0][3]*n[1][2]

class Torus:
    def __init__(self, N):
        self.N = N; self.sites = list(itertools.product(range(N), repeat=d))
    def sh(self, x, mu, s=1):
        y = list(x); y[mu] = (y[mu] + s) % self.N; return tuple(y)
    def plaq(self, U, x, mu, nu):
        return mul(mul(mul(U[(x, mu)], U[(self.sh(x, mu), nu)]), inv(U[(self.sh(x, nu), mu)])), inv(U[(x, nu)]))
    def action_local(self, U, x, mu):       # sum of 2 - tr over the 6 plaquettes containing the bond (x, mu)
        s = 0.0
        for nu in range(d):
            if nu == mu: continue
            s += 2 - _retr(self.plaq(U, x, mu, nu))
            s += 2 - _retr(self.plaq(U, self.sh(x, nu, -1), mu, nu))
        return s
def flux_config(T, n, nh):
    N = T.N; th = [[4*math.pi*n[a][b]/N**2 for b in range(d)] for a in range(d)]
    U = {}
    for x in T.sites:
        for mu in range(d):
            t = sum(th[nu][mu]*x[nu] for nu in range(mu))
            if x[mu] == N - 1: t -= N*sum(th[mu][nu]*x[nu] for nu in range(mu + 1, d))
            U[(x, mu)] = E(tuple(t*v for v in nh))
    return U, th
def act(T, r, U):
    """(rU)(r b) = U(b): r = (perm, signs), (r x)_{perm[i]} = signs[i] x_i (mod N), about the origin."""
    perm, sgn = r; out = {}
    for (x, mu), u in U.items():
        rx = [0]*d
        for i in range(d): rx[perm[i]] = (sgn[i]*x[i]) % T.N
        nu = perm[mu]; rx = tuple(rx)
        if sgn[mu] == 1: out[(rx, nu)] = u
        else: out[(T.sh(rx, nu, -1), nu)] = inv(u)
    return out
def rmat(r):
    perm, sgn = r; R = [[0]*d for _ in range(d)]
    for i in range(d): R[perm[i]][i] = sgn[i]
    return R
def detR(r):
    perm, sgn = r; s = 1
    for i in range(d):
        for j in range(i + 1, d):
            if perm[i] > perm[j]: s = -s
    for v in sgn: s *= v
    return s
def herm(h):   # Lie element ell(h) as the matrix x.sigma/2; tr(XY) = x.y/2
    return tuple(complex(z).real for z in ell(h))
def trXY(a, b): return 0.5*sum(p*q for p, q in zip(a, b))
def pf_sum(T, U):
    tot = 0.0
    for x in T.sites:
        Fx = {}
        for mu in range(d):
            for nu in range(d):
                if mu < nu: Fx[(mu, nu)] = herm(T.plaq(U, x, mu, nu))
        for mu in range(d):
            for nu in range(d):
                if mu == nu: continue
                f1 = Fx[(mu, nu)] if mu < nu else tuple(-v for v in Fx[(nu, mu)])
                for ka in range(d):
                    for la in range(d):
                        e = EPS[mu][nu][ka][la]
                        if e == 0: continue
                        f2 = Fx[(ka, la)] if ka < la else tuple(-v for v in Fx[(la, ka)])
                        tot += e*trXY(f1, f2)
    return tot
def current(T, U):
    worst = 0.0
    for x in T.sites:
        for mu in range(d):
            u0 = U[(x, mu)]
            for a in range(3):
                e = [0j, 0j, 0j]; e[a] = 1j*HC
                U[(x, mu)] = mul(E(tuple(e)), u0)
                worst = max(worst, abs(T.action_local(U, x, mu).imag/HC))
            U[(x, mu)] = u0
    return worst
def invariants(T, U):
    """Gauge-invariant data: plaquette traces; tr(P1 P2) for two plaquettes based at the same site; Polyakov loop traces;
    tr(Polyakov loop x plaquette) for loops based at the same site."""
    N = T.N; out = []
    for x in T.sites:
        Ps = [T.plaq(U, x, mu, nu) for mu in range(d) for nu in range(d) if mu != nu]
        out += [retr(p) for p in Ps]
        out += [retr(mul(Ps[i], Ps[j])) for i in range(len(Ps)) for j in range(i + 1, len(Ps))]
        for mu in range(d):
            W = ONE; y = x
            for _ in range(N): W = mul(W, U[(y, mu)]); y = T.sh(y, mu)
            out.append(retr(W))
            for nu in range(d):
                if nu != mu: out.append(retr(mul(W, T.plaq(U, x, mu, nu))))
    return out

for N, n in ((4, [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 2], [0, 0, -2, 0]]),
             (6, [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]),
             (5, [[0, 1, -1, 2], [-1, 0, 1, 1], [1, -1, 0, 1], [-2, -1, -1, 0]])):
    t0 = time.time(); T = Torus(N)
    v = [random.gauss(0, 1) for _ in range(3)]; nv = math.sqrt(sum(t*t for t in v)); nh = tuple(t/nv for t in v)
    U, th = flux_config(T, n, nh)
    print('N=%d, flux n = %s, Pf(n) = %d, U(1) direction %s' % (N, n, pf(n), ' '.join('%.3f' % z for z in nh)))
    e_pl = max(dist(T.plaq(U, x, mu, nu), E(tuple(th[mu][nu]*z for z in nh))) for x in T.sites for mu in range(d) for nu in range(d) if mu < nu)
    print('  plaquettes = exp(i 4 pi n_{mu nu}/N^2 n.sigma/2) at all %d sites, all planes: %.1e' % (len(T.sites), e_pl))
    J = current(T, U)
    Uc = dict(U); x0 = T.sites[7]; Uc[(x0, 1)] = mul(E((0.1, 0.0, 0.05)), Uc[(x0, 1)])
    print('  current (gradient of S, all bonds and colours): %.1e;  control with one bond perturbed by 0.1: %.1e' % (J, current(T, Uc)))
    q = pf_sum(T, U); pred = 8*0.5*N**4*pf(th)
    print('  naive Pfaffian sum sum_x eps tr(F F) = %.10f;  8 * (1/2) * N^4 * Pf(theta) = %.10f;  /(32 pi^2) = %.10f = 2 Pf(n)' % (q, pred, q/(32*math.pi**2)))
    # random gauge transformation leaves all of this unchanged
    g = {x: rand_su2() for x in T.sites}
    Ug = {(x, mu): mul(mul(g[x], u), inv(g[T.sh(x, mu)])) for (x, mu), u in U.items()}
    inv0 = invariants(T, U)
    print('  gauge check: invariants of U^g vs U %.1e, Pf sum %.1e' % (max(abs(a - b) for a, b in zip(invariants(T, Ug), inv0)), abs(pf_sum(T, Ug) - q)))
    # reflections and rotations
    rs = [((0, 1, 2, 3), (-1, 1, 1, 1)), ((0, 1, 2, 3), (1, 1, 1, -1)), ((1, 0, 2, 3), (1, 1, 1, 1)), ((2, 3, 0, 1), (1, 1, 1, 1)),
          ((0, 1, 2, 3), (-1, -1, -1, -1)), ((1, 2, 3, 0), (1, -1, 1, 1)), ((3, 1, 0, 2), (-1, 1, -1, 1))]
    rs += [random.choice([(p, s) for p in itertools.permutations(range(d)) for s in itertools.product((1, -1), repeat=d)]) for _ in range(3)]
    for r in rs:
        R = rmat(r); nr = [[sum(R[a][i]*n[i][j]*R[b][j] for i in range(d) for j in range(d)) for b in range(d)] for a in range(d)]
        rU = act(T, r, U); Ur, _ = flux_config(T, nr, nh)
        e_inv = max(abs(a - b) for a, b in zip(invariants(T, rU), invariants(T, Ur)))
        qr = pf_sum(T, rU)
        # control: compare r U_F with U_F itself (should differ unless r fixes n)
        e_ctrl = max(abs(a - b) for a, b in zip(invariants(T, rU), inv0))
        print('  r = perm %s signs %s (det %+d): |inv(r U_F) - inv(U_{R n R^T})| = %.1e  (vs inv(U_F): %.2f);  Pf sum(r U_F) / Pf sum(U_F) = %+.12f'
              % (r[0], r[1], detR(r), e_inv, e_ctrl, qr/q))
    print('  [%.0f s]' % (time.time() - t0), flush=True)
