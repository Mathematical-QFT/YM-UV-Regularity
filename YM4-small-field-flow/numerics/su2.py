# SU(2) holomorphic primitives (as in chart.py) and Balaban's group average M (CMP 109, (0.5)-(0.10)).
# h = E(x) = exp(i x.sigma/2) as a quaternion (h0, hv), ell = E^{-1} (principal branch), R(h) = Ad(h) on components.
# Real arithmetic for real input; complex input gives the holomorphic extension (used for complex-step derivatives).
import cmath, math, random

FORCE_ITER = False      # True: solve (0.10) iteratively also for two-element families (instead of the midpoint)

def mul(p, q):
    return (p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
            p[0]*q[1] + q[0]*p[1] - (p[2]*q[3] - p[3]*q[2]),
            p[0]*q[2] + q[0]*p[2] - (p[3]*q[1] - p[1]*q[3]),
            p[0]*q[3] + q[0]*p[3] - (p[1]*q[2] - p[2]*q[1]))
def inv(p): return (p[0], -p[1], -p[2], -p[3])
ONE = (1.0, 0.0, 0.0, 0.0)
def E(x):
    th2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
    if abs(th2) < 1e-12:
        c = 1 - th2/8 + th2*th2/384; s = 0.5 - th2/48 + th2*th2/3840
    else:
        th = cmath.sqrt(th2); c = cmath.cos(th/2); s = cmath.sin(th/2)/th
    return (c, s*x[0], s*x[1], s*x[2])
def ell(h):
    u = 1 - h[0]
    if abs(u) < 1e-5: f = 1 + u/3 + 2*u*u/15 + 2*u**3/35
    else: f = cmath.acos(h[0])/cmath.sqrt(1 - h[0]*h[0])
    return (2*f*h[1], 2*f*h[2], 2*f*h[3])
def adj(h):
    hi = inv(h); R = [[0.0]*3 for _ in range(3)]
    for b in range(3):
        e = [0.0, 0.0, 0.0, 0.0]; e[b+1] = 1.0
        m = mul(mul(h, tuple(e)), hi)
        for a in range(3): R[a][b] = m[a+1]
    return R
def adv(h, v):   # Ad(h) v
    m = mul(mul(h, (0.0, v[0], v[1], v[2])), inv(h)); return (m[1], m[2], m[3])
def retr(h): return 2*h[0]
def dist(a, b): return max(abs(z) for z in ell(mul(a, inv(b))))
def rand_su2(amp=None):   # amp None: Haar-distributed
    if amp is None:
        v = [random.gauss(0, 1) for _ in range(4)]; n = math.sqrt(sum(t*t for t in v)); return tuple(t/n for t in v)
    return E(tuple(random.gauss(0, amp) for _ in range(3)))
def real(h): return tuple(complex(z).real for z in h)

def gavg_iter(Us, tol=1e-13, maxit=200):
    """The solution U near Us[0] of sum_j ell(U_j U^{-1}) = 0, by U <- E(mean_j ell(U_j U^{-1})) U
    (a contraction for families of small diameter); two further steps after the mean drops below tol."""
    U = Us[0]; n = len(Us); extra = 0
    for _ in range(maxit):
        s = [0j, 0j, 0j]
        for Uj in Us:
            v = ell(mul(Uj, inv(U)))
            for a in range(3): s[a] += v[a]/n
        U = mul(E(tuple(s)), U)
        if max(abs(v) for v in s) < tol:
            extra += 1
            if extra == 3: return U
    raise RuntimeError('group average: no convergence')
def gavg(Us):
    if len(Us) == 1: return Us[0]
    if len(Us) == 2 and not FORCE_ITER: return mul(E(tuple(0.5*v for v in ell(mul(Us[1], inv(Us[0]))))), Us[0])   # geodesic midpoint
    return gavg_iter(Us)
