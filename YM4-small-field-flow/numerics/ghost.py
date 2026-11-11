# Check of Cor. ol-ghost (Jacobi identity) on a 2d periodic 4x4 lattice, random SU(2) background:
#   log|det(R X : N(Q') -> W)| = log det A' + 1/2 log det M',  X = D^T D, A' = X + a Q'^T Q', M' = Q' A'^{-2} Q'^T
# with W = X N(Q') and R the orthogonal projection onto W; left side in orthonormal frames, = 1/2 log det(P X^2 P |_N).
import random, math
random.seed(7)
n = 4; sites = [(i, j) for i in range(n) for j in range(n)]; S = len(sites)
def sidx(x): return sites.index((x[0] % n, x[1] % n))
def rot(axis, th):  # adjoint of exp(i th axis.sigma/2): rotation by th about axis (Rodrigues)
    k = axis; c, s = math.cos(th), math.sin(th)
    K = [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]
    return [[(1 if a == b else 0) + s*K[a][b] + (1-c)*sum(K[a][e]*K[e][b] for e in range(3)) for b in range(3)] for a in range(3)]
def randrot():
    v = [random.gauss(0, 1) for _ in range(3)]; nv = math.sqrt(sum(t*t for t in v))
    return rot([t/nv for t in v], random.uniform(0, math.pi))
links = [(x, mu) for x in range(S) for mu in range(2)]
Rl = {l: randrot() for l in links}
nd, nl = 3*S, 3*len(links)
D = [[0.0]*nd for _ in range(nl)]
for li, (x, mu) in enumerate(links):
    xs = sites[x]; y = sidx((xs[0] + (mu == 0), xs[1] + (mu == 1)))
    for a in range(3):
        D[3*li+a][3*x+a] -= 1
        for b in range(3): D[3*li+a][3*y+b] += Rl[(x, mu)][a][b]
def T(M): return [list(r) for r in zip(*M)]
def mm(A, B):
    Bt = T(B); return [[sum(a*b for a, b in zip(r, c)) for c in Bt] for r in A]
def logdet(M):
    A = [r[:] for r in M]; N = len(A); s = 0.0
    for i in range(N):
        p = max(range(i, N), key=lambda r: abs(A[r][i])); A[i], A[p] = A[p], A[i]
        s += math.log(abs(A[i][i]))
        for r in range(i+1, N):
            f = A[r][i]/A[i][i]
            if f: A[r] = [u - f*v for u, v in zip(A[r], A[i])]
    return s
def inv(M):
    N = len(M); A = [r[:] + [1.0 if i == j else 0.0 for j in range(N)] for i, r in enumerate(M)]
    for i in range(N):
        p = max(range(i, N), key=lambda r: abs(A[r][i])); A[i], A[p] = A[p], A[i]
        piv = A[i][i]; A[i] = [v/piv for v in A[i]]
        for r in range(N):
            if r != i and A[r][i]:
                f = A[r][i]; A[r] = [u - f*v for u, v in zip(A[r], A[i])]
    return [r[N:] for r in A]
X = mm(T(D), D)
# Q': block average over 2x2 blocks, scaled so that Q'Q'^T = I  (Jacobi needs only orthonormal rows)
blocks = [(i, j) for i in range(0, n, 2) for j in range(0, n, 2)]
Q = [[0.0]*nd for _ in range(3*len(blocks))]
for bi, (i, j) in enumerate(blocks):
    for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        x = sidx((i+di, j+dj))
        for a in range(3): Q[3*bi+a][3*x+a] = 0.5
# orthonormal basis of N(Q') by Gram-Schmidt of (1 - Q'^T Q') e_i
P = [[(1.0 if i == j else 0.0) - sum(Q[r][i]*Q[r][j] for r in range(len(Q))) for j in range(nd)] for i in range(nd)]
basis = []
for i in range(nd):
    v = P[i][:]
    for b in basis:
        c = sum(s*t for s, t in zip(v, b)); v = [s - c*t for s, t in zip(v, b)]
    nv = math.sqrt(sum(t*t for t in v))
    if nv > 1e-8: basis.append([t/nv for t in v])
print('dim N(Q\') =', len(basis), ' (expected', nd - len(Q), ')')
Nb = T(basis)                         # nd x dimN
XN = mm(X, Nb)
lhs = 0.5*logdet(mm(T(XN), XN))        # 1/2 log det(K^*K), K^*K = P X^2 P on N(Q')
for a in (0.3, 1.0, 4.0):
    Ap = [[X[i][j] + a*sum(Q[r][i]*Q[r][j] for r in range(len(Q))) for j in range(nd)] for i in range(nd)]
    G = inv(Ap); G2 = mm(G, G)
    M = mm(mm(Q, G2), T(Q))
    rhs = logdet(Ap) + 0.5*logdet(M)
    print('a=%.1f  lhs=%.12f  rhs=%.12f  diff=%.2e' % (a, lhs, rhs, lhs - rhs))
