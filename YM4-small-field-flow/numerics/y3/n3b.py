# N3 supplement: the families of n3.py evaluated on constant-flux tori. Pieces are quadratic forms in the plaquette
# variables; on a configuration with flux we insert the (gauge-invariant) principal plaquette angles. Periodic fields see
# no Pfaffian density (n3.py: the null family sums to 0); constant-flux configurations do, with a sign flip under reflections.
#   python3 n3b.py [seed]     (same seed as n3.py gives the same families)
import sys, math, itertools, random
sys.argv = [sys.argv[0]] + sys.argv[1:]
# rebuild the families exactly as n3.py does, without running its reports: copy its definitions up to the families
src = open(__file__.replace('n3b.py', 'n3.py')).read()
head = src[:src.index("report('R  random family")]
ns = {'__file__': __file__}
exec(compile(head, 'n3-head', 'exec'), ns)
famS, famSp, famN, tau_n, r_point, z, M, PAIRS, d = ns['famS'], ns['famSp'], ns['famN'], ns['tau_n'], ns['r_point'], ns['z'], ns['M'], ns['PAIRS'], ns['d']
B4, detr = ns['B4'], ns['detr']

def flux_angles(N, n):
    """principal plaquette angles of the U(1) constant-flux configuration: theta_{mu nu} = 4 pi n_{mu nu} / N^2 everywhere"""
    th = {}
    for (a, b) in PAIRS: th[(a, b)] = 4*math.pi*n[a][b]/N**2
    return th
def total_on_flux(pieces, N, th_of_plaq):
    s = 0.0
    for a in itertools.product(range(0, N, M), repeat=d):
        for q in pieces:
            for (c, p1, p2) in q:
                s += c*th_of_plaq(p1, a)*th_of_plaq(p2, a)
    return s
def rot_flux(r, n):
    perm, sgn = r; R = [[0]*d for _ in range(d)]
    for i in range(d): R[perm[i]][i] = sgn[i]
    return [[sum(R[a][i]*n[i][j]*R[b][j] for i in range(d) for j in range(d)) for b in range(d)] for a in range(d)]
N = 4
n = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]
print('Families of n3.py on the constant-flux torus N=%d, flux n12 = n34 = 1 (Pf = 1); r U_F has flux R n R^T (N2)' % N)
th = flux_angles(N, n)
Q = {}
for name, fam in (('S', famS), ('N', famN), ('S+N', famS + famN), ('S+', famSp)):
    Q[name] = total_on_flux(fam, N, lambda p, a: th[(p[1], p[2])])
for r in [((0, 1, 2, 3), (-1, 1, 1, 1)), ((1, 0, 2, 3), (1, 1, 1, 1)), ((0, 1, 2, 3), (-1, -1, -1, -1)), ((1, 2, 3, 0), (1, -1, 1, 1))]:
    nr = rot_flux(r, n); thr = flux_angles(N, nr)
    out = []
    for name, fam in (('S', famS), ('N', famN), ('S+N', famS + famN), ('S+', famSp)):
        Qr = total_on_flux(fam, N, lambda p, a: thr[(p[1], p[2])])
        out.append('%s: %+.6f -> %+.6f' % (name, Q[name], Qr))
    print('  r = perm %s signs %s (det %+d):  %s' % (r[0], r[1], detr(r), ';  '.join(out)))
print('  predicted for N: total = 8 tau Pf(theta) N^4 = %.6f' % (8*tau_n*(th[(0, 1)]*th[(2, 3)] - th[(0, 2)]*th[(1, 3)] + th[(0, 3)]*th[(1, 2)])*N**4))
