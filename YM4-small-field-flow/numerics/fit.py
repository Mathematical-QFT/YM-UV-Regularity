import math, itertools
rows = [l.split() for l in open('scan.txt')]
best = {}
for r in rows:
    m, N, f = float(r[0]), int(r[1]), float(r[2])
    if m not in best or N > best[m][0]: best[m] = (N, f)
# Richardson: f(step) = f + beta step^2; scan4.txt has step 4e-3, scan.txt step 1e-3
import os
if os.path.exists('scan4.txt'):
    four = {float(r.split()[0]): float(r.split()[2]) for r in open('scan4.txt')}
    for m in list(best):
        if m in four:
            N, f1 = best[m]; best[m] = (N, f1 + (f1 - four[m])/15)
            print('m = %.5f: step bias of f(1e-3) = %.2e' % (m, (four[m] - f1)/15))
ms = sorted(best, reverse=True)
b_th = 11*2/(24*math.pi**2*0.5)
d1 = 0.16995599*2 - 1/16
c_th = 2*d1 + 1/(12*math.pi**2)
print('b (theory) = %.6f ;  c_L predicted = 2 d1 + 1/(12 pi^2) = %.6f  (2 d1 = %.6f)' % (b_th, c_th, 2*d1))
print('   m        N    f(m)            f - b_th log(1/m)   local slope')
prev = None
for m in ms:
    N, f = best[m]
    sl = '' if prev is None else '%.5f' % ((f - prev[1])/math.log(prev[0]/m))
    print('%.5f  %4d  %.10f   %.6f          %s' % (m, N, f, f - b_th*math.log(1/m), sl)); prev = (m, f)
def solve(A, y):
    n = len(A[0]); M = [[sum(A[k][i]*A[k][j] for k in range(len(A))) for j in range(n)] + [sum(A[k][i]*y[k] for k in range(len(A)))] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i])); M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i: f = M[r][i]/M[i][i]; M[r] = [u - f*v for u, v in zip(M[r], M[i])]
    return [M[i][n]/M[i][i] for i in range(n)]
def basis(m, k):
    L = math.log(1/m); b = [L, 1, m*m*L, m*m, m**4*L, m**4]
    return b[:k]
for k in (4, 5, 6):
    for lo in range(0, len(ms) - k + 1):
        sub = ms[lo:]
        if len(sub) < k: continue
        sol = solve([basis(m, k) for m in sub], [best[m][1] for m in sub])
        print('fit %d params on m <= %.4f (%d pts): b = %.6f  c = %.6f' % (k, sub[0], len(sub), sol[0], sol[1]))
# b fixed at theory: fit c + m^2 corrections
for k, npts in ((3, 4), (3, 5), (5, 7)):
    sub = ms[-npts:]
    if len(sub) < npts: continue
    sol = solve([basis(m, k+1)[1:] for m in sub], [best[m][1] - b_th*math.log(1/m) for m in sub])
    print('b fixed, %d params on %d smallest m: c = %.6f' % (k, npts, sol[0]))
