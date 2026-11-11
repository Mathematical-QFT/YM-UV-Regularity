# Local Taylor tensors of the SU(2) Wilson plaquette action in the chart e^{ih}e^{iA T3}.
# F(h_1..h_4, A_1..A_4) = 2 - 1/2 (tr P + tr P^{-1}),  P = V1 V2 V3^{-1} V4^{-1},  V_l = E(h_l) E(A_l e3)
# E(x) = exp(i x.sigma/2).  Coefficients by Cauchy's formula on the holomorphic extension.
import cmath, math, itertools, sys

def E(x):
    x1,x2,x3 = x
    th2 = x1*x1+x2*x2+x3*x3
    if abs(th2) < 1e-30:
        c = 1.0; s = 0.5
    else:
        th = cmath.sqrt(th2)
        c = cmath.cos(th/2); s = cmath.sin(th/2)/th
    # c*I + i*s*(x.sigma);  x.sigma = [[x3, x1-i x2],[x1+i x2, -x3]]
    return ((c+1j*s*x3, 1j*s*(x1-1j*x2)), (1j*s*(x1+1j*x2), c-1j*s*x3))

def mul(a,b):
    return ((a[0][0]*b[0][0]+a[0][1]*b[1][0], a[0][0]*b[0][1]+a[0][1]*b[1][1]),
            (a[1][0]*b[0][0]+a[1][1]*b[1][0], a[1][0]*b[0][1]+a[1][1]*b[1][1]))

def neg(x): return (-x[0],-x[1],-x[2])

def V(h,A):      return mul(E(h), E((0,0,A)))
def Vinv(h,A):   return mul(E((0,0,-A)), E(neg(h)))

def F(h, A):
    P  = mul(mul(mul(V(h[0],A[0]), V(h[1],A[1])), Vinv(h[2],A[2])), Vinv(h[3],A[3]))
    Pi = mul(mul(mul(V(h[3],A[3]), V(h[2],A[2])), Vinv(h[1],A[1])), Vinv(h[0],A[0]))
    return 2 - 0.5*(P[0][0]+P[1][1] + Pi[0][0]+Pi[1][1])

def evalF(hdirs, adirs, svals, uvals):
    h = [[0j,0j,0j] for _ in range(4)]
    A = [0j]*4
    for (l,a),s in zip(hdirs,svals): h[l][a] += s
    for k,u in zip(adirs,uvals): A[k] += u
    return F([tuple(v) for v in h], A)

def coeff(hdirs, adirs, M, r):
    """mixed partial derivative d/dh_{hdirs} d/dA_{adirs} at 0 (each variable once)"""
    nv = len(hdirs)+len(adirs)
    w = [cmath.exp(2j*math.pi*j/M) for j in range(M)]
    tot = 0j
    for js in itertools.product(range(M), repeat=nv):
        vals = [r*w[j] for j in js]
        f = evalF(hdirs, adirs, vals[:len(hdirs)], vals[len(hdirs):])
        tot += f * cmath.exp(-2j*math.pi*sum(js)/M)
    return (tot/(M**nv)/(r**nv)).real

idx = [(l,a) for l in range(4) for a in range(3)]   # 12 h-variables
out = sys.argv[1]
M, r = 6, 0.03
H0 = {}; H1 = {}; H2 = {}; S2 = {}
for i in range(12):
    for j in range(i,12):
        H0[(i,j)] = coeff([idx[i],idx[j]], [], M, r)
        for k in range(4):
            H1[(i,j,k)] = coeff([idx[i],idx[j]], [k], M, r)
            for kk in range(k,4):
                H2[(i,j,k,kk)] = coeff([idx[i],idx[j]], [k,kk], M, r)
for k in range(4):
    for kk in range(k,4):
        S2[(k,kk)] = coeff([], [k,kk], 8, 0.05)
with open(out,'w') as fo:
    for (i,j),v in H0.items(): fo.write('H0 %d %d %.15e\n'%(i,j,v))
    for (i,j,k),v in H1.items(): fo.write('H1 %d %d %d %.15e\n'%(i,j,k,v))
    for (i,j,k,kk),v in H2.items(): fo.write('H2 %d %d %d %d %.15e\n'%(i,j,k,kk,v))
    for (k,kk),v in S2.items(): fo.write('S2 %d %d %.15e\n'%(k,kk,v))
print('done')
