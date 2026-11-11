# Sanity check of the algebra in Lemma later-normal-form (pure Python, SL(2,C)).
import random, cmath, itertools
random.seed(1)
D=4
def mz(): return [[0j,0j],[0j,0j]]
def eye(): return [[1+0j,0j],[0j,1+0j]]
def add(a,b): return [[a[i][j]+b[i][j] for j in range(2)] for i in range(2)]
def sub(a,b): return [[a[i][j]-b[i][j] for j in range(2)] for i in range(2)]
def sc(c,a): return [[c*a[i][j] for j in range(2)] for i in range(2)]
def mul(a,b): return [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def nrm(a): return sum(abs(a[i][j])**2 for i in range(2) for j in range(2))**0.5
def comm(a,b): return sub(mul(a,b),mul(b,a))
def expm(a):
    # exp(a) by scaling and squaring + Taylor
    s=0; n=nrm(a)
    while n>0.1: n/=2; s+=1
    a=sc(1/2**s,a); r=eye(); t=eye()
    for k in range(1,25):
        t=sc(1/k,mul(t,a)); r=add(r,t)
    for _ in range(s): r=mul(r,r)
    return r
def logm(h):
    x=sub(h,eye()); r=mz(); p=eye()
    assert nrm(x)<0.5, nrm(x)
    for n in range(1,80):
        p=mul(p,x); r=add(r,sc((-1)**(n+1)/n,p))
    return r
def ell(h): return sc(-1j,logm(h))          # e^{i ell(h)} = h
def E(X): return expm(sc(1j,X))               # e^{iX}
# basis of sl(2,C): Pauli/2
s1=[[0,0.5],[0.5,0]]; s2=[[0,-0.5j],[0.5j,0]]; s3=[[0.5,0],[0,-0.5]]
def rnd(amp):
    c=[complex(random.gauss(0,1),random.gauss(0,1))*amp for _ in range(3)]
    return add(add(sc(c[0],s1),sc(c[1],s2)),sc(c[2],s3))
eta=1/3.; r=3   # cube radius in lattice steps
eps=0.05
pts=list(itertools.product(range(-r,r+1),repeat=D))
def e(mu): return tuple(1 if i==mu else 0 for i in range(D))
def plus(x,y): return tuple(a+b for a,b in zip(x,y))
inside=lambda x: all(abs(c)<=r for c in x)
# smooth complex field A0_mu(x) = c_mu + sum_nu G[nu][mu] x_nu + sum Q x x  (x in physical units = eta*n)
c=[rnd(eps) for mu in range(D)]
G=[[rnd(eps**2) for mu in range(D)] for nu in range(D)]
Q=[[[rnd(eps**3) for mu in range(D)] for nu in range(D)] for rho in range(D)]
def A0(mu,n):
    x=[eta*k for k in n]
    v=c[mu]
    for nu in range(D): v=add(v,sc(x[nu],G[nu][mu]))
    for nu in range(D):
        for rho in range(D): v=add(v,sc(x[nu]*x[rho],Q[nu][rho][mu]))
    return v
U={}
for n in pts:
    for mu in range(D):
        m=plus(n,e(mu))
        if inside(m): U[(n,mu)]=E(sc(eta,A0(mu,n)))
O=tuple([0]*D)
a=[A0(mu,O) for mu in range(D)]
def lam1(n): 
    v=mz()
    for mu in range(D): v=add(v,sc(-eta*n[mu],a[mu]))
    return v
def gauge(U,lam):
    V={}
    for (n,mu),h in U.items():
        m=plus(n,e(mu))
        V[(n,mu)]=mul(mul(E(sc(-1,lam(n))),h),E(lam(m)))
    return V
U1=gauge(U,lam1)
B={k:sc(1/eta,ell(h)) for k,h in U1.items()}
print("max |B(x_X,.)| =",max(nrm(B[(O,mu)]) for mu in range(D)))
dB=[[sc(1/eta,sub(B[(e(mu),nu)],B[(O,nu)])) for nu in range(D)] for mu in range(D)]
S=[[sc(0.5,add(dB[mu][nu],dB[nu][mu])) for nu in range(D)] for mu in range(D)]
def lam2(n):
    y=[eta*k for k in n]; v=mz()
    for mu in range(D):
        for rho in range(D): v=add(v,sc(-0.5*y[mu]*y[rho],S[mu][rho]))
        v=add(v,sc(0.5*eta*y[mu],S[mu][mu]))
    return v
print("lam2 checks:",max(nrm(lam2(e(mu))) for mu in range(D)),
      max(nrm(add(lam2(plus(e(mu),e(nu))),sc(eta**2,S[mu][nu]))) for mu in range(D) for nu in range(D)))
W=gauge(U1,lam2)
A={k:sc(1/eta,ell(h)) for k,h in W.items()}
print("max |A(x_X,.)| =",max(nrm(A[(O,mu)]) for mu in range(D)))
dA=[[sc(1/eta,sub(A[(e(mu),nu)],A[(O,nu)])) for nu in range(D)] for mu in range(D)]
def curv(U,n,mu,nu):
    h=mul(mul(U[(n,mu)],U[(plus(n,e(mu)),nu)]),mul(expm(sc(-1,logm(U[(plus(n,e(nu)),mu)]))),expm(sc(-1,logm(U[(n,nu)])))))
    return sc(1/eta**2,ell(h))
FX=[[mz() for nu in range(D)] for mu in range(D)]
for mu in range(D):
    for nu in range(D):
        if mu<nu:
            FX[mu][nu]=curv(W,O,mu,nu); FX[nu][mu]=sc(-1,FX[mu][nu])
FU=[[curv(U,O,mu,nu) if mu<nu else None for nu in range(D)] for mu in range(D)]
print("covariance |F_X - F_U(x_X)| =",max(nrm(sub(FX[mu][nu],FU[mu][nu])) for mu in range(D) for nu in range(D) if mu<nu))
dBn=max(nrm(dB[mu][nu]) for mu in range(D) for nu in range(D))
rho=max(nrm(sub(dA[mu][nu],sc(0.5,FX[mu][nu]))) for mu in range(D) for nu in range(D))
print("|dB| =",dBn,"  |rho| = |dA(x_X)-F_X/2| =",rho,"  ratio rho/(eta^2|dB|^2) =",rho/(eta**2*dBn**2))
# remark (b): dB(x_X) - dA0(x_X) vs (i/2)[a_mu,a_nu] (leading order)
dA0=[[sc(1/eta,sub(A0(nu,e(mu)),A0(nu,O))) for nu in range(D)] for mu in range(D)]
dev=max(nrm(sub(sub(dB[mu][nu],dA0[mu][nu]),sc(0.5j,comm(a[mu],a[nu])))) for mu in range(D) for nu in range(D) if mu!=nu)
ref=max(nrm(sc(0.5j,comm(a[mu],a[nu]))) for mu in range(D) for nu in range(D) if mu!=nu)
print("remark(b): |dB-dA0-(i/2)[a,a]| =",dev," vs |(i/2)[a,a]| =",ref)
# antisymmetric part of dB vs curvature of U at x_X
dev2=max(nrm(sub(sub(dB[mu][nu],dB[nu][mu]),FU[mu][nu])) for mu in range(D) for nu in range(D) if mu<nu)
print("|(dB-dB^T) - F_U(x_X)| =",dev2)
# remainder R = A - A_lin[F_X] on the cube: sup and compare with A_lin
def Alin(nu,n):
    v=mz()
    for mu in range(D): v=add(v,sc(0.5*eta*n[mu],FX[mu][nu]))
    return v
Rmax=max(nrm(sub(A[(n,nu)],Alin(nu,n))) for (n,nu) in A)
Lmax=max(nrm(Alin(nu,n)) for (n,nu) in A)
print("sup|A_lin| =",Lmax," sup|R| =",Rmax)
