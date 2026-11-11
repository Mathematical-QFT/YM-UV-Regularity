# Dimension of B_4- (and B_4^+-) invariant quadratic forms on Lambda^2 R^4.
import itertools
pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
idx={p:i for i,p in enumerate(pairs)}
def act(perm,signs):
    # matrix of the induced action on Lambda^2: e_mu^e_nu -> (s_mu e_perm(mu)) ^ (s_nu e_perm(nu))
    M=[[0]*6 for _ in range(6)]
    for (m,n),i in idx.items():
        a,b=perm[m],perm[n]; s=signs[m]*signs[n]
        if a<b: M[idx[(a,b)]][i]+=s
        else: M[idx[(b,a)]][i]-=s
    return M
def det(perm,signs):
    inv=sum(1 for i in range(4) for j in range(i+1,4) if perm[i]>perm[j])
    s=(-1)**inv
    for x in signs: s*=x
    return s
group=[(p,s) for p in itertools.permutations(range(4)) for s in itertools.product([1,-1],repeat=4)]
def inv_dim(G):
    # basis of symmetric forms: E_ij+E_ji, i<=j ; average R^T Q R over G; rank of the averaging map
    basis=[]
    for i in range(6):
        for j in range(i,6):
            Q=[[0.0]*6 for _ in range(6)]; Q[i][j]+=1; Q[j][i]+=1 if i!=j else 0
            basis.append(Q)
    vecs=[]
    for Q in basis:
        A=[[0.0]*6 for _ in range(6)]
        for (p,s) in G:
            R=act(p,s)
            # R^T Q R
            QR=[[sum(Q[a][c]*R[c][b] for c in range(6)) for b in range(6)] for a in range(6)]
            for a in range(6):
                for b in range(6):
                    A[a][b]+=sum(R[c][a]*QR[c][b] for c in range(6))/len(G)
        vecs.append([A[a][b] for a in range(6) for b in range(a,6)])
    # rank by Gaussian elimination
    M=[v[:] for v in vecs]; r=0; ncol=len(M[0])
    for c in range(ncol):
        piv=None
        for i in range(r,len(M)):
            if abs(M[i][c])>1e-9: piv=i;break
        if piv is None: continue
        M[r],M[piv]=M[piv],M[r]
        for i in range(len(M)):
            if i!=r and abs(M[i][c])>1e-12:
                f=M[i][c]/M[r][c]
                M[i]=[x-f*y for x,y in zip(M[i],M[r])]
        r+=1
    return r, vecs
G=group; Gp=[g for g in group if det(*g)==1]
r,_=inv_dim(G); rp,vp=inv_dim(Gp)
print("|B4|=",len(G)," invariant forms:",r)
print("|B4+|=",len(Gp)," invariant forms:",rp)
# Pfaffian form F12F34 - F13F24 + F14F23 in basis order (01,02,03,12,13,23): pairs (01,23):+ , (02,13):- , (03,12):+
