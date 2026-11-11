import subprocess, sys, math
def run(m, N, p1, ed, sym=1, nproc=8):
    cuts = [round(i*N/nproc) for i in range(nproc+1)]
    procs = [subprocess.Popen(['./vacpol','tensors.txt',str(m),str(N),repr(p1),str(ed),str(cuts[i]),str(cuts[i+1]),str(sym)],
             stdout=subprocess.PIPE, text=True) for i in range(nproc)]
    tot = [0.0]*7
    for pr in procs:
        out = pr.communicate()[0].split()
        tot = [a+float(b) for a,b in zip(tot,out)]
    return [t/N**4 for t in tot]   # Pi(+p), Pi(-p), Pi(0), parts at +p
if __name__ == '__main__':
    m, N, p1, ed, sym = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    r = run(m, N, p1, ed, sym)
    print('Pi(+p)=%.12e Pi(-p)=%.12e Pi(0)=%.12e' % tuple(r[:3]))
    print('second difference / p^2 = %.10f' % ((r[0]+r[1]-2*r[2])/p1**2))
    print('parts(+p): vec tad %.6e vec bub %.6e gh tad %.6e gh bub %.6e' % tuple(r[3:]))
