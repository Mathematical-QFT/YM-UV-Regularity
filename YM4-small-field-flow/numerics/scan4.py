from run import run
import time
jobs = [(1.0,32),(0.5,48),(0.35355339059327373,64),(0.25,80),(0.17677669529663687,112),(0.125,128),(0.08838834764831843,176),(0.0625,224)]
with open('scan4.txt','a') as fo:
    for m,N in jobs:
        t=time.time(); d=4e-3
        r = run(m,N,d,1,1)
        f = (r[0]+r[1]-2*r[2])/d**2
        fo.write('%.17g %d %.12f %.3e %.1f\n'%(m,N,f,r[2],time.time()-t)); fo.flush()
