/* One-loop vacuum polarization of SU(2) Wilson lattice gauge theory on Z^4,
   background Feynman gauge with mass m, background A^3 (Cartan) in the chart e^{ih} e^{iA}.
   Gamma = -1/2 log det(Hess_W + 1/2 D D^T + 1/2 m^2) + log det(D^T D + m^2)   [component pairing]
   Pi(p) = second variation coefficient of a abar for A_mu(x) = a eps_mu e^{ipx} + c.c.
   usage: vacpol tensors.txt m N p1 epsdir j1lo j1hi sym   -> prints partial sums over q1-index slice
          vacpol tensors.txt test                          -> self-checks                                  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
typedef double complex cplx;

static double H0[12][12], H1[12][12][4], H2[12][12][4][4];
static const double J[3][3]  = {{0,1,0},{-1,0,0},{0,0,0}};
static const double J2[3][3] = {{-1,0,0},{0,-1,0},{0,0,0}};
static int PL[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

static void read_tensors(const char *fn) {
    FILE *f = fopen(fn, "r"); if (!f) { perror(fn); exit(1); }
    char tag[8]; int i, j, k, kk; double v;
    while (fscanf(f, "%7s", tag) == 1) {
        if (!strcmp(tag, "H0")) { fscanf(f, "%d %d %lf", &i, &j, &v); H0[i][j] = H0[j][i] = v; }
        else if (!strcmp(tag, "H1")) { fscanf(f, "%d %d %d %lf", &i, &j, &k, &v); H1[i][j][k] = H1[j][i][k] = v; }
        else if (!strcmp(tag, "H2")) { fscanf(f, "%d %d %d %d %lf", &i, &j, &k, &kk, &v);
            H2[i][j][k][kk] = H2[j][i][k][kk] = H2[i][j][kk][k] = H2[j][i][kk][k] = v; }
        else if (!strcmp(tag, "S2")) { fscanf(f, "%d %d %lf", &k, &kk, &v); }
    }
    fclose(f);
    /* clean numerical zeros */
    for (i = 0; i < 12; i++) for (j = 0; j < 12; j++) {
        if (fabs(H0[i][j]) < 1e-9) H0[i][j] = 0;
        for (k = 0; k < 4; k++) { if (fabs(H1[i][j][k]) < 1e-9) H1[i][j][k] = 0;
            for (kk = 0; kk < 4; kk++) if (fabs(H2[i][j][k][kk]) < 1e-9) H2[i][j][k][kk] = 0; }
    }
}

/* q . x_l for the four links of plane (mu,nu): x = 0, e_mu, e_nu, 0 */
static double qx(const double *q, int l, int mu, int nu) {
    return l == 1 ? q[mu] : (l == 2 ? q[nu] : 0.0);
}
static double hat2(const double *q) { double s = 0; for (int i = 0; i < 4; i++) s += 4*sin(q[i]/2)*sin(q[i]/2); return s; }

/* Wilson vertex X1(q+p <- q) and tadpole trace; with order 0 also W0(q) */
static void wilson(const double *q, const double *p, const double *eps, cplx X[12][12], cplx *trT, cplx W0[12][12]) {
    double qp[4]; for (int i = 0; i < 4; i++) qp[i] = q[i] + p[i];
    memset(X, 0, sizeof(cplx)*144); *trT = 0;
    if (W0) memset(W0, 0, sizeof(cplx)*144);
    for (int pl = 0; pl < 6; pl++) {
        int mu = PL[pl][0], nu = PL[pl][1];
        int dirs[4] = {mu, nu, mu, nu};
        cplx ak[4], Eq[4], Eqp[4]; int any = 0;
        for (int k = 0; k < 4; k++) { ak[k] = eps[dirs[k]] * cexp(I*qx(p, k, mu, nu)); if (eps[dirs[k]] != 0) any = 1;
            Eq[k] = cexp(I*qx(q, k, mu, nu)); Eqp[k] = cexp(I*qx(qp, k, mu, nu)); }
        for (int l = 0; l < 4; l++) for (int lp = 0; lp < 4; lp++) {
            if (W0) { cplx ph0 = cexp(-I*qx(q, l, mu, nu) + I*qx(q, lp, mu, nu));
                for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++)
                    W0[3*dirs[l]+a][3*dirs[lp]+b] += ph0 * H0[3*l+a][3*lp+b]; }
            if (!any) continue;
            cplx ph = conj(Eqp[l]) * Eq[lp];
            for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
                cplx v = 0;
                for (int k = 0; k < 4; k++) if (H1[3*l+a][3*lp+b][k] != 0) v += H1[3*l+a][3*lp+b][k] * ak[k];
                X[3*dirs[l]+a][3*dirs[lp]+b] += ph * v;
            }
            if (dirs[l] == dirs[lp]) {
                cplx ph2 = conj(Eq[l]) * Eq[lp];
                for (int a = 0; a < 3; a++) {
                    cplx v = 0;
                    for (int k = 0; k < 4; k++) for (int kk = 0; kk < 4; kk++)
                        if (H2[3*l+a][3*lp+a][k][kk] != 0) v += H2[3*l+a][3*lp+a][k][kk] * ak[k] * conj(ak[kk]);
                    *trT += ph2 * v;
                }
            }
        }
    }
}

/* D^T (1-forms -> scalars) symbols, 3x12:  B0(q), B1[u](q+p<-q), B1[ub](q<-q+p), B1[ub](q-p<-q), B2(q) */
static void Bsym(const double *q, const double *p, const double *eps,
                 cplx B0[3][12], cplx B1u[3][12], cplx B1up[3][12], cplx B1dn[3][12], cplx B2[3][12]) {
    memset(B0, 0, sizeof(cplx)*36); memset(B1u, 0, sizeof(cplx)*36); memset(B1up, 0, sizeof(cplx)*36);
    memset(B1dn, 0, sizeof(cplx)*36); memset(B2, 0, sizeof(cplx)*36);
    for (int mu = 0; mu < 4; mu++) {
        cplx eq = cexp(-I*q[mu]), eqp = cexp(-I*(q[mu]+p[mu])), eqm = cexp(-I*(q[mu]-p[mu]));
        for (int c = 0; c < 3; c++) {
        B0[c][3*mu+c] = eq - 1;
        if (eps[mu] == 0) continue;
        for (int b = 0; b < 3; b++) {
            B1u[c][3*mu+b]  = -eps[mu] * eqp * J[c][b];
            B1up[c][3*mu+b] = -eps[mu] * eq * J[c][b];
            B1dn[c][3*mu+b] = -eps[mu] * eqm * J[c][b];
            B2[c][3*mu+b]   = eps[mu]*eps[mu] * eq * J2[c][b];
        }
    }}
}
/* D (scalars -> 1-forms) symbols, 12x3 */
static void Csym(const double *q, const double *p, const double *eps,
                 cplx C0[12][3], cplx C1u[12][3], cplx C1up[12][3], cplx C1dn[12][3], cplx C2[12][3]) {
    memset(C0, 0, sizeof(cplx)*36); memset(C1u, 0, sizeof(cplx)*36); memset(C1up, 0, sizeof(cplx)*36);
    memset(C1dn, 0, sizeof(cplx)*36); memset(C2, 0, sizeof(cplx)*36);
    for (int mu = 0; mu < 4; mu++) {
        cplx eq = cexp(I*q[mu]), eqp = cexp(I*(q[mu]+p[mu]));
        for (int b = 0; b < 3; b++) {
        C0[3*mu+b][b] = eq - 1;
        if (eps[mu] == 0) continue;
        for (int c = 0; c < 3; c++) {
            C1u[3*mu+b][c]  = eps[mu] * eq * J[b][c];
            C1up[3*mu+b][c] = eps[mu] * eqp * J[b][c];
            C1dn[3*mu+b][c] = eps[mu] * eq * J[b][c];
            C2[3*mu+b][c]   = eps[mu]*eps[mu] * eq * J2[b][c];
        }
    }}
}

/* integrand P(q,p) = -1/2 [vec tadpole - vec bubble] + [gh tadpole - gh bubble]; parts returned in out[4] */
static double integrand(const double *q, const double *p, const double *eps, double m, double out[4]) {
    double qp[4]; for (int i = 0; i < 4; i++) qp[i] = q[i] + p[i];
    double gv = 2/(hat2(q)+m*m), gvp = 2/(hat2(qp)+m*m);
    double gg = 1/(hat2(q)+m*m), ggp = 1/(hat2(qp)+m*m);
    cplx X[12][12], trT;
    wilson(q, p, eps, X, &trT, NULL);
    cplx B0q[3][12], B1u[3][12], B1up[3][12], B1dn[3][12], B2[3][12], B0qp[3][12], d1[3][12], d2[3][12], d3[3][12], d4[3][12];
    Bsym(q, p, eps, B0q, B1u, B1up, B1dn, B2);
    Bsym(qp, p, eps, B0qp, d1, d2, d3, d4);
    /* X1_gf = 1/2 [B0(q+p)^+ B1u + B1up^+ B0(q)] */
    for (int i = 0; i < 12; i++) for (int j = 0; j < 12; j++) {
        cplx s = 0;
        for (int c = 0; c < 3; c++) s += conj(B0qp[c][i]) * B1u[c][j] + conj(B1up[c][i]) * B0q[c][j];
        X[i][j] += 0.5*s;
    }
    cplx trTgf = 0;
    for (int c = 0; c < 3; c++) for (int i = 0; i < 12; i++)
        trTgf += 0.5*(conj(B1u[c][i])*B1u[c][i] + conj(B1dn[c][i])*B1dn[c][i]
                      + conj(B0q[c][i])*B2[c][i] + conj(B2[c][i])*B0q[c][i]);
    double bub = 0; for (int i = 0; i < 12; i++) for (int j = 0; j < 12; j++) bub += creal(X[i][j]*conj(X[i][j]));
    double Pvec = gv*creal(trT + trTgf) - gvp*gv*bub;
    /* ghost */
    cplx C0q[12][3], C1u[12][3], C1up[12][3], C1dn[12][3], C2[12][3], C0qp[12][3], e1[12][3], e2[12][3], e3[12][3], e4[12][3];
    Csym(q, p, eps, C0q, C1u, C1up, C1dn, C2);
    Csym(qp, p, eps, C0qp, e1, e2, e3, e4);
    cplx M1[3][3]; double gbub = 0;
    for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) {
        cplx s = 0;
        for (int i = 0; i < 12; i++) s += conj(C0qp[i][a]) * C1u[i][c] + conj(C1up[i][a]) * C0q[i][c];
        M1[a][c] = s; gbub += creal(s*conj(s));
    }
    cplx trM2 = 0;
    for (int i = 0; i < 12; i++) for (int c = 0; c < 3; c++)
        trM2 += conj(C1u[i][c])*C1u[i][c] + conj(C1dn[i][c])*C1dn[i][c] + conj(C0q[i][c])*C2[i][c] + conj(C2[i][c])*C0q[i][c];
    double Pgh = gg*creal(trM2) - ggp*gg*gbub;
    if (out) { out[0] = gv*creal(trT+trTgf); out[1] = gvp*gv*bub; out[2] = gg*creal(trM2); out[3] = ggp*gg*gbub; }
    (void)M1;
    return -0.5*Pvec + Pgh;
}

static void selftest(void) {
    double q[4] = {0.3, -1.1, 2.0, 0.7}, p[4] = {0, 0, 0, 0}, e2[4] = {0, 1, 0, 0};
    cplx X[12][12], W0[12][12], trT;
    wilson(q, p, e2, X, &trT, W0);
    cplx B0[3][12], a[3][12], b[3][12], c[3][12], d[3][12];
    Bsym(q, p, e2, B0, a, b, c, d);
    double err = 0, h = hat2(q);
    for (int i = 0; i < 12; i++) for (int j = 0; j < 12; j++) {
        cplx s = W0[i][j];
        for (int cc = 0; cc < 3; cc++) s += 0.5*conj(B0[cc][i])*B0[cc][j];
        err = fmax(err, cabs(s - (i == j ? 0.5*h : 0)));
    }
    printf("free symbol W0 + 1/2 B0^+B0 - 1/2 qhat^2 I: max err %.2e\n", err);
    /* hermiticity of the Wilson vertex: X1[ub](q<-q+p) = X1[u](q+p<-q)^+  (check via p -> -p at shifted q) */
    double pp[4] = {0.37, 0, 0, 0}, qq[4], mp[4] = {-0.37, 0, 0, 0};
    for (int i = 0; i < 4; i++) qq[i] = q[i] + pp[i];
    cplx Xa[12][12], Xb[12][12], t1, t2;
    wilson(q, pp, e2, Xa, &t1, NULL);    /* (q+p <- q) with u   */
    wilson(qq, mp, e2, Xb, &t2, NULL);   /* (q <- q+p) with ub  */
    err = 0; for (int i = 0; i < 12; i++) for (int j = 0; j < 12; j++) err = fmax(err, cabs(Xb[i][j] - conj(Xa[j][i])));
    printf("Wilson vertex hermiticity: max err %.2e;  Im trT = %.2e\n", err, cimag(t1));
    /* reflection / swap symmetry of the integrand */
    double m = 0.3, o[4];
    double q1[4] = {0.3, -1.1, 2.0, 0.7}, q2[4] = {0.3, -1.1, -2.0, 0.7}, q3[4] = {0.3, -1.1, 0.7, 2.0}, q4[4] = {0.3, -1.1, 2.0, -0.7};
    double v1 = integrand(q1, pp, e2, m, o), v2 = integrand(q2, pp, e2, m, o), v3 = integrand(q3, pp, e2, m, o), v4 = integrand(q4, pp, e2, m, o);
    printf("symmetry q3->-q3, q3<->q4, q4->-q4: %.15f %.15f %.15f %.15f\n", v1, v2, v3, v4);
}

int main(int argc, char **argv) {
    read_tensors(argv[1]);
    if (argc > 2 && !strcmp(argv[2], "test")) { selftest(); return 0; }
    double m = atof(argv[2]); int N = atoi(argv[3]); double p1 = atof(argv[4]); int ed = atoi(argv[5]);
    int j1lo = atoi(argv[6]), j1hi = atoi(argv[7]); int sym = atoi(argv[8]);
    double eps[4] = {0, 0, 0, 0}; eps[ed] = 1;
    double pv[3][4] = {{p1, 0, 0, 0}, {-p1, 0, 0, 0}, {0, 0, 0, 0}};
    double S[3] = {0, 0, 0}, parts[4] = {0, 0, 0, 0};
    double dq = 2*M_PI/N;
    for (int j1 = j1lo; j1 < j1hi; j1++) for (int j2 = 0; j2 < N; j2++)
    for (int j3 = 0; j3 < (sym ? N/2+1 : N); j3++) for (int j4 = 0; j4 < (sym ? j3+1 : N); j4++) {
        double w = 1;
        if (sym) { w = (j3 == 0 || j3 == N/2 ? 1 : 2) * (j4 == 0 || j4 == N/2 ? 1 : 2) * (j4 < j3 ? 2 : 1); }
        double q[4] = {j1*dq, j2*dq, j3*dq, j4*dq}, o[4];
        for (int s = 0; s < 3; s++) {
            S[s] += w * integrand(q, pv[s], eps, m, o);
            if (s == 0) for (int t = 0; t < 4; t++) parts[t] += w*o[t];
        }
    }
    printf("%.17e %.17e %.17e %.17e %.17e %.17e %.17e\n", S[0], S[1], S[2], parts[0], parts[1], parts[2], parts[3]);
    return 0;
}
