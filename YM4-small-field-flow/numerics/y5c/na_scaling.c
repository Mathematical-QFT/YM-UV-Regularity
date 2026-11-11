/* N-a: scaling of the lattice integrand (Lemma ol-continuum-integrand, Section sec:ol-feynman).

   Reuses the SU(2) integrand of ../vacpol.c unchanged (lattice vertices from the exact Taylor
   tensors in ../tensors.txt, background along the Cartan generator, initial-point convention,
   propagators at q and q+p).  For a unit direction omega = (w_q, w_m) on the closed half-sphere
   |w_q|^2 + w_m^2 = 1, w_m >= 0, a real unit polarization c and a unit direction u of p, it computes

       R(rho) = rho^4 * I[c,u](rho w_q, rho w_m),   I[c,u](q,m) = d^2/dt^2 P_vac(q, t u, m) at t = 0,

   by a second difference in t with steps rho*h and rho*4h and Richardson extrapolation.  In the
   component pairing of vacpol.c, I[c,u] equals sum c_mu c_nu u_k u_l I_{mu nu k l} in the trace
   pairing of the note.  Lemma ol-continuum-integrand predicts |R(rho) - I0[c,u](omega)| <= C rho with

       I0[c,u](q,m) = kappa D^-2 [ 4 - 3 s^2 - 4 g^2/D + 16 a^2 g^2/D^2 - 8 a g s/D ],
       D = |q|^2 + m^2,  a = q.u,  g = q.c,  s = u.c,  kappa = C_A/T_rho = 4 for SU(2).

   usage: na_scaling ../tensors.txt ndir jmax seed   (the reported run: 200 12 7; under a second) */
#define main vacpol_main
#include "../vacpol.c"
#undef main

static unsigned long long rs;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) / 9007199254740992.0; }
static double grand(void) { return sqrt(-2*log(urand())) * cos(2*M_PI*urand()); }
static void unitvec(double *v, int n) {
    double s = 0; for (int i = 0; i < n; i++) { v[i] = grand(); s += v[i]*v[i]; }
    s = sqrt(s); for (int i = 0; i < n; i++) v[i] /= s;
}
static double dot4(const double *a, const double *b) { double s = 0; for (int i = 0; i < 4; i++) s += a[i]*b[i]; return s; }

static double I0(const double *q, double m, const double *c, const double *u, double kappa) {
    double D = dot4(q, q) + m*m, a = dot4(q, u), g = dot4(q, c), s = dot4(u, c);
    return kappa/(D*D) * (4 - 3*s*s - 4*g*g/D + 16*a*a*g*g/(D*D) - 8*a*g*s/D);
}
/* second difference in t of the integrand at loop momentum q, step dt along u */
static double sd(const double *q, double m, const double *c, const double *u, double dt) {
    double pp[4], pm[4], z[4] = {0, 0, 0, 0}, o[4];
    for (int i = 0; i < 4; i++) { pp[i] = dt*u[i]; pm[i] = -dt*u[i]; }
    return (integrand(q, pp, c, m, o) + integrand(q, pm, c, m, o) - 2*integrand(q, z, c, m, o)) / (dt*dt);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s tensors.txt ndir jmax seed\n", argv[0]); return 1; }
    read_tensors(argv[1]);
    int ndir = atoi(argv[2]), jmax = atoi(argv[3]); rs = 88172645463325252ULL ^ (unsigned long long)atoll(argv[4]);
    const double kappa = 4.0, h = 1e-3; const int nprint = 6;
    double worst[64], worst2[64]; for (int j = 0; j <= jmax; j++) worst[j] = worst2[j] = 0;
    printf("# N-a: R(rho) = rho^4 I[c,u](rho w) vs I0[c,u](w), %d directions; rows for the first %d\n", ndir, nprint);
    printf("# dir 0: m = 0, q = e1; dir 1: c = e2, u = e1 (Pi_22, p || e1); dir 2: c = u (longitudinal); others random\n");
    printf("# columns: dir j rho I0 R (R-I0)/rho\n");
    for (int k = 0; k < ndir; k++) {
        double w[5], qd[4], c[4], u[4], wm;
        unitvec(w, 5); for (int i = 0; i < 4; i++) qd[i] = w[i]; wm = fabs(w[4]);
        if (k == 0) { for (int i = 0; i < 4; i++) qd[i] = (i == 0); wm = 0; }      /* m = 0, q = e1 */
        unitvec(c, 4); unitvec(u, 4);
        if (k == 1) { for (int i = 0; i < 4; i++) { c[i] = (i == 1); u[i] = (i == 0); } }  /* Pi_22, p || e1 */
        if (k == 2) { for (int i = 0; i < 4; i++) c[i] = u[i]; }                           /* longitudinal */
        double i0 = I0(qd, wm, c, u, kappa);
        for (int j = 0; j <= jmax; j++) {
            double rho = ldexp(1.0, -j), q[4], m = rho*wm;
            for (int i = 0; i < 4; i++) q[i] = rho*qd[i];
            double r1 = sd(q, m, c, u, rho*h), r4 = sd(q, m, c, u, 4*rho*h);
            double R = pow(rho, 4) * (16*r1 - r4)/15;
            double e = (R - i0)/rho;
            if (fabs(e) > worst[j]) worst[j] = fabs(e);
            if (fabs(e)/rho > worst2[j]) worst2[j] = fabs(e)/rho;
            if (k < nprint) printf("%3d %2d %.6e % .10f % .10f % .6f\n", k, j, rho, i0, R, e);
        }
    }
    printf("# max over all %d directions, by rho:\n", ndir);
    for (int j = 0; j <= jmax; j++)
        printf("# rho = 2^-%-2d  max |R-I0| = %.3e   max |R-I0|/rho = %.6f   max |R-I0|/rho^2 = %.4f\n",
               j, worst[j]*ldexp(1.0, -j), worst[j], worst2[j]);
    return 0;
}
