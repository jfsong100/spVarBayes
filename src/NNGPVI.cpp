#define USE_FC_LEN_T
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include "util.h"
#include "nngp_fun.h"
#include <vector>
#include <algorithm>
#include <iterator>

#ifdef _OPENMP
#include <omp.h>
#endif

// #include <cppad/cppad.hpp>
// #include <cppad/example/cppad_eigen.hpp>
// #include <cassert>
// #include <Eigen/Dense>
// #include <Eigen/Core>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <Rconfig.h>
#include <R_ext/Linpack.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#include <R_ext/Utils.h>


using namespace std;


// using CppAD::AD;
// using CppAD::ADFun;
// using Eigen::Matrix;
// using Eigen::Dynamic;
//
// typedef Matrix< AD<double> , Dynamic, Dynamic > a_matrix;
// typedef Matrix< AD<double> , Dynamic , 1>       a_vector;
// typedef Matrix< double     , Dynamic, Dynamic > matrix;
// typedef Matrix< double ,     Dynamic , 1>       vector;
// typedef CppAD::AD<double> ADdouble;


#ifndef FCONE
#define FCONE
#endif






extern "C" {


  SEXP spVarBayes_NNGP_beta_w_jointcpp(SEXP y_r, SEXP X_r,
                               SEXP n_r, SEXP p_r, SEXP m_r, SEXP m_vi_r, SEXP coords_r, SEXP covModel_r, SEXP rho_r,
                               SEXP zetaSqIG_r, SEXP tauSqIG_r, SEXP phibeta_r, SEXP nuUnif_r,
                               SEXP zetaSqStarting_r, SEXP tauSqStarting_r, SEXP phiStarting_r, SEXP nuStarting_r,
                               SEXP sType_r, SEXP nThreads_r, SEXP verbose_r, SEXP fix_nugget_r, SEXP N_phi_r, SEXP Trace_N_r,
                               SEXP max_iter_r,
                               SEXP var_input_r,
                               SEXP phi_input_r, SEXP phi_iter_max_r, SEXP initial_mu_r,
                               SEXP min_iter_r, SEXP K_r, SEXP stop_K_r){


    int h, i, j, k, l, s, info, nProtect=0;
    const int inc = 1;
    const double one = 1.0;
    const double negOne = -1.0;
    const double zero = 0.0;
    char const *lower = "L";
    char const *upper = "U";
    char const *ntran = "N";
    char const *ytran = "T";
    char const *rside = "R";
    char const *lside = "L";
    const double pi = 3.1415926;
    //get args
    double *y = REAL(y_r);
    double *X = REAL(X_r);
    int p = INTEGER(p_r)[0];
    int n = INTEGER(n_r)[0];
    int m = INTEGER(m_r)[0];
    int m_vi = INTEGER(m_vi_r)[0];
    double *coords = REAL(coords_r);
    double fix_nugget = REAL(fix_nugget_r)[0];
    int covModel = INTEGER(covModel_r)[0];
    std::string corName = getCorName(covModel);
    //double converge_per  =  REAL(converge_per_r)[0];
    double phi_input  =  REAL(phi_input_r)[0];
    double *var_input  =  REAL(var_input_r);
    int initial_mu  =  INTEGER(initial_mu_r)[0];
    int phi_iter_max = INTEGER(phi_iter_max_r)[0];
    int nThreads = INTEGER(nThreads_r)[0];
    int verbose = INTEGER(verbose_r)[0];

    int K = INTEGER(K_r)[0];
    int stop_K = INTEGER(stop_K_r)[0];
    int min_iter = INTEGER(min_iter_r)[0];
    int max_iter = INTEGER(max_iter_r)[0];

    //double  vi_threshold  =  REAL(vi_threshold_r)[0];
    double  rho  =  REAL(rho_r)[0];
    //double  rho_phi  =  REAL(rho_phi_r)[0];
    //priors
    double zetaSqIGa = REAL(zetaSqIG_r)[0]; double zetaSqIGb = REAL(zetaSqIG_r)[1];
    double tauSqIGa = REAL(tauSqIG_r)[0]; double tauSqIGb = REAL(tauSqIG_r)[1];
    //double phiUnifa = REAL(phiUnif_r)[0]; double phiUnifb = REAL(phiUnif_r)[1];
    double phimin = REAL(phibeta_r)[0]; double phimax = REAL(phibeta_r)[1];

    double a_phi = (phi_input - phimin)/(phimax-phimin)*10;
    double b_phi = 10 - a_phi;

    double nuUnifa = 0, nuUnifb = 0;
    if(corName == "matern"){
      nuUnifa = REAL(nuUnif_r)[0]; nuUnifb = REAL(nuUnif_r)[1];
    }

#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#else
    if(nThreads > 1){
      warning("n.omp.threads > %i, but source not compiled with OpenMP support.", nThreads);
      nThreads = 1;
    }
#endif

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tModel description\n");
      Rprintf("----------------------------------------\n");
      Rprintf("NNGP Latent model fit with %i observations.\n\n", n);
      Rprintf("Number of covariates %i (including intercept if specified).\n\n", p);
      Rprintf("Using the %s spatial correlation model.\n\n", corName.c_str());
      Rprintf("Using %i nearest neighbors.\n\n", m);
      Rprintf("Priors and hyperpriors:\n");
      Rprintf("\tbeta flat.\n");
#ifdef _OPENMP
      Rprintf("\nSource compiled with OpenMP support and model fit using %i thread(s).\n", nThreads);
#else
      Rprintf("\n\nSource not compiled with OpenMP support.\n");
#endif
    }

    //parameters
    int nTheta, zetaSqIndx, tauSqIndx, phiIndx, nuIndx;

    if(corName != "matern"){
      nTheta = 3;//zeta^2, tau^2, phi
      zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
    }else{
      nTheta = 4;//zeta^2, tau^2, phi, nu
      zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2; nuIndx = 3;
    }

    //allocated for the nearest neighbor index vector (note, first location has no neighbors).
    int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);

    SEXP nnIndx_r; PROTECT(nnIndx_r = allocVector(INTSXP, nIndx)); nProtect++; int *nnIndx = INTEGER(nnIndx_r);

    //int *nnIndx = (int *) R_alloc(nIndx, sizeof(int));

    double *d = (double *) R_alloc(nIndx, sizeof(double));

    SEXP nnIndxLU_r; PROTECT(nnIndxLU_r = allocVector(INTSXP, 2*n)); nProtect++; int *nnIndxLU = INTEGER(nnIndxLU_r); //first column holds the nnIndx index for the i-th location and the second columns holds the number of neighbors the i-th location has (the second column is a bit of a waste but will simplifying some parallelization).

    //int *nnIndxLU = (int *) R_alloc(2*n, sizeof(int));

    //make the neighbor index
    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tBuilding neighbor index\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }

    if(INTEGER(sType_r)[0] == 0){
      mkNNIndx(n, m, coords, nnIndx, d, nnIndxLU);
    }
    if(INTEGER(sType_r)[0] == 1){
      mkNNIndxTree0(n, m, coords, nnIndx, d, nnIndxLU);
    }else{
      mkNNIndxCB(n, m, coords, nnIndx, d, nnIndxLU);
    }

    SEXP CIndx_r; PROTECT(CIndx_r = allocVector(INTSXP, 2*n)); nProtect++; int *CIndx = INTEGER(CIndx_r); //index for D and C.

    //int *CIndx = (int *) R_alloc(2*n, sizeof(int));
    for(i = 0, j = 0; i < n; i++){//zero should never be accessed
      j += nnIndxLU[n+i]*nnIndxLU[n+i];
      if(i == 0){
        CIndx[n+i] = 0;
        CIndx[i] = 0;
      }else{
        CIndx[n+i] = nnIndxLU[n+i]*nnIndxLU[n+i];
        CIndx[i] = CIndx[n+i-1] + CIndx[i-1];
      }
    }

    SEXP numIndxCol_r; PROTECT(numIndxCol_r = allocVector(INTSXP, n)); nProtect++; int *numIndxCol = INTEGER(numIndxCol_r); zeros_int(numIndxCol, n);
    get_num_nIndx_col(nnIndx, nIndx, numIndxCol);

    SEXP cumnumIndxCol_r; PROTECT(cumnumIndxCol_r = allocVector(INTSXP, n)); nProtect++; int *cumnumIndxCol = INTEGER(cumnumIndxCol_r); zeros_int(cumnumIndxCol,n);
    get_cumnum_nIndx_col(numIndxCol, n, cumnumIndxCol);

    SEXP nnIndxCol_r; PROTECT(nnIndxCol_r = allocVector(INTSXP, nIndx+n)); nProtect++; int *nnIndxCol = INTEGER(nnIndxCol_r); zeros_int(nnIndxCol, n);
    get_nnIndx_col(nnIndx, n, nIndx, cumnumIndxCol, numIndxCol, nnIndxCol);

    int *sumnnIndx = (int *) R_alloc(n-1, sizeof(int));; zeros_int(sumnnIndx, n-1);
    get_sum_nnIndx(sumnnIndx, n, m);

    SEXP nnIndxnnCol_r; PROTECT(nnIndxnnCol_r = allocVector(INTSXP, nIndx)); nProtect++; int *nnIndxnnCol = INTEGER(nnIndxnnCol_r); zeros_int(nnIndxnnCol, n);
    get_nnIndx_nn_col(nnIndx, n, m, nIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, sumnnIndx);


    double *D = (double *) R_alloc(j, sizeof(double));

    for(i = 0; i < n; i++){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        for(l = 0; l <= k; l++){
          D[CIndx[i]+l*nnIndxLU[n+i]+k] = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
        }
      }
    }
    int mm = m*m;
    SEXP B_r; PROTECT(B_r = allocVector(REALSXP, nIndx)); nProtect++; double *B = REAL(B_r);
    SEXP F_r; PROTECT(F_r = allocVector(REALSXP, n)); nProtect++; double *F = REAL(F_r);

    //double *B = (double *) R_alloc(nIndx, sizeof(double));
    //double *F = (double *) R_alloc(n, sizeof(double));

    double *c =(double *) R_alloc(m*nThreads, sizeof(double));
    double *C = (double *) R_alloc(mm*nThreads, sizeof(double));

    SEXP beta_r; PROTECT(beta_r = allocVector(REALSXP, p)); nProtect++; double *beta = REAL(beta_r); zeros(beta, p);

    SEXP beta_cov_r; PROTECT(beta_cov_r = allocVector(REALSXP, p*p)); nProtect++; double *beta_cov = REAL(beta_cov_r); zeros(beta_cov, p*p);

    SEXP theta_r; PROTECT(theta_r = allocVector(REALSXP, nTheta)); nProtect++; double *theta = REAL(theta_r);

    SEXP w_mu_r; PROTECT(w_mu_r = allocVector(REALSXP, n)); nProtect++; double *w_mu = REAL(w_mu_r);

    SEXP sigma_sq_r; PROTECT(sigma_sq_r = allocVector(REALSXP, n)); nProtect++; double *sigma_sq = REAL(sigma_sq_r);

    // SEXP L_beta_track_r; PROTECT(L_beta_track_r = allocVector(REALSXP, max_iter)); nProtect++; double *L_beta_track = REAL(L_beta_track_r);
    // SEXP A_beta_track_r; PROTECT(A_beta_track_r = allocVector(REALSXP, max_iter)); nProtect++; double *A_beta_track = REAL(A_beta_track_r);
    // SEXP E_vi_track_r; PROTECT(E_vi_track_r = allocVector(REALSXP, max_iter)); nProtect++; double *E_vi_track = REAL(E_vi_track_r);

    //double *beta = (double *) R_alloc(p, sizeof(double)); zeros(beta, p);
    //double *theta = (double *) R_alloc(nTheta, sizeof(double));

    // theta[0] = REAL(zetaSqStarting_r)[0];
    // theta[1] = REAL(phiStarting_r)[0];
    //
    // if(corName == "matern"){
    //   theta[2] = REAL(nuStarting_r)[0];
    // }
    //

    theta[zetaSqIndx] = REAL(zetaSqStarting_r)[0];
    theta[tauSqIndx] = REAL(tauSqStarting_r)[0];
    //theta[phiIndx] = REAL(phiStarting_r)[0];
    theta[phiIndx] = phi_input;
    if(corName == "matern"){
      theta[nuIndx] = REAL(nuStarting_r)[0];
    }

    //other stuff
    double logDetInv;
    int accept = 0, batchAccept = 0, status = 0;
    int jj, kk, pp = p*p, nn = n*n, np = n*p;
    double *one_n = (double *) R_alloc(n, sizeof(double)); ones(one_n, n);
    //double *tmp_n = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n, n);

    double *tmp_pp = (double *) R_alloc(pp, sizeof(double));
    double *tmp_p = (double *) R_alloc(p, sizeof(double));
    double *tmp_p2 = (double *) R_alloc(p, sizeof(double)); zeros(tmp_p2, p);
    double *tmp_n = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n, n);
    double *tmp_n2 = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n2, n);
    double *XtX = (double *) R_alloc(pp, sizeof(double));
    double *tau_sq_H = (double *) R_alloc(one, sizeof(double));
    double *tau_sq_I = (double *) R_alloc(one, sizeof(double));
    double *tau_sq_term = (double *) R_alloc(one, sizeof(double));

    ////////////
    SEXP A_beta_r; PROTECT(A_beta_r = allocVector(REALSXP, np)); nProtect++; double *A_beta = REAL(A_beta_r);
    SEXP L_beta_r; PROTECT(L_beta_r = allocVector(REALSXP, (p-1)*p/2)); nProtect++; double *L_beta = REAL(L_beta_r); zeros(L_beta,(p-1)*p/2);

    double *z_vec = (double *) R_alloc(p, sizeof(double));
    double *ub_vec = (double *) R_alloc(p, sizeof(double));
    double *gradient_beta = (double *) R_alloc(p, sizeof(double));
    double *tmp_Xtu = (double *) R_alloc(p, sizeof(double));

    double *l_gradient = (double *) R_alloc(p, sizeof(double));

    double *l_gradient_sum = (double *) R_alloc(p, sizeof(double));


    int L_beta_ind = p * (p-1)/2;

    double *E_A_beta_sq = (double *) R_alloc(np, sizeof(double)); zeros(E_A_beta_sq, np);
    double *delta_A_beta_sq = (double *) R_alloc(np, sizeof(double)); zeros(delta_A_beta_sq, np);
    double *delta_A_beta = (double *) R_alloc(np, sizeof(double)); zeros(delta_A_beta, np);


    double *E_L_beta_sq = (double *) R_alloc(L_beta_ind, sizeof(double)); zeros(E_L_beta_sq, L_beta_ind);
    double *delta_L_beta_sq = (double *) R_alloc(L_beta_ind, sizeof(double)); zeros(delta_L_beta_sq, L_beta_ind);
    double *delta_L_beta = (double *) R_alloc(L_beta_ind, sizeof(double)); zeros(delta_L_beta, L_beta_ind);

    SEXP E_vi_r; PROTECT(E_vi_r = allocVector(REALSXP, p)); nProtect++; double *E_vi = REAL(E_vi_r);

    for(int i = 0; i < np; i++){
      A_beta[i] = 0.1;
    }
    zeros(A_beta,np);
    double *E_E_vi_sq = (double *) R_alloc(p, sizeof(double)); zeros(E_E_vi_sq, p);
    double *delta_E_vi_sq = (double *) R_alloc(p, sizeof(double)); zeros(delta_E_vi_sq, p);
    double *delta_E_vi = (double *) R_alloc(p, sizeof(double)); zeros(delta_E_vi, p);
    double *A_beta_gradient = (double *) R_alloc(np, sizeof(double));
    double *L_beta_gradient = (double *) R_alloc(L_beta_ind, sizeof(double));

    double *A_beta_gradient_sum = (double *) R_alloc(np, sizeof(double));
    double *L_beta_gradient_sum = (double *) R_alloc(L_beta_ind, sizeof(double));

    SEXP IndxLU_beta_r; PROTECT(IndxLU_beta_r = allocVector(INTSXP, 2*p)); nProtect++;
    int *IndxLU_beta = INTEGER(IndxLU_beta_r); zeros_int(IndxLU_beta, 2*p);

    int L_index = 0;
    for (int i = 0; i < p; i++) {
      IndxLU_beta[i+p] = i; // Number of elements used before u_i
      IndxLU_beta[i] = L_index; // Start index in L vector
      L_index += i; // Move to the next position in L
    }

    int *numIndxCol_beta = (int *) R_alloc(p + p*(p-1)/2, sizeof(int));
    int *cumnumIndxCol_beta = (int *) R_alloc(p, sizeof(int));

    int *IndxCol_beta = (int *) R_alloc(p*(p-1)/2, sizeof(int));

    int *count = numIndxCol_beta;
    int *indices = numIndxCol_beta + p;

    int total_values = 0;
    for (int i = 0; i < p; i++) {
      count[i] = 0;
    }

    cumnumIndxCol_beta[0] = 0;
    if(p > 1){
      for (int j = 1; j < p; j++) {
        int start_j = IndxLU_beta[p + j];

        for (int k = 0; k < j; k++) {
          indices[total_values] = start_j + k - 1;
          IndxCol_beta[total_values] = k + 1;
          count[k]++;
          total_values++;
        }
      }

      for (int i = 1; i < p; i++) {
        cumnumIndxCol_beta[i] = cumnumIndxCol_beta[i - 1] + count[i - 1];
      }
    }


    ////////////



    //double *w_mu = (double *) R_alloc(n, sizeof(double));
    //zeros(w_mu, n);
    F77_NAME(dgemm)(ytran, ntran, &p, &p, &n, &one, X, &n, X, &n, &zero, XtX, &p FCONE FCONE);

    if(initial_mu){
      F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, y, &inc, &zero, tmp_p, &inc FCONE);

      for(i = 0; i < pp; i++){
        tmp_pp[i] = XtX[i];
      }

      F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotrf failed\n");}
      F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotri failed\n");}

      F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc FCONE);

      for(i = 0; i < n; i++){
        w_mu[i] = y[i] - F77_NAME(ddot)(&p, &X[i], &n, tmp_p2, &inc);
      }
      // F77_NAME(dcopy)(&n, y, &inc, w_mu, &inc);
    }else{
      zeros(w_mu, n);
    }
    //double *sigma_sq = (double *) R_alloc(n, sizeof(double));
    ones(sigma_sq, n);

    for(i = 0; i < pp; i++){
      tmp_pp[i] = XtX[i];
    }

    F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotrf failed\n");}
    F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotri failed\n");}

    F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc FCONE);


    for(int i = 0; i < p; i++){
      // E_vi[i] = tmp_pp[i * (p + 1)] * theta[tauSqIndx];
      E_vi[i] = 0.01;
    }

    double *w_mu_update = (double *) R_alloc(n, sizeof(double)); zeros(w_mu_update, n);
    double *E_mu_sq = (double *) R_alloc(n, sizeof(double)); zeros(E_mu_sq, n);
    double *delta_mu_sq = (double *) R_alloc(n, sizeof(double)); zeros(delta_mu_sq, n);
    double *delta_mu = (double *) R_alloc(n, sizeof(double)); zeros(delta_mu, n);
    double *m_mu = (double *) R_alloc(n, sizeof(double)); zeros(m_mu, n);

    double *sigma_sq_update = (double *) R_alloc(n, sizeof(double)); ones(sigma_sq_update, n);

    double a, v, b, e, mu, var, aij, phiCand, nuCand = 0, nu = 0;

    double a_tau_update = n * 0.5 + tauSqIGa;
    double b_tau_update = 0.0;
    double tau_sq = 0.0;

    double a_zeta_update = n * 0.5 + zetaSqIGa;
    double b_zeta_update = 0.0;
    double zeta_sq = 0.0;
    int N_phi = INTEGER(N_phi_r)[0];
    int Trace_N = INTEGER(Trace_N_r)[0];
    int one_int = 1;
    int three_int = 3;
    double adadelta_noise = 0.0000001;
    double *bk = (double *) R_alloc(nThreads*(1.0+5.0), sizeof(double));
    //double *bk = (double *) R_alloc(nThreads*(1.0+static_cast<int>(floor(nuUnifb))), sizeof(double));
    if(corName == "matern"){nu = theta[nuIndx];}

    //int iter = 1;

    //int iter = (int ) R_alloc(one_int, sizeof(int)); iter = 1;
    int iter = 1;

    double vi_error = 1.0;
    double rho1 = 0.9;
    double rho2 = 0.999;
    double adaptive_adam = 0.001;
    //double vi_threshold = 0.0001;


    // NNGP parameters

    //allocated for the nearest neighbor index vector (note, first location has no neighbors).
    int nIndx_vi = static_cast<int>(static_cast<double>(1+m_vi)/2*m_vi+(n-m_vi-1)*m_vi);

    SEXP nnIndx_vi_r; PROTECT(nnIndx_vi_r = allocVector(INTSXP, nIndx_vi)); nProtect++; int *nnIndx_vi = INTEGER(nnIndx_vi_r);

    double *d_vi = (double *) R_alloc(nIndx_vi, sizeof(double));

    SEXP nnIndxLU_vi_r; PROTECT(nnIndxLU_vi_r = allocVector(INTSXP, 2*n)); nProtect++; int *nnIndxLU_vi = INTEGER(nnIndxLU_vi_r); //first column holds the nnIndx index for the i-th location and the second columns holds the number of neighbors the i-th location has (the second column is a bit of a waste but will simplifying some parallelization).

    //make the neighbor index
    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tBuilding neighbor index for variational inference \n");
      Rprintf("Using %i nearest neighbors.\n\n", m_vi);
#ifdef Win32
      R_FlushConsole();
#endif
    }

    if(INTEGER(sType_r)[0] == 0){
      mkNNIndx(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }
    if(INTEGER(sType_r)[0] == 1){
      mkNNIndxTree0(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }else{
      mkNNIndxCB(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }


    int mm_vi = m_vi*m_vi;
    SEXP A_vi_r; PROTECT(A_vi_r = allocVector(REALSXP, nIndx_vi)); nProtect++; double *A_vi = REAL(A_vi_r); zeros(A_vi,nIndx_vi);
    SEXP S_vi_r; PROTECT(S_vi_r = allocVector(REALSXP, n)); nProtect++; double *S_vi = REAL(S_vi_r); ones(S_vi,n);
    for(int i = 0; i < n; i++){
      S_vi[i] = var_input[i];
    }
    SEXP numIndxCol_vi_r; PROTECT(numIndxCol_vi_r = allocVector(INTSXP, n)); nProtect++; int *numIndxCol_vi = INTEGER(numIndxCol_vi_r); zeros_int(numIndxCol_vi, n);
    get_num_nIndx_col(nnIndx_vi, nIndx_vi, numIndxCol_vi);

    SEXP cumnumIndxCol_vi_r; PROTECT(cumnumIndxCol_vi_r = allocVector(INTSXP, n)); nProtect++; int *cumnumIndxCol_vi = INTEGER(cumnumIndxCol_vi_r); zeros_int(cumnumIndxCol_vi,n);
    get_cumnum_nIndx_col(numIndxCol_vi, n, cumnumIndxCol_vi);

    SEXP nnIndxCol_vi_r; PROTECT(nnIndxCol_vi_r = allocVector(INTSXP, nIndx_vi+n)); nProtect++; int *nnIndxCol_vi = INTEGER(nnIndxCol_vi_r); zeros_int(nnIndxCol_vi, n);
    get_nnIndx_col(nnIndx_vi, n, nIndx_vi, cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi);

    int *sumnnIndx_vi = (int *) R_alloc(n-1, sizeof(int));; zeros_int(sumnnIndx_vi, n-1);
    get_sum_nnIndx(sumnnIndx_vi, n, m_vi);

    SEXP nnIndxnnCol_vi_r; PROTECT(nnIndxnnCol_vi_r = allocVector(INTSXP, nIndx_vi)); nProtect++; int *nnIndxnnCol_vi = INTEGER(nnIndxnnCol_vi_r); zeros_int(nnIndxnnCol_vi, n);
    get_nnIndx_nn_col(nnIndx_vi, n, m_vi, nIndx_vi, cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi, sumnnIndx_vi);

    double *E_a_sq = (double *) R_alloc(nIndx_vi, sizeof(double)); zeros(E_a_sq, nIndx_vi);
    double *delta_a_sq = (double *) R_alloc(nIndx_vi, sizeof(double)); zeros(delta_a_sq, nIndx_vi);
    double *delta_a = (double *) R_alloc(nIndx_vi, sizeof(double)); zeros(delta_a, nIndx_vi);

    double *E_gamma_sq = (double *) R_alloc(n, sizeof(double)); zeros(E_gamma_sq, n);
    double *delta_gamma_sq = (double *) R_alloc(n, sizeof(double)); zeros(delta_gamma_sq, n);
    double *delta_gamma = (double *) R_alloc(n, sizeof(double)); zeros(delta_gamma, n);
    double *gamma_vec = (double *) R_alloc(n, sizeof(double));zeros(gamma_vec, n);

    //double *B = (double *) R_alloc(nIndx, sizeof(double));
    //double *F = (double *) R_alloc(n, sizeof(double));



    double *epsilon_vec = (double *) R_alloc(n, sizeof(double));

    for(int i = 0; i < n; i++){
      epsilon_vec[i] = rnorm(0, 1);
    }

    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);



    // int n_per = nIndx_vi * converge_per;
    // int *sign_vec_old = (int *) R_alloc(n_per, sizeof(int));
    // int *sign_vec_new = (int *) R_alloc(n_per, sizeof(int));
    // int *check_vec = (int *) R_alloc(n_per, sizeof(int));
    int indicator_converge = 0;

    double *trace_vec = (double *) R_alloc(2, sizeof(double));
    double *u_vec = (double *) R_alloc(n, sizeof(double));
    double *u_vec_mean = (double *) R_alloc(n, sizeof(double));
    double ELBO_MC = 0.0;
    double ELBO = 0.0;
    //double *epsilon_vec = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp2 = (double *) R_alloc(n, sizeof(double));
    double *gradient_const = (double *) R_alloc(n, sizeof(double));
    double *gradient = (double *) R_alloc(n, sizeof(double));
    double *gamma_gradient_sum = (double *) R_alloc(n, sizeof(double));
    double *gamma_gradient = (double *) R_alloc(n, sizeof(double));
    double *a_gradient = (double *) R_alloc(nIndx_vi, sizeof(double));
    double *a_gradient_sum = (double *) R_alloc(nIndx_vi, sizeof(double));

    SEXP ELBO_vec_r; PROTECT(ELBO_vec_r = allocVector(REALSXP, max_iter)); nProtect++;
    double *ELBO_vec = REAL(ELBO_vec_r); zeros(ELBO_vec,max_iter);
    double max_ELBO = 0.0;
    int ELBO_convergence_count = 0;

    double E_phi_sq = 0.0;
    double delta_phi = 0.0;
    double delta_phi_sq = 0.0;
    double *rademacher_rv_vec = (double *) R_alloc(n, sizeof(double));
    double *rademacher_rv_temp = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp,n);
    double *rademacher_rv_temp2 = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp2,n);

    double *product_v = (double *) R_alloc(n, sizeof(double));zeros(product_v,n);
    double *product_v2 = (double *) R_alloc(n, sizeof(double));zeros(product_v2,n);
    double *phi_can_vec = (double *) R_alloc(N_phi*N_phi, sizeof(double));zeros(phi_can_vec,N_phi*N_phi);
    double *log_g_phi = (double *) R_alloc(N_phi*N_phi, sizeof(double));zeros(log_g_phi,N_phi*N_phi);
    double *sum_v = (double *) R_alloc(n, sizeof(double));zeros(sum_v,n);

    while(iter <= max_iter & !indicator_converge){

      if(verbose){
        Rprintf("----------------------------------------\n");
        Rprintf("\tIteration at %i \n",iter);
#ifdef Win32
        R_FlushConsole();
#endif
      }

      ///////////////
      //update beta
      ///////////////


      zeros(tau_sq_I, one_int);
      zeros(tau_sq_term, one_int);
      for(i = 0; i < n; i++){
        tmp_n[i] = y[i] - w_mu[i];
        tmp_n2[i] = y[i] - w_mu[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc);
        tau_sq_I[0] += pow(tmp_n[i],2);
        tau_sq_term[0] += pow(tmp_n2[i],2);
      }

      F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, tmp_n, &inc, &zero, tmp_p, &inc FCONE);

      for(i = 0; i < pp; i++){
        tmp_pp[i] = XtX[i];
      }

      F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotrf failed\n");}
      F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotri failed\n");}

      F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc FCONE);

      //F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 3 dpotrf failed\n");}

      F77_NAME(dcopy)(&p, tmp_p2, &inc, beta, &inc);

      for (int i = 0; i < p; i++) {
        for (int j = 0; j <= i; j++) {
          // Calculate the index for column-major format
          int idx = i + j * p;
          beta_cov[idx] = tmp_pp[idx] * theta[tauSqIndx];
        }
      }

      if(verbose){
        for(i = 0; i < p; i++){
          Rprintf("the value of beta[%i] : %f \n",i, beta[i]);
        }
        for(i = 0; i < pp; i++){
          Rprintf("the value of beta cov[%i] : %f \n",i, beta_cov[i]);
        }
#ifdef Win32
        R_FlushConsole();
#endif
      }


      ///////////////
      //update tausq
      ///////////////

      zeros(tau_sq_H, one_int);

      for(i = 0; i < p; i++){
        tau_sq_H[0] += tmp_p2[i]*tmp_p[i];
      }

      zeros(trace_vec,2);
      zeros(u_vec,n);

      int Trace_N_max = Trace_N;
      double tmp_sum;
      for(int i = 0; i < n; i++){
        epsilon_vec[i] = rnorm(0, 1);

      }
      update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);

      for(int k = 0; k < Trace_N_max; k++){
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }
        for(int i = 0; i < p; i++){
          z_vec[i] = rnorm(0, 1);
        }
        update_uvec_ubvec(u_vec, ub_vec, epsilon_vec,  z_vec, A_vi, A_beta, L_beta,
                          S_vi, E_vi, n, p, nnIndxLU_vi, nnIndx_vi, IndxLU_beta);

        for(i = 0; i < n; i++){
          tmp_sum = u_vec[i] + F77_NAME(ddot)(&p, &X[i], &n, ub_vec, &inc);
          trace_vec[0] += pow(tmp_sum,2);
        }

        trace_vec[1] += Q(B, F, u_vec, u_vec, n, nnIndx, nnIndxLU);
      }


      // b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N_max + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
      b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N_max + *tau_sq_term)*0.5;

      //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + *tau_sq_I)*0.5;
      tau_sq = b_tau_update/a_tau_update;
      theta[tauSqIndx] = tau_sq;


      if(verbose){
        Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
        R_FlushConsole();
#endif
      }

      //Rprintf("the value of add_tau : %f \n", trace_vec[0]/Trace_N + *tau_sq_I);
      ///////////////
      //update zetasq
      ///////////////

      updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

      double zeta_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
      b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N_max + zeta_Q)*theta[zetaSqIndx]*0.5;
      //Rprintf("zeta_Q: %f \n", zeta_Q);
      //b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q)*0.5;
      zeta_sq = b_zeta_update/a_zeta_update;
      //Rprintf("the value of add_zeta : %f \n", (trace_vec[1]/Trace_N + zeta_Q));
      theta[zetaSqIndx] = zeta_sq;

      if(verbose){
        Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
        R_FlushConsole();
#endif
      }
      updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
      ///////////////
      //update phi
      ///////////////

      if(iter < phi_iter_max){

        double *a_phi_vec = (double *) R_alloc(N_phi, sizeof(double));
        double *b_phi_vec = (double *) R_alloc(N_phi, sizeof(double));
        a_phi_vec[0] = a_phi;
        b_phi_vec[0] = b_phi;

        for(int i = 1; i < N_phi; i++){
          if (i % 2 == 0) {
            a_phi_vec[i] = a_phi_vec[0] + 0.01*i;
            b_phi_vec[i] = b_phi_vec[0] + 0.01*i;
            // a_phi_vec[i] = a_phi_vec[0]*(1+0.1*i);
            // b_phi_vec[i] = b_phi_vec[0]*(1+0.1*i);
          } else {
            a_phi_vec[i] = a_phi_vec[0] + 0.01*i*(-1);
            b_phi_vec[i] = b_phi_vec[0] + 0.01*i*(-1);
            // a_phi_vec[i] = a_phi_vec[0]*(1-0.1*i);
            // b_phi_vec[i] = b_phi_vec[0]*(1-0.1*i);
          }

        }

        double phi_Q = 0.0;
        double diag_sigma_sq_sum = 0.0;

        int max_index;
        zeros(phi_can_vec,N_phi*N_phi);
        zeros(log_g_phi,N_phi*N_phi);
        for(int i = 0; i < N_phi; i++){
          for(int j = 0; j < N_phi; j++){

            for(int k = 0; k < Trace_N; k++){
              phi_can_vec[i*N_phi+j] += rbeta(a_phi_vec[i], b_phi_vec[j]);  // Notice the indexing here
            }
            phi_can_vec[i*N_phi+j] /= Trace_N;
            phi_can_vec[i*N_phi+j] = phi_can_vec[i*N_phi+j]*(phimax - phimin) + phimin;
          }
        }


        for(i = 0; i < N_phi*N_phi; i++){

          updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                   theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb);

          //phi_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
          phi_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
          for(int i = 0; i < n; i++){
            epsilon_vec[i] = rnorm(0, 1);
          }
          for(int i = 0; i < p; i++){
            z_vec[i] = rnorm(0, 1);
          }
          update_uvec_ubvec(u_vec, ub_vec, epsilon_vec,  z_vec, A_vi, A_beta, L_beta,
                            S_vi, E_vi, n, p, nnIndxLU_vi, nnIndx_vi, IndxLU_beta);
          logDetInv = 0.0;
          diag_sigma_sq_sum = 0.0;
          for(j = 0; j < n; j++){
            logDetInv += log(1/F[j]);
          }

          log_g_phi[i] = logDetInv*0.5 - (phi_Q + Q(B, F, u_vec, u_vec, n, nnIndx, nnIndxLU))*0.5;
        }

        max_index = max_ind(log_g_phi,N_phi*N_phi);
        a_phi = a_phi_vec[max_index/N_phi];
        b_phi = b_phi_vec[max_index % N_phi];

        theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;


        updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                 theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
      }

      if(verbose){
        Rprintf("the value of theta[%i phiIndx] : %f \n", phiIndx, theta[phiIndx]);
#ifdef Win32
        R_FlushConsole();
#endif
      }

      ///////////////
      //update w
      ///////////////

      zeros(w_mu_temp,n);
      zeros(w_mu_temp2,n);

      product_B_F(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp);
      //product_B_F(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2);
      product_B_F_vec(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);

      double gradient_mu = 0.0;
      //Rprintf("w_mu_update: ");
      for(i = 0; i < n; i++){
        //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
        gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc))/theta[tauSqIndx]);

        //Rprintf("%i ",gradient_mu);
        //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i]/theta[zetaSqIndx] + (y[i])/theta[tauSqIndx]);
        E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
        delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
        delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
        w_mu_update[i] = w_mu[i] + delta_mu[i];
        //Rprintf("%f ",w_mu_update[i]);
      }

      product_B_F(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp);
      //product_B_F(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2);
      product_B_F_vec(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);


      zeros(gradient,n);
      zeros(gamma_gradient_sum, n);
      zeros(gamma_gradient,n);

      zeros(l_gradient,p);
      zeros(l_gradient_sum,p);

      for(int k = 0; k < Trace_N; k++){
        zeros(gamma_gradient,n);
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }

        for(int i = 0; i < p; i++){
          z_vec[i] = rnorm(0, 1);
        }

        gamma_l_gradient_fun(u_vec, epsilon_vec, gamma_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                           B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                           cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,w_mu_temp,w_mu_temp2,gradient,
                           ub_vec, z_vec, p, L_beta, A_beta, X, gradient_beta, XtX, tmp_Xtu, l_gradient, E_vi,
                           numIndxCol_beta, cumnumIndxCol_beta, IndxCol_beta, IndxLU_beta);


        vecsum(gamma_gradient_sum, gamma_gradient, Trace_N, n);
        vecsum(l_gradient_sum, l_gradient, Trace_N, p);
      }

      //free(gamma_gradient);
      //Rprintf("S_vi: ");
      for(i = 0; i < n; i++){
        E_gamma_sq[i] = rho * E_gamma_sq[i] + (1 - rho) * pow(gamma_gradient_sum[i],2);
        delta_gamma[i] = sqrt(delta_gamma_sq[i]+adadelta_noise)/sqrt(E_gamma_sq[i]+adadelta_noise)*gamma_gradient_sum[i];
        delta_gamma_sq[i] = rho*delta_gamma_sq[i] + (1 - rho) * pow(delta_gamma[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        //S_vi[i] = exp(pow((log(sqrt(S_vi[i])) + delta_gamma[i]),2));
        S_vi[i] = pow(exp(log(sqrt(S_vi[i])) + delta_gamma[i]),2);
        //Rprintf("%f ",S_vi[i]);
        //S_vi[i] = pow(exp(gamma_vec[i]),2);
      }

      for(i = 0; i < p; i++){
        E_E_vi_sq[i] = rho * E_E_vi_sq[i] + (1 - rho) * pow(l_gradient_sum[i],2);
        delta_E_vi[i] = sqrt(delta_E_vi_sq[i]+adadelta_noise)/sqrt(E_E_vi_sq[i]+adadelta_noise)*l_gradient_sum[i];
        delta_E_vi_sq[i] = rho*delta_E_vi_sq[i] + (1 - rho) * pow(delta_E_vi[i],2);
        E_vi[i] = pow(exp(log(sqrt(E_vi[i])) + delta_E_vi[i]),2);
      }

      //Rprintf("\n");
      zeros(a_gradient,nIndx_vi);
      zeros(a_gradient_sum, nIndx_vi);
      zeros(A_beta_gradient, np);
      zeros(A_beta_gradient_sum, np);

      zeros(L_beta_gradient, L_beta_ind);
      zeros(L_beta_gradient_sum, L_beta_ind);

      for(int k = 0; k < Trace_N; k++){
        zeros(a_gradient,nIndx_vi);
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }

        for(int i = 0; i < p; i++){
          z_vec[i] = rnorm(0, 1);
        }

        // a_gradient_fun(u_vec, epsilon_vec, a_gradient, gradient_const, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
        //                B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
        //                w_mu_temp,w_mu_temp2);

        a_Abeta_Lbeta_gradient_fun(u_vec, epsilon_vec, a_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                       B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                       w_mu_temp,w_mu_temp2,
                       ub_vec, z_vec, p, L_beta, A_beta, X, E_vi, IndxLU_beta,
                       A_beta_gradient, L_beta_gradient, gradient, gradient_beta, tmp_Xtu, XtX,
                       cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi);

        vecsum(a_gradient_sum, a_gradient, Trace_N, nIndx_vi);
        vecsum(A_beta_gradient_sum, A_beta_gradient, Trace_N, np);
        vecsum(L_beta_gradient_sum, L_beta_gradient, Trace_N, L_beta_ind);

      }
      //free(a_gradient);
      //update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);


      //Rprintf("A_vi: ");
      for(i = 0; i < nIndx_vi; i++){
        E_a_sq[i] = rho * E_a_sq[i] + (1 - rho) * pow(a_gradient_sum[i],2);
        delta_a[i] = sqrt(delta_a_sq[i]+adadelta_noise)/sqrt(E_a_sq[i]+adadelta_noise)*a_gradient_sum[i];
        delta_a_sq[i] = rho*delta_a_sq[i] + (1 - rho) * pow(delta_a[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        A_vi[i] = A_vi[i] + delta_a[i];
        //Rprintf("%f ",A_vi[i]);
      }

      for(i = 0; i < np; i++){
        E_A_beta_sq[i] = rho * E_A_beta_sq[i] + (1 - rho) * pow(A_beta_gradient_sum[i],2);
        delta_A_beta[i] = sqrt(delta_A_beta_sq[i]+adadelta_noise)/sqrt(E_A_beta_sq[i]+adadelta_noise)*A_beta_gradient_sum[i];
        delta_A_beta_sq[i] = rho*delta_A_beta_sq[i] + (1 - rho) * pow(delta_A_beta[i],2);
        A_beta[i] = A_beta[i] + delta_A_beta[i];
      }

      if(p > 1){
        for(i = 0; i < L_beta_ind; i++){
          E_L_beta_sq[i] = rho * E_L_beta_sq[i] + (1 - rho) * pow(L_beta_gradient_sum[i],2);
          delta_L_beta[i] = sqrt(delta_L_beta_sq[i]+adadelta_noise)/sqrt(E_L_beta_sq[i]+adadelta_noise)*L_beta_gradient_sum[i];
          delta_L_beta_sq[i] = rho*delta_L_beta_sq[i] + (1 - rho) * pow(delta_L_beta[i],2);
          L_beta[i] = L_beta[i] + delta_L_beta[i];
        }
      }

      ELBO = 0.0;
      zeros(sum_v,n);

      double sum2 = 0.0;
      double sum3 = 0.0;
      double sum31 = 0.0;
      double sum4 = 0.0;
      double sum41 = 0.0;
      double sum5 = 0.0;

      for(int k = 0; k < Trace_N; k++){
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }
        for(int i = 0; i < p; i++){
          z_vec[i] = rnorm(0, 1);
        }
        update_uvec_ubvec(u_vec, ub_vec, epsilon_vec,  z_vec,
                          A_vi,  A_beta,  L_beta, S_vi, E_vi, n, p, nnIndxLU_vi,  nnIndx_vi, IndxLU_beta);
        sum_two_vec(u_vec, w_mu_update, sum_v, n);
        for(int i = 0; i < n; i++){
          sum3 += pow((y[i] - w_mu_update[i] -F77_NAME(ddot)(&p, &X[i], &n, beta, &inc)),2)/theta[tauSqIndx]*0.5;
          sum31 += pow((u_vec[i] + F77_NAME(ddot)(&p, &X[i], &n, ub_vec, &inc)),2)/theta[tauSqIndx]*0.5;
        }
        sum2 += Q(B, F, sum_v, sum_v, n, nnIndx, nnIndxLU)*0.5;
      }

      for(int i = 0; i < n; i++){
        sum4 += log(2*pi*S_vi[i]);
        sum5 += log(2*pi*F[i]);
      }
      for(int i = 0; i < p; i++){
        sum41 += log(2*pi*E_vi[i]);
      }

      ELBO = (sum2 + sum3 + sum31)/Trace_N;

      ELBO += -0.5*sum4;

      ELBO += -0.5*sum41;

      ELBO += 0.5*n*log(2*pi*theta[tauSqIndx]);

      ELBO += 0.5*sum5;

      ELBO += -0.5*n;

      ELBO_vec[iter-1] = -ELBO;
      // L_beta_track[iter - 1] = L_beta[0];
      // A_beta_track[iter - 1] = A_beta[0];
      // E_vi_track[iter - 1] = E_vi[0];
      // if(iter == 1){max_ELBO = - ELBO;}
      // if(iter > min_iter & iter % 10){
      //   if(- ELBO<max_ELBO){ELBO_convergence_count+=1;}else{ELBO_convergence_count=0;}
      //   max_ELBO = max(max_ELBO, - ELBO);
      //   if(stop_K){
      //     indicator_converge = ELBO_convergence_count>=K;
      //   }
      // }

      if(iter == min_iter){max_ELBO = - ELBO;}
      if (iter > min_iter && iter % 10 == 0){

        int count = 0;
        double sum = 0.0;
        for (int i = iter - 10; i < iter; i++) {
          sum += ELBO_vec[i];
          count++;
        }

        double average =  sum / count;

        if(average < max_ELBO){ELBO_convergence_count+=1;}else{ELBO_convergence_count=0;}
        max_ELBO = max(max_ELBO, average);


        if(stop_K){
          indicator_converge = ELBO_convergence_count>=K;
        }
      }

      if(!verbose){
        int percent = (iter * 100) / max_iter;
        int progressMarks = percent / 10;

        if (iter == max_iter || iter % (max_iter / 10) == 0) {
          Rprintf("\r[");

          for (int j = 0; j < progressMarks; j++) {
            Rprintf("*");
          }

          for (int j = progressMarks; j < 10; j++) {
            Rprintf("-");
          }

          Rprintf("] %d%%\n", percent);

#ifdef Win32
          R_FlushConsole();
#endif
        }
      }

      if(indicator_converge == 1){
        Rprintf("Early convergence reached at iteration at %i \n", iter);
      }
#ifdef Win32
      R_FlushConsole();
#endif


      iter++;
      vi_error = 0.0;
      for(i = 0; i < n; i++){
        vi_error += abs(w_mu_update[i] - w_mu[i]) ;
      }

      F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);

    }



    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

    SEXP theta_para_r; PROTECT(theta_para_r = allocVector(REALSXP, nTheta*2)); nProtect++; double *theta_para = REAL(theta_para_r);

    theta_para[zetaSqIndx*2+0] = a_zeta_update;
    theta_para[zetaSqIndx*2+1] = b_zeta_update;

    theta_para[tauSqIndx*2+0] = a_tau_update;
    theta_para[tauSqIndx*2+1] = b_tau_update;

    theta_para[phiIndx*2+0] = a_phi;
    theta_para[phiIndx*2+1] = b_phi;

    //SEXP x_out; PROTECT(x_out = allocVector(REALSXP, n)); nProtect++;
    //REAL(x_out)[0] = phi_can;
    //F77_NAME(dcopy)(&n, w_mu, &inc, REAL(x_out), &inc);
    SEXP iter_r; PROTECT(iter_r = allocVector(INTSXP, 1)); nProtect++;
    INTEGER(iter_r)[0] = iter;

    SEXP result_r, resultName_r;
    int nResultListObjs = 28;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, nnIndxLU_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("nnIndxLU"));

    SET_VECTOR_ELT(result_r, 1, CIndx_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("CIndx"));

    SET_VECTOR_ELT(result_r, 2, nnIndx_r);
    SET_VECTOR_ELT(resultName_r, 2, mkChar("nnIndx"));

    SET_VECTOR_ELT(result_r, 3, numIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 3, mkChar("numIndxCol"));

    SET_VECTOR_ELT(result_r, 4, cumnumIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 4, mkChar("cumnumIndxCol"));

    SET_VECTOR_ELT(result_r, 5, nnIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 5, mkChar("nnIndxCol"));

    SET_VECTOR_ELT(result_r, 6, nnIndxnnCol_r);
    SET_VECTOR_ELT(resultName_r, 6, mkChar("nnIndxnnCol"));

    SET_VECTOR_ELT(result_r, 7, nnIndxLU_vi_r);
    SET_VECTOR_ELT(resultName_r, 7, mkChar("nnIndxLU_vi"));

    SET_VECTOR_ELT(result_r, 8, nnIndx_vi_r);
    SET_VECTOR_ELT(resultName_r, 8, mkChar("nnIndx_vi"));

    SET_VECTOR_ELT(result_r, 9, numIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 9, mkChar("numIndxCol_vi"));

    SET_VECTOR_ELT(result_r, 10, cumnumIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 10, mkChar("cumnumIndxCol_vi"));

    SET_VECTOR_ELT(result_r, 11, nnIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 11, mkChar("nnIndxCol_vi"));

    SET_VECTOR_ELT(result_r, 12, nnIndxnnCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 12, mkChar("nnIndxnnCol_vi"));

    SET_VECTOR_ELT(result_r, 13, B_r);
    SET_VECTOR_ELT(resultName_r, 13, mkChar("B"));

    SET_VECTOR_ELT(result_r, 14, F_r);
    SET_VECTOR_ELT(resultName_r, 14, mkChar("F"));

    SET_VECTOR_ELT(result_r, 15, theta_r);
    SET_VECTOR_ELT(resultName_r, 15, mkChar("theta"));

    SET_VECTOR_ELT(result_r, 16, w_mu_r);
    SET_VECTOR_ELT(resultName_r, 16, mkChar("w_mu"));

    SET_VECTOR_ELT(result_r, 17, A_vi_r);
    SET_VECTOR_ELT(resultName_r, 17, mkChar("A_vi"));

    SET_VECTOR_ELT(result_r, 18, S_vi_r);
    SET_VECTOR_ELT(resultName_r, 18, mkChar("S_vi"));

    SET_VECTOR_ELT(result_r, 19, iter_r);
    SET_VECTOR_ELT(resultName_r, 19, mkChar("iter"));

    SET_VECTOR_ELT(result_r, 20, ELBO_vec_r);
    SET_VECTOR_ELT(resultName_r, 20, mkChar("ELBO_vec"));

    SET_VECTOR_ELT(result_r, 21, theta_para_r);
    SET_VECTOR_ELT(resultName_r, 21, mkChar("theta_para"));

    SET_VECTOR_ELT(result_r, 22, beta_r);
    SET_VECTOR_ELT(resultName_r, 22, mkChar("beta"));

    SET_VECTOR_ELT(result_r, 23, beta_cov_r);
    SET_VECTOR_ELT(resultName_r, 23, mkChar("beta_cov"));

    SET_VECTOR_ELT(result_r, 24, A_beta_r);
    SET_VECTOR_ELT(resultName_r, 24, mkChar("A_beta"));

    SET_VECTOR_ELT(result_r, 25, L_beta_r);
    SET_VECTOR_ELT(resultName_r, 25, mkChar("L_beta"));

    SET_VECTOR_ELT(result_r, 26, E_vi_r);
    SET_VECTOR_ELT(resultName_r, 26, mkChar("E_vi"));

    // SET_VECTOR_ELT(result_r, 27, L_beta_track_r);
    // SET_VECTOR_ELT(resultName_r, 27, mkChar("L_beta_track"));
    // 
    // SET_VECTOR_ELT(result_r, 28, A_beta_track_r);
    // SET_VECTOR_ELT(resultName_r, 28, mkChar("A_beta_track"));
    // 
    // SET_VECTOR_ELT(result_r, 29, E_vi_track_r);
    // SET_VECTOR_ELT(resultName_r, 29, mkChar("E_vi_track"));

    SET_VECTOR_ELT(result_r, 27, IndxLU_beta_r);
    SET_VECTOR_ELT(resultName_r, 27, mkChar("IndxLU_beta"));
    // SET_VECTOR_ELT(result_r, 20, uiIndx_r);
    // SET_VECTOR_ELT(resultName_r, 20, mkChar("uiIndx"));
    //
    // SET_VECTOR_ELT(result_r, 21, uIndx_r);
    // SET_VECTOR_ELT(resultName_r, 21, mkChar("uIndx"));
    //
    // SET_VECTOR_ELT(result_r, 22, uIndxLU_r);
    // SET_VECTOR_ELT(resultName_r, 22, mkChar("uIndxLU"));

    namesgets(result_r, resultName_r);
    //unprotect
    UNPROTECT(nProtect);


    return(result_r);

  }

  SEXP spVarBayes_NNGP_betacpp(SEXP y_r, SEXP X_r,
                               SEXP n_r, SEXP p_r, SEXP m_r, SEXP m_vi_r, SEXP coords_r, SEXP covModel_r, SEXP rho_r,
                                                SEXP zetaSqIG_r, SEXP tauSqIG_r, SEXP phibeta_r, SEXP nuUnif_r,
                                                SEXP zetaSqStarting_r, SEXP tauSqStarting_r, SEXP phiStarting_r, SEXP nuStarting_r,
                                                SEXP sType_r, SEXP nThreads_r, SEXP verbose_r, SEXP fix_nugget_r, SEXP N_phi_r, SEXP Trace_N_r,
                                                SEXP max_iter_r,
                                                SEXP var_input_r,
                                                SEXP phi_input_r, SEXP phi_iter_max_r, SEXP initial_mu_r,
                                                SEXP min_iter_r, SEXP K_r, SEXP stop_K_r){


    int h, i, j, k, l, s, info, nProtect=0;
    const int inc = 1;
    const double one = 1.0;
    const double negOne = -1.0;
    const double zero = 0.0;
    char const *lower = "L";
    char const *upper = "U";
    char const *ntran = "N";
    char const *ytran = "T";
    char const *rside = "R";
    char const *lside = "L";
    const double pi = 3.1415926;
    //get args
    double *y = REAL(y_r);
    double *X = REAL(X_r);
    int p = INTEGER(p_r)[0];
    int n = INTEGER(n_r)[0];
    int m = INTEGER(m_r)[0];
    int m_vi = INTEGER(m_vi_r)[0];
    double *coords = REAL(coords_r);
    double fix_nugget = REAL(fix_nugget_r)[0];
    int covModel = INTEGER(covModel_r)[0];
    std::string corName = getCorName(covModel);
    //double converge_per  =  REAL(converge_per_r)[0];
    double phi_input  =  REAL(phi_input_r)[0];
    double *var_input  =  REAL(var_input_r);
    int initial_mu  =  INTEGER(initial_mu_r)[0];
    int phi_iter_max = INTEGER(phi_iter_max_r)[0];
    int nThreads = INTEGER(nThreads_r)[0];
    int verbose = INTEGER(verbose_r)[0];

    int K = INTEGER(K_r)[0];
    int stop_K = INTEGER(stop_K_r)[0];
    int min_iter = INTEGER(min_iter_r)[0];

    //double  vi_threshold  =  REAL(vi_threshold_r)[0];
    double  rho  =  REAL(rho_r)[0];
    //double  rho_phi  =  REAL(rho_phi_r)[0];
    //priors
    double zetaSqIGa = REAL(zetaSqIG_r)[0]; double zetaSqIGb = REAL(zetaSqIG_r)[1];
    double tauSqIGa = REAL(tauSqIG_r)[0]; double tauSqIGb = REAL(tauSqIG_r)[1];
    //double phiUnifa = REAL(phiUnif_r)[0]; double phiUnifb = REAL(phiUnif_r)[1];
    double phimin = REAL(phibeta_r)[0]; double phimax = REAL(phibeta_r)[1];

    double a_phi = (phi_input - phimin)/(phimax-phimin)*10;
    double b_phi = 10 - a_phi;

    double nuUnifa = 0, nuUnifb = 0;
    if(corName == "matern"){
      nuUnifa = REAL(nuUnif_r)[0]; nuUnifb = REAL(nuUnif_r)[1];
    }

#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#else
    if(nThreads > 1){
      warning("n.omp.threads > %i, but source not compiled with OpenMP support.", nThreads);
      nThreads = 1;
    }
#endif

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tModel description\n");
      Rprintf("----------------------------------------\n");
      Rprintf("NNGP Latent model fit with %i observations.\n\n", n);
      Rprintf("Number of covariates %i (including intercept if specified).\n\n", p);
      Rprintf("Using the %s spatial correlation model.\n\n", corName.c_str());
      Rprintf("Using %i nearest neighbors.\n\n", m);
      Rprintf("Priors and hyperpriors:\n");
      Rprintf("\tbeta flat.\n");
#ifdef _OPENMP
      Rprintf("\nSource compiled with OpenMP support and model fit using %i thread(s).\n", nThreads);
#else
      Rprintf("\n\nSource not compiled with OpenMP support.\n");
#endif
    }

    //parameters
    int nTheta, zetaSqIndx, tauSqIndx, phiIndx, nuIndx;

    if(corName != "matern"){
      nTheta = 3;//zeta^2, tau^2, phi
      zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
    }else{
      nTheta = 4;//zeta^2, tau^2, phi, nu
      zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2; nuIndx = 3;
    }

    //allocated for the nearest neighbor index vector (note, first location has no neighbors).
    int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);

    SEXP nnIndx_r; PROTECT(nnIndx_r = allocVector(INTSXP, nIndx)); nProtect++; int *nnIndx = INTEGER(nnIndx_r);

    //int *nnIndx = (int *) R_alloc(nIndx, sizeof(int));

    double *d = (double *) R_alloc(nIndx, sizeof(double));

    SEXP nnIndxLU_r; PROTECT(nnIndxLU_r = allocVector(INTSXP, 2*n)); nProtect++; int *nnIndxLU = INTEGER(nnIndxLU_r); //first column holds the nnIndx index for the i-th location and the second columns holds the number of neighbors the i-th location has (the second column is a bit of a waste but will simplifying some parallelization).

    //int *nnIndxLU = (int *) R_alloc(2*n, sizeof(int));

    //make the neighbor index
    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tBuilding neighbor index\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }

    if(INTEGER(sType_r)[0] == 0){
      mkNNIndx(n, m, coords, nnIndx, d, nnIndxLU);
    }
    if(INTEGER(sType_r)[0] == 1){
      mkNNIndxTree0(n, m, coords, nnIndx, d, nnIndxLU);
    }else{
      mkNNIndxCB(n, m, coords, nnIndx, d, nnIndxLU);
    }

    SEXP CIndx_r; PROTECT(CIndx_r = allocVector(INTSXP, 2*n)); nProtect++; int *CIndx = INTEGER(CIndx_r); //index for D and C.

    //int *CIndx = (int *) R_alloc(2*n, sizeof(int));
    for(i = 0, j = 0; i < n; i++){//zero should never be accessed
      j += nnIndxLU[n+i]*nnIndxLU[n+i];
      if(i == 0){
        CIndx[n+i] = 0;
        CIndx[i] = 0;
      }else{
        CIndx[n+i] = nnIndxLU[n+i]*nnIndxLU[n+i];
        CIndx[i] = CIndx[n+i-1] + CIndx[i-1];
      }
    }

    SEXP numIndxCol_r; PROTECT(numIndxCol_r = allocVector(INTSXP, n)); nProtect++; int *numIndxCol = INTEGER(numIndxCol_r); zeros_int(numIndxCol, n);
    get_num_nIndx_col(nnIndx, nIndx, numIndxCol);

    SEXP cumnumIndxCol_r; PROTECT(cumnumIndxCol_r = allocVector(INTSXP, n)); nProtect++; int *cumnumIndxCol = INTEGER(cumnumIndxCol_r); zeros_int(cumnumIndxCol,n);
    get_cumnum_nIndx_col(numIndxCol, n, cumnumIndxCol);

    SEXP nnIndxCol_r; PROTECT(nnIndxCol_r = allocVector(INTSXP, nIndx+n)); nProtect++; int *nnIndxCol = INTEGER(nnIndxCol_r); zeros_int(nnIndxCol, n);
    get_nnIndx_col(nnIndx, n, nIndx, cumnumIndxCol, numIndxCol, nnIndxCol);

    int *sumnnIndx = (int *) R_alloc(n-1, sizeof(int));; zeros_int(sumnnIndx, n-1);
    get_sum_nnIndx(sumnnIndx, n, m);

    SEXP nnIndxnnCol_r; PROTECT(nnIndxnnCol_r = allocVector(INTSXP, nIndx)); nProtect++; int *nnIndxnnCol = INTEGER(nnIndxnnCol_r); zeros_int(nnIndxnnCol, n);
    get_nnIndx_nn_col(nnIndx, n, m, nIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, sumnnIndx);


    double *D = (double *) R_alloc(j, sizeof(double));

    for(i = 0; i < n; i++){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        for(l = 0; l <= k; l++){
          D[CIndx[i]+l*nnIndxLU[n+i]+k] = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
        }
      }
    }
    int mm = m*m;
    SEXP B_r; PROTECT(B_r = allocVector(REALSXP, nIndx)); nProtect++; double *B = REAL(B_r);
    SEXP F_r; PROTECT(F_r = allocVector(REALSXP, n)); nProtect++; double *F = REAL(F_r);


    //double *B = (double *) R_alloc(nIndx, sizeof(double));
    //double *F = (double *) R_alloc(n, sizeof(double));

    double *c =(double *) R_alloc(m*nThreads, sizeof(double));
    double *C = (double *) R_alloc(mm*nThreads, sizeof(double));

    SEXP beta_r; PROTECT(beta_r = allocVector(REALSXP, p)); nProtect++; double *beta = REAL(beta_r); zeros(beta, p);

    SEXP beta_cov_r; PROTECT(beta_cov_r = allocVector(REALSXP, p*p)); nProtect++; double *beta_cov = REAL(beta_cov_r); zeros(beta_cov, p*p);

    SEXP theta_r; PROTECT(theta_r = allocVector(REALSXP, nTheta)); nProtect++; double *theta = REAL(theta_r);

    SEXP w_mu_r; PROTECT(w_mu_r = allocVector(REALSXP, n)); nProtect++; double *w_mu = REAL(w_mu_r);

    SEXP sigma_sq_r; PROTECT(sigma_sq_r = allocVector(REALSXP, n)); nProtect++; double *sigma_sq = REAL(sigma_sq_r);

    //double *beta = (double *) R_alloc(p, sizeof(double)); zeros(beta, p);
    //double *theta = (double *) R_alloc(nTheta, sizeof(double));

    // theta[0] = REAL(zetaSqStarting_r)[0];
    // theta[1] = REAL(phiStarting_r)[0];
    //
    // if(corName == "matern"){
    //   theta[2] = REAL(nuStarting_r)[0];
    // }
    //

    theta[zetaSqIndx] = REAL(zetaSqStarting_r)[0];
    theta[tauSqIndx] = REAL(tauSqStarting_r)[0];
    //theta[phiIndx] = REAL(phiStarting_r)[0];
    theta[phiIndx] = phi_input;
    if(corName == "matern"){
      theta[nuIndx] = REAL(nuStarting_r)[0];
    }

    //other stuff
    double logDetInv;
    int accept = 0, batchAccept = 0, status = 0;
    int jj, kk, pp = p*p, nn = n*n, np = n*p;
    double *one_n = (double *) R_alloc(n, sizeof(double)); ones(one_n, n);
    //double *tmp_n = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n, n);

    double *tmp_pp = (double *) R_alloc(pp, sizeof(double));
    double *tmp_p = (double *) R_alloc(p, sizeof(double));
    double *tmp_p2 = (double *) R_alloc(p, sizeof(double)); zeros(tmp_p2, p);
    double *tmp_n = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n, n);
    double *XtX = (double *) R_alloc(pp, sizeof(double));
    double *tau_sq_H = (double *) R_alloc(one, sizeof(double));
    double *tau_sq_I = (double *) R_alloc(one, sizeof(double));

    //double *w_mu = (double *) R_alloc(n, sizeof(double));
    //zeros(w_mu, n);
    F77_NAME(dgemm)(ytran, ntran, &p, &p, &n, &one, X, &n, X, &n, &zero, XtX, &p FCONE FCONE);

    if(initial_mu){
      F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, y, &inc, &zero, tmp_p, &inc FCONE);

      for(i = 0; i < pp; i++){
        tmp_pp[i] = XtX[i];
      }

      F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotrf failed\n");}
      F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotri failed\n");}

      F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc FCONE);

      for(i = 0; i < n; i++){
        w_mu[i] = y[i] - F77_NAME(ddot)(&p, &X[i], &n, tmp_p2, &inc);
      }
      // F77_NAME(dcopy)(&n, y, &inc, w_mu, &inc);
    }else{
      zeros(w_mu, n);
    }
    //double *sigma_sq = (double *) R_alloc(n, sizeof(double));
    ones(sigma_sq, n);

    double *w_mu_update = (double *) R_alloc(n, sizeof(double)); zeros(w_mu_update, n);
    double *E_mu_sq = (double *) R_alloc(n, sizeof(double)); zeros(E_mu_sq, n);
    double *delta_mu_sq = (double *) R_alloc(n, sizeof(double)); zeros(delta_mu_sq, n);
    double *delta_mu = (double *) R_alloc(n, sizeof(double)); zeros(delta_mu, n);
    double *m_mu = (double *) R_alloc(n, sizeof(double)); zeros(m_mu, n);

    double *sigma_sq_update = (double *) R_alloc(n, sizeof(double)); ones(sigma_sq_update, n);

    double a, v, b, e, mu, var, aij, phiCand, nuCand = 0, nu = 0;

    double a_tau_update = n * 0.5 + tauSqIGa;
    double b_tau_update = 0.0;
    double tau_sq = 0.0;

    double a_zeta_update = n * 0.5 + zetaSqIGa;
    double b_zeta_update = 0.0;
    double zeta_sq = 0.0;
    int N_phi = INTEGER(N_phi_r)[0];
    int Trace_N = INTEGER(Trace_N_r)[0];
    int one_int = 1;
    int three_int = 3;
    double adadelta_noise = 0.0000001;
    double *bk = (double *) R_alloc(nThreads*(1.0+5.0), sizeof(double));
    //double *bk = (double *) R_alloc(nThreads*(1.0+static_cast<int>(floor(nuUnifb))), sizeof(double));
    if(corName == "matern"){nu = theta[nuIndx];}

    //int iter = 1;
    int max_iter = INTEGER(max_iter_r)[0];
    //int iter = (int ) R_alloc(one_int, sizeof(int)); iter = 1;
    int iter = 1;

    double vi_error = 1.0;
    double rho1 = 0.9;
    double rho2 = 0.999;
    double adaptive_adam = 0.001;
    //double vi_threshold = 0.0001;

    // SEXP a_track_r; PROTECT(a_track_r = allocVector(REALSXP, max_iter)); nProtect++; double *a_track = REAL(a_track_r);
    // SEXP gamma_track_r; PROTECT(gamma_track_r = allocVector(REALSXP, max_iter)); nProtect++; double *gamma_track = REAL(gamma_track_r);

    // NNGP parameters

    //allocated for the nearest neighbor index vector (note, first location has no neighbors).
    int nIndx_vi = static_cast<int>(static_cast<double>(1+m_vi)/2*m_vi+(n-m_vi-1)*m_vi);

    SEXP nnIndx_vi_r; PROTECT(nnIndx_vi_r = allocVector(INTSXP, nIndx_vi)); nProtect++; int *nnIndx_vi = INTEGER(nnIndx_vi_r);

    double *d_vi = (double *) R_alloc(nIndx_vi, sizeof(double));

    SEXP nnIndxLU_vi_r; PROTECT(nnIndxLU_vi_r = allocVector(INTSXP, 2*n)); nProtect++; int *nnIndxLU_vi = INTEGER(nnIndxLU_vi_r); //first column holds the nnIndx index for the i-th location and the second columns holds the number of neighbors the i-th location has (the second column is a bit of a waste but will simplifying some parallelization).

    //make the neighbor index
    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tBuilding neighbor index for variational inference \n");
      Rprintf("Using %i nearest neighbors.\n\n", m_vi);
#ifdef Win32
      R_FlushConsole();
#endif
    }

    if(INTEGER(sType_r)[0] == 0){
      mkNNIndx(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }
    if(INTEGER(sType_r)[0] == 1){
      mkNNIndxTree0(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }else{
      mkNNIndxCB(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }


    int mm_vi = m_vi*m_vi;
    SEXP A_vi_r; PROTECT(A_vi_r = allocVector(REALSXP, nIndx_vi)); nProtect++; double *A_vi = REAL(A_vi_r); zeros(A_vi,nIndx_vi);
    SEXP S_vi_r; PROTECT(S_vi_r = allocVector(REALSXP, n)); nProtect++; double *S_vi = REAL(S_vi_r); ones(S_vi,n);
    for(int i = 0; i < n; i++){
      S_vi[i] = var_input[i];
    }
    SEXP numIndxCol_vi_r; PROTECT(numIndxCol_vi_r = allocVector(INTSXP, n)); nProtect++; int *numIndxCol_vi = INTEGER(numIndxCol_vi_r); zeros_int(numIndxCol_vi, n);
    get_num_nIndx_col(nnIndx_vi, nIndx_vi, numIndxCol_vi);

    SEXP cumnumIndxCol_vi_r; PROTECT(cumnumIndxCol_vi_r = allocVector(INTSXP, n)); nProtect++; int *cumnumIndxCol_vi = INTEGER(cumnumIndxCol_vi_r); zeros_int(cumnumIndxCol_vi,n);
    get_cumnum_nIndx_col(numIndxCol_vi, n, cumnumIndxCol_vi);

    SEXP nnIndxCol_vi_r; PROTECT(nnIndxCol_vi_r = allocVector(INTSXP, nIndx_vi+n)); nProtect++; int *nnIndxCol_vi = INTEGER(nnIndxCol_vi_r); zeros_int(nnIndxCol_vi, n);
    get_nnIndx_col(nnIndx_vi, n, nIndx_vi, cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi);

    int *sumnnIndx_vi = (int *) R_alloc(n-1, sizeof(int));; zeros_int(sumnnIndx_vi, n-1);
    get_sum_nnIndx(sumnnIndx_vi, n, m_vi);

    SEXP nnIndxnnCol_vi_r; PROTECT(nnIndxnnCol_vi_r = allocVector(INTSXP, nIndx_vi)); nProtect++; int *nnIndxnnCol_vi = INTEGER(nnIndxnnCol_vi_r); zeros_int(nnIndxnnCol_vi, n);
    get_nnIndx_nn_col(nnIndx_vi, n, m_vi, nIndx_vi, cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi, sumnnIndx_vi);

    double *E_a_sq = (double *) R_alloc(nIndx_vi, sizeof(double)); zeros(E_a_sq, nIndx_vi);
    double *delta_a_sq = (double *) R_alloc(nIndx_vi, sizeof(double)); zeros(delta_a_sq, nIndx_vi);
    double *delta_a = (double *) R_alloc(nIndx_vi, sizeof(double)); zeros(delta_a, nIndx_vi);

    double *E_gamma_sq = (double *) R_alloc(n, sizeof(double)); zeros(E_gamma_sq, n);
    double *delta_gamma_sq = (double *) R_alloc(n, sizeof(double)); zeros(delta_gamma_sq, n);
    double *delta_gamma = (double *) R_alloc(n, sizeof(double)); zeros(delta_gamma, n);
    double *gamma_vec = (double *) R_alloc(n, sizeof(double));zeros(gamma_vec, n);
    //double *B = (double *) R_alloc(nIndx, sizeof(double));
    //double *F = (double *) R_alloc(n, sizeof(double));



    double *epsilon_vec = (double *) R_alloc(n, sizeof(double));

    for(int i = 0; i < n; i++){
      epsilon_vec[i] = rnorm(0, 1);
    }

    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);



    // int n_per = nIndx_vi * converge_per;
    // int *sign_vec_old = (int *) R_alloc(n_per, sizeof(int));
    // int *sign_vec_new = (int *) R_alloc(n_per, sizeof(int));
    // int *check_vec = (int *) R_alloc(n_per, sizeof(int));
    int indicator_converge = 0;

    double *trace_vec = (double *) R_alloc(2, sizeof(double));
    double *u_vec = (double *) R_alloc(n, sizeof(double));
    double *u_vec_mean = (double *) R_alloc(n, sizeof(double));
    double ELBO_MC = 0.0;
    double ELBO = 0.0;
    //double *epsilon_vec = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp2 = (double *) R_alloc(n, sizeof(double));
    double *gradient_const = (double *) R_alloc(n, sizeof(double));
    double *gradient = (double *) R_alloc(n, sizeof(double));
    double *gamma_gradient_sum = (double *) R_alloc(n, sizeof(double));
    double *gamma_gradient = (double *) R_alloc(n, sizeof(double));
    double *a_gradient = (double *) R_alloc(nIndx_vi, sizeof(double));
    double *a_gradient_sum = (double *) R_alloc(nIndx_vi, sizeof(double));

    SEXP ELBO_vec_r; PROTECT(ELBO_vec_r = allocVector(REALSXP, max_iter)); nProtect++;
    double *ELBO_vec = REAL(ELBO_vec_r); zeros(ELBO_vec,max_iter);
    double max_ELBO = 0.0;
    int ELBO_convergence_count = 0;

    double E_phi_sq = 0.0;
    double delta_phi = 0.0;
    double delta_phi_sq = 0.0;
    double *rademacher_rv_vec = (double *) R_alloc(n, sizeof(double));
    double *rademacher_rv_temp = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp,n);
    double *rademacher_rv_temp2 = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp2,n);

    double *product_v = (double *) R_alloc(n, sizeof(double));zeros(product_v,n);
    double *product_v2 = (double *) R_alloc(n, sizeof(double));zeros(product_v2,n);
    double *phi_can_vec = (double *) R_alloc(N_phi*N_phi, sizeof(double));zeros(phi_can_vec,N_phi*N_phi);
    double *log_g_phi = (double *) R_alloc(N_phi*N_phi, sizeof(double));zeros(log_g_phi,N_phi*N_phi);
    double *sum_v = (double *) R_alloc(n, sizeof(double));zeros(sum_v,n);

    while(iter <= max_iter & !indicator_converge){

      if(verbose){
        Rprintf("----------------------------------------\n");
        Rprintf("\tIteration at %i \n",iter);
#ifdef Win32
        R_FlushConsole();
#endif
      }

      ///////////////
      //update beta
      ///////////////


      zeros(tau_sq_I, one_int);
      for(i = 0; i < n; i++){
        tmp_n[i] = y[i]-w_mu[i];
        tau_sq_I[0] += pow(tmp_n[i],2);
      }

      F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, tmp_n, &inc, &zero, tmp_p, &inc FCONE);

      for(i = 0; i < pp; i++){
        tmp_pp[i] = XtX[i];
      }

      F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotrf failed\n");}
      F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotri failed\n");}

      F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc FCONE);

      //F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 3 dpotrf failed\n");}

      F77_NAME(dcopy)(&p, tmp_p2, &inc, beta, &inc);

      for (int i = 0; i < p; i++) {
        for (int j = 0; j <= i; j++) {
          // Calculate the index for column-major format
          int idx = i + j * p;
          beta_cov[idx] = tmp_pp[idx] * theta[tauSqIndx];
        }
      }

      if(verbose){
        for(i = 0; i < p; i++){
          Rprintf("the value of beta[%i] : %f \n",i, beta[i]);
        }
        for(i = 0; i < pp; i++){
          Rprintf("the value of beta cov[%i] : %f \n",i, beta_cov[i]);
        }
#ifdef Win32
        R_FlushConsole();
#endif
      }


      ///////////////
      //update tausq
      ///////////////

      zeros(tau_sq_H, one_int);

      for(i = 0; i < p; i++){
        tau_sq_H[0] += tmp_p2[i]*tmp_p[i];
      }

      zeros(trace_vec,2);
      zeros(u_vec,n);

      int Trace_N_max = Trace_N;
      for(int i = 0; i < n; i++){
        epsilon_vec[i] = rnorm(0, 1);

      }
      update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);

      for(int k = 0; k < Trace_N_max; k++){
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }
        update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);

        double u_mean = 0.0;
        for(i = 0; i < n; i++){
          u_mean += u_vec[i];
        }
        u_mean = u_mean/n;

        for(i = 0; i < n; i++){
          trace_vec[0] += pow(u_vec[i]-u_mean,2);
        }

        trace_vec[1] += Q(B, F, u_vec, u_vec, n, nnIndx, nnIndxLU);
      }


      b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N_max + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
      //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + *tau_sq_I)*0.5;
      tau_sq = b_tau_update/a_tau_update;
      theta[tauSqIndx] = tau_sq;

      if(verbose){
        Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
        R_FlushConsole();
#endif
      }

      //Rprintf("the value of add_tau : %f \n", trace_vec[0]/Trace_N + *tau_sq_I);
      ///////////////
      //update zetasq
      ///////////////

      updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

      double zeta_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
      b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N_max + zeta_Q)*theta[zetaSqIndx]*0.5;
      //Rprintf("zeta_Q: %f \n", zeta_Q);
      //b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q)*0.5;
      zeta_sq = b_zeta_update/a_zeta_update;
      //Rprintf("the value of add_zeta : %f \n", (trace_vec[1]/Trace_N + zeta_Q));
      theta[zetaSqIndx] = zeta_sq;

      if(verbose){
        Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
        R_FlushConsole();
#endif
      }
      updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
      ///////////////
      //update phi
      ///////////////

      if(iter < phi_iter_max){

        double *a_phi_vec = (double *) R_alloc(N_phi, sizeof(double));
        double *b_phi_vec = (double *) R_alloc(N_phi, sizeof(double));
        a_phi_vec[0] = a_phi;
        b_phi_vec[0] = b_phi;

        for(int i = 1; i < N_phi; i++){
          if (i % 2 == 0) {
            a_phi_vec[i] = a_phi_vec[0] + 0.01*i;
            b_phi_vec[i] = b_phi_vec[0] + 0.01*i;
            // a_phi_vec[i] = a_phi_vec[0]*(1+0.1*i);
            // b_phi_vec[i] = b_phi_vec[0]*(1+0.1*i);
          } else {
            a_phi_vec[i] = a_phi_vec[0] + 0.01*i*(-1);
            b_phi_vec[i] = b_phi_vec[0] + 0.01*i*(-1);
            // a_phi_vec[i] = a_phi_vec[0]*(1-0.1*i);
            // b_phi_vec[i] = b_phi_vec[0]*(1-0.1*i);
          }

        }

        double phi_Q = 0.0;
        double diag_sigma_sq_sum = 0.0;

        int max_index;
        zeros(phi_can_vec,N_phi*N_phi);
        zeros(log_g_phi,N_phi*N_phi);
        for(int i = 0; i < N_phi; i++){
          for(int j = 0; j < N_phi; j++){

            for(int k = 0; k < Trace_N; k++){
              phi_can_vec[i*N_phi+j] += rbeta(a_phi_vec[i], b_phi_vec[j]);  // Notice the indexing here
            }
            phi_can_vec[i*N_phi+j] /= Trace_N;
            phi_can_vec[i*N_phi+j] = phi_can_vec[i*N_phi+j]*(phimax - phimin) + phimin;
          }
        }


        for(i = 0; i < N_phi*N_phi; i++){

          updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                   theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb);

          //phi_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
          phi_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
          update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);
          logDetInv = 0.0;
          diag_sigma_sq_sum = 0.0;
          for(j = 0; j < n; j++){
            logDetInv += log(1/F[j]);
          }

          log_g_phi[i] = logDetInv*0.5 - (phi_Q + Q(B, F, u_vec, u_vec, n, nnIndx, nnIndxLU))*0.5;
        }

        max_index = max_ind(log_g_phi,N_phi*N_phi);
        a_phi = a_phi_vec[max_index/N_phi];
        b_phi = b_phi_vec[max_index % N_phi];

        theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;


        updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                 theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
      }

      if(verbose){
        Rprintf("the value of theta[%i phiIndx] : %f \n", phiIndx, theta[phiIndx]);
#ifdef Win32
        R_FlushConsole();
#endif
      }

      ///////////////
      //update w
      ///////////////

      zeros(w_mu_temp,n);
      zeros(w_mu_temp2,n);

      product_B_F(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp);
      //product_B_F(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2);
      product_B_F_vec(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);

      double gradient_mu = 0.0;
      //Rprintf("w_mu_update: ");
      for(i = 0; i < n; i++){
        //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
        gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc))/theta[tauSqIndx]);

        //Rprintf("%i ",gradient_mu);
        //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i]/theta[zetaSqIndx] + (y[i])/theta[tauSqIndx]);
        E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
        delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
        delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
        w_mu_update[i] = w_mu[i] + delta_mu[i];
        //Rprintf("%f ",w_mu_update[i]);
      }

      product_B_F(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp);
      //product_B_F(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2);
      product_B_F_vec(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);


      // zeros(gradient_const,n);
      // for(i = 0; i < n; i++){
      //   //gradient_const[i] = -w_mu_update[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx];
      //   gradient_const[i] = -w_mu_update[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc))/theta[tauSqIndx];
      // }


      zeros(gradient,n);
      zeros(gamma_gradient_sum, n);
      zeros(gamma_gradient,n);

      for(int k = 0; k < Trace_N; k++){
        zeros(gamma_gradient,n);
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }

        // gamma_gradient_fun(u_vec, epsilon_vec, gamma_gradient, gradient_const, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
        //                    B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
        //                    cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,w_mu_temp,w_mu_temp2,gradient);

        gamma_gradient_fun(u_vec, epsilon_vec, gamma_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                           B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                           cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,w_mu_temp,w_mu_temp2,gradient);


        vecsum(gamma_gradient_sum, gamma_gradient, Trace_N, n);
      }

      //free(gamma_gradient);
      //Rprintf("S_vi: ");
      for(i = 0; i < n; i++){
        E_gamma_sq[i] = rho * E_gamma_sq[i] + (1 - rho) * pow(gamma_gradient_sum[i],2);
        delta_gamma[i] = sqrt(delta_gamma_sq[i]+adadelta_noise)/sqrt(E_gamma_sq[i]+adadelta_noise)*gamma_gradient_sum[i];
        delta_gamma_sq[i] = rho*delta_gamma_sq[i] + (1 - rho) * pow(delta_gamma[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        //S_vi[i] = exp(pow((log(sqrt(S_vi[i])) + delta_gamma[i]),2));
        S_vi[i] = pow(exp(log(sqrt(S_vi[i])) + delta_gamma[i]),2);
        //Rprintf("%f ",S_vi[i]);
        //S_vi[i] = pow(exp(gamma_vec[i]),2);
      }
      //Rprintf("\n");
      zeros(a_gradient,nIndx_vi);
      zeros(a_gradient_sum, nIndx_vi);

      for(int k = 0; k < Trace_N; k++){
        zeros(a_gradient,nIndx_vi);
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }

        // a_gradient_fun(u_vec, epsilon_vec, a_gradient, gradient_const, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
        //                B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
        //                w_mu_temp,w_mu_temp2);

        a_gradient_fun(u_vec, epsilon_vec, a_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                       B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                       w_mu_temp,w_mu_temp2);

        vecsum(a_gradient_sum, a_gradient, Trace_N, nIndx_vi);

      }
      //free(a_gradient);
      //update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);


      //Rprintf("A_vi: ");
      for(i = 0; i < nIndx_vi; i++){
        E_a_sq[i] = rho * E_a_sq[i] + (1 - rho) * pow(a_gradient_sum[i],2);
        delta_a[i] = sqrt(delta_a_sq[i]+adadelta_noise)/sqrt(E_a_sq[i]+adadelta_noise)*a_gradient_sum[i];
        delta_a_sq[i] = rho*delta_a_sq[i] + (1 - rho) * pow(delta_a[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        A_vi[i] = A_vi[i] + delta_a[i];
        //Rprintf("%f ",A_vi[i]);
      }
      ELBO = 0.0;
      zeros(sum_v,n);

      double sum2 = 0.0;
      double sum3 = 0.0;
      double sum4 = 0.0;
      double sum5 = 0.0;

      for(int k = 0; k < Trace_N; k++){
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }
        update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);
        sum_two_vec(u_vec, w_mu_update, sum_v, n);
        for(int i = 0; i < n; i++){
          sum3 += pow((y[i] - sum_v[i] -F77_NAME(ddot)(&p, &X[i], &n, beta, &inc)),2)/theta[tauSqIndx]*0.5;
        }
        sum2 += Q(B, F, sum_v, sum_v, n, nnIndx, nnIndxLU)*0.5;
      }

      for(int i = 0; i < n; i++){
        sum4 += log(2*pi*S_vi[i]);
        sum5 += log(2*pi*F[i]);
      }

      ELBO = (sum2 + sum3)/Trace_N;

      ELBO += -0.5*sum4;

      ELBO += 0.5*n*log(2*pi*theta[tauSqIndx]);

      ELBO += 0.5*sum5;

      ELBO += -0.5*n;

      ELBO_vec[iter-1] = -ELBO;
      // a_track[iter - 1] = A_vi[0];
      // gamma_track[iter - 1] = S_vi[0];

      // if(iter == 1){max_ELBO = - ELBO;}
      // if(iter > min_iter & iter % 10){
      //   if(- ELBO<max_ELBO){ELBO_convergence_count+=1;}else{ELBO_convergence_count=0;}
      //   max_ELBO = max(max_ELBO, - ELBO);
      //   if(stop_K){
      //     indicator_converge = ELBO_convergence_count>=K;
      //   }
      // }

      if(iter == min_iter){max_ELBO = - ELBO;}
      if (iter > min_iter && iter % 10 == 0){

        int count = 0;
        double sum = 0.0;
        for (int i = iter - 10; i < iter; i++) {
          sum += ELBO_vec[i];
          count++;
        }

        double average =  sum / count;

        if(average < max_ELBO){ELBO_convergence_count+=1;}else{ELBO_convergence_count=0;}
        max_ELBO = max(max_ELBO, average);


        if(stop_K){
          indicator_converge = ELBO_convergence_count>=K;
        }
      }

      if(!verbose){
        int percent = (iter * 100) / max_iter;
        int progressMarks = percent / 10;

        if (iter == max_iter || iter % (max_iter / 10) == 0) {
          Rprintf("\r[");

          for (int j = 0; j < progressMarks; j++) {
            Rprintf("*");
          }

          for (int j = progressMarks; j < 10; j++) {
            Rprintf("-");
          }

          Rprintf("] %d%%\n", percent);

#ifdef Win32
          R_FlushConsole();
#endif
        }
      }

      if(indicator_converge == 1){
        Rprintf("Early convergence reached at iteration at %i \n", iter);
      }
#ifdef Win32
      R_FlushConsole();
#endif


      iter++;
      vi_error = 0.0;
      for(i = 0; i < n; i++){
        vi_error += abs(w_mu_update[i] - w_mu[i]) ;
      }

      F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);

    }

    {

        Rprintf("----------------------------------------\n");
        Rprintf("\tWrapping Up the results \n",iter);
#ifdef Win32
        R_FlushConsole();
#endif


      ///////////////
      //update beta
      ///////////////


      zeros(tau_sq_I, one_int);
      for(i = 0; i < n; i++){
        tmp_n[i] = y[i]-w_mu[i];
        tau_sq_I[0] += pow(tmp_n[i],2);
      }

      F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, tmp_n, &inc, &zero, tmp_p, &inc FCONE);

      for(i = 0; i < pp; i++){
        tmp_pp[i] = XtX[i];
      }

      F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotrf failed\n");}
      F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotri failed\n");}

      F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc FCONE);

      //F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 3 dpotrf failed\n");}

      F77_NAME(dcopy)(&p, tmp_p2, &inc, beta, &inc);

      for (int i = 0; i < p; i++) {
        for (int j = 0; j <= i; j++) {
          // Calculate the index for column-major format
          int idx = i + j * p;
          beta_cov[idx] = tmp_pp[idx] * theta[tauSqIndx];
        }
      }


      ///////////////
      //update tausq
      ///////////////

      zeros(tau_sq_H, one_int);

      for(i = 0; i < p; i++){
        tau_sq_H[0] += tmp_p2[i]*tmp_p[i];
      }

      zeros(trace_vec,2);
      zeros(u_vec,n);

      int Trace_N_max = 2000;
      for(int i = 0; i < n; i++){
        epsilon_vec[i] = rnorm(0, 1);

      }
      update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);

      for(int k = 0; k < Trace_N_max; k++){
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }
        update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);

        double u_mean = 0.0;
        for(i = 0; i < n; i++){
          u_mean += u_vec[i];
        }
        u_mean = u_mean/n;

        for(i = 0; i < n; i++){
          trace_vec[0] += pow(u_vec[i]-u_mean,2);
        }

        trace_vec[1] += Q(B, F, u_vec, u_vec, n, nnIndx, nnIndxLU);
      }


      b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N_max + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
      //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + *tau_sq_I)*0.5;
      tau_sq = b_tau_update/a_tau_update;
      theta[tauSqIndx] = tau_sq;


      //Rprintf("the value of add_tau : %f \n", trace_vec[0]/Trace_N + *tau_sq_I);
      ///////////////
      //update zetasq
      ///////////////

      updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

      double zeta_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
      b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N_max + zeta_Q)*theta[zetaSqIndx]*0.5;
      //Rprintf("zeta_Q: %f \n", zeta_Q);
      //b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q)*0.5;
      zeta_sq = b_zeta_update/a_zeta_update;
      //Rprintf("the value of add_zeta : %f \n", (trace_vec[1]/Trace_N + zeta_Q));
      theta[zetaSqIndx] = zeta_sq;

    }

    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

    SEXP theta_para_r; PROTECT(theta_para_r = allocVector(REALSXP, nTheta*2)); nProtect++; double *theta_para = REAL(theta_para_r);

    theta_para[zetaSqIndx*2+0] = a_zeta_update;
    theta_para[zetaSqIndx*2+1] = b_zeta_update;

    theta_para[tauSqIndx*2+0] = a_tau_update;
    theta_para[tauSqIndx*2+1] = b_tau_update;

    theta_para[phiIndx*2+0] = a_phi;
    theta_para[phiIndx*2+1] = b_phi;

    //SEXP x_out; PROTECT(x_out = allocVector(REALSXP, n)); nProtect++;
    //REAL(x_out)[0] = phi_can;
    //F77_NAME(dcopy)(&n, w_mu, &inc, REAL(x_out), &inc);
    SEXP iter_r; PROTECT(iter_r = allocVector(INTSXP, 1)); nProtect++;
    INTEGER(iter_r)[0] = iter;

    SEXP result_r, resultName_r;
    int nResultListObjs = 24;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, nnIndxLU_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("nnIndxLU"));

    SET_VECTOR_ELT(result_r, 1, CIndx_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("CIndx"));

    SET_VECTOR_ELT(result_r, 2, nnIndx_r);
    SET_VECTOR_ELT(resultName_r, 2, mkChar("nnIndx"));

    SET_VECTOR_ELT(result_r, 3, numIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 3, mkChar("numIndxCol"));

    SET_VECTOR_ELT(result_r, 4, cumnumIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 4, mkChar("cumnumIndxCol"));

    SET_VECTOR_ELT(result_r, 5, nnIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 5, mkChar("nnIndxCol"));

    SET_VECTOR_ELT(result_r, 6, nnIndxnnCol_r);
    SET_VECTOR_ELT(resultName_r, 6, mkChar("nnIndxnnCol"));

    SET_VECTOR_ELT(result_r, 7, nnIndxLU_vi_r);
    SET_VECTOR_ELT(resultName_r, 7, mkChar("nnIndxLU_vi"));

    SET_VECTOR_ELT(result_r, 8, nnIndx_vi_r);
    SET_VECTOR_ELT(resultName_r, 8, mkChar("nnIndx_vi"));

    SET_VECTOR_ELT(result_r, 9, numIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 9, mkChar("numIndxCol_vi"));

    SET_VECTOR_ELT(result_r, 10, cumnumIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 10, mkChar("cumnumIndxCol_vi"));

    SET_VECTOR_ELT(result_r, 11, nnIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 11, mkChar("nnIndxCol_vi"));

    SET_VECTOR_ELT(result_r, 12, nnIndxnnCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 12, mkChar("nnIndxnnCol_vi"));

    SET_VECTOR_ELT(result_r, 13, B_r);
    SET_VECTOR_ELT(resultName_r, 13, mkChar("B"));

    SET_VECTOR_ELT(result_r, 14, F_r);
    SET_VECTOR_ELT(resultName_r, 14, mkChar("F"));

    SET_VECTOR_ELT(result_r, 15, theta_r);
    SET_VECTOR_ELT(resultName_r, 15, mkChar("theta"));

    SET_VECTOR_ELT(result_r, 16, w_mu_r);
    SET_VECTOR_ELT(resultName_r, 16, mkChar("w_mu"));

    SET_VECTOR_ELT(result_r, 17, A_vi_r);
    SET_VECTOR_ELT(resultName_r, 17, mkChar("A_vi"));

    SET_VECTOR_ELT(result_r, 18, S_vi_r);
    SET_VECTOR_ELT(resultName_r, 18, mkChar("S_vi"));

    SET_VECTOR_ELT(result_r, 19, iter_r);
    SET_VECTOR_ELT(resultName_r, 19, mkChar("iter"));

    SET_VECTOR_ELT(result_r, 20, ELBO_vec_r);
    SET_VECTOR_ELT(resultName_r, 20, mkChar("ELBO_vec"));

    SET_VECTOR_ELT(result_r, 21, theta_para_r);
    SET_VECTOR_ELT(resultName_r, 21, mkChar("theta_para"));

    SET_VECTOR_ELT(result_r, 22, beta_r);
    SET_VECTOR_ELT(resultName_r, 22, mkChar("beta"));

    SET_VECTOR_ELT(result_r, 23, beta_cov_r);
    SET_VECTOR_ELT(resultName_r, 23, mkChar("beta_cov"));

    // SET_VECTOR_ELT(result_r, 24, a_track_r);
    // SET_VECTOR_ELT(resultName_r, 24, mkChar("a_track"));
    // 
    // SET_VECTOR_ELT(result_r, 25, gamma_track_r);
    // SET_VECTOR_ELT(resultName_r, 25, mkChar("gamma_track"));


    // SET_VECTOR_ELT(result_r, 20, uiIndx_r);
    // SET_VECTOR_ELT(resultName_r, 20, mkChar("uiIndx"));
    //
    // SET_VECTOR_ELT(result_r, 21, uIndx_r);
    // SET_VECTOR_ELT(resultName_r, 21, mkChar("uIndx"));
    //
    // SET_VECTOR_ELT(result_r, 22, uIndxLU_r);
    // SET_VECTOR_ELT(resultName_r, 22, mkChar("uIndxLU"));

    namesgets(result_r, resultName_r);
    //unprotect
    UNPROTECT(nProtect);


    return(result_r);

  }

  SEXP spVarBayes_NNGP_nocovariates_betacpp(SEXP y_r,
                                                SEXP n_r, SEXP p_r, SEXP m_r, SEXP m_vi_r, SEXP coords_r, SEXP covModel_r, SEXP rho_r,
                                                SEXP zetaSqIG_r, SEXP tauSqIG_r, SEXP phibeta_r, SEXP nuUnif_r,
                                                SEXP zetaSqStarting_r, SEXP tauSqStarting_r, SEXP phiStarting_r, SEXP nuStarting_r,
                                                SEXP sType_r, SEXP nThreads_r, SEXP verbose_r, SEXP fix_nugget_r, SEXP N_phi_r, SEXP Trace_N_r,
                                                SEXP max_iter_r,
                                                SEXP var_input_r,
                                                SEXP phi_input_r, SEXP phi_iter_max_r, SEXP initial_mu_r,
                                                SEXP min_iter_r, SEXP K_r, SEXP stop_K_r){


    int h, i, j, k, l, s, info, nProtect=0;
    const int inc = 1;
    const double one = 1.0;
    const double negOne = -1.0;
    const double zero = 0.0;
    char const *lower = "L";
    char const *upper = "U";
    char const *ntran = "N";
    char const *ytran = "T";
    char const *rside = "R";
    char const *lside = "L";
    const double pi = 3.1415926;
    //get args
    double *y = REAL(y_r);
    int p = INTEGER(p_r)[0];
    int n = INTEGER(n_r)[0];
    int m = INTEGER(m_r)[0];
    int m_vi = INTEGER(m_vi_r)[0];
    double *coords = REAL(coords_r);
    double fix_nugget = REAL(fix_nugget_r)[0];
    int covModel = INTEGER(covModel_r)[0];
    std::string corName = getCorName(covModel);
    //double converge_per  =  REAL(converge_per_r)[0];

    int K = INTEGER(K_r)[0];
    int stop_K = INTEGER(stop_K_r)[0];
    int min_iter = INTEGER(min_iter_r)[0];

    double *var_input  =  REAL(var_input_r);
    double phi_input  =  REAL(phi_input_r)[0];
    int initial_mu  =  INTEGER(initial_mu_r)[0];
    int phi_iter_max = INTEGER(phi_iter_max_r)[0];
    int nThreads = INTEGER(nThreads_r)[0];
    int verbose = INTEGER(verbose_r)[0];
    //double  vi_threshold  =  REAL(vi_threshold_r)[0];
    double  rho  =  REAL(rho_r)[0];
    //double  rho_phi  =  REAL(rho_phi_r)[0];
    //priors
    double zetaSqIGa = REAL(zetaSqIG_r)[0]; double zetaSqIGb = REAL(zetaSqIG_r)[1];
    double tauSqIGa = REAL(tauSqIG_r)[0]; double tauSqIGb = REAL(tauSqIG_r)[1];
    //double phiUnifa = REAL(phiUnif_r)[0]; double phiUnifb = REAL(phiUnif_r)[1];
    double phimin = REAL(phibeta_r)[0]; double phimax = REAL(phibeta_r)[1];

    double a_phi = (phi_input - phimin)/(phimax-phimin)*10;
    double b_phi = 10 - a_phi;

    double nuUnifa = 0, nuUnifb = 0;
    if(corName == "matern"){
      nuUnifa = REAL(nuUnif_r)[0]; nuUnifb = REAL(nuUnif_r)[1];
    }

#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#else
    if(nThreads > 1){
      warning("n.omp.threads > %i, but source not compiled with OpenMP support.", nThreads);
      nThreads = 1;
    }
#endif

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tModel description\n");
      Rprintf("----------------------------------------\n");
      Rprintf("NNGP Latent model fit with %i observations.\n\n", n);
      Rprintf("Number of covariates %i (including intercept if specified).\n\n", p);
      Rprintf("Using the %s spatial correlation model.\n\n", corName.c_str());
      Rprintf("Using %i nearest neighbors.\n\n", m);
      Rprintf("Priors and hyperpriors:\n");
      Rprintf("\tbeta flat.\n");
#ifdef _OPENMP
      Rprintf("\nSource compiled with OpenMP support and model fit using %i thread(s).\n", nThreads);
#else
      Rprintf("\n\nSource not compiled with OpenMP support.\n");
#endif
    }

    //parameters
    int nTheta, zetaSqIndx, tauSqIndx, phiIndx, nuIndx;

    if(corName != "matern"){
      nTheta = 3;//zeta^2, tau^2, phi
      zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
    }else{
      nTheta = 4;//zeta^2, tau^2, phi, nu
      zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2; nuIndx = 3;
    }

    //allocated for the nearest neighbor index vector (note, first location has no neighbors).
    int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);

    SEXP nnIndx_r; PROTECT(nnIndx_r = allocVector(INTSXP, nIndx)); nProtect++; int *nnIndx = INTEGER(nnIndx_r);

    //int *nnIndx = (int *) R_alloc(nIndx, sizeof(int));

    double *d = (double *) R_alloc(nIndx, sizeof(double));

    SEXP nnIndxLU_r; PROTECT(nnIndxLU_r = allocVector(INTSXP, 2*n)); nProtect++; int *nnIndxLU = INTEGER(nnIndxLU_r); //first column holds the nnIndx index for the i-th location and the second columns holds the number of neighbors the i-th location has (the second column is a bit of a waste but will simplifying some parallelization).

    //int *nnIndxLU = (int *) R_alloc(2*n, sizeof(int));

    //make the neighbor index
    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tBuilding neighbor index\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }

    if(INTEGER(sType_r)[0] == 0){
      mkNNIndx(n, m, coords, nnIndx, d, nnIndxLU);
    }
    if(INTEGER(sType_r)[0] == 1){
      mkNNIndxTree0(n, m, coords, nnIndx, d, nnIndxLU);
    }else{
      mkNNIndxCB(n, m, coords, nnIndx, d, nnIndxLU);
    }

    SEXP CIndx_r; PROTECT(CIndx_r = allocVector(INTSXP, 2*n)); nProtect++; int *CIndx = INTEGER(CIndx_r); //index for D and C.

    //int *CIndx = (int *) R_alloc(2*n, sizeof(int));
    for(i = 0, j = 0; i < n; i++){//zero should never be accessed
      j += nnIndxLU[n+i]*nnIndxLU[n+i];
      if(i == 0){
        CIndx[n+i] = 0;
        CIndx[i] = 0;
      }else{
        CIndx[n+i] = nnIndxLU[n+i]*nnIndxLU[n+i];
        CIndx[i] = CIndx[n+i-1] + CIndx[i-1];
      }
    }

    SEXP numIndxCol_r; PROTECT(numIndxCol_r = allocVector(INTSXP, n)); nProtect++; int *numIndxCol = INTEGER(numIndxCol_r); zeros_int(numIndxCol, n);
    get_num_nIndx_col(nnIndx, nIndx, numIndxCol);

    SEXP cumnumIndxCol_r; PROTECT(cumnumIndxCol_r = allocVector(INTSXP, n)); nProtect++; int *cumnumIndxCol = INTEGER(cumnumIndxCol_r); zeros_int(cumnumIndxCol,n);
    get_cumnum_nIndx_col(numIndxCol, n, cumnumIndxCol);

    SEXP nnIndxCol_r; PROTECT(nnIndxCol_r = allocVector(INTSXP, nIndx+n)); nProtect++; int *nnIndxCol = INTEGER(nnIndxCol_r); zeros_int(nnIndxCol, n);
    get_nnIndx_col(nnIndx, n, nIndx, cumnumIndxCol, numIndxCol, nnIndxCol);

    int *sumnnIndx = (int *) R_alloc(n-1, sizeof(int));; zeros_int(sumnnIndx, n-1);
    get_sum_nnIndx(sumnnIndx, n, m);

    SEXP nnIndxnnCol_r; PROTECT(nnIndxnnCol_r = allocVector(INTSXP, nIndx)); nProtect++; int *nnIndxnnCol = INTEGER(nnIndxnnCol_r); zeros_int(nnIndxnnCol, n);
    get_nnIndx_nn_col(nnIndx, n, m, nIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, sumnnIndx);


    double *D = (double *) R_alloc(j, sizeof(double));

    for(i = 0; i < n; i++){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        for(l = 0; l <= k; l++){
          D[CIndx[i]+l*nnIndxLU[n+i]+k] = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
        }
      }
    }
    int mm = m*m;
    SEXP B_r; PROTECT(B_r = allocVector(REALSXP, nIndx)); nProtect++; double *B = REAL(B_r);
    SEXP F_r; PROTECT(F_r = allocVector(REALSXP, n)); nProtect++; double *F = REAL(F_r);

    //double *B = (double *) R_alloc(nIndx, sizeof(double));
    //double *F = (double *) R_alloc(n, sizeof(double));

    double *c =(double *) R_alloc(m*nThreads, sizeof(double));
    double *C = (double *) R_alloc(mm*nThreads, sizeof(double));


    SEXP theta_r; PROTECT(theta_r = allocVector(REALSXP, nTheta)); nProtect++; double *theta = REAL(theta_r);

    SEXP w_mu_r; PROTECT(w_mu_r = allocVector(REALSXP, n)); nProtect++; double *w_mu = REAL(w_mu_r);

    SEXP sigma_sq_r; PROTECT(sigma_sq_r = allocVector(REALSXP, n)); nProtect++; double *sigma_sq = REAL(sigma_sq_r);

    //double *beta = (double *) R_alloc(p, sizeof(double)); zeros(beta, p);
    //double *theta = (double *) R_alloc(nTheta, sizeof(double));

    // theta[0] = REAL(zetaSqStarting_r)[0];
    // theta[1] = REAL(phiStarting_r)[0];
    //
    // if(corName == "matern"){
    //   theta[2] = REAL(nuStarting_r)[0];
    // }
    //

    theta[zetaSqIndx] = REAL(zetaSqStarting_r)[0];
    theta[tauSqIndx] = REAL(tauSqStarting_r)[0];
    //theta[phiIndx] = REAL(phiStarting_r)[0];
    theta[phiIndx] = phi_input;
    if(corName == "matern"){
      theta[nuIndx] = REAL(nuStarting_r)[0];
    }

    //other stuff
    double logDetInv;
    int accept = 0, batchAccept = 0, status = 0;
    int jj, kk, nn = n*n;
    double *one_n = (double *) R_alloc(n, sizeof(double)); ones(one_n, n);
    double *tmp_n = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n, n);

    double *tau_sq_I = (double *) R_alloc(one, sizeof(double));

    //double *w_mu = (double *) R_alloc(n, sizeof(double));
    //zeros(w_mu, n);
    if(initial_mu){
      F77_NAME(dcopy)(&n, y, &inc, w_mu, &inc);
    }else{
      zeros(w_mu, n);
    }
    //double *sigma_sq = (double *) R_alloc(n, sizeof(double));
    ones(sigma_sq, n);

    double *w_mu_update = (double *) R_alloc(n, sizeof(double)); zeros(w_mu_update, n);
    double *E_mu_sq = (double *) R_alloc(n, sizeof(double)); zeros(E_mu_sq, n);
    double *delta_mu_sq = (double *) R_alloc(n, sizeof(double)); zeros(delta_mu_sq, n);
    double *delta_mu = (double *) R_alloc(n, sizeof(double)); zeros(delta_mu, n);
    double *m_mu = (double *) R_alloc(n, sizeof(double)); zeros(m_mu, n);

    double *sigma_sq_update = (double *) R_alloc(n, sizeof(double)); ones(sigma_sq_update, n);

    double a, v, b, e, mu, var, aij, phiCand, nuCand = 0, nu = 0;

    double a_tau_update = n * 0.5 + tauSqIGa;
    double b_tau_update = 0.0;
    double tau_sq = 0.0;

    double a_zeta_update = n * 0.5 + zetaSqIGa;
    double b_zeta_update = 0.0;
    double zeta_sq = 0.0;
    int N_phi = INTEGER(N_phi_r)[0];
    int Trace_N = INTEGER(Trace_N_r)[0];
    int one_int = 1;
    int three_int = 3;
    double adadelta_noise = 0.0000001;
    double *bk = (double *) R_alloc(nThreads*(1.0+5.0), sizeof(double));
    //double *bk = (double *) R_alloc(nThreads*(1.0+static_cast<int>(floor(nuUnifb))), sizeof(double));
    if(corName == "matern"){nu = theta[nuIndx];}

    //int iter = 1;
    int max_iter = INTEGER(max_iter_r)[0];
    //int iter = (int ) R_alloc(one_int, sizeof(int)); iter = 1;
    int iter = 1;

    double vi_error = 1.0;
    double rho1 = 0.9;
    double rho2 = 0.999;
    double adaptive_adam = 0.001;
    //double vi_threshold = 0.0001;

    //F77_NAME(dgemm)(ytran, ntran, &p, &p, &n, &one, X, &n, X, &n, &zero, XtX, &p FCONE FCONE);

    // NNGP parameters

    //allocated for the nearest neighbor index vector (note, first location has no neighbors).
    int nIndx_vi = static_cast<int>(static_cast<double>(1+m_vi)/2*m_vi+(n-m_vi-1)*m_vi);

    SEXP nnIndx_vi_r; PROTECT(nnIndx_vi_r = allocVector(INTSXP, nIndx_vi)); nProtect++; int *nnIndx_vi = INTEGER(nnIndx_vi_r);

    double *d_vi = (double *) R_alloc(nIndx_vi, sizeof(double));

    SEXP nnIndxLU_vi_r; PROTECT(nnIndxLU_vi_r = allocVector(INTSXP, 2*n)); nProtect++; int *nnIndxLU_vi = INTEGER(nnIndxLU_vi_r); //first column holds the nnIndx index for the i-th location and the second columns holds the number of neighbors the i-th location has (the second column is a bit of a waste but will simplifying some parallelization).

    //make the neighbor index
    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tBuilding neighbor index for variational inference \n");
      Rprintf("Using %i nearest neighbors.\n\n", m_vi);
#ifdef Win32
      R_FlushConsole();
#endif
    }

    if(INTEGER(sType_r)[0] == 0){
      mkNNIndx(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }
    if(INTEGER(sType_r)[0] == 1){
      mkNNIndxTree0(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }else{
      mkNNIndxCB(n, m_vi, coords, nnIndx_vi, d_vi, nnIndxLU_vi);
    }


    int mm_vi = m_vi*m_vi;
    SEXP A_vi_r; PROTECT(A_vi_r = allocVector(REALSXP, nIndx_vi)); nProtect++; double *A_vi = REAL(A_vi_r); zeros(A_vi,nIndx_vi);
    SEXP S_vi_r; PROTECT(S_vi_r = allocVector(REALSXP, n)); nProtect++; double *S_vi = REAL(S_vi_r); ones(S_vi,n);
    for(int i = 0; i < n; i++){
      S_vi[i] = var_input[i];
    }
    SEXP numIndxCol_vi_r; PROTECT(numIndxCol_vi_r = allocVector(INTSXP, n)); nProtect++; int *numIndxCol_vi = INTEGER(numIndxCol_vi_r); zeros_int(numIndxCol_vi, n);
    get_num_nIndx_col(nnIndx_vi, nIndx_vi, numIndxCol_vi);

    SEXP cumnumIndxCol_vi_r; PROTECT(cumnumIndxCol_vi_r = allocVector(INTSXP, n)); nProtect++; int *cumnumIndxCol_vi = INTEGER(cumnumIndxCol_vi_r); zeros_int(cumnumIndxCol_vi,n);
    get_cumnum_nIndx_col(numIndxCol_vi, n, cumnumIndxCol_vi);

    SEXP nnIndxCol_vi_r; PROTECT(nnIndxCol_vi_r = allocVector(INTSXP, nIndx_vi+n)); nProtect++; int *nnIndxCol_vi = INTEGER(nnIndxCol_vi_r); zeros_int(nnIndxCol_vi, n);
    get_nnIndx_col(nnIndx_vi, n, nIndx_vi, cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi);

    int *sumnnIndx_vi = (int *) R_alloc(n-1, sizeof(int));; zeros_int(sumnnIndx_vi, n-1);
    get_sum_nnIndx(sumnnIndx_vi, n, m_vi);

    SEXP nnIndxnnCol_vi_r; PROTECT(nnIndxnnCol_vi_r = allocVector(INTSXP, nIndx_vi)); nProtect++; int *nnIndxnnCol_vi = INTEGER(nnIndxnnCol_vi_r); zeros_int(nnIndxnnCol_vi, n);
    get_nnIndx_nn_col(nnIndx_vi, n, m_vi, nIndx_vi, cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi, sumnnIndx_vi);

    double *E_a_sq = (double *) R_alloc(nIndx_vi, sizeof(double)); zeros(E_a_sq, nIndx_vi);
    double *delta_a_sq = (double *) R_alloc(nIndx_vi, sizeof(double)); zeros(delta_a_sq, nIndx_vi);
    double *delta_a = (double *) R_alloc(nIndx_vi, sizeof(double)); zeros(delta_a, nIndx_vi);

    double *E_gamma_sq = (double *) R_alloc(n, sizeof(double)); zeros(E_gamma_sq, n);
    double *delta_gamma_sq = (double *) R_alloc(n, sizeof(double)); zeros(delta_gamma_sq, n);
    double *delta_gamma = (double *) R_alloc(n, sizeof(double)); zeros(delta_gamma, n);
    double *gamma_vec = (double *) R_alloc(n, sizeof(double));zeros(gamma_vec, n);
    //double *B = (double *) R_alloc(nIndx, sizeof(double));
    //double *F = (double *) R_alloc(n, sizeof(double));



    double *epsilon_vec = (double *) R_alloc(n, sizeof(double));

    for(int i = 0; i < n; i++){
      epsilon_vec[i] = rnorm(0, 1);
    }

    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

//
//     int nIndSqx = static_cast<int>(static_cast<double>(1+m)*(m+m+1)/6*m+(n-m-1)*m*m);
//     double *F_inv = (double *) R_alloc(n, sizeof(double)); zeros(F_inv,n);
//     double *B_over_F = (double *) R_alloc(nIndx, sizeof(double)); zeros(B_over_F,nIndx);
//     double *Bmat_over_F = (double *) R_alloc(nIndSqx, sizeof(double)); zeros(Bmat_over_F,nIndSqx);
//     int *nnIndxLUSq = (int *) R_alloc(2*n, sizeof(int));
//     for(int i = n; i < 2*n; i++){
//       nnIndxLUSq[i] = pow(nnIndxLU[i],2);
//     }
//
//     nnIndxLUSq[0] = 0;
//     for(int i = 1; i < n; i++){
//       nnIndxLUSq[i] = nnIndxLUSq[i-1] + nnIndxLUSq[n+i-1];
//     }
//
//     double F_inv_temp;
//     for(i = 0; i < n; i++){
//       F_inv[i] = 1/F[i];
//
//       for(j = 0; j < nnIndxLU[n+i]; j++){
//         B_over_F[nnIndxLU[i]+j] = B[nnIndxLU[i]+j]/F[i];
//       }
//
//       if(i > 0){
//         F_inv_temp = 1/F[i];
//         F77_NAME(dger)(&nnIndxLU[n+i], &nnIndxLU[n+i], &one, &B[nnIndxLU[i]], &inc, &B[nnIndxLU[i]], &inc, &Bmat_over_F[nnIndxLUSq[i]], &nnIndxLU[n+i]);
//         F77_NAME(dscal)(&nnIndxLUSq[n+i], &F_inv_temp, &Bmat_over_F[nnIndxLUSq[i]], &inc);
//       }
//
//     }
//
//     double q = 0;
//     int num_m;
//     for(i = 0; i < n; i++){
//       a = 0;
//       b = 0;
//       num_m = nnIndxLU[n+i];
//       for(j = 0; j < num_m; j++){
//         a += B_over_F[nnIndxLU[i]+j]*y[nnIndx[nnIndxLU[i]+j]];
//
//         for(k = 0; k < num_m; k++){
//           b += Bmat_over_F[nnIndxLUSq[i]+j*num_m+k]*y[nnIndx[nnIndxLU[i]+j]]*y[nnIndx[nnIndxLU[i]+k]];
//         }
//       }
//       //Rprintf("b at i %i is %f ",i,b);
//       q += y[i]*y[i]*F_inv[i] - 2*a*y[i] + b;
//     }
//
//     Rprintf("q is %f ", q);
//     Rprintf("Q is %f ", Q(B, F, y, y, n, nnIndx,nnIndxLU));
    // Rprintf("\t nIndSqx is : %i\n", nIndSqx);
    //
    // Rprintf("\t nnIndxLU\n");
    // for(int i = 0; i< 2*n; i++){
    //   Rprintf("%i ",nnIndxLU[i]);
    // }
    // Rprintf("\n");
    // Rprintf("\t nnIndxLUSq\n");
    // for(int i = 0; i< 2*n; i++){
    //   Rprintf("%i ",nnIndxLUSq[i]);
    // }
    // Rprintf("\n");
    //
    // for(int i = 0; i < n; i++){
    //   Rprintf("B is ");
    //   for (int l = 0; l < nnIndxLU[n + i]; l++){
    //     Rprintf("%f ", B[nnIndxLU[i] + l]);
    //   }
    //   Rprintf("\n");
    //   Rprintf("F is %f ", F[i]);
    //   Rprintf("Bmat_over_F is ");
    //   for (int k = 0; k < nnIndxLUSq[n + i]; k++){
    //     Rprintf("%f ", Bmat_over_F[nnIndxLUSq[i] + k]);
    //   }
    //   Rprintf("\n");
    // }

    // int n_per = nIndx_vi * converge_per;
    // int *sign_vec_old = (int *) R_alloc(n_per, sizeof(int));
    // int *sign_vec_new = (int *) R_alloc(n_per, sizeof(int));
    // int *check_vec = (int *) R_alloc(n_per, sizeof(int));
    int indicator_converge = 0;

    double *trace_vec = (double *) R_alloc(2, sizeof(double));
    double *u_vec = (double *) R_alloc(n, sizeof(double));
    double *u_vec_mean = (double *) R_alloc(n, sizeof(double));
    double ELBO_MC = 0.0;
    double ELBO = 0.0;
    //double *epsilon_vec = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp2 = (double *) R_alloc(n, sizeof(double));
    double *gradient_const = (double *) R_alloc(n, sizeof(double));
    double *gradient = (double *) R_alloc(n, sizeof(double));
    double *gamma_gradient_sum = (double *) R_alloc(n, sizeof(double));
    double *gamma_gradient = (double *) R_alloc(n, sizeof(double));
    double *a_gradient = (double *) R_alloc(nIndx_vi, sizeof(double));
    double *a_gradient_sum = (double *) R_alloc(nIndx_vi, sizeof(double));

    SEXP ELBO_vec_r; PROTECT(ELBO_vec_r = allocVector(REALSXP, max_iter)); nProtect++;
    double *ELBO_vec = REAL(ELBO_vec_r); zeros(ELBO_vec,max_iter);
    double max_ELBO = 0.0;
    int ELBO_convergence_count = 0;
    // double *derivative_neighbour = (double *) R_alloc(n, sizeof(double));zeros(derivative_neighbour,n);
    // double *derivative_neighbour_a = (double *) R_alloc(n, sizeof(double));zeros(derivative_neighbour_a,n);
    //
    // for(int i = 1; i < n; i++){
    //   for (int l = 0; l < nnIndxLU_vi[n + i]; l++){
    //     if((i-1) == (nnIndx_vi[nnIndxLU_vi[i] + l]) ){
    //       derivative_neighbour[i] = 1;
    //       //derivative_neighbour_a[i+1] = A_vi[nnIndxLU_vi[i] + l];
    //     }
    //   }
    // }

    // double *derivative_store = (double *) R_alloc(m_vi*n, sizeof(double)); zeros(derivative_store,m_vi*n);
    // double *derivative_store_gamma = (double *) R_alloc(n, sizeof(double)); zeros(derivative_store_gamma,n);

    double E_phi_sq = 0.0;
    double delta_phi = 0.0;
    double delta_phi_sq = 0.0;
    double *rademacher_rv_vec = (double *) R_alloc(n, sizeof(double));
    double *rademacher_rv_temp = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp,n);
    double *rademacher_rv_temp2 = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp2,n);

    double *product_v = (double *) R_alloc(n, sizeof(double));zeros(product_v,n);
    double *product_v2 = (double *) R_alloc(n, sizeof(double));zeros(product_v2,n);
    double *phi_can_vec = (double *) R_alloc(N_phi*N_phi, sizeof(double));zeros(phi_can_vec,N_phi*N_phi);
    double *log_g_phi = (double *) R_alloc(N_phi*N_phi, sizeof(double));zeros(log_g_phi,N_phi*N_phi);
    double *sum_v = (double *) R_alloc(n, sizeof(double));zeros(sum_v,n);

    while(iter <= max_iter & !indicator_converge){
      if(verbose){
        Rprintf("----------------------------------------\n");
        Rprintf("\tIteration at %i \n",iter);
#ifdef Win32
        R_FlushConsole();
#endif
      }
      zeros(tau_sq_I, one_int);
      for(i = 0; i < n; i++){
        tmp_n[i] = y[i]-w_mu[i];
        tau_sq_I[0] += pow(tmp_n[i],2);
      }

      ///////////////
      //update tausq
      ///////////////

      zeros(trace_vec,2);
      zeros(u_vec,n);

      for(int i = 0; i < n; i++){
        epsilon_vec[i] = rnorm(0, 1);

      }
      update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);

      for(int k = 0; k < Trace_N; k++){
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }
        update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);

        double u_mean = 0.0;
        for(i = 0; i < n; i++){
          u_mean += u_vec[i];
        }
        u_mean = u_mean/n;

        for(i = 0; i < n; i++){
          trace_vec[0] += pow(u_vec[i]-u_mean,2);
        }

        trace_vec[1] += Q(B, F, u_vec, u_vec, n, nnIndx, nnIndxLU);
      }




      //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
      b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + *tau_sq_I)*0.5;
      // Rprintf("trace_vec[0]/Trace_N is : %f \n",trace_vec[0]/Trace_N);
      // Rprintf("b_tau_update is : %f \n",b_tau_update);
      tau_sq = b_tau_update/a_tau_update;
      theta[tauSqIndx] = tau_sq;


      if(verbose){
        Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
        R_FlushConsole();
#endif
      }

      //Rprintf("the value of add_tau : %f \n", trace_vec[0]/Trace_N + *tau_sq_I);
      ///////////////
      //update zetasq
      ///////////////

      updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

      double zeta_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
      b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q)*theta[zetaSqIndx]*0.5;
      //Rprintf("zeta_Q: %f \n", zeta_Q);
      //b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q)*0.5;
      zeta_sq = b_zeta_update/a_zeta_update;
      //Rprintf("the value of add_zeta : %f \n", (trace_vec[1]/Trace_N + zeta_Q));
      theta[zetaSqIndx] = zeta_sq;
      // Rprintf("zeta_Q_mb is : %f \n",zeta_Q);
      // Rprintf("trace_vec1_re/Trace_N is : %f \n",trace_vec[1]/Trace_N);
      // Rprintf("b_zeta_update is : %f \n",b_zeta_update);

      if(verbose){
        Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
        R_FlushConsole();
#endif
      }
      updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
      ///////////////
      //update phi
      ///////////////
      // clock_t start, mid, end;
      // double cpu_time_used;
      // start = clock();

      if(iter < phi_iter_max){

        double *a_phi_vec = (double *) R_alloc(N_phi, sizeof(double));
        double *b_phi_vec = (double *) R_alloc(N_phi, sizeof(double));
        a_phi_vec[0] = a_phi;
        b_phi_vec[0] = b_phi;

        for(int i = 1; i < N_phi; i++){
          if (i % 2 == 0) {
            a_phi_vec[i] = a_phi_vec[0] + 0.01*i;
            b_phi_vec[i] = b_phi_vec[0] + 0.01*i;
            // a_phi_vec[i] = a_phi_vec[0]*(1+0.1*i);
            // b_phi_vec[i] = b_phi_vec[0]*(1+0.1*i);
          } else {
            a_phi_vec[i] = a_phi_vec[0] + 0.01*i*(-1);
            b_phi_vec[i] = b_phi_vec[0] + 0.01*i*(-1);
            // a_phi_vec[i] = a_phi_vec[0]*(1-0.1*i);
            // b_phi_vec[i] = b_phi_vec[0]*(1-0.1*i);
          }

        }

        double phi_Q = 0.0;
        double diag_sigma_sq_sum = 0.0;

        int max_index;
        zeros(phi_can_vec,N_phi*N_phi);
        zeros(log_g_phi,N_phi*N_phi);
        for(int i = 0; i < N_phi; i++){
          for(int j = 0; j < N_phi; j++){

            for(int k = 0; k < Trace_N; k++){
              phi_can_vec[i*N_phi+j] += rbeta(a_phi_vec[i], b_phi_vec[j]);  // Notice the indexing here
            }
            phi_can_vec[i*N_phi+j] /= Trace_N;
            phi_can_vec[i*N_phi+j] = phi_can_vec[i*N_phi+j]*(phimax - phimin) + phimin;
          }
        }


        for(i = 0; i < N_phi*N_phi; i++){

          updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                   theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb);

          //phi_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
          phi_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
          update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);
          logDetInv = 0.0;
          diag_sigma_sq_sum = 0.0;
          for(j = 0; j < n; j++){
            logDetInv += log(1/F[j]);
          }

          log_g_phi[i] = logDetInv*0.5 - (phi_Q + Q(B, F, u_vec, u_vec, n, nnIndx, nnIndxLU))*0.5;
        }

        max_index = max_ind(log_g_phi,N_phi*N_phi);
        a_phi = a_phi_vec[max_index/N_phi];
        b_phi = b_phi_vec[max_index % N_phi];

        theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;

        updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                 theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
      }
      // end = clock();
      // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
      // printf("phi %f seconds to execute \n", cpu_time_used);
      if(verbose){
        Rprintf("the value of theta[%i phiIndx] : %f \n", phiIndx, theta[phiIndx]);
#ifdef Win32
        R_FlushConsole();
#endif
      }

      ///////////////
      //update w
      ///////////////

      zeros(w_mu_temp,n);
      zeros(w_mu_temp2,n);

      product_B_F(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp);
      //product_B_F(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2);
      product_B_F_vec(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);

      double gradient_mu = 0.0;
      //Rprintf("w_mu_update: ");
      for(i = 0; i < n; i++){
        gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
        //Rprintf("%i ",gradient_mu);
        //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i]/theta[zetaSqIndx] + (y[i])/theta[tauSqIndx]);
        E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
        delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
        delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
        w_mu_update[i] = w_mu[i] + delta_mu[i];
        //Rprintf("%f ",w_mu_update[i]);
      }


      product_B_F(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp);
      //product_B_F(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2);
      product_B_F_vec(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);



      // zeros(gradient_const,n);
      // for(i = 0; i < n; i++){
      //   gradient_const[i] = -w_mu_update[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx];
      // }


      zeros(gradient,n);
      zeros(gamma_gradient_sum, n);
      zeros(gamma_gradient,n);

      for(int k = 0; k < Trace_N; k++){
        zeros(gamma_gradient,n);
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }
        // gamma_gradient_fun(u_vec, epsilon_vec, gamma_gradient, gradient_const, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
        //                    B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
        //                    cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,w_mu_temp,w_mu_temp2,gradient);

        gamma_gradient_fun(u_vec, epsilon_vec, gamma_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                           B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                           cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,w_mu_temp,w_mu_temp2,gradient);

        vecsum(gamma_gradient_sum, gamma_gradient, Trace_N, n);
      }

      //free(gamma_gradient);
      //Rprintf("S_vi: ");
      for(i = 0; i < n; i++){
        E_gamma_sq[i] = rho * E_gamma_sq[i] + (1 - rho) * pow(gamma_gradient_sum[i],2);
        delta_gamma[i] = sqrt(delta_gamma_sq[i]+adadelta_noise)/sqrt(E_gamma_sq[i]+adadelta_noise)*gamma_gradient_sum[i];
        delta_gamma_sq[i] = rho*delta_gamma_sq[i] + (1 - rho) * pow(delta_gamma[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        //S_vi[i] = exp(pow((log(sqrt(S_vi[i])) + delta_gamma[i]),2));
        S_vi[i] = pow(exp(log(sqrt(S_vi[i])) + delta_gamma[i]),2);
        //Rprintf("%f ",S_vi[i]);
        //S_vi[i] = pow(exp(gamma_vec[i]),2);
      }
      //Rprintf("\n");


      zeros(a_gradient,nIndx_vi);
      zeros(a_gradient_sum, nIndx_vi);

      for(int k = 0; k < Trace_N; k++){
        zeros(a_gradient,nIndx_vi);
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }

        // a_gradient_fun(u_vec, epsilon_vec, a_gradient, gradient_const, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
        //                B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
        //                w_mu_temp,w_mu_temp2);

        a_gradient_fun(u_vec, epsilon_vec, a_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                       B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                       w_mu_temp,w_mu_temp2);
        //
        // for(int i = 0; i < nIndx_vi; i++){
        //   Rprintf("\tError is %i, %f \n",i, a_gradient[i]);
        // }
        vecsum(a_gradient_sum, a_gradient, Trace_N, nIndx_vi);
        // for(int i = 0; i < nIndx_vi; i++){
        //   a_gradient_sum[i] = a_gradient[i];
        // }
      }
      //free(a_gradient);
      //update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);


      //Rprintf("A_vi: ");
      for(i = 0; i < nIndx_vi; i++){
        E_a_sq[i] = rho * E_a_sq[i] + (1 - rho) * pow(a_gradient_sum[i],2);
        delta_a[i] = sqrt(delta_a_sq[i]+adadelta_noise)/sqrt(E_a_sq[i]+adadelta_noise)*a_gradient_sum[i];
        delta_a_sq[i] = rho*delta_a_sq[i] + (1 - rho) * pow(delta_a[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        A_vi[i] = A_vi[i] + delta_a[i];
        //Rprintf("%f ",A_vi[i]);
      }


      ELBO = 0.0;
      zeros(sum_v,n);

      double sum2 = 0.0;
      double sum3 = 0.0;
      double sum4 = 0.0;
      double sum5 = 0.0;

      for(int k = 0; k < Trace_N; k++){
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }
        update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);
        sum_two_vec(u_vec, w_mu_update, sum_v, n);
        for(int i = 0; i < n; i++){
          sum3 += pow((y[i] - sum_v[i]),2)/theta[tauSqIndx]*0.5;
        }
        sum2 += Q(B, F, sum_v, sum_v, n, nnIndx, nnIndxLU)*0.5;
      }

      for(int i = 0; i < n; i++){
        sum4 += log(2*pi*S_vi[i]);
        sum5 += log(2*pi*F[i]);
      }

      ELBO = (sum2 + sum3)/Trace_N;

      ELBO += -0.5*sum4;

      ELBO += 0.5*n*log(2*pi*theta[tauSqIndx]);

      ELBO += 0.5*sum5;

      ELBO += -0.5*n;

      //Rprintf("the value of ELBO: %f \n", ELBO);
      ELBO_vec[iter-1] = -ELBO;


      if(iter == min_iter){max_ELBO = - ELBO;}
      if (iter > min_iter && iter % 10 == 0){

        int count = 0;
        double sum = 0.0;
        for (int i = iter - 10; i < iter; i++) {
          sum += ELBO_vec[i];
          count++;
        }

        double average =  sum / count;

        if(average < max_ELBO){ELBO_convergence_count+=1;}else{ELBO_convergence_count=0;}
        max_ELBO = max(max_ELBO, average);


        if(stop_K){
          indicator_converge = ELBO_convergence_count>=K;
        }
      }

      if(!verbose){
        int percent = (iter * 100) / max_iter;
        int progressMarks = percent / 10;

        if (iter == max_iter || iter % (max_iter / 10) == 0) {
          Rprintf("\r[");

          for (int j = 0; j < progressMarks; j++) {
            Rprintf("*");
          }

          for (int j = progressMarks; j < 10; j++) {
            Rprintf("-");
          }

          Rprintf("] %d%%\n", percent);

#ifdef Win32
          R_FlushConsole();
#endif
        }
      }

      if(indicator_converge == 1){
        Rprintf("Early convergence reached at iteration at %i \n", iter);
      }
#ifdef Win32
      R_FlushConsole();
#endif

      iter++;
      vi_error = 0.0;
      for(i = 0; i < n; i++){
        vi_error += abs(w_mu_update[i] - w_mu[i]) ;
      }

      F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);


    }
    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

    SEXP theta_para_r; PROTECT(theta_para_r = allocVector(REALSXP, nTheta*2)); nProtect++; double *theta_para = REAL(theta_para_r);

    theta_para[zetaSqIndx*2+0] = a_zeta_update;
    theta_para[zetaSqIndx*2+1] = b_zeta_update;

    theta_para[tauSqIndx*2+0] = a_tau_update;
    theta_para[tauSqIndx*2+1] = b_tau_update;

    theta_para[phiIndx*2+0] = a_phi;
    theta_para[phiIndx*2+1] = b_phi;

    //SEXP x_out; PROTECT(x_out = allocVector(REALSXP, n)); nProtect++;
    //REAL(x_out)[0] = phi_can;
    //F77_NAME(dcopy)(&n, w_mu, &inc, REAL(x_out), &inc);
    SEXP iter_r; PROTECT(iter_r = allocVector(INTSXP, 1)); nProtect++;
    INTEGER(iter_r)[0] = iter;

    SEXP result_r, resultName_r;
    int nResultListObjs = 22;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, nnIndxLU_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("nnIndxLU"));

    SET_VECTOR_ELT(result_r, 1, CIndx_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("CIndx"));

    SET_VECTOR_ELT(result_r, 2, nnIndx_r);
    SET_VECTOR_ELT(resultName_r, 2, mkChar("nnIndx"));

    SET_VECTOR_ELT(result_r, 3, numIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 3, mkChar("numIndxCol"));

    SET_VECTOR_ELT(result_r, 4, cumnumIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 4, mkChar("cumnumIndxCol"));

    SET_VECTOR_ELT(result_r, 5, nnIndxCol_r);
    SET_VECTOR_ELT(resultName_r, 5, mkChar("nnIndxCol"));

    SET_VECTOR_ELT(result_r, 6, nnIndxnnCol_r);
    SET_VECTOR_ELT(resultName_r, 6, mkChar("nnIndxnnCol"));

    SET_VECTOR_ELT(result_r, 7, nnIndxLU_vi_r);
    SET_VECTOR_ELT(resultName_r, 7, mkChar("nnIndxLU_vi"));

    SET_VECTOR_ELT(result_r, 8, nnIndx_vi_r);
    SET_VECTOR_ELT(resultName_r, 8, mkChar("nnIndx_vi"));

    SET_VECTOR_ELT(result_r, 9, numIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 9, mkChar("numIndxCol_vi"));

    SET_VECTOR_ELT(result_r, 10, cumnumIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 10, mkChar("cumnumIndxCol_vi"));

    SET_VECTOR_ELT(result_r, 11, nnIndxCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 11, mkChar("nnIndxCol_vi"));

    SET_VECTOR_ELT(result_r, 12, nnIndxnnCol_vi_r);
    SET_VECTOR_ELT(resultName_r, 12, mkChar("nnIndxnnCol_vi"));

    SET_VECTOR_ELT(result_r, 13, B_r);
    SET_VECTOR_ELT(resultName_r, 13, mkChar("B"));

    SET_VECTOR_ELT(result_r, 14, F_r);
    SET_VECTOR_ELT(resultName_r, 14, mkChar("F"));

    SET_VECTOR_ELT(result_r, 15, theta_r);
    SET_VECTOR_ELT(resultName_r, 15, mkChar("theta"));

    SET_VECTOR_ELT(result_r, 16, w_mu_r);
    SET_VECTOR_ELT(resultName_r, 16, mkChar("w_mu"));

    SET_VECTOR_ELT(result_r, 17, A_vi_r);
    SET_VECTOR_ELT(resultName_r, 17, mkChar("A_vi"));

    SET_VECTOR_ELT(result_r, 18, S_vi_r);
    SET_VECTOR_ELT(resultName_r, 18, mkChar("S_vi"));

    SET_VECTOR_ELT(result_r, 19, iter_r);
    SET_VECTOR_ELT(resultName_r, 19, mkChar("iter"));

    SET_VECTOR_ELT(result_r, 20, ELBO_vec_r);
    SET_VECTOR_ELT(resultName_r, 20, mkChar("ELBO_vec"));

    SET_VECTOR_ELT(result_r, 21, theta_para_r);
    SET_VECTOR_ELT(resultName_r, 21, mkChar("theta_para"));

    namesgets(result_r, resultName_r);
    //unprotect
    UNPROTECT(nProtect);


    return(result_r);

  }



}
