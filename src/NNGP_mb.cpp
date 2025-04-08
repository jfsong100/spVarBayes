#define USE_FC_LEN_T
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include "util.h"
#include "nngp_fun.h"
#include <vector>
#include <algorithm>


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

  SEXP spVarBayes_NNGP_mb_betacpp(SEXP y_r, SEXP X_r,
                                                 SEXP n_r, SEXP p_r, SEXP m_r, SEXP m_vi_r, SEXP coords_r, SEXP covModel_r, SEXP rho_r,
                                                 SEXP zetaSqIG_r, SEXP tauSqIG_r, SEXP phibeta_r, SEXP nuUnif_r,
                                                 SEXP zetaSqStarting_r, SEXP tauSqStarting_r, SEXP phiStarting_r, SEXP nuStarting_r,
                                                 SEXP sType_r, SEXP nThreads_r, SEXP verbose_r, SEXP fix_nugget_r, SEXP N_phi_r, SEXP Trace_N_r,
                                                 SEXP max_iter_r,
                                                 SEXP var_input_r,
                                                 SEXP phi_input_r, SEXP phi_iter_max_r,  SEXP initial_mu_r,
                                                 SEXP mini_batch_size_r,
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
    int n_mb = INTEGER(mini_batch_size_r)[0];


    int K = INTEGER(K_r)[0];
    int stop_K = INTEGER(stop_K_r)[0];
    int min_iter = INTEGER(min_iter_r)[0];

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
    int nBatch = static_cast<int>(std::ceil(static_cast<double>(n)/static_cast<double>(n_mb)));
    int *nBatchIndx = (int *) R_alloc(nBatch, sizeof(int));
    int *nBatchLU = (int *) R_alloc(nBatch, sizeof(int));
    get_nBatchIndx(n, nBatch, n_mb, nBatchIndx, nBatchLU);
    if(verbose){
      Rprintf("Using %i nBatch \n", nBatch);

      for(int i = 0; i < nBatch; i++){
        Rprintf("the value of nBatchIndx[%i] : %i \n",i, nBatchIndx[i]);
        Rprintf("the value of nBatchLU[%i] : %i \n",i, nBatchLU[i]);
      }
#ifdef Win32
      R_FlushConsole();
#endif
    }



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
      //F77_NAME(dcopy)(&n, y, &inc, w_mu, &inc);
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
    double *w_mu_temp_dF = (double *) R_alloc(n, sizeof(double));
    double *u_vec_temp_dF = (double *) R_alloc(n, sizeof(double));
    double *gradient_mu_vec = (double *) R_alloc(n, sizeof(double));

    double *gradient_const = (double *) R_alloc(n, sizeof(double));
    double *gradient = (double *) R_alloc(n, sizeof(double));
    double *gamma_gradient_sum = (double *) R_alloc(n, sizeof(double));

    double *u_vec_temp = (double *) R_alloc(n, sizeof(double));
    double *u_vec_temp2 = (double *) R_alloc(n, sizeof(double));

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

    double *tmp_n_mb = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n_mb, n);

    double *diag_input_mb = (double *) R_alloc(n, sizeof(double)); zeros(diag_input_mb, n);

    int BatchSize;
    double sum_diags= 0.0;
    int i_mb;
    double *rademacher_rv_vec = (double *) R_alloc(n, sizeof(double));
    double *rademacher_rv_temp = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp,n);
    double *rademacher_rv_temp2 = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp2,n);
    double *sum_v = (double *) R_alloc(n, sizeof(double));zeros(sum_v,n);
    double *product_v = (double *) R_alloc(n, sizeof(double));zeros(product_v,n);
    double *product_v2 = (double *) R_alloc(n, sizeof(double));zeros(product_v2,n);
    double *e_i = (double *) R_alloc(n, sizeof(double));zeros(e_i,n);

    int batch_index = 0;

    int max_result_size = nBatch * n;
    int max_temp_size = n;

    int* result_arr = (int *) R_alloc(max_result_size, sizeof(int));
    int* temp_arr = (int *) R_alloc(max_temp_size, sizeof(int));
    int result_index = 0;
    int temp_index = 0;

    int* tempsize_vec = (int *) R_alloc(nBatch, sizeof(int));

    // Usage:
    int *seen_values = (int *) R_alloc(n, sizeof(int));

    // Assuming max possible size for all results is n
    int *intersect_result = (int *) R_alloc(max_result_size, sizeof(int));
    int *complement_first_result = (int *) R_alloc(max_result_size, sizeof(int));
    int *complement_second_result = (int *) R_alloc(max_result_size, sizeof(int));

    // Allocate and initialize indices and sizes arrays
    int *intersect_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *intersect_sizes = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_first_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_first_sizes = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_second_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_second_sizes = (int *) R_alloc(nBatch, sizeof(int));

    // Initialize result indices
    int intersect_result_index = 0;
    int complement_first_result_index = 0;
    int complement_second_result_index = 0;

    for (int batch_index = 0; batch_index < nBatch; ++batch_index) {
      zeros_int(seen_values,n);
      BatchSize = nBatchIndx[batch_index];
      find_set_nngp(n, nnIndx, nnIndxLU, BatchSize, nBatchLU, batch_index,
                    seen_values,
                    intersect_result, intersect_sizes, intersect_start_indices,
                    complement_first_result, complement_first_sizes, complement_first_start_indices,
                    complement_second_result, complement_second_sizes, complement_second_start_indices,
                    intersect_result_index, complement_first_result_index, complement_second_result_index);

      zeros_int(seen_values,n);

      find_set_mb(n, nnIndx, nnIndxLU, nnIndxCol, numIndxCol, nnIndxnnCol, cumnumIndxCol,
                  BatchSize, nBatchLU, batch_index, result_arr, result_index, temp_arr, temp_index, tempsize_vec, seen_values);


    }

    int total_size_intersect = 0;
    int total_size_complement_first  = 0;
    int total_size_complement_second = 0;

    for (int i = 0; i < nBatch; ++i) {
      total_size_intersect         += intersect_sizes[i];
      total_size_complement_first  += complement_first_sizes[i];
      total_size_complement_second += complement_second_sizes[i];
    }

    if(verbose){

      for (int i = 0; i < nBatch; ++i) {
        Rprintf("intersect_sizes is %i \n", intersect_sizes[i]);
        Rprintf("complement_first_sizes is %i \n", complement_first_sizes[i]);
        Rprintf("complement_second_sizes is %i \n", complement_second_sizes[i]);
      }
      Rprintf("\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }

    int* final_intersect_vec = (int *) R_alloc(total_size_intersect, sizeof(int));
    int* final_complement_1_vec = (int *) R_alloc(total_size_complement_first, sizeof(int));
    int* final_complement_2_vec = (int *) R_alloc(total_size_complement_second, sizeof(int));
    for(int i = 0; i < total_size_intersect; i++) {
      final_intersect_vec[i] = intersect_result[i];
    }
    for(int i = 0; i < total_size_complement_first; i++) {
      final_complement_1_vec[i] = complement_first_result[i];
    }
    for(int i = 0; i < total_size_complement_second; i++) {
      final_complement_2_vec[i] = complement_second_result[i];
    }


    int* final_result_vec = (int *) R_alloc(result_index, sizeof(int));
    for(int i = 0; i < result_index; i++) {
      final_result_vec[i] = result_arr[i];
    }

    int tempsize;
    int *nBatchLU_temp = (int *) R_alloc(nBatch, sizeof(int));

    nBatchLU_temp[0] = 0; // starting with the first value

    for(int i = 1; i < nBatch; i++) {
      nBatchLU_temp[i] = nBatchLU_temp[i-1] + tempsize_vec[i-1];
    }

    double *phi_can_vec = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    double *log_g_phi = (double *) R_alloc(N_phi*N_phi, sizeof(double));

    // Calculate XtX
    // Creatte pp*nBatch length vector
    double *XtXmb = (double *) R_alloc(pp*nBatch, sizeof(double));
    double *XtX_temp = (double *) R_alloc(pp, sizeof(double));


    for (int batch_index = 0; batch_index < nBatch; ++batch_index){

      BatchSize = nBatchIndx[batch_index];
      int startRow = nBatchLU[batch_index]; // Calculate based on batch_index
      int endRow = nBatchLU[batch_index]+BatchSize;

      double *X_subset = (double *) R_alloc(BatchSize * p, sizeof(double));

      // Copy the data from X to X_subset
      for (int j = 0; j < p; ++j) { // Iterate over columns
        for (int i = startRow; i < endRow; ++i) { // Iterate over rows within the column
          X_subset[j * BatchSize + (i - startRow)] = X[j * n + i];
        }
      }

      // Use X_subset in dgemm
      F77_NAME(dgemm)(ytran, ntran, &p, &p, &BatchSize, &one, X_subset, &BatchSize, X_subset, &BatchSize, &zero, XtX_temp, &p FCONE FCONE);


      for(i = 0; i < pp; i++){
        XtXmb[batch_index*pp+i] = XtX_temp[i];
        //Rprintf("XtXmb is %f \n", XtXmb[batch_index*pp+i]);
      }
    }
    //double *sum_v = (double *) R_alloc(n, sizeof(double));zeros(sum_v,n);
    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("Initialize Process \n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    if(initial_mu){

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
        for(i = 0; i < p*p; i++){
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



      b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
      //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + *tau_sq_I)*0.5;

      tau_sq = b_tau_update/a_tau_update;
      theta[tauSqIndx] = tau_sq;


      if(verbose){
        Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
        R_FlushConsole();
#endif
      }

      ///////////////
      //update zetasq
      ///////////////

      updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

      double zeta_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
      b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q)*theta[zetaSqIndx]*0.5;
      //Rprintf("zeta_Q: %f \n", zeta_Q);
      //b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q)*0.5;
      zeta_sq = b_zeta_update/a_zeta_update;

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
      for(i = 0; i < n; i++){
        //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
        //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i]/theta[zetaSqIndx] + (y[i])/theta[tauSqIndx]);
        gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc))/theta[tauSqIndx]);

        E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
        delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
        delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
        w_mu_update[i] = w_mu[i] + delta_mu[i];
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

        gamma_gradient_fun(u_vec, epsilon_vec, gamma_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                           B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                           cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,w_mu_temp,w_mu_temp2,gradient);

        vecsum(gamma_gradient_sum, gamma_gradient, Trace_N, n);
      }

      //free(gamma_gradient);

      for(i = 0; i < n; i++){
        E_gamma_sq[i] = rho * E_gamma_sq[i] + (1 - rho) * pow(gamma_gradient_sum[i],2);
        delta_gamma[i] = sqrt(delta_gamma_sq[i]+adadelta_noise)/sqrt(E_gamma_sq[i]+adadelta_noise)*gamma_gradient_sum[i];
        delta_gamma_sq[i] = rho*delta_gamma_sq[i] + (1 - rho) * pow(delta_gamma[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        //S_vi[i] = exp(pow((log(sqrt(S_vi[i])) + delta_gamma[i]),2));
        S_vi[i] = pow(exp(log(sqrt(S_vi[i])) + delta_gamma[i]),2);
        //S_vi[i] = pow(exp(gamma_vec[i]),2);
      }

      zeros(a_gradient,nIndx_vi);
      zeros(a_gradient_sum, nIndx_vi);

      for(int k = 0; k < Trace_N; k++){
        zeros(a_gradient,nIndx_vi);
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }

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



      for(i = 0; i < nIndx_vi; i++){
        E_a_sq[i] = rho * E_a_sq[i] + (1 - rho) * pow(a_gradient_sum[i],2);
        delta_a[i] = sqrt(delta_a_sq[i]+adadelta_noise)/sqrt(E_a_sq[i]+adadelta_noise)*a_gradient_sum[i];
        delta_a_sq[i] = rho*delta_a_sq[i] + (1 - rho) * pow(delta_a[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        A_vi[i] = A_vi[i] + delta_a[i];
      }

      F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);

    }
    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("Updating Process \n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    while(iter <= max_iter & !indicator_converge){
      if(verbose){
        Rprintf("----------------------------------------\n");
        Rprintf("\tIteration at %i \n",iter);
#ifdef Win32
        R_FlushConsole();
#endif
      }
      zeros(gradient_const,n);
      zeros(gradient,n);
      zeros(gamma_gradient_sum, n);
      zeros(gamma_gradient,n);
      zeros(a_gradient,nIndx_vi);
      zeros(a_gradient_sum, nIndx_vi);

      for(int batch_index = 0; batch_index < nBatch; batch_index++){
        tempsize = tempsize_vec[batch_index];
        BatchSize = nBatchIndx[batch_index];
        for(i_mb = 0; i_mb < tempsize; i_mb++){
          epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
        }
        //BatchSize = nBatchIndx[batch_index];

        if(batch_index == iter % nBatch){
          if(verbose){
            Rprintf("the value of batch_index global : %i \n", batch_index);
          }
          a_tau_update = BatchSize * 0.5 + tauSqIGa;
          a_zeta_update = BatchSize * 0.5 + zetaSqIGa;


          ///////////////
          //update beta
          ///////////////

          zeros(tau_sq_I, one_int);
          zeros(tmp_n_mb, n);

          for(i = 0; i < BatchSize; i++){
            tmp_n_mb[nBatchLU[batch_index] + i] = y[nBatchLU[batch_index] + i]-w_mu[nBatchLU[batch_index] + i];
            tau_sq_I[0] += pow(tmp_n_mb[nBatchLU[batch_index] + i],2);
          }

          F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, tmp_n_mb, &inc, &zero, tmp_p, &inc FCONE);

          for(i = 0; i < pp; i++){
            tmp_pp[i] = XtXmb[batch_index*pp+i];
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
              beta_cov[idx] = tmp_pp[idx] * theta[tauSqIndx] * BatchSize / n;
            }
          }

          if(verbose){
            for(i = 0; i < p; i++){
              Rprintf("the value of beta[%i] : %f \n",i, beta[i]);
            }
            for(i = 0; i < p*p; i++){
              Rprintf("the value of beta cov[%i] : %f \n",i, beta_cov[i]);
            }
#ifdef Win32
            R_FlushConsole();
#endif
          }

          ///////////////
          //update tausq
          ///////////////
          int Trace_N_max = Trace_N;
          zeros(tau_sq_H, one_int);
          for(i = 0; i < p; i++){
            tau_sq_H[0] += tmp_p2[i]*tmp_p[i];
          }

          zeros(trace_vec,2);

          for(i_mb = 0; i_mb < tempsize; i_mb++){
            epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
          }


          update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                     batch_index, final_result_vec, nBatchLU_temp, tempsize);

          for(int k = 0; k < Trace_N_max; k++){
            for(i_mb = 0; i_mb < tempsize; i_mb++){
              epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
            }
            update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                       batch_index, final_result_vec, nBatchLU_temp, tempsize);
            double u_mean = 0.0;
            for(i_mb = 0; i_mb < BatchSize; i_mb++){
              u_mean += u_vec[nBatchLU[batch_index] + i_mb];
            }
            u_mean = u_mean/BatchSize;
            for(i_mb = 0; i_mb < BatchSize; i_mb++){
              trace_vec[0] += pow(u_vec[nBatchLU[batch_index] + i_mb]-u_mean,2);
            }
            //trace_vec[1] += Q_mini_batch_plus(B, F, u_vec, u_vec, batch_index, n, nnIndx, nnIndxLU, final_result_vec, nBatchLU_temp, tempsize);
            trace_vec[1] += Q_mini_batch(B, F, u_vec, u_vec, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU);
          }

          //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
          if (!isnan(trace_vec[0])){
            //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + *tau_sq_I)*0.5;
            b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N_max + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5/BatchSize*n;
            a_tau_update = n * 0.5 + tauSqIGa;
            //Rprintf("add_tau is : %f \n",trace_vec[0]/Trace_N + *tau_sq_I);
            tau_sq = b_tau_update/a_tau_update;
            theta[tauSqIndx] = tau_sq;
          }else{
            theta[tauSqIndx] = 1;
          }



          if(verbose){
            Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
            R_FlushConsole();
#endif
          }

          ///////////////
          //update zetasq
          ///////////////
          updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                                  theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
                                  batch_index, final_result_vec, nBatchLU_temp, tempsize);

          //double zeta_Q_mb = Q_mini_batch_plus(B, F, w_mu, w_mu, batch_index, n, nnIndx, nnIndxLU, final_result_vec, nBatchLU_temp, tempsize);
          double zeta_Q_mb = Q_mini_batch(B, F, w_mu, w_mu, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU);

          //Rprintf("zeta_Q_mb: %f \n", zeta_Q_mb);
          if (!isnan(trace_vec[1])){
           // b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q_mb)*theta[zetaSqIndx]*0.5;
            b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N_max + zeta_Q_mb)*theta[zetaSqIndx]*0.5/BatchSize*n;
            a_zeta_update = n * 0.5 + zetaSqIGa;

            zeta_sq = b_zeta_update/a_zeta_update;
            if(zeta_sq > 1000){zeta_sq = 1000;}
            theta[zetaSqIndx] = zeta_sq;
          }else{
            theta[zetaSqIndx] = 1;
          }



          if(verbose){
            Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
            R_FlushConsole();
#endif
          }

          updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                                  theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
                                  batch_index, final_result_vec, nBatchLU_temp, tempsize);

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

              // updateBF_minibatch(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
              //                    theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb,
              //                    BatchSize, nBatchLU, batch_index);
              updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                                      theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb,
                                      batch_index, final_result_vec, nBatchLU_temp, tempsize);
              // update_uvec_minibatch(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
              //                       BatchSize, nBatchLU, batch_index);
              update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                         batch_index, final_result_vec, nBatchLU_temp, tempsize);
              logDetInv = 0.0;
              diag_sigma_sq_sum = 0.0;
              for(i_mb = 0; i_mb < BatchSize; i_mb++){
                j = nBatchLU[batch_index] + i_mb;
                logDetInv += log(1/F[j]);
              }

              log_g_phi[i] = logDetInv*0.5 -
                (Q_mini_batch(B, F, u_vec, u_vec, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU)+
                Q_mini_batch(B, F, w_mu, w_mu, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU))*0.5;
            }

            max_index = max_ind(log_g_phi,N_phi*N_phi);
            a_phi = a_phi_vec[max_index/N_phi];
            b_phi = b_phi_vec[max_index % N_phi];

            theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;

            // updateBF_minibatch(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
            //                    theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb,
            //                    BatchSize, nBatchLU, batch_index);
            updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                                    theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
                                    batch_index, final_result_vec, nBatchLU_temp, tempsize);
          }

          if(verbose){
            Rprintf("the value of theta[%i phiIndx] : %f \n", phiIndx, theta[phiIndx]);
#ifdef Win32
            R_FlushConsole();
#endif
          }
        }
        //for(int batch_index = 0; batch_index < nBatch; batch_index++)
        if(verbose){
          Rprintf("the value of batch_index for w : %i \n", batch_index);
#ifdef Win32
          R_FlushConsole();
#endif
        }
        tempsize = tempsize_vec[batch_index];
        BatchSize = nBatchIndx[batch_index];
        ///////////////
        //update w
        ///////////////

        //zeros_minibatch(w_mu_temp,n, BatchSize, nBatchLU, batch_index);
        //zeros_minibatch(w_mu_temp2,n, BatchSize, nBatchLU, batch_index);
        double gradient_mu;
        zeros(w_mu_temp,n);
        zeros(w_mu_temp_dF,n);
        zeros(w_mu_temp2,n);
        zeros(gradient_const,n);
        zeros(gradient,n);
        zeros(gamma_gradient_sum, n);
        zeros(gamma_gradient,n);
        zeros(a_gradient,nIndx_vi);
        zeros(a_gradient_sum, nIndx_vi);


        product_B_F_minibatch_plus(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
        product_B_F_minibatch_term1(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);
        product_B_F_vec_minibatch_plus_fix(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);

        for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
          i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
          //gradient_mu_vec[i] =  (y[i] - w_mu[i])/theta[tauSqIndx] - w_mu_temp2[i];
          gradient_mu_vec[i] =  (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc) - w_mu[i])/theta[tauSqIndx] - w_mu_temp2[i];
        }

        for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
          i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
          //gradient_mu_vec[i] =  (y[i] - w_mu[i])/theta[tauSqIndx] - w_mu_temp_dF[i];
          gradient_mu_vec[i] =  (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc) - w_mu[i])/theta[tauSqIndx] - w_mu_temp_dF[i];
        }

        for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
          i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
          gradient_mu_vec[i] = - w_mu_temp2[i] + w_mu_temp_dF[i];
        }

        for(i_mb = 0; i_mb < tempsize; i_mb++){
          i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
          //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
          gradient_mu = gradient_mu_vec[i];
          E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
          delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
          delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
          w_mu_update[i] = w_mu[i] + delta_mu[i];
        }

        zeros(w_mu_temp,n);
        zeros(w_mu_temp_dF,n);
        zeros(w_mu_temp2,n);
        product_B_F_minibatch_plus(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
        product_B_F_minibatch_term1(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);
        product_B_F_vec_minibatch_plus_fix(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);



        zeros(gamma_gradient_sum, n);
        for(int k = 0; k < Trace_N; k++){
          //zeros_minibatch_plus(gamma_gradient,n, batch_index,final_result_vec, nBatchLU_temp, tempsize);
          zeros(gradient,n);
          zeros(gamma_gradient,n);
          for(i_mb = 0; i_mb < tempsize; i_mb++){
            epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
          }

          gamma_gradient_fun_minibatch_beta(y, X, beta, p, w_mu_update,
                                            w_mu_temp_dF, w_mu_temp2,
                                            u_vec, epsilon_vec, gamma_gradient,
                                            A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                            B, F, nnIndx, nnIndxLU, theta, tauSqIndx,
                                            cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                                            cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,
                                            u_vec_temp, u_vec_temp2, u_vec_temp_dF, gradient,
                                            batch_index, BatchSize, nBatchLU,
                                            final_result_vec, nBatchLU_temp, tempsize,
                                            intersect_start_indices, intersect_sizes, final_intersect_vec,
                                            complement_first_start_indices, complement_first_sizes, final_complement_1_vec,
                                            complement_second_start_indices, complement_second_sizes, final_complement_2_vec);

          vecsum_minibatch_plus(gamma_gradient_sum, gamma_gradient, Trace_N, n, batch_index, final_result_vec, nBatchLU_temp, tempsize);

        }

//         if(verbose){
//           for(int i_mb = 0; i_mb < 25; i_mb++){
//             i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
//             Rprintf("the value of gamma_gradient_sum[%i] : %f \n",i, gamma_gradient_sum[i]);
//           }
//           for(int i_mb = 0; i_mb < 25; i_mb++){
//             i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
//             Rprintf("the value of delta_gamma[%i] : %f \n",i, delta_gamma[i]);
//           }
// #ifdef Win32
//           R_FlushConsole();
// #endif
//         }

        for(i_mb = 0; i_mb < tempsize; i_mb++){
          i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
          //Rprintf("gamma gradient[%i],: %f \n",i, gamma_gradient_sum[i]);
          E_gamma_sq[i] = rho * E_gamma_sq[i] + (1 - rho) * pow(gamma_gradient_sum[i],2);
          delta_gamma[i] = sqrt(delta_gamma_sq[i]+adadelta_noise)/sqrt(E_gamma_sq[i]+adadelta_noise)*gamma_gradient_sum[i];
          delta_gamma_sq[i] = rho*delta_gamma_sq[i] + (1 - rho) * pow(delta_gamma[i],2);
          S_vi[i] = pow(exp(log(sqrt(S_vi[i])) + delta_gamma[i]),2);
        }

        zeros(a_gradient_sum, nIndx_vi);
        for(int k = 0; k < Trace_N; k++){
          zeros(gradient,n);
          zeros(a_gradient,nIndx_vi);
          zeros(w_mu_temp,n);
          for(i_mb = 0; i_mb < tempsize; i_mb++){
            epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
          }

          a_gradient_fun_minibatch_beta(y, X, beta, p, w_mu_update,
                                        w_mu_temp_dF, w_mu_temp2,
                                        u_vec, epsilon_vec, a_gradient, gradient_const,
                                        A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                        B, F, nnIndx, nnIndxLU, theta, tauSqIndx,
                                        cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                                        u_vec_temp, u_vec_temp2, u_vec_temp_dF, gradient,
                                        batch_index, BatchSize, nBatchLU,
                                        final_result_vec, nBatchLU_temp, tempsize,
                                        intersect_start_indices, intersect_sizes, final_intersect_vec,
                                        complement_first_start_indices, complement_first_sizes, final_complement_1_vec,
                                        complement_second_start_indices, complement_second_sizes, final_complement_2_vec);

          //vecsum(a_gradient_sum, a_gradient, Trace_N, nIndx_vi);
          for(int i_mb = 0; i_mb < tempsize; i_mb++){
            i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
            for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
              a_gradient_sum[nnIndxLU_vi[i] + l] += a_gradient[nnIndxLU_vi[i] + l]/Trace_N;
            }
          }


        }

//         if(verbose){
//           for(i_mb = 0; i_mb < 25; i_mb++){
//             i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
//             for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
//               int sub_index = nnIndxLU_vi[i] + l;
//               Rprintf("the value of a_gradient_sum[%i] : %f \n",sub_index, a_gradient_sum[sub_index]);
//
//             }
//           }
//
// #ifdef Win32
//               R_FlushConsole();
// #endif
//         }

        int sub_index;
        //Rprintf("A_vi: ");
        for(int i_mb = 0; i_mb < tempsize; i_mb++){
          i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
          for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
            sub_index = nnIndxLU_vi[i] + l;
            //Rprintf("a gradient[%i],: %f \n",sub_index, a_gradient_sum[sub_index]);
            //a_gradient_sum[nnIndxLU_vi[i] + l] += a_gradient[nnIndxLU_vi[i] + l]/Trace_N;
            E_a_sq[sub_index] = rho * E_a_sq[sub_index] + (1 - rho) * pow(a_gradient_sum[sub_index],2);
            delta_a[sub_index] = sqrt(delta_a_sq[sub_index]+adadelta_noise)/sqrt(E_a_sq[sub_index]+adadelta_noise)*a_gradient_sum[sub_index];
            delta_a_sq[sub_index] = rho*delta_a_sq[sub_index] + (1 - rho) * pow(delta_a[sub_index],2);
            A_vi[sub_index] = A_vi[sub_index] + delta_a[sub_index];
            //Rprintf("A_vi[i]: %i, %f \n",sub_index, A_vi[sub_index]);
            //Rprintf("\t Updated a index is %i \n",sub_index);
          }
        }
        //Rprintf("\n");
        F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);

      }


      //Calculate the first part of ELBO
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

      ELBO_vec[iter-1] = - ELBO;


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


      F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);


    }

    //
    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
    //zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
    SEXP theta_para_r; PROTECT(theta_para_r = allocVector(REALSXP, nTheta*2)); nProtect++; double *theta_para = REAL(theta_para_r);

    theta_para[zetaSqIndx*2+0] = a_zeta_update;
    theta_para[zetaSqIndx*2+1] = b_zeta_update;

    theta_para[tauSqIndx*2+0] = a_tau_update;
    theta_para[tauSqIndx*2+1] = b_tau_update;

    theta_para[phiIndx*2+0] = a_phi;
    theta_para[phiIndx*2+1] = b_phi;

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

  SEXP spVarBayes_NNGP_nocovariates_mb_betacpp(SEXP y_r,
                                               SEXP n_r, SEXP p_r, SEXP m_r, SEXP m_vi_r, SEXP coords_r, SEXP covModel_r, SEXP rho_r,
                                               SEXP zetaSqIG_r, SEXP tauSqIG_r, SEXP phibeta_r, SEXP nuUnif_r,
                                               SEXP zetaSqStarting_r, SEXP tauSqStarting_r, SEXP phiStarting_r, SEXP nuStarting_r,
                                               SEXP sType_r, SEXP nThreads_r, SEXP verbose_r, SEXP fix_nugget_r, SEXP N_phi_r, SEXP Trace_N_r,
                                               SEXP max_iter_r,
                                               SEXP var_input_r,
                                               SEXP phi_input_r, SEXP phi_iter_max_r, SEXP initial_mu_r,
                                               SEXP mini_batch_size_r,
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
    double phi_input  =  REAL(phi_input_r)[0];
    double *var_input  =  REAL(var_input_r);
    int initial_mu  =  INTEGER(initial_mu_r)[0];
    int phi_iter_max = INTEGER(phi_iter_max_r)[0];
    int n_mb = INTEGER(mini_batch_size_r)[0];

    int K = INTEGER(K_r)[0];
    int stop_K = INTEGER(stop_K_r)[0];
    int min_iter = INTEGER(min_iter_r)[0];

    int nThreads = INTEGER(nThreads_r)[0];
    int verbose = INTEGER(verbose_r)[0];
    //double  vi_threshold  =  REAL(vi_threshold_r)[0];
    double  rho_input  =  REAL(rho_r)[0];
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
    int nBatch = static_cast<int>(std::ceil(static_cast<double>(n)/static_cast<double>(n_mb)));
    int *nBatchIndx = (int *) R_alloc(nBatch, sizeof(int));
    int *nBatchLU = (int *) R_alloc(nBatch, sizeof(int));
    get_nBatchIndx(n, nBatch, n_mb, nBatchIndx, nBatchLU);
    if(verbose){
      Rprintf("Using %i nBatch \n", nBatch);

      for(int i = 0; i < nBatch; i++){
        Rprintf("the value of nBatchIndx[%i] : %i \n",i, nBatchIndx[i]);
        Rprintf("the value of nBatchLU[%i] : %i \n",i, nBatchLU[i]);
      }
#ifdef Win32
      R_FlushConsole();
#endif
    }



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
    double *w_mu_temp_dF = (double *) R_alloc(n, sizeof(double));
    double *u_vec_temp_dF = (double *) R_alloc(n, sizeof(double));
    double *gradient_mu_vec = (double *) R_alloc(n, sizeof(double));

    double *gradient_const = (double *) R_alloc(n, sizeof(double));
    double *gradient = (double *) R_alloc(n, sizeof(double));
    double *gamma_gradient_sum = (double *) R_alloc(n, sizeof(double));

    double *u_vec_temp = (double *) R_alloc(n, sizeof(double));
    double *u_vec_temp2 = (double *) R_alloc(n, sizeof(double));

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

    double *tmp_n_mb = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n_mb, n);
    double *diag_input_mb = (double *) R_alloc(n, sizeof(double)); zeros(diag_input_mb, n);

    int BatchSize;
    double sum_diags= 0.0;
    int i_mb;
    double *rademacher_rv_vec = (double *) R_alloc(n, sizeof(double));
    double *rademacher_rv_temp = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp,n);
    double *rademacher_rv_temp2 = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp2,n);

    double *product_v = (double *) R_alloc(n, sizeof(double));zeros(product_v,n);
    double *product_v2 = (double *) R_alloc(n, sizeof(double));zeros(product_v2,n);
    double *e_i = (double *) R_alloc(n, sizeof(double));zeros(e_i,n);

    int batch_index = 0;

    int max_result_size = nBatch * n;
    int max_temp_size = n;

    int* result_arr = (int *) R_alloc(max_result_size, sizeof(int));
    int* temp_arr = (int *) R_alloc(max_temp_size, sizeof(int));
    int result_index = 0;
    int temp_index = 0;

    int* tempsize_vec = (int *) R_alloc(nBatch, sizeof(int));

    // Usage:
    int *seen_values = (int *) R_alloc(n, sizeof(int));

    // Assuming max possible size for all results is n
    int *intersect_result = (int *) R_alloc(max_result_size, sizeof(int));
    int *complement_first_result = (int *) R_alloc(max_result_size, sizeof(int));
    int *complement_second_result = (int *) R_alloc(max_result_size, sizeof(int));

    // Allocate and initialize indices and sizes arrays
    int *intersect_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *intersect_sizes = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_first_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_first_sizes = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_second_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_second_sizes = (int *) R_alloc(nBatch, sizeof(int));

    // Initialize result indices
    int intersect_result_index = 0;
    int complement_first_result_index = 0;
    int complement_second_result_index = 0;

    for (int batch_index = 0; batch_index < nBatch; ++batch_index) {
      zeros_int(seen_values,n);
      BatchSize = nBatchIndx[batch_index];
      find_set_nngp(n, nnIndx, nnIndxLU, BatchSize, nBatchLU, batch_index,
                    seen_values,
                    intersect_result, intersect_sizes, intersect_start_indices,
                    complement_first_result, complement_first_sizes, complement_first_start_indices,
                    complement_second_result, complement_second_sizes, complement_second_start_indices,
                    intersect_result_index, complement_first_result_index, complement_second_result_index);

      zeros_int(seen_values,n);

      find_set_mb(n, nnIndx, nnIndxLU, nnIndxCol, numIndxCol, nnIndxnnCol, cumnumIndxCol,
                  BatchSize, nBatchLU, batch_index, result_arr, result_index, temp_arr, temp_index, tempsize_vec, seen_values);


    }

    int total_size_intersect = 0;
    int total_size_complement_first  = 0;
    int total_size_complement_second = 0;

    for (int i = 0; i < nBatch; ++i) {
      total_size_intersect         += intersect_sizes[i];
      total_size_complement_first  += complement_first_sizes[i];
      total_size_complement_second += complement_second_sizes[i];
    }

    if(verbose){

      for (int i = 0; i < nBatch; ++i) {
        Rprintf("intersect_sizes is %i ", intersect_sizes[i]);
        Rprintf("complement_first_sizes is %i ", complement_first_sizes[i]);
        Rprintf("complement_second_sizes is %i ", complement_second_sizes[i]);
      }
      Rprintf("\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }

    int* final_intersect_vec = (int *) R_alloc(total_size_intersect, sizeof(int));
    int* final_complement_1_vec = (int *) R_alloc(total_size_complement_first, sizeof(int));
    int* final_complement_2_vec = (int *) R_alloc(total_size_complement_second, sizeof(int));
    for(int i = 0; i < total_size_intersect; i++) {
      final_intersect_vec[i] = intersect_result[i];
    }
    for(int i = 0; i < total_size_complement_first; i++) {
      final_complement_1_vec[i] = complement_first_result[i];
    }
    for(int i = 0; i < total_size_complement_second; i++) {
      final_complement_2_vec[i] = complement_second_result[i];
    }


    int* final_result_vec = (int *) R_alloc(result_index, sizeof(int));
    for(int i = 0; i < result_index; i++) {
      final_result_vec[i] = result_arr[i];
    }



    if(verbose){
      Rprintf("tempsize_vec: ");
      for (int i = 0; i < nBatch; i++) {
        Rprintf("%i ", tempsize_vec[i]);
      }
      Rprintf("\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    int tempsize;
    int *nBatchLU_temp = (int *) R_alloc(nBatch, sizeof(int));

    nBatchLU_temp[0] = 0; // starting with the first value

    for(int i = 1; i < nBatch; i++) {
      nBatchLU_temp[i] = nBatchLU_temp[i-1] + tempsize_vec[i-1];
    }

    double *phi_can_vec = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    double *log_g_phi = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    double *sum_v = (double *) R_alloc(n, sizeof(double));zeros(sum_v,n);

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("Initialize Process \n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    if(initial_mu){
      double rho = 0.1;
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

      tau_sq = b_tau_update/a_tau_update;
      theta[tauSqIndx] = tau_sq;


      if(verbose){
        Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
        R_FlushConsole();
#endif
      }

      ///////////////
      //update zetasq
      ///////////////

      updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

      double zeta_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
      b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q)*theta[zetaSqIndx]*0.5;
      //Rprintf("zeta_Q: %f \n", zeta_Q);
      //b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q)*0.5;
      zeta_sq = b_zeta_update/a_zeta_update;

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
      for(i = 0; i < n; i++){
        gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
        //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i]/theta[zetaSqIndx] + (y[i])/theta[tauSqIndx]);
        E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
        delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
        delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
        w_mu_update[i] = w_mu[i] + delta_mu[i];
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

        gamma_gradient_fun(u_vec, epsilon_vec, gamma_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                           B, F, nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                           cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,w_mu_temp,w_mu_temp2,gradient);

        vecsum(gamma_gradient_sum, gamma_gradient, Trace_N, n);
      }

      //free(gamma_gradient);

      for(i = 0; i < n; i++){
        E_gamma_sq[i] = rho * E_gamma_sq[i] + (1 - rho) * pow(gamma_gradient_sum[i],2);
        delta_gamma[i] = sqrt(delta_gamma_sq[i]+adadelta_noise)/sqrt(E_gamma_sq[i]+adadelta_noise)*gamma_gradient_sum[i];
        delta_gamma_sq[i] = rho*delta_gamma_sq[i] + (1 - rho) * pow(delta_gamma[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        //S_vi[i] = exp(pow((log(sqrt(S_vi[i])) + delta_gamma[i]),2));
        S_vi[i] = pow(exp(log(sqrt(S_vi[i])) + delta_gamma[i]),2);
        //S_vi[i] = pow(exp(gamma_vec[i]),2);
      }

      zeros(a_gradient,nIndx_vi);
      zeros(a_gradient_sum, nIndx_vi);

      for(int k = 0; k < Trace_N; k++){
        zeros(a_gradient,nIndx_vi);
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }

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



      for(i = 0; i < nIndx_vi; i++){
        E_a_sq[i] = rho * E_a_sq[i] + (1 - rho) * pow(a_gradient_sum[i],2);
        delta_a[i] = sqrt(delta_a_sq[i]+adadelta_noise)/sqrt(E_a_sq[i]+adadelta_noise)*a_gradient_sum[i];
        delta_a_sq[i] = rho*delta_a_sq[i] + (1 - rho) * pow(delta_a[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        A_vi[i] = A_vi[i] + delta_a[i];
      }

      F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);

    }
    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("Updating Process \n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    while(iter <= max_iter & !indicator_converge){
      if(verbose){
        Rprintf("----------------------------------------\n");
        Rprintf("\tIteration at %i \n",iter);
#ifdef Win32
        R_FlushConsole();
#endif
      }
      zeros(gradient_const,n);
      zeros(gradient,n);
      zeros(gamma_gradient_sum, n);
      zeros(gamma_gradient,n);
      zeros(a_gradient,nIndx_vi);
      zeros(a_gradient_sum, nIndx_vi);
      double rho = rho_input;
      // if(iter < 100){
      //   rho = 0.95;
      // }else{
      //   rho = rho_input;
      // }
      for(int batch_index = 0; batch_index < nBatch; batch_index++){
        tempsize = tempsize_vec[batch_index];
        BatchSize = nBatchIndx[batch_index];
        for(i_mb = 0; i_mb < tempsize; i_mb++){
          epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
        }
        //BatchSize = nBatchIndx[batch_index];

        if(batch_index == iter % nBatch){
          if(verbose){
            Rprintf("the value of batch_index global : %i \n", batch_index);
#ifdef Win32
            R_FlushConsole();
#endif
          }
          a_tau_update = BatchSize * 0.5 + tauSqIGa;
          a_zeta_update = BatchSize * 0.5 + zetaSqIGa;

          zeros(tau_sq_I, one_int);
          zeros(tmp_n_mb, n);

          for(i = 0; i < BatchSize; i++){
            tmp_n_mb[i] = y[nBatchLU[batch_index] + i]-w_mu[nBatchLU[batch_index] + i];
            tau_sq_I[0] += pow(tmp_n_mb[i],2);
          }

          ///////////////
          //update tausq
          ///////////////

          zeros(trace_vec,2);

          for(i_mb = 0; i_mb < tempsize; i_mb++){
            epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
          }


          update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                     batch_index, final_result_vec, nBatchLU_temp, tempsize);

          for(int k = 0; k < Trace_N; k++){
            for(i_mb = 0; i_mb < tempsize; i_mb++){
              epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
            }
            update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                       batch_index, final_result_vec, nBatchLU_temp, tempsize);

            double u_mean = 0.0;
            for(i_mb = 0; i_mb < BatchSize; i_mb++){
              u_mean += u_vec[nBatchLU[batch_index] + i_mb];
            }
            u_mean = u_mean/BatchSize;
            for(i_mb = 0; i_mb < BatchSize; i_mb++){
              trace_vec[0] += pow(u_vec[nBatchLU[batch_index] + i_mb]-u_mean,2);
            }

            //trace_vec[1] += Q_mini_batch_plus(B, F, u_vec, u_vec, batch_index, n, nnIndx, nnIndxLU, final_result_vec, nBatchLU_temp, tempsize);
            trace_vec[1] += Q_mini_batch(B, F, u_vec, u_vec, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU);
          }

          //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
          if (!isnan(trace_vec[0])){
            a_tau_update = n * 0.5 + tauSqIGa;
            b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + *tau_sq_I)*0.5/BatchSize*n;
            //Rprintf("add_tau is : %f \n",trace_vec[0]/Trace_N + *tau_sq_I);
            tau_sq = b_tau_update/a_tau_update;
            theta[tauSqIndx] = tau_sq;
          }else{
            theta[tauSqIndx] = 1;
          }


          if(verbose){
            Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
            R_FlushConsole();
#endif
          }

          ///////////////
          //update zetasq
          ///////////////
          updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                                  theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
                                  batch_index, final_result_vec, nBatchLU_temp, tempsize);
          double zeta_Q_mb = Q_mini_batch(B, F, w_mu, w_mu, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU);

          //Rprintf("zeta_Q_mb: %f \n", zeta_Q_mb);
          if (!isnan(trace_vec[1])){
            b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q_mb)*theta[zetaSqIndx]*0.5/BatchSize*n;
            a_zeta_update = BatchSize/BatchSize*n * 0.5 + zetaSqIGa;
            //b_zeta_update = zetaSqIGb + (trace_vec[1]/Trace_N + zeta_Q_mb)*0.5;
            //b_zeta_update = zetaSqIGb + trace_vec[1]/Trace_N*theta[zetaSqIndx]*0.5;
            //Rprintf("add zeta is : %f \n",trace_vec[1]/Trace_N);
            zeta_sq = b_zeta_update/a_zeta_update;
            //zeta_sq = 13.227241;
            if(zeta_sq > 1000){zeta_sq = 1000;}
            theta[zetaSqIndx] = zeta_sq;
          }else{
            theta[zetaSqIndx] = 1;
          }


          if(verbose){
            Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
            R_FlushConsole();
#endif
          }

          // updateBF_minibatch(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //                    theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
          //                    BatchSize, nBatchLU, batch_index);
          updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                                  theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
                                  batch_index, final_result_vec, nBatchLU_temp, tempsize);

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

              // updateBF_minibatch(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
              //                    theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb,
              //                    BatchSize, nBatchLU, batch_index);
              updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                                      theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb,
                                      batch_index, final_result_vec, nBatchLU_temp, tempsize);
              // update_uvec_minibatch(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
              //                       BatchSize, nBatchLU, batch_index);
              update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                         batch_index, final_result_vec, nBatchLU_temp, tempsize);
              logDetInv = 0.0;
              diag_sigma_sq_sum = 0.0;
              for(i_mb = 0; i_mb < BatchSize; i_mb++){
                j = nBatchLU[batch_index] + i_mb;
                logDetInv += log(1/F[j]);
              }

              log_g_phi[i] = logDetInv*0.5 -
                (Q_mini_batch(B, F, u_vec, u_vec, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU)+
                Q_mini_batch(B, F, w_mu, w_mu, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU))*0.5;
            }

            max_index = max_ind(log_g_phi,N_phi*N_phi);
            a_phi = a_phi_vec[max_index/N_phi];
            b_phi = b_phi_vec[max_index % N_phi];

            theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;

            // updateBF_minibatch(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
            //                    theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb,
            //                    BatchSize, nBatchLU, batch_index);
            updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
                                    theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
                                    batch_index, final_result_vec, nBatchLU_temp, tempsize);
          }

          if(verbose){
            Rprintf("the value of theta[%i phiIndx] : %f \n", phiIndx, theta[phiIndx]);
#ifdef Win32
            R_FlushConsole();
#endif
          }
        }
        //for(int batch_index = 0; batch_index < nBatch; batch_index++)
          if(verbose){
            Rprintf("the value of batch_index for w : %i \n", batch_index);
#ifdef Win32
            R_FlushConsole();
#endif
          }
          tempsize = tempsize_vec[batch_index];
          BatchSize = nBatchIndx[batch_index];
          ///////////////
          //update w
          ///////////////

          //zeros_minibatch(w_mu_temp,n, BatchSize, nBatchLU, batch_index);
          //zeros_minibatch(w_mu_temp2,n, BatchSize, nBatchLU, batch_index);
          double gradient_mu;
          zeros(w_mu_temp,n);
          zeros(w_mu_temp_dF,n);
          zeros(w_mu_temp2,n);
          product_B_F_minibatch_plus(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
          product_B_F_minibatch_term1(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);
          product_B_F_vec_minibatch_plus_fix(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);

          for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
            i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
            gradient_mu_vec[i] =  (y[i] - w_mu[i])/theta[tauSqIndx] - w_mu_temp2[i];
          }

          for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
            i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
            gradient_mu_vec[i] =  (y[i] - w_mu[i])/theta[tauSqIndx] - w_mu_temp_dF[i];
          }

          for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
            i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
            gradient_mu_vec[i] = - w_mu_temp2[i] + w_mu_temp_dF[i];
          }

          for(i_mb = 0; i_mb < tempsize; i_mb++){
            i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
            //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
            gradient_mu = gradient_mu_vec[i];
            E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
            delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
            delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
            w_mu_update[i] = w_mu[i] + delta_mu[i];
          }

          zeros(w_mu_temp,n);
          zeros(w_mu_temp_dF,n);
          zeros(w_mu_temp2,n);
          product_B_F_minibatch_plus(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
          product_B_F_minibatch_term1(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);
          product_B_F_vec_minibatch_plus_fix(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);



          zeros(gamma_gradient_sum, n);
          for(int k = 0; k < Trace_N; k++){
            //zeros_minibatch_plus(gamma_gradient,n, batch_index,final_result_vec, nBatchLU_temp, tempsize);
            zeros(gradient,n);
            zeros(gamma_gradient,n);
            for(i_mb = 0; i_mb < tempsize; i_mb++){
              epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
            }

            gamma_gradient_fun_minibatch_test(y, w_mu_update,
                                              w_mu_temp_dF, w_mu_temp2,
                                              u_vec, epsilon_vec, gamma_gradient,
                                              A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                              B, F, nnIndx, nnIndxLU, theta, tauSqIndx,
                                              cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                                              cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,
                                              u_vec_temp, u_vec_temp2, u_vec_temp_dF, gradient,
                                              batch_index, BatchSize, nBatchLU,
                                              final_result_vec, nBatchLU_temp, tempsize,
                                              intersect_start_indices, intersect_sizes, final_intersect_vec,
                                              complement_first_start_indices, complement_first_sizes, final_complement_1_vec,
                                              complement_second_start_indices, complement_second_sizes, final_complement_2_vec);

            vecsum_minibatch_plus(gamma_gradient_sum, gamma_gradient, Trace_N, n, batch_index, final_result_vec, nBatchLU_temp, tempsize);

          }

          for(i_mb = 0; i_mb < tempsize; i_mb++){
            i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
            //Rprintf("gamma gradient[%i],: %f \n",i, gamma_gradient_sum[i]);
            E_gamma_sq[i] = rho * E_gamma_sq[i] + (1 - rho) * pow(gamma_gradient_sum[i],2);
            delta_gamma[i] = sqrt(delta_gamma_sq[i]+adadelta_noise)/sqrt(E_gamma_sq[i]+adadelta_noise)*gamma_gradient_sum[i];
            delta_gamma_sq[i] = rho*delta_gamma_sq[i] + (1 - rho) * pow(delta_gamma[i],2);
            S_vi[i] = pow(exp(log(sqrt(S_vi[i])) + delta_gamma[i]),2);
          }

          zeros(a_gradient_sum, nIndx_vi);
          for(int k = 0; k < Trace_N; k++){
            zeros(gradient,n);
            zeros(a_gradient,nIndx_vi);
            zeros(w_mu_temp,n);
            for(i_mb = 0; i_mb < tempsize; i_mb++){
              epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
            }

            a_gradient_fun_minibatch_test(y, w_mu_update,
                                          w_mu_temp_dF, w_mu_temp2,
                                          u_vec, epsilon_vec, a_gradient, gradient_const,
                                          A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                          B, F, nnIndx, nnIndxLU, theta, tauSqIndx,
                                          cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                                          u_vec_temp, u_vec_temp2, u_vec_temp_dF, gradient,
                                          batch_index, BatchSize, nBatchLU,
                                          final_result_vec, nBatchLU_temp, tempsize,
                                          intersect_start_indices, intersect_sizes, final_intersect_vec,
                                          complement_first_start_indices, complement_first_sizes, final_complement_1_vec,
                                          complement_second_start_indices, complement_second_sizes, final_complement_2_vec);

            //vecsum(a_gradient_sum, a_gradient, Trace_N, nIndx_vi);
            for(int i_mb = 0; i_mb < tempsize; i_mb++){
              i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
              for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
                a_gradient_sum[nnIndxLU_vi[i] + l] += a_gradient[nnIndxLU_vi[i] + l]/Trace_N;
              }
            }


          }
          int sub_index;
          //Rprintf("A_vi: ");
          for(int i_mb = 0; i_mb < tempsize; i_mb++){
            i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
            for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
              sub_index = nnIndxLU_vi[i] + l;
              //Rprintf("a gradient[%i],: %f \n",sub_index, a_gradient_sum[sub_index]);
              //a_gradient_sum[nnIndxLU_vi[i] + l] += a_gradient[nnIndxLU_vi[i] + l]/Trace_N;
              E_a_sq[sub_index] = rho * E_a_sq[sub_index] + (1 - rho) * pow(a_gradient_sum[sub_index],2);
              delta_a[sub_index] = sqrt(delta_a_sq[sub_index]+adadelta_noise)/sqrt(E_a_sq[sub_index]+adadelta_noise)*a_gradient_sum[sub_index];
              delta_a_sq[sub_index] = rho*delta_a_sq[sub_index] + (1 - rho) * pow(delta_a[sub_index],2);
              A_vi[sub_index] = A_vi[sub_index] + delta_a[sub_index];
              //Rprintf("A_vi[i]: %i, %f \n",sub_index, A_vi[sub_index]);
              //Rprintf("\t Updated a index is %i \n",sub_index);
            }
          }
          //Rprintf("\n");
          F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);



      }

      // Rprintf("\t rho at %i is %f\n",iter,rho);
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

      ELBO_vec[iter-1] = -ELBO;
      // if(iter == 1){create_sign(delta_a, sign_vec_old, n_per);}
      // if(iter % 10){
      //   create_sign(delta_mu, sign_vec_new, n_per);
      //   checksign(sign_vec_old, sign_vec_new, check_vec, n_per);
      //   indicator_converge = prodsign(check_vec ,n_per);
      //   memcpy(sign_vec_old, sign_vec_new, n_per * sizeof(int));
      // }

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


      F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);


    }

    //
    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
    //zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
    SEXP theta_para_r; PROTECT(theta_para_r = allocVector(REALSXP, nTheta*2)); nProtect++; double *theta_para = REAL(theta_para_r);

    theta_para[zetaSqIndx*2+0] = a_zeta_update;
    theta_para[zetaSqIndx*2+1] = b_zeta_update;

    theta_para[tauSqIndx*2+0] = a_tau_update;
    theta_para[tauSqIndx*2+1] = b_tau_update;

    theta_para[phiIndx*2+0] = a_phi;
    theta_para[phiIndx*2+1] = b_phi;


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

  SEXP spVarBayes_NNGP_mb_beta_rephicpp(SEXP y_r, SEXP X_r,
                                        SEXP n_r, SEXP p_r, SEXP m_r, SEXP m_vi_r, SEXP coords_r, SEXP covModel_r, SEXP rho_r,
                                        SEXP zetaSqIG_r, SEXP tauSqIG_r, SEXP phibeta_r, SEXP nuUnif_r,
                                        SEXP zetaSqStarting_r, SEXP tauSqStarting_r, SEXP phiStarting_r, SEXP nuStarting_r,
                                        SEXP sType_r, SEXP nThreads_r, SEXP verbose_r, SEXP fix_nugget_r, SEXP N_phi_r, SEXP Trace_N_r,
                                        SEXP max_iter_r,
                                        SEXP var_input_r,
                                        SEXP phi_input_r, SEXP phi_iter_max_r,  SEXP initial_mu_r,
                                        SEXP mini_batch_size_r,
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
    int n_mb = INTEGER(mini_batch_size_r)[0];

    int K = INTEGER(K_r)[0];
    int stop_K = INTEGER(stop_K_r)[0];
    int min_iter = INTEGER(min_iter_r)[0];

    int nThreads = INTEGER(nThreads_r)[0];
    int verbose = INTEGER(verbose_r)[0];
    //double  vi_threshold  =  REAL(vi_threshold_r)[0];
    double  rho_input  =  REAL(rho_r)[0];
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
    int nBatch = static_cast<int>(std::ceil(static_cast<double>(n)/static_cast<double>(n_mb)));
    int *nBatchIndx = (int *) R_alloc(nBatch, sizeof(int));
    int *nBatchLU = (int *) R_alloc(nBatch, sizeof(int));
    get_nBatchIndx(n, nBatch, n_mb, nBatchIndx, nBatchLU);
    if(verbose){
      Rprintf("Using %i nBatch \n", nBatch);

      for(int i = 0; i < nBatch; i++){
        Rprintf("the value of nBatchIndx[%i] : %i \n",i, nBatchIndx[i]);
        Rprintf("the value of nBatchLU[%i] : %i \n",i, nBatchLU[i]);
      }
#ifdef Win32
      R_FlushConsole();
#endif
    }



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

    double *tmp_pp = (double *) R_alloc(pp, sizeof(double));
    double *tmp_p = (double *) R_alloc(p, sizeof(double));
    double *tmp_p2 = (double *) R_alloc(p, sizeof(double)); zeros(tmp_p2, p);
    double *tmp_n = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n, n);
    double *XtX = (double *) R_alloc(pp, sizeof(double));
    double *tau_sq_H = (double *) R_alloc(one, sizeof(double));

    double *tau_sq_I = (double *) R_alloc(one, sizeof(double));

    F77_NAME(dgemm)(ytran, ntran, &p, &p, &n, &one, X, &n, X, &n, &zero, XtX, &p FCONE FCONE);
    //double *w_mu = (double *) R_alloc(n, sizeof(double));
    //zeros(w_mu, n);
    if(initial_mu){
      //F77_NAME(dcopy)(&n, y, &inc, w_mu, &inc);
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
    int Trace_phi = 10;
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

    //updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
    updateBF2(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[phiIndx], nu, covModel, bk, nuUnifb);
    int nIndSqx = static_cast<int>(static_cast<double>(1+m)*(m+m+1)/6*m+(n-m-1)*m*m);
    double *F_inv = (double *) R_alloc(n, sizeof(double)); zeros(F_inv,n);
    double *B_over_F = (double *) R_alloc(nIndx, sizeof(double)); zeros(B_over_F,nIndx);
    double *Bmat_over_F = (double *) R_alloc(nIndSqx, sizeof(double)); zeros(Bmat_over_F,nIndSqx);
    int *nnIndxLUSq = (int *) R_alloc(2*n, sizeof(int));

    double *F_temp = (double *) R_alloc(n, sizeof(double)); zeros(F_temp,n);
    double *B_temp = (double *) R_alloc(nIndx, sizeof(double)); zeros(B_temp,nIndx);
    double *Bmat_over_F_temp = (double *) R_alloc(nIndSqx, sizeof(double)); zeros(Bmat_over_F_temp,nIndSqx);


    for(int i = n; i < 2*n; i++){
      nnIndxLUSq[i] = pow(nnIndxLU[i],2);
    }

    nnIndxLUSq[0] = 0;
    for(int i = 1; i < n; i++){
      nnIndxLUSq[i] = nnIndxLUSq[i-1] + nnIndxLUSq[n+i-1];
    }

    double F_inv_temp;
    for(i = 0; i < n; i++){
      F_inv[i] = 1/F[i];

      for(j = 0; j < nnIndxLU[n+i]; j++){
        B_over_F[nnIndxLU[i]+j] = B[nnIndxLU[i]+j]/F[i];
      }

      if(i > 0){
        F_inv_temp = 1/F[i];
        F77_NAME(dger)(&nnIndxLU[n+i], &nnIndxLU[n+i], &one, &B[nnIndxLU[i]], &inc, &B[nnIndxLU[i]], &inc, &Bmat_over_F[nnIndxLUSq[i]], &nnIndxLU[n+i]);
        F77_NAME(dscal)(&nnIndxLUSq[n+i], &F_inv_temp, &Bmat_over_F[nnIndxLUSq[i]], &inc);
      }

    }
    int *nnIndxwhich = (int *) R_alloc(nIndx, sizeof(int));; zeros_int(nnIndxwhich, nIndx);
    get_nnIndxwhich(nnIndxwhich, n, nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);



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
    double *w_mu_temp_dF = (double *) R_alloc(n, sizeof(double));
    double *u_vec_temp_dF = (double *) R_alloc(n, sizeof(double));
    double *gradient_mu_vec = (double *) R_alloc(n, sizeof(double));

    double *gradient_const = (double *) R_alloc(n, sizeof(double));
    double *gradient = (double *) R_alloc(n, sizeof(double));
    double *gamma_gradient_sum = (double *) R_alloc(n, sizeof(double));

    double *u_vec_temp = (double *) R_alloc(n, sizeof(double));
    double *u_vec_temp2 = (double *) R_alloc(n, sizeof(double));

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

    double *tmp_n_mb = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n_mb, n);
    double *diag_input_mb = (double *) R_alloc(n, sizeof(double)); zeros(diag_input_mb, n);

    int BatchSize;
    double sum_diags= 0.0;
    int i_mb;
    double *rademacher_rv_vec = (double *) R_alloc(n, sizeof(double));
    double *rademacher_rv_temp = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp,n);
    double *rademacher_rv_temp2 = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp2,n);

    double *product_v = (double *) R_alloc(n, sizeof(double));zeros(product_v,n);
    double *product_v2 = (double *) R_alloc(n, sizeof(double));zeros(product_v2,n);
    double *e_i = (double *) R_alloc(n, sizeof(double));zeros(e_i,n);

    int batch_index = 0;

    int max_result_size = nBatch * n;
    int max_temp_size = n;

    int* result_arr = (int *) R_alloc(max_result_size, sizeof(int));
    int* temp_arr = (int *) R_alloc(max_temp_size, sizeof(int));
    int result_index = 0;
    int temp_index = 0;

    int* tempsize_vec = (int *) R_alloc(nBatch, sizeof(int));

    // Usage:
    int *seen_values = (int *) R_alloc(n, sizeof(int));

    // Assuming max possible size for all results is n
    int *intersect_result = (int *) R_alloc(max_result_size, sizeof(int));
    int *complement_first_result = (int *) R_alloc(max_result_size, sizeof(int));
    int *complement_second_result = (int *) R_alloc(max_result_size, sizeof(int));

    // Allocate and initialize indices and sizes arrays
    int *intersect_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *intersect_sizes = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_first_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_first_sizes = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_second_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_second_sizes = (int *) R_alloc(nBatch, sizeof(int));

    // Initialize result indices
    int intersect_result_index = 0;
    int complement_first_result_index = 0;
    int complement_second_result_index = 0;

    for (int batch_index = 0; batch_index < nBatch; ++batch_index) {
      zeros_int(seen_values,n);
      BatchSize = nBatchIndx[batch_index];
      find_set_nngp(n, nnIndx, nnIndxLU, BatchSize, nBatchLU, batch_index,
                    seen_values,
                    intersect_result, intersect_sizes, intersect_start_indices,
                    complement_first_result, complement_first_sizes, complement_first_start_indices,
                    complement_second_result, complement_second_sizes, complement_second_start_indices,
                    intersect_result_index, complement_first_result_index, complement_second_result_index);

      zeros_int(seen_values,n);

      find_set_mb(n, nnIndx, nnIndxLU, nnIndxCol, numIndxCol, nnIndxnnCol, cumnumIndxCol,
                  BatchSize, nBatchLU, batch_index, result_arr, result_index, temp_arr, temp_index, tempsize_vec, seen_values);


    }

    int total_size_intersect = 0;
    int total_size_complement_first  = 0;
    int total_size_complement_second = 0;

    for (int i = 0; i < nBatch; ++i) {
      total_size_intersect         += intersect_sizes[i];
      total_size_complement_first  += complement_first_sizes[i];
      total_size_complement_second += complement_second_sizes[i];
    }

    if(verbose){

      for (int i = 0; i < nBatch; ++i) {
        Rprintf("intersect_sizes is %i ", intersect_sizes[i]);
        Rprintf("complement_first_sizes is %i ", complement_first_sizes[i]);
        Rprintf("complement_second_sizes is %i ", complement_second_sizes[i]);
      }
      Rprintf("\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }

    int* final_intersect_vec = (int *) R_alloc(total_size_intersect, sizeof(int));
    int* final_complement_1_vec = (int *) R_alloc(total_size_complement_first, sizeof(int));
    int* final_complement_2_vec = (int *) R_alloc(total_size_complement_second, sizeof(int));
    for(int i = 0; i < total_size_intersect; i++) {
      final_intersect_vec[i] = intersect_result[i];
    }
    for(int i = 0; i < total_size_complement_first; i++) {
      final_complement_1_vec[i] = complement_first_result[i];
    }
    for(int i = 0; i < total_size_complement_second; i++) {
      final_complement_2_vec[i] = complement_second_result[i];
    }


    int* final_result_vec = (int *) R_alloc(result_index, sizeof(int));
    for(int i = 0; i < result_index; i++) {
      final_result_vec[i] = result_arr[i];
    }



    if(verbose){
      Rprintf("tempsize_vec: ");
      for (int i = 0; i < nBatch; i++) {
        Rprintf("%i ", tempsize_vec[i]);
      }
      Rprintf("\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    int tempsize;
    int *nBatchLU_temp = (int *) R_alloc(nBatch, sizeof(int));

    nBatchLU_temp[0] = 0; // starting with the first value

    for(int i = 1; i < nBatch; i++) {
      nBatchLU_temp[i] = nBatchLU_temp[i-1] + tempsize_vec[i-1];
    }

    double *phi_can_vec = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    double *log_g_phi = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    double *sum_v = (double *) R_alloc(n, sizeof(double));zeros(sum_v,n);

    double *XtXmb = (double *) R_alloc(pp*nBatch, sizeof(double));
    double *XtX_temp = (double *) R_alloc(pp, sizeof(double));


    for (int batch_index = 0; batch_index < nBatch; ++batch_index){

      BatchSize = nBatchIndx[batch_index];
      int startRow = nBatchLU[batch_index]; // Calculate based on batch_index
      int endRow = nBatchLU[batch_index]+BatchSize;

      double *X_subset = (double *) R_alloc(BatchSize * p, sizeof(double));

      // Copy the data from X to X_subset
      for (int j = 0; j < p; ++j) { // Iterate over columns
        for (int i = startRow; i < endRow; ++i) { // Iterate over rows within the column
          X_subset[j * BatchSize + (i - startRow)] = X[j * n + i];
        }
      }

      // Use X_subset in dgemm
      F77_NAME(dgemm)(ytran, ntran, &p, &p, &BatchSize, &one, X_subset, &BatchSize, X_subset, &BatchSize, &zero, XtX_temp, &p FCONE FCONE);


      for(i = 0; i < pp; i++){
        XtXmb[batch_index*pp+i] = XtX_temp[i];
        //Rprintf("XtXmb is %f \n", XtXmb[batch_index*pp+i]);
      }
    }

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("Initialize Process \n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    if(initial_mu){
      double rho = 0.1;

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
        tmp_pp[i] = XtX[i]/theta[tauSqIndx];
      }

      for (i = 0; i < p; i++) {
        tmp_pp[i * (p + 1)] += 0.1;  // Add 0.1 to the diagonal elements
      }

      F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotrf failed\n");}
      F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 2 dpotri failed\n");}

      F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc FCONE);

      //F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info FCONE); if(info != 0){error("c++ error: 3 dpotrf failed\n");}

      for (i = 0; i < p; i++) {
        tmp_p2[i] = tmp_p2[i]/theta[tauSqIndx];  // Add 0.1 to the diagonal elements
      }

      F77_NAME(dcopy)(&p, tmp_p2, &inc, beta, &inc);

      for (int i = 0; i < p; i++) {
        for (int j = 0; j <= i; j++) {
          // Calculate the index for column-major format
          int idx = i + j * p;
          beta_cov[idx] = tmp_pp[idx] ;
        }
      }

      if(verbose){
        for(i = 0; i < p; i++){
          Rprintf("the value of beta[%i] : %f \n",i, beta[i]);
        }
        for(i = 0; i < p*p; i++){
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

      for(int i = 0; i < n; i++){
        epsilon_vec[i] = rnorm(0, 1);

      }
      update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);
      double trace_vec1_re = 0.0;
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
        trace_vec1_re += E_quadratic(u_vec, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);
      }




      //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
      b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;

      //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + *tau_sq_I)*0.5;

      tau_sq = b_tau_update/a_tau_update;
      theta[tauSqIndx] = tau_sq;

      ///////////////
      //update zetasq
      ///////////////

      double zeta_Q_re = E_quadratic(w_mu, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);
      b_zeta_update = zetaSqIGb + (trace_vec1_re/Trace_N + zeta_Q_re)*0.5;
      zeta_sq = b_zeta_update/a_zeta_update;
      theta[zetaSqIndx] = zeta_sq;

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

            // for(int k = 0; k < Trace_N; k++){
            //   phi_can_vec[i*N_phi+j] += rbeta(a_phi_vec[i], b_phi_vec[j]);  // Notice the indexing here
            // }

            updateBF_quadratic(B_temp, F_temp, Bmat_over_F_temp,
                               F_inv, B_over_F, Bmat_over_F,
                               nIndx, nIndSqx,
                               nnIndxLUSq,
                               Trace_phi,
                               c, C, coords, nnIndx, nnIndxLU,
                               n,  m,
                               nu,  covModel, bk,  nuUnifb,
                               a_phi_vec[i],  b_phi_vec[j],
                                                       phimax,  phimin);

            //phi_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
            phi_Q = E_quadratic(w_mu, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);

            update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);
            logDetInv = 0.0;
            diag_sigma_sq_sum = 0.0;
            for(int s = 0; s < n; s++){
              logDetInv += log(F_inv[s]);
            }

            log_g_phi[i*N_phi+j] = logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) -
              (phi_Q +  E_quadratic(u_vec, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq))*0.5/theta[zetaSqIndx];


            //Rprintf("log_g_phi is %f ", log_g_phi[i*N_phi+j]);
            // phi_can_vec[i*N_phi+j] /= Trace_N;
            // phi_can_vec[i*N_phi+j] = phi_can_vec[i*N_phi+j]*(phimax - phimin) + phimin;
          }
        }


        max_index = max_ind(log_g_phi,N_phi*N_phi);
        a_phi = a_phi_vec[max_index/N_phi];
        b_phi = b_phi_vec[max_index % N_phi];

        theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;

        updateBF_quadratic(B_temp, F_temp, Bmat_over_F_temp,
                           F_inv, B_over_F, Bmat_over_F,
                           nIndx, nIndSqx,
                           nnIndxLUSq,
                           Trace_N,
                           c, C, coords, nnIndx, nnIndxLU,
                           n,  m,
                           nu,  covModel, bk,  nuUnifb,
                           a_phi,  b_phi,  phimax,  phimin);
        // updateBF2(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
        //           theta[phiIndx], nu, covModel, bk, nuUnifb);

        // updateBF2(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[phiIndx], nu, covModel, bk, nuUnifb);

        // for(int i = 0; i < n; i++){
        //   Rprintf("B_over_F is ");
        //   for (int l = 0; l < nnIndxLU[n + i]; l++){
        //     Rprintf("%f ", B_over_F[nnIndxLU[i] + l]);
        //   }
        //   Rprintf("\n");
        //
        //   Rprintf("B/F is ");
        //   for (int l = 0; l < nnIndxLU[n + i]; l++){
        //     Rprintf("%f ", B[nnIndxLU[i] + l]/F[i]);
        //   }
        //   Rprintf("\n");
        //
        //
        //   Rprintf("F_inv is %f ", F_inv[i]);
        //   Rprintf("1/F is %f ", 1/F[i]);
        //
        //   Rprintf("B is ");
        //   for (int l = 0; l < nnIndxLU[n + i]; l++){
        //     Rprintf("%f ", B[nnIndxLU[i] + l]);
        //   }
        //   Rprintf("\n");
        //
        //   Rprintf("Bmat_over_F is ");
        //   for (int k = 0; k < nnIndxLUSq[n + i]; k++){
        //     Rprintf("%f ", Bmat_over_F[nnIndxLUSq[i] + k]);
        //   }
        //   Rprintf("\n");
        //
        // }

      }

      ///////////////
      //update w
      ///////////////

      double one_over_zeta_sq = 1.0/theta[zetaSqIndx];
      zeros(w_mu_temp,n);
      zeros(w_mu_temp2,n);
      product_B_F_combine(w_mu, w_mu_temp, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                          n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                          nnIndxwhich);

      F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);

      double gradient_mu = 0.0;
      //Rprintf("w_mu_update: ");
      for(i = 0; i < n; i++){
        gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i]- F77_NAME(ddot)(&p, &X[i], &n, beta, &inc))/theta[tauSqIndx]);
        E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
        delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
        delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
        w_mu_update[i] = w_mu[i] + delta_mu[i];
      }
      product_B_F_combine(w_mu_update, w_mu_temp, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                          n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                          nnIndxwhich);

      F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);

      zeros(gradient,n);
      zeros(gamma_gradient_sum, n);
      zeros(gamma_gradient,n);

      for(int k = 0; k < Trace_N; k++){
        zeros(gamma_gradient,n);
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }

        gamma_gradient_fun2(u_vec, epsilon_vec, gamma_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                            nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                            cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,w_mu_temp,w_mu_temp2,gradient,zetaSqIndx,
                            F_inv, B_over_F, Bmat_over_F, nnIndxLUSq, nnIndxwhich);

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


        a_gradient_fun2(u_vec, epsilon_vec, a_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                        nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                        w_mu_temp,w_mu_temp2,zetaSqIndx, F_inv, B_over_F, Bmat_over_F, nnIndxLUSq, nnIndxwhich);
        //
        // for(int i = 0; i < nIndx_vi; i++){
        //   Rprintf("\tError is %i, %f \n",i, a_gradient[i]);
        // }
        vecsum(a_gradient_sum, a_gradient, Trace_N, nIndx_vi);
        // for(int i = 0; i < nIndx_vi; i++){
        //   a_gradient_sum[i] = a_gradient[i];
        // }
      }
      for(i = 0; i < nIndx_vi; i++){
        E_a_sq[i] = rho * E_a_sq[i] + (1 - rho) * pow(a_gradient_sum[i],2);
        delta_a[i] = sqrt(delta_a_sq[i]+adadelta_noise)/sqrt(E_a_sq[i]+adadelta_noise)*a_gradient_sum[i];
        delta_a_sq[i] = rho*delta_a_sq[i] + (1 - rho) * pow(delta_a[i],2);
        //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
        A_vi[i] = A_vi[i] + delta_a[i];
        //Rprintf("%f ",A_vi[i]);
      }

      F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);

    }

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("Updating Process \n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    while(iter <= max_iter & !indicator_converge){
      if(verbose){
        Rprintf("----------------------------------------\n");
        Rprintf("\tIteration at %i \n",iter);
#ifdef Win32
        R_FlushConsole();
#endif
      }
      zeros(gradient_const,n);
      zeros(gradient,n);
      zeros(gamma_gradient_sum, n);
      zeros(gamma_gradient,n);
      zeros(a_gradient,nIndx_vi);
      zeros(a_gradient_sum, nIndx_vi);
      double rho = rho_input;
      // if(iter < 100){
      //   rho = 0.95;
      // }else{
      //   rho = rho_input;
      // }
      for(int batch_index = 0; batch_index < nBatch; batch_index++){
        tempsize = tempsize_vec[batch_index];
        BatchSize = nBatchIndx[batch_index];
        for(i_mb = 0; i_mb < tempsize; i_mb++){
          epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
        }
        //BatchSize = nBatchIndx[batch_index];

        if(batch_index == iter % nBatch){
          if(verbose){
            Rprintf("the value of batch_index global : %i \n", batch_index);
#ifdef Win32
            R_FlushConsole();
#endif
          }
          a_tau_update = BatchSize * 0.5 + tauSqIGa;
          a_zeta_update = BatchSize * 0.5 + zetaSqIGa;

          ///////////////
          //update beta
          ///////////////

          zeros(tau_sq_I, one_int);
          zeros(tmp_n_mb, n);

          for(i = 0; i < BatchSize; i++){
            tmp_n_mb[nBatchLU[batch_index] + i] = y[nBatchLU[batch_index] + i]-w_mu[nBatchLU[batch_index] + i];
            tau_sq_I[0] += pow(tmp_n_mb[nBatchLU[batch_index] + i],2);
          }

          F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, tmp_n_mb, &inc, &zero, tmp_p, &inc FCONE);

          for(i = 0; i < pp; i++){
            tmp_pp[i] = XtXmb[batch_index*pp+i];
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
            for(i = 0; i < p*p; i++){
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
          double trace_vec1_re = 0.0;

          for(i_mb = 0; i_mb < tempsize; i_mb++){
            epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
          }


          update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                     batch_index, final_result_vec, nBatchLU_temp, tempsize);

          for(int k = 0; k < Trace_N; k++){
            for(i_mb = 0; i_mb < tempsize; i_mb++){
              epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
            }
            update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                       batch_index, final_result_vec, nBatchLU_temp, tempsize);

            double u_mean = 0.0;
            for(i_mb = 0; i_mb < BatchSize; i_mb++){
              u_mean += u_vec[nBatchLU[batch_index] + i_mb];
            }
            u_mean = u_mean/BatchSize;
            for(i_mb = 0; i_mb < BatchSize; i_mb++){
              trace_vec[0] += pow(u_vec[nBatchLU[batch_index] + i_mb]-u_mean,2);
            }

            //trace_vec[1] += Q_mini_batch_plus(B, F, u_vec, u_vec, batch_index, n, nnIndx, nnIndxLU, final_result_vec, nBatchLU_temp, tempsize);
            //trace_vec[1] += Q_mini_batch(B, F, u_vec, u_vec, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU);
            trace_vec1_re += E_quadratic_mb(u_vec, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                            n, nnIndx, nnIndxLU, nnIndxLUSq);

          }

          //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
          if (!isnan(trace_vec[0])){
            //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + *tau_sq_I)*0.5;
            //Rprintf("add_tau is : %f \n",trace_vec[0]/Trace_N + *tau_sq_I);
            b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;

            tau_sq = b_tau_update/a_tau_update;
            theta[tauSqIndx] = tau_sq;
          }else{
            theta[tauSqIndx] = 1;
          }



          if(verbose){
            Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
            R_FlushConsole();
#endif
          }

          ///////////////
          //update zetasq
          ///////////////
          // updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //                         theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
          //                         batch_index, final_result_vec, nBatchLU_temp, tempsize);
          //double zeta_Q_mb = Q_mini_batch(B, F, w_mu, w_mu, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU);
          double zeta_Q_mb_re = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                               n, nnIndx, nnIndxLU, nnIndxLUSq);


          //Rprintf("zeta_Q_mb: %f \n", zeta_Q_mb);
          if (!isnan(trace_vec[1])){
            b_zeta_update = zetaSqIGb + (trace_vec1_re/Trace_N + zeta_Q_mb_re)*0.5;

            //Rprintf("add zeta is : %f \n",trace_vec[1]/Trace_N);
            zeta_sq = b_zeta_update/a_zeta_update;
            //zeta_sq = 13.227241;
            if(zeta_sq > 1000){zeta_sq = 1000;}
            theta[zetaSqIndx] = zeta_sq;
          }else{
            theta[zetaSqIndx] = 1;
          }



          if(verbose){
            Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
            R_FlushConsole();
#endif
          }

          // updateBF_minibatch(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //                    theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
          //                    BatchSize, nBatchLU, batch_index);
          // updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //                         theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
          //                         batch_index, final_result_vec, nBatchLU_temp, tempsize);

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

                updateBF_quadratic_mb(B_temp, F_temp, Bmat_over_F_temp,
                                      F_inv, B_over_F, Bmat_over_F,
                                      nIndx, nIndSqx,
                                      nnIndxLUSq,
                                      Trace_phi,
                                      c, C, coords, nnIndx, nnIndxLU,
                                      BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                                      n,  m,
                                      nu,  covModel, bk,  nuUnifb,
                                      a_phi_vec[i],  b_phi_vec[j], phimax,  phimin);

                update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                           batch_index, final_result_vec, nBatchLU_temp, tempsize);

                phi_Q = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                       n, nnIndx, nnIndxLU, nnIndxLUSq);

                logDetInv = 0.0;
                diag_sigma_sq_sum = 0.0;
                for(i_mb = 0; i_mb < BatchSize; i_mb++){
                  s = nBatchLU[batch_index] + i_mb;
                  logDetInv += log(F_inv[s]);
                }

                log_g_phi[i*N_phi+j] = logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) -
                  (phi_Q +  E_quadratic_mb(u_vec, F_inv, B_over_F, Bmat_over_F,  BatchSize, nBatchLU, batch_index,
                                           n, nnIndx, nnIndxLU, nnIndxLUSq))*0.5/theta[zetaSqIndx];


              }
            }

            max_index = max_ind(log_g_phi,N_phi*N_phi);
            a_phi = a_phi_vec[max_index/N_phi];
            b_phi = b_phi_vec[max_index % N_phi];

            theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;

            // updateBF_minibatch(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
            //                    theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb,
            //                    BatchSize, nBatchLU, batch_index);
            updateBF_quadratic(B_temp, F_temp, Bmat_over_F_temp,
                               F_inv, B_over_F, Bmat_over_F,
                               nIndx, nIndSqx,
                               nnIndxLUSq,
                               Trace_phi,
                               c, C, coords, nnIndx, nnIndxLU,
                               n,  m,
                               nu,  covModel, bk,  nuUnifb,
                               a_phi,  b_phi,  phimax,  phimin);
          }

          if(verbose){
            Rprintf("the value of theta[%i phiIndx] : %f \n", phiIndx, theta[phiIndx]);
#ifdef Win32
            R_FlushConsole();
#endif
          }
        }
        //for(int batch_index = 0; batch_index < nBatch; batch_index++)
        if(verbose){
          Rprintf("the value of batch_index for w : %i \n", batch_index);
#ifdef Win32
          R_FlushConsole();
#endif
        }
        tempsize = tempsize_vec[batch_index];
        BatchSize = nBatchIndx[batch_index];

        ///////////////
        //update w
        ///////////////

        //zeros_minibatch(w_mu_temp,n, BatchSize, nBatchLU, batch_index);
        //zeros_minibatch(w_mu_temp2,n, BatchSize, nBatchLU, batch_index);
        double gradient_mu;
        double one_over_zeta_sq = 1.0/theta[zetaSqIndx];
        zeros(w_mu_temp,n);
        zeros(w_mu_temp_dF,n);
        zeros(w_mu_temp2,n);
        product_B_F_combine_mb(w_mu, w_mu_temp, w_mu_temp_dF, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                               BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                               n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                               nnIndxwhich);

        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp, &inc);
        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp_dF, &inc);
        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);

        for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
          i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
          gradient_mu_vec[i] =  (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc) - w_mu[i])/theta[tauSqIndx] - w_mu_temp2[i];
        }
        for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
          i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
          //gradient_mu_vec[i] =  (y[i] - w_mu[i])/theta[tauSqIndx] - w_mu_temp_dF[i];
          gradient_mu_vec[i] =  (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc) - w_mu[i])/theta[tauSqIndx] - w_mu_temp[i];

        }


        for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
          i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
          //gradient_mu_vec[i] = - w_mu_temp2[i] + w_mu_temp_dF[i];
          gradient_mu_vec[i] = w_mu_temp_dF[i];
        }

        for(i_mb = 0; i_mb < tempsize; i_mb++){
          i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
          //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
          gradient_mu = gradient_mu_vec[i];
          E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
          delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
          delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
          w_mu_update[i] = w_mu[i] + delta_mu[i];
        }

        zeros(w_mu_temp,n);
        zeros(w_mu_temp_dF,n);
        zeros(w_mu_temp2,n);
        // product_B_F_minibatch_plus(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
        // product_B_F_minibatch_term1(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);
        // product_B_F_vec_minibatch_plus_fix(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);

        product_B_F_combine_mb(w_mu, w_mu_temp, w_mu_temp_dF, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                               BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                               n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                               nnIndxwhich);

        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp, &inc);
        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp_dF, &inc);
        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);


        zeros(gamma_gradient_sum, n);
        for(int k = 0; k < Trace_N; k++){
          //zeros_minibatch_plus(gamma_gradient,n, batch_index,final_result_vec, nBatchLU_temp, tempsize);
          zeros(gradient,n);
          zeros(gamma_gradient,n);
          for(i_mb = 0; i_mb < tempsize; i_mb++){
            epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
          }

          gamma_gradient_mb_fun2(u_vec, epsilon_vec, gamma_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                 nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                                 cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,
                                 u_vec_temp, u_vec_temp2, u_vec_temp_dF, gradient, zetaSqIndx,
                                 F_inv, B_over_F, Bmat_over_F,
                                 nnIndxLUSq, nnIndxwhich,
                                 batch_index, BatchSize, nBatchLU,
                                 final_result_vec, nBatchLU_temp, tempsize,
                                 intersect_start_indices, intersect_sizes, final_intersect_vec,
                                 complement_first_start_indices, complement_first_sizes, final_complement_1_vec,
                                 complement_second_start_indices, complement_second_sizes, final_complement_2_vec);

          vecsum_minibatch_plus(gamma_gradient_sum, gamma_gradient, Trace_N, n, batch_index, final_result_vec, nBatchLU_temp, tempsize);

        }

        for(i_mb = 0; i_mb < tempsize; i_mb++){
          i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
          //Rprintf("gamma gradient[%i],: %f \n",i, gamma_gradient_sum[i]);
          E_gamma_sq[i] = rho * E_gamma_sq[i] + (1 - rho) * pow(gamma_gradient_sum[i],2);
          delta_gamma[i] = sqrt(delta_gamma_sq[i]+adadelta_noise)/sqrt(E_gamma_sq[i]+adadelta_noise)*gamma_gradient_sum[i];
          delta_gamma_sq[i] = rho*delta_gamma_sq[i] + (1 - rho) * pow(delta_gamma[i],2);
          S_vi[i] = pow(exp(log(sqrt(S_vi[i])) + delta_gamma[i]),2);
        }

        zeros(a_gradient_sum, nIndx_vi);
        for(int k = 0; k < Trace_N; k++){
          zeros(gradient,n);
          zeros(a_gradient,nIndx_vi);
          zeros(w_mu_temp,n);
          for(i_mb = 0; i_mb < tempsize; i_mb++){
            epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
          }

          a_gradient_mb_fun2(u_vec, epsilon_vec, a_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                             cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,
                             u_vec_temp, u_vec_temp2, u_vec_temp_dF, gradient, zetaSqIndx,
                             F_inv, B_over_F, Bmat_over_F,
                             nnIndxLUSq, nnIndxwhich,
                             batch_index, BatchSize, nBatchLU,
                             final_result_vec, nBatchLU_temp, tempsize,
                             intersect_start_indices, intersect_sizes, final_intersect_vec,
                             complement_first_start_indices, complement_first_sizes, final_complement_1_vec,
                             complement_second_start_indices, complement_second_sizes, final_complement_2_vec);

          //vecsum(a_gradient_sum, a_gradient, Trace_N, nIndx_vi);
          for(int i_mb = 0; i_mb < tempsize; i_mb++){
            i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
            for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
              a_gradient_sum[nnIndxLU_vi[i] + l] += a_gradient[nnIndxLU_vi[i] + l]/Trace_N;
            }
          }


        }
        int sub_index;
        //Rprintf("A_vi: ");
        for(int i_mb = 0; i_mb < tempsize; i_mb++){
          i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
          for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
            sub_index = nnIndxLU_vi[i] + l;
            //Rprintf("a gradient[%i],: %f \n",sub_index, a_gradient_sum[sub_index]);
            //a_gradient_sum[nnIndxLU_vi[i] + l] += a_gradient[nnIndxLU_vi[i] + l]/Trace_N;
            E_a_sq[sub_index] = rho * E_a_sq[sub_index] + (1 - rho) * pow(a_gradient_sum[sub_index],2);
            delta_a[sub_index] = sqrt(delta_a_sq[sub_index]+adadelta_noise)/sqrt(E_a_sq[sub_index]+adadelta_noise)*a_gradient_sum[sub_index];
            delta_a_sq[sub_index] = rho*delta_a_sq[sub_index] + (1 - rho) * pow(delta_a[sub_index],2);
            A_vi[sub_index] = A_vi[sub_index] + delta_a[sub_index];
            //Rprintf("A_vi[i]: %i, %f \n",sub_index, A_vi[sub_index]);
            //Rprintf("\t Updated a index is %i \n",sub_index);
          }
        }
        //Rprintf("\n");
        F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);

      }

      // Rprintf("\t rho at %i is %f\n",iter,rho);
      ELBO = 0.0;
      zeros(sum_v,n);
      double sum1 = 0.0;
      double sum2 = 0.0;
      double sum3 = 0.0;
      double sum4 = 0.0;
      double sum5 = 0.0;
      double sum6 = 0.0;

      for(int k = 0; k < Trace_N; k++){
        for(int i = 0; i < n; i++){
          epsilon_vec[i] = rnorm(0, 1);
        }
        update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);

        sum2 += sumsq(u_vec, n);

        sum4 += E_quadratic(u_vec, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);
      }

      for(int i = 0; i < n; i++){
        sum1 += pow((y[i] -F77_NAME(ddot)(&p, &X[i], &n, beta, &inc) - w_mu_update[i]),2);
        sum5 += log(2*pi*S_vi[i]);
        sum6 += log(2*pi*F_inv[i]/theta[zetaSqIndx]);
      }

      sum1 = sum1/theta[tauSqIndx]*0.5;
      sum2 = sum2/Trace_N/theta[tauSqIndx]*0.5;
      sum3 = E_quadratic(w_mu_update, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq)/theta[zetaSqIndx]*0.5;
      sum4 = sum4/Trace_N/theta[zetaSqIndx]*0.5;

      ELBO = -sum1 - sum2 - sum3 - sum4 + sum5*0.5 + sum6*0.5 + 0.5*n - 0.5*n*log(2*pi*theta[tauSqIndx]);

      //Rprintf("the value of ELBO: %f \n", ELBO);
      ELBO_vec[iter-1] = ELBO;


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


      F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);


    }

    //
    //updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
    updateBF2(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[phiIndx], nu, covModel, bk, nuUnifb);
    //zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
    SEXP theta_para_r; PROTECT(theta_para_r = allocVector(REALSXP, nTheta*2)); nProtect++; double *theta_para = REAL(theta_para_r);

    theta_para[zetaSqIndx*2+0] = a_zeta_update;
    theta_para[zetaSqIndx*2+1] = b_zeta_update;

    theta_para[tauSqIndx*2+0] = a_tau_update;
    theta_para[tauSqIndx*2+1] = b_tau_update;

    theta_para[phiIndx*2+0] = a_phi;
    theta_para[phiIndx*2+1] = b_phi;


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

    namesgets(result_r, resultName_r);
    //unprotect
    UNPROTECT(nProtect);


    return(result_r);

  }

  SEXP spVarBayes_NNGP_nocovariates_mb_beta_rephicpp(SEXP y_r,
                                               SEXP n_r, SEXP p_r, SEXP m_r, SEXP m_vi_r, SEXP coords_r, SEXP covModel_r, SEXP rho_r,
                                               SEXP zetaSqIG_r, SEXP tauSqIG_r, SEXP phibeta_r, SEXP nuUnif_r,
                                               SEXP zetaSqStarting_r, SEXP tauSqStarting_r, SEXP phiStarting_r, SEXP nuStarting_r,
                                               SEXP sType_r, SEXP nThreads_r, SEXP verbose_r, SEXP fix_nugget_r, SEXP N_phi_r, SEXP Trace_N_r,
                                               SEXP max_iter_r,
                                               SEXP var_input_r,
                                               SEXP phi_input_r, SEXP phi_iter_max_r, SEXP initial_mu_r,
                                               SEXP mini_batch_size_r,
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
    double phi_input  =  REAL(phi_input_r)[0];
    double *var_input  =  REAL(var_input_r);
    int initial_mu  =  INTEGER(initial_mu_r)[0];
    int phi_iter_max = INTEGER(phi_iter_max_r)[0];
    int max_iter = INTEGER(max_iter_r)[0];
    int phi_start = max_iter - phi_iter_max;
    int n_mb = INTEGER(mini_batch_size_r)[0];

    int K = INTEGER(K_r)[0];
    int stop_K = INTEGER(stop_K_r)[0];
    int min_iter = INTEGER(min_iter_r)[0];

    int nThreads = INTEGER(nThreads_r)[0];
    int verbose = INTEGER(verbose_r)[0];
    //double  vi_threshold  =  REAL(vi_threshold_r)[0];
    double  rho_input  =  REAL(rho_r)[0];
    //double  rho_phi  =  REAL(rho_phi_r)[0];
    //priors
    double zetaSqIGa = REAL(zetaSqIG_r)[0]; double zetaSqIGb = REAL(zetaSqIG_r)[1];
    double tauSqIGa = REAL(tauSqIG_r)[0]; double tauSqIGb = REAL(tauSqIG_r)[1];
    //double phiUnifa = REAL(phiUnif_r)[0]; double phiUnifb = REAL(phiUnif_r)[1];
    double phimin = REAL(phibeta_r)[0]; double phimax = REAL(phibeta_r)[1];

    double a_phi = (phi_input - phimin)/(phimax-phimin)*50;
    double b_phi = 50 - a_phi;

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
    int nBatch = static_cast<int>(std::ceil(static_cast<double>(n)/static_cast<double>(n_mb)));
    int *nBatchIndx = (int *) R_alloc(nBatch, sizeof(int));
    int *nBatchLU = (int *) R_alloc(nBatch, sizeof(int));
    get_nBatchIndx(n, nBatch, n_mb, nBatchIndx, nBatchLU);
    if(verbose){
      Rprintf("Using %i nBatch \n", nBatch);

      for(int i = 0; i < nBatch; i++){
        Rprintf("the value of nBatchIndx[%i] : %i \n",i, nBatchIndx[i]);
        Rprintf("the value of nBatchLU[%i] : %i \n",i, nBatchLU[i]);
      }
#ifdef Win32
      R_FlushConsole();
#endif
    }



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
    int Trace_phi = 10;
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

    //updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
    updateBF2(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[phiIndx], nu, covModel, bk, nuUnifb);
    int nIndSqx = static_cast<int>(static_cast<double>(1+m)*(m+m+1)/6*m+(n-m-1)*m*m);
    double *F_inv = (double *) R_alloc(n, sizeof(double)); zeros(F_inv,n);
    double *B_over_F = (double *) R_alloc(nIndx, sizeof(double)); zeros(B_over_F,nIndx);
    double *Bmat_over_F = (double *) R_alloc(nIndSqx, sizeof(double)); zeros(Bmat_over_F,nIndSqx);
    int *nnIndxLUSq = (int *) R_alloc(2*n, sizeof(int));

    double *F_temp = (double *) R_alloc(n, sizeof(double)); zeros(F_temp,n);
    double *B_temp = (double *) R_alloc(nIndx, sizeof(double)); zeros(B_temp,nIndx);
    double *Bmat_over_F_temp = (double *) R_alloc(nIndSqx, sizeof(double)); zeros(Bmat_over_F_temp,nIndSqx);


    for(int i = n; i < 2*n; i++){
      nnIndxLUSq[i] = pow(nnIndxLU[i],2);
    }

    nnIndxLUSq[0] = 0;
    for(int i = 1; i < n; i++){
      nnIndxLUSq[i] = nnIndxLUSq[i-1] + nnIndxLUSq[n+i-1];
    }

    double F_inv_temp;
    for(i = 0; i < n; i++){
      F_inv[i] = 1/F[i];

      for(j = 0; j < nnIndxLU[n+i]; j++){
        B_over_F[nnIndxLU[i]+j] = B[nnIndxLU[i]+j]/F[i];
      }

      if(i > 0){
        F_inv_temp = 1/F[i];
        F77_NAME(dger)(&nnIndxLU[n+i], &nnIndxLU[n+i], &one, &B[nnIndxLU[i]], &inc, &B[nnIndxLU[i]], &inc, &Bmat_over_F[nnIndxLUSq[i]], &nnIndxLU[n+i]);
        F77_NAME(dscal)(&nnIndxLUSq[n+i], &F_inv_temp, &Bmat_over_F[nnIndxLUSq[i]], &inc);
      }

    }
    int *nnIndxwhich = (int *) R_alloc(nIndx, sizeof(int));; zeros_int(nnIndxwhich, nIndx);
    get_nnIndxwhich(nnIndxwhich, n, nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);



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
    double *w_mu_temp_dF = (double *) R_alloc(n, sizeof(double));
    double *u_vec_temp_dF = (double *) R_alloc(n, sizeof(double));
    double *gradient_mu_vec = (double *) R_alloc(n, sizeof(double));

    double *gradient_const = (double *) R_alloc(n, sizeof(double));
    double *gradient = (double *) R_alloc(n, sizeof(double));
    double *gamma_gradient_sum = (double *) R_alloc(n, sizeof(double));

    double *u_vec_temp = (double *) R_alloc(n, sizeof(double));
    double *u_vec_temp2 = (double *) R_alloc(n, sizeof(double));

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

    double *tmp_n_mb = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n_mb, n);
    double *diag_input_mb = (double *) R_alloc(n, sizeof(double)); zeros(diag_input_mb, n);

    int BatchSize;
    double sum_diags= 0.0;
    int i_mb;
    double *rademacher_rv_vec = (double *) R_alloc(n, sizeof(double));
    double *rademacher_rv_temp = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp,n);
    double *rademacher_rv_temp2 = (double *) R_alloc(n, sizeof(double));zeros(rademacher_rv_temp2,n);

    double *product_v = (double *) R_alloc(n, sizeof(double));zeros(product_v,n);
    double *product_v2 = (double *) R_alloc(n, sizeof(double));zeros(product_v2,n);
    double *e_i = (double *) R_alloc(n, sizeof(double));zeros(e_i,n);

    int batch_index = 0;

    int max_result_size = nBatch * n;
    int max_temp_size = n;

    int* result_arr = (int *) R_alloc(max_result_size, sizeof(int));
    int* temp_arr = (int *) R_alloc(max_temp_size, sizeof(int));
    int result_index = 0;
    int temp_index = 0;

    int* tempsize_vec = (int *) R_alloc(nBatch, sizeof(int));

    // Usage:
    int *seen_values = (int *) R_alloc(n, sizeof(int));

    // Assuming max possible size for all results is n
    int *intersect_result = (int *) R_alloc(max_result_size, sizeof(int));
    int *complement_first_result = (int *) R_alloc(max_result_size, sizeof(int));
    int *complement_second_result = (int *) R_alloc(max_result_size, sizeof(int));

    // Allocate and initialize indices and sizes arrays
    int *intersect_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *intersect_sizes = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_first_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_first_sizes = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_second_start_indices = (int *) R_alloc(nBatch, sizeof(int));
    int *complement_second_sizes = (int *) R_alloc(nBatch, sizeof(int));

    // Initialize result indices
    int intersect_result_index = 0;
    int complement_first_result_index = 0;
    int complement_second_result_index = 0;

    for (int batch_index = 0; batch_index < nBatch; ++batch_index) {
      zeros_int(seen_values,n);
      BatchSize = nBatchIndx[batch_index];
      find_set_nngp(n, nnIndx, nnIndxLU, BatchSize, nBatchLU, batch_index,
                    seen_values,
                    intersect_result, intersect_sizes, intersect_start_indices,
                    complement_first_result, complement_first_sizes, complement_first_start_indices,
                    complement_second_result, complement_second_sizes, complement_second_start_indices,
                    intersect_result_index, complement_first_result_index, complement_second_result_index);

      zeros_int(seen_values,n);

      find_set_mb(n, nnIndx, nnIndxLU, nnIndxCol, numIndxCol, nnIndxnnCol, cumnumIndxCol,
                  BatchSize, nBatchLU, batch_index, result_arr, result_index, temp_arr, temp_index, tempsize_vec, seen_values);


    }

    int total_size_intersect = 0;
    int total_size_complement_first  = 0;
    int total_size_complement_second = 0;

    for (int i = 0; i < nBatch; ++i) {
      total_size_intersect         += intersect_sizes[i];
      total_size_complement_first  += complement_first_sizes[i];
      total_size_complement_second += complement_second_sizes[i];
    }

    if(verbose){

      for (int i = 0; i < nBatch; ++i) {
        Rprintf("intersect_sizes is %i ", intersect_sizes[i]);
        Rprintf("complement_first_sizes is %i ", complement_first_sizes[i]);
        Rprintf("complement_second_sizes is %i ", complement_second_sizes[i]);
      }
      Rprintf("\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }

    int* final_intersect_vec = (int *) R_alloc(total_size_intersect, sizeof(int));
    int* final_complement_1_vec = (int *) R_alloc(total_size_complement_first, sizeof(int));
    int* final_complement_2_vec = (int *) R_alloc(total_size_complement_second, sizeof(int));
    for(int i = 0; i < total_size_intersect; i++) {
      final_intersect_vec[i] = intersect_result[i];
    }
    for(int i = 0; i < total_size_complement_first; i++) {
      final_complement_1_vec[i] = complement_first_result[i];
    }
    for(int i = 0; i < total_size_complement_second; i++) {
      final_complement_2_vec[i] = complement_second_result[i];
    }


    int* final_result_vec = (int *) R_alloc(result_index, sizeof(int));
    for(int i = 0; i < result_index; i++) {
      final_result_vec[i] = result_arr[i];
    }



    if(verbose){
      Rprintf("tempsize_vec: ");
      for (int i = 0; i < nBatch; i++) {
        Rprintf("%i ", tempsize_vec[i]);
      }
      Rprintf("\n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    int tempsize;
    int *nBatchLU_temp = (int *) R_alloc(nBatch, sizeof(int));

    nBatchLU_temp[0] = 0; // starting with the first value

    for(int i = 1; i < nBatch; i++) {
      nBatchLU_temp[i] = nBatchLU_temp[i-1] + tempsize_vec[i-1];
    }

    double *phi_can_vec = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    double *log_g_phi = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    double *sum_v = (double *) R_alloc(n, sizeof(double));zeros(sum_v,n);
    double grad_a = 0.0, grad_b = 0.0;
    double Eg2_a = 0.0, Eg2_b = 0.0;  // Running average of squared gradients
    double Ex2_a = 0.0, Ex2_b = 0.0;  // Running average of squared parameter updates


    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("Initialize Process \n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    if(initial_mu){
      double rho = 0.1;
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
        double trace_vec1_re = 0.0;
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
          trace_vec1_re += E_quadratic(u_vec, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);
        }




        //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
        b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + *tau_sq_I)*0.5;

        tau_sq = b_tau_update/a_tau_update;
        theta[tauSqIndx] = tau_sq;

        ///////////////
        //update zetasq
        ///////////////

        double zeta_Q_re = E_quadratic(w_mu, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);
        b_zeta_update = zetaSqIGb + (trace_vec1_re/Trace_N + zeta_Q_re)*0.5;
        zeta_sq = b_zeta_update/a_zeta_update;
        theta[zetaSqIndx] = zeta_sq;

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

              // for(int k = 0; k < Trace_N; k++){
              //   phi_can_vec[i*N_phi+j] += rbeta(a_phi_vec[i], b_phi_vec[j]);  // Notice the indexing here
              // }

              updateBF_quadratic(B_temp, F_temp, Bmat_over_F_temp,
                                 F_inv, B_over_F, Bmat_over_F,
                                 nIndx, nIndSqx,
                                 nnIndxLUSq,
                                 Trace_phi,
                                 c, C, coords, nnIndx, nnIndxLU,
                                 n,  m,
                                 nu,  covModel, bk,  nuUnifb,
                                 a_phi_vec[i],  b_phi_vec[j],
                                                         phimax,  phimin);

              //phi_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
              phi_Q = E_quadratic(w_mu, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);

              update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);
              logDetInv = 0.0;
              diag_sigma_sq_sum = 0.0;
              for(int s = 0; s < n; s++){
                logDetInv += log(F_inv[s]);
              }

              log_g_phi[i*N_phi+j] = logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) -
                (phi_Q +  E_quadratic(u_vec, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq))*0.5/theta[zetaSqIndx];


              //Rprintf("log_g_phi is %f ", log_g_phi[i*N_phi+j]);
              // phi_can_vec[i*N_phi+j] /= Trace_N;
              // phi_can_vec[i*N_phi+j] = phi_can_vec[i*N_phi+j]*(phimax - phimin) + phimin;
            }
          }


          max_index = max_ind(log_g_phi,N_phi*N_phi);
          a_phi = a_phi_vec[max_index/N_phi];
          b_phi = b_phi_vec[max_index % N_phi];

          theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;

          updateBF_quadratic(B_temp, F_temp, Bmat_over_F_temp,
                             F_inv, B_over_F, Bmat_over_F,
                             nIndx, nIndSqx,
                             nnIndxLUSq,
                             Trace_N,
                             c, C, coords, nnIndx, nnIndxLU,
                             n,  m,
                             nu,  covModel, bk,  nuUnifb,
                             a_phi,  b_phi,  phimax,  phimin);
          // updateBF2(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //           theta[phiIndx], nu, covModel, bk, nuUnifb);

          // updateBF2(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[phiIndx], nu, covModel, bk, nuUnifb);

          // for(int i = 0; i < n; i++){
          //   Rprintf("B_over_F is ");
          //   for (int l = 0; l < nnIndxLU[n + i]; l++){
          //     Rprintf("%f ", B_over_F[nnIndxLU[i] + l]);
          //   }
          //   Rprintf("\n");
          //
          //   Rprintf("B/F is ");
          //   for (int l = 0; l < nnIndxLU[n + i]; l++){
          //     Rprintf("%f ", B[nnIndxLU[i] + l]/F[i]);
          //   }
          //   Rprintf("\n");
          //
          //
          //   Rprintf("F_inv is %f ", F_inv[i]);
          //   Rprintf("1/F is %f ", 1/F[i]);
          //
          //   Rprintf("B is ");
          //   for (int l = 0; l < nnIndxLU[n + i]; l++){
          //     Rprintf("%f ", B[nnIndxLU[i] + l]);
          //   }
          //   Rprintf("\n");
          //
          //   Rprintf("Bmat_over_F is ");
          //   for (int k = 0; k < nnIndxLUSq[n + i]; k++){
          //     Rprintf("%f ", Bmat_over_F[nnIndxLUSq[i] + k]);
          //   }
          //   Rprintf("\n");
          //
          // }

        }

        ///////////////
        //update w
        ///////////////

        double one_over_zeta_sq = 1.0/theta[zetaSqIndx];
        zeros(w_mu_temp,n);
        zeros(w_mu_temp2,n);
        product_B_F_combine(w_mu, w_mu_temp, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                            n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                            nnIndxwhich);

        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);

        double gradient_mu = 0.0;
        //Rprintf("w_mu_update: ");
        for(i = 0; i < n; i++){
          gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
          E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
          delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
          delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
          w_mu_update[i] = w_mu[i] + delta_mu[i];
        }
        product_B_F_combine(w_mu_update, w_mu_temp, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                            n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                            nnIndxwhich);

        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);

        zeros(gradient,n);
        zeros(gamma_gradient_sum, n);
        zeros(gamma_gradient,n);

        for(int k = 0; k < Trace_N; k++){
          zeros(gamma_gradient,n);
          for(int i = 0; i < n; i++){
            epsilon_vec[i] = rnorm(0, 1);
          }

          gamma_gradient_fun2(u_vec, epsilon_vec, gamma_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                              nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                              cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,w_mu_temp,w_mu_temp2,gradient,zetaSqIndx,
                              F_inv, B_over_F, Bmat_over_F, nnIndxLUSq, nnIndxwhich);

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


          a_gradient_fun2(u_vec, epsilon_vec, a_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                          nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                          w_mu_temp,w_mu_temp2,zetaSqIndx, F_inv, B_over_F, Bmat_over_F, nnIndxLUSq, nnIndxwhich);
          //
          // for(int i = 0; i < nIndx_vi; i++){
          //   Rprintf("\tError is %i, %f \n",i, a_gradient[i]);
          // }
          vecsum(a_gradient_sum, a_gradient, Trace_N, nIndx_vi);
          // for(int i = 0; i < nIndx_vi; i++){
          //   a_gradient_sum[i] = a_gradient[i];
          // }
        }
        for(i = 0; i < nIndx_vi; i++){
          E_a_sq[i] = rho * E_a_sq[i] + (1 - rho) * pow(a_gradient_sum[i],2);
          delta_a[i] = sqrt(delta_a_sq[i]+adadelta_noise)/sqrt(E_a_sq[i]+adadelta_noise)*a_gradient_sum[i];
          delta_a_sq[i] = rho*delta_a_sq[i] + (1 - rho) * pow(delta_a[i],2);
          //gamma_vec[i] = gamma_vec[i] + delta_gamma[i];
          A_vi[i] = A_vi[i] + delta_a[i];
          //Rprintf("%f ",A_vi[i]);
        }

        F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);

    }

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("Updating Process \n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    while(iter <= max_iter & !indicator_converge){
      if(verbose){
        Rprintf("----------------------------------------\n");
        Rprintf("\tIteration at %i \n",iter);
#ifdef Win32
        R_FlushConsole();
#endif
      }
      zeros(gradient_const,n);
      zeros(gradient,n);
      zeros(gamma_gradient_sum, n);
      zeros(gamma_gradient,n);
      zeros(a_gradient,nIndx_vi);
      zeros(a_gradient_sum, nIndx_vi);
      double rho = rho_input;
      // if(iter < 100){
      //   rho = 0.95;
      // }else{
      //   rho = rho_input;
      // }
      for(int batch_index = 0; batch_index < nBatch; batch_index++){
        tempsize = tempsize_vec[batch_index];
        BatchSize = nBatchIndx[batch_index];
        for(i_mb = 0; i_mb < tempsize; i_mb++){
          epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
        }
        //BatchSize = nBatchIndx[batch_index];

        if(batch_index == iter % nBatch){
          if(verbose){
            Rprintf("the value of batch_index global : %i \n", batch_index);
#ifdef Win32
            R_FlushConsole();
#endif
          }
          a_tau_update = BatchSize * 0.5 + tauSqIGa;
          a_zeta_update = BatchSize * 0.5 + zetaSqIGa;

          zeros(tau_sq_I, one_int);
          zeros(tmp_n_mb, n);

          for(i = 0; i < BatchSize; i++){
            tmp_n_mb[i] = y[nBatchLU[batch_index] + i]-w_mu[nBatchLU[batch_index] + i];
            tau_sq_I[0] += pow(tmp_n_mb[i],2);
          }

          ///////////////
          //update tausq
          ///////////////
          int Trace_N_para = 1000;
          zeros(trace_vec,2);
          double trace_vec1_re = 0.0;

          for(i_mb = 0; i_mb < tempsize; i_mb++){
            epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
            // int random_num = std::rand() % 2;
            // int bernoulli_point = (random_num == 0) ? -1 : 1;
            // epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = bernoulli_point;
          }


          update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                     batch_index, final_result_vec, nBatchLU_temp, tempsize);
          // clock_t start, mid, end;
          // double cpu_time_used;
          // start = clock();
          for(int k = 0; k < Trace_N_para; k++){
            for(i_mb = 0; i_mb < tempsize; i_mb++){
              epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
              // int random_num = std::rand() % 2;
              // int bernoulli_point = (random_num == 0) ? -1 : 1;
              // epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = bernoulli_point;
            }
            update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                       batch_index, final_result_vec, nBatchLU_temp, tempsize);

            double u_mean = 0.0;
            for(i_mb = 0; i_mb < BatchSize; i_mb++){
              u_mean += u_vec[nBatchLU[batch_index] + i_mb];
            }
            u_mean = u_mean/BatchSize;
            for(i_mb = 0; i_mb < BatchSize; i_mb++){
              trace_vec[0] += pow(u_vec[nBatchLU[batch_index] + i_mb]-u_mean,2);
            }

            //trace_vec[1] += Q_mini_batch_plus(B, F, u_vec, u_vec, batch_index, n, nnIndx, nnIndxLU, final_result_vec, nBatchLU_temp, tempsize);
            //trace_vec[1] += Q_mini_batch(B, F, u_vec, u_vec, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU);
            trace_vec1_re += E_quadratic_mb(u_vec, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                            n, nnIndx, nnIndxLU, nnIndxLUSq);

          }
          // end = clock();
          // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
          // printf("trace %f seconds to execute \n", cpu_time_used);

          //b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
          if (!isnan(trace_vec[0])){
            b_tau_update = tauSqIGb + (trace_vec[0]/Trace_N_para + *tau_sq_I)*0.5;
            //Rprintf("add_tau is : %f \n",trace_vec[0]/Trace_N + *tau_sq_I);
            tau_sq = b_tau_update/a_tau_update;
            theta[tauSqIndx] = tau_sq;
          }else{
            theta[tauSqIndx] = 1;
          }

          theta[tauSqIndx] = 0.5;

          if(verbose){
            Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
            R_FlushConsole();
#endif
          }

          ///////////////
          //update zetasq
          ///////////////
          // updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //                         theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
          //                         batch_index, final_result_vec, nBatchLU_temp, tempsize);
          //double zeta_Q_mb = Q_mini_batch(B, F, w_mu, w_mu, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU);
          // clock_t start, mid, end;
          // double cpu_time_used;
          // start = clock();
          double zeta_Q_mb_re = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                    n, nnIndx, nnIndxLU, nnIndxLUSq);
          // end = clock();
          // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
          // printf("zeta_Q_mb_re %f seconds to execute \n", cpu_time_used);

          //Rprintf("zeta_Q_mb: %f \n", zeta_Q_mb);
          if (!isnan(trace_vec[1])){
            b_zeta_update = zetaSqIGb + (trace_vec1_re/Trace_N_para + zeta_Q_mb_re)*0.5;

            //Rprintf("add zeta is : %f \n",trace_vec[1]/Trace_N);
            zeta_sq = b_zeta_update/a_zeta_update;
            //zeta_sq = 13.227241;
            if(zeta_sq > 1000){zeta_sq = 1000;}
            theta[zetaSqIndx] = zeta_sq;
          }else{
            theta[zetaSqIndx] = 1;
          }

          // theta[zetaSqIndx] = 10;
          if(verbose){
            Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
            R_FlushConsole();
#endif
          }
          theta[zetaSqIndx] = 10;
          // updateBF_minibatch(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //                    theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
          //                    BatchSize, nBatchLU, batch_index);
          // updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //                         theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
          //                         batch_index, final_result_vec, nBatchLU_temp, tempsize);

          ///////////////
          //update phi
          ///////////////

          // if(iter > phi_start){
          //
          //   double *a_phi_vec = (double *) R_alloc(N_phi, sizeof(double));
          //   double *b_phi_vec = (double *) R_alloc(N_phi, sizeof(double));
          //   a_phi_vec[0] = a_phi;
          //   b_phi_vec[0] = b_phi;
          //
          //   for(int i = 1; i < N_phi; i++) {
          //     double adjustment = 0.25 * ((i % 2 == 0) ? (i / 2) : -(i / 2 + 1));
          //
          //     a_phi_vec[i] = std::max(a_phi_vec[0] + adjustment, 0.05);
          //     b_phi_vec[i] = std::max(b_phi_vec[0] + adjustment, 0.05);
          //   }
          //
          //   double phi_Q = 0.0;
          //   double diag_sigma_sq_sum = 0.0;
          //   int max_index;
          //
          //   zeros(phi_can_vec,N_phi*N_phi);
          //   zeros(log_g_phi,N_phi*N_phi);
          //
          //   // start = clock();
          //   for(int i = 0; i < N_phi; i++){
          //     for(int j = 0; j < N_phi; j++){
          //
          //       updateBF_quadratic_mb(B_temp, F_temp, Bmat_over_F_temp,
          //                             F_inv, B_over_F, Bmat_over_F,
          //                             nIndx, nIndSqx,
          //                             nnIndxLUSq,
          //                             Trace_phi,
          //                             c, C, coords, nnIndx, nnIndxLU,
          //                             BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
          //                             n,  m,
          //                             nu,  covModel, bk,  nuUnifb,
          //                             a_phi_vec[i],  b_phi_vec[j], phimax,  phimin);
          //
          //       double phi_re = 0.0;
          //
          //       update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
          //                                  batch_index, final_result_vec, nBatchLU_temp, tempsize);
          //
          //       phi_re = E_quadratic_mb(u_vec, F_inv, B_over_F, Bmat_over_F,  BatchSize, nBatchLU, batch_index,
          //                               n, nnIndx, nnIndxLU, nnIndxLUSq);
          //
          //       phi_Q = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
          //                              n, nnIndx, nnIndxLU, nnIndxLUSq);
          //
          //       logDetInv = 0.0;
          //       diag_sigma_sq_sum = 0.0;
          //       for(i_mb = 0; i_mb < BatchSize; i_mb++){
          //         s = nBatchLU[batch_index] + i_mb;
          //         logDetInv += log(F_inv[s]);
          //       }
          //
          //       log_g_phi[i*N_phi+j] = logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) -
          //         (phi_Q +  phi_re)*0.5/theta[zetaSqIndx];
          //       // Rprintf("the value of a_phi_vec[i]: %i, %f \n", i, a_phi_vec[i]);
          //       // Rprintf("the value of b_phi_vec[j]: %i, %f \n", j, b_phi_vec[j]);
          //       //
          //       // Rprintf("the value of log_g_phi[i*N_phi+j] : %f \n",log_g_phi[i*N_phi+j]);
          //       //
          //       // Rprintf("the value of phi_re: %f \n",phi_re);
          //
          //     }
          //   }
          //   // end = clock();
          //   // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
          //   // printf("phi %f seconds to execute \n", cpu_time_used);
          //   max_index = max_ind(log_g_phi,N_phi*N_phi);
          //   a_phi = a_phi_vec[max_index/N_phi];
          //   b_phi = b_phi_vec[max_index % N_phi];
          //
          //
          //
          //
          //
          //   theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;
          //
          //   // updateBF_minibatch(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //   //                    theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb,
          //   //                    BatchSize, nBatchLU, batch_index);
          //   // start = clock();
          //   updateBF_quadratic(B_temp, F_temp, Bmat_over_F_temp,
          //                      F_inv, B_over_F, Bmat_over_F,
          //                      nIndx, nIndSqx,
          //                      nnIndxLUSq,
          //                      Trace_phi,
          //                      c, C, coords, nnIndx, nnIndxLU,
          //                      n,  m,
          //                      nu,  covModel, bk,  nuUnifb,
          //                      a_phi,  b_phi,  phimax,  phimin);
          //   // updateBF_quadratic_mb(B_temp, F_temp, Bmat_over_F_temp,
          //   //                       F_inv, B_over_F, Bmat_over_F,
          //   //                       nIndx, nIndSqx,
          //   //                       nnIndxLUSq,
          //   //                       Trace_N,
          //   //                       c, C, coords, nnIndx, nnIndxLU,
          //   //                       BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
          //   //                       n, m,
          //   //                       nu,  covModel, bk,  nuUnifb,
          //   //                       a_phi,  b_phi,  phimax,  phimin);
          //   // end = clock();
          //   // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
          //   // printf("updateBF_quadratic %f seconds to execute \n", cpu_time_used);
          //
          // }

          if(iter < phi_iter_max){

            //double step_size = 0.1;
            double rho_phi = 0.7;
            double noise_phi = 1;
            double L_a_up = 0;
            double L_b_up = 0;
            double current_L = 0;
            double a_up = a_phi + noise_phi;
            double b_up = b_phi + noise_phi;
            double phi_Q, diag_sigma_sq_sum, delta_a, delta_b;


            update_uvec_minibatch_plus(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                       batch_index, final_result_vec, nBatchLU_temp, tempsize);
            // Calculate current_L
           {
            updateBF_quadratic_mb(B_temp, F_temp, Bmat_over_F_temp,
                                  F_inv, B_over_F, Bmat_over_F,
                                  nIndx, nIndSqx,
                                  nnIndxLUSq,
                                  Trace_phi,
                                  c, C, coords, nnIndx, nnIndxLU,
                                  BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                                  n,  m,
                                  nu,  covModel, bk,  nuUnifb,
                                  a_phi, b_phi, phimax,  phimin);

             double phi_re = 0.0;

             phi_re = E_quadratic_mb(u_vec, F_inv, B_over_F, Bmat_over_F,  BatchSize, nBatchLU, batch_index,
                                     n, nnIndx, nnIndxLU, nnIndxLUSq);

             phi_Q = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                    n, nnIndx, nnIndxLU, nnIndxLUSq);

             logDetInv = 0.0;
             diag_sigma_sq_sum = 0.0;
             for(i_mb = 0; i_mb < BatchSize; i_mb++){
               s = nBatchLU[batch_index] + i_mb;
               logDetInv += log(F_inv[s]);
             }

             current_L = logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) -
               (phi_Q +  phi_re)*0.5/theta[zetaSqIndx];
             }
            // Calculate L_b_up
            for(int i = 0; i < N_phi; i++) {

              updateBF_quadratic_mb(B_temp, F_temp, Bmat_over_F_temp,
                                    F_inv, B_over_F, Bmat_over_F,
                                    nIndx, nIndSqx,
                                    nnIndxLUSq,
                                    Trace_phi,
                                    c, C, coords, nnIndx, nnIndxLU,
                                    BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                                    n,  m,
                                    nu,  covModel, bk,  nuUnifb,
                                    a_phi, b_up, phimax,  phimin);

              double phi_re = 0.0;

              phi_re = E_quadratic_mb(u_vec, F_inv, B_over_F, Bmat_over_F,  BatchSize, nBatchLU, batch_index,
                                      n, nnIndx, nnIndxLU, nnIndxLUSq);

              phi_Q = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                     n, nnIndx, nnIndxLU, nnIndxLUSq);

              logDetInv = 0.0;
              diag_sigma_sq_sum = 0.0;
              for(i_mb = 0; i_mb < BatchSize; i_mb++){
                s = nBatchLU[batch_index] + i_mb;
                logDetInv += log(F_inv[s]);
              }

              L_b_up += logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) -
                (phi_Q +  phi_re)*0.5/theta[zetaSqIndx];

            }
            // Calculate L_a_up
            for(int i = 0; i < N_phi; i++) {

              updateBF_quadratic_mb(B_temp, F_temp, Bmat_over_F_temp,
                                    F_inv, B_over_F, Bmat_over_F,
                                    nIndx, nIndSqx,
                                    nnIndxLUSq,
                                    Trace_phi,
                                    c, C, coords, nnIndx, nnIndxLU,
                                    BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                                    n,  m,
                                    nu,  covModel, bk,  nuUnifb,
                                    a_up, b_phi, phimax,  phimin);

              double phi_re = 0.0;

              phi_re = E_quadratic_mb(u_vec, F_inv, B_over_F, Bmat_over_F,  BatchSize, nBatchLU, batch_index,
                                      n, nnIndx, nnIndxLU, nnIndxLUSq);

              phi_Q = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                     n, nnIndx, nnIndxLU, nnIndxLUSq);

              logDetInv = 0.0;
              diag_sigma_sq_sum = 0.0;
              for(i_mb = 0; i_mb < BatchSize; i_mb++){
                s = nBatchLU[batch_index] + i_mb;
                logDetInv += log(F_inv[s]);
              }

              L_a_up += logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) -
                (phi_Q +  phi_re)*0.5/theta[zetaSqIndx];

            }

            grad_a = (L_a_up/N_phi - current_L) / noise_phi;
            grad_b = (L_b_up/N_phi - current_L) / noise_phi;

            // Update running averages of squared gradients
            Eg2_a = rho_phi * Eg2_a + (1 - rho_phi) * grad_a * grad_a;
            Eg2_b = rho_phi * Eg2_b + (1 - rho_phi) * grad_b * grad_b;

            // Compute parameter updates using AdaDelta formula
            delta_a = sqrt(Ex2_a + noise_phi) / sqrt(Eg2_a + noise_phi) * grad_a;
            delta_b = sqrt(Ex2_b + noise_phi) / sqrt(Eg2_b + noise_phi) * grad_b;

            // Update running averages of squared updates
            Ex2_a = rho_phi * Ex2_a + (1 - rho_phi) * delta_a * delta_a;
            Ex2_b = rho_phi * Ex2_b + (1 - rho_phi) * delta_b * delta_b;

            // Update parameters
            a_phi += delta_a;
            b_phi += delta_b;

            // Rprintf("the value of current_L: %f \n", current_L);
            // Rprintf("the value of L_a_up: %f \n", L_a_up/N_phi);
            // Rprintf("the value of L_b_up: %f \n", L_b_up/N_phi);
            //
            // Rprintf("the value of a_phi: %f \n", a_phi);
            // Rprintf("the value of b_phi : %f \n", b_phi);

            theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;
            theta[phiIndx] = 1;
            // updateBF_minibatch(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
            //                    theta[zetaSqIndx], phi_can_vec[i], nu, covModel, bk, nuUnifb,
            //                    BatchSize, nBatchLU, batch_index);
            // start = clock();
            updateBF_quadratic(B_temp, F_temp, Bmat_over_F_temp,
                               F_inv, B_over_F, Bmat_over_F,
                               nIndx, nIndSqx,
                               nnIndxLUSq,
                               Trace_phi,
                               c, C, coords, nnIndx, nnIndxLU,
                               n,  m,
                               nu,  covModel, bk,  nuUnifb,
                               a_phi,  b_phi,  phimax,  phimin);
            // updateBF_quadratic_mb(B_temp, F_temp, Bmat_over_F_temp,
            //                       F_inv, B_over_F, Bmat_over_F,
            //                       nIndx, nIndSqx,
            //                       nnIndxLUSq,
            //                       Trace_N,
            //                       c, C, coords, nnIndx, nnIndxLU,
            //                       BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
            //                       n, m,
            //                       nu,  covModel, bk,  nuUnifb,
            //                       a_phi,  b_phi,  phimax,  phimin);
            // end = clock();
            // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            // printf("updateBF_quadratic %f seconds to execute \n", cpu_time_used);

          }

          if(verbose){
            Rprintf("the value of theta[%i phiIndx] : %f \n", phiIndx, theta[phiIndx]);
#ifdef Win32
            R_FlushConsole();
#endif
          }
        }
        //for(int batch_index = 0; batch_index < nBatch; batch_index++)
        if(verbose){
          Rprintf("the value of batch_index for w : %i \n", batch_index);
#ifdef Win32
          R_FlushConsole();
#endif
        }
        tempsize = tempsize_vec[batch_index];
        BatchSize = nBatchIndx[batch_index];

        ///////////////
        //update w
        ///////////////

        //zeros_minibatch(w_mu_temp,n, BatchSize, nBatchLU, batch_index);
        //zeros_minibatch(w_mu_temp2,n, BatchSize, nBatchLU, batch_index);
        double gradient_mu;
        double one_over_zeta_sq = 1.0/theta[zetaSqIndx];
        zeros(w_mu_temp,n);
        zeros(w_mu_temp_dF,n);
        zeros(w_mu_temp2,n);
        // clock_t start, mid, end;
        // double cpu_time_used;
        // start = clock();
        product_B_F_combine_mb(w_mu, w_mu_temp, w_mu_temp_dF, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                               BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                            n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                            nnIndxwhich);
        // end = clock();
        // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        // printf("product_B_F_combine_mb %f seconds to execute \n", cpu_time_used);

        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp, &inc);
        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp_dF, &inc);
        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);

        for (i_mb = 0; i_mb < intersect_sizes[batch_index]; i_mb++) {
          i = final_intersect_vec[intersect_start_indices[batch_index] + i_mb];
          gradient_mu_vec[i] =  (y[i] - w_mu[i])/theta[tauSqIndx] - w_mu_temp2[i];
        }
        for (i_mb = 0; i_mb < complement_first_sizes[batch_index]; i_mb++) {
          i = final_complement_1_vec[complement_first_start_indices[batch_index] + i_mb];
          //gradient_mu_vec[i] =  (y[i] - w_mu[i])/theta[tauSqIndx] - w_mu_temp_dF[i];
          gradient_mu_vec[i] =  (y[i] - w_mu[i])/theta[tauSqIndx] - w_mu_temp[i];

        }


        for (i_mb = 0; i_mb < complement_second_sizes[batch_index]; i_mb++) {
          i = final_complement_2_vec[complement_second_start_indices[batch_index] + i_mb];
          //gradient_mu_vec[i] = - w_mu_temp2[i] + w_mu_temp_dF[i];
          gradient_mu_vec[i] = w_mu_temp_dF[i];
        }



        for(i_mb = 0; i_mb < tempsize; i_mb++){
          i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
          //gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
          gradient_mu = gradient_mu_vec[i];
          E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
          delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
          delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
          w_mu_update[i] = w_mu[i] + delta_mu[i];
        }

        zeros(w_mu_temp,n);
        zeros(w_mu_temp_dF,n);
        zeros(w_mu_temp2,n);
        // product_B_F_minibatch_plus(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
        // product_B_F_minibatch_term1(B, F, w_mu_update, n, nnIndxLU, nnIndx, w_mu_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);
        // product_B_F_vec_minibatch_plus_fix(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);

        product_B_F_combine_mb(w_mu, w_mu_temp, w_mu_temp_dF, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                               BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                               n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                               nnIndxwhich);

        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp, &inc);
        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp_dF, &inc);
        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);

        // start = clock();
        zeros(gamma_gradient_sum, n);
        for(int k = 0; k < Trace_N; k++){
          //zeros_minibatch_plus(gamma_gradient,n, batch_index,final_result_vec, nBatchLU_temp, tempsize);
          zeros(gradient,n);
          zeros(gamma_gradient,n);
          for(i_mb = 0; i_mb < tempsize; i_mb++){
            epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
            // int random_num = std::rand() % 2;
            // int bernoulli_point = (random_num == 0) ? -1 : 1;
            // epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = bernoulli_point;
          }

          gamma_gradient_mb_fun2(u_vec, epsilon_vec, gamma_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                                 nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                                 cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,
                                 u_vec_temp, u_vec_temp2, u_vec_temp_dF, gradient, zetaSqIndx,
                                 F_inv, B_over_F, Bmat_over_F,
                                 nnIndxLUSq, nnIndxwhich,
                                 batch_index, BatchSize, nBatchLU,
                                 final_result_vec, nBatchLU_temp, tempsize,
                                 intersect_start_indices, intersect_sizes, final_intersect_vec,
                                 complement_first_start_indices, complement_first_sizes, final_complement_1_vec,
                                 complement_second_start_indices, complement_second_sizes, final_complement_2_vec);

          vecsum_minibatch_plus(gamma_gradient_sum, gamma_gradient, Trace_N, n, batch_index, final_result_vec, nBatchLU_temp, tempsize);

        }
        // end = clock();
        // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        // printf("gamma_gradient_mb_fun2 %f seconds to execute \n", cpu_time_used);

        for(i_mb = 0; i_mb < tempsize; i_mb++){
          i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
          //Rprintf("gamma gradient[%i],: %f \n",i, gamma_gradient_sum[i]);
          E_gamma_sq[i] = rho * E_gamma_sq[i] + (1 - rho) * pow(gamma_gradient_sum[i],2);
          delta_gamma[i] = sqrt(delta_gamma_sq[i]+adadelta_noise)/sqrt(E_gamma_sq[i]+adadelta_noise)*gamma_gradient_sum[i];
          delta_gamma_sq[i] = rho*delta_gamma_sq[i] + (1 - rho) * pow(delta_gamma[i],2);
          S_vi[i] = pow(exp(log(sqrt(S_vi[i])) + delta_gamma[i]),2);
        }
        // start = clock();
        zeros(a_gradient_sum, nIndx_vi);
        for(int k = 0; k < Trace_N; k++){
          zeros(gradient,n);
          zeros(a_gradient,nIndx_vi);
          zeros(w_mu_temp,n);
          for(i_mb = 0; i_mb < tempsize; i_mb++){
            epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = rnorm(0, 1);
            // int random_num = std::rand() % 2;
            // int bernoulli_point = (random_num == 0) ? -1 : 1;
            // epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = bernoulli_point;
          }

          a_gradient_mb_fun2(u_vec, epsilon_vec, a_gradient, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi,
                             nnIndx, nnIndxLU, theta, tauSqIndx, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                             cumnumIndxCol_vi, numIndxCol_vi, nnIndxCol_vi, nnIndxnnCol_vi,
                             u_vec_temp, u_vec_temp2, u_vec_temp_dF, gradient, zetaSqIndx,
                             F_inv, B_over_F, Bmat_over_F,
                             nnIndxLUSq, nnIndxwhich,
                             batch_index, BatchSize, nBatchLU,
                             final_result_vec, nBatchLU_temp, tempsize,
                             intersect_start_indices, intersect_sizes, final_intersect_vec,
                             complement_first_start_indices, complement_first_sizes, final_complement_1_vec,
                             complement_second_start_indices, complement_second_sizes, final_complement_2_vec);

          //vecsum(a_gradient_sum, a_gradient, Trace_N, nIndx_vi);
          for(int i_mb = 0; i_mb < tempsize; i_mb++){
            i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
            for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
              a_gradient_sum[nnIndxLU_vi[i] + l] += a_gradient[nnIndxLU_vi[i] + l]/Trace_N;
            }
          }


        }

        // end = clock();
        // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        // printf("a_gradient_mb_fun2 %f seconds to execute \n", cpu_time_used);

        int sub_index;
        //Rprintf("A_vi: ");
        for(int i_mb = 0; i_mb < tempsize; i_mb++){
          i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
          for (int l = 0; l < nnIndxLU_vi[n + i]; l++) {
            sub_index = nnIndxLU_vi[i] + l;
            //Rprintf("a gradient[%i],: %f \n",sub_index, a_gradient_sum[sub_index]);
            //a_gradient_sum[nnIndxLU_vi[i] + l] += a_gradient[nnIndxLU_vi[i] + l]/Trace_N;
            E_a_sq[sub_index] = rho * E_a_sq[sub_index] + (1 - rho) * pow(a_gradient_sum[sub_index],2);
            delta_a[sub_index] = sqrt(delta_a_sq[sub_index]+adadelta_noise)/sqrt(E_a_sq[sub_index]+adadelta_noise)*a_gradient_sum[sub_index];
            delta_a_sq[sub_index] = rho*delta_a_sq[sub_index] + (1 - rho) * pow(delta_a[sub_index],2);
            A_vi[sub_index] = A_vi[sub_index] + delta_a[sub_index];
            //Rprintf("A_vi[i]: %i, %f \n",sub_index, A_vi[sub_index]);
            //Rprintf("\t Updated a index is %i \n",sub_index);
          }
        }
        //Rprintf("\n");
        F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);

      }

      // Rprintf("\t rho at %i is %f\n",iter,rho);
      ELBO = 0.0;
      zeros(sum_v,n);
      double sum1 = 0.0;
      double sum2 = 0.0;
      double sum3 = 0.0;
      double sum4 = 0.0;
      double sum5 = 0.0;
      double sum6 = 0.0;

      for(int k = 0; k < Trace_N; k++){
        for(int i = 0; i < n; i++){
          //epsilon_vec[i] = rnorm(0, 1);
          int random_num = std::rand() % 2;
          int bernoulli_point = (random_num == 0) ? -1 : 1;
          epsilon_vec[final_result_vec[nBatchLU_temp[batch_index] + i_mb]] = bernoulli_point;
        }
        update_uvec(u_vec, epsilon_vec, A_vi, S_vi, n, nnIndxLU_vi, nnIndx_vi);

        sum2 += sumsq(u_vec, n);

        sum4 += E_quadratic(u_vec, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);
      }

      for(int i = 0; i < n; i++){
        sum1 += pow((y[i] - w_mu_update[i]),2);
        sum5 += log(2*pi*S_vi[i]);
        sum6 += log(2*pi*F_inv[i]/theta[zetaSqIndx]);
      }

      sum1 = sum1/theta[tauSqIndx]*0.5;
      sum2 = sum2/Trace_N/theta[tauSqIndx]*0.5;
      sum3 = E_quadratic(w_mu_update, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq)/theta[zetaSqIndx]*0.5;
      sum4 = sum4/Trace_N/theta[zetaSqIndx]*0.5;

      ELBO = -sum1 - sum2 - sum3 - sum4 + sum5*0.5 + sum6*0.5 + 0.5*n - 0.5*n*log(2*pi*theta[tauSqIndx]);

      //Rprintf("the value of ELBO: %f \n", ELBO);
      ELBO_vec[iter-1] = ELBO;


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


      F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);


    }

    //
    //updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
    updateBF2(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[phiIndx], nu, covModel, bk, nuUnifb);
    //zetaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
    SEXP theta_para_r; PROTECT(theta_para_r = allocVector(REALSXP, nTheta*2)); nProtect++; double *theta_para = REAL(theta_para_r);

    theta_para[zetaSqIndx*2+0] = a_zeta_update;
    theta_para[zetaSqIndx*2+1] = b_zeta_update;

    theta_para[tauSqIndx*2+0] = a_tau_update;
    theta_para[tauSqIndx*2+1] = b_tau_update;

    theta_para[phiIndx*2+0] = a_phi;
    theta_para[phiIndx*2+1] = b_phi;


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
