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

#ifndef FCONE
#define FCONE
#endif


extern "C" {
  SEXP updateBFq_cpp(SEXP H_, 
                          SEXP V_diag_, 
                          SEXP nnIndx_, 
                          SEXP nnIndxLU_, 
                          SEXP B_q_, 
                          SEXP F_q_);
}

extern "C" {


  SEXP spVarBayes_MFA_LR_update_nocovariates_bigcpp(SEXP y_r,
                                       SEXP n_r, SEXP m_r, SEXP coords_r, SEXP covModel_r,
                                       SEXP zetaSqIG_r, SEXP tauSqIG_r,
                                       SEXP sType_r, SEXP nThreads_r, SEXP fix_nugget_r,  SEXP nuUnif_r,SEXP nuStarting_r,
                                       SEXP w_mu_r,
                                       SEXP theta_input_r,
                                       SEXP H_r, SEXP V_diag_r){


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
    // double *Sigma_star = REAL(Sigma_star_r);
    double *w_mu = REAL(w_mu_r);
    int n = INTEGER(n_r)[0];
    int m = INTEGER(m_r)[0];
    double *coords = REAL(coords_r);
    double fix_nugget = REAL(fix_nugget_r)[0];
    int covModel = INTEGER(covModel_r)[0];
    std::string corName = getCorName(covModel);
    int nThreads = INTEGER(nThreads_r)[0];
    double *theta_input = REAL(theta_input_r);

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

    int *nnIndx = (int *) R_alloc(nIndx, sizeof(int));

    double *d = (double *) R_alloc(nIndx, sizeof(double));

    int *nnIndxLU = (int *) R_alloc(2*n, sizeof(int));

    if(INTEGER(sType_r)[0] == 0){
      mkNNIndx(n, m, coords, nnIndx, d, nnIndxLU);
    }
    if(INTEGER(sType_r)[0] == 1){
      mkNNIndxTree0(n, m, coords, nnIndx, d, nnIndxLU);
    }else{
      mkNNIndxCB(n, m, coords, nnIndx, d, nnIndxLU);
    }

    int *CIndx = (int *) R_alloc(2*n, sizeof(int));
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

    double *D = (double *) R_alloc(j, sizeof(double));

    for(i = 0; i < n; i++){
      for(k = 0; k < nnIndxLU[n+i]; k++){
        for(l = 0; l <= k; l++){
          D[CIndx[i]+l*nnIndxLU[n+i]+k] = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
        }
      }
    }
    int mm = m*m;

    double *B = (double *) R_alloc(nIndx, sizeof(double));
    double *F = (double *) R_alloc(n, sizeof(double));

    double *c =(double *) R_alloc(m*nThreads, sizeof(double));
    double *C = (double *) R_alloc(mm*nThreads, sizeof(double));

    SEXP theta_r; PROTECT(theta_r = allocVector(REALSXP, nTheta)); nProtect++; double *theta = REAL(theta_r);

    theta[zetaSqIndx] = theta_input[zetaSqIndx];
    theta[tauSqIndx] = theta_input[tauSqIndx];
    //theta[phiIndx] = REAL(phiStarting_r)[0];
    theta[phiIndx] = theta_input[phiIndx];
    if(corName == "matern"){
      theta[nuIndx] = REAL(nuStarting_r)[0];
    }

    //other stuff
    double logDetInv;
    int accept = 0, batchAccept = 0, status = 0;
    int jj, kk, nn = n*n;
    double *one_n = (double *) R_alloc(n, sizeof(double)); ones(one_n, n);
    //double *tmp_n = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n, n);

    double *tmp_n = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n, n);

    double *tau_sq_H = (double *) R_alloc(one, sizeof(double));
    double *tau_sq_I = (double *) R_alloc(one, sizeof(double));
    double *bk = (double *) R_alloc(nThreads*(1.0+5.0), sizeof(double));
    double a, v, b, e, mu, var, aij, phiCand, nuCand = 0, nu = 0;
    if(corName == "matern"){nu = theta[nuIndx];}

    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

    double zetaSqIGa = REAL(zetaSqIG_r)[0]; double zetaSqIGb = REAL(zetaSqIG_r)[1];
    double tauSqIGa = REAL(tauSqIG_r)[0]; double tauSqIGb = REAL(tauSqIG_r)[1];


    // *** NEW: Compute Sigma_star internally ***
    // Call your parallel function compute_Hinv_V_full_p_parallel.
    // It expects H, V_top, V_diag, and p as SEXP arguments.
    int n_plus_p = n; // assumed square: (n+p) x (n+p)

    int n_new = n_plus_p;

    SEXP B_q_r; PROTECT(B_q_r = allocVector(REALSXP, nIndx)); nProtect++; 
    SEXP F_q_r; PROTECT(F_q_r = allocVector(REALSXP, n)); nProtect++; 
    
    double *c_q =(double *) R_alloc(m*nThreads, sizeof(double));
    double *C_q = (double *) R_alloc(mm*nThreads, sizeof(double));
    
    int p = 0;
    
    Rprintf("------------------------------------------------------\n");
    Rprintf("\tDecompose the updated covariance matrix \n");
    
    // Call the updateBFq_cpp function
    SEXP nnIndx_SEXP = PROTECT(allocVector(INTSXP, nIndx)); nProtect++; 
    int *nnIndx_SEXP_ptr = INTEGER(nnIndx_SEXP);
    for (int i = 0; i < nIndx; i++) {
      nnIndx_SEXP_ptr[i] = nnIndx[i];  
    }
    
    SEXP nnIndxLU_SEXP = PROTECT(allocVector(INTSXP, 2*n)); nProtect++; 
    int *nnIndxLU_SEXP_ptr = INTEGER(nnIndxLU_SEXP);
    for (int i = 0; i < 2*n; i++) {
      nnIndxLU_SEXP_ptr[i] = nnIndxLU[i];
    }

    SEXP updateRes; nProtect++; 
    PROTECT(updateRes = updateBFq_cpp(H_r, V_diag_r, 
                                      nnIndx_SEXP, 
                                      nnIndxLU_SEXP, 
                                      B_q_r,      
                                      F_q_r));      
    // updateBFq_cpp returns a list with elements "B_q" and "F_q"
  
    B_q_r = VECTOR_ELT(updateRes, 0);
    F_q_r = VECTOR_ELT(updateRes, 1);

    double* B_q = REAL(B_q_r);
    double* F_q = REAL(F_q_r);
    
    double trace = 0.0;
    
    double *epsilon_vec = (double *) R_alloc(n, sizeof(double));
    double *u_vec = (double *) R_alloc(n, sizeof(double));
    
    int Trace_N = 50; 
    for(int k = 0; k < Trace_N; k++){
      for(int i = 0; i < n; i++){
        epsilon_vec[i] = rnorm(0, 1);
      }
      update_uvec_lr(u_vec, epsilon_vec, B_q, F_q, n, nnIndxLU, nnIndx);
      
      trace += Q(B, F, u_vec, u_vec, n, nnIndx, nnIndxLU);
    }

    Rprintf("------------------------------------------------------\n");
    Rprintf("\tUpdating the spatial parameters \n");
    Rprintf("------------------------------------------------------\n");
#ifdef Win32
    R_FlushConsole();
#endif

    ////////////////// Update tausq //////////////////
    double a_tau_update = n * 0.5 + tauSqIGa;
    double b_tau_update = 0.0;
    int one_int = 1;

    double sum_diags = 0;
    zeros(tau_sq_I, one_int);
    zeros(tmp_n, n);

    for(i = 0; i < n; i++){
      tmp_n[i] = y[i]-w_mu[i];
      tau_sq_I[0] += pow(tmp_n[i],2);
      // sum_diags += HinvV_full[i + i*n];
    }

    b_tau_update = tauSqIGb + (sum_diags + *tau_sq_I)*0.5;
    a_tau_update = tauSqIGa + n*0.5;


    theta[tauSqIndx] = b_tau_update/a_tau_update;

    ////////////////// Update sigmasq //////////////////

    double a_zeta_update = n * 0.5 + zetaSqIGa;
    double b_zeta_update = 0.0;

    double zeta_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);

    b_zeta_update = zetaSqIGb + (trace/Trace_N + zeta_Q)*theta[zetaSqIndx]*0.5;

    theta[zetaSqIndx] = b_zeta_update/a_zeta_update;

    SEXP theta_para_r; PROTECT(theta_para_r = allocVector(REALSXP, nTheta*2)); nProtect++; double *theta_para = REAL(theta_para_r);

    theta_para[zetaSqIndx*2+0] = a_zeta_update;
    theta_para[zetaSqIndx*2+1] = b_zeta_update;

    theta_para[tauSqIndx*2+0] = a_tau_update;
    theta_para[tauSqIndx*2+1] = b_tau_update;

    SEXP result_r, resultName_r;
    int nResultListObjs = 4;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, theta_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("theta"));

    SET_VECTOR_ELT(result_r, 1, theta_para_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("theta_para"));
    
    SET_VECTOR_ELT(result_r, 2, B_q_r);
    SET_VECTOR_ELT(resultName_r, 2, mkChar("B_q"));
    
    SET_VECTOR_ELT(result_r, 3, F_q_r);
    SET_VECTOR_ELT(resultName_r, 3, mkChar("F_q"));
    
    namesgets(result_r, resultName_r);
    //unprotect
    UNPROTECT(nProtect);


    return(result_r);

  }

}
