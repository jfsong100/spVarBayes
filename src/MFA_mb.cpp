#define USE_FC_LEN_T
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include "util.h"
#include "nngp_fun.h"
#include <vector>

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
// typedef Matrix< double ,     Dynamic , 1>       vector;


#ifndef FCONE
#define FCONE
#endif

//Description: update B and F.


extern "C" {


  SEXP spVarBayes_MFA_nocovariates_mb_beta_rephicpp(SEXP y_r, SEXP n_r, SEXP p_r, SEXP m_r, SEXP coords_r, SEXP covModel_r, SEXP rho_r,
                                              SEXP zetaSqIG_r, SEXP tauSqIG_r, SEXP phibeta_r, SEXP nuUnif_r,
                                              SEXP zetaSqStarting_r, SEXP tauSqStarting_r, SEXP phiStarting_r, SEXP nuStarting_r,
                                              SEXP sType_r, SEXP nThreads_r, SEXP verbose_r, SEXP fix_nugget_r, SEXP N_phi_r, SEXP Trace_N_r,
                                              SEXP max_iter_r,
                                              SEXP var_input_r,
                                              SEXP phi_input_r, SEXP phi_iter_max_r, SEXP initial_mu_r,
                                              SEXP mini_batch_size_r,
                                              SEXP min_iter_r, SEXP K_r, SEXP stop_K_r, SEXP tausq_input_r, SEXP zetasq_input_r, SEXP LR_r){

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
    double pi = 3.1415926;
    //get args
    double *y = REAL(y_r);
    int p = INTEGER(p_r)[0];
    int n = INTEGER(n_r)[0];
    int m = INTEGER(m_r)[0];
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

    double tausq_input  =  REAL(tausq_input_r)[0];
    double zetasq_input  =  REAL(zetasq_input_r)[0];
    int LR = INTEGER(LR_r)[0];

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
      Rprintf("Model fit with %i observations.\n\n", n);
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
    int jj, kk, pp = p*p, nn = n*n, np = n*p;
    double *one_n = (double *) R_alloc(n, sizeof(double)); ones(one_n, n);
    double *tmp_n = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n, n);


    double *tau_sq_I = (double *) R_alloc(one, sizeof(double));

    //double *w_mu = (double *) R_alloc(n, sizeof(double));

    if(initial_mu){
      F77_NAME(dcopy)(&n, y, &inc, w_mu, &inc);
    }else{
      zeros(w_mu, n);
    }
    //double *sigma_sq = (double *) R_alloc(n, sizeof(double));
    for(int i = 0; i < n; i++){
      sigma_sq[i] = var_input[i];
    }

    double *w_mu_update = (double *) R_alloc(n, sizeof(double)); zeros(w_mu_update, n);
    double *E_mu_sq = (double *) R_alloc(n, sizeof(double)); zeros(E_mu_sq, n);
    double *delta_mu_sq = (double *) R_alloc(n, sizeof(double)); zeros(delta_mu_sq, n);
    double *delta_mu = (double *) R_alloc(n, sizeof(double)); zeros(delta_mu, n);
    double *m_mu = (double *) R_alloc(n, sizeof(double)); zeros(m_mu, n);

    double *E_sigmasq_sq = (double *) R_alloc(n, sizeof(double)); zeros(E_sigmasq_sq, n);
    double *delta_sigmasq_sq = (double *) R_alloc(n, sizeof(double)); zeros(delta_sigmasq_sq, n);
    double *delta_sigmasq = (double *) R_alloc(n, sizeof(double)); zeros(delta_sigmasq, n);
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
    double adadelta_noise = 0.0000001;
    double *bk = (double *) R_alloc(nThreads*(1.0+5.0), sizeof(double));
    //double *bk = (double *) R_alloc(nThreads*(1.0+static_cast<int>(floor(nuUnifb))), sizeof(double));
    if(corName == "matern"){nu = theta[nuIndx];}

    int iter = 1;
    int max_iter = INTEGER(max_iter_r)[0];

    double vi_error = 1.0;
    double rho1 = 0.9;
    double rho2 = 0.999;
    double adaptive_adam = 0.001;
    // int n_per = n * converge_per;
    // int *sign_vec_old = (int *) R_alloc(n_per, sizeof(int));
    // int *sign_vec_new = (int *) R_alloc(n_per, sizeof(int));
    // int *check_vec = (int *) R_alloc(n_per, sizeof(int));

    int nIndSqx = static_cast<int>(static_cast<double>(1+m)*(m+m+1)/6*m+(n-m-1)*m*m);
    double *F_inv = (double *) R_alloc(n, sizeof(double)); zeros(F_inv,n);
    double *B_over_F = (double *) R_alloc(nIndx, sizeof(double)); zeros(B_over_F,nIndx);
    double *Bsq_over_F = (double *) R_alloc(nIndx, sizeof(double)); zeros(Bsq_over_F,nIndx);
    double *Bmat_over_F = (double *) R_alloc(nIndSqx, sizeof(double)); zeros(Bmat_over_F,nIndSqx);
    int *nnIndxLUSq = (int *) R_alloc(2*n, sizeof(int));

    double *F_temp = (double *) R_alloc(n, sizeof(double)); zeros(F_temp,n);
    double *B_temp = (double *) R_alloc(nIndx, sizeof(double)); zeros(B_temp,nIndx);
    double *Bmat_over_F_temp = (double *) R_alloc(nIndSqx, sizeof(double)); zeros(Bmat_over_F_temp,nIndSqx);

    updateBF2(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[phiIndx], nu, covModel, bk, nuUnifb);

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
        Bsq_over_F[nnIndxLU[i]+j] = B[nnIndxLU[i]+j]*B[nnIndxLU[i]+j]/F[i];
      }

      if(i > 0){
        F_inv_temp = 1/F[i];
        F77_NAME(dger)(&nnIndxLU[n+i], &nnIndxLU[n+i], &one, &B[nnIndxLU[i]], &inc, &B[nnIndxLU[i]], &inc, &Bmat_over_F[nnIndxLUSq[i]], &nnIndxLU[n+i]);
        F77_NAME(dscal)(&nnIndxLUSq[n+i], &F_inv_temp, &Bmat_over_F[nnIndxLUSq[i]], &inc);
      }

    }
    int *nnIndxwhich = (int *) R_alloc(nIndx, sizeof(int));; zeros_int(nnIndxwhich, nIndx);
    get_nnIndxwhich(nnIndxwhich, n, nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);

    int indicator_converge = 0;

    double *trace_vec = (double *) R_alloc(2, sizeof(double));
    double ELBO_MC = 0.0;
    double ELBO = 0.0;

    SEXP ELBO_vec_r; PROTECT(ELBO_vec_r = allocVector(REALSXP, max_iter)); nProtect++;
    double *ELBO_vec = REAL(ELBO_vec_r); zeros(ELBO_vec,max_iter);
    double max_ELBO = 0.0;
    int ELBO_convergence_count = 0;


    double *diag_ouput = (double *) R_alloc(n, sizeof(double));
    double *diag_input = (double *) R_alloc(n, sizeof(double));
    double *u_vec = (double *) R_alloc(n, sizeof(double));
    double *epsilon_vec = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp_dF = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp2 = (double *) R_alloc(n, sizeof(double));
    double *gradient_sigmasq_temp = (double *) R_alloc(n, sizeof(double));
    double *MFA_sigmasq_grad_vec = (double *) R_alloc(n, sizeof(double));
    double *MFA_sigmasq_grad_vec_cum = (double *) R_alloc(n, sizeof(double));
    double *gradient_mu_vec = (double *) R_alloc(n, sizeof(double));

    double E_phi_sq = 0.0;
    double delta_phi = 0.0;
    double delta_phi_sq = 0.0;

    double *tmp_n_mb = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n_mb, n);
    double *diag_input_mb = (double *) R_alloc(n, sizeof(double)); zeros(diag_input_mb, n);
    double *phi_can_vec = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    double *log_g_phi = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    int BatchSize;
    double sum_diags= 0.0;
    int i_mb;

    int batch_index = 0;

    int max_result_size = nBatch * n;
    int max_temp_size = n;

    int* result_arr = (int *) R_alloc(max_result_size, sizeof(int));
    int* temp_arr = (int *) R_alloc(max_temp_size, sizeof(int));
    int result_index = 0;
    int temp_index = 0;

    int* tempsize_vec = (int *) R_alloc(nBatch, sizeof(int));

    int *seen_values = (int *) R_alloc(n, sizeof(int)); // initialized to zeros


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

    for(int batch_index = 0; batch_index < nBatch; batch_index++) {
      BatchSize = nBatchIndx[batch_index];
      zeros_int(seen_values,n);

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


    // Print all elements of tempsize_vec
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

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("Initialize Process \n");
#ifdef Win32
      R_FlushConsole();
#endif
    }
    if(initial_mu){

      zeros(tmp_n, n);
      zeros(tau_sq_I, one_int);
      for(i = 0; i < n; i++){
        tmp_n[i] = y[i]-w_mu[i];
        tau_sq_I[0] += pow(tmp_n[i],2);
      }

      ///////////////
      //update tausq
      ///////////////

      b_tau_update = tauSqIGb + (F77_NAME(dasum)(&n, sigma_sq, &inc) +
        *tau_sq_I )*0.5;

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

        //updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
        //double zeta_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
        double zeta_Q_re = E_quadratic(w_mu, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);

        double sum1 = 0;
        double sum2 = 0;
        for(int i = 0; i < n; i++){
          sum1 = sigma_sq[i] * F_inv[i];
          int num_m = nnIndxLU[n+i];
          // Rprintf("i's sons \n");
          if(i > 0){
            for (int l = 0; l < nnIndxLU[n + i]; l++) {
              sum1 = sum1 + Bsq_over_F[nnIndxLU[i] + l] * sigma_sq[nnIndx[nnIndxLU[i] + l]];
              // Rprintf("%i ",nnIndx[nnIndxLU[i] + l]);
            }
          }
          sum2 += sum1;
        }

        b_zeta_update = zetaSqIGb + (sum2 + zeta_Q_re)*0.5;
        //b_zeta_update = zetaSqIGb + (sum2 + zeta_Q)*theta[zetaSqIndx]*0.5;


        zeta_sq = b_zeta_update/a_zeta_update;

        theta[zetaSqIndx] = zeta_sq;

        if(verbose){
          Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
          R_FlushConsole();
#endif
        }
        // updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

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

              updateBF_quadratic(B_temp, F_temp, Bmat_over_F_temp,
                                 F_inv, B_over_F, Bmat_over_F,
                                 nIndx, nIndSqx,
                                 nnIndxLUSq,
                                 Trace_phi,
                                 c, C, coords, nnIndx, nnIndxLU,
                                 n,  m,
                                 nu,  covModel, bk,  nuUnifb,
                                 a_phi_vec[i],  b_phi_vec[j],  phimax,  phimin);

              phi_Q = E_quadratic(w_mu, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);

              sum1 = 0;
              sum2 = 0;
              for(int s = 0; s < n; s++){
                sum1 = sigma_sq[s] * F_inv[s];
                // Rprintf("i's sons \n");
                if(s > 0){
                  for (int l = 0; l < nnIndxLU[n + s]; l++) {
                    sum1 = sum1 + Bsq_over_F[nnIndxLU[s] + l] * sigma_sq[nnIndx[nnIndxLU[s] + l]];
                    // Rprintf("%j ",nnIndx[nnIndxLU[j] + l]);
                  }
                }
                sum2 += sum1;
              }

              logDetInv = 0.0;
              //diag_sigma_sq_sum = 0.0;
              for(int s = 0; s < n; s++){
                logDetInv += log(F_inv[s]);
              }

              log_g_phi[i*N_phi+j] = logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) - (phi_Q + sum2)*0.5/theta[zetaSqIndx];

            }
          }


          max_index = max_ind(log_g_phi,N_phi*N_phi);
          a_phi = a_phi_vec[max_index/N_phi];
          b_phi = b_phi_vec[max_index % N_phi];

          theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;

          MFA_updateBF_quadratic(B_temp,  F_temp,  Bmat_over_F_temp,
                                 F_inv, B_over_F, Bmat_over_F, Bsq_over_F,
                                 nIndx, nIndSqx,
                                 nnIndxLUSq,
                                 Trace_N,
                                 c, C, coords, nnIndx, nnIndxLU,
                                 n, m,
                                 nu, covModel, bk, nuUnifb,
                                 a_phi, b_phi,
                                 phimax, phimin);

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
        double one_over_zeta_sq = 1.0/theta[zetaSqIndx];
        zeros(w_mu_temp,n);
        zeros(w_mu_temp2,n);
        // product_B_F(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp);
        // product_B_F_vec(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
        product_B_F_combine(w_mu, w_mu_temp, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                            n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                            nnIndxwhich);
        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);

        double gradient_mu = 0.0;
        for(i = 0; i < n; i++){
          gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i])/theta[tauSqIndx]);
          E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
          delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
          delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
          w_mu_update[i] = w_mu[i] + delta_mu[i];
        }

        zeros(gradient_sigmasq_temp,n);
        zeros(MFA_sigmasq_grad_vec,n);
        double sum3;
        for (int i = 0; i < n; i++) {
          // Rprintf("i is : %i \n", i);
          sum3 = 0;
          int i_l, num_m, j_ind;
          // Rprintf("i's neighbor: \n ");
          if(numIndxCol[i] > 0){
            for (int l = 0; l < numIndxCol[i]; l++) {
              // l is the lth that the i is whose neighbor
              // transfer to i_l
              i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];

                // Rprintf("%i \n", i_l);
                //sum3 += B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]*B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]/F[i_l];
                num_m = nnIndxLU[n+i_l];
                j_ind = nnIndxwhich[cumnumIndxCol[i] - i + l];
                sum3 += Bsq_over_F[nnIndxCol[ 1 + cumnumIndxCol[i] + l] ];

            }
          }
          // Rprintf("\n");
          gradient_sigmasq_temp[i] =  sum3;
        }
        F77_NAME(dscal)(&n, &one_over_zeta_sq, gradient_sigmasq_temp, &inc);

        for (int i = 0; i < n; i++) {
          MFA_sigmasq_grad_vec[i] = (1/sigma_sq[i] - 1/theta[tauSqIndx] - F_inv[i]/theta[zetaSqIndx] - gradient_sigmasq_temp[i]) * 0.5 * sigma_sq[i];
        }

        double gradient_sigmasq;
        for(int i = 0; i < n; i++){
          gradient_sigmasq = MFA_sigmasq_grad_vec[i];
          E_sigmasq_sq[i] = rho * E_sigmasq_sq[i] + (1 - rho) * pow(gradient_sigmasq,2);
          delta_sigmasq[i] = sqrt(delta_sigmasq_sq[i]+adadelta_noise)/sqrt(E_sigmasq_sq[i]+adadelta_noise)*gradient_sigmasq;
          delta_sigmasq_sq[i] = rho*delta_sigmasq_sq[i] + (1 - rho) * pow(delta_sigmasq[i],2);
          sigma_sq_update[i] = exp(log(sigma_sq[i]) + delta_sigmasq[i]);
        }



        F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);
        F77_NAME(dcopy)(&n, sigma_sq_update, &inc, sigma_sq, &inc);
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
      zeros(diag_ouput,n);
      //diagonal of C^-1(\theta)
      zeros(u_vec,n);
      zeros(MFA_sigmasq_grad_vec_cum,n);
      for(int batch_index = 0; batch_index < nBatch; batch_index++){
        tempsize = tempsize_vec[batch_index];
        BatchSize = nBatchIndx[batch_index];
        //if(batch_index == iter % nBatch)
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
          //Rprintf("the value of 1 over E[1/tau_sq]_I : %f \n", *tau_sq_I);
          sum_diags = 0;
          for(i = 0; i < BatchSize; i++){
            tmp_n_mb[i] = y[nBatchLU[batch_index] + i]-w_mu[nBatchLU[batch_index] + i];
            tau_sq_I[0] += pow(tmp_n_mb[i],2);
            sum_diags += sigma_sq[nBatchLU[batch_index] + i];
          }


          ///////////////
          //update tausq
          ///////////////
          if(LR){
            theta[tauSqIndx] = tausq_input;
          }else{
            b_tau_update = tauSqIGb + (sum_diags + *tau_sq_I )*0.5/BatchSize*n;
            a_tau_update = n * 0.5 + tauSqIGa;
            //b_tau_update = tauSqIGb + (sum4/Trace_N + *tau_sq_I )*0.5;
            //
            tau_sq = b_tau_update/a_tau_update;
            theta[tauSqIndx] = tau_sq;

            if(verbose){
              Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
              R_FlushConsole();
#endif
            }
          }



          ///////////////
          //update zetasq
          ///////////////

          // updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //                         theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
          //                         batch_index, final_result_vec, nBatchLU_temp, tempsize);
          // double zeta_Q_mb = Q_mini_batch(B, F, w_mu, w_mu, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU);
          // clock_t start, mid, end;
          // double cpu_time_used;
          // start = clock();
          int ini_point = nBatchLU[batch_index] - 1;
          int end_point = nBatchLU[batch_index] + BatchSize;
          int i_l;
          double sum1 = 0;
          double sum2 = 0;

          if(LR){
            theta[zetaSqIndx] = zetasq_input;
          }else{
            double zeta_Q_mb_re = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                                 n, nnIndx, nnIndxLU, nnIndxLUSq);

            //           end = clock();
            //           cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            //           printf("zeta_Q_mb_re %f seconds to execute \n", cpu_time_used);

            // start = clock();

            for(int i_mb = 0; i_mb < BatchSize; i_mb++){
              i = nBatchLU[batch_index] + i_mb;
              sum1 = sigma_sq[i] * F_inv[i];
              // Rprintf("i's sons \n");
              if(i > 0){
                for (int l = 0; l < nnIndxLU[n + i]; l++) {
                  sum1 = sum1 + Bsq_over_F[nnIndxLU[i] + l] * sigma_sq[nnIndx[nnIndxLU[i] + l]];
                  // Rprintf("%i ",nnIndx[nnIndxLU[i] + l]);
                }
              }
              sum2 += sum1;
            }
            // end = clock();
            // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            // printf("sum2 %f seconds to execute \n", cpu_time_used);

            b_zeta_update = zetaSqIGb + (sum2 + zeta_Q_mb_re)*0.5/BatchSize*n;
            a_zeta_update = n * 0.5 + zetaSqIGa;

            zeta_sq = b_zeta_update/a_zeta_update;

            theta[zetaSqIndx] = zeta_sq;

            //theta[zetaSqIndx] = 17.046391;
            if(verbose){
              Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
              R_FlushConsole();
#endif
            }

          }


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
            // start = clock();
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

                phi_Q = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                       n, nnIndx, nnIndxLU, nnIndxLUSq);

                sum1 = 0;
                sum2 = 0;
                for(int i_mb = 0; i_mb < BatchSize; i_mb++){
                  s = nBatchLU[batch_index] + i_mb;
                  sum1 = sigma_sq[s] * F_inv[s];
                  // Rprintf("i's sons \n");
                  if(s > 0){
                    for (int l = 0; l < nnIndxLU[n + s]; l++) {
                      sum1 = sum1 + Bsq_over_F[nnIndxLU[s] + l] * sigma_sq[nnIndx[nnIndxLU[s] + l]];
                      // Rprintf("%j ",nnIndx[nnIndxLU[j] + l]);
                    }
                  }
                  sum2 += sum1;
                }

                logDetInv = 0.0;
                //diag_sigma_sq_sum = 0.0;
                for(i_mb = 0; i_mb < BatchSize; i_mb++){
                  s = nBatchLU[batch_index] + i_mb;
                  logDetInv += log(F_inv[s]);
                }

                log_g_phi[i*N_phi+j] = logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) - (phi_Q + sum2)*0.5/theta[zetaSqIndx];



              }
            }
            // end = clock();
            // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            // printf("phi %f seconds to execute \n", cpu_time_used);

            max_index = max_ind(log_g_phi,N_phi*N_phi);
            a_phi = a_phi_vec[max_index/N_phi];
            b_phi = b_phi_vec[max_index % N_phi];

            theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;
            // start = clock();
            MFA_updateBF_quadratic(B_temp,  F_temp,  Bmat_over_F_temp,
                                      F_inv, B_over_F, Bmat_over_F, Bsq_over_F,
                                      nIndx, nIndSqx,
                                      nnIndxLUSq,
                                      Trace_N,
                                      c, C, coords, nnIndx, nnIndxLU,
                                      n, m,
                                      nu, covModel, bk, nuUnifb,
                                      a_phi, b_phi,
                                      phimax, phimin);

            // end = clock();
            // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            // printf("MFA_updateBF_quadratic %f seconds to execute \n", cpu_time_used);

          }


          if(verbose){
            Rprintf("the value of theta[%i phiIndx] : %f \n", phiIndx, theta[phiIndx]);
#ifdef Win32
            R_FlushConsole();
#endif
          }
        }


        ///////////////
        //update w
        ///////////////
        // for(int batch_index = 0; batch_index < nBatch; batch_index++)
        {
          if(verbose){
            Rprintf("the value of batch_index for w : %i \n", batch_index);
#ifdef Win32
            R_FlushConsole();
#endif
          }
          tempsize = tempsize_vec[batch_index];
          BatchSize = nBatchIndx[batch_index];
          double gradient_mu;
          double one_over_zeta_sq = 1.0/theta[zetaSqIndx];
          zeros(w_mu_temp,n);
          zeros(w_mu_temp_dF,n);
          zeros(w_mu_temp2,n);
          zeros(gradient_mu_vec,n);
          // product_B_F_minibatch_plus(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
          // product_B_F_minibatch_term1(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);
          // product_B_F_vec_minibatch_plus_fix(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);
          // clock_t start, mid, end;
          // double cpu_time_used;
          // start = clock();
          product_B_F_combine_mb(w_mu, w_mu_temp, w_mu_temp_dF, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                                 BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize,
                                 n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                                 nnIndxwhich);

          F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp, &inc);
          F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp_dF, &inc);
          F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);
          // end = clock();
          // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
          // printf("product_B_F_combine_mb %f seconds to execute \n", cpu_time_used);

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


          // start = clock();
          zeros(gradient_sigmasq_temp,n);
          zeros(MFA_sigmasq_grad_vec,n);

          MFA_sigmasq_grad_term1_rephi(n, nnIndx, nnIndxLU, nnIndxCol,
                                 BatchSize, nBatchLU, batch_index,
                                 numIndxCol, nnIndxnnCol, cumnumIndxCol,
                                 theta, tauSqIndx,
                                 Bsq_over_F, nnIndxLUSq, nnIndxwhich,
                                 final_result_vec, nBatchLU_temp, tempsize,
                                 gradient_sigmasq_temp);
          F77_NAME(dscal)(&n, &one_over_zeta_sq, gradient_sigmasq_temp, &inc);
          // end = clock();
          // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
          // printf("MFA_sigmasq_grad_term1_rephi %f seconds to execute \n", cpu_time_used);
          // start = clock();
          MFA_sigmasq_grad_rephi(MFA_sigmasq_grad_vec, gradient_sigmasq_temp, sigma_sq,
                           n, nnIndx, nnIndxLU, nnIndxCol,
                           BatchSize, nBatchLU, batch_index,
                           numIndxCol, nnIndxnnCol, cumnumIndxCol,
                           theta, tauSqIndx, zetaSqIndx,
                           F_inv, final_result_vec, nBatchLU_temp, tempsize,
                           intersect_start_indices, intersect_sizes, final_intersect_vec,
                           complement_first_start_indices, complement_first_sizes, final_complement_1_vec,
                           complement_second_start_indices, complement_second_sizes, final_complement_2_vec);
          // end = clock();
          // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
          // printf("MFA_sigmasq_grad_rephi %f seconds to execute \n", cpu_time_used);

          double gradient_sigmasq;
          for(i_mb = 0; i_mb < tempsize; i_mb++){
            i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
            gradient_sigmasq = MFA_sigmasq_grad_vec[i];
            E_sigmasq_sq[i] = rho * E_sigmasq_sq[i] + (1 - rho) * pow(gradient_sigmasq,2);
            delta_sigmasq[i] = sqrt(delta_sigmasq_sq[i]+adadelta_noise)/sqrt(E_sigmasq_sq[i]+adadelta_noise)*gradient_sigmasq;
            delta_sigmasq_sq[i] = rho*delta_sigmasq_sq[i] + (1 - rho) * pow(delta_sigmasq[i],2);
            sigma_sq_update[i] = exp(log(sigma_sq[i]) + delta_sigmasq[i]);
          }


          F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);
          F77_NAME(dcopy)(&n, sigma_sq_update, &inc, sigma_sq, &inc);

        }

        // Rprintf("sigma_sq_update: \n");
        // for (int i = 0; i < n; i++) {
        //   Rprintf("%f ", sigma_sq_update[i]);
        // }
        // Rprintf("\n");
      }

      // Rprintf("MFA_sigmasq_grad_vec_cum: \n");
      // for (int i = 0; i < n; i++) {
      //   Rprintf("%f ", MFA_sigmasq_grad_vec_cum[i]);
      // }
      // Rprintf("\n");
      double sum1 = 0;
      double sum2 = 0;
      for(int i = 0; i < n; i++){
        sum1 = sigma_sq[i] * F_inv[i];
        // Rprintf("i's sons \n");
        if(i > 0){
          for (int l = 0; l < nnIndxLU[n + i]; l++) {
            sum1 = sum1 + Bsq_over_F[nnIndxLU[i] + l] * sigma_sq[nnIndx[nnIndxLU[i] + l]];
            // Rprintf("%i ",nnIndx[nnIndxLU[i] + l]);
          }
        }
        sum2 += sum1;
      }



      double sum3 = 0.0;
      double sum4 = 0.0;
      double sum5 = 0.0;
      for(int i = 0; i < n; i++){
        sum3 += pow((y[i]- w_mu_update[i]),2);
        sum3 += sigma_sq_update[i];
        sum4 += log(2*pi*sigma_sq_update[i]);
        sum5 += log(2*pi*F_inv[i]/theta[zetaSqIndx]);
      }

      ELBO = 0.0;

      ELBO += sum3/theta[tauSqIndx] * 0.5;

      ELBO += E_quadratic(w_mu_update, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq)/theta[zetaSqIndx]*0.5;

      ELBO += -sum5 * 0.5;

      ELBO += sum2/theta[zetaSqIndx] * 0.5;

      ELBO += -sum4 * 0.5;

      ELBO += n*log(2*pi*theta[tauSqIndx]) * 0.5;



      ELBO += -n * 0.5;

      ELBO_vec[iter-1] = -0.5*ELBO;

      if(iter == min_iter){max_ELBO = - 0.5*ELBO;}
      if (iter > min_iter && iter % 10 == 0){

        int count = 0;
        double sum = 0.0;
        for (int i = iter - 10; i < iter; i++) {
          sum += ELBO_vec[i];
          count++;
        }

        double average = sum / count;

        if (average < max_ELBO) {
          ELBO_convergence_count += 1;
        } else {
          ELBO_convergence_count = 0;
        }
        max_ELBO = max(max_ELBO, average);

        // Rprintf("Max ELBO at interation at %i is %f\n", iter, max_ELBO);
        // Rprintf("Average window ELBO at interation at %i is %f\n", iter, average);

        if(stop_K){
          indicator_converge = ELBO_convergence_count>=K;
          //Rprintf("indicator_converge %i \n", indicator_converge);
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
    int nResultListObjs = 15;

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

    SET_VECTOR_ELT(result_r, 7, B_r);
    SET_VECTOR_ELT(resultName_r, 7, mkChar("B"));

    SET_VECTOR_ELT(result_r, 8, F_r);
    SET_VECTOR_ELT(resultName_r, 8, mkChar("F"));

    SET_VECTOR_ELT(result_r, 9, theta_r);
    SET_VECTOR_ELT(resultName_r, 9, mkChar("theta"));

    SET_VECTOR_ELT(result_r, 10, w_mu_r);
    SET_VECTOR_ELT(resultName_r, 10, mkChar("w_mu"));

    SET_VECTOR_ELT(result_r, 11, sigma_sq_r);
    SET_VECTOR_ELT(resultName_r, 11, mkChar("w_sigma_sq"));

    SET_VECTOR_ELT(result_r, 12, iter_r);
    SET_VECTOR_ELT(resultName_r, 12, mkChar("iter"));

    SET_VECTOR_ELT(result_r, 13, ELBO_vec_r);
    SET_VECTOR_ELT(resultName_r, 13, mkChar("ELBO_vec"));

    SET_VECTOR_ELT(result_r, 14, theta_para_r);
    SET_VECTOR_ELT(resultName_r, 14, mkChar("theta_para"));

    namesgets(result_r, resultName_r);
    //unprotect
    UNPROTECT(nProtect);

    return(result_r);

  }

  SEXP spVarBayes_MFA_mb_beta_rephicpp(SEXP y_r, SEXP X_r, SEXP n_r, SEXP p_r, SEXP m_r, SEXP coords_r, SEXP covModel_r, SEXP rho_r,
                                       SEXP zetaSqIG_r, SEXP tauSqIG_r, SEXP phibeta_r, SEXP nuUnif_r,
                                       SEXP zetaSqStarting_r, SEXP tauSqStarting_r, SEXP phiStarting_r, SEXP nuStarting_r,
                                       SEXP sType_r, SEXP nThreads_r, SEXP verbose_r, SEXP fix_nugget_r, SEXP N_phi_r, SEXP Trace_N_r,
                                       SEXP max_iter_r,
                                       SEXP var_input_r,
                                       SEXP phi_input_r, SEXP phi_iter_max_r,SEXP initial_mu_r,
                                       SEXP mini_batch_size_r,
                                       SEXP min_iter_r, SEXP K_r, SEXP stop_K_r, SEXP tausq_input_r, SEXP zetasq_input_r, SEXP LR_r){

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
    double pi = 3.1415926;
    //get args
    double *y = REAL(y_r);
    double *X = REAL(X_r);
    int p = INTEGER(p_r)[0];
    int n = INTEGER(n_r)[0];
    int m = INTEGER(m_r)[0];
    double *coords = REAL(coords_r);
    double fix_nugget = REAL(fix_nugget_r)[0];
    int covModel = INTEGER(covModel_r)[0];
    std::string corName = getCorName(covModel);
    //double converge_per  =  REAL(converge_per_r)[0];
    double phi_input  =  REAL(phi_input_r)[0];
    double tausq_input  =  REAL(tausq_input_r)[0];
    double zetasq_input  =  REAL(zetasq_input_r)[0];

    double *var_input  =  REAL(var_input_r);
    int initial_mu  =  INTEGER(initial_mu_r)[0];
    int phi_iter_max = INTEGER(phi_iter_max_r)[0];
    int n_mb = INTEGER(mini_batch_size_r)[0];

    int K = INTEGER(K_r)[0];
    int stop_K = INTEGER(stop_K_r)[0];
    int min_iter = INTEGER(min_iter_r)[0];

    int nThreads = INTEGER(nThreads_r)[0];
    int verbose = INTEGER(verbose_r)[0];
    int LR = INTEGER(LR_r)[0];
    //double  vi_threshold  =  REAL(vi_threshold_r)[0];
    double  rho  =  REAL(rho_r)[0];
    //double  rho_phi  =  REAL(rho_phi_r)[0];
    // double  tau_input  =  REAL(tau_input_r)[0];

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
      Rprintf("Model fit with %i observations.\n\n", n);
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
    for(int i = 0; i < n; i++){
      sigma_sq[i] = var_input[i];
    }

    double *w_mu_update = (double *) R_alloc(n, sizeof(double)); zeros(w_mu_update, n);
    double *E_mu_sq = (double *) R_alloc(n, sizeof(double)); zeros(E_mu_sq, n);
    double *delta_mu_sq = (double *) R_alloc(n, sizeof(double)); zeros(delta_mu_sq, n);
    double *delta_mu = (double *) R_alloc(n, sizeof(double)); zeros(delta_mu, n);
    double *m_mu = (double *) R_alloc(n, sizeof(double)); zeros(m_mu, n);

    double *E_sigmasq_sq = (double *) R_alloc(n, sizeof(double)); zeros(E_sigmasq_sq, n);
    double *delta_sigmasq_sq = (double *) R_alloc(n, sizeof(double)); zeros(delta_sigmasq_sq, n);
    double *delta_sigmasq = (double *) R_alloc(n, sizeof(double)); zeros(delta_sigmasq, n);
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
    double adadelta_noise = 0.0000001;
    double *bk = (double *) R_alloc(nThreads*(1.0+5.0), sizeof(double));
    //double *bk = (double *) R_alloc(nThreads*(1.0+static_cast<int>(floor(nuUnifb))), sizeof(double));
    if(corName == "matern"){nu = theta[nuIndx];}

    int iter = 1;
    int max_iter = INTEGER(max_iter_r)[0];

    double vi_error = 1.0;
    double rho1 = 0.9;
    double rho2 = 0.999;
    double adaptive_adam = 0.001;
    // int n_per = n * converge_per;
    // int *sign_vec_old = (int *) R_alloc(n_per, sizeof(int));
    // int *sign_vec_new = (int *) R_alloc(n_per, sizeof(int));
    // int *check_vec = (int *) R_alloc(n_per, sizeof(int));

    int nIndSqx = static_cast<int>(static_cast<double>(1+m)*(m+m+1)/6*m+(n-m-1)*m*m);
    double *F_inv = (double *) R_alloc(n, sizeof(double)); zeros(F_inv,n);
    double *B_over_F = (double *) R_alloc(nIndx, sizeof(double)); zeros(B_over_F,nIndx);
    double *Bsq_over_F = (double *) R_alloc(nIndx, sizeof(double)); zeros(Bsq_over_F,nIndx);
    double *Bmat_over_F = (double *) R_alloc(nIndSqx, sizeof(double)); zeros(Bmat_over_F,nIndSqx);
    int *nnIndxLUSq = (int *) R_alloc(2*n, sizeof(int));

    double *F_temp = (double *) R_alloc(n, sizeof(double)); zeros(F_temp,n);
    double *B_temp = (double *) R_alloc(nIndx, sizeof(double)); zeros(B_temp,nIndx);
    double *Bmat_over_F_temp = (double *) R_alloc(nIndSqx, sizeof(double)); zeros(Bmat_over_F_temp,nIndSqx);

    updateBF2(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[phiIndx], nu, covModel, bk, nuUnifb);

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
        Bsq_over_F[nnIndxLU[i]+j] = B[nnIndxLU[i]+j]*B[nnIndxLU[i]+j]/F[i];
      }

      if(i > 0){
        F_inv_temp = 1/F[i];
        F77_NAME(dger)(&nnIndxLU[n+i], &nnIndxLU[n+i], &one, &B[nnIndxLU[i]], &inc, &B[nnIndxLU[i]], &inc, &Bmat_over_F[nnIndxLUSq[i]], &nnIndxLU[n+i]);
        F77_NAME(dscal)(&nnIndxLUSq[n+i], &F_inv_temp, &Bmat_over_F[nnIndxLUSq[i]], &inc);
      }

    }
    int *nnIndxwhich = (int *) R_alloc(nIndx, sizeof(int));; zeros_int(nnIndxwhich, nIndx);
    get_nnIndxwhich(nnIndxwhich, n, nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);

    int indicator_converge = 0;

    double *trace_vec = (double *) R_alloc(2, sizeof(double));
    double ELBO_MC = 0.0;
    double ELBO = 0.0;

    SEXP ELBO_vec_r; PROTECT(ELBO_vec_r = allocVector(REALSXP, max_iter)); nProtect++;
    double *ELBO_vec = REAL(ELBO_vec_r); zeros(ELBO_vec,max_iter);
    double max_ELBO = 0.0;
    int ELBO_convergence_count = 0;


    double *diag_ouput = (double *) R_alloc(n, sizeof(double));
    double *diag_input = (double *) R_alloc(n, sizeof(double));
    double *u_vec = (double *) R_alloc(n, sizeof(double));
    double *epsilon_vec = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp_dF = (double *) R_alloc(n, sizeof(double));
    double *w_mu_temp2 = (double *) R_alloc(n, sizeof(double));
    double *gradient_sigmasq_temp = (double *) R_alloc(n, sizeof(double));
    double *MFA_sigmasq_grad_vec = (double *) R_alloc(n, sizeof(double));
    double *MFA_sigmasq_grad_vec_cum = (double *) R_alloc(n, sizeof(double));
    double *gradient_mu_vec = (double *) R_alloc(n, sizeof(double));

    double E_phi_sq = 0.0;
    double delta_phi = 0.0;
    double delta_phi_sq = 0.0;

    double *tmp_n_mb = (double *) R_alloc(n, sizeof(double)); zeros(tmp_n_mb, n);
    double *diag_input_mb = (double *) R_alloc(n, sizeof(double)); zeros(diag_input_mb, n);
    double *phi_can_vec = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    double *log_g_phi = (double *) R_alloc(N_phi*N_phi, sizeof(double));
    int BatchSize;
    double sum_diags= 0.0;
    int i_mb;

    int batch_index = 0;

    int max_result_size = nBatch * n;
    int max_temp_size = n;

    int* result_arr = (int *) R_alloc(max_result_size, sizeof(int));
    int* temp_arr = (int *) R_alloc(max_temp_size, sizeof(int));
    int result_index = 0;
    int temp_index = 0;

    int* tempsize_vec = (int *) R_alloc(nBatch, sizeof(int));

    int *seen_values = (int *) R_alloc(n, sizeof(int)); // initialized to zeros


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

    for(int batch_index = 0; batch_index < nBatch; batch_index++) {
      BatchSize = nBatchIndx[batch_index];
      zeros_int(seen_values,n);

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


    // Print all elements of tempsize_vec
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

      ///////////////
      //update beta
      ///////////////

      zeros(tmp_n, n);
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

      b_tau_update = tauSqIGb + (F77_NAME(dasum)(&n, sigma_sq, &inc) +
        p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;

        tau_sq = b_tau_update/a_tau_update;
        theta[tauSqIndx] = tau_sq;
        //theta[tauSqIndx] = tau_input;
        if(verbose){
          Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
          R_FlushConsole();
#endif
        }
        ///////////////
        //update zetasq
        ///////////////

        //updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);
        //double zeta_Q = Q(B, F, w_mu, w_mu, n, nnIndx, nnIndxLU);
        double zeta_Q_re = E_quadratic(w_mu, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);

        double sum1 = 0;
        double sum2 = 0;
        for(int i = 0; i < n; i++){
          sum1 = sigma_sq[i] * F_inv[i];
          int num_m = nnIndxLU[n+i];
          // Rprintf("i's sons \n");
          if(i > 0){
            for (int l = 0; l < nnIndxLU[n + i]; l++) {
              sum1 = sum1 + Bsq_over_F[nnIndxLU[i] + l] * sigma_sq[nnIndx[nnIndxLU[i] + l]];
              // Rprintf("%i ",nnIndx[nnIndxLU[i] + l]);
            }
          }
          sum2 += sum1;
        }

        b_zeta_update = zetaSqIGb + (sum2 + zeta_Q_re)*0.5;
        //b_zeta_update = zetaSqIGb + (sum2 + zeta_Q)*theta[zetaSqIndx]*0.5;


        zeta_sq = b_zeta_update/a_zeta_update;

        theta[zetaSqIndx] = zeta_sq;

        if(verbose){
          Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
          R_FlushConsole();
#endif
        }
        // updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb);

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

              updateBF_quadratic(B_temp, F_temp, Bmat_over_F_temp,
                                 F_inv, B_over_F, Bmat_over_F,
                                 nIndx, nIndSqx,
                                 nnIndxLUSq,
                                 Trace_phi,
                                 c, C, coords, nnIndx, nnIndxLU,
                                 n,  m,
                                 nu,  covModel, bk,  nuUnifb,
                                 a_phi_vec[i],  b_phi_vec[j],  phimax,  phimin);

              phi_Q = E_quadratic(w_mu, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq);

              sum1 = 0;
              sum2 = 0;
              for(int s = 0; s < n; s++){
                sum1 = sigma_sq[s] * F_inv[s];
                // Rprintf("i's sons \n");
                if(s > 0){
                  for (int l = 0; l < nnIndxLU[n + s]; l++) {
                    sum1 = sum1 + Bsq_over_F[nnIndxLU[s] + l] * sigma_sq[nnIndx[nnIndxLU[s] + l]];
                    // Rprintf("%j ",nnIndx[nnIndxLU[j] + l]);
                  }
                }
                sum2 += sum1;
              }

              logDetInv = 0.0;
              //diag_sigma_sq_sum = 0.0;
              for(int s = 0; s < n; s++){
                logDetInv += log(F_inv[s]);
              }

              log_g_phi[i*N_phi+j] = logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) - (phi_Q + sum2)*0.5/theta[zetaSqIndx];

            }
          }


          max_index = max_ind(log_g_phi,N_phi*N_phi);
          a_phi = a_phi_vec[max_index/N_phi];
          b_phi = b_phi_vec[max_index % N_phi];

          theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;;

          MFA_updateBF_quadratic(B_temp,  F_temp,  Bmat_over_F_temp,
                                 F_inv, B_over_F, Bmat_over_F, Bsq_over_F,
                                 nIndx, nIndSqx,
                                 nnIndxLUSq,
                                 Trace_N,
                                 c, C, coords, nnIndx, nnIndxLU,
                                 n, m,
                                 nu, covModel, bk, nuUnifb,
                                 a_phi, b_phi,
                                 phimax, phimin);

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
        double one_over_zeta_sq = 1.0/theta[zetaSqIndx];
        zeros(w_mu_temp,n);
        zeros(w_mu_temp2,n);
        // product_B_F(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp);
        // product_B_F_vec(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol);
        product_B_F_combine(w_mu, w_mu_temp, w_mu_temp2, F_inv,  B_over_F, Bmat_over_F,
                            n,  nnIndx, nnIndxLU, nnIndxLUSq, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol,
                            nnIndxwhich);
        F77_NAME(dscal)(&n, &one_over_zeta_sq, w_mu_temp2, &inc);

        double gradient_mu = 0.0;
        for(i = 0; i < n; i++){
          gradient_mu = ( - w_mu[i]/theta[tauSqIndx] - w_mu_temp2[i] + (y[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc))/theta[tauSqIndx]);
          E_mu_sq[i] = rho * E_mu_sq[i] + (1 - rho) * pow(gradient_mu,2);
          delta_mu[i] = sqrt(delta_mu_sq[i]+adadelta_noise)/sqrt(E_mu_sq[i]+adadelta_noise)*gradient_mu;
          delta_mu_sq[i] = rho*delta_mu_sq[i] + (1 - rho) * pow(delta_mu[i],2);
          w_mu_update[i] = w_mu[i] + delta_mu[i];
        }

        zeros(gradient_sigmasq_temp,n);
        zeros(MFA_sigmasq_grad_vec,n);
        double sum3;
        for (int i = 0; i < n; i++) {
          // Rprintf("i is : %i \n", i);
          sum3 = 0;
          int i_l, num_m, j_ind;
          // Rprintf("i's neighbor: \n ");
          if(numIndxCol[i] > 0){
            for (int l = 0; l < numIndxCol[i]; l++) {
              // l is the lth that the i is whose neighbor
              // transfer to i_l
              i_l = nnIndxnnCol[cumnumIndxCol[i] - i + l];

              // Rprintf("%i \n", i_l);
              //sum3 += B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]*B[ nnIndxCol[ 1 + cumnumIndxCol[i] + l] ]/F[i_l];
              num_m = nnIndxLU[n+i_l];
              j_ind = nnIndxwhich[cumnumIndxCol[i] - i + l];
              sum3 += Bsq_over_F[nnIndxCol[ 1 + cumnumIndxCol[i] + l] ];

            }
          }
          // Rprintf("\n");
          gradient_sigmasq_temp[i] =  sum3;
        }
        F77_NAME(dscal)(&n, &one_over_zeta_sq, gradient_sigmasq_temp, &inc);

        for (int i = 0; i < n; i++) {
          MFA_sigmasq_grad_vec[i] = (1/sigma_sq[i] - 1/theta[tauSqIndx] - F_inv[i]/theta[zetaSqIndx] - gradient_sigmasq_temp[i]) * 0.5 * sigma_sq[i];
        }

        double gradient_sigmasq;
        for(int i = 0; i < n; i++){
          gradient_sigmasq = MFA_sigmasq_grad_vec[i];
          E_sigmasq_sq[i] = rho * E_sigmasq_sq[i] + (1 - rho) * pow(gradient_sigmasq,2);
          delta_sigmasq[i] = sqrt(delta_sigmasq_sq[i]+adadelta_noise)/sqrt(E_sigmasq_sq[i]+adadelta_noise)*gradient_sigmasq;
          delta_sigmasq_sq[i] = rho*delta_sigmasq_sq[i] + (1 - rho) * pow(delta_sigmasq[i],2);
          sigma_sq_update[i] = exp(log(sigma_sq[i]) + delta_sigmasq[i]);
        }



        F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);
        F77_NAME(dcopy)(&n, sigma_sq_update, &inc, sigma_sq, &inc);
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
      zeros(diag_ouput,n);
      //diagonal of C^-1(\theta)
      zeros(u_vec,n);
      zeros(MFA_sigmasq_grad_vec_cum,n);
      for(int batch_index = 0; batch_index < nBatch; batch_index++){
        tempsize = tempsize_vec[batch_index];
        BatchSize = nBatchIndx[batch_index];
        //if(batch_index == iter % nBatch)
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
          sum_diags = 0;
          zeros(tau_sq_I, one_int);
          zeros(tmp_n_mb, n);

          for(i = 0; i < BatchSize; i++){
            tmp_n_mb[nBatchLU[batch_index] + i] = y[nBatchLU[batch_index] + i]-w_mu[nBatchLU[batch_index] + i];
            tau_sq_I[0] += pow(tmp_n_mb[nBatchLU[batch_index] + i],2);
            sum_diags += sigma_sq[nBatchLU[batch_index] + i];
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
              beta_cov[idx] = tmp_pp[idx] * theta[tauSqIndx] / BatchSize * n;
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
          if(LR){
            theta[tauSqIndx] = tausq_input;
          }else{
            zeros(tau_sq_H, one_int);
            for(i = 0; i < p; i++){
              tau_sq_H[0] += tmp_p2[i]*tmp_p[i];
            }

            // b_tau_update = tauSqIGb + (sum_diags + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5;
            b_tau_update = tauSqIGb + (sum_diags + p*theta[tauSqIndx] + *tau_sq_I - *tau_sq_H)*0.5/ BatchSize * n;
            a_tau_update = tauSqIGa + n*0.5;

            //b_tau_update = tauSqIGb + (sum4/Trace_N + *tau_sq_I )*0.5;
            //
            // Rprintf("the value of sum4/trace : %f \n", sum4/Trace_N);

            // b_tau_update = tauSqIGb + (186.9186 + p*theta[tauSqIndx] + 332.6374)*0.5;

            tau_sq = b_tau_update/a_tau_update;
            theta[tauSqIndx] = tau_sq;
            //theta[tauSqIndx] = 0.5;
            if(verbose){
              Rprintf("the value of 1 over E[1/tau_sq] : %f \n", tau_sq);
#ifdef Win32
              R_FlushConsole();
#endif
            }
          }






          ///////////////
          //update zetasq
          ///////////////

          // updateBF_minibatch_plus(B, F, c, C, coords, nnIndx, nnIndxLU, n, m,
          //                         theta[zetaSqIndx], theta[phiIndx], nu, covModel, bk, nuUnifb,
          //                         batch_index, final_result_vec, nBatchLU_temp, tempsize);
          // double zeta_Q_mb = Q_mini_batch(B, F, w_mu, w_mu, BatchSize, nBatchLU, batch_index, n, nnIndx, nnIndxLU);
          int ini_point = nBatchLU[batch_index] - 1;
          int end_point = nBatchLU[batch_index] + BatchSize;
          int i_l;
          double sum1 = 0;
          double sum2 = 0;

          if(LR){
            theta[zetaSqIndx] = zetasq_input;
          }else{
            double zeta_Q_mb_re = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                                 n, nnIndx, nnIndxLU, nnIndxLUSq);



            for(int i_mb = 0; i_mb < BatchSize; i_mb++){
              i = nBatchLU[batch_index] + i_mb;
              sum1 = sigma_sq[i] * F_inv[i];
              // Rprintf("i's sons \n");
              if(i > 0){
                for (int l = 0; l < nnIndxLU[n + i]; l++) {
                  sum1 = sum1 + Bsq_over_F[nnIndxLU[i] + l] * sigma_sq[nnIndx[nnIndxLU[i] + l]];
                  // Rprintf("%i ",nnIndx[nnIndxLU[i] + l]);
                }
              }
              sum2 += sum1;
            }

            b_zeta_update = zetaSqIGb + (sum2 + zeta_Q_mb_re)*0.5/BatchSize*n;
            a_zeta_update = n * 0.5 + zetaSqIGa;

            zeta_sq = b_zeta_update/a_zeta_update;

            theta[zetaSqIndx] = zeta_sq;
          }




          //theta[zetaSqIndx] = 2;
          //theta[zetaSqIndx] = 10;
          //theta[zetaSqIndx] = 17.046391;
          if(verbose){
            Rprintf("the value of 1 over E[1/sigma_sq] : %f \n", zeta_sq);
#ifdef Win32
            R_FlushConsole();
#endif
          }


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

                phi_Q = E_quadratic_mb(w_mu, F_inv, B_over_F, Bmat_over_F, BatchSize, nBatchLU, batch_index,
                                       n, nnIndx, nnIndxLU, nnIndxLUSq);
                sum1 = 0;
                sum2 = 0;
                for(int i_mb = 0; i_mb < BatchSize; i_mb++){
                  s = nBatchLU[batch_index] + i_mb;
                  sum1 = sigma_sq[s] * F_inv[s];
                  // Rprintf("i's sons \n");
                  if(s > 0){
                    for (int l = 0; l < nnIndxLU[n + s]; l++) {
                      sum1 = sum1 + Bsq_over_F[nnIndxLU[s] + l] * sigma_sq[nnIndx[nnIndxLU[s] + l]];
                      // Rprintf("%j ",nnIndx[nnIndxLU[j] + l]);
                    }
                  }
                  sum2 += sum1;
                }

                logDetInv = 0.0;
                //diag_sigma_sq_sum = 0.0;
                for(i_mb = 0; i_mb < BatchSize; i_mb++){
                  s = nBatchLU[batch_index] + i_mb;
                  logDetInv += log(F_inv[s]);
                }

                log_g_phi[i*N_phi+j] = logDetInv*0.5 + 0.5*log(1/theta[zetaSqIndx]) - (phi_Q + sum2)*0.5/theta[zetaSqIndx];



              }
            }

            max_index = max_ind(log_g_phi,N_phi*N_phi);
            a_phi = a_phi_vec[max_index/N_phi];
            b_phi = b_phi_vec[max_index % N_phi];

            theta[phiIndx] = a_phi/(a_phi+b_phi)*(phimax - phimin) + phimin;

            MFA_updateBF_quadratic(B_temp,  F_temp,  Bmat_over_F_temp,
                                   F_inv, B_over_F, Bmat_over_F, Bsq_over_F,
                                   nIndx, nIndSqx,
                                   nnIndxLUSq,
                                   Trace_N,
                                   c, C, coords, nnIndx, nnIndxLU,
                                   n, m,
                                   nu, covModel, bk, nuUnifb,
                                   a_phi, b_phi,
                                   phimax, phimin);



          }


          if(verbose){
            Rprintf("the value of theta[%i phiIndx] : %f \n", phiIndx, theta[phiIndx]);
#ifdef Win32
            R_FlushConsole();
#endif
          }
        }


        ///////////////
        //update w
        ///////////////
        // for(int batch_index = 0; batch_index < nBatch; batch_index++)
        {
          if(verbose){
            Rprintf("the value of batch_index for w : %i \n", batch_index);
#ifdef Win32
            R_FlushConsole();
#endif
          }
          tempsize = tempsize_vec[batch_index];
          BatchSize = nBatchIndx[batch_index];
          double gradient_mu;
          double one_over_zeta_sq = 1.0/theta[zetaSqIndx];
          zeros(w_mu_temp,n);
          zeros(w_mu_temp_dF,n);
          zeros(w_mu_temp2,n);
          zeros(gradient_mu_vec,n);
          // product_B_F_minibatch_plus(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp, batch_index, final_result_vec, nBatchLU_temp, tempsize);
          // product_B_F_minibatch_term1(B, F, w_mu, n, nnIndxLU, nnIndx, w_mu_temp_dF, batch_index, final_result_vec, nBatchLU_temp, tempsize);
          // product_B_F_vec_minibatch_plus_fix(B, F, w_mu_temp, n, nnIndxLU, nnIndx, w_mu_temp2, cumnumIndxCol, numIndxCol, nnIndxCol, nnIndxnnCol, BatchSize, nBatchLU, batch_index, final_result_vec, nBatchLU_temp, tempsize);
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

          MFA_sigmasq_grad_term1_rephi(n, nnIndx, nnIndxLU, nnIndxCol,
                                       BatchSize, nBatchLU, batch_index,
                                       numIndxCol, nnIndxnnCol, cumnumIndxCol,
                                       theta, tauSqIndx,
                                       Bsq_over_F, nnIndxLUSq, nnIndxwhich,
                                       final_result_vec, nBatchLU_temp, tempsize,
                                       gradient_sigmasq_temp);
          F77_NAME(dscal)(&n, &one_over_zeta_sq, gradient_sigmasq_temp, &inc);

          MFA_sigmasq_grad_rephi(MFA_sigmasq_grad_vec, gradient_sigmasq_temp, sigma_sq,
                                 n, nnIndx, nnIndxLU, nnIndxCol,
                                 BatchSize, nBatchLU, batch_index,
                                 numIndxCol, nnIndxnnCol, cumnumIndxCol,
                                 theta, tauSqIndx, zetaSqIndx,
                                 F_inv, final_result_vec, nBatchLU_temp, tempsize,
                                 intersect_start_indices, intersect_sizes, final_intersect_vec,
                                 complement_first_start_indices, complement_first_sizes, final_complement_1_vec,
                                 complement_second_start_indices, complement_second_sizes, final_complement_2_vec);

          double gradient_sigmasq;
          for(i_mb = 0; i_mb < tempsize; i_mb++){
            i = final_result_vec[nBatchLU_temp[batch_index] + i_mb];
            gradient_sigmasq = MFA_sigmasq_grad_vec[i];
            E_sigmasq_sq[i] = rho * E_sigmasq_sq[i] + (1 - rho) * pow(gradient_sigmasq,2);
            delta_sigmasq[i] = sqrt(delta_sigmasq_sq[i]+adadelta_noise)/sqrt(E_sigmasq_sq[i]+adadelta_noise)*gradient_sigmasq;
            delta_sigmasq_sq[i] = rho*delta_sigmasq_sq[i] + (1 - rho) * pow(delta_sigmasq[i],2);
            sigma_sq_update[i] = exp(log(sigma_sq[i]) + delta_sigmasq[i]);
          }


          F77_NAME(dcopy)(&n, w_mu_update, &inc, w_mu, &inc);
          F77_NAME(dcopy)(&n, sigma_sq_update, &inc, sigma_sq, &inc);

        }

        // Rprintf("sigma_sq_update: \n");
        // for (int i = 0; i < n; i++) {
        //   Rprintf("%f ", sigma_sq_update[i]);
        // }
        // Rprintf("\n");
      }

      // Rprintf("MFA_sigmasq_grad_vec_cum: \n");
      // for (int i = 0; i < n; i++) {
      //   Rprintf("%f ", MFA_sigmasq_grad_vec_cum[i]);
      // }
      // Rprintf("\n");
      double sum1 = 0;
      double sum2 = 0;
      for(int i = 0; i < n; i++){
        sum1 = sigma_sq[i] * F_inv[i];
        // Rprintf("i's sons \n");
        if(i > 0){
          for (int l = 0; l < nnIndxLU[n + i]; l++) {
            sum1 = sum1 + Bsq_over_F[nnIndxLU[i] + l] * sigma_sq[nnIndx[nnIndxLU[i] + l]];
            // Rprintf("%i ",nnIndx[nnIndxLU[i] + l]);
          }
        }
        sum2 += sum1;
      }



      double sum3 = 0.0;
      double sum4 = 0.0;
      double sum5 = 0.0;
      for(int i = 0; i < n; i++){
        sum3 += pow((y[i]- w_mu_update[i] - F77_NAME(ddot)(&p, &X[i], &n, beta, &inc)),2);
        sum3 += sigma_sq_update[i];
        sum4 += log(2*pi*sigma_sq_update[i]);
        //sum5 += log(2*pi*F_inv[i]/theta[zetaSqIndx]);
        sum5 += log(2*pi*F_inv[i]) + digamma(a_zeta_update) - log(b_zeta_update);
      }

      ELBO = 0.0;

      ELBO += sum3/theta[tauSqIndx] * 0.5;

      ELBO += E_quadratic(w_mu_update, F_inv, B_over_F, Bmat_over_F, n, nnIndx, nnIndxLU, nnIndxLUSq)/theta[zetaSqIndx]*0.5;

      ELBO += -sum5 * 0.5;

      ELBO += sum2/theta[zetaSqIndx] * 0.5;

      ELBO += -sum4 * 0.5;

      ELBO += -n*(log(0.5/pi) + digamma(a_tau_update) - log(b_tau_update))* 0.5;



      ELBO += -n * 0.5;

      ELBO_vec[iter-1] = -0.5*ELBO;

      if(iter == min_iter){max_ELBO = - 0.5*ELBO;}
      if (iter > min_iter && iter % 10 == 0){

        int count = 0;
        double sum = 0.0;
        for (int i = iter - 10; i < iter; i++) {
          sum += ELBO_vec[i];
          count++;
        }

        double average = sum / count;

        if (average < max_ELBO) {
          ELBO_convergence_count += 1;
        } else {
          ELBO_convergence_count = 0;
        }
        max_ELBO = max(max_ELBO, average);

        // Rprintf("Max ELBO at interation at %i is %f\n", iter, max_ELBO);
        // Rprintf("Average window ELBO at interation at %i is %f\n", iter, average);

        if(stop_K){
          indicator_converge = ELBO_convergence_count>=K;
          //Rprintf("indicator_converge %i \n", indicator_converge);
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



    }


    SEXP quadratic_term_r; PROTECT(quadratic_term_r = allocVector(REALSXP, 1)); nProtect++;
    double *quadratic_term = REAL(quadratic_term_r);

    zeros(tmp_n, n);
    zeros(tau_sq_I, one_int);
    for(i = 0; i < n; i++){
      tmp_n[i] = y[i]-w_mu[i];
      tau_sq_I[0] += pow(tmp_n[i],2);
    }

    F77_NAME(dgemv)(ytran, &n, &p, &one, X, &n, tmp_n, &inc, &zero, tmp_p, &inc FCONE);

    zeros(tau_sq_H, one_int);
    for(i = 0; i < p; i++){
      tau_sq_H[0] += tmp_p2[i]*tmp_p[i];
    }
    quadratic_term[0] = tau_sq_I[0] - tau_sq_H[0];


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
    int nResultListObjs = 18;

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

    SET_VECTOR_ELT(result_r, 7, B_r);
    SET_VECTOR_ELT(resultName_r, 7, mkChar("B"));

    SET_VECTOR_ELT(result_r, 8, F_r);
    SET_VECTOR_ELT(resultName_r, 8, mkChar("F"));

    SET_VECTOR_ELT(result_r, 9, theta_r);
    SET_VECTOR_ELT(resultName_r, 9, mkChar("theta"));

    SET_VECTOR_ELT(result_r, 10, w_mu_r);
    SET_VECTOR_ELT(resultName_r, 10, mkChar("w_mu"));

    SET_VECTOR_ELT(result_r, 11, sigma_sq_r);
    SET_VECTOR_ELT(resultName_r, 11, mkChar("w_sigma_sq"));

    SET_VECTOR_ELT(result_r, 12, iter_r);
    SET_VECTOR_ELT(resultName_r, 12, mkChar("iter"));

    SET_VECTOR_ELT(result_r, 13, ELBO_vec_r);
    SET_VECTOR_ELT(resultName_r, 13, mkChar("ELBO_vec"));

    SET_VECTOR_ELT(result_r, 14, theta_para_r);
    SET_VECTOR_ELT(resultName_r, 14, mkChar("theta_para"));

    SET_VECTOR_ELT(result_r, 15, beta_r);
    SET_VECTOR_ELT(resultName_r, 15, mkChar("beta"));

    SET_VECTOR_ELT(result_r, 16, beta_cov_r);
    SET_VECTOR_ELT(resultName_r, 16, mkChar("beta_cov"));

    SET_VECTOR_ELT(result_r, 17, quadratic_term_r);
    SET_VECTOR_ELT(resultName_r, 17, mkChar("quadratic_term"));

    namesgets(result_r, resultName_r);
    //unprotect
    UNPROTECT(nProtect);

    return(result_r);

  }



 


}
